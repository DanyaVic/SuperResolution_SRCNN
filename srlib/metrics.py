"""Метрики качества: PSNR, SSIM (по каналу Y, как в статьях по SR), LPIPS и анализ ошибок деталей
(ложноположительные = «выдуманные» детали, ложноотрицательные = потерянные детали)."""
import cv2
import numpy as np
from skimage.metrics import structural_similarity

from .imgproc import bgr2y


def _crop(img, border):
    return img[border:-border, border:-border] if border > 0 else img


def psnr(a, b, border=0, data_range=255.0):
    """PSNR = 10·log10(MAX² / MSE). a, b — float/uint8 массивы одного размера."""
    a = _crop(np.asarray(a, np.float64), border)
    b = _crop(np.asarray(b, np.float64), border)
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(data_range**2 / mse))


def ssim(a, b, border=0, data_range=255.0):
    """SSIM с гауссовым окном σ=1.5 (реализация Wang et al. 2004)."""
    a = _crop(np.asarray(a, np.float64), border)
    b = _crop(np.asarray(b, np.float64), border)
    return float(structural_similarity(a, b, data_range=data_range, gaussian_weights=True, sigma=1.5,
                                       use_sample_covariance=False, channel_axis=-1 if a.ndim == 3 else None))


def psnr_ssim_y(sr_bgr, hr_bgr, border):
    """Стандартный протокол SR-бенчмарков: метрики на Y (YCbCr), обрезка рамки = масштабу."""
    ys, yh = bgr2y(sr_bgr.astype(np.uint8)), bgr2y(hr_bgr.astype(np.uint8))
    return psnr(ys, yh, border), ssim(ys, yh, border)


class LPIPSMetric:
    """LPIPS (Zhang et al., 2018): расстояние между глубокими признаками AlexNet. Меньше — лучше.
    Требует пакет `lpips` (при первом запуске скачивает веса AlexNet ~230 МБ)."""

    def __init__(self, device="cpu"):
        import lpips  # noqa: WPS433
        import torch

        self.torch = torch
        self.device = device
        self.net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    def __call__(self, sr_bgr, hr_bgr, border=0):
        t = self.torch

        def prep(img):
            img = _crop(img, border)[..., ::-1].astype(np.float32) / 127.5 - 1.0
            return t.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None].to(self.device)

        with t.no_grad():
            return float(self.net(prep(sr_bgr), prep(hr_bgr)).item())


def try_lpips(device="cpu"):
    try:
        return LPIPSMetric(device)
    except Exception as exc:  # нет пакета или нет доступа к весам
        print(f"[!] LPIPS недоступен ({type(exc).__name__}: {str(exc)[:120]}). Метрика будет пропущена.")
        return None


# ------------------------------------------------------------------ анализ ошибок 1 и 2 рода
def detail_map(y, sigma=1.5):
    """Карта высокочастотных деталей: |Y − GaussBlur(Y)|."""
    y = y.astype(np.float32)
    return np.abs(y - cv2.GaussianBlur(y, (0, 0), sigma))


def detail_errors(sr_bgr, hr_bgr, border, thr=6.0, tol=1):
    """Сравнивает, где есть детали (текстура, края) в эталоне и в результате.

    D_gt = detail(HR) > thr, D_sr = detail(SR) > thr; допуск смещения `tol` пикселей (дилатация).
      FP (ошибка I рода)  — деталь есть в SR, но нет в HR: «галлюцинация»/артефакт (звон, шум, ложная текстура);
      FN (ошибка II рода) — деталь есть в HR, но SR её не восстановил: размытие, потеря текстуры.
    Возвращает precision, recall, F1, доли FP/FN и бинарные маски для визуализации.
    """
    yg = _crop(bgr2y(hr_bgr.astype(np.uint8)), border)
    ys = _crop(bgr2y(sr_bgr.astype(np.uint8)), border)
    d_gt, d_sr = detail_map(yg) > thr, detail_map(ys) > thr
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    d_gt_dil = cv2.dilate(d_gt.astype(np.uint8), k).astype(bool)
    d_sr_dil = cv2.dilate(d_sr.astype(np.uint8), k).astype(bool)
    fp = d_sr & ~d_gt_dil
    fn = d_gt & ~d_sr_dil
    tp_p = (d_sr & d_gt_dil).sum()
    tp_r = (d_gt & d_sr_dil).sum()
    precision = tp_p / max(d_sr.sum(), 1)
    recall = tp_r / max(d_gt.sum(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return dict(precision=float(precision), recall=float(recall), f1=float(f1),
                fp_rate=float(fp.mean()), fn_rate=float(fn.sum() / max(d_gt.sum(), 1)),
                fp_mask=fp, fn_mask=fn)


def lr_consistency_psnr(sr_bgr, lr_bgr, scale):
    """Безэталонная проверка: уменьшаем результат обратно и сравниваем со входом (LR-PSNR, NTIRE/PIRM).
    Высокое значение = SR не противоречит исходному изображению (не «перерисовал» его)."""
    from .imgproc import degrade

    back = degrade(sr_bgr.astype(np.uint8), scale)
    h, w = min(back.shape[0], lr_bgr.shape[0]), min(back.shape[1], lr_bgr.shape[1])
    return psnr(back[:h, :w].astype(np.float64), lr_bgr[:h, :w].astype(np.float64))
