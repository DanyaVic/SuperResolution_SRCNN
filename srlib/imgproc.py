"""Работа с изображениями: чтение/запись (в т.ч. пути с кириллицей в Windows), цветовые пространства,
бикубический ресайз «как в MATLAB» и модель деградации HR → LR."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT)


def imread(path):
    """BGR uint8. np.fromfile + imdecode работает с не-ASCII путями в Windows (cv2.imread — нет)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Не удалось прочитать изображение: {path}")
    return img


def imwrite(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise IOError(f"Не удалось сохранить: {path}")
    buf.tofile(str(path))


# --------------------------------------------------------------------- цветовые пространства
# ITU-R BT.601, «студийный» диапазон, как rgb2ycbcr в MATLAB — стандарт для подсчёта PSNR/SSIM в SR.
def bgr2ycbcr(img):
    """uint8/float BGR [0..255] → float32 YCbCr [0..255] (Y ∈ [16, 235])."""
    img = img.astype(np.float32) / 255.0
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    y = 16.0 + 65.481 * r + 128.553 * g + 24.966 * b
    cb = 128.0 - 37.797 * r - 74.203 * g + 112.0 * b
    cr = 128.0 + 112.0 * r - 93.786 * g - 18.214 * b
    return np.stack([y, cb, cr], -1)


def ycbcr2bgr(ycc):
    """float YCbCr [0..255] → uint8 BGR."""
    y, cb, cr = ycc[..., 0] - 16.0, ycc[..., 1] - 128.0, ycc[..., 2] - 128.0
    r = 1.164383 * y + 1.596027 * cr
    g = 1.164383 * y - 0.391762 * cb - 0.812968 * cr
    b = 1.164383 * y + 2.017232 * cb
    return np.clip(np.stack([b, g, r], -1), 0, 255).round().astype(np.uint8)


def bgr2y(img):
    return bgr2ycbcr(img)[..., 0]


# --------------------------------------------------------------------- ресайз и деградация
def imresize(img, size):
    """Бикубический ресайз с антиалиасингом при уменьшении (PIL ≈ MATLAB imresize).
    img: uint8 HxW или HxWx3 (BGR); size: (width, height)."""
    if img.ndim == 2:
        return np.asarray(Image.fromarray(img).resize(size, Image.BICUBIC))
    rgb = Image.fromarray(img[..., ::-1])
    return np.ascontiguousarray(np.asarray(rgb.resize(size, Image.BICUBIC))[..., ::-1])


def modcrop(img, scale):
    """Обрезает изображение до размеров, кратных масштабу — иначе LR и HR не совпадают по сетке."""
    h, w = img.shape[:2]
    return img[: h - h % scale, : w - w % scale]


def degrade(hr, scale):
    """Модель деградации «bicubic» (стандарт DIV2K track 1): LR = (HR ↓s)_bicubic, квантование в uint8."""
    h, w = hr.shape[:2]
    return imresize(hr, (w // scale, h // scale))


def upscale_bicubic(lr, scale):
    h, w = lr.shape[:2]
    return imresize(lr, (w * scale, h * scale))


# --------------------------------------------------------------------- тензоры
def to_tensor(img):
    """uint8/float HxW(xC) → float tensor 1xCxHxW в [0, 1]."""
    import torch

    arr = img.astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[None]
    else:
        arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(arr))[None]


def to_image(t):
    """float tensor 1xCxHxW [0,1] → float HxW(xC) [0, 255] (без округления)."""
    arr = t.detach().float().clamp(0, 1).cpu().numpy()[0] * 255.0
    return arr[0] if arr.shape[0] == 1 else arr.transpose(1, 2, 0)
