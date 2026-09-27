"""Единый интерфейс апскейлеров: bicubic, SRCNN (наши веса), ESRGAN и Real-ESRGAN (официальные веса).
Все принимают BGR uint8 и возвращают BGR uint8 увеличенного размера."""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .imgproc import bgr2ycbcr, imresize, to_image, to_tensor, upscale_bicubic, ycbcr2bgr
from .models import RRDBNet, SRCNN

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"
CHECKPOINTS = ROOT / "checkpoints"

GITHUB = "https://github.com/xinntao/Real-ESRGAN/releases/download"
PRETRAINED = {
    # ESRGAN (официальный, обучен на DF2K+OST с бикубической деградацией, Perceptual+GAN loss)
    ("esrgan", 4): (f"{GITHUB}/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth", 4),
    # Real-ESRGAN: тот же генератор, обучен на синтетических «реальных» деградациях (blur, шум, JPEG)
    ("realesrgan", 4): (f"{GITHUB}/v0.1.0/RealESRGAN_x4plus.pth", 4),
    ("realesrgan", 2): (f"{GITHUB}/v0.2.1/RealESRGAN_x2plus.pth", 2),
}


def pick_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def download(url, dst):
    dst = Path(dst)
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Скачиваю веса: {url}")
    torch.hub.download_url_to_file(url, str(dst), progress=True)
    return dst


@torch.no_grad()
def tiled_forward(net, x, scale, tile=0, pad=16):
    """Прогон по тайлам с перекрытием: экономит видеопамять на больших изображениях (4 ГБ на 3050 Ti).
    x: 1xCxHxW; scale — во сколько раз выход больше входа (для SRCNN = 1)."""
    _, _, h, w = x.shape
    if tile <= 0 or (h <= tile and w <= tile):
        return net(x)
    out = None
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            py0, px0 = max(y0 - pad, 0), max(x0 - pad, 0)
            py1, px1 = min(y1 + pad, h), min(x1 + pad, w)
            res = net(x[:, :, py0:py1, px0:px1])
            if out is None:
                out = x.new_zeros((1, res.shape[1], h * scale, w * scale))
            oy, ox = (y0 - py0) * scale, (x0 - px0) * scale
            out[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] = \
                res[:, :, oy:oy + (y1 - y0) * scale, ox:ox + (x1 - x0) * scale]
    return out


class Upscaler:
    name = "base"

    def __init__(self, scale, device="cpu", tile=0, half=False):
        self.scale, self.device, self.tile = scale, torch.device(device), tile
        self.half = half and self.device.type == "cuda"
        self.last_time = 0.0

    def params(self):
        return {"model": self.name, "scale": self.scale, "device": str(self.device), "tile": self.tile,
                "fp16": self.half}

    def __call__(self, lr_bgr):
        t0 = time.perf_counter()
        out = self._run(lr_bgr)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.last_time = time.perf_counter() - t0
        return out


class BicubicUpscaler(Upscaler):
    name = "bicubic"

    def _run(self, lr_bgr):
        return upscale_bicubic(lr_bgr, self.scale)


class SRCNNUpscaler(Upscaler):
    """Y-канал обрабатывает сеть, Cb/Cr увеличиваются бикубически (глаз менее чувствителен к цвету)."""

    name = "srcnn"

    def __init__(self, scale, checkpoint=None, **kw):
        super().__init__(scale, **kw)
        checkpoint = Path(checkpoint or CHECKPOINTS / f"srcnn_x{scale}.pth")
        if not checkpoint.exists():
            raise FileNotFoundError(f"Нет весов SRCNN: {checkpoint}. Сначала обучите: python train.py --scale {scale}")
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if ckpt.get("scale", scale) != scale:
            print(f"[!] Чекпоинт обучен для x{ckpt['scale']}, используется для x{scale}")
        self.net = SRCNN(**ckpt["config"])
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval().to(self.device)
        if self.half:
            self.net.half()
        self.checkpoint = str(checkpoint)
        self.info = {k: ckpt.get(k) for k in ("iter", "best_psnr", "bicubic_psnr")}

    def params(self):
        return {**super().params(), "checkpoint": Path(self.checkpoint).name}

    @torch.no_grad()
    def _run(self, lr_bgr):
        up = upscale_bicubic(lr_bgr, self.scale)
        ycc = bgr2ycbcr(up)
        x = to_tensor(ycc[..., 0]).to(self.device)
        if self.half:
            x = x.half()
        y = tiled_forward(self.net, x, 1, self.tile)
        ycc[..., 0] = to_image(y)
        return ycbcr2bgr(ycc)


class RRDBUpscaler(Upscaler):
    """ESRGAN / Real-ESRGAN. Если нужный масштаб отличается от «родного» (ESRGAN бывает только ×4),
    результат ×4 уменьшается до нужного размера (так делает и официальный скрипт, параметр outscale)."""

    def __init__(self, scale, variant="esrgan", **kw):
        super().__init__(scale, **kw)
        self.name = variant
        key = (variant, scale) if (variant, scale) in PRETRAINED else (variant, 4)
        url, native = PRETRAINED[key]
        path = download(url, WEIGHTS / url.rsplit("/", 1)[-1])
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = state.get("params_ema", state.get("params", state))
        self.net = RRDBNet(scale=native)
        self.net.load_state_dict(state, strict=True)
        self.net.eval().to(self.device)
        if self.half:
            self.net.half()
        self.native = native
        self.weights = path.name

    def params(self):
        return {**super().params(), "weights": self.weights, "native_scale": self.native}

    @torch.no_grad()
    def _run(self, lr_bgr):
        x = to_tensor(lr_bgr[..., ::-1].copy()).to(self.device)
        if self.half:
            x = x.half()
        # pixel-unshuffle в x2-модели требует чётных размеров
        h, w = x.shape[2:]
        ph, pw = (h % 2, w % 2) if self.native == 2 else (0, 0)
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        y = tiled_forward(self.net, x, self.native, self.tile)
        y = y[:, :, : h * self.native, : w * self.native]
        out = np.clip(to_image(y), 0, 255).round().astype(np.uint8)[..., ::-1]
        out = np.ascontiguousarray(out)
        if self.native != self.scale:
            out = imresize(out, (w * self.scale, h * self.scale))
        return out


METHODS = ("bicubic", "srcnn", "esrgan", "realesrgan")


def build_upscaler(method, scale, device="auto", tile=0, half=False, checkpoint=None):
    device = pick_device(device)
    kw = dict(device=device, tile=tile, half=half)
    if method == "bicubic":
        return BicubicUpscaler(scale, **kw)
    if method == "srcnn":
        return SRCNNUpscaler(scale, checkpoint=checkpoint, **kw)
    if method in ("esrgan", "realesrgan"):
        return RRDBUpscaler(scale, variant=method, **kw)
    raise ValueError(f"Неизвестный метод: {method}. Доступны: {', '.join(METHODS)}")
