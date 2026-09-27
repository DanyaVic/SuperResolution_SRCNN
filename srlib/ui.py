"""Утилиты интерфейса: подписи на кириллице поверх изображений (OpenCV сам кириллицу не рисует)."""
import cv2
import numpy as np

_FONT_CACHE = {}


def _font(size):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font = None
    try:
        from PIL import ImageFont

        try:
            from matplotlib import font_manager

            path = font_manager.findfont("DejaVu Sans", fallback_to_default=True)
        except Exception:
            path = "DejaVuSans.ttf"
        font = ImageFont.truetype(path, size)
    except Exception:
        font = None
    _FONT_CACHE[size] = font
    return font


def put_text(img, lines, org=(10, 10), size=18, color=(255, 255, 255), bg=(0, 0, 0), alpha=0.55):
    """Рисует несколько строк текста на полупрозрачной подложке. img — BGR uint8 (изменяется на месте)."""
    if isinstance(lines, str):
        lines = [lines]
    font = _font(size)
    pad, gap = 6, 4
    if font is None:  # запасной вариант: только латиница
        for i, t in enumerate(lines):
            cv2.putText(img, t.encode("ascii", "replace").decode(), (org[0], org[1] + 20 * (i + 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, size / 30, color, 1, cv2.LINE_AA)
        return img
    from PIL import Image, ImageDraw

    widths = [font.getbbox(t)[2] for t in lines]
    lh = size + gap
    w, h = max(widths) + 2 * pad, lh * len(lines) + 2 * pad - gap
    x0, y0 = org
    x1, y1 = min(x0 + w, img.shape[1]), min(y0 + h, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return img
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi * (1 - alpha) + np.array(bg) * alpha).astype(np.uint8)
    pil = Image.fromarray(np.ascontiguousarray(img[y0:y1, x0:x1][..., ::-1]))
    d = ImageDraw.Draw(pil)
    for i, t in enumerate(lines):
        d.text((pad, pad + i * lh), t, font=font, fill=tuple(int(c) for c in color[::-1]))
    img[y0:y1, x0:x1] = np.asarray(pil)[..., ::-1]
    return img


def fit(img, max_w, max_h, interp=cv2.INTER_AREA):
    h, w = img.shape[:2]
    k = min(max_w / w, max_h / h, 1.0)
    if k >= 1.0:
        return img, 1.0
    return cv2.resize(img, (max(1, int(w * k)), max(1, int(h * k))), interpolation=interp), k


def gui_available():
    try:
        cv2.namedWindow("__probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__probe__")
        return True
    except cv2.error:
        return False
