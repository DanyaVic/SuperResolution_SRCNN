"""Увеличение разрешения изображения (Super-Resolution) — файл или веб-камера.

Примеры:
  python upscale.py --input photo.jpg --model srcnn --scale 4
  python upscale.py --input photo.jpg --model esrgan --scale 4 --mode eval    # фото = эталон: ↓4 → ↑4, PSNR/SSIM
  python upscale.py --input small.png --ref original.png --model srcnn --scale 2   # PSNR относительно эталона
  python upscale.py --input folder_with_images/ --model realesrgan --scale 4 --no-show
  python upscale.py --webcam 0 --model srcnn --scale 2                           # реальное время

Режимы (--mode):
  sr   — обычный апскейл: входное изображение считается низкого разрешения и увеличивается в S раз.
         Эталона нет, поэтому логируется LR-PSNR (результат уменьшают обратно и сравнивают со входом —
         показывает, что сеть не исказила исходное содержимое). С --ref считается настоящий PSNR.
  eval — входное изображение считается эталоном HR: оно уменьшается в S раз (бикубически), затем
         восстанавливается моделью, и считаются PSNR / SSIM (+ LPIPS с --lpips) против оригинала.
         По умолчанию для веб-камеры.

Окно: слева бикубическая интерполяция, справа модель; снизу — увеличенные фрагменты под курсором мыши.
Клавиши: q/Esc — выход, s — сохранить снимок окна, пробел — пауза (веб-камера).
Каждый результат дублируется строкой в log.txt: время | входной файл | параметры апскейла | метрики | файл результата.
"""
import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from srlib.imgproc import IMG_EXT, degrade, imread, imwrite, modcrop, upscale_bicubic
from srlib.metrics import lr_consistency_psnr, psnr_ssim_y, try_lpips
from srlib.ui import fit, gui_available, put_text
from srlib.upscalers import METHODS, build_upscaler

ROOT = Path(__file__).resolve().parent
TITLES = {"bicubic": "Бикубическая", "srcnn": "SRCNN", "esrgan": "ESRGAN", "realesrgan": "Real-ESRGAN"}
WIN = "Super-Resolution"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", "-i", help="путь к изображению или папке с изображениями")
    src.add_argument("--webcam", nargs="?", const=0, type=int, help="номер веб-камеры (по умолчанию 0)")
    ap.add_argument("--model", "-m", choices=METHODS, default="srcnn")
    ap.add_argument("--scale", "-s", type=int, choices=[2, 4], default=4)
    ap.add_argument("--mode", choices=["sr", "eval"], default=None,
                    help="sr — увеличить как есть; eval — считать вход эталоном (по умолчанию: файл→sr, камера→eval)")
    ap.add_argument("--ref", help="эталонное HR-изображение для подсчёта PSNR/SSIM в режиме sr")
    ap.add_argument("--checkpoint", help="свой чекпоинт SRCNN (по умолчанию checkpoints/srcnn_x{S}.pth)")
    ap.add_argument("--device", default="auto", help="auto / cuda / cpu")
    ap.add_argument("--tile", type=int, default=None,
                    help="размер тайла (0 = без тайлов). По умолчанию: ESRGAN 256 (вход), SRCNN 1024 (выход)")
    ap.add_argument("--fp32", action="store_true", help="не использовать FP16 на GPU")
    ap.add_argument("--lpips", action="store_true", help="считать LPIPS в режиме eval (нужен пакет lpips)")
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "upscaled"))
    ap.add_argument("--log", default=str(ROOT / "log.txt"))
    ap.add_argument("--no-show", action="store_true", help="не открывать окно (только файлы и лог)")
    ap.add_argument("--cam-width", type=int, default=640, help="ширина кадра веб-камеры до обработки")
    ap.add_argument("--log-every", type=int, default=30, help="веб-камера: писать в лог каждые N кадров")
    return ap.parse_args()


# ------------------------------------------------------------------------------------ лог
def rel(p):
    """Короткий путь для лога: относительно текущей папки, если возможно."""
    try:
        return Path(p).resolve().relative_to(Path.cwd()).as_posix()
    except (ValueError, OSError):
        return str(p)


def write_log(path, source, params, metrics, output):
    """Одна строка на результат: время | вход | параметры апскейла | метрики | выход."""
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    p = " ".join(f"{k}={v}" for k, v in params.items())
    m = " ".join(f"{k}={v}" for k, v in metrics.items())
    source = rel(source) if Path(str(source)).exists() else source
    output = rel(output) if output != "-" else output
    line = f"{ts} | input={source} | {p} | {m} | output={output}"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return line


def fmt_metrics(res):
    out = {}
    for k, v in res.items():
        if v is None:
            continue
        if k.startswith("PSNR"):
            out[k] = f"{v:.2f}dB"
        elif k.startswith("time"):
            out[k] = f"{v:.3f}s"
        elif isinstance(v, float):
            out[k] = f"{v:.4f}"
        else:
            out[k] = v
    return out


# ------------------------------------------------------------------------------------ обработка
def process(img, up, mode, ref=None, lpips_fn=None):
    """Возвращает dict: lr, bicubic, sr, hr (или None), метрики."""
    s = up.scale
    if mode == "eval":
        hr = modcrop(img, s)
        lr = degrade(hr, s)
    else:
        lr, hr = img, None
        if ref is not None:
            hr = ref[: lr.shape[0] * s, : lr.shape[1] * s]
    sr = up(lr)
    bic = upscale_bicubic(lr, s)
    res = {"in_size": f"{lr.shape[1]}x{lr.shape[0]}", "out_size": f"{sr.shape[1]}x{sr.shape[0]}",
           "time": up.last_time}
    if hr is not None and hr.shape == sr.shape:
        res["PSNR"], res["SSIM"] = psnr_ssim_y(sr, hr, s)
        res["PSNR_bicubic"], res["SSIM_bicubic"] = psnr_ssim_y(bic, hr, s)
        if lpips_fn is not None:
            res["LPIPS"] = lpips_fn(sr, hr, s)
    else:
        res["PSNR_LR"] = lr_consistency_psnr(sr, lr, s)
    return {"lr": lr, "bicubic": bic, "sr": sr, "hr": hr, "metrics": res}


def info_lines(up, mode, m):
    lines = [f"{TITLES[up.name]} ×{up.scale} | режим {mode} | {m['in_size']} → {m['out_size']} | "
             f"{m['time'] * 1000:.0f} мс ({up.device.type})"]
    if "PSNR" in m:
        lines.append(f"PSNR {m['PSNR']:.2f} дБ (бикуб. {m['PSNR_bicubic']:.2f}, "
                     f"{m['PSNR'] - m['PSNR_bicubic']:+.2f})   SSIM {m['SSIM']:.4f} (бикуб. {m['SSIM_bicubic']:.4f})"
                     + (f"   LPIPS {m['LPIPS']:.3f}" if "LPIPS" in m else ""))
    else:
        lines.append(f"LR-PSNR {m['PSNR_LR']:.2f} дБ (согласованность со входом; эталона нет)")
    return lines


# ------------------------------------------------------------------------------------ окно
class Viewer:
    """Сверху: бикубическая | модель. Снизу: фрагменты 1:1 под курсором (LR, бикубическая, модель, эталон)."""

    def __init__(self, out, up, mode, max_w=1600, max_h=980, zoom=128, zoom_px=240):
        self.o, self.up, self.mode = out, up, mode
        self.zoom, self.zoom_px = zoom, zoom_px
        self.panel_w = (max_w - 10) // 2
        self.panel_h = max_h - zoom_px - 40
        h, w = out["sr"].shape[:2]
        self.cx, self.cy = w // 2, h // 2
        _, self.k = fit(out["sr"], self.panel_w, self.panel_h)
        self.info = info_lines(up, mode, out["metrics"])

    def on_mouse(self, event, x, y, flags, _):
        if event == cv2.EVENT_MOUSEMOVE:
            pw = int(self.o["sr"].shape[1] * self.k) + 10
            px = x - pw if x >= pw else x
            h, w = self.o["sr"].shape[:2]
            self.cx = int(np.clip(px / self.k, 0, w - 1))
            self.cy = int(np.clip(y / self.k, 0, h - 1))

    def crop(self, img, s=1):
        z = self.zoom // s
        cx, cy = self.cx // s, self.cy // s
        h, w = img.shape[:2]
        x0, y0 = int(np.clip(cx - z // 2, 0, max(w - z, 0))), int(np.clip(cy - z // 2, 0, max(h - z, 0)))
        c = img[y0:y0 + z, x0:x0 + z]
        return cv2.resize(c, (self.zoom_px, self.zoom_px), interpolation=cv2.INTER_NEAREST)

    def render(self):
        o, s = self.o, self.up.scale
        top = []
        for key, title in (("bicubic", "Бикубическая"), ("sr", TITLES[self.up.name])):
            p, _ = fit(o[key], self.panel_w, self.panel_h)
            p = p.copy()
            z = int(self.zoom * self.k)
            x0 = int(np.clip(self.cx * self.k - z / 2, 0, p.shape[1] - z))
            y0 = int(np.clip(self.cy * self.k - z / 2, 0, p.shape[0] - z))
            cv2.rectangle(p, (x0, y0), (x0 + z, y0 + z), (0, 255, 255), 1)
            put_text(p, title, (8, p.shape[0] - 34), size=18)
            top.append(p)
        gap = np.zeros((top[0].shape[0], 10, 3), np.uint8)
        top = np.hstack([top[0], gap, top[1]])
        tiles = [("Вход LR", self.crop(o["lr"], s)), ("Бикубическая", self.crop(o["bicubic"])),
                 (TITLES[self.up.name], self.crop(o["sr"]))]
        if o["hr"] is not None and o["hr"].shape == o["sr"].shape:
            tiles.append(("Эталон HR", self.crop(o["hr"])))
        row = []
        for t, c in tiles:
            c = c.copy()
            put_text(c, t, (4, 4), size=15)
            row += [c, np.zeros((self.zoom_px, 6, 3), np.uint8)]
        row = np.hstack(row[:-1])
        width = max(top.shape[1], row.shape[1])
        pad = lambda a: np.pad(a, ((0, 0), (0, width - a.shape[1]), (0, 0)))  # noqa: E731
        canvas = np.vstack([pad(top), np.zeros((8, width, 3), np.uint8), pad(row)])
        put_text(canvas, self.info, (8, 8), size=17)
        return canvas

    def run(self, snapshot_path):
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self.on_mouse)
        print("Окно открыто: наведите мышь для увеличенного фрагмента; s — снимок окна, q/Esc — далее/выход")
        while True:
            canvas = self.render()
            cv2.imshow(WIN, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                imwrite(snapshot_path, canvas)
                print(f"Снимок окна: {snapshot_path}")
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyWindow(WIN)


# ------------------------------------------------------------------------------------ режимы
def run_files(args, up, lpips_fn, show):
    src = Path(args.input)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT) if src.is_dir() else [src]
    if not files:
        sys.exit(f"Нет изображений: {src}")
    mode = args.mode or "sr"
    ref = imread(args.ref) if args.ref else None
    out_dir = Path(args.out_dir)
    for f in files:
        img = imread(f)
        o = process(img, up, mode, ref, lpips_fn)
        stem = f"{f.stem}_{up.name}_x{up.scale}" + ("_eval" if mode == "eval" else "")
        out_path = out_dir / f"{stem}.png"
        imwrite(out_path, o["sr"])
        if mode == "eval":
            imwrite(out_dir / f"{stem}_input_lr.png", o["lr"])
        params = {"mode": mode, **up.params()}
        line = write_log(args.log, f, params, fmt_metrics(o["metrics"]), out_path)
        print(line)
        if show:
            Viewer(o, up, mode).run(out_dir / f"{stem}_window.png")


def run_webcam(args, up, lpips_fn, show):
    cap = cv2.VideoCapture(args.webcam)
    if not cap.isOpened():
        sys.exit(f"Не удалось открыть веб-камеру {args.webcam}")
    mode = args.mode or "eval"
    out_dir = Path(args.out_dir)
    n, fps, paused, o = 0, 0.0, False, None
    params = {"mode": mode, **up.params()}
    print("Веб-камера: q/Esc — выход, s — сохранить кадр, пробел — пауза")
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                print("Кадр не получен — выход")
                break
            h, w = frame.shape[:2]
            if w > args.cam_width:
                frame = cv2.resize(frame, (args.cam_width, int(h * args.cam_width / w)), interpolation=cv2.INTER_AREA)
            t0 = time.perf_counter()
            o = process(frame, up, mode, None, lpips_fn if n % args.log_every == 0 else None)
            fps = 0.9 * fps + 0.1 / max(time.perf_counter() - t0, 1e-6) if n else 1 / max(time.perf_counter() - t0, 1e-6)
            n += 1
            if n % args.log_every == 1:
                write_log(args.log, f"webcam:{args.webcam}#frame{n}", params, fmt_metrics(o["metrics"]), "-")
        if show:
            left, _ = fit(o["bicubic"], 790, 700)
            right, _ = fit(o["sr"], 790, 700)
            left, right = left.copy(), right.copy()
            put_text(left, "Бикубическая", (8, left.shape[0] - 34), size=18)
            put_text(right, TITLES[up.name], (8, right.shape[0] - 34), size=18)
            canvas = np.hstack([left, np.zeros((left.shape[0], 10, 3), np.uint8), right])
            put_text(canvas, info_lines(up, mode, o["metrics"]) + [f"FPS {fps:.1f}  кадр {n}"], (8, 8), size=16)
            cv2.imshow(WIN, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            if key == ord("s"):
                ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                p = out_dir / f"webcam_{ts}_{up.name}_x{up.scale}.png"
                imwrite(p, o["sr"])
                imwrite(out_dir / f"webcam_{ts}_window.png", canvas)
                print(write_log(args.log, f"webcam:{args.webcam}#frame{n}", params, fmt_metrics(o["metrics"]), p))
        elif n >= args.log_every * 10:
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    tile = args.tile
    if tile is None:
        tile = 256 if args.model in ("esrgan", "realesrgan") else 1024
    try:
        up = build_upscaler(args.model, args.scale, args.device, tile=tile, half=not args.fp32,
                            checkpoint=args.checkpoint)
    except FileNotFoundError as e:
        sys.exit(str(e))
    print("Параметры апскейла:", json.dumps(up.params(), ensure_ascii=False))
    lpips_fn = try_lpips(str(up.device)) if args.lpips else None
    show = not args.no_show
    if show and not gui_available():
        print("[!] Графический вывод недоступен (нет дисплея) — работаю без окна.")
        show = False
    if args.webcam is not None:
        run_webcam(args, up, lpips_fn, show)
    else:
        run_files(args, up, lpips_fn, show)
    print(f"Лог: {args.log}")


if __name__ == "__main__":
    main()
