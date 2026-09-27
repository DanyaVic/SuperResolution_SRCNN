"""Оценка на тестовых наборах: PSNR, SSIM, LPIPS, анализ ошибок (ложные / потерянные детали) + иллюстрации.

  python evaluate.py                                   # все test-наборы из data/splits.json, ×2 и ×4, все методы
  python evaluate.py --scales 4 --methods bicubic srcnn esrgan --lpips
  python evaluate.py --sets Set5 Set14 --max-per-set 20

Результаты (папка results/):
  metrics_per_image.csv  — метрики по каждому изображению и методу
  summary.md / .json     — средние значения по наборам (эту таблицу можно вставить в отчёт)
  figures/examples_x{S}.png  — успешные кейсы (сравнение фрагментов)
  figures/errors_x{S}.png    — ошибки: FP (красный) — ложные детали, FN (синий) — потерянные детали
  images/…                — сохранённые увеличенные изображения для примеров
"""
import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from srlib.imgproc import degrade, imread, imwrite, modcrop
from srlib.metrics import detail_errors, detail_map, psnr_ssim_y, try_lpips
from srlib.upscalers import METHODS, build_upscaler

ROOT = Path(__file__).resolve().parent
TITLES = {"bicubic": "Бикубическая", "srcnn": "SRCNN", "esrgan": "ESRGAN", "realesrgan": "Real-ESRGAN"}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", default=str(ROOT / "data" / "splits.json"))
    ap.add_argument("--sets", nargs="*", help="какие тестовые наборы (по умолчанию все)")
    ap.add_argument("--scales", nargs="+", type=int, default=[2, 4])
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    ap.add_argument("--max-per-set", type=int, default=0, help="ограничить число изображений в наборе (0 = все)")
    ap.add_argument("--lpips", action="store_true", help="считать LPIPS (нужен пакет lpips)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--n-examples", type=int, default=3, help="сколько примеров в каждой иллюстрации")
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--figures-only", action="store_true",
                    help="только перерисовать иллюстрации по готовому metrics_per_image.csv")
    ap.add_argument("--append", action="store_true",
                    help="дописать к существующему metrics_per_image.csv (заменяя пересчитанные строки)")
    return ap.parse_args()


def best_window(score_map, win):
    """Координаты окна win×win с максимальной суммой score_map."""
    s = cv2.boxFilter(score_map.astype(np.float32), -1, (win, win), normalize=False, borderType=cv2.BORDER_CONSTANT)
    h, w = s.shape
    half = win // 2
    s[:half, :] = s[h - half:, :] = -1
    s[:, :half] = s[:, w - half:] = -1
    y, x = np.unravel_index(np.argmax(s), s.shape)
    return max(0, x - half), max(0, y - half)


def main():
    args = parse_args()
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    sets = {k: v for k, v in splits["test"].items() if not args.sets or k in args.sets}
    if args.max_per_set:
        sets = {k: v[: args.max_per_set] for k, v in sets.items()}
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    lp = try_lpips("cuda" if args.device in ("auto", "cuda") and _cuda() else "cpu") if args.lpips else None

    if args.figures_only:
        rows = list(csv.DictReader(open(out / "metrics_per_image.csv", encoding="utf-8")))
        for r in rows:
            r["scale"] = int(r["scale"])
            for k in ("psnr", "ssim", "fp_rate", "fn_rate"):
                r[k] = float(r[k])
            r["lpips"] = float(r["lpips"]) if r["lpips"] not in ("", "None") else None
        for s in args.scales:
            ms = [m for m in METHODS if any(r["method"] == m and r["scale"] == s for r in rows)]
            ups = {m: build_upscaler(m, s, args.device, tile=args.tile) for m in ms}
            make_figures(rows, ups, s, out, args.n_examples, any(r["lpips"] is not None for r in rows))
        return

    rows = []
    for s in args.scales:
        ups = {}
        for m in args.methods:
            try:
                ups[m] = build_upscaler(m, s, args.device, tile=args.tile, half=False)
            except FileNotFoundError as e:
                print(f"[!] {m} x{s} пропущен: {e}")
        for set_name, files in sets.items():
            t_set = time.time()
            for f in files:
                hr = modcrop(imread(f), s)
                lr = degrade(hr, s)
                for m, up in ups.items():
                    sr = up(lr)
                    p, ss = psnr_ssim_y(sr, hr, s)
                    de = detail_errors(sr, hr, s)
                    rows.append({"set": set_name, "image": Path(f).name, "path": f, "scale": s, "method": m,
                                 "psnr": p, "ssim": ss, "lpips": lp(sr, hr, s) if lp else None,
                                 "detail_precision": de["precision"], "detail_recall": de["recall"],
                                 "detail_f1": de["f1"], "fp_rate": de["fp_rate"], "fn_rate": de["fn_rate"],
                                 "time_s": up.last_time})
            means = {m: np.mean([r["psnr"] for r in rows if r["set"] == set_name and r["scale"] == s
                                 and r["method"] == m]) for m in ups}
            print(f"x{s} {set_name:>18}: " + "  ".join(f"{m} {v:.2f}" for m, v in means.items())
                  + f"  ({time.time() - t_set:.0f} c)")
        make_figures(rows, ups, s, out, args.n_examples, lp)

    # ---------- сохранение таблиц (с --append старые строки других масштабов/наборов/методов сохраняются)
    old = out / "metrics_per_image.csv"
    if args.append and old.exists():
        prev = list(csv.DictReader(open(old, encoding="utf-8")))
        keep = [r for r in prev if not (int(r["scale"]) in args.scales and r["method"] in args.methods
                                        and r["set"] in sets)]
        for r in keep:
            for k in ("psnr", "ssim", "detail_precision", "detail_recall", "detail_f1", "fp_rate", "fn_rate",
                      "time_s"):
                r[k] = float(r[k])
            r["scale"] = int(r["scale"])
            r["lpips"] = float(r["lpips"]) if r["lpips"] not in ("", "None") else None
        rows = keep + rows
    keys = list(rows[-1].keys())
    with open(out / "metrics_per_image.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)
    summary = defaultdict(dict)
    agg = defaultdict(list)
    for r in rows:
        agg[(r["scale"], r["set"], r["method"])].append(r)
    for (s, st, m), rs in agg.items():
        mean = lambda k: float(np.mean([r[k] for r in rs])) if rs[0][k] is not None else None  # noqa: E731
        summary[f"x{s}"][f"{st}/{m}"] = {"n": len(rs), "psnr": mean("psnr"), "ssim": mean("ssim"),
                                         "lpips": mean("lpips"), "detail_precision": mean("detail_precision"),
                                         "detail_recall": mean("detail_recall"), "detail_f1": mean("detail_f1"),
                                         "time_s": mean("time_s")}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    write_markdown(summary, out / "summary.md")
    print((out / "summary.md").read_text(encoding="utf-8"))


def _cuda():
    import torch

    return torch.cuda.is_available()


def write_markdown(summary, path):
    lines = ["# Результаты на тестовых наборах", "",
             "PSNR / SSIM — канал Y, обрезка рамки = масштабу. LPIPS — RGB (меньше = лучше). "
             "Детали: precision = 1 − доля ложных деталей (ошибка I рода), recall = 1 − доля потерянных "
             "(ошибка II рода).", ""]
    for sc, d in summary.items():
        lines += [f"## Масштаб {sc}", "",
                  "| Набор | Метод | N | PSNR, дБ ↑ | SSIM ↑ | LPIPS ↓ | Precision деталей ↑ | Recall деталей ↑ | F1 ↑ | Время, с |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for key, v in d.items():
            st, m = key.split("/")
            lp = f"{v['lpips']:.4f}" if v["lpips"] is not None else "—"
            lines.append(f"| {st} | {TITLES[m]} | {v['n']} | {v['psnr']:.2f} | {v['ssim']:.4f} | {lp} | "
                         f"{v['detail_precision']:.3f} | {v['detail_recall']:.3f} | {v['detail_f1']:.3f} | "
                         f"{v['time_s']:.3f} |")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------------------- иллюстрации
def make_figures(rows, ups, s, out, n_ex, lp):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rs = [r for r in rows if r["scale"] == s]
    by = defaultdict(dict)
    for r in rs:
        by[r["path"]][r["method"]] = r
    methods = list(ups)
    win = 64 if s == 2 else 96

    def crop(img, x, y):
        return img[y:y + win, x:x + win, ::-1]

    def name_of(path):
        return f"{Path(path).parent.name}/{Path(path).name}"

    def run(path):
        hr = modcrop(imread(path), s)
        lr = degrade(hr, s)
        return hr, lr, {m: ups[m](lr) for m in methods}

    img_dir = out / "images" / f"x{s}"

    # ---- успешные кейсы: наибольший прирост лучшего обученного метода над бикубической
    main_m = "srcnn" if "srcnn" in methods else methods[-1]
    gains = sorted(by, key=lambda p: -(by[p][main_m]["psnr"] - by[p]["bicubic"]["psnr"])
                   if "bicubic" in by[p] else 0)[:n_ex]
    fig, axes = plt.subplots(len(gains), len(methods) + 2, figsize=(2.6 * (len(methods) + 2), 2.9 * len(gains)),
                             squeeze=False)
    for i, p in enumerate(gains):
        hr, lr, srs = run(p)
        x, y = best_window(detail_map(cv2.cvtColor(hr, cv2.COLOR_BGR2GRAY)), win)
        full = hr[..., ::-1].copy()
        cv2.rectangle(full, (x, y), (x + win, y + win), (255, 0, 0), max(2, hr.shape[1] // 200))
        axes[i, 0].imshow(full)
        axes[i, 0].set_title(name_of(p), fontsize=9)
        axes[i, 1].imshow(crop(hr, x, y))
        axes[i, 1].set_title("Эталон HR", fontsize=9)
        for j, m in enumerate(methods):
            r = by[p][m]
            axes[i, j + 2].imshow(crop(srs[m], x, y))
            t = f"{TITLES[m]}\n{r['psnr']:.2f} дБ / {r['ssim']:.3f}"
            if r["lpips"] is not None:
                t += f" / {r['lpips']:.3f}"
            axes[i, j + 2].set_title(t, fontsize=9)
            imwrite(img_dir / f"{Path(p).stem}_{m}_x{s}.png", srs[m])
        imwrite(img_dir / f"{Path(p).stem}_HR.png", hr)
        imwrite(img_dir / f"{Path(p).stem}_LR_x{s}.png", lr)
    for a in axes.ravel():
        a.axis("off")
    fig.suptitle(f"Успешные кейсы, ×{s} (PSNR / SSIM" + (" / LPIPS)" if lp else ")"), fontsize=11)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.32)
    fig.savefig(out / "figures" / f"examples_x{s}.png", dpi=130)
    plt.close(fig)

    # ---- ошибки: I рода — метод с наибольшей долей ложных деталей; II рода — с наибольшей долей потерянных
    cases = []
    fp_m = max(methods, key=lambda m: np.mean([by[p][m]["fp_rate"] for p in by]))
    fn_m = main_m
    fp_imgs = sorted(by, key=lambda p: -by[p][fp_m]["fp_rate"])[: max(1, n_ex - 1)]
    fn_imgs = sorted(by, key=lambda p: by[p][fn_m]["psnr"] - by[p]["bicubic"]["psnr"]
                     if "bicubic" in by[p] else by[p][fn_m]["fn_rate"])[:1]
    fn_imgs += [p for p in sorted(by, key=lambda p: -by[p][fn_m]["fn_rate"]) if p not in fn_imgs][: max(1, n_ex - 1) - 1]
    cases += [("FP", fp_m, p) for p in fp_imgs] + [("FN", fn_m, p) for p in fn_imgs]

    fig, axes = plt.subplots(len(cases), 4, figsize=(11, 2.9 * len(cases)), squeeze=False)
    for i, (kind, m, p) in enumerate(cases):
        hr, lr, srs = run(p)
        sr = srs[m]
        de = detail_errors(sr, hr, s)
        mask = de["fp_mask"] if kind == "FP" else de["fn_mask"]
        full_mask = np.zeros(hr.shape[:2], np.float32)
        full_mask[s:s + mask.shape[0], s:s + mask.shape[1]] = mask
        x, y = best_window(full_mask, win)
        g = cv2.cvtColor(cv2.cvtColor(hr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
        ov = (g * 0.6).astype(np.uint8)
        fpm = np.zeros(hr.shape[:2], bool)
        fnm = np.zeros(hr.shape[:2], bool)
        fpm[s:s + mask.shape[0], s:s + mask.shape[1]] = de["fp_mask"]
        fnm[s:s + mask.shape[0], s:s + mask.shape[1]] = de["fn_mask"]
        ov[fpm] = (230, 40, 40)
        ov[fnm] = (40, 90, 255)
        r = by[p][m]
        label = "Ошибка I рода (FP): ложные детали" if kind == "FP" else "Ошибка II рода (FN): потерянные детали"
        axes[i, 0].imshow(crop(hr, x, y))
        axes[i, 0].set_title(f"Эталон · {name_of(p)}", fontsize=9)
        axes[i, 1].imshow(crop(srs["bicubic"], x, y) if "bicubic" in srs else crop(lr, x // s, y // s))
        axes[i, 1].set_title("Бикубическая", fontsize=9)
        axes[i, 2].imshow(crop(sr, x, y))
        axes[i, 2].set_title(f"{TITLES[m]}: {r['psnr']:.2f} дБ", fontsize=9)
        axes[i, 3].imshow(ov[y:y + win, x:x + win])
        axes[i, 3].set_title(f"{label}\nprecision {de['precision']:.2f} · recall {de['recall']:.2f}", fontsize=8)
        imwrite(img_dir / f"error_{kind}_{Path(p).stem}_{m}_x{s}.png", sr)
    for a in axes.ravel():
        a.axis("off")
    fig.suptitle(f"Ошибочные результаты, ×{s}: красный — FP (деталь есть в SR, нет в эталоне), "
                 f"синий — FN (деталь есть в эталоне, не восстановлена)", fontsize=10)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3)
    fig.savefig(out / "figures" / f"errors_x{s}.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
