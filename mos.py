"""Субъективная оценка качества (MOS, Mean Opinion Score) по методике ACR (ITU-T P.910 / BT.500).

1) Подготовка: для N тестовых изображений строятся результаты всех методов, файлы перемешиваются
   и анонимизируются (оценщик не знает, какой метод перед ним):
     python mos.py prepare --images datasets/Set14 --scale 4 --n 10
2) Оценка (каждый оценщик отдельно, порядок показа свой для каждого):
     python mos.py rate --rater ivan
   Клавиши 1–5: 1 — плохо, 2 — посредственно, 3 — удовлетворительно, 4 — хорошо, 5 — отлично;
   ← (Backspace) — назад, q — сохранить и выйти (можно продолжить позже).
3) Итог — MOS и 95% доверительный интервал по каждому методу:
     python mos.py summary
Рекомендуется ≥ 10 оценщиков, экран с одинаковыми настройками, изображения показываются в масштабе 1:1.
"""
import argparse
import csv
import json
import random
import zlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from srlib.imgproc import degrade, imread, imwrite, list_images, modcrop
from srlib.ui import fit, put_text
from srlib.upscalers import METHODS, build_upscaler

ROOT = Path(__file__).resolve().parent
MOS_DIR = ROOT / "results" / "mos"
LABELS = {1: "Плохо", 2: "Посредственно", 3: "Удовлетворительно", 4: "Хорошо", 5: "Отлично"}


def prepare(args):
    files = list_images(args.images)[: args.n]
    stim = MOS_DIR / "stimuli"
    stim.mkdir(parents=True, exist_ok=True)
    key, all_variants = {}, []
    ups = {m: build_upscaler(m, args.scale, args.device, tile=256) for m in args.methods}
    rng = random.Random(args.seed)
    for f in files:
        hr = modcrop(imread(f), args.scale)
        lr = degrade(hr, args.scale)
        variants = [(m, ups[m](lr)) for m in args.methods] + ([("reference", hr)] if args.with_reference else [])
        all_variants += [(m, f.name, img) for m, img in variants]
        print(f"{f.name}: {len(variants)} вариантов")
    rng.shuffle(all_variants)  # имена файлов не должны выдавать метод
    for idx, (m, image, img) in enumerate(all_variants):
        name = f"s{idx:04d}.png"
        imwrite(stim / name, img)
        key[name] = {"method": m, "image": image, "scale": args.scale}
    (MOS_DIR / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Готово: {len(key)} стимулов в {stim}. Ключ (не показывать оценщикам): {MOS_DIR / 'key.json'}")


def rate(args):
    key = json.loads((MOS_DIR / "key.json").read_text(encoding="utf-8"))
    out = MOS_DIR / f"ratings_{args.rater}.csv"
    done = {}
    if out.exists():
        done = {r["stimulus"]: int(r["score"]) for r in csv.DictReader(open(out, encoding="utf-8"))}
    names = list(key)
    random.Random(zlib.crc32(args.rater.encode())).shuffle(names)
    i = next((k for k, n in enumerate(names) if n not in done), len(names))
    win = "MOS"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    while i < len(names):
        img, _ = fit(imread(MOS_DIR / "stimuli" / names[i]), 1600, 900)
        img = img.copy()
        put_text(img, [f"Изображение {i + 1} из {len(names)}. Оцените качество: 1–5",
                       "  ".join(f"{k} — {v}" for k, v in LABELS.items())], (8, 8), size=16)
        cv2.imshow(win, img)
        k = cv2.waitKey(0) & 0xFF
        if k in (ord("q"), 27):
            break
        if k in (8, 81, 2) and i > 0:
            i -= 1
            continue
        if ord("1") <= k <= ord("5"):
            done[names[i]] = k - ord("0")
            i += 1
            with open(out, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["stimulus", "score"])
                w.writerows(done.items())
    cv2.destroyAllWindows()
    print(f"Сохранено оценок: {len(done)}/{len(names)} → {out}")


def summary(_):
    key = json.loads((MOS_DIR / "key.json").read_text(encoding="utf-8"))
    scores = defaultdict(list)
    raters = sorted(MOS_DIR.glob("ratings_*.csv"))
    for f in raters:
        for r in csv.DictReader(open(f, encoding="utf-8")):
            scores[key[r["stimulus"]]["method"]].append(int(r["score"]))
    lines = [f"# MOS (оценщиков: {len(raters)})", "", "| Метод | N оценок | MOS | 95% ДИ |", "|---|---|---|---|"]
    for m, v in sorted(scores.items(), key=lambda kv: -np.mean(kv[1])):
        v = np.array(v, float)
        ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        lines.append(f"| {m} | {len(v)} | {v.mean():.2f} | ±{ci:.2f} |")
    text = "\n".join(lines)
    (MOS_DIR / "mos_summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--images", required=True)
    p.add_argument("--scale", type=int, default=4, choices=[2, 4])
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--with-reference", action="store_true", help="добавить оригиналы (верхняя «якорная» оценка)")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("rate")
    r.add_argument("--rater", required=True)
    sub.add_parser("summary")
    a = ap.parse_args()
    {"prepare": prepare, "rate": rate, "summary": summary}[a.cmd](a)
