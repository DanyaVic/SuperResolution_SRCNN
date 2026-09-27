"""Подготовка данных: разбиение train / val / test и нарезка обучающих HR-изображений на фрагменты.

Пример для DIV2K (рекомендуется):
  python prepare_data.py --train-dirs datasets/DIV2K_train_HR \
      --val-dir datasets/DIV2K_valid_HR --val-count 10 \
      --test-dirs Set5=datasets/Set5 Set14=datasets/Set14 BSD100=datasets/BSD100

  → data/train_y.npy (+ .json)  — фрагменты 192×192 канала Y (memmap, читается мгновенно)
  → data/splits.json            — списки файлов val и test (DIV2K_valid: 10 → val, 90 → test «DIV2K_valid_rest»)

Картинки из val и test в обучении не участвуют; дубликаты между наборами отсекаются по хешу.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from srlib.imgproc import bgr2y, imread, list_images

ROOT = Path(__file__).resolve().parent


def file_hash(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def crop_positions(length, size, stride):
    if length < size:
        return []
    pos = list(range(0, length - size + 1, stride))
    if pos[-1] != length - size:
        pos.append(length - size)
    return pos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-dirs", nargs="+", required=True, help="папки с HR-изображениями для обучения")
    ap.add_argument("--val-dir", required=True, help="папка для валидации (напр. DIV2K_valid_HR)")
    ap.add_argument("--val-count", type=int, default=10,
                    help="сколько картинок из val-dir взять в val; остальные уйдут в test (0 = все в val)")
    ap.add_argument("--test-dirs", nargs="*", default=[], help="тестовые наборы в виде Имя=папка")
    ap.add_argument("--out", default=str(ROOT / "data"))
    ap.add_argument("--size", type=int, default=192, help="размер нарезаемого фрагмента")
    ap.add_argument("--stride", type=int, default=192, help="шаг нарезки (меньше size → перекрытие)")
    ap.add_argument("--min-std", type=float, default=2.0,
                    help="отбрасывать почти однотонные фрагменты (std яркости ниже порога) — они ничему не учат")
    ap.add_argument("--max-images", type=int, default=0, help="ограничить число train-картинок (0 = все)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- разбиение
    val_all = list_images(args.val_dir)
    n_val = len(val_all) if args.val_count <= 0 else min(args.val_count, len(val_all))
    val, val_rest = val_all[:n_val], val_all[n_val:]
    tests = {}
    if val_rest:
        tests[Path(args.val_dir).name + "_rest"] = val_rest
    for spec in args.test_dirs:
        name, folder = spec.split("=", 1) if "=" in spec else (Path(spec).name, spec)
        tests[name] = list_images(folder)

    held_out = {file_hash(p) for p in val} | {file_hash(p) for ps in tests.values() for p in ps}
    train = [p for d in args.train_dirs for p in list_images(d)]
    before = len(train)
    train = [p for p in train if file_hash(p) not in held_out]
    excluded = before - len(train)
    if args.max_images:
        train = train[: args.max_images]
    print(f"train: {len(train)} изобр. (исключено дубликатов с val/test: {excluded})")
    print(f"val:   {len(val)} изобр. из {args.val_dir}")
    for k, v in tests.items():
        print(f"test:  {k}: {len(v)} изобр.")

    if not train:
        raise SystemExit("Нет обучающих изображений: проверьте --train-dirs (картинки, совпадающие с val/test, исключаются).")
    splits = {"train": [str(p) for p in train], "val": [str(p) for p in val],
              "test": {k: [str(p) for p in v] for k, v in tests.items()}}
    (out / "splits.json").write_text(json.dumps(splits, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------------- нарезка train на фрагменты (два прохода: подсчёт → запись в memmap)
    s, st = args.size, args.stride
    boxes = []
    for i, p in enumerate(tqdm(train, desc="Анализ train")):
        y = bgr2y(imread(p))
        for y0 in crop_positions(y.shape[0], s, st):
            for x0 in crop_positions(y.shape[1], s, st):
                if y[y0:y0 + s, x0:x0 + s].std() >= args.min_std:
                    boxes.append((i, y0, x0))
    if not boxes:
        raise SystemExit(f"Ни одного фрагмента {s}×{s}: изображения меньше --size? Уменьшите --size.")
    arr = np.lib.format.open_memmap(out / "train_y.npy", mode="w+", dtype=np.uint8, shape=(len(boxes), s, s))
    k = 0
    by_img = {}
    for i, y0, x0 in boxes:
        by_img.setdefault(i, []).append((y0, x0))
    for i in tqdm(sorted(by_img), desc="Нарезка"):
        y = np.clip(bgr2y(imread(train[i])).round(), 0, 255).astype(np.uint8)
        for y0, x0 in by_img[i]:
            arr[k] = y[y0:y0 + s, x0:x0 + s]
            k += 1
    arr.flush()
    meta = {"count": len(boxes), "size": s, "stride": st, "images": len(train), "channel": "Y (BT.601)"}
    (out / "train_y.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Готово: {len(boxes)} фрагментов {s}×{s} → {out / 'train_y.npy'} "
          f"({len(boxes) * s * s / 2**20:.0f} МБ)")


if __name__ == "__main__":
    main()
