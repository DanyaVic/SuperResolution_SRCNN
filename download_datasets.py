"""Скачивание датасетов.

  python download_datasets.py div2k        # DIV2K train HR (800 шт., ~3.5 ГБ) + valid HR (100 шт., ~450 МБ)
  python download_datasets.py flickr2k     # Flickr2K (2650 шт., ~20 ГБ) — по желанию, для лучшего качества
  python download_datasets.py benchmarks   # Set5, Set14, BSD100, Urban100 (HR) — классические тестовые наборы

Все наборы кладутся в папку datasets/.
"""
import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets"

DIV2K = {
    "DIV2K_train_HR": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "DIV2K_valid_HR": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
}
FLICKR2K = {"Flickr2K": "https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar"}

RAW = "https://raw.githubusercontent.com"
SET5 = [f"{RAW}/JingyunLiang/SwinIR/main/testsets/Set5/HR/{n}.png"
        for n in ("baby", "bird", "butterfly", "head", "woman")]


def selfexsr(name, count):
    return [f"{RAW}/jbhuang0604/SelfExSR/master/data/{name}/image_SRF_2/img_{i:03d}_SRF_2_HR.png"
            for i in range(1, count + 1)]


BENCHMARKS = {"Set5": SET5, "Set14": selfexsr("Set14", 14), "BSD100": selfexsr("BSD100", 100),
              "Urban100": selfexsr("Urban100", 100)}


def fetch(url, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return

    def hook(n, bs, total):
        if total > 0:
            sys.stdout.write(f"\r  {dst.name}: {min(n * bs / total, 1) * 100:5.1f}%")
            sys.stdout.flush()

    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(url, tmp, reporthook=hook if dst.suffix in (".zip", ".tar") else None)
    tmp.rename(dst)
    if dst.suffix in (".zip", ".tar"):
        print()


def get_archives(items):
    for name, url in items.items():
        if (DATA / name).exists():
            print(f"{name}: уже есть")
            continue
        arch = DATA / Path(url).name
        print(f"Скачиваю {name} ...")
        fetch(url, arch)
        print(f"Распаковываю {arch.name} ...")
        if arch.suffix == ".zip":
            with zipfile.ZipFile(arch) as z:
                z.extractall(DATA)
        else:
            shutil.unpack_archive(str(arch), str(DATA))
        arch.unlink()


def get_benchmarks():
    for name, urls in BENCHMARKS.items():
        out = DATA / name
        print(f"{name}: {len(urls)} изображений")
        for url in urls:
            fn = Path(url).name.replace("_SRF_2_HR", "")
            fetch(url, out / fn)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="+", choices=["div2k", "flickr2k", "benchmarks"])
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    if "div2k" in args.what:
        get_archives(DIV2K)
    if "flickr2k" in args.what:
        get_archives(FLICKR2K)
    if "benchmarks" in args.what:
        get_benchmarks()
    print("Готово:", DATA)
