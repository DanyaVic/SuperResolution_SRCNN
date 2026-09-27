"""Графики для отчёта — строятся только из реальных логов обучения (runs/) и результатов теста (results/).

  python make_report_figures.py      → results/figures/report_*.png
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from srlib import plotstyle
from srlib.plotstyle import COLORS, INK2, TITLES, plt

ROOT = Path(__file__).resolve().parent
RUNS, RES = ROOT / "runs", ROOT / "results"
FIG = RES / "figures"
SCALE_COLORS = {2: "#2a78d6", 4: "#eb6834"}


def read_csv(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def training():
    scales = [s for s in (2, 4) if (RUNS / f"srcnn_x{s}" / "train_log.csv").exists()]
    if not scales:
        return
    fig, ax = plt.subplots(1, 1 + len(scales), figsize=(4.6 * (1 + len(scales)), 3.8))
    for s in scales:
        rows = read_csv(RUNS / f"srcnn_x{s}" / "train_log.csv")
        it = np.array([int(r["iter"]) for r in rows])
        loss = np.array([float(r["train_loss"]) for r in rows])
        k = max(len(loss) // 30, 1)
        sm = np.convolve(loss, np.ones(k) / k, mode="valid")
        ax[0].plot(it, loss, color=SCALE_COLORS[s], alpha=0.2, lw=1)
        ax[0].plot(it[k - 1:], sm, color=SCALE_COLORS[s], label=f"×{s}")
        ax[0].annotate(f"×{s}", (it[-1], sm[-1]), xytext=(4, 0), textcoords="offset points", va="center",
                       color=INK2, fontsize=9)
    ax[0].set(title="MSE на обучении", xlabel="итерация", yscale="log")
    ax[0].legend(loc="upper right")
    for i, s in enumerate(scales, 1):
        rows = read_csv(RUNS / f"srcnn_x{s}" / "train_log.csv")
        summ = json.loads((RUNS / f"srcnn_x{s}" / "summary.json").read_text(encoding="utf-8"))
        vit = [int(r["iter"]) for r in rows if r["val_psnr"]]
        vp = [float(r["val_psnr"]) for r in rows if r["val_psnr"]]
        a = ax[i]
        a.plot(vit, vp, "o-", color=COLORS["srcnn"], label="SRCNN", markersize=6)
        a.axhline(summ["bicubic_val_psnr"], color=COLORS["bicubic"], ls="--", lw=1.5, label="Бикубическая")
        j = int(np.argmax(vp))
        a.annotate(f"{vp[j]:.2f} дБ\n(+{vp[j] - summ['bicubic_val_psnr']:.2f})", (vit[j], vp[j]),
                   xytext=(-6, -32), textcoords="offset points", ha="right", color=INK2, fontsize=9)
        a.annotate(f"{summ['bicubic_val_psnr']:.2f} дБ", (vit[0], summ["bicubic_val_psnr"]), xytext=(0, 5),
                   textcoords="offset points", color=INK2, fontsize=9)
        a.set(title=f"PSNR на валидации, ×{s}", xlabel="итерация", ylabel="дБ")
        lo = min(min(vp), summ["bicubic_val_psnr"])
        a.set_ylim(lo - 0.3, max(vp) + 0.4)
        a.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(FIG / "report_training.png")
    plt.close(fig)


def ablation():
    p = RUNS / "ablation_lr_x4_600it.csv"
    if not p.exists():
        return
    rows = [r for r in read_csv(p) if not r["config"].startswith("#")]
    names = {"paper_lr": "lr 1e-4 / 1e-5 (как в статье)", "high_lr": "lr 1e-3 / 1e-4 (выбрано)",
             "equal_lr": "lr 5e-4 для всех слоёв", "residual": "residual + lr 1e-4 / 1e-5"}
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, a = plt.subplots(figsize=(7.2, 4))
    x = [200, 400, 600]
    finals = []
    for r, c in zip([r for r in rows if r["config"] in names], cols):
        y = [float(r["psnr_200"]), float(r["psnr_400"]), float(r["psnr_600"])]
        a.plot(x, y, "o-", color=c, label=names[r["config"]], markersize=6)
        finals.append(y[-1])
    # подписи конечных значений без наложения: раздвигаем по вертикали не меньше чем на 0.09 дБ
    order = np.argsort(finals)
    pos = [finals[i] for i in order]
    for k in range(1, len(pos)):
        pos[k] = max(pos[k], pos[k - 1] + 0.09)
    for i, yy in zip(order, pos):
        a.annotate(f"{finals[i]:.2f}", (x[-1], finals[i]), xytext=(612, yy), textcoords="data", va="center",
                   fontsize=9, color=INK2)
    bic = float(next(r for r in rows if r["config"] == "bicubic")["psnr_600"])
    a.axhline(bic, color=COLORS["bicubic"], ls="--", lw=1.5, label=f"Бикубическая ({bic:.2f})")
    a.set(title="Влияние скорости обучения: SRCNN ×4, первые 600 итераций (Set5)", xlabel="итерация",
          ylabel="PSNR, дБ", xticks=x, xlim=(170, 660))
    a.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "report_ablation_lr.png")
    plt.close(fig)


def gains():
    p = RES / "summary.json"
    if not p.exists():
        return
    summ = json.loads(p.read_text(encoding="utf-8"))
    scales = [k for k in ("x2", "x4") if k in summ]
    all_methods = [m for m in ("srcnn", "esrgan", "realesrgan")
                   if any(k.endswith("/" + m) for sc in scales for k in summ[sc])]
    fig, ax = plt.subplots(1, len(scales), figsize=(6.2 * len(scales), 4.4), squeeze=False, sharey=True)
    w = 0.8 / len(all_methods)
    for a, sc in zip(ax[0], scales):
        d = summ[sc]
        sets = list(dict.fromkeys(k.split("/")[0] for k in d))
        for j, m in enumerate(all_methods):
            vals = [d[f"{st}/{m}"]["psnr"] - d[f"{st}/bicubic"]["psnr"] if f"{st}/{m}" in d else np.nan
                    for st in sets]
            xs = np.arange(len(sets)) + (j - (len(all_methods) - 1) / 2) * w
            a.bar(xs, vals, w * 0.9, color=COLORS[m], label=TITLES[m])
            for xx, v in zip(xs, vals):
                if np.isnan(v):
                    a.annotate("нет модели ×2", (xx, 0), xytext=(0, 4), textcoords="offset points", ha="center",
                               fontsize=7.5, color=INK2, rotation=90, va="bottom")
                else:
                    a.annotate(f"{v:+.2f}", (xx, v), xytext=(0, 3 if v >= 0 else -11), textcoords="offset points",
                               ha="center", fontsize=8.5, color=INK2)
        a.axhline(0, color=INK2, lw=0.8)
        n = {st: d[f"{st}/bicubic"]["n"] for st in sets}
        a.set_xticks(range(len(sets)), [f"{st.replace('_rest', '')} ({n[st]} изобр.)" for st in sets])
        a.set(title=f"×{sc[1:]}", ylabel="ΔPSNR к бикубической, дБ")
        a.grid(axis="x", visible=False)
    h, l = ax[0][-1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(all_methods), bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Прирост PSNR относительно бикубической интерполяции", y=1.1, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "report_psnr_gain.png")
    plt.close(fig)


def tradeoff():
    p = RES / "summary.json"
    if not p.exists():
        return
    summ = json.loads(p.read_text(encoding="utf-8"))
    rows = read_csv(RES / "metrics_per_image.csv")
    scales = [k for k in ("x2", "x4") if k in summ]
    fig, ax = plt.subplots(1, len(scales), figsize=(5.6 * len(scales), 4.6), squeeze=False)
    for a, sc in zip(ax[0], scales):
        s = int(sc[1:])
        agg = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if int(r["scale"]) == s:
                agg[r["method"]]["fp"].append(1 - float(r["detail_precision"]))
                agg[r["method"]]["fn"].append(1 - float(r["detail_recall"]))
                agg[r["method"]]["psnr"].append(float(r["psnr"]))
        for m, v in agg.items():
            x, y = 100 * np.mean(v["fp"]), 100 * np.mean(v["fn"])
            a.scatter(x, y, s=90, color=COLORS[m], edgecolor="white", linewidth=2, zorder=3)
            a.annotate(f"{TITLES[m]}\n{np.mean(v['psnr']):.2f} дБ", (x, y), xytext=(8, 4),
                       textcoords="offset points", fontsize=9, color=INK2)
        a.set(title=f"Ошибки восстановления деталей, ×{s}", xlabel="Ложные детали, % (ошибка I рода, FP)",
              ylabel="Потерянные детали, % (ошибка II рода, FN)")
        xs_ = [100 * np.mean(v["fp"]) for v in agg.values()]
        ys_ = [100 * np.mean(v["fn"]) for v in agg.values()]
        a.set_xlim(0, max(xs_) * 1.35)
        a.set_ylim(0, max(ys_) * 1.2)
        a.annotate("↙ лучше", (0.02, 0.03), xycoords="axes fraction", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(FIG / "report_errors_tradeoff.png")
    plt.close(fig)


def per_image():
    full = RES / "full_bicubic_srcnn" / "metrics_per_image.csv"  # все изображения наборов (без GAN-моделей)
    p = full if full.exists() else RES / "metrics_per_image.csv"
    if not p.exists():
        return
    rows = read_csv(p)
    idx = {(r["scale"], r["set"], r["image"], r["method"]): float(r["psnr"]) for r in rows}
    scales = sorted({int(r["scale"]) for r in rows})
    fig, ax = plt.subplots(1, len(scales), figsize=(6 * len(scales), 4), squeeze=False)
    rng = np.random.default_rng(0)
    for a, s in zip(ax[0], scales):
        sets = list(dict.fromkeys(r["set"] for r in rows if int(r["scale"]) == s))
        counts = []
        for i, st in enumerate(sets):
            g = [idx[(str(s), st, im, "srcnn")] - idx[(str(s), st, im, "bicubic")]
                 for (sc, se, im, m) in idx if sc == str(s) and se == st and m == "srcnn"]
            xs = i + rng.uniform(-0.18, 0.18, len(g))
            a.scatter(xs, g, s=16, color=COLORS["srcnn"], alpha=0.55, edgecolor="none")
            a.plot([i - 0.28, i + 0.28], [np.median(g)] * 2, color="#0b0b0b", lw=2)
            a.annotate(f"медиана {np.median(g):+.2f}\nмин {min(g):+.2f}", (i, max(g)), xytext=(0, 6),
                       textcoords="offset points", fontsize=8.5, color=INK2, ha="center", va="bottom")
            counts.append(len(g))
        a.axhline(0, color=COLORS["bicubic"], ls="--", lw=1.2)
        a.set_xticks(range(len(sets)), [f"{x.replace('_rest', '')} ({c})" for x, c in zip(sets, counts)])
        a.set_xlim(-0.5, len(sets) - 0.5)
        a.set_ylim(min(0, a.get_ylim()[0]), a.get_ylim()[1] * 1.12)
        a.grid(axis="x", visible=False)
        a.set(title=f"Прирост SRCNN к бикубической по изображениям, ×{s}", ylabel="ΔPSNR, дБ")
    fig.tight_layout()
    fig.savefig(FIG / "report_per_image_gain.png")
    plt.close(fig)


if __name__ == "__main__":
    plotstyle.apply()
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (training, ablation, gains, tradeoff, per_image):
        f()
        print("ok:", f.__name__)
    print("Готово:", FIG)
