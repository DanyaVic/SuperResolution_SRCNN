"""Схема архитектуры SRCNN для отчёта (matplotlib, без внешних зависимостей)."""
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from srlib import plotstyle
from srlib.plotstyle import COLORS, INK, INK2, plt


def box(ax, x, y, w, h, title, sub="", fc="#f4f3f0", ec="#c9c8c2", tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2 + (0.13 if sub else 0), title, ha="center", va="center", fontsize=10.5,
            color=tc, weight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.2, sub, ha="center", va="center", fontsize=8.5, color=INK2)


def arrow(ax, p, q, label=""):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, color=INK2, lw=1.3,
                                 shrinkA=2, shrinkB=2, connectionstyle="arc3,rad=0"))
    if label:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.12, label, ha="center", fontsize=8.5, color=INK2)


def draw(path):
    plotstyle.apply()
    fig, ax = plt.subplots(figsize=(13, 3.9))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.9)
    ax.axis("off")
    blue_fc, blue_ec = "#e6effb", COLORS["srcnn"]
    # верхний ряд: подготовка
    box(ax, 0.1, 2.5, 1.6, 1.0, "LR", "RGB, H×W")
    box(ax, 2.2, 2.5, 2.0, 1.0, "Бикубическое ↑s", "sH × sW")
    box(ax, 4.7, 2.5, 1.8, 1.0, "RGB → YCbCr", "BT.601")
    arrow(ax, (1.7, 3.0), (2.2, 3.0))
    arrow(ax, (4.2, 3.0), (4.7, 3.0))
    # нижний ряд: сеть на канале Y
    box(ax, 0.1, 0.4, 1.4, 1.0, "Y", "1 канал")
    box(ax, 2.0, 0.4, 2.1, 1.0, "Conv 9×9, 64", "+ ReLU · извлечение", fc=blue_fc, ec=blue_ec)
    box(ax, 4.55, 0.4, 2.1, 1.0, "Conv 5×5, 32", "+ ReLU · отображение", fc=blue_fc, ec=blue_ec)
    box(ax, 7.1, 0.4, 2.1, 1.0, "Conv 5×5, 1", "реконструкция", fc=blue_fc, ec=blue_ec)
    box(ax, 9.7, 0.4, 1.3, 1.0, "Y′", "1 канал")
    arrow(ax, (1.5, 0.9), (2.0, 0.9))
    arrow(ax, (4.1, 0.9), (4.55, 0.9))
    arrow(ax, (6.65, 0.9), (7.1, 0.9))
    arrow(ax, (9.2, 0.9), (9.7, 0.9))
    ax.add_patch(FancyArrowPatch((5.6, 2.5), (0.8, 1.4), arrowstyle="-|>", mutation_scale=12, color=INK2, lw=1.3,
                                 connectionstyle="arc,angleA=-90,angleB=90,armA=25,armB=25,rad=12"))
    ax.text(4.3, 1.72, "канал Y", fontsize=8.5, color=INK2)
    # объединение
    box(ax, 7.5, 2.5, 2.2, 1.0, "Cb, Cr", "бикубические")
    arrow(ax, (6.5, 3.0), (7.5, 3.0))
    box(ax, 11.4, 1.45, 1.5, 1.0, "SR", "RGB, sH×sW", fc="#e6effb", ec=COLORS["srcnn"])
    arrow(ax, (11.0, 0.9), (11.55, 1.45))
    arrow(ax, (9.7, 3.0), (11.55, 2.45))
    ax.text(10.3, 1.9, "YCbCr → RGB", fontsize=8.5, color=INK2)
    ax.text(5.6, 0.08, "SRCNN: 57 281 параметр, рецептивное поле 17×17 пикселей, «same»-свёртки с replicate-паддингом",
            ha="center", fontsize=9, color=INK2)
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    import sys

    draw(sys.argv[1] if len(sys.argv) > 1 else "srcnn_architecture.png")
