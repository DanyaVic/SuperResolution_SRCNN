"""Единый стиль графиков: фиксированный цвет за каждым методом (не зависит от порядка/числа серий),
тонкие линии, приглушённые оси и сетка, подписи на русском (шрифт DejaVu Sans есть в matplotlib)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Категориальная палитра (проверена на различимость при дальтонизме: первые 3 слота — все пары).
# Бикубическая интерполяция — это базовая линия, поэтому нейтральный серый, а не цвет серии.
COLORS = {"srcnn": "#2a78d6", "esrgan": "#eb6834", "realesrgan": "#1baf7a", "bicubic": "#8a8984"}
TITLES = {"bicubic": "Бикубическая", "srcnn": "SRCNN", "esrgan": "ESRGAN", "realesrgan": "Real-ESRGAN"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def apply():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelcolor": INK2, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.size": 0, "ytick.major.size": 0, "text.color": INK, "legend.frameon": False,
        "lines.linewidth": 2, "lines.markersize": 5, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.bbox": "tight", "savefig.dpi": 160,
    })
