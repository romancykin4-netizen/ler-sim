import matplotlib as mpl

NM = 1e9
NM3 = 1e27

RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 0.8, "patch.linewidth": 0.6,
    "axes.prop_cycle": mpl.cycler(color=["k", "0.5", "#1b3a6b", "#8c1c13"]),
    "axes.grid": True, "axes.grid.which": "major",
    "grid.color": "0.85", "grid.linewidth": 0.35, "grid.linestyle": "-",
    "axes.axisbelow": True,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "xtick.minor.size": 1.6, "ytick.minor.size": 1.6,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "legend.frameon": False, "legend.handlelength": 1.8,
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": True,
}


def apply():
    mpl.rcParams.update(RC)


def tag(ax, s):
    ax.text(0.02, 0.97, f"({s})", transform=ax.transAxes, va="top", ha="left",
            fontweight="bold", bbox=dict(fc="w", ec="none", pad=0.6))
