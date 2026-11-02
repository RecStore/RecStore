"""Shared drawing helpers for the storage-layer schematic figures.

Every figure is authored in pixel-like coordinates (origin at top-left) so the
geometry reads the same way the SVG is laid out. Output is vector SVG plus a PNG
preview for quick review.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D

_FONT_DIR = "/home/hadoop-quanta/.local/share/fonts/noto-sc"
for _f in ("NotoSansSC-Regular.otf", "NotoSansSC-Medium.otf", "NotoSansSC-Bold.otf"):
    fm.fontManager.addfont(f"{_FONT_DIR}/{_f}")

matplotlib.rcParams["svg.fonttype"] = "path"
matplotlib.rcParams["font.family"] = "Noto Sans SC"
matplotlib.rcParams["axes.unicode_minus"] = False

INK = "#1f2933"
MUT = "#5b6570"
LINE = "#c3ccd6"
HEAD = "#2c3e50"
BLUE = "#2f6feb"
BLUE_F = "#eaf1fe"
GREEN = "#2e8555"
GREEN_F = "#e8f5ec"
ORANGE = "#c26401"
ORANGE_F = "#fdf1e0"
PURPLE = "#7048c4"
PURPLE_F = "#f1ecfb"
RED = "#c0392b"
RED_F = "#fbeae8"
TEAL = "#0f7b8a"
TEAL_F = "#e2f2f4"
GRAY_F = "#f1f4f7"
PAD_F = "#ffffff"


def new_fig(W, H, dpi=100):
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def rbox(ax, x, y, w, h, fc="white", ec=LINE, lw=1.4, r=8, z=2, ls="solid", alpha=1.0):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
                       linestyle=ls, alpha=alpha, mutation_aspect=1)
    ax.add_patch(p)
    return p


def sbox(ax, x, y, w, h, fc="white", ec=LINE, lw=1.2, z=2, alpha=1.0):
    p = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw,
                  zorder=z, alpha=alpha)
    ax.add_patch(p)
    return p


def T(ax, x, y, s, size=11, color=INK, ha="left", va="top", w="normal",
      style="normal", z=6, alpha=1.0, rot=0):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
                   fontweight=w, style=style, zorder=z, alpha=alpha,
                   rotation=rot)


def arrow(ax, x1, y1, x2, y2, color=HEAD, lw=1.6, style="-|>", ms=13,
          z=4, ls="solid", conn="arc3,rad=0"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                        mutation_scale=ms, lw=lw, color=color, zorder=z,
                        linestyle=ls, connectionstyle=conn,
                        shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    return a


def vline(ax, x, y1, y2, color=LINE, lw=1.0, ls=(0, (2, 3)), z=3):
    ax.add_line(Line2D([x, x], [y1, y2], color=color, lw=lw, linestyle=ls,
                       zorder=z))


def hline(ax, x1, x2, y, color=LINE, lw=1.0, ls=(0, (2, 3)), z=3):
    ax.add_line(Line2D([x1, x2], [y, y], color=color, lw=lw, linestyle=ls,
                       zorder=z))


def bitstrip(ax, x, y, w, h, fields, z=3, tick_size=9, seg_pad=0.0):
    """fields: list of dicts {f, label, fc, ec, tc, lw, ticks:[(pos,'txt')]}.

    Draws contiguous segments across [x, x+w]; the boundary ticks under the bar
    carry the bit indices.
    """
    cx = x
    segs = []
    for fd in fields:
        sw = w * fd["f"]
        sbox(ax, cx, y, sw, h, fc=fd.get("fc", GRAY_F), ec=fd.get("ec", LINE),
             lw=fd.get("lw", 1.3), z=z)
        segs.append((cx, sw, fd))
        cx += sw
    for sx, sw, fd in segs:
        lab = fd.get("label")
        if lab:
            T(ax, sx + sw / 2, y + h / 2, lab, size=fd.get("size", 10.5),
              color=fd.get("tc", INK), ha="center", va="center",
              w=fd.get("w", "normal"), z=z + 1)
    for sx, sw, fd in segs:
        for pos, txt in fd.get("ticks", []):
            ax.add_line(Line2D([sx + sw * pos, sx + sw * pos], [y + h, y + h + 6],
                               color=MUT, lw=1.0, zorder=z + 1))
            T(ax, sx + sw * pos, y + h + 7, txt, size=tick_size, color=MUT,
              ha="center", va="top", z=z + 1)
    return segs


def save(fig, name, outdir):
    fig.savefig(f"{outdir}/{name}.svg")
    fig.savefig(f"{outdir}/{name}.png", dpi=160)
    plt.close(fig)
    print("wrote", name)
