"""Shared plotting style for the smile-smoothing-shootout figures.

Every figure must survive being printed in greyscale, so each model carries a
distinct *luminance*, a distinct *marker* and a distinct *linestyle/hatch*;
hue is never the only channel.
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, "..", "results").replace("\\", "/") + "/"
OUT = _HERE.replace("\\", "/") + "/"


def res(name):
    """Path of a released result, plain or gzipped."""
    p = RESULTS + name
    if not os.path.exists(p) and os.path.exists(p + ".gz"):
        p += ".gz"
    return p

# display order: the eSSVI arms first, the benchmark last
MODELS = ["essvi_sg05", "essvi_g05", "essvi_gfree", "essvi", "ssvi", "svi"]
# the practical menu (benchmark excluded) for dispersion figures
MENU = ["essvi_sg05", "essvi_g05", "essvi_gfree", "essvi", "ssvi"]
GLOBALS = ["essvi_g05", "essvi_gfree"]
LABEL = {"essvi_sg05": "S-eSSVI ($\gamma{=}1/2$)",
         "essvi_g05": "G-eSSVI ($\gamma{=}1/2$)",
         "essvi_gfree": "G-eSSVI ($\gamma$ free)",
         "essvi": "eSSVI (seq.)",
         "ssvi": "SSVI",
         "svi": "SVI (benchmark)"}

# luminance-separated palette; the two global arms share the blue family
# (solid vs dashed separates them in greyscale), the benchmark is grey.
COLOR = {"essvi_sg05": "#5C2D91", "essvi_g05": "#0B4F8A", "essvi_gfree": "#4E9CD6", "essvi": "#1E7B5E",
         "ssvi": "#C08A00", "svi": "#8A8A8A"}
MARKER = {"essvi_sg05": "P", "essvi_g05": "s", "essvi_gfree": "D", "essvi": "o", "ssvi": "^", "svi": "x"}
LS = {"essvi_sg05": (0, (5, 1.4)), "essvi_g05": "-", "essvi_gfree": "--", "essvi": "-.",
      "ssvi": (0, (1, 1.2)), "svi": (0, (4, 1.5))}
HATCH = {"essvi_sg05": "++", "essvi_g05": "///", "essvi_gfree": "\\\\\\", "essvi": "...",
         "ssvi": "xxx", "svi": ""}

# product styling for the barrier figure
PCOLOR = {"down-out P": "#7A1C6B", "up-out C": "#1E7B5E", "vanilla C": "#4D4D4D"}
PMARKER = {"down-out P": "D", "up-out C": "v", "vanilla C": "o"}
PLS = {"down-out P": "-", "up-out C": "--", "vanilla C": (0, (1, 1.2))}
PLABEL = {"down-out P": "down-and-out put", "up-out C": "up-and-out call",
          "vanilla C": "vanilla call (control)"}


def use_paper_style():
    mpl.rcParams.update({
        "pdf.fonttype": 42,            # embed TrueType, keep text selectable
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "0.6",
        "legend.borderpad": 0.4,
        "axes.linewidth": 0.7,
        "axes.grid": True,
        "grid.color": "0.88",
        "grid.linewidth": 0.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.1,
        "savefig.bbox": "standard",
        "figure.dpi": 110,
    })


def note(fig, text, y=0.012, x=0.012, size=6.8):
    """A small provenance / caveat footnote pinned inside the bottom margin.

    Figures are saved with an explicit size and an explicit axes rectangle
    (savefig.bbox is "standard", never "tight"), so the PDF page is exactly
    the figsize asked for and the note can never push the page wider.
    """
    fig.text(x, y, text, ha="left", va="bottom", fontsize=size, color="0.28",
             linespacing=1.4)


def save(fig, name):
    path = OUT + name
    fig.savefig(path)
    print("wrote", path)
    plt.close(fig)
