"""fig4 -- what the Dupire clamp had to absorb.

Formerly a four-panel admissibility figure. Panels (a) and (b) repeated
Table 5 bar for bar, and panel (d) plotted the self-certificate that
table no longer carries, so all three are gone. What is left is the one
quantity with no table of its own at name-day resolution: how much of
the grid the $0.01 \\le \\sigma_loc \\le 3.0$ clamp had to touch, and how
often a surface reached the upper bound anywhere.

Both series are counts or medians over name-days. min_g, the worst
butterfly density on the grid, is deliberately not plotted: degenerate
benchmark slices give it neither a usable scale nor a meaningful mean.

Source: results/localvol.csv (`study --panel daily`).
"""
import numpy as np
import matplotlib.pyplot as plt

import _data as D
from _style import COLOR, HATCH, LABEL, MODELS, use_paper_style, save, note

use_paper_style()

lv = D.load_localvol("daily")
N = int((lv.model == "svi").sum())

SHORT = {"essvi_sg05": "S-eSSVI\n$\\gamma{=}1/2$",
         "essvi_g05": "G-eSSVI\n$\\gamma{=}1/2$",
         "essvi_gfree": "G-eSSVI\n$\\gamma$ free",
         "essvi": "eSSVI\nseq.", "ssvi": "SSVI", "svi": "SVI\nbench."}

X = np.arange(len(MODELS))
W = 0.34

fig = plt.figure(figsize=(7.0, 3.1))
ax = fig.add_axes([0.085, 0.235, 0.895, 0.60])

clamp = [100 * lv[lv.model == m].clamped_fraction.median() for m in MODELS]
pin = [100 * (lv[lv.model == m].lv_max >= 2.999).mean() for m in MODELS]
ytop = 1.28 * max(max(clamp), max(pin), 1.0)

for j, (vals, off, alpha, lab) in enumerate(
        [(clamp, -W / 2, 1.0, r"median % of grid clamped"),
         (pin, W / 2, 0.42, r"% of name-days pinning $\sigma_{\rm loc}=3.0$")]):
    for i, mod in enumerate(MODELS):
        ax.bar(X[i] + off, vals[i], W, color=COLOR[mod], alpha=alpha,
               edgecolor="black", linewidth=0.5,
               hatch=HATCH[mod] if j == 0 else None,
               label=lab if i == 0 else None, zorder=3)
        ax.text(X[i] + off, vals[i] + 0.022 * ytop, "%.2f" % vals[i],
                ha="center", va="bottom", fontsize=6.8, color="0.15")

ax.set_xticks(X)
ax.set_xticklabels([SHORT[m] for m in MODELS], fontsize=7.2)
ax.set_ylim(0, ytop)
ax.set_axisbelow(True)
ax.set_ylabel("percent", fontsize=8.4)

leg = ax.legend(loc="upper left", fontsize=7, borderaxespad=0.4,
                handlelength=1.5, labelspacing=0.3)
leg.get_frame().set_linewidth(0.5)

# the SVI pinning bar reaches ~35, so the inset sits left of it rather
# than in the top-right corner it used to occupy
ax.text(0.345, 0.97,
        "median $(\\sigma_{\\rm loc}^{\\min},\\sigma_{\\rm loc}^{\\max})$\n"
        + "\n".join("%-6s (%.3f, %.3f)" % (LABEL[m],
                                           lv[lv.model == m].lv_min.median(),
                                           lv[lv.model == m].lv_max.median())
                    for m in MODELS),
        transform=ax.transAxes, fontsize=6.2, color="0.15", va="top", ha="left",
        linespacing=1.4, family="monospace",
        bbox=dict(fc="white", ec="0.7", lw=0.5, pad=2.6))

fig.text(0.03, 0.985, "What the Dupire clamp absorbed",
         fontsize=10.5, ha="left", va="top")
fig.text(0.03, 0.925,
         "Daily slice, %d name-days per arm. Clamp fractions are medians over name-days; the pinning bar counts name-days "
         "whose\nlocal volatility reaches the upper clamp anywhere on the grid." % N,
         fontsize=7.5, ha="left", va="top", color="0.25", linespacing=1.5)

note(fig,
     "Counts and medians only. min_g, the worst butterfly density on the grid, is not plotted and must not be quoted as a "
     "headline:\ndegenerate benchmark slices give it neither a usable scale nor a meaningful mean.",
     y=0.02)

save(fig, "fig4_clamp.pdf")
