"""fig8 -- the fitted exponent session by session, all eight names.

Figure 7 follows two names at intraday resolution, which answers whether
the exponent is stable *within* a day. This one answers the other half:
whether it is stable *across* the nineteen sessions of the panel, for
every name in it.

One capture per name-session (the daily slice, the 15:30 cut), the
G-eSSVI backbone exponent fitted freely. Index ETFs above, single names
below, on a shared y axis so the two halves are directly comparable.
The horizontal rule is the gamma = 1/2 pin.

Each legend entry carries that name's median day-over-day absolute move,
which is the quantity the figure exists to show: a series that wanders
would make the pooled medians of Table 4 meaningless.

Source: results/fits.csv (`study --panel daily`).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _style import RESULTS, use_paper_style, save, note, res

use_paper_style()

IDX = ["SPY", "QQQ", "IWM", "DIA"]
SGL = ["META", "AMD", "TSLA", "GOOGL"]
PANELS = [(IDX, "Index ETFs"), (SGL, "Single names")]

# luminance-separated within each panel; hue is never the only channel
STYLE = {
    "SPY":   ("#0B4F8A", "s", "-"),
    "QQQ":   ("#4E9CD6", "D", "--"),
    "IWM":   ("#5C2D91", "P", (0, (5, 1.4))),
    "DIA":   ("#C08A00", "^", (0, (1, 1.2))),
    "META":  ("#0B4F8A", "s", "-"),
    "AMD":   ("#4E9CD6", "D", "--"),
    "TSLA":  ("#5C2D91", "P", (0, (5, 1.4))),
    "GOOGL": ("#C08A00", "^", (0, (1, 1.2))),
}

f = pd.read_csv(res("fits.csv"),
                usecols=["ticker", "date", "time", "model", "status", "essvi_gamma"])
f = f[(f.status == "ok") & (f.model == "essvi_gfree")]
g = (f.groupby(["ticker", "date"], as_index=False)
       .essvi_gamma.first()
       .rename(columns={"essvi_gamma": "gamma"}))

dates = sorted(g.date.unique())
DI = {d: i for i, d in enumerate(dates)}

fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.4), sharex=True, sharey=True)
fig.subplots_adjust(left=0.085, right=0.985, top=0.945, bottom=0.20, hspace=0.16)

steps = {}
for ax, (names, title) in zip(axes, PANELS):
    for t in names:
        d = g[g.ticker == t].sort_values("date")
        x = d.date.map(DI).values
        y = d.gamma.values.astype(float)
        steps[t] = float(np.median(np.abs(np.diff(y)))) if len(y) > 1 else np.nan
        color, marker, ls = STYLE[t]
        ax.plot(x, y, ls=ls, color=color, lw=1.0, marker=marker, ms=3.0,
                mfc="white", mew=0.8, zorder=3,
                label="%s  (step %.3f)" % (t, steps[t]))
    ax.axhline(0.5, color="0.25", lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax.set_ylabel(r"fitted $\gamma$", fontsize=8.5)
    ax.set_title(title, loc="left", fontsize=9)
    ax.legend(loc="upper right", fontsize=7, ncol=4, columnspacing=1.0,
              handlelength=2.2)

axes[1].text(0.004, 0.5, r"$\gamma=1/2$", transform=axes[1].get_yaxis_transform(),
             ha="left", va="top", fontsize=7, color="0.25", zorder=4)

lo, hi = float(g.gamma.min()), float(g.gamma.max())
pad = 0.06 * (hi - lo)
axes[0].set_ylim(lo - pad, hi + pad + 0.05)
axes[-1].set_xlim(-0.4, len(dates) - 0.6)
axes[-1].set_xticks(range(len(dates)))
axes[-1].set_xticklabels([d[5:] for d in dates], fontsize=6.8, rotation=90)
axes[-1].set_xlabel("session (one capture per name-day, the 15:30 cut)", fontsize=8)

note(fig,
     "Daily slice, %d sessions, G-eSSVI with the exponent fitted freely. "
     "\"step\" is the median absolute day-over-day move in $\\gamma$."
     % len(dates), y=0.018)

save(fig, "fig8_gamma_daily.pdf")

print("day-over-day median |dgamma|:")
for t in IDX + SGL:
    print("  %-6s %.4f   median gamma %.3f" % (t, steps[t], g[g.ticker == t].gamma.median()))
print("cross-name spread of gamma: %.3f to %.3f" % (lo, hi))
