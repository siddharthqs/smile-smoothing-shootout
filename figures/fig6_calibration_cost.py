"""fig6 -- what each calibration costs.

Distribution of the per-capture calibration wall clock, per arm, on the
intraday slice (box = IQR, whisker = 5th-95th percentile, log axis). The two
global arms solve one joint problem over all expiries; the per-expiry arms
solve many small ones; SSVI solves one three-parameter problem (rho, eta,
gamma -- the theta pillars are read off the quotes, not fitted).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _style import RESULTS, MODELS, COLOR, LABEL, use_paper_style, save, note, res

use_paper_style()

t = pd.read_csv(res("intraday/timing.csv"))
t = t[np.isfinite(t.cold_ms)]

fig = plt.figure(figsize=(7.0, 4.0))
ax = fig.add_axes([0.165, 0.250, 0.800, 0.610])

data, cols = [], []
for m in MODELS:
    x = t[t.model == m].cold_ms.dropna().values
    data.append(x)
    cols.append(COLOR[m])
bp = ax.boxplot(data, vert=False, widths=0.62, whis=(5, 95),
                showfliers=False, patch_artist=True,
                medianprops=dict(color="black", lw=1.2))
for patch, c in zip(bp["boxes"], cols):
    patch.set_facecolor(c)
    patch.set_alpha(0.55)
    patch.set_edgecolor("black")
    patch.set_linewidth(0.6)
for j, x in enumerate(data):
    ax.text(np.median(x), j + 1.42, "%.0f ms" % np.median(x),
            ha="center", va="bottom", fontsize=6.8, color="0.15")
ax.set_yticks(range(1, len(MODELS) + 1))
ax.set_yticklabels([LABEL[m] for m in MODELS], fontsize=8)
ax.set_xscale("log")
ax.set_xlabel("calibration wall clock per capture (ms, log scale)",
              fontsize=8.4)
ax.invert_yaxis()

fig.text(0.03, 0.985, "Calibration cost per capture, in milliseconds",
         fontsize=10.5, ha="left", va="top")

gm = t[t.model == "essvi_g05"]
gf = t[t.model == "essvi_gfree"]
IDX = ["SPY", "QQQ", "IWM", "DIA"]
gmi = gm[gm.ticker.isin(IDX)]
gms = gm[~gm.ticker.isin(IDX)]
spy_g = gm[gm.ticker == "SPY"]
spy_f = gf[gf.ticker == "SPY"]
note(fig,
     "Intraday slice, %d captures per arm, both families pooled -- the joint problem scales with the expiry count, so the\n"
     "global arms' boxes straddle their two regimes: median %.0f ms on the ~30-expiry index chains (dim %.0f) against %.0f ms\n"
     "on the single names (dim %.0f). Where the pin is misspecified the optimizer grinds: on SPY the pinned arm runs a median\n"
     "%.0f iterations vs the free arm's %.0f."
     % (len(gm), gmi.cold_ms.median(), gmi.dim.median(), gms.cold_ms.median(), gms.dim.median(),
        spy_g.cold_iterations.median(), spy_f.cold_iterations.median()))

save(fig, "fig6_calibration_cost.pdf")
