"""fig7 -- the fitted exponent as a time series, one index and one single name.

Table 4 reports the exponent's median and IQR per name; this figure shows
its path. Every intraday capture over the panel window, for SPY (index) and
TSLA (single name), for the two fits that let the exponent move:

    G-eSSVI (gamma free)      the joint global eSSVI's backbone exponent
    SSVI, exponent free       the unconstrained SSVI refit, warm-started
                              from the constrained arm (a diagnostic: where
                              the data takes the exponent when the guarantee
                              is not imposed)

The constrained SSVI arm is not drawn: it sits on its bound gamma = 1/2 by
construction wherever the data asks for a slower decay, which the note
quantifies. Vertical rules are overnights; the horizontal rule is the pin.

Each panel carries the within-session and overnight step sizes, so that the
reader can tell a slowly drifting series from one that jumps day to day.

Source: results/gamma_ext.csv (`study --panel gamma`), the same ingest and
the same fits as the intraday panel.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _style import COLOR, RESULTS, use_paper_style, save, note, res

use_paper_style()

NAMES = [("SPY", "S&P 500 ETF (index)"), ("TSLA", "Tesla (single name)")]
SERIES = [("essvi_gfree", "G-eSSVI ($\\gamma$ free)", COLOR["essvi_gfree"], "-"),
          ("ssvi_free", "SSVI, exponent unconstrained (diagnostic)", COLOR["ssvi"], "-")]

g = pd.read_csv(res("gamma_ext.csv"))
g = g[(g.status == "ok") & g.ticker.isin([n for n, _ in NAMES])].copy()
g["time"] = g["time"].astype(int)
dates = sorted(g.date.unique())
DI = {d: i for i, d in enumerate(dates)}


def xcoord(df):
    mins = (df.time // 100) * 60 + (df.time % 100) - (9 * 60 + 30)
    return df.date.map(DI).values + np.clip(mins.values / 390.0, 0.0, 1.0)


def draw(ax, q, color, ls, label):
    """One path, broken across sessions and across gaps longer than 25 min."""
    first = True
    for _, d in q.groupby("date"):
        d = d.sort_values("time")
        x = xcoord(d)
        y = d.gamma.values.astype(float)
        brk = np.where(np.diff(x) > 25.0 / 390.0)[0]
        xs = np.insert(x, brk + 1, np.nan)
        ys = np.insert(y, brk + 1, np.nan)
        ax.plot(xs, ys, ls=ls, color=color, lw=1.0, zorder=3,
                label=label if first else None, solid_capstyle="round")
        first = False


def steps(q):
    """Median absolute move of gamma between consecutive snapshots within a
    session, and between the last snapshot of one session and the first of
    the next."""
    q = q.sort_values(["date", "time"])
    within, overnight = [], []
    prev_last = None
    for _, d in q.groupby("date", sort=True):
        v = d.gamma.values.astype(float)
        if len(v) > 1:
            within.extend(np.abs(np.diff(v)))
        if prev_last is not None:
            overnight.append(abs(v[0] - prev_last))
        prev_last = v[-1]
    return (float(np.median(within)) if within else np.nan,
            float(np.median(overnight)) if overnight else np.nan)


fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.9), sharex=True)
fig.subplots_adjust(left=0.085, right=0.985, top=0.92, bottom=0.24, hspace=0.19)

summary = {}
for ax, (tick, title) in zip(axes, NAMES):
    q = g[g.ticker == tick]
    lines = []
    for model, label, color, ls in SERIES:
        s = q[q.model == model]
        draw(ax, s, color, ls, label)
        w, o = steps(s)
        summary[(tick, model)] = (float(s.gamma.median()), float(s.gamma.quantile(0.25)),
                                  float(s.gamma.quantile(0.75)), w, o,
                                  float(100 * (s.gamma > 0.5).mean()))
        lines.append("%s: median %.3f, IQR [%.3f, %.3f]; median step %.4f within session, %.4f overnight"
                     % (label.replace("$\\gamma$", "gamma"), *summary[(tick, model)][:5]))
    ax.axhline(0.5, color="0.25", lw=0.8, ls=(0, (3, 2)), zorder=2)
    if tick == NAMES[0][0]:
        ax.text(0.005, 0.5, r"$\gamma=1/2$: the pin, and the SSVI bound", transform=ax.get_yaxis_transform(),
                ha="left", va="bottom", fontsize=7, color="0.25", zorder=4)
    for i in range(1, len(dates)):
        ax.axvline(i, color="0.55", lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.set_xlim(-0.06, len(dates) - 0.94)
    ax.set_ylabel(r"fitted $\gamma$", fontsize=8.5)
    ax.set_title(title, loc="left", fontsize=9)
    ax.text(0.008, 0.965, "\n".join(lines), transform=ax.transAxes, ha="left", va="top",
            fontsize=6.4, color="0.2", zorder=5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", lw=0.5, alpha=0.92))

axes[0].legend(loc="lower right", fontsize=7, ncol=2)
axes[-1].set_xticks([i + 0.5 for i in range(len(dates))])
axes[-1].set_xticklabels([d[5:] for d in dates], fontsize=6.6, rotation=90)
axes[-1].set_xlabel("session (vertical rules = overnight; x within a session = time of day)", fontsize=8)

ymin = min(min(v) for v in [g[g.ticker == t].gamma.values for t, _ in NAMES])
ymax = max(max(v) for v in [g[g.ticker == t].gamma.values for t, _ in NAMES])
for ax in axes:
    ax.set_ylim(min(0.45, ymin - 0.02), max(0.7, ymax + 0.02))

n_cap = int(g[g.model == "ssvi"].groupby("ticker").size().sum())
at_bound = 100.0 * (g[g.model == "ssvi"].gamma > 0.4999).mean()
note(fig, "Intraday captures over %d sessions, %d captures across the two names. The constrained SSVI arm (not drawn) sits on its\n"
          "bound gamma = 1/2 on %.0f%% of these captures. Source: results/gamma_ext.csv (study --panel gamma); identical ingest to the\n"
          "intraday panel, so the G-eSSVI series is the intraday panel's own fit." % (len(dates), n_cap, at_bound), y=0.008)
save(fig, "fig7_gamma_timeseries.pdf")

for k, v in summary.items():
    print("%-5s %-12s median %.3f IQR [%.3f, %.3f] step within %.4f overnight %.4f  >1/2 %.0f%%" % (k[0], k[1], *v))
print("ssvi at bound: %.1f%%" % at_bound)
