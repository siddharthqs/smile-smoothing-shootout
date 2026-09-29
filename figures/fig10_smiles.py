"""Figures: the smiles of every name, four fitted arms against the market.

Three figures on one daily capture - the index ETFs, the single names, and
the JPM/XOM extension - at the pillar expiries nearest one,
three and six months.  Dots are the repaired pillar vols every arm was
fitted to (results/pillars.csv, written by the driver); the shaded band
is the raw bid-ask in implied vol, inverted with the Black formula on
the pillar's parity forward; the curves are the arms' own slice
formulas evaluated from the parameters in fits.csv.  Beyond the quoted
span the curves are extrapolation, and the band stops.
"""
import gzip
import json
import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import ndtr

from _style import use_paper_style, RESULTS, OUT, COLOR, LS, LABEL, res

use_paper_style()

DATE = "2026-08-27"
TARGETS = [("1M", 1 / 12), ("3M", 0.25), ("6M", 0.5)]
ARMS = ["ssvi", "essvi_g05", "essvi_gfree", "essvi_sg05"]
RAW = os.path.join(os.path.dirname(RESULTS.rstrip("/")), "data", "raw").replace("\\", "/")

FIGS = [("fig10_smiles_index.pdf", ["SPY", "QQQ", "IWM", "DIA"], ""),
        ("fig10_smiles_single.pdf", ["AMD", "GOOGL", "META", "TSLA"], ""),
        ("fig10_smiles_ext.pdf", ["JPM", "XOM"], "extension/")]


def load(sub):
    pil = pd.read_csv(res(sub + "pillars.csv"))
    fits = pd.read_csv(res(sub + "fits.csv"), low_memory=False)
    fits = fits[(fits.status == "ok") & (fits.date == DATE)]
    panel = pd.read_csv(res(sub + "panel.csv"), low_memory=False)
    panel = panel[(panel.slice == "daily") & (panel.date == DATE)].set_index("ticker")
    return pil, fits, panel


def smile(row, model, k):
    """Total variance of one fitted slice at log-moneyness k."""
    if model == "svi":
        a, b, r, m, s = (row[f"svi_{c}"] for c in ("a", "b", "rho", "m", "sigma"))
        d = k - m
        return a + b * (r * d + np.sqrt(d * d + s * s))
    if model == "ssvi":
        th, ph, r = row["ssvi_theta"], row["ssvi_phi"], row["ssvi_rho"]
    else:
        th, r = row["essvi_theta"], row["essvi_rho"]
        ph = row["essvi_psi"] / th
    u = ph * k + r
    return 0.5 * th * (1 + r * ph * k + np.sqrt(u * u + 1 - r * r))


def black(F, K, t, sig, call, df):
    v = sig * np.sqrt(t)
    d1 = (np.log(F / K) + 0.5 * v * v) / v
    d2 = d1 - v
    return df * np.where(call, F * ndtr(d1) - K * ndtr(d2), K * ndtr(-d2) - F * ndtr(-d1))


def invert(price, F, K, t, call, df):
    lo, hi = np.full_like(price, 1e-4), np.full_like(price, 5.0)
    sig = np.full_like(price, 0.3)
    for _ in range(60):
        f = black(F, K, t, sig, call, df) - price
        lo = np.where(f < 0, sig, lo)
        hi = np.where(f > 0, sig, hi)
        sig = 0.5 * (lo + hi)
    return sig


def rate(row, t):
    xs = np.array([1 / 12, 0.25, 0.5, 1.0])
    ys = np.array([row.rate_1M, row.rate_3M, row.rate_6M, row.rate_1Y])
    return float(np.interp(t, xs, ys))


def draw(out, NAMES, sub):
    pil, fits, panel = load(sub)
    n = len(NAMES)
    fig, axes = plt.subplots(n, 3, figsize=(7.4, 0.5 + 2.1 * n))
    for i, name in enumerate(NAMES):
        prow = panel.loc[name]
        raw = json.load(gzip.open(f"{RAW}/{name}/{DATE}/{int(prow.time):04d}.json.gz", "rt", encoding="utf-8"))
        quotes = pd.DataFrame(raw["quotes"])
        pn = pil[(pil.ticker == name) & (pil.date == DATE)]
        fn = fits[fits.ticker == name]
        for j, (lab, target) in enumerate(TARGETS):
            ax = axes[i, j]
            # the pillar nearest the target tenor
            tens = np.sort(pn.tenor_years.unique())
            t = tens[np.argmin(np.abs(tens - target))]
            p = pn[np.isclose(pn.tenor_years, t)]
            F = float(p.forward.iloc[0])
            expiry = p.expiry_date.iloc[0]
            k_p = np.log(p.strike.values / F)
            lo, hi = k_p.min(), k_p.max()
            # raw bid-ask band, OTM side, inside the quoted span
            q = quotes[(quotes.expiry == expiry) & (quotes.bid > 0) & (quotes.ask > quotes.bid)].copy()
            q["k"] = np.log(q.strike / F)
            q = q[(q.k >= lo) & (q.k <= hi)]
            q = q[((q.right == "C") & (q.k >= 0)) | ((q.right == "P") & (q.k < 0))].sort_values("k")
            df = math.exp(-rate(prow, t) * t)
            call = (q.right == "C").values
            vb = invert(q.bid.values, F, q.strike.values, t, call, df)
            va = invert(q.ask.values, F, q.strike.values, t, call, df)
            ax.fill_between(q.k.values, 100 * vb, 100 * va, color="#A6A6A6", lw=0, alpha=0.9, label="bid--ask (raw)")
            # fitted curves, drawn past the quoted span into the shaded extrapolation
            kk = np.linspace(lo - 0.12, hi + 0.12, 400)
            for arm in ARMS:
                r = fn[(fn.model == arm) & np.isclose(fn.tenor_years, t)]
                if r.empty:
                    continue
                w = smile(r.iloc[0], arm, kk)
                ax.plot(kk, 100 * np.sqrt(np.maximum(w, 0) / t), color=COLOR[arm], ls=LS[arm], lw=1.3, label=LABEL[arm], zorder=4)
            ax.plot(k_p, 100 * p.vol.values, "o", ms=3.4, mfc="white", mec="#8C8C8C", mew=0.7, label="repaired pillar (fit target)", zorder=3)
            ax.axvspan(lo - 0.12, lo, color="#F2F2F2", lw=0, zorder=0)
            ax.axvspan(hi, hi + 0.12, color="#F2F2F2", lw=0, zorder=0)
            ax.set_xlim(lo - 0.12, hi + 0.12)
            ymin = 100 * np.nanmin(np.minimum(vb, p.vol.values.min()))
            ymax = 100 * np.nanmax(np.maximum(va, p.vol.values.max()))
            ax.set_ylim(max(0, ymin - 0.12 * (ymax - ymin)), ymax + 0.12 * (ymax - ymin))
            ax.set_title(f"{name}, {expiry} ({lab}, $t={t:.2f}$)", fontsize=8.5)
            if i == n - 1:
                ax.set_xlabel("log-moneyness $k = \\ln(K/F)$")
            if j == 0:
                ax.set_ylabel("implied vol (%)")
            ax.grid(True, lw=0.3, alpha=0.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.4 / (0.5 + 2.1 * n), 1, 1))
    fig.savefig(OUT + out)
    plt.close(fig)
    print("wrote", out, NAMES, DATE)


for out, names, sub in FIGS:
    draw(out, names, sub)
