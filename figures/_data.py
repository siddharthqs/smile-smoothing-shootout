"""Loaders and derived quantities for the smile-smoothing-shootout figures.

Nothing here invents a number: every column read below exists in the CSVs
written by the study driver, and the derived quantities are closed forms of
the fitted parameters that the driver already exports.

Robustness convention used throughout: raw SVI produces a small number of
degenerate slices (published implied vol up to 5.6e4 and down to 7e-6, which
drives fit_rmse_vol_bp to 1.8e8), so every summary here is a MEDIAN or a
COUNT.  Means and min_g are never used as summary statistics.
"""
import numpy as np
import pandas as pd

from _style import RESULTS, res

KEY = ["ticker", "date", "time"]


def _p(pref, name):
    return res(("intraday/" if pref == "intraday" else "") + name)


def load_panel(slice_="daily"):
    return pd.read_csv(_p(slice_, "panel.csv"))


def load_fits(slice_="daily", usecols=None):
    f = pd.read_csv(_p(slice_, "fits.csv"), usecols=usecols)
    return f[f.status == "ok"].copy()


def load_localvol(slice_="daily"):
    # match make_tables.py: keep every "ok*" status.  ok_no_usability means
    # only the (unused) ur_* usability block failed; the surface, the local
    # volatility and the round trip on that row are all valid.
    lv = pd.read_csv(_p(slice_, "localvol.csv"))
    return lv[lv.status.str.startswith("ok")].copy()


def load_arb(slice_="daily"):
    a = pd.read_csv(_p(slice_, "arbitrage.csv"))
    return a[a.status.str.startswith("ok")].copy()


def load_martingale(slice_="daily"):
    m = pd.read_csv(_p(slice_, "martingale.csv"))
    return m[m.status.str.startswith("ok")].copy()


# ---------------------------------------------------------------- fit error
def nameday_fit_error(slice_="daily"):
    """Per name-day, per model: MEDIAN over the fitted expiry slices of the
    in-sample RMSE in vol bp.  Median, not mean, because of the SVI tail."""
    f = load_fits(slice_, usecols=["ticker", "date", "time", "model", "status",
                                   "fit_rmse_vol_bp", "converged"])
    g = (f.groupby(KEY + ["model"])
           .agg(fit_bp=("fit_rmse_vol_bp", "median"),
                fit_bp_p90=("fit_rmse_vol_bp", lambda s: s.quantile(0.9)),
                converged=("converged", "mean"),
                n_slices=("fit_rmse_vol_bp", "size"))
           .reset_index())
    return g


# ------------------------------------------------------- closed-form ATM/skew
def atm_quantities(df, k=0.0):
    """Total variance w(k), its k-slope and its k-curvature for whichever
    model each row belongs to.

    raw SVI   w = a + b[rho(k-m) + sqrt((k-m)^2 + sigma^2)]
    SSVI      w = theta/2 [1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2)]
    eSSVI     same slice shape with phi = psi/theta  (so w(0)=theta,
              dw/dk|_0 = rho psi)

    These are exactly the parameterizations in src/equity/models/svi.rs and
    src/equity/models/essvi.rs, evaluated on the parameters the driver wrote.
    """
    n = len(df)
    w = np.full(n, np.nan)
    wk = np.full(n, np.nan)
    wkk = np.full(n, np.nan)
    m = df.model.values

    i = m == "svi"
    if i.any():
        a, b, r, mm, sg = (df[c].values[i] for c in
                           ["svi_a", "svi_b", "svi_rho", "svi_m", "svi_sigma"])
        d = k - mm
        rt = np.sqrt(d * d + sg * sg)
        w[i] = a + b * (r * d + rt)
        wk[i] = b * (r + d / rt)
        wkk[i] = b * sg * sg / rt ** 3

    for mod in ("ssvi", "essvi", "essvi_g05", "essvi_gfree", "essvi_sg05"):
        i = m == mod
        if not i.any():
            continue
        if mod == "ssvi":
            th = df["ssvi_theta"].values[i]
            ph = df["ssvi_phi"].values[i]
            r = df["ssvi_rho"].values[i]
        else:
            th = df["essvi_theta"].values[i]
            ph = df["essvi_psi"].values[i] / th
            r = df["essvi_rho"].values[i]
        u = ph * k + r
        R = np.sqrt(u * u + 1.0 - r * r)
        w[i] = 0.5 * th * (1.0 + r * ph * k + R)
        wk[i] = 0.5 * th * ph * (r + u / R)
        wkk[i] = 0.5 * th * ph * ph * (1.0 - r * r) / R ** 3
    return w, wk, wkk


PARAM_COLS = ["svi_a", "svi_b", "svi_rho", "svi_m", "svi_sigma",
              "ssvi_rho", "ssvi_theta", "ssvi_phi", "ssvi_eta", "ssvi_gamma",
              "essvi_theta", "essvi_psi", "essvi_rho",
              "essvi_eta", "essvi_gamma"]

# Per-slice parameters used for the snapshot-to-snapshot jump statistic.
# The global arms carry the same slice triple as the sequential eSSVI; their
# per-capture backbone (eta, gamma) is analysed separately in the gamma panel.
MODEL_PARAMS = {
    "svi": ["svi_a", "svi_b", "svi_rho", "svi_m", "svi_sigma"],
    "essvi": ["essvi_theta", "essvi_psi", "essvi_rho"],
    "essvi_g05": ["essvi_theta", "essvi_psi", "essvi_rho"],
    "essvi_gfree": ["essvi_theta", "essvi_psi", "essvi_rho"],
    "essvi_sg05": ["essvi_theta", "essvi_psi", "essvi_rho"],
    "ssvi": ["ssvi_theta", "ssvi_phi", "ssvi_rho"],
}


def pillar_series(target_t=1.0 / 12.0, slice_="intraday"):
    """For every (ticker, date, time, model) pick the fitted expiry closest to
    `target_t` and return the ATM implied vol and ATM vol skew there.

    All three models fit the SAME repaired pillars, so the selected expiry is
    identical across models at each snapshot: any expiry-roll jump moves all
    three lines together and cannot masquerade as model instability.
    """
    cols = (["ticker", "date", "time", "model", "status", "tenor_years",
             "fit_rmse_vol_bp", "fit_vol_min", "fit_vol_max", "converged"]
            + PARAM_COLS)
    f = load_fits(slice_, usecols=cols)
    f["dist"] = (f.tenor_years - target_t).abs()
    sel = f.loc[f.groupby(KEY + ["model"]).dist.idxmin()].copy()
    w, wk, wkk = atm_quantities(sel, 0.0)
    t = sel.tenor_years.values
    sel["atm_vol"] = np.sqrt(np.clip(w, 1e-12, None) / t)
    # d sigma / dk at the money  =  w_k / (2 sqrt(w t))
    sel["atm_skew"] = wk / (2.0 * np.sqrt(np.clip(w, 1e-12, None) * t))
    sel["atm_curv"] = wkk
    return sel.sort_values(["ticker", "model", "date", "time"])


SESSION_MINUTES = 390.0


def clock_axis(df):
    """Continuous x coordinate: session index + fraction of the 09:30-16:00
    session elapsed, so overnight gaps appear as gaps and intraday spacing is
    faithful."""
    dates = sorted(df.date.unique())
    di = {d: i for i, d in enumerate(dates)}
    hhmm = df.time.astype(int)
    mins = (hhmm // 100) * 60 + (hhmm % 100) - (9 * 60 + 30)
    return df.date.map(di).values + np.clip(mins / SESSION_MINUTES, 0, 1).values, dates


# --------------------------------------------------------------- barriers
BARRIER_COLS = ["ticker", "date", "time", "status", "maturity",
                "barrier_type", "barrier_dir", "barrier_mult", "right", "pv",
                "std_err", "spot", "t_below_first_pillar", "t_beyond_last_pillar"]


def barrier_dispersion(slice_="daily", models=None):
    """Cross-arm spread (max - min over the chosen arms) of every priced
    instrument, expressed BOTH relative to the mean price and in bp of spot,
    plus its size in Monte-Carlo standard errors.  One row per
    (name-day, maturity, product, barrier level).  Default arm set is the
    practical menu (the SVI benchmark excluded)."""
    from _style import MENU
    if models is None:
        models = MENU
    b = pd.read_csv(_p(slice_, "barriers.csv"), usecols=["model"] + BARRIER_COLS,
                    low_memory=False)
    b = b[(b.status == "ok") & b.model.isin(models)].copy()
    b["fam"] = np.where(b.barrier_type == "none", "vanilla " + b.right,
                        b.barrier_dir + "-out " + b.right)
    g = (b.groupby(KEY + ["maturity", "fam", "barrier_mult"], dropna=False)
           .agg(n=("pv", "size"), mx=("pv", "max"), mn=("pv", "min"),
                mean=("pv", "mean"), se=("std_err", "mean"),
                spot=("spot", "first"),
                below=("t_below_first_pillar", "max"),
                beyond=("t_beyond_last_pillar", "max"))
           .reset_index())
    g = g[g.n == len(models)].copy()
    spread = g.mx - g.mn
    g["rel_pct"] = 100.0 * spread / g["mean"]
    g["abs_bp"] = 1e4 * spread / g.spot
    g["mc_sigma"] = spread / g.se
    g["calibrated"] = (~g.below) & (~g.beyond)
    return g


def med_iqr(s):
    s = pd.Series(s).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return np.nan, np.nan, np.nan
    return s.median(), s.quantile(0.25), s.quantile(0.75)
