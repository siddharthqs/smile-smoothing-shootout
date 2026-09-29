//! Smile-smoothing shootout: the study driver.
//!
//! One causal chain per (ticker, date, time), run three times -- once per
//! parameterization -- on **identical** inputs:
//!
//! ```text
//!   chain.json.gz
//!     -> OptionChain
//!     -> clean + parity forwards + Black-76 implied vols   (FILTERS below)
//!     -> minimal-change static-arbitrage repair            (repair_arbitrage)
//!     -> the SAME repaired pillars fitted by SVI / SSVI / eSSVI
//!     -> arbitrage on the fitted surface (common dense grid)
//!     -> Dupire local vol (the library's shared SmoothedSurface path)
//!     -> martingale forward recovery (fixed seed)
//!     -> single-barrier grid under local vol (same seed, same paths)
//! ```
//!
//! Emits fits.csv, arbitrage.csv, localvol.csv, martingale.csv,
//! barriers.csv, panel.csv and run_manifest.json.
//!
//! Usage:
//! ```text
//!   study --panel daily     --out ../results
//!   study --panel intraday  --out ../results/intraday
//!   study --panel loo       --out ../results            (leave-one-expiry-out; loocv.csv)
//!   study --panel gamma     --out ../results --tickers A,B,...
//!                                                        (fitted exponents only; gamma_ext.csv)
//! ```

mod model;
mod sim;

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use chrono::NaiveDate;
use rayon::prelude::*;

use rustyqlib::core::curves::{Compounding, InterpolationMethod, Tenor, YieldCurve};
use rustyqlib::core::daycount::DayCountConvention;
use rustyqlib::core::trade::PutOrCall;
use rustyqlib::core::traits::Instrument;
use rustyqlib::core::vols::{SmileCoordinate, VolInput, VolSurface};
use rustyqlib::equity::blackscholes::{bs_price, implied_vol_from_price};
use rustyqlib::equity::builder::EquityOptionBuilder;
use rustyqlib::equity::essvi::{EssviFitConfig, EssviGlobalConfig, EssviSurfaceFit};
use rustyqlib::equity::option_chain::{
    de_americanize_chain, implied_vol_surface_from_chain, DeAmericanizeConfig, FilterConfig,
    OptionChain,
};
use rustyqlib::equity::smoothed_surface::SmoothedSurface;
use rustyqlib::equity::surface_repair::repair_arbitrage;
use rustyqlib::equity::svi::{SsviSurfaceFit, SviSurfaceFit};
use rustyqlib::equity::usability::{usability_report, UsabilityConfig};
use rustyqlib::equity::utils::{Engine, Model};
use rustyqlib::validation::martingale::{martingale_report, MartingaleConfig};

use model::{Fitted, ModelId};
use sim::{BarrierLevel, SimConfig};

// ════════════════════════════════════════════════════════════════════════
//  EVERY KNOB THE STUDY TURNS, IN ONE PLACE.
//  These are the numbers the paper states; changing the study means
//  changing this block and nothing else.
// ════════════════════════════════════════════════════════════════════════

// ── Quote cleaning (FilterConfig) ──────────────────────────────────────
/// Reject quotes with `(ask - bid) / mid` above this.
const MAX_RELATIVE_SPREAD: f64 = 0.25;
/// Keep strikes with `K/F` inside this window. Chosen to **bracket the
/// evaluation grid** below (`GRID_K_LO/HI` = +/-0.40 in log-moneyness,
/// i.e. K/F in [0.670, 1.492]), so every arbitrage, roughness and
/// local-vol number the study reports is measured where quotes existed
/// rather than in the extrapolated wings.
const MIN_MONEYNESS: f64 = 0.60;
const MAX_MONEYNESS: f64 = 1.60;
/// Skip expiries closer than this many calendar days ...
const MIN_DAYS_TO_EXPIRY: i64 = 1;
/// ... or beyond this many years (Act/365).
const MAX_YEARS_TO_EXPIRY: f64 = 1.5;
/// Drop an expiry left with fewer usable vols than this. Set to 5 (raw
/// SVI's parameter count) so all three models are handed exactly the
/// same set of expiry slices -- SVI skips slices with fewer than five
/// pillars, eSSVI would accept three, and an unequal slice set would
/// make the downstream comparison unattributable.
const MIN_QUOTES_PER_EXPIRY: usize = 5;
/// Out-of-the-money quotes only per side (the liquid side, free of the
/// American early-exercise premium in ITM calls).
const OTM_ONLY: bool = true;
/// Nearest-the-money put/call pairs the parity-forward median uses.
const FORWARD_PAIRS: usize = 5;

// ── eSSVI calibration weights (SVI and SSVI have no equivalent knob) ───
const ESSVI_CALENDAR_PENALTY: f64 = 1000.0;
const ESSVI_BUTTERFLY_PENALTY: f64 = 10.0;

// ── The common dense evaluation grid ───────────────────────────────────
// Arbitrage, roughness and local-vol statistics are all measured here,
// identically for the three models, so the numbers are comparable.
/// Log-moneyness span (relative to each expiry's forward).
const GRID_K_LO: f64 = -0.40;
const GRID_K_HI: f64 = 0.40;
/// dk = 0.01 at 81 points.
const GRID_K_POINTS: usize = 81;
/// Expiry points, spanning the fitted pillar range.
const GRID_T_POINTS: usize = 41;
/// Strikes sampled per slice when a fit is written out as a `VolSurface`
/// (the finite-difference round trip is a `VolSurface` consumer).
const SURFACE_SAMPLES: usize = 41;

// ── Round-trip repricing of the calibrating vanillas ───────────────────
/// Log-moneyness offsets, relative to the forward, of the repriced
/// vanillas.
const RT_K_OFFSETS: [f64; 5] = [-0.10, -0.05, 0.0, 0.05, 0.10];
/// Finite-difference grid for the round trip (spot steps, time steps).
const RT_FD_GRID: (usize, usize) = (400, 400);
/// The library usability report's sampling.
const USABILITY_MAX_EXPIRIES: usize = 4;
const USABILITY_STRIKES_PER_EXPIRY: usize = 3;
const USABILITY_MONEYNESS: (f64, f64) = (0.85, 1.15);

// ── Fixed instrument grid (barriers, round trip, martingale horizons) ──
/// Maturities, in years. Fixed across every name so the cross-sectional
/// dispersion result is not contaminated by different listed expiries.
const MATURITIES: [f64; 3] = [1.0 / 12.0, 0.25, 0.5];
const MATURITY_LABELS: [&str; 3] = ["1M", "3M", "6M"];
/// Barrier levels as multiples of spot.
const BARRIER_LEVELS: [(bool, f64); 6] = [
    (true, 0.95), // down-and-out
    (true, 0.90),
    (true, 0.85),
    (false, 1.05), // up-and-out
    (false, 1.10),
    (false, 1.15),
];
/// Every barrier and vanilla in the grid is struck at spot.
const STRIKE_MULT: f64 = 1.0;
/// Run options for the alternative contract: strike at the forward, and
/// one symmetric barrier pair per maturity at K(1 -/+ p). Set from the
/// command line; unset reproduces the released grid.
static STRIKE_AT_FORWARD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static BARRIER_PCT: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
/// Per-maturity barrier distance in ATM standard deviations (0 = no barrier
/// at that maturity), from the market ATM vol of the repaired surface.
static BARRIER_SD: std::sync::OnceLock<Option<Vec<f64>>> = std::sync::OnceLock::new();
/// Out-of-the-money strikes: call at F exp(+m sigma sqrt T), put at F exp(-m sigma sqrt T).
static STRIKE_SD: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();

// ── Stochastic steps: one seed, identical across models ────────────────
/// Barrier / vanilla Monte Carlo.
const MC_SEED: u64 = 20260824;
const MC_PATHS: usize = 40_000;
const MC_STEPS_PER_YEAR: usize = 252;
/// Floor on Euler substeps between consecutive maturities. A flat
/// steps-per-year rate leaves the 1M leg with only ~21 steps, which is
/// where log-Euler discretization bias actually bites; this makes the
/// short end as finely stepped as the long one for a small extra cost.
const MC_MIN_SUBSTEPS_PER_LEG: usize = 64;
/// Martingale forward-recovery check (the library's own simulation).
const MARTINGALE_SEED: u64 = 20260824;
const MARTINGALE_PATHS: usize = 32_768;
const MARTINGALE_STEPS_PER_YEAR: usize = 104;
const MARTINGALE_Z_THRESHOLD: f64 = 3.0;

// ── De-Americanization (applied ONCE, before the surface builder) ──────
/// CRR binomial steps for the early-exercise premium trees.
const DEA_TREE_STEPS: usize = 201;
/// Fixed-point rounds (forward -> implied carry -> EEP -> forward).
const DEA_ROUNDS: usize = 2;

// ── Panel layout ───────────────────────────────────────────────────────
const DAILY_TARGET_TIME: &str = "1530";
/// The study universe: the four index ETFs plus four liquid single
/// names, both panels. Overridable with `--tickers` for exploration;
/// the released CSVs use this set.
const STUDY_TICKERS: [&str; 8] = [
    "AMD", "DIA", "GOOGL", "IWM", "META", "QQQ", "SPY", "TSLA",
];
/// The study window (inclusive): two calendar weeks, ten sessions.
const STUDY_DATE_MIN: &str = "2026-08-14";
const STUDY_DATE_MAX: &str = "2026-09-11";
/// Dates inside the window that are not trading sessions. 2026-09-07 is
/// Labor Day: the collector ran and left seven stray captures, which are
/// not a session. (2026-08-28 needs no entry: the collector did not run.)
const STUDY_SKIP_DATES: [&str; 1] = ["2026-09-07"];

// ════════════════════════════════════════════════════════════════════════

fn filter_config() -> FilterConfig {
    FilterConfig {
        max_relative_spread: MAX_RELATIVE_SPREAD,
        min_moneyness: MIN_MONEYNESS,
        max_moneyness: MAX_MONEYNESS,
        min_days_to_expiry: MIN_DAYS_TO_EXPIRY,
        max_years_to_expiry: MAX_YEARS_TO_EXPIRY,
        min_quotes_per_expiry: MIN_QUOTES_PER_EXPIRY,
        otm_only: OTM_ONLY,
        forward_pairs: FORWARD_PAIRS,
    }
}

fn dea_config() -> DeAmericanizeConfig {
    DeAmericanizeConfig {
        tree_steps: DEA_TREE_STEPS,
        rounds: DEA_ROUNDS,
        // trees are skipped where the surface builder would drop the
        // quote anyway
        max_years_to_expiry: Some(MAX_YEARS_TO_EXPIRY),
        forward_pairs: FORWARD_PAIRS,
        ..DeAmericanizeConfig::default()
    }
}

// ── CSV plumbing ───────────────────────────────────────────────────────

/// Numbers are written at 10 decimals; non-finite values are written
/// empty rather than as `NaN`, so a reader never has to guess.
fn num(x: f64) -> String {
    if x.is_finite() {
        format!("{x:.10}")
    } else {
        String::new()
    }
}

/// Free text that has to survive a CSV field.
fn clean(s: &str) -> String {
    // Long enough that the library's per-reason dropped-quote map -- the
    // whole diagnosis of a failed name-day -- survives into the CSV.
    s.replace([',', '\n', '\r', '"'], ";")
        .chars()
        .take(400)
        .collect()
}

/// Some library pricing paths assert rather than return an error (e.g.
/// the finite-difference solver refuses a non-positive reference vol,
/// which a degenerate fitted slice can produce). A study must record
/// that, not die of it: every such call is wrapped, the failure is
/// counted, and the reason reaches `status_detail`.
fn guarded<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(|e| {
        e.downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "panic".to_string())
    })
}

/// Print each distinct panic once. Silencing them entirely would hide a
/// real bug; printing every one drowns the progress log.
fn install_panic_hook() {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<String>>> = Mutex::new(None);
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let key = format!("{}@{:?}", info, info.location());
        let mut guard = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        let seen = guard.get_or_insert_with(HashSet::new);
        if seen.insert(key) {
            previous(info);
        }
    }));
}

fn columns(header: &str) -> usize {
    header.matches(',').count() + 1
}

/// A CSV row that checks its own width against the header it is destined
/// for. Miscounting a column is the single easiest way to silently
/// corrupt a published dataset, so it panics instead.
struct Row(Vec<String>);

impl Row {
    fn new() -> Self {
        Row(Vec::with_capacity(48))
    }
    fn s(mut self, v: impl AsRef<str>) -> Self {
        self.0.push(v.as_ref().to_string());
        self
    }
    fn f(mut self, v: f64) -> Self {
        self.0.push(num(v));
        self
    }
    fn of(mut self, v: Option<f64>) -> Self {
        self.0.push(v.map(num).unwrap_or_default());
        self
    }
    fn ou(self, v: Option<usize>) -> Self {
        match v {
            Some(x) => self.u(x),
            None => self.s(""),
        }
    }
    fn u(mut self, v: usize) -> Self {
        self.0.push(v.to_string());
        self
    }
    fn n(mut self, v: u64) -> Self {
        self.0.push(v.to_string());
        self
    }
    fn b(mut self, v: bool) -> Self {
        self.0.push(v.to_string());
        self
    }
    /// Pad the remaining columns with empty fields (failure rows).
    fn fill(mut self, header: &str) -> String {
        while self.0.len() < columns(header) {
            self.0.push(String::new());
        }
        self.finish(header)
    }
    fn finish(self, header: &str) -> String {
        assert_eq!(
            self.0.len(),
            columns(header),
            "row width {} != header width {} for `{}`",
            self.0.len(),
            columns(header),
            header.split(',').next().unwrap_or("")
        );
        self.0.join(",")
    }
}

// ── Panel enumeration ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Capture {
    ticker: String,
    date: NaiveDate,
    time: String,
    path: PathBuf,
}

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data")
}

fn list_captures(slice: &str, tickers: &Option<Vec<String>>) -> Vec<Capture> {
    let raw = data_root().join("raw");
    let mut out: Vec<Capture> = Vec::new();
    let mut symbols: Vec<String> = std::fs::read_dir(&raw)
        .expect("data/raw must exist")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    symbols.sort();
    if let Some(keep) = tickers {
        symbols.retain(|s| keep.iter().any(|k| k.eq_ignore_ascii_case(s)));
    }
    for symbol in symbols {
        let mut dates: Vec<String> = std::fs::read_dir(raw.join(&symbol))
            .expect("ticker directory")
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        dates.sort();
        for date_str in dates {
            let Ok(date) = NaiveDate::parse_from_str(&date_str, "%Y-%m-%d") else {
                continue;
            };
            if date_str.as_str() < STUDY_DATE_MIN || date_str.as_str() > STUDY_DATE_MAX {
                continue;
            }
            if STUDY_SKIP_DATES.contains(&date_str.as_str()) {
                continue;
            }
            let mut times: Vec<String> = std::fs::read_dir(raw.join(&symbol).join(&date_str))
                .expect("session directory")
                .filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().to_string())
                .filter(|n| n.ends_with(".json.gz"))
                .map(|n| n.trim_end_matches(".json.gz").to_string())
                .collect();
            times.sort();
            if times.is_empty() {
                continue;
            }
            let chosen: Vec<String> = if slice != "intraday" && slice != "gamma" {
                // the 1530 capture, or -- when a short session never
                // reached it -- the capture closest to it. A name-day is
                // never skipped for want of the exact stamp; the `time`
                // column records what was actually used.
                let target: i32 = DAILY_TARGET_TIME.parse().unwrap();
                vec![times
                    .iter()
                    .min_by_key(|t| (t.parse::<i32>().unwrap_or(0) - target).abs())
                    .cloned()
                    .expect("times is non-empty")]
            } else {
                times.clone()
            };
            for time in chosen {
                out.push(Capture {
                    ticker: symbol.clone(),
                    date,
                    time: time.clone(),
                    path: raw
                        .join(&symbol)
                        .join(&date_str)
                        .join(format!("{time}.json.gz")),
                });
            }
        }
    }
    out
}

fn read_gz(path: &Path) -> std::io::Result<String> {
    let bytes = std::fs::read(path)?;
    let mut text = String::new();
    flate2::read::GzDecoder::new(&bytes[..]).read_to_string(&mut text)?;
    Ok(text)
}

/// The bootstrapped Treasury curve for a session, re-anchored at the
/// chain's own as-of date (the published curve is dated T-1).
fn load_curve(date: NaiveDate, as_of: NaiveDate) -> Result<(YieldCurve, NaiveDate), String> {
    let path = data_root().join(format!("curves/ust-{date}.json"));
    let text = std::fs::read_to_string(&path).map_err(|e| format!("curve {date}: {e}"))?;
    let value: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("curve {date}: {e}"))?;
    let (curve, _) =
        rustyqlib::data::treasury::bootstrap_from_document(&value).map_err(|e| e.to_string())?;
    let curve_date = curve.reference_date();
    let rolled = if as_of > curve_date {
        curve.rolled(as_of).map_err(|e| e.to_string())?
    } else {
        curve
    };
    Ok((rolled, curve_date))
}

// ── Small helpers ──────────────────────────────────────────────────────

/// Linear interpolation on `(x, y)` pairs, flat outside.
fn lerp_pairs(points: &[(f64, f64)], x: f64) -> f64 {
    match points.len() {
        0 => f64::NAN,
        1 => points[0].1,
        n => {
            if x <= points[0].0 {
                return points[0].1;
            }
            if x >= points[n - 1].0 {
                return points[n - 1].1;
            }
            let i = points.partition_point(|p| p.0 < x).max(1);
            let (x0, y0) = points[i - 1];
            let (x1, y1) = points[i];
            y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        }
    }
}

/// The pillar smiles of a surface, on absolute strikes, in time order.
fn pillar_smiles(
    surface: &VolSurface,
    forward: &dyn Fn(f64) -> f64,
) -> Vec<(f64, Vec<(f64, f64)>)> {
    let VolInput::StrikeSmiles {
        expiries,
        smiles,
        coordinate,
        ..
    } = surface.to_input()
    else {
        return Vec::new();
    };
    expiries
        .iter()
        .zip(&smiles)
        .filter_map(|(tenor, smile)| {
            let t = match tenor {
                Tenor::YearFraction(t) => *t,
                Tenor::Date(_) => return None,
            };
            let f = forward(t);
            let points = smile
                .iter()
                .map(|&(x, vol)| {
                    let strike = match coordinate {
                        SmileCoordinate::Strike => x,
                        SmileCoordinate::Moneyness => x * f,
                        SmileCoordinate::LogMoneyness => x.exp() * f,
                    };
                    (strike, vol)
                })
                .collect();
            Some((t, points))
        })
        .collect()
}

/// `carry(t) = ln(F(t) / spot)` for a fitted surface -- the accumulated
/// drift the local vol was built against.
///
/// The library's Dupire (`smoothed_surface::dupire`) is written in
/// forward log-moneyness `k = ln(K/F(t))`. A local vol read off such a
/// surface is only consistent with dynamics that drift at
/// `d ln F(t)/dt`; drifting at the risk-free rate with zero dividends
/// instead leaves `E[S_T] = S e^{rT}`, which differs from `F_T` by
/// exactly the implied dividend/borrow yield. Every simulation in the
/// study therefore uses this, and only discounting comes from the curve.
///
/// `F` here is `SmoothedSurface::forward`, i.e. the same function the
/// martingale targets and the `forward` column are read from -- flat
/// outside the fitted pillar range -- so the accumulated drift hits the
/// stated target exactly at every horizon, including extrapolated ones.
fn carry_of(fit: &Fitted, spot: f64) -> impl Fn(f64) -> f64 + '_ {
    move |t: f64| {
        if t <= 0.0 {
            return 0.0;
        }
        let f = SmoothedSurface::forward(fit, t);
        if f.is_finite() && f > 0.0 && spot > 0.0 {
            (f / spot).ln()
        } else {
            0.0
        }
    }
}

/// A curve whose continuous zero rate at each `time` is `carry(t)/t`.
///
/// `martingale_report` uses its curve for one thing only -- the per-step
/// drift `z(t1) t1 - z(t0) t0`, which telescopes -- so handing it this
/// curve makes the simulated drift to each checkpoint exactly
/// `ln(F(t)/S)` without touching the library. Discounting elsewhere in
/// the driver keeps using the real curve.
fn growth_curve(
    reference: NaiveDate,
    carry: &dyn Fn(f64) -> f64,
    times: &[f64],
) -> Result<YieldCurve, String> {
    let mut tenors = Vec::new();
    let mut rates = Vec::new();
    for &t in times {
        if t > 0.0 {
            tenors.push(Tenor::YearFraction(t));
            rates.push(carry(t) / t);
        }
    }
    YieldCurve::from_zero_rates(
        &tenors,
        &rates,
        reference,
        DayCountConvention::Act365,
        Compounding::Continuous,
        InterpolationMethod::LogLinearDf,
    )
    .map_err(|e| e.to_string())
}

/// The element of `xs` whose `.0` is nearest `target`.
fn nearest<'a, T>(xs: &'a [(f64, T)], target: f64) -> Option<&'a (f64, T)> {
    xs.iter().min_by(|a, b| {
        (a.0 - target)
            .abs()
            .partial_cmp(&(b.0 - target).abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

// ── Headers ────────────────────────────────────────────────────────────

const PANEL_HEADER: &str = "slice,ticker,date,time,status,status_detail,file,as_of,quote_timestamp,spot,\
curve_date,n_quotes_raw,n_expiries_raw,n_quotes_used,n_quotes_dropped,\
drop_unquoted_or_crossed,drop_wide_spread,drop_expiry_window,drop_moneyness_window,\
drop_in_the_money,drop_vol_out_of_bounds,drop_solver_failed,drop_sparse_expiry,drop_no_forward,\
n_expiries_used,t_first,t_last,fwd_1M,fwd_3M,fwd_6M,rate_1M,rate_3M,rate_6M,rate_1Y,\
raw_butterfly_violations,raw_calendar_violations,repair_butterfly_adjustments,\
repair_calendar_adjustments,repair_dropped_points,repair_max_vol_change,repair_iterations,\
repair_clean,elapsed_ms";

const FITS_HEADER: &str = "slice,ticker,date,time,model,status,status_detail,slice_index,expiry_date,\
tenor_years,forward,n_quotes_raw,n_quotes_used,fit_rmse_vol_bp,fit_max_err_bp,lib_rmse_vol_bp,\
fit_vol_min,fit_vol_max,pillar_vol_min,pillar_vol_max,\
converged,min_butterfly_g,k_lo,k_hi,n_slices_fitted,n_slices_skipped,self_validates,\
svi_a,svi_b,svi_rho,svi_m,svi_sigma,\
ssvi_rho,ssvi_eta,ssvi_gamma,ssvi_theta,ssvi_phi,\
essvi_theta,essvi_psi,essvi_rho,essvi_eta,essvi_gamma,essvi_tube_clamped";

const ARB_HEADER: &str = "slice,ticker,date,time,model,status,status_detail,grid_k_lo,grid_k_hi,grid_k_points,\
grid_t_lo,grid_t_hi,grid_t_points,n_grid_points,butterfly_violations,butterfly_violation_rate,\
min_g,calendar_pairs,calendar_crossings,calendar_crossing_rate,max_calendar_crossing,\
fit_max_calendar_crossing,self_validates,grid_inside_quoted_fraction,\
butterfly_violations_inside,min_g_inside,calendar_crossings_inside,\
max_calendar_crossing_inside";

const LOCALVOL_HEADER: &str = "slice,ticker,date,time,model,status,status_detail,grid_levels,grid_times,\
lv_min,lv_max,lv_mean,clamped_fraction,guard_fraction,roughness_dk,roughness_mean_abs_d2,\
roughness_max_abs_d2,roughness_mean_abs_d2_per_dk2,lv_analytic_minus_numeric_mean_abs,\
lv_analytic_minus_numeric_max_abs,rt_points,rt_failures,rt_panics,rt_vs_fit_mean_vol_bp,\
rt_vs_fit_max_vol_bp,rt_vs_fit_mean_abs_price,rt_vs_fit_max_abs_price,\
rt_vs_fit_mean_rel_price,rt_vs_mkt_mean_vol_bp,rt_vs_mkt_max_vol_bp,\
ur_roundtrip_points,ur_roundtrip_failures,ur_roundtrip_mean_vol_bp,ur_roundtrip_max_vol_bp,\
ur_clamped_fraction,ur_fallback_fraction,ur_within_desk_tolerance,\
trusted_strike_lo,trusted_strike_hi,trusted_t_lo,trusted_t_hi";

const MARTINGALE_HEADER: &str = "slice,ticker,date,time,model,status,status_detail,horizon,t,\
t_beyond_last_pillar,t_below_first_pillar,target_forward,simulated_mean,relative_error,standard_error,z_score,\
paths,seed,steps_per_year,z_threshold,within_threshold,max_abs_z_all_horizons,\
mc_mean_spot,mc_relative_error,mc_standard_error,mc_paths,mc_seed,mc_steps,\
carry_rate,zero_dividend_forward_error,drift_source";

/// One row per (capture, model): what the calibration cost. `cold` is
/// the canonical fit every published number comes from (independent
/// captures); `warm` re-runs the same fit seeded with the previous
/// capture's solution for the same ticker and model, and exists to
/// measure what warm-starting buys. Warm results are recorded here and
/// used nowhere else.
const TIMING_HEADER: &str = "slice,ticker,date,time,model,n_pillars,n_quotes,dim,\
cold_ms,cold_iterations,cold_converged,cold_rmse_bp,\
warm_available,warm_ms,warm_iterations,warm_converged,warm_rmse_bp,warm_x_maxdiff";

const BARRIERS_HEADER: &str = "slice,ticker,date,time,model,status,status_detail,maturity,t,\
t_beyond_last_pillar,t_below_first_pillar,spot,strike,forward,discount_factor,barrier_type,barrier_dir,\
barrier_mult,barrier_level,right,pv,std_err,knock_prob,vanilla_pv,vanilla_std_err,\
ko_over_vanilla,paths,steps,seed";

/// One row per (capture, expiry): what de-Americanization did. The
/// last three columns are chain-level and repeat on every row of the
/// capture.
const DEA_HEADER: &str = "slice,ticker,date,time,expiry,tenor_years,forward_raw,forward_dea,\
forward_shift_bp,implied_carry,corrected_quotes,median_eep_call_bp,median_eep_put_bp,\
max_eep_bp,rounds,skipped_floor,max_forward_shift_bp";

/// One row per repaired pillar quote of the daily slice: the smiles every
/// arm was handed, so a fitted smile can be drawn against its target.
const PILLARS_HEADER: &str = "slice,ticker,date,time,expiry_date,tenor_years,forward,strike,vol";

/// One row per (capture, model, held-out interior expiry): the
/// leave-one-expiry-out interpolation test. The reduced fit never sees
/// the held-out slice's quotes; signed errors are model minus market.
const LOO_HEADER: &str = "slice,ticker,date,time,model,status,status_detail,holdout_index,\
n_pillars,expiry_date,tenor_years,forward,n_eval_quotes,gap_lo_years,gap_hi_years,\
insample_rmse_bp,insample_max_bp,loo_rmse_bp,loo_max_bp,loo_mean_signed_bp,\
loo_atm_signed_bp,fit_ms";

// ── The output of one capture ──────────────────────────────────────────

#[derive(Default)]
struct Output {
    panel: Vec<String>,
    fits: Vec<String>,
    arbitrage: Vec<String>,
    localvol: Vec<String>,
    martingale: Vec<String>,
    barriers: Vec<String>,
    timing: Vec<String>,
    dea: Vec<String>,
    pillars: Vec<String>,
}

impl Output {
    fn merge(&mut self, other: Output) {
        self.panel.extend(other.panel);
        self.fits.extend(other.fits);
        self.arbitrage.extend(other.arbitrage);
        self.localvol.extend(other.localvol);
        self.martingale.extend(other.martingale);
        self.barriers.extend(other.barriers);
        self.timing.extend(other.timing);
        self.dea.extend(other.dea);
        self.pillars.extend(other.pillars);
    }

    /// A failure row on every model CSV, so no name-day disappears.
    /// `code` is a short groupable category, `detail` the message.
    fn fail_models(&mut self, key: &Key, code: &str, detail: &str) {
        for m in ModelId::ALL {
            self.fail_model(key, m, code, detail);
        }
    }

    fn fail_model(&mut self, key: &Key, id: ModelId, code: &str, detail: &str) {
        let detail = clean(detail);
        let row = |h: &str| key.row(id).s(code).s(&detail).fill(h);
        self.fits.push(row(FITS_HEADER));
        self.arbitrage.push(row(ARB_HEADER));
        self.localvol.push(row(LOCALVOL_HEADER));
        self.martingale.push(row(MARTINGALE_HEADER));
        self.barriers.push(row(BARRIERS_HEADER));
    }
}

/// The (slice, ticker, date, time) key every row starts with.
struct Key {
    slice: String,
    ticker: String,
    date: String,
    time: String,
}

impl Key {
    fn base(&self) -> Row {
        Row::new()
            .s(&self.slice)
            .s(&self.ticker)
            .s(&self.date)
            .s(&self.time)
    }
    fn row(&self, id: ModelId) -> Row {
        self.base().s(id.name())
    }
}

// ── The driver ─────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let slice = arg("--panel").unwrap_or_else(|| "daily".to_string());
    assert!(
        slice == "daily" || slice == "intraday" || slice == "loo" || slice == "gamma",
        "--panel must be `daily`, `intraday`, `loo` or `gamma`"
    );
    let out_dir = PathBuf::from(arg("--out").unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../results")
            .to_string_lossy()
            .to_string()
    }));
    let tickers: Option<Vec<String>> = arg("--tickers")
        .map(|s| s.split(',').map(|t| t.trim().to_string()).collect())
        .or_else(|| Some(STUDY_TICKERS.iter().map(|s| s.to_string()).collect()));
    let limit: Option<usize> = arg("--limit").and_then(|s| s.parse().ok());
    let _ = STRIKE_AT_FORWARD.set(arg("--strike").map(|s| s == "forward").unwrap_or(false));
    let _ = BARRIER_PCT.set(arg("--barrier-pct").and_then(|s| s.parse::<f64>().ok()));
    let _ = STRIKE_SD.set(arg("--strike-sd").and_then(|s| s.parse::<f64>().ok()));
    let _ = BARRIER_SD.set(arg("--barrier-sd").map(|s| {
        let v: Vec<f64> = s.split(',').map(|x| x.trim().parse::<f64>().expect("--barrier-sd: numbers")).collect();
        assert_eq!(v.len(), MATURITIES.len(), "--barrier-sd needs one multiple per maturity");
        v
    }));
    let mc_paths: usize = arg("--paths")
        .and_then(|s| s.parse().ok())
        .unwrap_or(MC_PATHS);
    let mc_steps: usize = arg("--mc-steps")
        .and_then(|s| s.parse().ok())
        .unwrap_or(MC_STEPS_PER_YEAR);
    if let Some(n) = arg("--threads").and_then(|s| s.parse::<usize>().ok()) {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .ok();
    }
    install_panic_hook();
    std::fs::create_dir_all(&out_dir).expect("results directory");

    let mut captures = list_captures(&slice, &tickers);
    if let Some(n) = limit {
        captures.truncate(n);
    }
    eprintln!(
        "panel `{slice}`: {} captures over {} tickers -> {}",
        captures.len(),
        captures
            .iter()
            .map(|c| c.ticker.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        out_dir.display()
    );

    if slice == "loo" {
        run_loo(&captures, &out_dir);
        return;
    }
    if slice == "gamma" {
        run_gamma(&captures, &out_dir);
        return;
    }

    let started = Instant::now();
    let done = std::sync::atomic::AtomicUsize::new(0);
    let total = captures.len();
    // Group by ticker, sequential in time inside a group, so each
    // capture can be warm-started from the previous one for the same
    // name. Parallelism is across tickers (the MC engine parallelizes
    // inside each capture as well).
    let mut groups: std::collections::BTreeMap<String, Vec<&Capture>> = Default::default();
    for c in &captures {
        groups.entry(c.ticker.clone()).or_default().push(c);
    }
    let mut groups: Vec<Vec<&Capture>> = groups.into_values().collect();
    for g in &mut groups {
        g.sort_by(|a, b| (a.date, &a.time).cmp(&(b.date, &b.time)));
    }
    let outputs: Vec<Output> = groups
        .par_iter()
        .map(|group| {
            let mut warm = WarmCache::default();
            let mut acc = Output::default();
            for capture in group {
                let out = process(capture, &slice, mc_paths, mc_steps, &mut warm);
                acc.merge(out);
                let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if n % 25 == 0 || n == total {
                    eprintln!(
                        "  {n}/{total}  ({:.0}s elapsed)",
                        started.elapsed().as_secs_f64()
                    );
                }
            }
            acc
        })
        .collect();

    let mut merged = Output::default();
    for o in outputs {
        merged.merge(o);
    }

    let files: [(&str, &str, &Vec<String>); 9] = [
        ("panel.csv", PANEL_HEADER, &merged.panel),
        ("fits.csv", FITS_HEADER, &merged.fits),
        ("arbitrage.csv", ARB_HEADER, &merged.arbitrage),
        ("localvol.csv", LOCALVOL_HEADER, &merged.localvol),
        ("martingale.csv", MARTINGALE_HEADER, &merged.martingale),
        ("barriers.csv", BARRIERS_HEADER, &merged.barriers),
        ("timing.csv", TIMING_HEADER, &merged.timing),
        ("dea.csv", DEA_HEADER, &merged.dea),
        ("pillars.csv", PILLARS_HEADER, &merged.pillars),
    ];
    let mut row_counts = serde_json::Map::new();
    for (name, header, rows) in files {
        let mut text = String::with_capacity(header.len() + 128 * rows.len());
        text.push_str(header);
        text.push('\n');
        for row in rows {
            text.push_str(row);
            text.push('\n');
        }
        std::fs::write(out_dir.join(name), text).expect("write csv");
        row_counts.insert(name.to_string(), serde_json::json!(rows.len()));
        eprintln!("  wrote {name}: {} rows", rows.len());
    }

    let elapsed = started.elapsed().as_secs_f64();
    write_manifest(&out_dir, &slice, &captures, mc_paths, mc_steps, elapsed, row_counts);
    eprintln!("done in {elapsed:.1}s");
}

/// Where the library actually came from.
///
/// A commit hash alone is not provenance when the worktree is dirty --
/// and this one is: the study runs against uncommitted changes to
/// `src/`. So the manifest carries the commit, the list of modified
/// tracked files, and a content hash of the working-tree diff, and the
/// diff itself is written next to the CSVs as `library_diff.patch`.
/// Commit + patch reproduces the exact code these numbers came from.
struct Provenance {
    commit: String,
    dirty: bool,
    modified: Vec<String>,
    diff_hash: String,
    diff: String,
}

fn provenance() -> Provenance {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../RustyQLib");
    let root = root.to_string_lossy().to_string();
    let git = |args: &[&str]| -> String {
        let mut full = vec!["-C", &root];
        full.extend_from_slice(args);
        std::process::Command::new("git")
            .args(&full)
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .unwrap_or_default()
    };
    let status = git(&["status", "--porcelain"]);
    let diff = git(&["diff", "HEAD", "--", "src", "Cargo.toml", "Cargo.lock"]);
    let diff_hash = {
        // `git hash-object` gives a stable content id without pulling a
        // hashing crate into the driver.
        use std::io::Write;
        std::process::Command::new("git")
            .args(["-C", &root, "hash-object", "--stdin"])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .ok()
            .and_then(|mut child| {
                {
                    let stdin = child.stdin.as_mut()?;
                    stdin.write_all(diff.as_bytes()).ok()?;
                }
                drop(child.stdin.take());
                child.wait_with_output().ok()
            })
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_default()
    };
    Provenance {
        commit: git(&["rev-parse", "HEAD"]).trim().to_string(),
        dirty: !status.trim().is_empty(),
        modified: status
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect(),
        diff_hash,
        diff,
    }
}

fn write_manifest(
    out_dir: &Path,
    slice: &str,
    captures: &[Capture],
    mc_paths: usize,
    mc_steps: usize,
    elapsed: f64,
    row_counts: serde_json::Map<String, serde_json::Value>,
) {
    let prov = provenance();
    if !prov.diff.is_empty() {
        let _ = std::fs::write(out_dir.join("library_diff.patch"), &prov.diff);
    }
    let tickers: std::collections::BTreeSet<&str> =
        captures.iter().map(|c| c.ticker.as_str()).collect();
    let dates: std::collections::BTreeSet<String> =
        captures.iter().map(|c| c.date.to_string()).collect();
    let times: std::collections::BTreeSet<&str> =
        captures.iter().map(|c| c.time.as_str()).collect();
    let manifest = serde_json::json!({
        "study": "smile-smoothing-shootout",
        "driver": "siddh_papers/rustyqlib_papers/smile-smoothing-shootout/study (bin `study`)",
        "generated_at": chrono::Local::now().to_rfc3339(),
        "elapsed_seconds": elapsed,
        "library": {
            "package": "RustyQLib",
            "lib_target": "rustyqlib",
            "git_commit": prov.commit,
            "git_worktree_dirty": prov.dirty,
            "git_status": prov.modified,
            "working_tree_diff_hash": prov.diff_hash,
            "working_tree_diff_file": if prov.diff.is_empty() {
                serde_json::Value::Null
            } else {
                serde_json::json!("library_diff.patch")
            },
            "reproduce": "check out `git_commit` of RustyQLib, apply `library_diff.patch`, \
                          then run the paper's study driver (its Cargo.toml points at the \
                          RustyQLib checkout via a path dependency)",
            "features": ["fetch"],
            "rustc": option_env!("RUSTC_VERSION").unwrap_or("see rustc --version"),
        },
        "panel": {
            "slice": slice,
            "captures": captures.len(),
            "tickers": tickers.iter().collect::<Vec<_>>(),
            "dates": dates.iter().collect::<Vec<_>>(),
            "capture_times": times.iter().collect::<Vec<_>>(),
            "daily_target_time": DAILY_TARGET_TIME,
            "daily_selection": "the capture nearest 1530 in each session (short sessions \
                                fall back to their closest stamp; the `time` column records \
                                what was used)",
            "source": "data/raw/<TICKER>/<DATE>/<HHMM>.json.gz (Cboe delayed quotes)",
            "curves": "data/curves/ust-<DATE>.json (US Treasury par yields), bootstrapped \
                       via data::treasury::bootstrap_from_document and rolled to the chain \
                       as-of date",
        },
        "filters": {
            "max_relative_spread": MAX_RELATIVE_SPREAD,
            "min_moneyness": MIN_MONEYNESS,
            "max_moneyness": MAX_MONEYNESS,
            "min_days_to_expiry": MIN_DAYS_TO_EXPIRY,
            "max_years_to_expiry": MAX_YEARS_TO_EXPIRY,
            "min_quotes_per_expiry": MIN_QUOTES_PER_EXPIRY,
            "otm_only": OTM_ONLY,
            "forward_pairs": FORWARD_PAIRS,
            "implied_vol_bounds": [0.005, 5.0],
        },
        "repair": "rustyqlib::equity::surface_repair::repair_arbitrage (convex-hull \
                   butterfly projection + total-variance calendar sweep), applied ONCE; \
                   all three models fit the same repaired pillars",
        "models": {
            "svi": "per-expiry raw SVI (SviSurfaceFit::fit) -- unstructured benchmark",
            "ssvi": "global SSVI with power-law curvature, calibrated inside the Gatheral-Jacquier sufficient set gamma <= 1/2, eta (1 + |rho|) <= 2, hence free of static arbitrage at its pillars by construction (SsviSurfaceFit::fit); the unconstrained exponent is recorded as a diagnostic in gamma_ext.csv (--panel gamma)",
            "essvi": "per-expiry eSSVI, sequential + penalties (EssviSurfaceFit::fit_with)",
            "essvi_g05": "global eSSVI (Mingone 2022), power-law psi backbone, gamma = 1/2 (EssviSurfaceFit::fit_global)",
            "essvi_gfree": "global eSSVI (Mingone 2022), power-law psi backbone, gamma fitted (EssviSurfaceFit::fit_global)",
            "essvi_sg05": "sequential greedy-tube eSSVI, power-law backbone, gamma = 1/2, Hendriks-Martini conditions by construction slice-by-slice (EssviSurfaceFit::fit_sequential_backbone)",
            "universe": "AMD, DIA, GOOGL, IWM, META, QQQ, SPY, TSLA over 2026-08-14..2026-09-11: nineteen sessions (the collector did not run on 2026-08-28; 2026-09-07 was a market holiday); expiries 1 day to 1.5 years",
            "de_americanization": "rustyqlib::equity::option_chain::de_americanize_chain, \
                                   applied to every chain before the surface builder: \
                                   parity forward -> implied carry -> CRR early-exercise \
                                   premium subtracted from each mid, iterated twice \
                                   (tree_steps = 201). The carry is measured from the \
                                   chain's own parity forwards, never assumed. Per-expiry \
                                   outcomes land in dea.csv.",
            "essvi_calendar_penalty": ESSVI_CALENDAR_PENALTY,
            "essvi_butterfly_penalty": ESSVI_BUTTERFLY_PENALTY,
            "local_vol": "rustyqlib::equity::smoothed_surface::dupire -- one shared \
                          implementation; only variance_derivatives differs by model",
            "local_vol_clamps": [0.01, 3.0],
        },
        "grid": {
            "k_lo": GRID_K_LO, "k_hi": GRID_K_HI, "k_points": GRID_K_POINTS,
            "t_points": GRID_T_POINTS,
            "t_span": "first fitted pillar to last fitted pillar",
            "surface_samples_per_slice": SURFACE_SAMPLES,
        },
        "instruments": {
            "maturities_years": MATURITIES,
            "maturity_labels": MATURITY_LABELS,
            "strike_multiple_of_spot": STRIKE_MULT,
            "strike_at": if STRIKE_AT_FORWARD.get().copied().unwrap_or(false) { "forward" } else { "spot" },
            "barrier_pct": BARRIER_PCT.get().copied().flatten(),
            "barrier_sd": BARRIER_SD.get().cloned().flatten(),
            "strike_sd": STRIKE_SD.get().copied().flatten(),
            "vanilla_rows": "pv is the closed-form Black price at the arm's fitted vol at the strike; vanilla_pv is the local-vol Monte Carlo vanilla on the shared paths",
            "barrier_levels": BARRIER_LEVELS
                .iter()
                .map(|&(down, m)| serde_json::json!({
                    "dir": if down { "down" } else { "up" }, "mult": m
                }))
                .collect::<Vec<_>>(),
            "barrier_style": "continuously monitored knock-out, Brownian-bridge corrected",
        },
        "monte_carlo": {
            "barrier_seed": MC_SEED,
            "barrier_paths": mc_paths,
            "barrier_steps_per_year": mc_steps,
            "barrier_min_substeps_per_leg": MC_MIN_SUBSTEPS_PER_LEG,
            "scheme": "log-Euler, antithetic pairs; identical seed and normal draws \
                       across the three models, and every barrier maturity priced off \
                       one path set",
            "drift": "d ln F(t)/dt from the chain's parity forwards \
                      (SmoothedSurface::forward on the fitted surface), NOT the \
                      risk-free zero rate. The library's Dupire is written in forward \
                      log-moneyness, so this is the only drift its local vol is \
                      consistent with; drifting at r with zero dividends leaves \
                      E[S_T] = S e^{rT} and breaks put/call parity against F_T by the \
                      implied dividend/borrow yield. Discounting still comes from the \
                      bootstrapped curve.",
            "martingale_seed": MARTINGALE_SEED,
            "martingale_paths": MARTINGALE_PATHS,
            "martingale_steps_per_year": MARTINGALE_STEPS_PER_YEAR,
            "martingale_z_threshold": MARTINGALE_Z_THRESHOLD,
            "martingale_impl": "rustyqlib::validation::martingale::martingale_report, \
                                fed a synthetic growth curve whose continuous zero at t \
                                is ln(F(t)/S)/t so its internal drift matches the \
                                targets (the `drift_source` column records which curve \
                                each row used)",
        },
        "roundtrip": {
            "engine": "finite difference (Crank-Nicolson), Model::LocalVol on the sampled \
                       fitted VolSurface -- i.e. the pipeline a desk actually runs",
            "fd_spot_steps": RT_FD_GRID.0,
            "fd_time_steps": RT_FD_GRID.1,
            "k_offsets": RT_K_OFFSETS,
            "usability_report": "rustyqlib::equity::usability::usability_report, fed the \
                                 model's ANALYTIC local_vol_checked",
            "usability_max_expiries": USABILITY_MAX_EXPIRIES,
            "usability_strikes_per_expiry": USABILITY_STRIKES_PER_EXPIRY,
            "usability_moneyness": USABILITY_MONEYNESS,
            "carry_yield": "the repricing passes q = r - ln(F(t)/S)/t so the engine, \
                            the target Black-Scholes price and the implied-vol \
                            inversion all put the forward where the fitted smile is \
                            quoted. `usability_report` takes no carry argument, so its \
                            ur_* columns still assume zero dividends.",
        },
        "rows": row_counts,
    });
    std::fs::write(
        out_dir.join("run_manifest.json"),
        serde_json::to_string_pretty(&manifest).expect("manifest serializes"),
    )
    .expect("write manifest");
}

// ── One capture, end to end ────────────────────────────────────────────

/// Per-(ticker, model-name) warm-start state: the previous capture's
/// transformed parameter vector for the global eSSVI fits.
type WarmCache = std::collections::HashMap<&'static str, Vec<f64>>;

fn process(
    capture: &Capture,
    slice: &str,
    mc_paths: usize,
    mc_steps: usize,
    warm: &mut WarmCache,
) -> Output {
    let t0 = Instant::now();
    let mut out = Output::default();
    let key = Key {
        slice: slice.to_string(),
        ticker: capture.ticker.clone(),
        date: capture.date.to_string(),
        time: capture.time.clone(),
    };
    let rel = format!(
        "{}/{}/{}.json.gz",
        capture.ticker, capture.date, capture.time
    );

    // Provenance learned so far, so a failure row is still informative
    // about how far the capture got before it stopped.
    let mut ctx_as_of = String::new();
    let mut ctx_stamp = String::new();
    let mut ctx_spot = f64::NAN;
    let mut ctx_curve_date = String::new();
    let mut ctx_quotes = String::new();
    let mut ctx_expiries = String::new();

    macro_rules! bail {
        ($code:expr, $detail:expr) => {{
            let detail = clean(&$detail);
            out.panel.push(
                key.base()
                    .s($code)
                    .s(&detail)
                    .s(&rel)
                    .s(&ctx_as_of)
                    .s(&ctx_stamp)
                    .f(ctx_spot)
                    .s(&ctx_curve_date)
                    .s(&ctx_quotes)
                    .s(&ctx_expiries)
                    .fill(PANEL_HEADER),
            );
            out.fail_models(&key, $code, &detail);
            return out;
        }};
    }

    // ── 1. the chain ───────────────────────────────────────────────────
    let text = match read_gz(&capture.path) {
        Ok(t) => t,
        Err(e) => bail!("read_failed", format!("{e}")),
    };
    let chain: OptionChain = match OptionChain::from_json(&text) {
        Ok(c) => c,
        Err(e) => bail!("chain_parse_failed", format!("{e}")),
    };
    ctx_as_of = chain.as_of.to_string();
    ctx_stamp = clean(chain.timestamp.as_deref().unwrap_or(""));
    let n_quotes_raw = chain.quotes.len();
    let expiries_raw: std::collections::BTreeSet<NaiveDate> =
        chain.quotes.iter().map(|q| q.expiry).collect();
    ctx_quotes = n_quotes_raw.to_string();
    ctx_expiries = expiries_raw.len().to_string();
    let Some(spot) = chain.spot.filter(|s| s.is_finite() && *s > 0.0) else {
        bail!("no_spot", "the chain carries no usable underlying price")
    };
    ctx_spot = spot;
    let raw_per_expiry: BTreeMap<NaiveDate, usize> =
        chain.quotes.iter().fold(BTreeMap::new(), |mut m, q| {
            *m.entry(q.expiry).or_insert(0) += 1;
            m
        });

    // ── 2. the curve ───────────────────────────────────────────────────
    let (curve, curve_date) = match load_curve(capture.date, chain.as_of) {
        Ok(c) => c,
        Err(e) => bail!("curve_failed", format!("{e}")),
    };
    ctx_curve_date = curve_date.to_string();

    // ── 2b. de-Americanize: strip the early-exercise premium ───────────
    // Every quote's mid is lowered by its CRR-measured premium before
    // any implied vol is solved, so the parity forwards and the smiles
    // all six arms fit are European-equivalent. The carry is implied
    // from the chain's own parity forwards, never assumed.
    let (chain, dea) = match guarded(|| de_americanize_chain(&chain, &curve, &dea_config())) {
        Ok(Ok(x)) => x,
        Ok(Err(e)) => bail!("dea_failed", format!("{e}")),
        Err(p) => bail!("dea_failed", p),
    };
    for e in &dea.expiries {
        let shift_bp = if e.forward_raw > 0.0 {
            (e.forward_dea - e.forward_raw) / e.forward_raw * 1e4
        } else {
            f64::NAN
        };
        out.dea.push(
            key.base()
                .s(e.expiry.to_string())
                .f(e.t)
                .f(e.forward_raw)
                .f(e.forward_dea)
                .f(shift_bp)
                .f(e.implied_carry)
                .u(e.corrected)
                .f(e.median_eep_call_bp)
                .f(e.median_eep_put_bp)
                .f(e.max_eep_bp)
                .u(dea.rounds)
                .u(dea.skipped_floor)
                .f(dea.max_forward_shift_bp)
                .finish(DEA_HEADER),
        );
    }

    // ── 3. clean -> parity forwards -> Black-76 implied vols ───────────
    let (raw_surface, report) =
        match implied_vol_surface_from_chain(&chain, &curve, &filter_config()) {
            Ok(x) => x,
            Err(e) => bail!("surface_failed", format!("{e}")),
        };
    let day_count = DayCountConvention::Act365;
    let forward_points: Vec<(f64, f64)> = report
        .forwards
        .iter()
        .map(|&(expiry, f)| (day_count.year_fraction(chain.as_of, expiry), f))
        .collect();
    let expiry_by_t: Vec<(f64, NaiveDate)> = report
        .forwards
        .iter()
        .map(|&(expiry, _)| (day_count.year_fraction(chain.as_of, expiry), expiry))
        .collect();
    let fwd = |t: f64| lerp_pairs(&forward_points, t);

    // ── 4. minimal-change static-arbitrage repair (ONCE, shared) ───────
    let (repaired, repair) = match repair_arbitrage(&raw_surface, fwd) {
        Ok(x) => x,
        Err(e) => bail!("repair_failed", format!("{e}")),
    };
    let pillars = pillar_smiles(&repaired, &fwd);
    // The quoted log-moneyness span per pillar, from the repaired input
    // every model was handed -- so the "inside the quotes" mask below is
    // literally identical for the three of them.
    let quoted_spans: Vec<(f64, (f64, f64))> = pillars
        .iter()
        .map(|(t, smile)| {
            let f = fwd(*t);
            let (lo, hi) = smile
                .iter()
                .fold((f64::MAX, f64::MIN), |(lo, hi), &(k, _)| {
                    (lo.min(k), hi.max(k))
                });
            (*t, ((lo / f).ln(), (hi / f).ln()))
        })
        .collect();
    if pillars.is_empty() {
        bail!("no_pillars_after_repair", "the repaired surface carries no strike smiles")
    }
    let t_first = pillars.first().map(|p| p.0).unwrap_or(f64::NAN);
    let t_last = pillars.last().map(|p| p.0).unwrap_or(f64::NAN);

    // The repaired pillar quotes themselves, daily slice only (the
    // intraday slice would multiply the file by twenty-five).
    if slice == "daily" {
        for (t, smile) in &pillars {
            let expiry = nearest(&expiry_by_t, *t)
                .map(|&(_, d)| d.to_string())
                .unwrap_or_default();
            let f = fwd(*t);
            for &(strike, vol) in smile {
                out.pillars.push(
                    key.base()
                        .s(&expiry)
                        .f(*t)
                        .f(f)
                        .f(strike)
                        .f(vol)
                        .finish(PILLARS_HEADER),
                );
            }
        }
    }

    // ── 5. the five parameterizations, on those same pillars ───────────
    let n_pillar_quotes: usize = pillars.iter().map(|(_, sm)| sm.len()).sum();
    for id in ModelId::ALL {
        let t_fit = Instant::now();
        let mut cold_iters: Option<usize> = None;
        let mut dim: Option<usize> = None;
        let mut global_diag: Option<rustyqlib::equity::essvi::EssviGlobalDiag> = None;
        let fitted = match id {
            ModelId::Svi => SviSurfaceFit::fit(&repaired, fwd).map(Fitted::Svi),
            ModelId::Ssvi => SsviSurfaceFit::fit(&repaired, fwd).map(Fitted::Ssvi),
            ModelId::Essvi => EssviSurfaceFit::fit_with(
                &repaired,
                fwd,
                &EssviFitConfig {
                    calendar_penalty: ESSVI_CALENDAR_PENALTY,
                    butterfly_penalty: ESSVI_BUTTERFLY_PENALTY,
                },
            )
            .map(Fitted::Essvi),
            ModelId::EssviG05 | ModelId::EssviGfree => {
                let gamma = match id {
                    ModelId::EssviG05 => Some(0.5),
                    _ => None,
                };
                EssviSurfaceFit::fit_global(
                    &repaired,
                    fwd,
                    &EssviGlobalConfig {
                        gamma,
                        start: None,
                        max_iterations: 200,
                    },
                )
                .map(|(f, d)| {
                    cold_iters = Some(d.iterations);
                    dim = Some(d.x.len());
                    global_diag = Some(d.clone());
                    Fitted::EssviGlobal(f, d)
                })
            }
            ModelId::EssviSeqG05 => {
                EssviSurfaceFit::fit_sequential_backbone(&repaired, fwd, 0.5).map(|(f, d)| {
                    cold_iters = Some(d.iterations);
                    // one rho per slice; no joint parameter vector, no warm start
                    Fitted::EssviGlobal(f, d)
                })
            }
        };
        let cold_ms = t_fit.elapsed().as_secs_f64() * 1e3;

        // timing row -- plus, for the global arms, the warm-start
        // experiment seeded from the previous capture of this ticker.
        {
            let (n_pill, cold_conv, cold_rmse) = match &fitted {
                Ok(f) => {
                    let rows = f.slice_rows();
                    let n = rows.len();
                    let conv = rows.iter().all(|r| r.converged);
                    let rmse = if n == 0 {
                        f64::NAN
                    } else {
                        (rows.iter().map(|r| r.lib_rmse * r.lib_rmse).sum::<f64>() / n as f64)
                            .sqrt()
                            * 1e4
                    };
                    (n, conv, rmse)
                }
                Err(_) => (0, false, f64::NAN),
            };
            let mut row = key
                .row(id)
                .u(n_pill)
                .u(n_pillar_quotes)
                .ou(dim)
                .f(cold_ms)
                .ou(cold_iters)
                .b(cold_conv)
                .f(cold_rmse);
            let mut warm_done = false;
            if let (Ok(_), Some(d)) = (&fitted, &global_diag) {
                if let Some(prev_x) = warm.get(id.name()) {
                    if prev_x.len() == d.x.len() {
                        let gamma = match id {
                            ModelId::EssviG05 => Some(0.5),
                            _ => None,
                        };
                        let t_warm = Instant::now();
                        if let Ok((wf, wd)) = EssviSurfaceFit::fit_global(
                            &repaired,
                            fwd,
                            &EssviGlobalConfig {
                                gamma,
                                start: Some(prev_x.clone()),
                                max_iterations: 200,
                            },
                        ) {
                            let warm_ms = t_warm.elapsed().as_secs_f64() * 1e3;
                            let n = wf.slices.len().max(1);
                            let warm_rmse = (wf
                                .slices
                                .iter()
                                .map(|sl| sl.rmse * sl.rmse)
                                .sum::<f64>()
                                / n as f64)
                                .sqrt()
                                * 1e4;
                            let x_maxdiff = wd
                                .x
                                .iter()
                                .zip(&d.x)
                                .map(|(a, b)| (a - b).abs())
                                .fold(0.0f64, f64::max);
                            row = row
                                .b(true)
                                .f(warm_ms)
                                .u(wd.iterations)
                                .b(wd.converged)
                                .f(warm_rmse)
                                .f(x_maxdiff);
                            warm_done = true;
                        }
                    }
                }
                warm.insert(id.name(), d.x.clone());
            }
            if !warm_done {
                row = row.b(false);
            }
            out.timing.push(row.fill(TIMING_HEADER));
        }

        match fitted {
            Err(e) => out.fail_model(&key, id, "fit_failed", &format!("{e}")),
            Ok(fit) => emit_model(
                &mut out,
                &key,
                id,
                &fit,
                &pillars,
                &quoted_spans,
                &expiry_by_t,
                &raw_per_expiry,
                &repaired,
                &curve,
                spot,
                mc_paths,
                mc_steps,
            ),
        }
    }

    // The provenance row is written last so `elapsed_ms` covers the whole
    // capture -- ingest, cleaning, repair and all three models.
    let drop = |k: &str| report.dropped.get(k).copied().unwrap_or(0);
    let rate = |t: f64| curve.zero_rate_with(t, Compounding::Continuous);
    out.panel.push(
        key.base()
            .s("ok")
            .s("")
            .s(&rel)
            .s(chain.as_of.to_string())
            .s(clean(chain.timestamp.as_deref().unwrap_or("")))
            .f(spot)
            .s(curve_date.to_string())
            .u(n_quotes_raw)
            .u(expiries_raw.len())
            .u(report.quotes_used)
            .u(report.quotes_dropped())
            .u(drop("unquoted_or_crossed"))
            .u(drop("wide_spread"))
            .u(drop("expiry_window"))
            .u(drop("moneyness_window"))
            .u(drop("in_the_money"))
            .u(drop("vol_out_of_bounds"))
            .u(drop("solver_failed"))
            .u(drop("sparse_expiry"))
            .u(drop("no_forward"))
            .u(pillars.len())
            .f(t_first)
            .f(t_last)
            .f(fwd(MATURITIES[0]))
            .f(fwd(MATURITIES[1]))
            .f(fwd(MATURITIES[2]))
            .f(rate(MATURITIES[0]))
            .f(rate(MATURITIES[1]))
            .f(rate(MATURITIES[2]))
            .f(rate(1.0))
            .u(report.diagnostics.butterfly.len())
            .u(report.diagnostics.calendar.len())
            .u(repair.butterfly_adjustments)
            .u(repair.calendar_adjustments)
            .u(repair.dropped_points)
            .f(repair.max_vol_change)
            .u(repair.iterations)
            .b(repair.clean)
            .f(t0.elapsed().as_secs_f64() * 1e3)
            .finish(PANEL_HEADER),
    );

    out
}

#[allow(clippy::too_many_arguments)]
fn emit_model(
    out: &mut Output,
    key: &Key,
    id: ModelId,
    fit: &Fitted,
    pillars: &[(f64, Vec<(f64, f64)>)],
    quoted_spans: &[(f64, (f64, f64))],
    expiry_by_t: &[(f64, NaiveDate)],
    raw_per_expiry: &BTreeMap<NaiveDate, usize>,
    repaired: &VolSurface,
    curve: &YieldCurve,
    spot: f64,
    mc_paths: usize,
    mc_steps: usize,
) {
    let slice_rows = fit.slice_rows();
    let n_fitted = slice_rows.len();
    let n_skipped = fit.skipped_slices();
    let self_ok = fit.self_validates();

    // ── fits.csv: one row per fitted expiry slice ──────────────────────
    if slice_rows.is_empty() {
        out.fits.push(
            key.row(id)
                .s("no_slices_fitted")
                .s("the parameterization fitted zero expiry slices")
                .fill(FITS_HEADER),
        );
    }
    for (i, row) in slice_rows.iter().enumerate() {
        // Fit error is measured HERE, identically for the three models,
        // against the repaired pillars all three were handed -- rather
        // than trusting each model's own reported RMSE, which is defined
        // per model. The library figure is kept alongside it.
        //
        // `SmoothedSurface::vol` is the *published* surface, which for
        // the per-expiry parameterizations floors total variance against
        // the previous slice. A degenerate neighbouring slice therefore
        // shows up here even when the model's own per-slice RMSE is
        // small -- which is the point, but it also means this column is
        // heavy-tailed and must be summarized by medians. The fitted and
        // market vol ranges are written alongside so an implausible
        // implied vol is visible in the data rather than only in a
        // blown-up error statistic.
        let mut sq = 0.0;
        let mut worst: f64 = 0.0;
        let mut n_used = 0usize;
        let (mut fit_lo, mut fit_hi) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut mkt_lo, mut mkt_hi) = (f64::INFINITY, f64::NEG_INFINITY);
        if let Some((_, smile)) = nearest(pillars, row.t) {
            for &(strike, vol) in smile {
                let model_vol = SmoothedSurface::vol(fit, strike, row.t);
                let e = (model_vol - vol).abs();
                sq += e * e;
                worst = worst.max(e);
                fit_lo = fit_lo.min(model_vol);
                fit_hi = fit_hi.max(model_vol);
                mkt_lo = mkt_lo.min(vol);
                mkt_hi = mkt_hi.max(vol);
                n_used += 1;
            }
        }
        let (rmse_bp, max_bp) = if n_used == 0 {
            (f64::NAN, f64::NAN)
        } else {
            ((sq / n_used as f64).sqrt() * 1e4, worst * 1e4)
        };
        let expiry = nearest(expiry_by_t, row.t).map(|&(_, d)| d);
        let n_raw = expiry
            .and_then(|d| raw_per_expiry.get(&d).copied())
            .unwrap_or(0);
        out.fits.push(
            key.row(id)
                .s("ok")
                .s("")
                .u(i)
                .s(expiry.map(|d| d.to_string()).unwrap_or_default())
                .f(row.t)
                .f(row.forward)
                .u(n_raw)
                .u(n_used)
                .f(rmse_bp)
                .f(max_bp)
                .f(row.lib_rmse * 1e4)
                .f(fit_lo)
                .f(fit_hi)
                .f(mkt_lo)
                .f(mkt_hi)
                .b(row.converged)
                .f(row.min_g)
                .f(row.k_lo)
                .f(row.k_hi)
                .u(n_fitted)
                .u(n_skipped)
                .b(self_ok)
                .of(row.svi_a)
                .of(row.svi_b)
                .of(row.svi_rho)
                .of(row.svi_m)
                .of(row.svi_sigma)
                .of(row.ssvi_rho)
                .of(row.ssvi_eta)
                .of(row.ssvi_gamma)
                .of(row.ssvi_theta)
                .of(row.ssvi_phi)
                .of(row.essvi_theta)
                .of(row.essvi_psi)
                .of(row.essvi_rho)
                .of(row.essvi_eta)
                .of(row.essvi_gamma)
                .of(row.essvi_tube_clamped)
                .finish(FITS_HEADER),
        );
    }

    // The accumulated drift the model's own local vol is consistent
    // with. Everything stochastic downstream uses it.
    let carry = carry_of(fit, spot);

    let pillar_times = fit.pillar_times();
    let (t_lo, t_hi) = match (pillar_times.first(), pillar_times.last()) {
        (Some(&a), Some(&b)) if b > a && a > 0.0 => (a, b),
        (Some(&a), Some(&b)) if a > 0.0 => (a, b + 1e-3),
        _ => {
            out.fail_model(key, id, "no_usable_pillars", "the fit produced no positive-time pillar");
            return;
        }
    };

    // ── the common dense grid ──────────────────────────────────────────
    let ks: Vec<f64> = (0..GRID_K_POINTS)
        .map(|i| GRID_K_LO + (GRID_K_HI - GRID_K_LO) * i as f64 / (GRID_K_POINTS - 1) as f64)
        .collect();
    let ts: Vec<f64> = (0..GRID_T_POINTS)
        .map(|j| t_lo + (t_hi - t_lo) * j as f64 / (GRID_T_POINTS - 1) as f64)
        .collect();
    let dk = ks[1] - ks[0];
    let n_grid = ks.len() * ts.len();

    // ── arbitrage.csv: measured on the FITTED surface ──────────────────
    // `inside[j][i]` = grid point (k_i, t_j) sits within the quoted
    // log-moneyness span of the nearest input pillar. Built from the
    // shared repaired pillars, so it is the same mask for all three
    // models and the restricted statistics stay comparable.
    let inside: Vec<Vec<bool>> = ts
        .iter()
        .map(|&t| {
            let (lo, hi) = nearest(quoted_spans, t).map(|s| s.1).unwrap_or((0.0, 0.0));
            ks.iter().map(|&k| k >= lo && k <= hi).collect()
        })
        .collect();
    let n_inside: usize = inside.iter().flatten().filter(|b| **b).count();

    let mut butterfly_violations = 0usize;
    let mut butterfly_inside = 0usize;
    let mut min_g = f64::MAX;
    let mut min_g_inside = f64::MAX;
    for (j, &t) in ts.iter().enumerate() {
        for (i, &k) in ks.iter().enumerate() {
            let g = fit.butterfly_g_at(k, t);
            if g < 0.0 {
                butterfly_violations += 1;
                if inside[j][i] {
                    butterfly_inside += 1;
                }
            }
            min_g = min_g.min(g);
            if inside[j][i] {
                min_g_inside = min_g_inside.min(g);
            }
        }
    }
    let mut calendar_crossings = 0usize;
    let mut calendar_inside = 0usize;
    let mut max_crossing: f64 = 0.0;
    let mut max_crossing_inside: f64 = 0.0;
    for (i, &k) in ks.iter().enumerate() {
        for j in 1..ts.len() {
            let dw = SmoothedSurface::total_variance(fit, k, ts[j])
                - SmoothedSurface::total_variance(fit, k, ts[j - 1]);
            if dw < -1e-12 {
                calendar_crossings += 1;
                max_crossing = max_crossing.max(-dw);
                if inside[j][i] && inside[j - 1][i] {
                    calendar_inside += 1;
                    max_crossing_inside = max_crossing_inside.max(-dw);
                }
            }
        }
    }
    let min_g_inside = if n_inside == 0 { f64::NAN } else { min_g_inside };
    let n_cal_pairs = ks.len() * (ts.len() - 1);
    out.arbitrage.push(
        key.row(id)
            .s("ok")
            .s("")
            .f(GRID_K_LO)
            .f(GRID_K_HI)
            .u(GRID_K_POINTS)
            .f(t_lo)
            .f(t_hi)
            .u(GRID_T_POINTS)
            .u(n_grid)
            .u(butterfly_violations)
            .f(butterfly_violations as f64 / n_grid as f64)
            .f(min_g)
            .u(n_cal_pairs)
            .u(calendar_crossings)
            .f(calendar_crossings as f64 / n_cal_pairs as f64)
            .f(max_crossing)
            .f(fit.fit_max_calendar_crossing())
            .b(self_ok)
            .f(n_inside as f64 / n_grid as f64)
            .u(butterfly_inside)
            .f(min_g_inside)
            .u(calendar_inside)
            .f(max_crossing_inside)
            .finish(ARB_HEADER),
    );

    // ── localvol.csv ───────────────────────────────────────────────────
    // levels are underlying levels (spot-anchored), the coordinate the
    // local vol function actually takes
    let levels: Vec<f64> = ks.iter().map(|&k| spot * k.exp()).collect();
    let mut lv: Vec<Vec<f64>> = Vec::with_capacity(levels.len());
    let mut guard_hits = 0usize;
    let mut clamped = 0usize;
    let (mut lv_min, mut lv_max, mut lv_sum) = (f64::MAX, f64::MIN, 0.0);
    for &level in &levels {
        let mut row = Vec::with_capacity(ts.len());
        for &t in &ts {
            let (v, g) = fit.local_vol_checked(level, t);
            if g {
                guard_hits += 1;
            }
            // pinned at the shared [1%, 300%] clamps (1% margin)
            if v <= 0.01 * 1.01 || v >= 3.0 * 0.999 {
                clamped += 1;
            }
            lv_min = lv_min.min(v);
            lv_max = lv_max.max(v);
            lv_sum += v;
            row.push(v);
        }
        lv.push(row);
    }
    // roughness: mean / max |second difference| in log-strike at fixed t
    let mut rough_sum = 0.0;
    let mut rough_max: f64 = 0.0;
    let mut rough_n = 0usize;
    for i in 1..levels.len() - 1 {
        for j in 0..ts.len() {
            let d2 = (lv[i + 1][j] - 2.0 * lv[i][j] + lv[i - 1][j]).abs();
            rough_sum += d2;
            rough_max = rough_max.max(d2);
            rough_n += 1;
        }
    }
    let rough_mean = rough_sum / rough_n.max(1) as f64;

    // the sampled artifact and its NUMERICAL Dupire: the gap between
    // "the model's analytic local vol" and "what a VolSurface consumer
    // computes from the published surface"
    let sampled = fit.to_vol_surface(SURFACE_SAMPLES);
    // `LocalVol` rebuilds its own forward as `S e^{(r-q)t}`. With q = 0
    // that is not the forward the fitted smile is quoted against, and
    // the mismatch would land in this gap as a spurious skew term. Use
    // the carry the fit actually carries, at the middle of the grid.
    let t_mid = 0.5 * (t_lo + t_hi);
    let q_grid = curve.zero_rate_with(t_mid, Compounding::Continuous) - carry(t_mid) / t_mid;
    let (gap_mean, gap_max) = match &sampled {
        Ok(surface) => {
            let numeric =
                rustyqlib::equity::local_vol::LocalVol::new(surface, curve, spot, q_grid, 0.0);
            let mut sum = 0.0;
            let mut worst: f64 = 0.0;
            for (i, &level) in levels.iter().enumerate() {
                for (j, &t) in ts.iter().enumerate() {
                    let d = (lv[i][j] - numeric.vol(level, t)).abs();
                    sum += d;
                    worst = worst.max(d);
                }
            }
            (sum / n_grid as f64, worst)
        }
        Err(_) => (f64::NAN, f64::NAN),
    };

    let rt = sampled
        .as_ref()
        .ok()
        .map(|surface| round_trip(surface, repaired, fit, curve, &carry, spot, pillars))
        .unwrap_or_default();

    // the library's usability verdict, on the model's ANALYTIC local vol
    let ur_attempt = sampled.as_ref().ok().map(|surface| {
        guarded(|| {
            usability_report(
                surface,
                &|level, t| fit.local_vol_checked(level, t),
                &levels,
                &ts,
                curve,
                spot,
                &UsabilityConfig {
                    moneyness: USABILITY_MONEYNESS,
                    max_expiries: USABILITY_MAX_EXPIRIES,
                    strikes_per_expiry: USABILITY_STRIKES_PER_EXPIRY,
                    martingale: None,
                },
            )
        })
    });
    let ur = match &ur_attempt {
        Some(Ok(u)) => Some(u),
        _ => None,
    };
    // A panic inside the library's usability report used to leave the
    // ur_* columns blank on a row still marked `ok`, which is
    // indistinguishable from "not attempted". Say so instead.
    let ur_error = match &ur_attempt {
        Some(Err(e)) => Some(e.clone()),
        _ => None,
    };

    let lv_status = match (&sampled, rt.points, &rt.detail) {
        (Err(e), _, _) => ("sample_failed".to_string(), clean(e)),
        (Ok(_), 0, Some(d)) => ("roundtrip_failed".to_string(), clean(d)),
        (Ok(_), _, Some(d)) => ("ok_partial_roundtrip".to_string(), clean(d)),
        (Ok(_), _, None) => match &ur_error {
            Some(e) => ("ok_no_usability".to_string(), clean(e)),
            None => ("ok".to_string(), String::new()),
        },
    };
    let mut lv_row = key
        .row(id)
        .s(&lv_status.0)
        .s(&lv_status.1)
        .u(levels.len())
        .u(ts.len())
        .f(lv_min)
        .f(lv_max)
        .f(lv_sum / n_grid as f64)
        .f(clamped as f64 / n_grid as f64)
        .f(guard_hits as f64 / n_grid as f64)
        .f(dk)
        .f(rough_mean)
        .f(rough_max)
        .f(rough_mean / (dk * dk))
        .f(gap_mean)
        .f(gap_max)
        .u(rt.points)
        .u(rt.failures)
        .u(rt.panics)
        .f(rt.fit_mean_bp)
        .f(rt.fit_max_bp)
        .f(rt.fit_mean_abs_price)
        .f(rt.fit_max_abs_price)
        .f(rt.fit_mean_rel_price)
        .f(rt.mkt_mean_bp)
        .f(rt.mkt_max_bp);
    lv_row = match &ur {
        Some(u) => lv_row
            .u(u.roundtrip.points)
            .u(u.roundtrip.failures)
            .f(u.roundtrip.mean_vol_bps)
            .f(u.roundtrip.max_vol_bps)
            .f(u.clamped_fraction)
            .f(u.fallback_fraction)
            .b(u.within_desk_tolerance)
            .f(u.trusted_region.strike_lo)
            .f(u.trusted_region.strike_hi)
            .f(u.trusted_region.t_lo)
            .f(u.trusted_region.t_hi),
        None => lv_row,
    };
    out.localvol.push(lv_row.fill(LOCALVOL_HEADER));

    // ── one path set: vanillas + every barrier + forward recovery ──────
    let strike_fwd = STRIKE_AT_FORWARD.get().copied().unwrap_or(false);
    let barrier_pct = BARRIER_PCT.get().copied().flatten();
    let strike_sd = STRIKE_SD.get().copied().flatten();
    let (call_strikes, put_strikes): (Vec<f64>, Vec<f64>) = MATURITIES
        .iter()
        .map(|&t| {
            let base = if strike_fwd { SmoothedSurface::forward(fit, t) } else { spot * STRIKE_MULT };
            match strike_sd {
                Some(ms) => {
                    let f = SmoothedSurface::forward(fit, t);
                    let d = ms * repaired.vol(f, f, t) * t.sqrt();
                    (f * d.exp(), f * (-d).exp())
                }
                None => (base, base),
            }
        })
        .unzip();
    // `owner[b]` is the maturity a barrier belongs to under --barrier-pct,
    // None on the released grid where every level applies at every maturity.
    let barrier_sd = BARRIER_SD.get().cloned().flatten();
    let (barriers, owner): (Vec<BarrierLevel>, Vec<Option<usize>>) = match (barrier_sd.as_ref(), barrier_pct) {
        (Some(sd), _) => {
            // one pair per maturity at K exp(-/+ m sigma sqrt(T)), sigma the market
            // ATM vol at that maturity, identical for every arm
            let mut bs = Vec::new();
            let mut ow = Vec::new();
            for (mi, &t) in MATURITIES.iter().enumerate() {
                let mult = sd[mi];
                if mult <= 0.0 {
                    continue;
                }
                let f = SmoothedSurface::forward(fit, t);
                let sigma = repaired.vol(f, f, t);
                let dist = mult * sigma * t.sqrt();
                // centred on the forward when the strikes are out of the money,
                // on the (common) strike otherwise
                let centre = if strike_sd.is_some() { f } else { call_strikes[mi] };
                bs.push(BarrierLevel { down: true, mult: centre * (-dist).exp() / spot });
                ow.push(Some(mi));
                bs.push(BarrierLevel { down: false, mult: centre * dist.exp() / spot });
                ow.push(Some(mi));
            }
            (bs, ow)
        }
        (None, Some(p)) => {
            let mut bs = Vec::new();
            let mut ow = Vec::new();
            for (mi, &k) in call_strikes.iter().enumerate() {
                bs.push(BarrierLevel { down: true, mult: k * (1.0 - p) / spot });
                ow.push(Some(mi));
                bs.push(BarrierLevel { down: false, mult: k * (1.0 + p) / spot });
                ow.push(Some(mi));
            }
            (bs, ow)
        }
        (None, None) => (
            BARRIER_LEVELS.iter().map(|&(down, mult)| BarrierLevel { down, mult }).collect(),
            vec![None; BARRIER_LEVELS.len()],
        ),
    };
    let mc = SimConfig {
        paths: mc_paths,
        steps_per_year: mc_steps,
        min_substeps_per_leg: MC_MIN_SUBSTEPS_PER_LEG,
        seed: MC_SEED,
    };
    let result = sim::simulate(
        &|level, t| fit.local_vol(level, t),
        &carry,
        curve,
        spot,
        &call_strikes,
        &put_strikes,
        &MATURITIES,
        &barriers,
        &mc,
    );

    for (m, &t) in MATURITIES.iter().enumerate() {
        // Both directions of extrapolation matter and only one of them
        // was flagged. A quarter of the panel has no listed expiry
        // inside a month, so its 1M row is the surface extended *below*
        // its first pillar -- where variance is accrued proportionally
        // from zero and the smile is simply the first slice's. Anything
        // that treats those rows as calibrated is measuring the
        // extrapolation rule, not the parameterization.
        let beyond = t > t_hi;
        let below = t < t_lo;
        let forward = SmoothedSurface::forward(fit, t);
        let df = curve.df(t);
        let vc = result.vanilla_call[m];
        let vp = result.vanilla_put[m];
        let base = |k: f64| {
            key.row(id)
                .s("ok")
                .s("")
                .s(MATURITY_LABELS[m])
                .f(t)
                .b(beyond)
                .b(below)
                .f(spot)
                .f(k)
                .f(forward)
                .f(df)
        };
        // the vanillas: closed form at the arm's own fitted vol at the strike
        // (pv), with the local-vol Monte Carlo vanilla on the shared paths
        // beside it (vanilla_pv) and their ratio as a per-row round-trip check
        let rate = -df.ln() / t;
        let q = rate - carry(t) / t;
        for (right, stat, k, side) in [
            ("C", vc, call_strikes[m], PutOrCall::Call),
            ("P", vp, put_strikes[m], PutOrCall::Put),
        ] {
            let fitted_vol = SmoothedSurface::vol(fit, k, t);
            let closed = bs_price(spot, k, rate, q, fitted_vol, t, side);
            out.barriers.push(
                base(k)
                    .s("none")
                    .s("")
                    .s("")
                    .s("")
                    .s(right)
                    .f(closed)
                    .f(0.0)
                    .s("")
                    .f(stat.mean)
                    .f(stat.std_err)
                    .f(if stat.mean > 1e-12 { closed / stat.mean } else { f64::NAN })
                    .u(result.paths)
                    .u(result.steps)
                    .n(MC_SEED)
                    .finish(BARRIERS_HEADER),
            );
        }
        for (b, spec) in barriers.iter().enumerate() {
            if let Some(o) = owner[b] {
                if o != m {
                    continue;
                }
            }
            let dir = if spec.down { "down" } else { "up" };
            let level = spot * spec.mult;
            let mult_out = match (barrier_sd.as_ref(), barrier_pct) {
                (Some(sd), _) => sd[m],
                (None, Some(p)) => if spec.down { 1.0 - p } else { 1.0 + p },
                (None, None) => spec.mult,
            };
            for (right, stat, vanilla) in [
                ("C", result.ko_call[m][b], vc),
                ("P", result.ko_put[m][b], vp),
            ] {
                out.barriers.push(
                    base(if right == "C" { call_strikes[m] } else { put_strikes[m] })
                        .s("out")
                        .s(dir)
                        .f(mult_out)
                        .f(level)
                        .s(right)
                        .f(stat.mean)
                        .f(stat.std_err)
                        .f(result.knock_prob[m][b])
                        .f(vanilla.mean)
                        .f(vanilla.std_err)
                        .f(if vanilla.mean > 1e-12 {
                            stat.mean / vanilla.mean
                        } else {
                            f64::NAN
                        })
                        .u(result.paths)
                        .u(result.steps)
                        .n(MC_SEED)
                        .finish(BARRIERS_HEADER),
                );
            }
        }
    }

    // ── martingale.csv ─────────────────────────────────────────────────
    let targets: Vec<(f64, f64)> = MATURITIES
        .iter()
        .map(|&t| (t, SmoothedSurface::forward(fit, t)))
        .collect();
    // `martingale_report` uses its curve only for the per-step drift, so
    // handing it the growth curve makes the simulation consistent with
    // the forward the targets are read from. Without this the reported
    // relative error is (to 3 significant figures) just the implied
    // dividend/borrow yield and says nothing about the local vol.
    let mart_curve = growth_curve(curve.reference_date(), &carry, &MATURITIES);
    let (drift_curve, drift_source) = match &mart_curve {
        Ok(c) => (c, "parity_forward"),
        Err(_) => (curve, "zero_dividend_fallback"),
    };
    let mart = martingale_report(
        &|level, t| fit.local_vol(level, t),
        drift_curve,
        spot,
        &targets,
        &MartingaleConfig {
            paths: MARTINGALE_PATHS,
            steps_per_year: MARTINGALE_STEPS_PER_YEAR,
            seed: MARTINGALE_SEED,
            z_threshold: MARTINGALE_Z_THRESHOLD,
        },
    );
    if mart.checks.is_empty() {
        out.fail_model(key, id, "martingale_no_targets", "no positive-time forward target");
    }
    for check in mart.checks.iter() {
        // `martingale_report` sorts and filters its targets, so the index
        // into `checks` is not automatically the index into MATURITIES.
        // Pair on the horizon time itself.
        let m = MATURITIES
            .iter()
            .position(|&t| (t - check.t).abs() < 1e-12);
        let mc_mean = m
            .and_then(|m| result.mean_spot.get(m).copied())
            .unwrap_or_default();
        out.martingale.push(
            key.row(id)
                .s("ok")
                .s("")
                .s(m.map(|m| MATURITY_LABELS[m]).unwrap_or(""))
                .f(check.t)
                .b(check.t > t_hi)
                .b(check.t < t_lo)
                .f(check.target_forward)
                .f(check.simulated_mean)
                .f(check.relative_error)
                .f(check.standard_error)
                .f(check.z_score)
                .u(mart.paths)
                .n(mart.seed)
                .u(MARTINGALE_STEPS_PER_YEAR)
                .f(MARTINGALE_Z_THRESHOLD)
                .b(mart.within_threshold)
                .f(mart.max_abs_z)
                .f(mc_mean.mean)
                .f(mc_mean.mean / check.target_forward - 1.0)
                .f(mc_mean.std_err / check.target_forward)
                .u(result.paths)
                .n(MC_SEED)
                .u(result.steps)
                // the carry the simulation now drifts at, and the error
                // the old zero-dividend drift produced -- kept so the
                // modelling gap between S e^{rT} and the parity forward
                // is still reported, under its own name
                .f(carry(check.t) / check.t)
                .f(
                    spot * (curve.zero_rate_with(check.t, Compounding::Continuous) * check.t).exp()
                        / check.target_forward
                        - 1.0,
                )
                .s(drift_source)
                .finish(MARTINGALE_HEADER),
        );
    }
}

// ── Round-trip repricing ───────────────────────────────────────────────

#[derive(Default)]
struct RoundTrip {
    points: usize,
    failures: usize,
    /// Failures that came from a library assertion rather than a
    /// returned error (see `guarded`).
    panics: usize,
    /// First failure reason seen, for `status_detail`.
    detail: Option<String>,
    fit_mean_bp: f64,
    fit_max_bp: f64,
    fit_mean_abs_price: f64,
    fit_max_abs_price: f64,
    fit_mean_rel_price: f64,
    mkt_mean_bp: f64,
    mkt_max_bp: f64,
}

/// Reprice the calibrating vanillas under the local vol implied by the
/// fitted surface, on the library's finite-difference `Model::LocalVol`
/// engine -- what a desk gets when it publishes the fitted surface and
/// prices off it.
///
/// Two errors are reported: against the **fit's own** implied vol (which
/// isolates the Dupire + PDE step) and against the **market** pillar
/// implied vol read off the repaired input surface (which additionally
/// carries the smoothing error). Strikes outside the expiry's quoted
/// span are skipped -- extrapolation is not a round trip.
#[allow(clippy::too_many_arguments)]
fn round_trip(
    sampled: &VolSurface,
    market: &VolSurface,
    fit: &Fitted,
    curve: &YieldCurve,
    carry: &dyn Fn(f64) -> f64,
    spot: f64,
    pillars: &[(f64, Vec<(f64, f64)>)],
) -> RoundTrip {
    let mut rt = RoundTrip::default();
    let reference = sampled.reference_date();
    let mut fit_bp: Vec<f64> = Vec::new();
    let mut mkt_bp: Vec<f64> = Vec::new();
    let mut abs_price: Vec<f64> = Vec::new();
    let mut rel_price: Vec<f64> = Vec::new();
    let mut seen: Vec<f64> = Vec::new();

    // the pillar expiries nearest the study's fixed maturities, so the
    // round trip is measured where the barriers are priced
    for &target in MATURITIES.iter() {
        let Some((t, smile)) = nearest(pillars, target) else {
            continue;
        };
        if smile.is_empty() || seen.iter().any(|s| (s - t).abs() < 1e-12) {
            continue;
        }
        seen.push(*t);
        let (k_lo, k_hi) = smile
            .iter()
            .fold((f64::MAX, f64::MIN), |(lo, hi), &(s, _)| {
                (lo.min(s), hi.max(s))
            });
        let forward = SmoothedSurface::forward(fit, *t);
        let maturity = reference + chrono::Days::new((t * 365.0).round().max(3.0) as u64);
        let t_eff = DayCountConvention::Act365.year_fraction(reference, maturity);
        let rate = curve.zero_rate_with(t_eff, Compounding::Continuous);
        // The engine, the target price and the inversion all have to
        // agree on where the forward is. The fitted smile is quoted
        // against the chain's parity forward, so the carry yield that
        // reproduces it is r - ln(F/S)/t; leaving it at zero prices a
        // different option than the one the fit describes and shows up
        // as a fake round-trip error proportional to the skew.
        let q = rate - carry(t_eff) / t_eff;
        for &k in RT_K_OFFSETS.iter() {
            let strike = forward * k.exp();
            if strike < k_lo || strike > k_hi {
                continue;
            }
            let side = if strike >= forward {
                PutOrCall::Call
            } else {
                PutOrCall::Put
            };
            let fit_vol = SmoothedSurface::vol(fit, strike, *t);
            let mkt_vol = market.vol(strike, forward, *t);
            let priced = guarded(|| {
                EquityOptionBuilder::new()
                    .spot(spot)
                    .strike(strike)
                    .vol_surface(sampled.clone())
                    .flat_rate(rate)
                    .valuation_date(reference)
                    .maturity_date(maturity)
                    .dividend_yield(q)
                    .vanilla(side)
                    .engine(Engine::FiniteDifference)
                    .model(Model::LocalVol)
                    .fd_grid(RT_FD_GRID.0, RT_FD_GRID.1)
                    .build()
                    .and_then(|o| o.price())
                    .map(|r| r.pv)
                    .map_err(|e| e.to_string())
            });
            let pv = match priced {
                Ok(Ok(pv)) if pv.is_finite() => pv,
                Ok(Ok(_)) => {
                    rt.failures += 1;
                    rt.detail
                        .get_or_insert_with(|| "non-finite finite-difference pv".to_string());
                    continue;
                }
                Ok(Err(e)) => {
                    rt.failures += 1;
                    rt.detail.get_or_insert(e);
                    continue;
                }
                Err(panic) => {
                    rt.failures += 1;
                    rt.panics += 1;
                    rt.detail.get_or_insert(panic);
                    continue;
                }
            };
            let target_price = bs_price(spot, strike, rate, q, fit_vol, t_eff, side);
            match implied_vol_from_price(spot, strike, rate, q, t_eff, pv, side) {
                Ok(re) => {
                    fit_bp.push((re - fit_vol).abs() * 1e4);
                    mkt_bp.push((re - mkt_vol).abs() * 1e4);
                }
                Err(_) => rt.failures += 1,
            }
            abs_price.push((pv - target_price).abs());
            rel_price.push((pv - target_price).abs() / target_price.max(1e-6));
        }
    }
    let mean = |v: &[f64]| {
        if v.is_empty() {
            f64::NAN
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    };
    let max = |v: &[f64]| v.iter().copied().fold(f64::NAN, f64::max);
    rt.points = fit_bp.len();
    rt.fit_mean_bp = mean(&fit_bp);
    rt.fit_max_bp = max(&fit_bp);
    rt.mkt_mean_bp = mean(&mkt_bp);
    rt.mkt_max_bp = max(&mkt_bp);
    rt.fit_mean_abs_price = mean(&abs_price);
    rt.fit_max_abs_price = max(&abs_price);
    rt.fit_mean_rel_price = mean(&rel_price);
    rt
}

// ── Leave-one-expiry-out (`--panel loo`) ───────────────────────────────
//
// The out-of-sample counterpart of fits.csv: for each daily capture and
// each *interior* pillar, refit every arm on the remaining pillars and
// ask the refitted surface for the held-out smile -- which it can only
// produce by interpolating across the gap. Errors are measured against
// the same repaired pillar vols the in-sample RMSE is measured against,
// at the held-out slice's own strikes, so the two columns are directly
// comparable. The first and last pillars are never held out: a surface
// can only extrapolate to them, which is a different question.
//
// The parity forward at the held-out expiry is NOT handed to the fit:
// evaluation goes through `SmoothedSurface::vol`, whose strike-to-
// moneyness conversion interpolates the *remaining* pillar forwards --
// identically for every arm.

/// Fit one arm on a (possibly reduced) repaired surface. The same calls
/// `process` makes, without the timing/warm-start plumbing.
fn fit_arm(
    id: ModelId,
    surface: &VolSurface,
    fwd: &(dyn Fn(f64) -> f64 + Sync),
) -> Result<Fitted, String> {
    match id {
        ModelId::Svi => SviSurfaceFit::fit(surface, fwd)
            .map(Fitted::Svi)
            .map_err(|e| e.to_string()),
        ModelId::Ssvi => SsviSurfaceFit::fit(surface, fwd)
            .map(Fitted::Ssvi)
            .map_err(|e| e.to_string()),
        ModelId::Essvi => EssviSurfaceFit::fit_with(
            surface,
            fwd,
            &EssviFitConfig {
                calendar_penalty: ESSVI_CALENDAR_PENALTY,
                butterfly_penalty: ESSVI_BUTTERFLY_PENALTY,
            },
        )
        .map(Fitted::Essvi)
        .map_err(|e| e.to_string()),
        ModelId::EssviG05 | ModelId::EssviGfree => {
            let gamma = match id {
                ModelId::EssviG05 => Some(0.5),
                _ => None,
            };
            EssviSurfaceFit::fit_global(
                surface,
                fwd,
                &EssviGlobalConfig {
                    gamma,
                    start: None,
                    max_iterations: 200,
                },
            )
            .map(|(f, d)| Fitted::EssviGlobal(f, d))
            .map_err(|e| e.to_string())
        }
        ModelId::EssviSeqG05 => EssviSurfaceFit::fit_sequential_backbone(surface, fwd, 0.5)
            .map(|(f, d)| Fitted::EssviGlobal(f, d))
            .map_err(|e| e.to_string()),
    }
}

/// Errors of a fitted surface against one repaired pillar smile, at the
/// smile's own strikes: (rmse_bp, max_bp, mean_signed_bp, atm_signed_bp,
/// n). Signed errors are model minus market; `atm` is the quote whose
/// strike is nearest `f_market`.
fn slice_errors(
    fit: &Fitted,
    smile: &[(f64, f64)],
    t: f64,
    f_market: f64,
) -> (f64, f64, f64, f64, usize) {
    let mut sq = 0.0;
    let mut worst: f64 = 0.0;
    let mut signed_sum = 0.0;
    let mut atm = (f64::MAX, f64::NAN); // (|K - F|, signed error)
    let mut n = 0usize;
    for &(strike, vol) in smile {
        let model_vol = SmoothedSurface::vol(fit, strike, t);
        if !model_vol.is_finite() {
            continue;
        }
        let e = model_vol - vol;
        sq += e * e;
        worst = worst.max(e.abs());
        signed_sum += e;
        let d = (strike - f_market).abs();
        if d < atm.0 {
            atm = (d, e);
        }
        n += 1;
    }
    if n == 0 {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN, 0);
    }
    (
        (sq / n as f64).sqrt() * 1e4,
        worst * 1e4,
        signed_sum / n as f64 * 1e4,
        atm.1 * 1e4,
        n,
    )
}

/// A single-model failure row for the LOO CSV.
fn loo_fail_row(key: &Key, id: ModelId, code: &str, detail: &str, rows: &mut Vec<String>) {
    rows.push(key.row(id).s(code).s(clean(detail)).fill(LOO_HEADER));
}

/// One capture's LOO rows: ingest identically to `process`, then hold
/// out each interior pillar in turn.
fn process_loo(capture: &Capture) -> Vec<String> {
    let mut rows: Vec<String> = Vec::new();
    let key = Key {
        slice: "loo".to_string(),
        ticker: capture.ticker.clone(),
        date: capture.date.to_string(),
        time: capture.time.clone(),
    };

    macro_rules! bail {
        ($code:expr, $detail:expr) => {{
            let detail = clean(&$detail);
            for id in ModelId::ALL {
                rows.push(key.row(id).s($code).s(&detail).fill(LOO_HEADER));
            }
            return rows;
        }};
    }

    // ── ingest: read -> parse -> curve -> de-Americanize -> surface ────
    let text = match read_gz(&capture.path) {
        Ok(t) => t,
        Err(e) => bail!("read_failed", format!("{e}")),
    };
    let chain: OptionChain = match OptionChain::from_json(&text) {
        Ok(c) => c,
        Err(e) => bail!("chain_parse_failed", format!("{e}")),
    };
    if chain.spot.filter(|s| s.is_finite() && *s > 0.0).is_none() {
        bail!("no_spot", "no usable underlying price");
    }
    let (curve, _curve_date) = match load_curve(capture.date, chain.as_of) {
        Ok(c) => c,
        Err(e) => bail!("curve_failed", e),
    };
    let (chain, _dea) = match guarded(|| de_americanize_chain(&chain, &curve, &dea_config())) {
        Ok(Ok(x)) => x,
        Ok(Err(e)) => bail!("dea_failed", format!("{e}")),
        Err(p) => bail!("dea_failed", p),
    };
    let (raw_surface, report) =
        match implied_vol_surface_from_chain(&chain, &curve, &filter_config()) {
            Ok(x) => x,
            Err(e) => bail!("surface_failed", format!("{e}")),
        };
    let day_count = DayCountConvention::Act365;
    let forward_points: Vec<(f64, f64)> = report
        .forwards
        .iter()
        .map(|&(expiry, f)| (day_count.year_fraction(chain.as_of, expiry), f))
        .collect();
    let expiry_by_t: Vec<(f64, NaiveDate)> = report
        .forwards
        .iter()
        .map(|&(expiry, _)| (day_count.year_fraction(chain.as_of, expiry), expiry))
        .collect();
    let fwd = |t: f64| lerp_pairs(&forward_points, t);
    let (repaired, _repair) = match repair_arbitrage(&raw_surface, fwd) {
        Ok(x) => x,
        Err(e) => bail!("repair_failed", format!("{e}")),
    };
    let pillars = pillar_smiles(&repaired, &fwd);
    let n_pillars = pillars.len();
    if n_pillars < 3 {
        bail!("too_few_pillars", "need at least 3 pillars to hold one out");
    }

    // ── the full fits: the in-sample side of every comparison ──────────
    let mut full: Vec<Option<Fitted>> = Vec::new();
    for id in ModelId::ALL {
        let f = match guarded(|| fit_arm(id, &repaired, &fwd)) {
            Ok(Ok(f)) => Some(f),
            Ok(Err(e)) => {
                loo_fail_row(&key, id, "full_fit_failed", &e, &mut rows);
                None
            }
            Err(p) => {
                loo_fail_row(&key, id, "full_fit_failed", &p, &mut rows);
                None
            }
        };
        full.push(f);
    }

    // ── the reduced input, one interior pillar at a time ───────────────
    let VolInput::StrikeSmiles {
        expiries,
        smiles,
        coordinate,
        day_count: dc,
    } = repaired.to_input()
    else {
        bail!("bad_input_form", "repaired surface is not strike smiles");
    };

    for i in 1..n_pillars - 1 {
        let (t_out, ref smile_out) = pillars[i];
        let f_market = fwd(t_out);
        let gap_lo = t_out - pillars[i - 1].0;
        let gap_hi = pillars[i + 1].0 - t_out;
        let expiry = nearest(&expiry_by_t, t_out).map(|&(_, d)| d);

        let mut e2 = expiries.clone();
        let mut s2 = smiles.clone();
        e2.remove(i);
        s2.remove(i);
        let input = VolInput::StrikeSmiles {
            expiries: e2,
            smiles: s2,
            coordinate,
            day_count: dc,
        };
        let reduced = match VolSurface::from_input(&input, chain.as_of) {
            Ok(s) => s,
            Err(e) => {
                let detail = clean(&format!("{e}"));
                for id in ModelId::ALL {
                    rows.push(
                        key.row(id)
                            .s("reduced_surface_failed")
                            .s(&detail)
                            .u(i)
                            .fill(LOO_HEADER),
                    );
                }
                continue;
            }
        };

        for (mi, id) in ModelId::ALL.into_iter().enumerate() {
            let t_fit = Instant::now();
            let fitted = match guarded(|| fit_arm(id, &reduced, &fwd)) {
                Ok(r) => r,
                Err(p) => Err(p),
            };
            let fit_ms = t_fit.elapsed().as_secs_f64() * 1e3;
            match fitted {
                Err(e) => {
                    rows.push(
                        key.row(id)
                            .s("fit_failed")
                            .s(clean(&e))
                            .u(i)
                            .u(n_pillars)
                            .s(expiry.map(|d| d.to_string()).unwrap_or_default())
                            .f(t_out)
                            .fill(LOO_HEADER),
                    );
                }
                Ok(fit) => {
                    let (in_rmse, in_max) = match &full[mi] {
                        Some(ff) => {
                            let (r, m, _, _, _) = slice_errors(ff, smile_out, t_out, f_market);
                            (r, m)
                        }
                        None => (f64::NAN, f64::NAN),
                    };
                    let (loo_rmse, loo_max, loo_signed, loo_atm, n_eval) =
                        slice_errors(&fit, smile_out, t_out, f_market);
                    rows.push(
                        key.row(id)
                            .s("ok")
                            .s("")
                            .u(i)
                            .u(n_pillars)
                            .s(expiry.map(|d| d.to_string()).unwrap_or_default())
                            .f(t_out)
                            .f(f_market)
                            .u(n_eval)
                            .f(gap_lo)
                            .f(gap_hi)
                            .f(in_rmse)
                            .f(in_max)
                            .f(loo_rmse)
                            .f(loo_max)
                            .f(loo_signed)
                            .f(loo_atm)
                            .f(fit_ms)
                            .finish(LOO_HEADER),
                    );
                }
            }
        }
    }
    rows
}

fn run_loo(captures: &[Capture], out_dir: &Path) {
    let started = Instant::now();
    let done = std::sync::atomic::AtomicUsize::new(0);
    let total = captures.len();
    let mut groups: std::collections::BTreeMap<String, Vec<&Capture>> = Default::default();
    for c in captures {
        groups.entry(c.ticker.clone()).or_default().push(c);
    }
    let mut groups: Vec<Vec<&Capture>> = groups.into_values().collect();
    for g in &mut groups {
        g.sort_by(|a, b| (a.date, &a.time).cmp(&(b.date, &b.time)));
    }
    let all_rows: Vec<Vec<String>> = groups
        .par_iter()
        .map(|group| {
            let mut acc = Vec::new();
            for capture in group {
                acc.extend(process_loo(capture));
                let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                eprintln!(
                    "  loo {n}/{total} ({:.0}s elapsed)",
                    started.elapsed().as_secs_f64()
                );
            }
            acc
        })
        .collect();
    let rows: Vec<String> = all_rows.into_iter().flatten().collect();
    let mut text = String::with_capacity(LOO_HEADER.len() + 128 * rows.len());
    text.push_str(LOO_HEADER);
    text.push('\n');
    for row in &rows {
        text.push_str(row);
        text.push('\n');
    }
    std::fs::write(out_dir.join("loocv.csv"), text).expect("write loocv.csv");
    eprintln!("  wrote loocv.csv: {} rows", rows.len());
    let manifest = serde_json::json!({
        "panel": "loo",
        "generated_by": "study --panel loo",
        "protocol": "for each daily capture (the 1530 snapshot) and each INTERIOR \
                     repaired pillar, refit every arm on the remaining pillars and \
                     evaluate the refitted surface at the held-out slice's strikes \
                     against the repaired pillar vols (the same target the in-sample \
                     RMSE uses). First and last pillars are never held out \
                     (extrapolation, not interpolation). The held-out expiry's parity \
                     forward is not handed to the fit; strike-to-moneyness conversion \
                     uses each fit's own interpolation over the remaining pillar \
                     forwards, which is identical code across arms.",
        "dea": "chains are de-Americanized first, as in the daily/intraday panels",
        "captures": captures.len(),
        "rows": rows.len(),
        "elapsed_seconds": started.elapsed().as_secs_f64(),
    });
    std::fs::write(
        out_dir.join("loo_manifest.json"),
        serde_json::to_string_pretty(&manifest).expect("manifest serializes"),
    )
    .expect("write loo manifest");
    eprintln!("done in {:.1}s", started.elapsed().as_secs_f64());
}

// ── Fitted exponents only (`--panel gamma`) ─────────────────────────────
//
// The exponent cross-section. The ingest is identical to the other panels
// (same curve, same de-Americanization, same filters, same one-shot
// arbitrage repair) and nothing downstream of the fit is computed, so it
// runs on every name in the archive in minutes. Three fits per capture:
//
//   essvi_gfree  the joint global eSSVI with the backbone exponent free;
//   ssvi         SSVI as the study calibrates it, inside the sufficient
//                set gamma <= 1/2, eta (1 + |rho|) <= 2, so its exponent
//                can only sit at or below the boundary;
//   ssvi_free    the same SSVI re-optimised with the exponent free on
//                (0, 1) and eta free, warm-started from the constrained
//                optimum: where the data would take the exponent if the
//                guarantee were not imposed, and what imposing it costs.
//
// `rmse_bp` is the median over pillar slices of the slice RMSE in implied
// vol, the statistic fits.csv reports slice by slice, so the two files
// agree on the arms they share.

const GAMMA_HEADER: &str = "ticker,date,time,model,status,status_detail,n_pillars,n_quotes,gamma,eta,rho,rmse_bp,converged,fit_ms";

const GAMMA_MODELS: [&str; 3] = ["essvi_gfree", "ssvi", "ssvi_free"];

fn process_gamma(capture: &Capture) -> Vec<String> {
    let mut rows: Vec<String> = Vec::new();
    let base = |model: &str| -> Row {
        Row::new()
            .s(&capture.ticker)
            .s(capture.date.to_string())
            .s(&capture.time)
            .s(model)
    };

    macro_rules! bail {
        ($code:expr, $detail:expr) => {{
            let detail = clean(&$detail);
            for m in GAMMA_MODELS {
                rows.push(base(m).s($code).s(&detail).fill(GAMMA_HEADER));
            }
            return rows;
        }};
    }

    let text = match read_gz(&capture.path) {
        Ok(t) => t,
        Err(e) => bail!("read_failed", format!("{e}")),
    };
    let chain: OptionChain = match OptionChain::from_json(&text) {
        Ok(c) => c,
        Err(e) => bail!("chain_parse_failed", format!("{e}")),
    };
    if chain.spot.filter(|s| s.is_finite() && *s > 0.0).is_none() {
        bail!("no_spot", "no usable underlying price");
    }
    let (curve, _) = match load_curve(capture.date, chain.as_of) {
        Ok(c) => c,
        Err(e) => bail!("curve_failed", e),
    };
    let (chain, _) = match guarded(|| de_americanize_chain(&chain, &curve, &dea_config())) {
        Ok(Ok(x)) => x,
        Ok(Err(e)) => bail!("dea_failed", format!("{e}")),
        Err(p) => bail!("dea_failed", p),
    };
    let (raw_surface, report) =
        match implied_vol_surface_from_chain(&chain, &curve, &filter_config()) {
            Ok(x) => x,
            Err(e) => bail!("surface_failed", format!("{e}")),
        };
    let day_count = DayCountConvention::Act365;
    let forward_points: Vec<(f64, f64)> = report
        .forwards
        .iter()
        .map(|&(expiry, f)| (day_count.year_fraction(chain.as_of, expiry), f))
        .collect();
    let fwd = |t: f64| lerp_pairs(&forward_points, t);
    let (repaired, _) = match repair_arbitrage(&raw_surface, fwd) {
        Ok(x) => x,
        Err(e) => bail!("repair_failed", format!("{e}")),
    };
    let smiles = pillar_smiles(&repaired, &fwd);
    let n_pillars = smiles.len();
    let n_quotes: usize = smiles.iter().map(|(_, s)| s.len()).sum();

    // median over pillar slices of the slice RMSE, as fits.csv reports it
    let rmse_of = |fit: &Fitted| -> f64 {
        let mut v: Vec<f64> = smiles
            .iter()
            .map(|(t, smile)| slice_errors(fit, smile, *t, fwd(*t)).0)
            .filter(|x| x.is_finite())
            .collect();
        if v.is_empty() {
            return f64::NAN;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = v.len();
        if n % 2 == 1 {
            v[n / 2]
        } else {
            0.5 * (v[n / 2 - 1] + v[n / 2])
        }
    };
    let params = |fit: &Fitted, essvi: bool| -> (Option<f64>, Option<f64>, Option<f64>, bool) {
        match fit.slice_rows().first() {
            Some(r) if essvi => (r.essvi_gamma, r.essvi_eta, r.essvi_rho, r.converged),
            Some(r) => (r.ssvi_gamma, r.ssvi_eta, r.ssvi_rho, r.converged),
            None => (None, None, None, false),
        }
    };
    let ok_row = |name: &str, fit: &Fitted, essvi: bool, ms: f64| -> String {
        let (g, eta, rho, conv) = params(fit, essvi);
        base(name)
            .s("ok")
            .s("")
            .u(n_pillars)
            .u(n_quotes)
            .of(g)
            .of(eta)
            .of(rho)
            .f(rmse_of(fit))
            .b(conv)
            .f(ms)
            .finish(GAMMA_HEADER)
    };
    let fail_row = |name: &str, detail: &str| -> String {
        base(name).s("fit_failed").s(clean(detail)).fill(GAMMA_HEADER)
    };

    // 1. the joint global eSSVI, exponent free
    let t0 = Instant::now();
    match guarded(|| fit_arm(ModelId::EssviGfree, &repaired, &fwd)) {
        Ok(Ok(fit)) => rows.push(ok_row(
            "essvi_gfree",
            &fit,
            true,
            t0.elapsed().as_secs_f64() * 1e3,
        )),
        Ok(Err(e)) => rows.push(fail_row("essvi_gfree", &e)),
        Err(p) => rows.push(fail_row("essvi_gfree", &p)),
    }

    // 2. SSVI inside the sufficient set, then 3. its unconstrained refit,
    //    warm-started from the constrained optimum
    let t0 = Instant::now();
    match guarded(|| fit_arm(ModelId::Ssvi, &repaired, &fwd)) {
        Ok(Ok(fit)) => {
            rows.push(ok_row("ssvi", &fit, false, t0.elapsed().as_secs_f64() * 1e3));
            let start = match &fit {
                Fitted::Ssvi(f) => (f.ssvi.rho, f.ssvi.eta, f.ssvi.gamma),
                _ => (-0.5, 0.5, 0.25),
            };
            let t1 = Instant::now();
            match guarded(|| {
                SsviSurfaceFit::fit_unconstrained_from(&repaired, &fwd, start)
                    .map(Fitted::Ssvi)
                    .map_err(|e| e.to_string())
            }) {
                Ok(Ok(free)) => rows.push(ok_row(
                    "ssvi_free",
                    &free,
                    false,
                    t1.elapsed().as_secs_f64() * 1e3,
                )),
                Ok(Err(e)) => rows.push(fail_row("ssvi_free", &e)),
                Err(p) => rows.push(fail_row("ssvi_free", &p)),
            }
        }
        Ok(Err(e)) => {
            rows.push(fail_row("ssvi", &e));
            rows.push(fail_row("ssvi_free", "constrained fit failed"));
        }
        Err(p) => {
            rows.push(fail_row("ssvi", &p));
            rows.push(fail_row("ssvi_free", "constrained fit failed"));
        }
    }
    rows
}

fn run_gamma(captures: &[Capture], out_dir: &Path) {
    let started = Instant::now();
    let done = std::sync::atomic::AtomicUsize::new(0);
    let total = captures.len();
    let mut groups: std::collections::BTreeMap<String, Vec<&Capture>> = Default::default();
    for c in captures {
        groups.entry(c.ticker.clone()).or_default().push(c);
    }
    let groups: Vec<Vec<&Capture>> = groups.into_values().collect();
    let all: Vec<Vec<String>> = groups
        .par_iter()
        .map(|group| {
            let mut acc = Vec::new();
            for capture in group {
                acc.extend(process_gamma(capture));
                let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if n % 100 == 0 || n == total {
                    eprintln!(
                        "  gamma {n}/{total} ({:.0}s elapsed)",
                        started.elapsed().as_secs_f64()
                    );
                }
            }
            acc
        })
        .collect();
    let rows: Vec<String> = all.into_iter().flatten().collect();
    let mut text = String::with_capacity(GAMMA_HEADER.len() + 64 * rows.len());
    text.push_str(GAMMA_HEADER);
    text.push('\n');
    for r in &rows {
        text.push_str(r);
        text.push('\n');
    }
    std::fs::write(out_dir.join("gamma_ext.csv"), text).expect("write gamma_ext.csv");
    eprintln!("  wrote gamma_ext.csv: {} rows", rows.len());
    let mut tickers: Vec<&str> = captures.iter().map(|c| c.ticker.as_str()).collect();
    tickers.sort();
    tickers.dedup();
    let mut dates: Vec<String> = captures.iter().map(|c| c.date.to_string()).collect();
    dates.sort();
    dates.dedup();
    let manifest = serde_json::json!({
        "panel": "gamma",
        "generated_by": "study --panel gamma --tickers ...",
        "purpose": "the fitted power-law exponent over a wider cross-section of names: identical ingest to the daily and intraday panels (curve, de-Americanization, filters, one-shot repair), the fit and nothing downstream of it, on every intraday capture",
        "models": {
            "essvi_gfree": "joint global eSSVI (Mingone 2022) with the backbone exponent free; the intraday panel's own fit on the names the two files share",
            "ssvi": "SSVI as the study calibrates it, inside the Gatheral-Jacquier sufficient set gamma <= 1/2, eta (1 + |rho|) <= 2 (SsviSurfaceFit::fit)",
            "ssvi_free": "the same SSVI re-optimised with gamma free on (0, 1) and eta free, warm-started from the constrained optimum (SsviSurfaceFit::fit_unconstrained_from): a diagnostic of where the data takes the exponent without the guarantee, and of what imposing it costs in fit",
        },
        "rmse_bp": "median over pillar slices of the slice RMSE in implied vol (basis points), the statistic fits.csv reports slice by slice",
        "tickers": tickers,
        "sessions": dates.len(),
        "first_session": dates.first().cloned().unwrap_or_default(),
        "last_session": dates.last().cloned().unwrap_or_default(),
        "captures": captures.len(),
        "rows": rows.len(),
        "elapsed_seconds": started.elapsed().as_secs_f64(),
    });
    std::fs::write(
        out_dir.join("gamma_manifest.json"),
        serde_json::to_string_pretty(&manifest).expect("manifest serializes"),
    )
    .expect("write gamma manifest");
    eprintln!("done in {:.1}s", started.elapsed().as_secs_f64());
}
