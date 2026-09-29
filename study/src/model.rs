//! The three contenders behind one interface.
//!
//! Every parameterization is reached only through
//! [`rustyqlib::equity::smoothed_surface::SmoothedSurface`], which is the
//! library's own guarantee that Dupire's formula, the butterfly
//! function, the guards and the clamps are literally the same code for
//! all three. The only thing that differs downstream of this file is
//! `variance_derivatives` -- i.e. the parameterization itself, which is
//! exactly what the study is trying to measure.

use rustyqlib::core::vols::VolSurface;
use rustyqlib::equity::essvi::{EssviGlobalDiag, EssviSurfaceFit};
use rustyqlib::equity::smoothed_surface::{SmoothedSurface, VarianceDerivatives};
use rustyqlib::equity::svi::{SsviSurfaceFit, SviSurfaceFit};

/// Model identity, in the order rows are emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelId {
    Svi,
    Ssvi,
    Essvi,
    /// Global (Mingone 2022) eSSVI, power-law psi backbone, gamma = 1/2.
    EssviG05,
    /// Global (Mingone 2022) eSSVI, power-law psi backbone, gamma fitted.
    EssviGfree,
    /// Sequential greedy-tube eSSVI, power-law backbone, gamma = 1/2:
    /// the Hendriks-Martini guarantee at per-expiry speed.
    EssviSeqG05,
}

impl ModelId {
    pub const ALL: [ModelId; 6] = [
        ModelId::Svi,
        ModelId::Ssvi,
        ModelId::Essvi,
        ModelId::EssviG05,
        ModelId::EssviGfree,
        ModelId::EssviSeqG05,
    ];

    pub fn name(self) -> &'static str {
        match self {
            ModelId::Svi => "svi",
            ModelId::Ssvi => "ssvi",
            ModelId::Essvi => "essvi",
            ModelId::EssviG05 => "essvi_g05",
            ModelId::EssviGfree => "essvi_gfree",
            ModelId::EssviSeqG05 => "essvi_sg05",
        }
    }
}

/// A fitted surface, whichever parameterization produced it.
pub enum Fitted {
    Svi(SviSurfaceFit),
    Ssvi(SsviSurfaceFit),
    Essvi(EssviSurfaceFit),
    /// The Mingone-global fit reuses the sequential fit's surface type,
    /// so everything downstream (Dupire, sampling, validation) is the
    /// same code; the diagnostics ride along for fits.csv / timing.csv.
    EssviGlobal(EssviSurfaceFit, EssviGlobalDiag),
}

/// One fitted expiry slice, normalized across parameterizations so a
/// single `fits.csv` schema covers all three. Columns a model does not
/// have are `None` and are written empty.
#[derive(Debug, Clone, Default)]
pub struct SliceRow {
    pub t: f64,
    pub forward: f64,
    pub k_lo: f64,
    pub k_hi: f64,
    /// The library's own reported fit RMSE (implied vol) where it is a
    /// per-slice quantity; SSVI reports one global number on every row.
    pub lib_rmse: f64,
    pub converged: bool,
    /// Minimum of Gatheral's `g(k)` over the slice's quoted span, as
    /// reported by the fit (SSVI computes it here on the same span).
    pub min_g: f64,
    // raw SVI
    pub svi_a: Option<f64>,
    pub svi_b: Option<f64>,
    pub svi_rho: Option<f64>,
    pub svi_m: Option<f64>,
    pub svi_sigma: Option<f64>,
    // SSVI: three global shape parameters + this pillar's theta and the
    // power-law curvature phi(theta) evaluated there
    pub ssvi_rho: Option<f64>,
    pub ssvi_eta: Option<f64>,
    pub ssvi_gamma: Option<f64>,
    pub ssvi_theta: Option<f64>,
    pub ssvi_phi: Option<f64>,
    // eSSVI
    pub essvi_theta: Option<f64>,
    pub essvi_psi: Option<f64>,
    pub essvi_rho: Option<f64>,
    // eSSVI-global only: the power-law backbone level and exponent, and
    // how many of the fit's slices had their psi target clamped onto the
    // arbitrage-free tube (a per-fit number, repeated on each row the
    // way SSVI's global parameters are).
    pub essvi_eta: Option<f64>,
    pub essvi_gamma: Option<f64>,
    pub essvi_tube_clamped: Option<f64>,
}

impl Fitted {
    /// Expiries the fit is anchored on, in increasing time order.
    pub fn pillar_times(&self) -> Vec<f64> {
        match self {
            Fitted::Svi(f) => f.slices.iter().map(|s| s.t).collect(),
            Fitted::Ssvi(f) => f.forwards.iter().map(|&(t, _)| t).collect(),
            Fitted::Essvi(f) => f.slices.iter().map(|s| s.t).collect(),
            Fitted::EssviGlobal(f, _) => f.slices.iter().map(|s| s.t).collect(),
        }
    }

    /// Input expiries the fit refused (too few pillars for its own
    /// parameter count).
    pub fn skipped_slices(&self) -> usize {
        match self {
            Fitted::Svi(f) => f.skipped_slices,
            Fitted::Ssvi(f) => f.skipped_slices,
            Fitted::Essvi(f) => f.skipped_slices,
            Fitted::EssviGlobal(f, _) => f.skipped_slices,
        }
    }

    /// The fit's own calendar-tension figure, where the model reports
    /// one (SSVI is calendar-clean by construction from floored theta
    /// pillars, so it reports 0).
    pub fn fit_max_calendar_crossing(&self) -> f64 {
        match self {
            Fitted::Svi(f) => f.max_calendar_crossing,
            Fitted::Ssvi(_) => 0.0,
            Fitted::Essvi(f) => f.max_calendar_crossing,
            Fitted::EssviGlobal(f, _) => f.max_calendar_crossing,
        }
    }

    /// `true` when the fit passes the model's own admissibility check.
    pub fn self_validates(&self) -> bool {
        match self {
            Fitted::Svi(f) => f.slices.iter().all(|s| s.params.validate().is_ok()),
            Fitted::Ssvi(f) => f.ssvi.validate().is_ok(),
            Fitted::Essvi(f) => f.validate().is_ok(),
            Fitted::EssviGlobal(f, _) => f.validate().is_ok(),
        }
    }

    /// Sample into the canonical pricing surface (used by the
    /// finite-difference round trip, which is a `VolSurface` consumer).
    pub fn to_vol_surface(&self, samples: usize) -> Result<VolSurface, String> {
        match self {
            Fitted::Svi(f) => f.to_vol_surface(samples).map_err(|e| e.to_string()),
            Fitted::Ssvi(f) => f.to_vol_surface(samples).map_err(|e| e.to_string()),
            Fitted::Essvi(f) => f.to_vol_surface(samples).map_err(|e| e.to_string()),
            Fitted::EssviGlobal(f, _) => f.to_vol_surface(samples).map_err(|e| e.to_string()),
        }
    }

    /// One normalized row per fitted expiry slice.
    pub fn slice_rows(&self) -> Vec<SliceRow> {
        match self {
            Fitted::Svi(f) => f
                .slices
                .iter()
                .map(|s| SliceRow {
                    t: s.t,
                    forward: s.forward,
                    k_lo: s.k_range.0,
                    k_hi: s.k_range.1,
                    lib_rmse: s.rmse,
                    converged: s.converged,
                    min_g: s.min_g,
                    svi_a: Some(s.params.a),
                    svi_b: Some(s.params.b),
                    svi_rho: Some(s.params.rho),
                    svi_m: Some(s.params.m),
                    svi_sigma: Some(s.params.sigma),
                    ..SliceRow::default()
                })
                .collect(),
            Fitted::Ssvi(f) => f
                .forwards
                .iter()
                .zip(&f.k_ranges)
                .map(|(&(t, fwd), &(lo, hi))| {
                    let theta = f.ssvi.theta(t);
                    // g(k) on the same quoted span the per-expiry fits
                    // report it on, through the shared butterfly function
                    let min_g = (0..=200)
                        .map(|i| lo + (hi - lo) * i as f64 / 200.0)
                        .map(|k| f.butterfly_g_at(k, t))
                        .fold(f64::MAX, f64::min);
                    SliceRow {
                        t,
                        forward: fwd,
                        k_lo: lo,
                        k_hi: hi,
                        lib_rmse: f.rmse,
                        converged: f.converged,
                        min_g,
                        ssvi_rho: Some(f.ssvi.rho),
                        ssvi_eta: Some(f.ssvi.eta),
                        ssvi_gamma: Some(f.ssvi.gamma),
                        ssvi_theta: Some(theta),
                        ssvi_phi: Some(f.ssvi.phi(theta)),
                        ..SliceRow::default()
                    }
                })
                .collect(),
            Fitted::Essvi(f) => f
                .slices
                .iter()
                .map(|s| SliceRow {
                    t: s.t,
                    forward: s.forward,
                    k_lo: s.k_range.0,
                    k_hi: s.k_range.1,
                    lib_rmse: s.rmse,
                    converged: s.converged,
                    min_g: s.min_g,
                    essvi_theta: Some(s.params.theta),
                    essvi_psi: Some(s.params.psi),
                    essvi_rho: Some(s.params.rho),
                    ..SliceRow::default()
                })
                .collect(),
            Fitted::EssviGlobal(f, diag) => f
                .slices
                .iter()
                .map(|s| SliceRow {
                    t: s.t,
                    forward: s.forward,
                    k_lo: s.k_range.0,
                    k_hi: s.k_range.1,
                    lib_rmse: s.rmse,
                    converged: s.converged,
                    min_g: s.min_g,
                    essvi_theta: Some(s.params.theta),
                    essvi_psi: Some(s.params.psi),
                    essvi_rho: Some(s.params.rho),
                    essvi_eta: Some(diag.eta),
                    essvi_gamma: Some(diag.gamma),
                    essvi_tube_clamped: Some(diag.tube_clamped as f64),
                    ..SliceRow::default()
                })
                .collect(),
        }
    }
}

// Everything downstream of the fit -- implied vol, Dupire local vol, the
// guard flag, the butterfly function -- comes from the library's shared
// implementation, so the three models reach it through identical code.
impl SmoothedSurface for Fitted {
    fn forward(&self, t: f64) -> f64 {
        match self {
            Fitted::Svi(f) => SmoothedSurface::forward(f, t),
            Fitted::Ssvi(f) => SmoothedSurface::forward(f, t),
            Fitted::Essvi(f) => SmoothedSurface::forward(f, t),
            Fitted::EssviGlobal(f, _) => SmoothedSurface::forward(f, t),
        }
    }

    fn total_variance(&self, k: f64, t: f64) -> f64 {
        match self {
            Fitted::Svi(f) => SmoothedSurface::total_variance(f, k, t),
            Fitted::Ssvi(f) => SmoothedSurface::total_variance(f, k, t),
            Fitted::Essvi(f) => SmoothedSurface::total_variance(f, k, t),
            Fitted::EssviGlobal(f, _) => SmoothedSurface::total_variance(f, k, t),
        }
    }

    fn variance_derivatives(&self, k: f64, t: f64) -> VarianceDerivatives {
        match self {
            Fitted::Svi(f) => SmoothedSurface::variance_derivatives(f, k, t),
            Fitted::Ssvi(f) => SmoothedSurface::variance_derivatives(f, k, t),
            Fitted::Essvi(f) => SmoothedSurface::variance_derivatives(f, k, t),
            Fitted::EssviGlobal(f, _) => SmoothedSurface::variance_derivatives(f, k, t),
        }
    }
}
