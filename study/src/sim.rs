//! One local-volatility Monte Carlo, shared by every model.
//!
//! log-Euler on `dS/S = mu(t) dt + sigma_loc(S, t) dW`, antithetic
//! pairs, and the Brownian-bridge survival weight of
//! `rustyqlib::equity::montecarlo::bridge_survival` for the barriers.
//!
//! **The drift is the growth rate of the chain's own parity forward,
//! not the risk-free zero rate.** The library's Dupire implementation
//! (`smoothed_surface::dupire`) is written in forward log-moneyness
//! `k = ln(K/F(t))`, so the local vol it produces is only consistent
//! with dynamics whose drift is `d ln F(t)/dt`. Drifting at `r` with
//! zero dividends instead -- the build pipeline's convention -- makes
//! `E[S_T] = S e^{rT}` rather than `F_T`, which shows up as a
//! forward-recovery error equal to the implied dividend/borrow yield
//! and as a put/call parity error in the vanilla references. The
//! caller therefore supplies `carry(t) = ln(F(t)/S)`, and discounting
//! still comes from the curve.
//!
//! It lives in the driver rather than being reached through
//! `EquityOptionBuilder` for one reason: the builder's `Model::LocalVol`
//! route re-derives Dupire *numerically* from a sampled `VolSurface`,
//! which would fold the sampling resolution and the finite-difference
//! stencil into the model comparison. Here every model hands in its own
//! `SmoothedSurface::local_vol_checked` and nothing else changes -- same
//! seed, same normal draws, same time grid, same barrier monitoring,
//! same path count. That is what makes `barriers.csv` apples to apples.

use rand::distributions::Distribution;
use rand::SeedableRng;
use rustyqlib::core::curves::YieldCurve;

/// One barrier level, as a multiple of spot. `down` selects the
/// direction; every barrier here is continuously monitored and
/// knock-out (knock-in follows from in + out = vanilla, which the CSV
/// carries as the `none` rows).
#[derive(Debug, Clone, Copy)]
pub struct BarrierLevel {
    pub down: bool,
    pub mult: f64,
}

/// Simulation choices. Identical across models by construction: the
/// driver builds one of these and hands the same value to all three.
#[derive(Debug, Clone, Copy)]
pub struct SimConfig {
    /// Simulated paths (rounded down to an even count for antithetics).
    pub paths: usize,
    /// Euler steps per year.
    pub steps_per_year: usize,
    /// Floor on substeps between consecutive maturities, so a short
    /// first leg is not left coarser than the rest of the grid.
    pub min_substeps_per_leg: usize,
    pub seed: u64,
}

/// Mean and standard error of one Monte Carlo statistic.
#[derive(Debug, Clone, Copy, Default)]
pub struct Stat {
    pub mean: f64,
    pub std_err: f64,
}

/// Everything one path set produces, indexed `[maturity]` and
/// `[maturity][barrier]`.
#[derive(Debug, Clone)]
pub struct SimResult {
    pub paths: usize,
    pub steps: usize,
    /// `E[S_T]` -- the forward-recovery statistic, on the same paths the
    /// barriers were priced on.
    pub mean_spot: Vec<Stat>,
    /// Undiscounted-then-discounted vanilla reference prices.
    pub vanilla_call: Vec<Stat>,
    pub vanilla_put: Vec<Stat>,
    /// `[maturity][barrier]` knock-out call and put.
    pub ko_call: Vec<Vec<Stat>>,
    pub ko_put: Vec<Vec<Stat>>,
    /// Share of paths that had knocked out of each barrier by each
    /// maturity (1 - mean survival weight).
    pub knock_prob: Vec<Vec<f64>>,
}

/// Price the whole fixed instrument grid off one path set.
///
/// * `local_vol(level, t) -> sigma` and `carry(t) = ln(F(t) / spot)`
///   are the only model-dependent inputs; both come from the same
///   fitted surface, which is what keeps the drift and the local vol
///   consistent with each other.
/// * `maturities` must be increasing and positive.
/// * `call_strikes[m]` / `put_strikes[m]` strike the calls and puts at maturity `m`.
#[allow(clippy::too_many_arguments)]
pub fn simulate(
    local_vol: &dyn Fn(f64, f64) -> f64,
    carry: &dyn Fn(f64) -> f64,
    curve: &YieldCurve,
    spot: f64,
    call_strikes: &[f64],
    put_strikes: &[f64],
    maturities: &[f64],
    barriers: &[BarrierLevel],
    cfg: &SimConfig,
) -> SimResult {
    let pairs = (cfg.paths / 2).max(1);
    let n_mat = maturities.len();
    assert_eq!(call_strikes.len(), n_mat, "one call strike per maturity");
    assert_eq!(put_strikes.len(), n_mat, "one put strike per maturity");
    let n_bar = barriers.len();

    // ── time grid: the configured resolution, never coarser than the
    // maturity checkpoints themselves (same rule as the library's
    // martingale check) ────────────────────────────────────────────────
    let mut grid: Vec<f64> = vec![0.0];
    for &t in maturities {
        let previous = *grid.last().expect("grid starts at 0");
        let substeps = (((t - previous) * cfg.steps_per_year as f64).ceil() as usize)
            .max(cfg.min_substeps_per_leg)
            .max(1);
        for i in 1..=substeps {
            grid.push(previous + (t - previous) * i as f64 / substeps as f64);
        }
    }
    let checkpoints: Vec<usize> = maturities
        .iter()
        .map(|&t| grid.partition_point(|&g| g < t - 1e-12))
        .collect();
    let steps = grid.len() - 1;

    // exact forward growth per substep: the increment of ln F(t). This
    // telescopes, so the accumulated drift to any maturity is exactly
    // ln(F(T) / spot) and `E[S_T]` targets the parity forward the local
    // vol was built against.
    let drift: Vec<f64> = grid
        .windows(2)
        .map(|w| carry(w[1]) - carry(w[0]))
        .collect();

    let levels: Vec<f64> = barriers.iter().map(|b| spot * b.mult).collect();
    let dfs: Vec<f64> = maturities.iter().map(|&t| curve.df(t)).collect();

    // statistics laid out flat: per maturity, [S, call, put, then
    // (ko_call, ko_put) per barrier, then survival]
    let per_mat = 3 + 3 * n_bar;
    let n_stat = per_mat * n_mat;
    let mut sums = vec![0.0f64; n_stat];
    let mut sum_sq = vec![0.0f64; n_stat];

    let mut rng = rand_pcg::Pcg64::seed_from_u64(cfg.seed);
    let normal = rand_distr::StandardNormal;
    let ln_spot = spot.ln();

    let mut surv_up = vec![0.0f64; n_bar];
    let mut surv_dn = vec![0.0f64; n_bar];

    for _ in 0..pairs {
        let mut log_up = ln_spot;
        let mut log_dn = ln_spot;
        surv_up.iter_mut().for_each(|s| *s = 1.0);
        surv_dn.iter_mut().for_each(|s| *s = 1.0);
        let mut checkpoint = 0usize;

        for step in 0..steps {
            let (t0, t1) = (grid[step], grid[step + 1]);
            let dt = t1 - t0;
            let sqrt_dt = dt.sqrt();
            let shock: f64 = normal.sample(&mut rng);
            let t_eval = t0.max(1e-4);

            let s_up = log_up.exp();
            let s_dn = log_dn.exp();
            let vol_up = local_vol(s_up, t_eval).max(1e-8);
            let vol_dn = local_vol(s_dn, t_eval).max(1e-8);

            let next_up =
                log_up + drift[step] - 0.5 * vol_up * vol_up * dt + vol_up * sqrt_dt * shock;
            let next_dn =
                log_dn + drift[step] - 0.5 * vol_dn * vol_dn * dt - vol_dn * sqrt_dt * shock;
            let s_up_next = next_up.exp();
            let s_dn_next = next_dn.exp();

            for b in 0..n_bar {
                let h = levels[b];
                let down = barriers[b].down;
                bridge_step(&mut surv_up[b], s_up, s_up_next, h, down, vol_up, dt);
                bridge_step(&mut surv_dn[b], s_dn, s_dn_next, h, down, vol_dn, dt);
            }

            log_up = next_up;
            log_dn = next_dn;

            while checkpoint < checkpoints.len() && step + 1 == checkpoints[checkpoint] {
                let base = per_mat * checkpoint;
                let df = dfs[checkpoint];
                let (su, sd) = (s_up_next, s_dn_next);
                let (kc, kp) = (call_strikes[checkpoint], put_strikes[checkpoint]);
                let call = |s: f64| (s - kc).max(0.0);
                let put = |s: f64| (kp - s).max(0.0);

                accumulate(&mut sums, &mut sum_sq, base, 0.5 * (su + sd));
                accumulate(
                    &mut sums,
                    &mut sum_sq,
                    base + 1,
                    df * 0.5 * (call(su) + call(sd)),
                );
                accumulate(
                    &mut sums,
                    &mut sum_sq,
                    base + 2,
                    df * 0.5 * (put(su) + put(sd)),
                );
                for b in 0..n_bar {
                    let (wu, wd) = (surv_up[b], surv_dn[b]);
                    accumulate(
                        &mut sums,
                        &mut sum_sq,
                        base + 3 + 3 * b,
                        df * 0.5 * (wu * call(su) + wd * call(sd)),
                    );
                    accumulate(
                        &mut sums,
                        &mut sum_sq,
                        base + 4 + 3 * b,
                        df * 0.5 * (wu * put(su) + wd * put(sd)),
                    );
                    accumulate(&mut sums, &mut sum_sq, base + 5 + 3 * b, 0.5 * (wu + wd));
                }
                checkpoint += 1;
            }
        }
    }

    let n = pairs as f64;
    let stat = |i: usize| -> Stat {
        let mean = sums[i] / n;
        let variance = (sum_sq[i] / n - mean * mean).max(0.0);
        Stat {
            mean,
            std_err: (variance / n).sqrt(),
        }
    };

    let mut result = SimResult {
        paths: pairs * 2,
        steps,
        mean_spot: Vec::with_capacity(n_mat),
        vanilla_call: Vec::with_capacity(n_mat),
        vanilla_put: Vec::with_capacity(n_mat),
        ko_call: Vec::with_capacity(n_mat),
        ko_put: Vec::with_capacity(n_mat),
        knock_prob: Vec::with_capacity(n_mat),
    };
    for m in 0..n_mat {
        let base = per_mat * m;
        result.mean_spot.push(stat(base));
        result.vanilla_call.push(stat(base + 1));
        result.vanilla_put.push(stat(base + 2));
        result
            .ko_call
            .push((0..n_bar).map(|b| stat(base + 3 + 3 * b)).collect());
        result
            .ko_put
            .push((0..n_bar).map(|b| stat(base + 4 + 3 * b)).collect());
        result.knock_prob.push(
            (0..n_bar)
                .map(|b| 1.0 - stat(base + 5 + 3 * b).mean)
                .collect(),
        );
    }
    result
}

#[inline]
fn accumulate(sums: &mut [f64], sum_sq: &mut [f64], i: usize, x: f64) {
    sums[i] += x;
    sum_sq[i] += x * x;
}

/// One step of the Brownian-bridge survival weight against a single
/// continuously-monitored barrier -- crossing at the node kills the
/// path, otherwise the bridge crossing probability
/// `exp(-2ab / (sigma^2 dt))` discounts the survival. Transcribed from
/// `bridge_survival` in `rustyqlib::equity::montecarlo`.
#[inline]
fn bridge_step(survival: &mut f64, s_prev: f64, s_next: f64, h: f64, down: bool, sigma: f64, dt: f64) {
    if *survival <= 0.0 {
        return;
    }
    let crossed = if down { s_next <= h } else { s_next >= h };
    if crossed {
        *survival = 0.0;
        return;
    }
    let (a, b) = if down {
        ((s_prev / h).ln(), (s_next / h).ln())
    } else {
        ((h / s_prev).ln(), (h / s_next).ln())
    };
    let sigma = sigma.max(1e-8);
    *survival *= 1.0 - (-2.0 * a * b / (sigma * sigma * dt)).exp();
}
