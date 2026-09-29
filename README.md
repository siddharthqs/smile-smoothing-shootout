# smile-smoothing-shootout

Replication material for **Fit Is Not Enough: An Empirical Study of eSSVI
Calibrations Through the Local Volatility They Induce** (Siddharth Singh, 2026).
The paper is [`paper.pdf`](paper.pdf).

Six calibrations of one equity volatility surface are compared on the same
quotes: S-eSSVI (greedy forward pass), two joint global eSSVI arms
(Mingone 2022, exponent pinned at 1/2 or free), sequential eSSVI with
penalties, SSVI, and raw SVI as a benchmark. Every arm is fitted to the same
de-Americanized, arbitrage-repaired pillars, scored in and out of sample and
on a full-grid static-arbitrage scan, pushed through one shared Dupire
construction, and used to price six-month vanillas in closed form and
knock-out barriers by Monte Carlo under the induced local volatility.

Every number in the paper is computed by the scripts here from the CSVs
here; none is transcribed.

## Contents

| Path | Contents |
|---|---|
| `paper.pdf` | the paper |
| `study/` | Rust driver: ingest, repair, the six calibrations, arbitrage scans, local volatility, martingale gate, barrier pricing; writes every CSV |
| `make_tables.py` | builds the paper's tables (`tables_*.tex`) from the CSVs |
| `figures/` | one script per figure, shared helpers, and `make_figures.py` to run them all |
| `data/raw/` | the 190 daily option-chain captures: eight study names and the two extension names (JPM, XOM), 19 sessions each |
| `data/curves/` | one US Treasury par-yield curve per session |
| `results/` | the daily result CSVs for the eight study names |
| `results/extension/` | the same for JPM and XOM |

## Reproducing the results

**Requirements.** Rust (built with 1.91), and Python 3 with `pandas`,
`numpy`, `scipy` and `matplotlib`.

**1. The library.** `study/Cargo.toml` pins
[RustyQLib](https://github.com/siddharthqs/RustyQLib) to commit
`ee2103d3131b90d242520cb51b4b57fd5e260395`; Cargo fetches it on the first
build. Every CSV except the wall-clock columns (`elapsed_ms`, `cold_ms`,
`warm_ms`, `fit_ms`) reproduces bit for bit at that commit.

**2. The driver.** From `study/`:

```bash
# daily panel, out-of-the-money barrier design -> results/*.csv
cargo run --release -- --panel daily --strike forward --strike-sd 0.5 --barrier-sd 0,0.5,1.0 --paths 400000

# at-the-money barrier design; only barriers.csv differs from the run above
cargo run --release -- --panel daily --strike forward --barrier-sd 0,0.5,1.0 --paths 400000 --out ../results/forward
cp ../results/forward/barriers.csv ../results/barriers_forward.csv

# leave-one-expiry-out -> results/loocv.csv
cargo run --release -- --panel loo

# the JPM/XOM extension
cargo run --release -- --panel daily --strike forward --strike-sd 0.5 --barrier-sd 0,0.5,1.0 --paths 400000 --tickers JPM,XOM --out ../results/extension
cargo run --release -- --panel loo --tickers JPM,XOM --out ../results/extension
```

The commands write into `results/` in place; pass `--out` to a fresh folder
to compare a rerun against the released CSVs. On one laptop the daily panel
takes about 45 minutes, most of it the 400,000-path barrier stage, and the
leave-one-out run about 25. Each run also writes a `run_manifest.json`
recording its settings.

**3. Tables and figures.** From the repository root:

```bash
python make_tables.py
```

```bash
python figures/make_figures.py
```

They read `results/` directly and write `tables_*.tex` beside
`make_tables.py` and the figure PDFs into `figures/`.

## Settings

| Stage | Setting |
|---|---|
| Sessions | 2026-08-14 to 2026-09-11, nineteen sessions (no capture on 2026-08-28; 2026-09-07 was a market holiday) |
| Daily capture | the capture nearest 15:30 each session; the `time` column records the one used |
| Quote filters | OTM only; relative spread at most 25%; moneyness K/F in [0.6, 1.6]; 1 day to 1.5 years to expiry; at least 5 quotes per expiry; implied vol in [0.005, 5] |
| Forwards and rates | parity forward, the median over the 5 put–call pairs nearest the money; discounting from the session's Treasury curve |
| De-Americanization | CRR early-exercise premium removed from each mid, 201 steps, carry implied from the chain's own forwards, iterated twice (`dea.csv`) |
| Repair | one minimal-change static-arbitrage repair; every arm fits the same repaired pillars (`pillars.csv`) |
| Sequential eSSVI penalties | butterfly 10, calendar 1000 |
| Scan and local-vol grid | 81 points in k over [-0.4, 0.4] by 41 in t from the first to the last fitted pillar; local vol clamped to [0.01, 3.0] |
| Round trip | Crank–Nicolson local-vol repricing, 400 × 400 grid, at k = -0.1, -0.05, 0, 0.05, 0.1 |
| Martingale gate | log-Euler with antithetic pairs, 32,768 paths, 104 steps a year, seed 20260824, drift from the parity forwards |
| Barriers | continuously monitored knock-outs, Brownian-bridge corrected, 400,000 paths, 252 steps a year (at least 64 per leg), seed 20260824, one path set shared by every arm; vanillas in closed form at each arm's fitted vol |

The barrier files hold one-, three- and six-month contracts; the paper uses
the six-month ones. In `barriers.csv` the strike is half a standard
deviation out of the money, K = F exp(±½σ√T); in `barriers_forward.csv` it
is the forward. In both the barrier sits one standard deviation away,
H = F exp(±σ√T), with σ the market at-the-money vol.

## Result files

| File | One row per | Holds |
|---|---|---|
| `panel.csv` | capture | status, spot, quote counts through each cleaning step, and the repair |
| `dea.csv` | capture × expiry | the de-Americanization outcome |
| `pillars.csv` | capture × expiry × strike | the repaired pillar vols every arm is fitted to |
| `fits.csv` | capture × arm × expiry | fitted parameters and the slice RMSE |
| `arbitrage.csv` | capture × arm | butterfly and calendar checks at the pillars and on the full grid |
| `localvol.csv` | capture × arm | local-vol roughness, clamp activity and round-trip error |
| `martingale.csv` | capture × arm × horizon | the martingale gate at one, three and six months |
| `barriers.csv`, `barriers_forward.csv` | capture × arm × contract | vanilla and knock-out prices for the two designs |
| `loocv.csv` | capture × arm × held-out expiry | leave-one-expiry-out errors and the ATM bias |
| `timing.csv` | capture × arm | calibration wall clocks and iteration counts |

## Not included

The **intraday panel**, every quarter-hour capture rather than one a
session, is not distributed: its captures run to some 470 MB and would
amount to bulk redistribution of vendor quotes. Without its results,
`make_tables.py` prints the intraday entries as `--` and skips the two
fitted-exponent tables, and `make_figures.py` skips the two figures drawn
from it (the exponent through the session and the calibration cost). Every
daily number, including every barrier result, regenerates from this
repository. The intraday captures and results are available from the
author on request: siddharth_qs@outlook.com.

## Data and licence

The captures in `data/raw/` are snapshots of Cboe's public delayed option
feed, collected for this study and included for replication only; they
remain subject to Cboe's terms. Each is a gzipped JSON object with the
symbol, spot, timestamp and a list of quotes (expiry, strike, right, bid,
ask, last, volume, open interest). The code is MIT licensed (see
[`LICENSE`](LICENSE)); the licence covers the code, not the data.

## Citing

Singh, S. (2026). *Fit Is Not Enough: An Empirical Study of eSSVI
Calibrations Through the Local Volatility They Induce.*
