# Calibrated Heston Bermudan Put Pricer

ENSF 620 Fall 2026 final project (individual). Owner: Jared Crebo.

`SPEC.md` holds the requirements and `DATA_CONTRACT.md` holds the data rules. Where they differ on data handling, the contract wins.

## Question

For a one-year Bermudan put on GOOG, what is the risk-neutral price implied by published Heston parameters for Google options, and how much of that price is the early-exercise premium relative to the corresponding European put?

Any headline value is a **model-implied, low-biased Monte Carlo estimate** for its stated data mode, snapshot date and assumptions. It is not a market price, a tradable quote or a trading signal. Results appear only in `results/`, each labelled with its data mode.

## Scope

| Part | Status |
|---|---|
| CORE: single-asset Heston pricer, calibration, LSMC, verification | Implemented |
| EXT-A: optional call or put on n correlated assets (worst-of or weighted basket), including the three-asset worst-of put | Not started; may start only after gate G1 passes |
| EXT-B: C++ kernel | Closed unless the runtime gate in SPEC.md 9.4 is met |

Out of scope: live data, hedging or backtesting, stochastic rates, jumps, discrete dividends in the dynamics, and duality upper bounds.

## Data boundary

Two licensed datasets (Massive minute bars for NVDA, AMD, INTC, GOOG and GOOGL, and iVolatility Raw IV for NVDA, AMD and INTC) stay on the owner's laptop under `data/`. They are never committed, submitted, shown to an AI tool, or written to a log.

Only aggregated results leave that boundary: fitted parameters, RMS fit residuals, correlations and prices as fractions of spot. No figure plots a market price, quote or implied-volatility point.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
ln -sf ../../scripts/check_no_data.sh .git/hooks/pre-commit
```

## Data modes and inputs

| Mode | Parameters come from | Who can run it |
|---|---|---|
| `synthetic` | `tests/fixtures/` (made-up parameters) | Anyone |
| `published` | `configs/published_params.yaml` (cited papers) | Anyone, once the owner has entered the values |
| `market` | `results/calibration.json` | Only where the licensed data is present, and only when authorized in `configs/governance.yaml` |

Inputs entered by hand: the Treasury yields in `configs/market_inputs.yaml`, the published parameters in `configs/published_params.yaml`, and the benchmarks in `tests/reference_values.yaml`.

`market` runs stop with `E_GOV_UNAUTHORIZED` unless `configs/governance.yaml` contains `market_use_authorized: true`, along with the `basis` for that authority and `recorded_on`.

## Commands

```bash
python python/make_fixtures.py                                   # regenerate synthetic fixtures
python python/validate_raw.py --source fixtures                  # check data against the contract
python scripts/inspect_schema.py <path-under-data>               # structure only, never values
python scripts/estimate_correlations.py                          # NVDA, AMD, INTC, GOOG correlations

python scripts/run.py calibrate --config configs/<name>.yaml
python scripts/run.py price     --config configs/<name>.yaml
python scripts/run.py sweep     --config configs/<name>.yaml
python scripts/run.py compare   --config configs/<name>.yaml
python scripts/compare_to_market.py                              # model against the published Feb 2015 Google puts

bash   scripts/smoke.sh                                          # reviewer smoke path
python scripts/verify_outputs.py                                 # run checks, write the table
python scripts/make_figures.py                                   # figures and evidence map
bash   scripts/check_no_data.sh                                  # data boundary check
```

Runs marked `reported: true` refuse a working tree with uncommitted changes.

Exit codes: `0` success, `1` internal error, `2` input or schema failure, `3` a check or gate failed, `4` governance refusal.

## Outputs

| Path | Contents |
|---|---|
| `results/runs/<run_id>/` | `config.yaml`, `manifest.json`, `result.json`, `run.log` |
| `results/calibration.json` | Fitted parameters and aggregate fit statistics |
| `results/market_comparison.json` | Model against the published Google puts of 2015-02-02 (in-sample) |
| `results/estimates.json` | Correlations and realized volatility for 126, 252 and 504-day windows |
| `results/sweeps/` | Convergence sweep summaries |
| `results/verification_table.csv` | PASS, FAIL or NOT_RUN for each check |
| `results/evidence_map.csv` | Links each figure and table to its source |
| `results/figures/` | Figures and tables |

All prices are fractions of initial spot.

## What the checks establish

The checks test the code on known answers: published benchmarks, closed forms, a binomial tree, an independent scalar implementation, and a synthetic surface with known parameters. A check that could not run is recorded as `NOT_RUN`, never as a pass.

They do **not** show that Heston is the right model for AMD, or that a model price matches what the option would trade at.

## Limitations

- LSMC prices are low-biased, because the exercise rule is only an estimate of the optimal one.
- The Euler scheme adds discretization bias, measured by the sub-step sweep.
- American quotes are converted to European-equivalent volatilities (de-Americanization), which is an approximation.
- The rate is flat; the term structure is ignored.
- Heston has no jumps and no earnings effects.
- Calibration uses a single snapshot date, with two neighbouring dates for stability only.
- Any step of the calibration failure ladder that was used is recorded in `results/calibration.json`.

## What a reviewer can and cannot reproduce

**Can reproduce**, from a fresh clone with no licensed data: `bash scripts/smoke.sh`, the whole pipeline on synthetic fixtures, and `published` mode.

**Cannot reproduce**: `market` mode, because the licensed data cannot be shared. For that result a reviewer can inspect the committed code, configs, run manifests, Internal outputs and evidence map.

## Repository layout

```text
configs/    run configurations and owner-entered inputs
python/     hestonpricer package, validate_raw.py, make_fixtures.py
scripts/    run.py, verify_outputs.py, make_figures.py, inspect_schema.py, smoke.sh, check_no_data.sh
tests/      verification checks, unit tests, reference implementation, fixtures
results/    Internal outputs, written only by the pipeline scripts
prototype/  earlier exploratory code, not part of the assessed workflow
data/       Git-ignored and Restricted; never in a commit or submission
```

AI use is recorded in `AI_USE.md`.
