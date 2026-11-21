# Heston LSMC Pricer: Out-of-Sample Accuracy on GOOG Options

ENSF 620 Fall 2026 final project (individual). Owner: Jared Crebo.

`SPEC.md` (v1.13) holds the requirements and `DATA_CONTRACT.md` (v0.9) holds the data rules. Where they differ on data handling, the contract wins.

## Question

How accurately does a Heston model price options contracts within its calibrated sample data and outside its calibrated sample data?

Within the sample means the contracts in the calibration grid; outside means held-out GOOG contracts that traded the same day (the test population below).

1. Calibrate the five Heston parameters to GOOG's option surface of 2026-10-13.
2. Price every contract in the test population as an American option with a Least-Squares Monte Carlo (LSMC) engine, with a Monte Carlo interval on each price.
3. Compare the model prices with the market's bid, ask and mid, and report aggregate accuracy with bootstrap intervals. The same test is run with published 2015 Google parameters as a comparison model.

Each model price is a **low-biased LSMC estimate** with a Monte Carlo standard error. It is not a market price, a tradable quote or a trading signal. A parameter set that fails the calibration acceptance rule is labelled **REJECTED** wherever its results appear, with its RMS implied-volatility residual and the 0.01 limit.

## Test population

GOOG options in the 2026-10-13 iVolatility file that:

- pass filters F1 to F8 (`SPEC.md` §6.2): 21 to 511 days to expiry, strike between 0.70 and 1.30 times the forward, out of the money, spread at most 50% of mid, open interest at least 10, and a valid implied-volatility root;
- traded that day (`volume > 0`);
- are not in the calibration grid.

Calibration and test use the same snapshot, so this is a test across contracts on one day, not over time. In-the-money contracts are excluded.

## Scope

In scope: one asset (GOOG), the Python/NumPy engine, puts and calls with daily exercise as an approximation to American exercise.

Out of scope: contracts on more than one asset, C++ kernels, live data, hedging or backtesting, stochastic rates, jumps, discrete dividends in the dynamics, and duality upper bounds.

## Data boundary

The licensed datasets stay on the owner's laptop under `data/`: iVolatility Raw IV and Massive minute bars. They are never committed, submitted, shown to an AI tool, or written to a log. The reported analysis reads only `data/goog_IV_Oct13.csv`; the Massive bars are used for raw-data validation only.

A committed script may read `data/` once the owner has reviewed it, provided everything it prints, logs or writes outside `data/` is aggregate. Per-contract model prices are written only to `data/derived/goog_test_prices.parquet`, which is Restricted. No figure plots a market price, quote, implied-volatility point or per-contract residual.

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
| `published` | `configs/published_params.yaml` (Burkovska et al., arXiv:1611.06452, Table 8) | Anyone |
| `market` | `results/calibration_goog/calibration.json` | Only where the licensed data is present, and only when authorized in `configs/governance.yaml` |

Inputs entered by hand:

- `configs/market_inputs_goog_oct13.yaml`: the Treasury yield and the GOOG dividend yield for 2026-10-13.
- `configs/published_params.yaml`: the published parameters.
- `tests/reference_values.yaml`: the benchmarks.

`market` runs stop with `E_GOV_UNAUTHORIZED` unless `configs/governance.yaml` contains `market_use_authorized: true`.

## Commands

```bash
python python/make_fixtures.py                                       # regenerate synthetic fixtures
python python/validate_raw.py --source fixtures                      # check fixtures against the contract
python scripts/inspect_schema.py <path-under-data>                   # structure only, never values

python scripts/run.py calibrate --config configs/calibrate_goog.yaml          # GOOG, 2026-10-13
python scripts/run.py price     --config configs/core_goog_reference.yaml     # reference contract
python scripts/run.py sweep     --config configs/core_goog_reference.yaml     # convergence sweeps
python scripts/run.py compare   --config configs/core_goog_reference.yaml     # numerical vs analytical
python scripts/market_test.py --config configs/market_test_goog_2026_10_13.yaml --workers 8
                                                                     # accuracy within and outside the calibrated sample (licensed data)
python scripts/run.py price --config configs/goog_2015_contract.yaml # the $525 Google put of 2015-02-02
python scripts/market_test.py --config configs/market_test_goog_2015.yaml --workers 5
                                                                     # engine check: the 401 published Feb 2015 Google puts
python scripts/run.py grid --config configs/grid_goog_out_of_range.yaml --workers 4
                                                                     # strikes and maturities outside the 2015 paper's range

bash   scripts/smoke.sh                                              # reviewer smoke path
python scripts/verify_outputs.py                                     # run checks, write the table
python scripts/make_figures.py                                       # figures and evidence map
bash   scripts/check_no_data.sh                                      # data boundary check
```

Each analysis is described by a config, not by its own script. To compare another asset or date with the market, copy a `configs/market_test_*.yaml` and change:
- `source`, with either `table` (a published price table) or `raw_file` (a licensed iVolatility file);
- `symbol` and `date`;
- `market_inputs`;
- `parameter_sets`;
- `outputs`.

To price another strike × maturity grid, copy `configs/grid_goog_out_of_range.yaml` and edit `grid.cases`.

Runs marked `reported: true` refuse a working tree with uncommitted changes. To try a contract without committing, set `reported: false` and pass `--results-root <folder>`.

Exit codes: `0` success, `1` internal error, `2` input or schema failure, `3` a check or gate failed, `4` governance refusal.

## Pricing your own contract

These steps price one American put or call of your choice, using a Heston model calibrated to one day of the licensed iVolatility data in `data/`. They are exploratory: the runs are not reported, and their output is kept outside `results/`.

### Inputs

There are two Raw IV files. The minute bars in `data/` are for validation only and must not be used for pricing.

| File | Tickers | Dates |
|---|---|---|
| `data/goog_IV_Oct13.csv` | GOOG | 2026-10-13 |
| `data/nvda_amd_intc_IV_Oct13-15.csv` | NVDA, AMD, INTC | 2026-10-13 to 2026-10-15 |

You can check a file's structure with `python scripts/inspect_schema.py <path>`. It prints structure only, never values.

### Steps

**1. Check the market inputs.** The `market_inputs` file must have the Treasury 1-year yield for the date and a dividend yield for the ticker:
- `configs/market_inputs_goog_oct13.yaml` covers GOOG on 2026-10-13;
- `configs/market_inputs.yaml` covers NVDA, AMD and INTC on 2026-10-13 to 2026-10-15.

The run stops with `E_CONFIG` if the yield is missing. The governance gate in `configs/governance.yaml` must also be open (`market_use_authorized: true`).

**2. Write a config.** Copy this, for example to `configs/explore_nvda_2026-10-13.yaml`, and change the asset, date, file and contract:

```yaml
schema_version: 1
run_name: explore_nvda_2026-10-13
reported: false
data_mode: market
assets: [NVDA]                                   # one ticker that is in the raw file
snapshot_date: 2026-10-13                        # a date that is in the raw file
market_inputs: configs/market_inputs.yaml
params_source: results/calibration.json         # with --results-root only the file name is used
raw_iv_file: data/nvda_amd_intc_IV_Oct13-15.csv
contract: {strike: 1.05, maturity: 0.25, n_exercise: 63, kind: put}
simulation: {n_paths: 100000, n_substeps: 4, basis_degree: 3}
seeds: [620001, 620002, 620003, 620004, 620005]
```

The contract is given in the model's units:

| Key | Meaning | Example |
|---|---|---|
| `strike` | Strike divided by the stock price on `snapshot_date` | a $230 strike with the stock at $219 is 230 / 219 ≈ 1.05 |
| `maturity` | Time to expiry in years, as calendar days / 365 (the calibration uses the same convention) | 91 days is 0.25 |
| `n_exercise` | Number of exercise dates. About `ceil(252 × maturity)` approximates daily American exercise; `1` gives the European price | 63 |
| `kind` | `put` or `call` | `put` |

Any key not listed in the example is rejected with `E_CONFIG`.

**3. Calibrate.**

```bash
python scripts/run.py calibrate --config configs/explore_nvda_2026-10-13.yaml --results-root explore/nvda_2026-10-13
```

Always pass `--results-root` for an exploratory calibration. Without it, `calibrate` writes the Restricted calibration grid to `data/derived/calibration_grid.parquet`. That would overwrite the grid behind the reported GOOG results.

The command prints the ladder step and the RMS implied-volatility error:

| Ladder step | Meaning | Can you price? |
|---|---|---|
| 1 or 2 | Accepted: RMS ≤ 0.01 and stable | Yes |
| 3 | No fit accepted. The lowest-RMS fit is used | Yes, but every result is labelled **REJECTED** |
| 4 | Nothing usable | No |

Also look at `params` in `explore/.../calibration.json`. A parameter sitting on its bound (for example `kappa` at 20) means the day's surface did not pin it down.

**4. Price.** Use the same `--results-root`, so `price` reads the `calibration.json` that step 3 wrote:

```bash
python scripts/run.py price --config configs/explore_nvda_2026-10-13.yaml --results-root explore/nvda_2026-10-13
```

`result.json` in the run folder holds:
- the American (Bermudan) price, its standard error and 95% interval;
- the Monte Carlo European price;
- the early-exercise premium.

**5. Convert to dollars.** Prices are fractions of the stock price, so multiply by the stock price on `snapshot_date`. For example, 0.062 × $219 ≈ $13.58 per share, or about $1,358 for a 100-share contract.

**6. Other dates and tickers.** Copy the config and change `run_name`, `assets`, `snapshot_date`, `raw_iv_file` and `market_inputs` as needed. Recompute `strike` from that day's stock price, and use a new `--results-root` for each date.

### Limits

- **What the number is:** a low-biased LSMC estimate with a Monte Carlo error. It is not a market price, a quote or a trading signal.
- **Comparing with the market:** to compare model prices with the market's bid and ask, write a `configs/market_test_*.yaml` and run `scripts/market_test.py` (see Commands). Do not compare them by hand: per-contract market values may only be written under `data/derived/` (`DATA_CONTRACT.md`).
- **Committing:** do not commit `explore/` output, or configs that name licensed rows or values.

## Outputs

| Path | Contents |
|---|---|
| `results/goog_market_test.json` | Out-of-sample accuracy by parameter set, with breakdowns and intervals (aggregates only) |
| `results/calibration_goog/calibration.json` | The 2026-10-13 fit, its ladder step, and aggregate fit statistics |
| `results/runs/<run_id>/` | `config.yaml`, `manifest.json`, `result.json`, `run.log` |
| `results/market_comparison.json` | Engine check against the published Google puts of 2015-02-02 (in-sample) |
| `results/sweeps/` | Convergence sweep summaries |
| `results/verification_table.csv` | PASS, FAIL or NOT_RUN for each check |
| `results/evidence_map.csv` | Links each figure and table to its source |
| `results/figures/` | Figures and tables |

All model and market quantities are fractions of the stock price. The only absolute prices are the published 2015 Google prices.

## What the checks establish

The checks test the code on known answers: published benchmarks, closed forms, a binomial tree, an independent scalar implementation, and a synthetic surface with known parameters. A check that could not run is recorded as `NOT_RUN`, never as a pass.

They do **not** show that Heston is the right model for GOOG. The out-of-sample test measures how well it prices held-out contracts on one day; it has no pass rule.

## Limitations

- **Same-day test:** calibration and test use the same snapshot, and test contracts sit next to grid points, so accuracy is optimistic.
- **LSMC low bias:** prices are low-biased, because the exercise rule is only an estimate of the optimal one.
- **Discretization:** the Euler scheme adds bias, measured by the sub-step sweep.
- **Exercise approximation:** daily exercise approximates American exercise.
- **De-Americanization:** calibration targets are American quotes converted to European-equivalent volatilities, which is an approximation.
- **Rate and dividend:** the rate is flat, ignoring the term structure, and the dividend yield is approximate.
- **Model limits:** Heston has no jumps and no earnings effects.
- **Exclusions:** in-the-money contracts are not tested.
- **Calibration status:** the ladder step used is recorded in `results/calibration_goog/calibration.json`.

## What a reviewer can and cannot reproduce

**Can reproduce**, from a fresh clone with no licensed data: `bash scripts/smoke.sh`, the whole pipeline on synthetic fixtures, and `published` mode.

**Cannot reproduce**: `market` mode and the out-of-sample test, because the licensed data cannot be shared. For those results a reviewer can inspect the committed code, configs, run manifests, Internal outputs and evidence map.

## Repository layout

```text
configs/    run configurations and owner-entered inputs
python/     hestonpricer package, validate_raw.py, make_fixtures.py
scripts/    run.py, market_test.py, verify_outputs.py, make_figures.py, inspect_schema.py, smoke.sh, check_no_data.sh
tests/      verification checks, unit tests, reference implementation, fixtures
results/    Internal outputs, written only by the pipeline scripts
prototype/  earlier exploratory code, not part of the assessed workflow
data/       Git-ignored and Restricted; never in a commit or submission
```

AI use is recorded in `AI_USE.md`.
