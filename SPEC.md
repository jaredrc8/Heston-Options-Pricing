# SPEC.md — Calibrated Heston Bermudan Put Pricer

| | |
|---|---|
| **Project** | ENSF 620 Fall 2026 final project (individual) |
| **Owner** | Jared Crebo |
| **Spec version** | 1.2 (2026-11-08) |
| **Companion document** | `DATA_CONTRACT.md` v0.3 |
| **Audience** | Engineers and AI coding agents writing implementation code and automated tests |

The key words MUST, MUST NOT, SHOULD, and MAY are normative. Requirements are binding until changed through §1.3.

---

## 1. Purpose, status, and precedence

### 1.1 What this document is

This spec defines what must be true of the pricing system and how each requirement is verified. It carries project decisions across chat sessions. It records requirements, not implementation status: nothing here asserts that any component already exists or passes.

### 1.2 Precedence

When documents conflict, the higher item wins:

2. `DATA_CONTRACT.md`, for schemas, data classification, allowed paths, and what may appear in prompts, logs, and submissions.
3. This spec, for everything else.
4. The September 29 proposal. §20 lists where it has been superseded.

### 1.3 Change control

- Pass rules, tolerances, filters, and defaults in this spec are frozen at v1.0.
- A change requires a new row in §22 stating the date, the reason, and whether the affected main result had already been viewed.
- A change made after viewing an affected main result MUST be disclosed in the report.
- No person or agent may change a pass rule, tolerance, seed list, reference value, or data filter in order to make a failing check pass.
- A failed check is a recorded result. It is resolved by fixing a defect or applying §17, never by loosening the check.

---

## 2. Scope

### 2.1 Required core (CORE)

**Primary question.** For a one-year Bermudan put on AMD, what is the risk-neutral price implied by Heston parameters calibrated to AMD's listed volatility surface, and how much of that price is the early-exercise premium relative to the corresponding European put?

CORE is the committed result. Its required components are:

- Heston semi-analytic European pricer.
- Quote filtering, implied-volatility conversion, and single-asset calibration.
- Python/NumPy Monte Carlo simulator.
- Two-pass Least-Squares Monte Carlo (LSMC) pricer.
- Verification suite, run provenance, and reviewer smoke path.

CORE MUST also run with published parameters and synthetic fixtures so that it does not depend on licensed data (§5.1).

### 2.2 Conditional extensions

| ID | Extension | May start only when | If not completed |
|---|---|---|---|
| EXT-A | Optional n-asset correlated extension: a Bermudan call or put on `d >= 2` correlated assets, with a worst-of or weighted-basket payoff (§4.1). It includes the three-asset worst-of put on NVDA, AMD, and INTC | Gate G1 passes (§17.1) | Omit it from the claims. The final question stays as stated in §2.1. |
| EXT-B | C++ pricing kernel exposed through pybind11 | The C++ gate passes (§9.4) | The Python engine remains the production engine. |

The report, slides, and README MUST NOT present an unfinished extension as a result.

### 2.3 Removed or out of scope

The system MUST NOT implement these within this project:

- Live data feeds or re-downloading data.
- P&L simulation, delta hedging, backtesting, trading signals, or paper trading.
- Stochastic rates, jumps, or discrete dividends in the simulated dynamics.
- Duality-based upper bounds.

### 2.4 Permitted claims

The headline value is a **model-implied, low-biased Monte Carlo estimate** for the stated data mode, snapshot date, and assumptions. It MUST NOT be described as a market price, tradable quote, or trading signal.

---

## 3. Conventions

| Item | Convention |
|---|---|
| Floating point | IEEE-754 `float64` throughout |
| Spot normalization | Pricing uses S(0) = 1. All reported values are fractions of initial spot. |
| Time | Years, ACT/365 from snapshot date to expiration |
| Rates and yields | Decimal, continuously compounded |
| Volatility and variance | σ is annualized decimal volatility; ν is instantaneous variance |
| Standard error | `SE = s / sqrt(N)`, where `s` is the sample standard deviation with `ddof=1` over `N` independent evaluation paths |
| Paired SE | SE of the path-by-path difference on the same evaluation paths |
| Pooled result over R seeds | `mean(V_r)`, with `SE = sqrt(sum(SE_r^2)) / R` |
| Absolute prices | Absolute market prices MUST NOT appear in `results/`, logs, figures, or submitted files (§15) |

---

## 4. Contract and model

### 4.1 Contract

For `d` assets, the payoff at exercise date `t_k` is:

```text
g(t_k) = max( K - min_i[ S_i(t_k) / S_i(0) ], 0 )
```

CORE is the case `d = 1`.

| Term | Default | Rule |
|---|---|---|
| Normalized strike `K` | `1.00` | `K > 0` |
| Maturity `T` | `1.00` year | `T > 0` |
| Exercise dates `M` | `50` | Integer `M >= 1` |
| Exercise grid | `t_k = k T / M`, `k = 1..M` | No exercise at `t = 0`; `t_M = T` |

The corresponding European contract uses the same `K` and `T` and may be exercised only at `T`.

**n-asset payoffs (EXT-A, optional).** For `d >= 2` correlated assets, let `P_i(t) = S_i(t) / S_i(0)`, and define the underlying value `U(t)` by the payoff type:

| Payoff type | `U(t)` | Rule |
|---|---|---|
| `worst_of` | `min_i P_i(t)` | |
| `basket` | `sum_i w_i P_i(t)` | `w_i >= 0`, `sum_i w_i = 1`; equal weights by default |

A put pays `max(K - U(t), 0)` and a call pays `max(U(t) - K, 0)`. The payoff stated above is the `worst_of` put, and CORE is unchanged by these definitions.

### 4.2 Risk-neutral dynamics

```text
dS_i = (r - q_i) S_i dt + sqrt(nu_i) S_i dW_i^S
dnu_i = kappa_i (theta_i - nu_i) dt + xi_i sqrt(nu_i) dW_i^nu
d<W_i^S, W_i^nu> = rho_i dt
d<W_i^S, W_j^S> = R_ij dt
```

Parameter vector for each asset:

```text
phi_i = (nu_i0, kappa_i, theta_i, xi_i, rho_i)
```

`nu_i0` is **fitted**, not fixed. `r` and `q_i` are constants.

### 4.3 Rate and dividend inputs

These values live in `configs/market_inputs.yaml`.

- **Rate.** Let `y` be the U.S. Treasury 1-Year CMT par yield on the snapshot date as a decimal. Then `r = 2 ln(1 + y / 2)`.
- **Term structure.** The same flat `r` is used for implied-volatility conversion, calibration, and pricing. Ignoring the term structure is a stated limitation.
- **CORE dividend yield.** `q_AMD = 0`.
- **EXT-A dividend yields.** `q_AMD = 0`. `q_INTC = 0` unless the owner records otherwise. `q_NVDA` is the annualized declared dividend divided by the snapshot spot, rounded to the nearest `0.0001` before any use.
- **Discrete dividends.** They are approximated by this continuous yield in both the tree and the simulation.

### 4.4 Cross-asset shock structure (EXT-A)

Let `Z^S` have correlation matrix `R`. Let `Z^perp` be standard normals independent of `Z^S` and independent across assets. Then:

```text
Z_i^nu = rho_i Z_i^S + sqrt(1 - rho_i^2) Z_i^perp
```

This fully determines the 2d × 2d driver correlation matrix:

| Pair | Correlation |
|---|---|
| `corr(Z_i^S, Z_j^S)` | `R_ij` |
| `corr(Z_i^S, Z_i^nu)` | `rho_i` |
| `corr(Z_i^S, Z_j^nu)` for `i != j` | `rho_j R_ij` |
| `corr(Z_i^nu, Z_j^nu)` for `i != j` | `rho_i rho_j R_ij` |

`R` is factored by symmetric eigendecomposition, so a singular positive semi-definite `R` is valid input.

---

## 5. Input data and modes

### 5.1 Data modes

Every run MUST declare one mode.

| Mode | Parameter source | Purpose | Licensed data read |
|---|---|---|---|
| `synthetic` | Known parameters and `tests/fixtures/` | Tests, CI, reviewer smoke path | None |
| `published` | `configs/published_params.yaml` | Fallback result and known-answer checks | None |
| `market` | `results/calibration.json` | Market-calibrated result | Read by the committed scripts only (§15) |

Each published parameter set MUST include its citation and table or page. Every reported artifact MUST name its mode. A `published` result MUST NOT be described as calibrated to AMD.

### 5.2 Authority gate for market mode

`configs/governance.yaml` contains:

```yaml
market_use_authorized: false
basis: ""
recorded_on: null
```

- `market` runs MUST fail with `E_GOV_UNAUTHORIZED` while `market_use_authorized` is `false`.
- It may be set to `true` only by the owner, or by an agent on the owner's recorded instruction. `basis` must then state where the authority is documented.
- While it stays `false`, or if it cannot be settled by G1, the `published` result is the reported result (§19, OD-1).

### 5.3 Data sources

Schemas, classes, and paths are defined by `DATA_CONTRACT.md`; this spec does not duplicate them.

| Dataset | Used by | Notes |
|---|---|---|
| iVolatility Raw IV, 2026-10-13 to 2026-10-15 | CORE calibration; EXT-A calibration | Restricted. Primary snapshot date is `2026-10-14`. |
| Massive minute bars, November 2011 to July 2026 | EXT-A correlations only | Restricted. **CORE MUST NOT read Massive data.** |
| U.S. Treasury CMT yields | `r` | Public |
| Published parameters and benchmarks | Fallback and checks | Public; transcribed by the owner |
| Synthetic fixtures | Tests, CI, smoke path | Public |

The other two iVolatility dates are for day-to-day stability reporting only. The primary date MUST NOT be changed after viewing fit quality.

### 5.4 Correlation estimation (EXT-A)

1. Take split-adjusted last regular-session closes and compute daily log returns.
2. Inner-join tickers on date.
3. Estimate the Pearson correlation over the trailing 252 trading days ending at the last available date.
4. Require a symmetric matrix with unit diagonal, entries in `[-1, 1]`, and minimum eigenvalue `>= -1e-10`. Otherwise raise `E_CORR_NOT_PSD`.
5. Report sensitivity using 126-day and 504-day windows.

The report MUST state two limitations: the correlation window ends before the option snapshot, and historical correlation is being used inside a risk-neutral model.

---

## 6. Calibration

### 6.1 Pipeline

```text
validate_raw -> filter quotes -> convert to implied vol -> select grid
             -> multi-start least squares -> stability check -> calibration.json
```

### 6.2 Quote filters

Filters are applied in this order, and each drop count is recorded. `S` is `stock_price_for_iv`, `mid = (bid + ask) / 2`, and `F = S exp((r - q) T)`.

| ID | Keep a row only if |
|---|---|
| F1 | It matches the requested symbol and snapshot date |
| F2 | `iv` is valid under `DATA_CONTRACT.md`, `bid > 0`, and `ask >= bid` |
| F3 | Days to expiry are in `[21, 511]` |
| F4 | `K / F` is in `[0.70, 1.30]` |
| F5 | It is out of the money: calls with `K >= F`, puts with `K < F` |
| F6 | `(ask - bid) / mid <= 0.50` |
| F7 | `open_interest >= 10` |
| F8 | `mid` exceeds intrinsic value and an implied-volatility root exists in `[0.01, 5.00]` |

### 6.3 Implied-volatility conversion

- Style `A` quotes are de-Americanized. Invert a Cox–Ross–Rubinstein (CRR) American price with 500 steps for `sigma`, using Brent's method with `xtol = 1e-6`. That `sigma` is taken as the European-equivalent implied volatility.
- Style `E` quotes are inverted with Black–Scholes.
- The vendor `iv` field is never a calibration target. It is used only in V-14.

### 6.4 Grid selection

Selection is deterministic and has no manual override.

1. For each target tenor in `{0.10, 0.25, 0.50, 0.75, 1.00, 1.25}` years, take the nearest surviving expiry. Remove duplicates.
2. Within each selected expiry and each target `K/F` in `{0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20}`, take the nearest surviving strike within `±0.025`. Break ties toward the lower strike. Remove duplicates.
3. If step 2 produced no contract point (defined below), add the surviving quote nearest the contract strike at the selected expiry nearest the contract maturity.

The grid MUST satisfy all of the following, or `E_SPARSE_GRID` is raised:

- At least 20 points.
- At least 3 expiries.
- A **contract point**: at least one point with tenor in `[0.70, 1.30]` and `|K/S - K_c| <= 0.025`, where `K_c` is the normalized strike in §4.1. The contract strike is a fraction of spot, not of the forward.

The selected grid is written only to `data/derived/calibration_grid.parquet` and is Restricted. Its `iv` column holds the §6.3 value, never the vendor value.

### 6.5 Objective and solver

```text
phi_hat = argmin sum_j [ sigma_mkt(K_j, T_j) - sigma_Heston(K_j, T_j; phi) ]^2
```

- The objective is unweighted in implied-volatility space.
- `sigma_Heston` inverts the Heston European price using the numerically stable characteristic function of Albrecher et al. Calls are used for `K >= F`, puts otherwise.
- If model implied volatility cannot be inverted for a point, its residual is `1.0` and the event is counted.

| Solver setting | Value |
|---|---|
| Routine | `scipy.optimize.least_squares(method="trf", x_scale="jac")` |
| `ftol`, `xtol`, `gtol` | `1e-10` |
| `max_nfev` | `200` per start |
| Starts | 8, from a scrambled Sobol sequence with `calibration_seed = 620000` |

| Parameter | Bounds |
|---|---|
| `nu0` | `[1e-4, 4.0]` |
| `kappa` | `[0.01, 20.0]` |
| `theta` | `[1e-4, 4.0]` |
| `xi` | `[0.01, 5.0]` |
| `rho` | `[-0.999, 0.999]` |

The Feller condition MUST NOT constrain calibration or block pricing. `feller_ratio = 2 kappa theta / xi^2` is recorded.

### 6.6 Selection and stability

- The result is the converged start with the lowest RMS implied-volatility residual.
- Let `A` be the converged starts with `RMS <= max(1.05 × best RMS, 1e-8)`. The floor stops a near-zero residual from rejecting starts that recovered the same parameters.
- The fit is **stable** if `|A| >= 3` and, for every parameter, `(max - min) / |median| <= 0.10` over `A`.

### 6.7 Failure ladder

Apply these in order. The first satisfied step is used and recorded in `calibration.json`.

| Step | Condition | Action |
|---|---|---|
| 1 | Stable and `RMS <= 0.01` | Accept the full five-parameter fit |
| 2 | Otherwise | Fix `kappa` to the cited value recorded in `configs/published_params.yaml` and refit four parameters; state it as an assumption |
| 3 | Step 2 still fails, or `E_SPARSE_GRID` | Report no market-calibrated result; use `published` mode |

### 6.8 Synthetic surface fixture

`tests/fixtures/synthetic_raw_iv.csv` follows the `DATA_CONTRACT.md` schema, with these rules:

- The symbol is `SYN1`. `validate_raw.py --source fixtures` accepts `SYN1` to `SYN3` in place of the three real tickers.
- `style` is `E`, so de-Americanization error cannot enter V-12.
- Mids are exact Heston European prices with no added noise. `bid` and `ask` are `mid × 0.99` and `mid × 1.01`. `iv` is the Black–Scholes implied volatility of the mid.
- Strikes and expiries MUST satisfy every §6.4 requirement.

---

## 7. Simulation

### 7.1 Discretization

Full-truncation Euler for variance and log-Euler for spot (Lord et al.). For one step of size `Δ = T / (M × n_sub)`:

```text
nu_plus = max(nu_n, 0)
ln S_{n+1} = ln S_n + (r - q - 0.5 nu_plus) Δ + sqrt(nu_plus Δ) Z^S
nu_{n+1} = nu_n + kappa (theta - nu_plus) Δ + xi sqrt(nu_plus Δ) Z^nu
```

- `nu_n` may be negative in storage. Only `nu_plus` enters drift, diffusion, and regression.
- There is no Feller gating, reflection, or absorption.
- The simulator MUST accept `xi = 0` and `|rho| = 1`.

### 7.2 Randomness

- The generator is NumPy `PCG64`. Streams are spawned with `SeedSequence(master_seed)`.
- Regression and evaluation sets use independent child streams.
- No antithetic variates or variance-reduction scheme is used.
- The same config, seed, commit, and environment MUST produce identical results.

### 7.3 Path output

The simulator returns `(S, nu)` at `t_0..t_M` only, with shape `(n_paths, M + 1, d)`.

---

## 8. LSMC pricing

### 8.1 Algorithm

Use two independent path sets: regression set A and evaluation set B, each with `n_paths` paths.

**Fit on set A:**

1. Initialize `CF = g(T)` with cash-flow time `tau = T`.
2. For `k = M-1` down to `1`:
   1. Let `I` be the paths with `g(t_k) > 0`.
   2. If `|I| < n_min = max(50, 10p)`, store no policy for `t_k` and increment `skipped_dates`.
   3. Otherwise build features for `I` and standardize each feature using its mean and standard deviation over `I`.
   4. Drop any feature whose standard deviation is below `1e-12`.
   5. Regress `exp(-r (tau - t_k)) × CF` on the polynomial basis with `numpy.linalg.lstsq(rcond=1e-10)`.
   6. Compute continuation estimate `C_hat`. Exercise where `g(t_k) > C_hat`; update `CF` and `tau`.
   7. Store coefficients, retained features, means, and standard deviations.

**Evaluate on set B:**

3. Each path exercises at the first `t_k` with a stored policy where `g(t_k) > 0` and `g(t_k) > C_hat`, using coefficients and scaling from set A only. Otherwise it receives `g(T)` at `T`.
4. `Y_i = exp(-r tau_i) g_i`, and `V_B = mean(Y)`.
5. `Y_i^E = exp(-r T) g_i(T)`, and `V_E = mean(Y^E)`.
6. `Premium = mean(Y - Y^E)`, with paired SE.

The evaluation-set price is the reported price. The in-sample price from set A is a diagnostic only. A tie (`g = C_hat`) continues.

### 8.2 Regression basis

| Case | Variables | Default | Basis size `p` |
|---|---|---|---|
| CORE | `(S/S0, nu_plus)` | Total degree 3 | Degree 2: 6; degree 3: 10; degree 4: 15 |
| EXT-A `worst_of` | `(w_1, ..., w_d, nu_(1), ..., nu_(d))`, where `w_1 <= ... <= w_d` are sorted performances and `nu_(j)` is the variance of the asset ranked `j` | Total degree 2 | `d = 3`, degree 2: 28; degree 3: 84 |
| EXT-A `basket` | `(P_1, ..., P_d, nu_1, ..., nu_d)`, using `nu_plus` for each variance | Total degree 2 | `d = 2`: 15; `d = 3`: 28; `d = 5`: 66 |

Variance MUST be part of the regression state.

### 8.3 Baselines

CORE reports two baselines:

1. **European put.** Both the Monte Carlo `V_E` and the semi-analytic price are reported. The premium uses the paired Monte Carlo difference, so simulation noise and discretization error are shared by both legs.
2. **Constant-volatility Bermudan.** `crr_price` with `n_steps = 50 M` and exercise at the same dates. `sigma` is the Black–Scholes implied volatility of the fitted Heston European put at the contract `(K, T)`. It uses model output only, so it stays Internal. The gap to the Heston Bermudan is what stochastic volatility adds beyond matching the European level.

EXT-A has only the Monte Carlo European comparator.

---

## 9. Precision, workload, and runtime

### 9.1 Run profiles

| Setting | Smoke | Production |
|---|---|---|
| `n_paths` per set | `20_000` | `100_000` |
| `M` | `50` | `50` |
| `n_sub` | `4` | `4` |
| Basis degree | 3 | 3 |
| Master seeds | `[620001]` | `[620001, 620002, 620003, 620004, 620005]` |

One production CORE seed performs 8 × 10⁷ state updates (two path sets, two state variables). One production three-asset seed performs 2.4 × 10⁸.

### 9.2 Precision targets

These apply to the pooled production estimate.

| ID | Target | If missed |
|---|---|---|
| T1 | `SE(V_B) / V_B <= 0.005` | Raise `n_paths` within the runtime budget; the headline is not reported as meeting T1 otherwise |
| T2 | `SE(Premium) / Premium <= 0.10` | Raise `n_paths` within the budget; if still missed, set `premium_resolved = false` and report the 95% interval without a point claim |

A premium whose 95% interval includes zero is a valid finding and is reported that way.

**Path-reduction rule.** `n_paths` may be reduced for computing reasons only if T1 still holds on the pooled estimate. `n_paths` MUST NOT go below `20_000`. Below that rule, the fallback is to reduce the tier, not the precision.

### 9.3 Required sweeps

Sweeps vary one factor from the production settings. Each configuration uses all production seeds.

| Factor | Values |
|---|---|
| `n_paths` | `20_000`, `50_000`, `100_000`, `200_000` |
| `n_sub` | `1`, `2`, `4`, `8` |
| `M`, at fixed `Δ = 1/200` | `10`, `25`, `50`, `100` |
| Basis degree | `2`, `3`, `4` (EXT-A: `2`, `3`) |
| `r` | Base value, `+0.01`, `-0.01` |
| Calibration date (market mode) | `2026-10-13`, `2026-10-14`, `2026-10-15` |

The `M` sweep is a contract-convergence study, not a numerical convergence test of the 50-date contract.

### 9.4 Runtime budgets and C++ gate

| ID | Budget |
|---|---|
| B1 | One production pricing at one seed: `<= 30 s` median wall time over 5 repeats, single process |
| B2 | Full required sweep: `<= 6 h` |
| B3 | Reviewer smoke path from a clean clone: `<= 10 min` |

EXT-B may start only when all of these hold:

1. B1 or B2 has been measured and missed on the local baseline environment.
2. The bottleneck has been profiled.
3. Vectorized NumPy and Numba have been tried.
4. Reducing paths under §9.2 would break T1.

The local baseline environment is recorded automatically in each manifest.

---

## 10. Python API contract

The installable package is `hestonpricer`, located at `python/hestonpricer/`. Public names and semantics below are binding. Private helpers are not.

### 10.1 Data structures

The package uses plain functions and plain dictionaries. It MUST NOT define dataclasses or other classes for model inputs or results. The only classes in the package are the exceptions in §14.1, because Python exceptions must be classes.

The dictionaries below are binding: keys and meanings are part of the public API. Where a key has a default, a caller may leave it out.

```python
# One asset's Heston parameters. Functions take one dict, or a list with one dict per asset.
params = {"v0": 0.04, "kappa": 1.5, "theta": 0.04, "xi": 0.5, "rho": -0.7}

# Rate and dividend yields.
market = {"r": 0.04, "q": [0.0]}          # q has one entry per asset

# The contract. Defaults are the values shown.
contract = {"strike": 1.0, "maturity": 1.0, "n_exercise": 50}

# Optional contract keys, EXT-A only. These defaults leave CORE unchanged.
#   contract["payoff"]  = "worst_of"    # or "basket"
#   contract["kind"]    = "put"         # or "call"
#   contract["weights"] = None          # basket weights; None means equal

# Simulation settings. n_paths is required; the other defaults are the values shown.
sim = {
    "n_paths": 100_000,
    "n_substeps": 4,
    "basis_degree": 3,
    "master_seed": 620001,
    "identical_assets_test_mode": False,   # shares Z^perp across assets; V-21 only
}

# Returned by simulate(). Arrays have shape (n_paths, M + 1, d).
paths = {"S": ..., "nu": ...}

# Returned by fit_policy().
policy = {
    "dates": {},                    # k -> {"coef", "keep", "mean", "std"}; missing k means no policy
    "basis_degree": 3,
    "in_sample_bermudan": 0.0,
    "skipped_dates": 0,
    "rank_deficient_dates": 0,
    "min_itm_count": 0,
}

# Returned by evaluate_policy() and price_bermudan().
price_result = {
    "bermudan": 0.0, "se_bermudan": 0.0,
    "european": 0.0, "se_european": 0.0,
    "premium": 0.0, "se_premium": 0.0,
    "in_sample_bermudan": 0.0,
    "exercise_fraction": [],        # length M
    "skipped_dates": 0,
    "rank_deficient_dates": 0,
    "min_itm_count": 0,
    "n_paths": 0,
}
```

### 10.2 Functions

```python
# analytic.py
def heston_european(kind, s0, strike, maturity, params, r, q): ...   # kind is "call" or "put"; returns float

# iv.py
def bs_price(kind, s0, strike, maturity, sigma, r, q): ...
def bs_implied_vol(kind, price, s0, strike, maturity, r, q): ...                  # nan if no root
def american_implied_vol(kind, price, s0, strike, maturity, r, q, n_steps=500): ... # nan if no root

# tree.py
def crr_price(kind, s0, strike, maturity, sigma, r, q, n_steps, exercise_steps=None): ...
# exercise_steps=None means American. A step list prices a Bermudan.

# simulate.py
def simulate(params, corr, market, contract, sim, rng, normals=None): ...   # returns a paths dict
# params is a list of dicts. normals injects a shared fixture; rng is ignored when normals is given.

# lsmc.py
def fit_policy(paths, contract, market, basis_degree): ...          # returns a policy dict
def evaluate_policy(policy, paths, contract, market): ...           # returns a price_result dict
def price_bermudan(params, corr, market, contract, sim): ...        # returns a price_result dict

# surface.py
def filter_quotes(raw, cfg): ...      # raw is a DataFrame, cfg a dict; returns (DataFrame, dict of counts)
def select_grid(quotes, cfg): ...     # returns a DataFrame

# calibrate.py
def calibrate_asset(grid, r, q, cfg): ...                 # returns a dict with the §12.5 asset fields

# correlation.py
def estimate_correlation(log_returns, window=252): ...    # returns a dict with the §12.6 fields
```

### 10.3 Preconditions

Violations raise `InputError` (§14.1).

| Function | Requirement |
|---|---|
| `heston_european` | `s0, strike, maturity > 0`; `xi > 0`; `\|rho\| < 1`; `v0, kappa, theta > 0` |
| `simulate` | `xi >= 0`; `\|rho\| <= 1`; `kappa >= 0`; `v0, theta >= 0`; `corr` is d×d, symmetric, unit diagonal, PSD; `len(params) == len(market["q"]) == d` |
| `fit_policy` | `basis_degree in {1, 2, 3, 4}` |
| All pricing functions | All numeric inputs finite |

---

## 11. CLI and workflow

```bash
python python/validate_raw.py --source {ivolatility|massive|fixtures}
python python/make_fixtures.py
python scripts/inspect_schema.py <path-under-data>
python scripts/run.py calibrate --config configs/<name>.yaml
python scripts/run.py price     --config configs/<name>.yaml [--results-root <dir>]
python scripts/run.py sweep     --config configs/<name>.yaml
python scripts/verify_outputs.py
python scripts/make_figures.py
bash   scripts/smoke.sh
bash   scripts/check_no_data.sh
```

| Exit code | Meaning |
|---|---|
| `0` | Success |
| `1` | Unhandled internal error |
| `2` | Input or schema validation failed |
| `3` | A check, gate, or precision requirement failed |
| `4` | Governance refusal |

`inspect_schema.py` prints only file names, column names, dtypes, row counts, and null counts. `make_figures.py` MUST NOT read anything under `data/`.

`smoke.sh` MUST, using only Public material:

1. Regenerate the synthetic fixtures.
2. Calibrate to the synthetic surface.
3. Price the CORE contract with the smoke profile, using the fixture's known parameters rather than the fitted ones, so that the comparison does not depend on solver differences across machines.
4. Compare the price with `results/smoke/expected.json` at relative tolerance `1e-8`.
5. Run the `fast` test set (§13) and return non-zero on any failure.

---

## 12. Artifact schemas

### 12.1 Run configuration

```yaml
schema_version: 1
run_name: core_amd_production
tier: CORE                    # CORE | EXT-A | EXT-B
reported: true
data_mode: market             # synthetic | published | market
assets: [AMD]
snapshot_date: 2026-10-14
market_inputs: configs/market_inputs.yaml
params_source: results/calibration.json
contract: {strike: 1.0, maturity: 1.0, n_exercise: 50}
simulation: {n_paths: 100000, n_substeps: 4, basis_degree: 3}
seeds: [620001, 620002, 620003, 620004, 620005]
```

Unknown keys MUST raise `E_CONFIG`.

### 12.2 Run directory

`run_id` is `<run_name>-<first 8 characters of commit>`. A run writes:

```text
results/runs/<run_id>/
  config.yaml     exact copy of the input config
  manifest.json   provenance
  result.json     numbers; written only if every value is finite
  run.log         sanitized log
```

A run with `reported: true` MUST refuse a dirty working tree (`E_DIRTY_TREE`). Changes under `results/` do not count as dirty, because each run writes there. `--results-root` redirects all run output and is how V-08 repeats a run without touching a completed `run_id`.

### 12.3 `manifest.json`

```json
{
  "schema_version": 1,
  "spec_version": "1.2",
  "data_contract_version": "0.3",
  "run_id": "core_amd_production-1a2b3c4d",
  "git_commit": "<full hash>",
  "git_dirty": false,
  "command": "python scripts/run.py price --config configs/core_amd_production.yaml",
  "config_sha256": "<hash>",
  "data_mode": "market",
  "seeds": [620001, 620002, 620003, 620004, 620005],
  "environment": {
    "python": "3.x.y",
    "packages": {"numpy": "x.y.z", "scipy": "x.y.z", "pandas": "x.y.z"},
    "os": "...", "cpu": "...", "cores": 0, "ram_gb": 0
  },
  "inputs": [
    {"dataset": "ivolatility_raw_iv", "tickers": ["AMD"], "snapshot_date": "2026-10-14",
     "file_name": "<name>", "row_count": 0}
  ],
  "outputs": [{"path": "result.json", "sha256": "<hash>"}],
  "started_at": "<UTC ISO-8601>",
  "finished_at": "<UTC ISO-8601>",
  "wall_seconds": 0.0
}
```

Input records contain identifiers and counts only, never data values. Output hashes apply to Public and Internal files only.

### 12.4 `result.json`

```json
{
  "schema_version": 1,
  "run_id": "core_amd_production-1a2b3c4d",
  "tier": "CORE",
  "data_mode": "market",
  "units": "fraction_of_initial_spot",
  "assets": ["AMD"],
  "contract": {"strike": 1.0, "maturity": 1.0, "n_exercise": 50},
  "simulation": {"n_paths": 100000, "n_substeps": 4, "basis_degree": 3},
  "per_seed": [
    {"seed": 620001, "bermudan": 0.0, "se_bermudan": 0.0, "european": 0.0,
     "se_european": 0.0, "premium": 0.0, "se_premium": 0.0,
     "in_sample_bermudan": 0.0, "wall_seconds": 0.0}
  ],
  "pooled": {
    "bermudan": 0.0, "se_bermudan": 0.0,
    "european_mc": 0.0, "se_european": 0.0, "european_analytic": 0.0,
    "baseline_const_vol_bermudan": 0.0,
    "premium": 0.0, "se_premium": 0.0, "premium_ci95": [0.0, 0.0],
    "premium_resolved": true
  },
  "diagnostics": {
    "exercise_fraction_by_date": [],
    "min_itm_count": 0,
    "skipped_dates": 0,
    "rank_deficient_dates": 0,
    "feller_ratio": [0.0]
  },
  "precision": {"T1": true, "T2": true}
}
```

`european_analytic` and `baseline_const_vol_bermudan` are `null` for EXT-A.

### 12.5 `results/calibration.json`

```json
{
  "schema_version": 1,
  "data_mode": "market",
  "snapshot_date": "2026-10-14",
  "r": 0.0,
  "assets": {
    "AMD": {
      "q": 0.0,
      "ladder_step": 1,
      "params": {"v0": 0.0, "kappa": 0.0, "theta": 0.0, "xi": 0.0, "rho": 0.0},
      "fixed_params": [],
      "feller_ratio": 0.0,
      "grid_size": 0,
      "n_expiries": 0,
      "rms_iv_residual": 0.0,
      "stable": true,
      "filter_drops": {"F1": 0, "F2": 0, "F3": 0, "F4": 0, "F5": 0, "F6": 0, "F7": 0, "F8": 0},
      "starts": [{"start": 0, "converged": true, "status": 1, "rms_iv_residual": 0.0}],
      "residual_buckets": [{"tenor_bucket": "...", "moneyness_bucket": "...", "n": 5, "rms": 0.0}]
    }
  }
}
```

`residual_buckets` MUST contain only buckets with `n >= 5`. Per-contract residuals are prohibited.

### 12.6 `results/estimates.json` (EXT-A)

This file contains the window dates, per-ticker sample size and realized volatility, and the correlation matrix. Return series and prices are prohibited.

### 12.7 Evidence map and verification table

`results/evidence_map.csv` has these columns:

```text
claim_id, location, artifact_type, run_id, result_path, json_pointer, check_ids
```

Every quantitative claim, figure, and table in the report and slides MUST have one row.

`results/verification_table.csv` has these columns:

```text
check_id, status, measured, threshold, run_id, commit, evaluated_at
```

`status` is `PASS`, `FAIL`, or `NOT_RUN`. `NOT_RUN` MUST include a reason in `measured`.

### 12.8 Required evidence

| Evidence | Source |
|---|---|
| Verification table | `results/verification_table.csv` |
| Convergence plots in paths, sub-steps, exercise dates, and basis degree | §9.3 sweeps |
| Fitted surface versus known surface | Synthetic fixture only |
| Market fit quality | Aggregate statistics from `calibration.json`; no market data points plotted |
| Price, European value, and premium with uncertainty | `result.json` |
| Seed robustness table | `per_seed` |
| Test and CI logs, run manifests | CI and `results/runs/` |

Every principal figure or table MUST state:

- Its conclusion, units, sample or workload, repetitions, uncertainty, exclusions, source run, and generating code.
- A readable encoding that does not rely on colour alone.

### 12.9 Visualization plan for the report and presentation

This is the plan for the figures in the report and the eight-minute presentation. It is a requirement for G3, not a statement that any figure exists. Each figure MUST follow §12.8 and have an evidence-map row. All are generated by `scripts/make_figures.py`, which MUST NOT read anything under `data/`.

| ID | Figure | Source | Class | Report | Slides |
|---|---|---|---|---|---|
| VZ-1 | PyVista 3D surface of the calibrated Heston implied-volatility surface | Fitted parameters in `results/calibration.json` | Internal | Yes | Yes |
| VZ-2 | Monte Carlo European vs. semi-analytic prices over the V-03 grid of strikes and maturities, with the difference in SE | Comparison run on the reported parameters | Internal | Yes | Yes |
| VZ-3 | LSMC Bermudan vs. CRR tree on the V-04 cases | Comparison run | Public | Yes | Yes |
| VZ-4 | Heston Bermudan, European, and constant-volatility Bermudan across strikes | Comparison run on the reported parameters | Internal | Yes | No |
| VZ-5 | Convergence panels in paths, sub-steps, exercise dates, and basis degree | §9.3 sweeps | Internal | Yes | One panel |
| VZ-6 | Price, European value, and premium with 95% intervals | `result.json` | Internal | Yes | Yes |
| VZ-7 | Fitted vs. known surface on the synthetic fixture | `results/smoke/calibration.json` | Public | Yes | No |
| VZ-8 | Seed robustness table and aggregate fit-quality table | `per_seed`; `residual_buckets` | Internal | Yes | No |

**VZ-1 requirements.**

- Evaluate `sigma_Heston` with `model_implied_vols` on a dense regular grid of `K/F` in `[0.80, 1.20]` and tenor in `[0.10, 1.25]`, using the fitted parameters only.
- Show `K/F`, tenor, and implied volatility as labelled axes, with height encoding implied volatility and a single-hue colour ramp with a labelled scale, so the figure does not rely on colour alone.
- Render off-screen with PyVista to a PNG for the report and slides.
- Label it as a fitted model surface and state the data mode and snapshot date.
- It MUST NOT overlay market implied-volatility points, quotes, or per-contract residuals (`DATA_CONTRACT.md` §4.3).

**Comparison runs.** VZ-2, VZ-3, and VZ-4 use a pipeline command that prices the listed contracts and writes Internal summaries under `results/comparisons/`. All values are fractions of initial spot. Numerical-versus-analytical figures test the engine on known answers and do not show that Heston fits AMD (§16).

**Presentation order.** Question and contract; data boundary; VZ-1; VZ-3 and VZ-2 as engine checks; one VZ-5 panel; VZ-6 as the result; limitations.

**Implementation.** PyVista and the comparison command are added to the lock file and pipeline when this plan is implemented. The report's conclusions and narrative are written by the owner.

---

## 13. Repository contract

```text
.
├── SPEC.md  README.md  AI_USE.md  LICENSE  DATA_CONTRACT.md
├── pyproject.toml  <pinned lock file>
├── .gitignore  .claude/settings.json  .github/workflows/
├── configs/        governance.yaml, market_inputs.yaml, published_params.yaml, one YAML per reported run
├── python/         validate_raw.py, make_fixtures.py, hestonpricer/
├── scripts/        run.py, verify_outputs.py, make_figures.py, inspect_schema.py,
│                   smoke.sh, check_no_data.sh
├── tests/          test_*.py, reference_values.yaml, reference_impl.py, fixtures/
├── results/        calibration.json, estimates.json, runs/, figures/, smoke/,
│                   evidence_map.csv, verification_table.csv
├── prototype/      exploratory notebooks with outputs cleared; not part of the assessed workflow
└── data/           Git-ignored; Restricted; never in a commit or submission
```

Directory `cpp/` is added only if EXT-B opens.

**Tests.** Each check in §16 MUST have a test whose name starts with its ID, such as `test_V03_european_mc_grid`. Statistical tests use fixed seeds. Market-dependent checks are skipped in CI and report `NOT_RUN`, not `PASS`.

Tests use one of two pytest markers:

- `fast`: V-01, V-02, V-05, V-07, V-08, V-12, and part (a) of V-14, using the smoke profile where paths are needed. These run within B3.
- `slow`: every other check. The owner or an agent runs these, and `verify_outputs.py` records them from completed runs.

Tests for V-25 to V-28 are added when EXT-A opens.

**CI.** Every push runs linting, the `fast` test set, `smoke.sh`, and `check_no_data.sh`. CI has no market data and no data-provider secrets.

**README.** It MUST cover the question, scope, data boundary, setup, inputs, run and verify commands, outputs, limitations, and exactly what a reviewer can and cannot reproduce.

**Reviewer boundary.** A reviewer can reproduce the full pipeline on Public fixtures and `published` mode. A reviewer cannot rerun `market` mode. For that result, the reviewer inspects committed code, configs, manifests, Internal outputs, and evidence maps.

**Final version.** The final commit is annotated tag `v1.0-final`.

**Submission (Route A).**

1. Create a fresh `git clone` in a temporary directory, so no ignored file can enter the package.
2. Check out `v1.0-final`, and zip the clone with `.git`.
3. Extract the ZIP elsewhere. Run `git status`, `git log`, `scripts/check_no_data.sh`, and `scripts/smoke.sh`.
4. Upload it to D2L, then open the exact uploaded file.

---

## 14. Errors and edge cases

### 14.1 Error codes

All errors derive from `HestonError` and carry a `code`.

| Code | Trigger | Required behaviour | Exit |
|---|---|---|---|
| `E_CONFIG` | Unknown or invalid config key | Stop before computing | 2 |
| `E_INPUT` (`InputError`) | §10.3 precondition violated | Raise immediately | 2 |
| `E_SCHEMA` | Data-contract rule violated | Stop; report rule, file, count, row indices only | 2 |
| `E_SPARSE_GRID` | §6.4 requirements not met | Apply ladder step 3 | 3 |
| `E_CALIB_FAILED` | No converged start | Apply ladder step 2, then step 3 | 3 |
| `E_CORR_NOT_PSD` | §5.4 check fails | Stop EXT-A | 3 |
| `E_NONFINITE` | Any reported output is NaN or infinite | Write no `result.json` | 3 |
| `E_CHECK_FAILED` | Gated check failed in `verify_outputs.py` | Record `FAIL`; stop dependent reported runs | 3 |
| `E_DIRTY_TREE` | Reported run with a dirty tree | Stop before computing | 4 |
| `E_GOV_UNAUTHORIZED` | Market run while unauthorized | Stop before reading `data/` | 4 |
| `E_GOV_RESTRICTED_LEAK` | `data/` path in Git, or prohibited output detected | Stop; follow `DATA_CONTRACT.md` §4.4 | 4 |

### 14.2 Sanitized exceptions

Exceptions raised while reading Restricted data MUST be caught and re-raised with only file name, row index, column name, and rule name. Row values, pandas repr output, and chained tracebacks that could echo values MUST NOT reach logs. Use `raise ... from None`.

### 14.3 Required edge-case behaviour

| Case | Required behaviour |
|---|---|
| Feller condition violated | Proceed; record `feller_ratio` |
| Negative stored variance | Use `nu_plus = 0` in the next step |
| `xi = 0` in simulation | Variance feature may be degenerate and is dropped; pricing continues |
| `xi = 0` in `heston_european` | Raise `InputError`; do not silently substitute Black–Scholes |
| `\|rho\| = 1` in simulation | `Z^nu = ± Z^S` without numerical error |
| `M = 1` | No regression; Bermudan and European estimates are identical; premium and its SE are exactly 0 |
| No in-the-money paths at a date, or fewer than `n_min` | Skip exercise at that date; count it |
| Rank-deficient regression | Use the minimum-norm solution; count it; do not raise |
| All values zero, so SE is 0 | A relative target is undefined: report it unmet; comparison checks use `max(SE, 1e-12)` |
| Singular `R` | Valid; eigendecomposition; do not call Cholesky |
| Quote `mid` differs from `mean_price` | Use computed `mid`; count rows differing by more than `1e-6` |
| No implied-volatility root | Drop under F8; count it |
| Model implied volatility not invertible during fit | Residual `1.0`; count it |
| `maturity` or `strike` passed as an integer | Cast to `float64` at the API boundary. Time and payoff arrays MUST NOT have an integer dtype, because `np.full(n, 1)` silently truncates later exercise times to 0 and removes discounting |
| Re-run with a completed `run_id` | Do not overwrite; exit with a clear message, unless `--results-root` names a new location |
| Run interrupted | Treat the run directory as incomplete; V-15 fails until re-run |

---

## 15. Agent and governance rules

`DATA_CONTRACT.md` governs data handling. These rules apply to every AI agent on the project.

An agent MUST NOT:

- Read, print, plot, paste, sample, or summarize at row level any value under `data/`. Structure is available only through `scripts/inspect_schema.py`.
- Run `cat`, `head`, `zcat`, DataFrame previews, or ad hoc scripts against `data/`.
- Set `market_use_authorized` or write `configs/governance.yaml` without the owner's recorded instruction.
- Run `market` mode by any means other than the committed scripts listed below.
- Hand-edit `results/**`. Those files are written only by pipeline scripts.
- Generate, recall, or fill in reference values in `tests/reference_values.yaml` or `configs/published_params.yaml`. The owner transcribes them from the cited source and records the table or page.
- Write the report, slides, or claims. Reported results and figures come only from pipeline scripts.
- Change a pass rule, threshold, seed, or data filter (§1.3).
- Access credentials or commit anything under `data/`.
- Create misleading or artificial Git history.

An agent MAY:

- Write code, tests, fixtures, and documentation. Owner review may follow the commit.
- Commit and push when the owner asks, but only if `data/` is listed in `.gitignore` and `scripts/check_no_data.sh` passes immediately before the push.
- Run tests and pipelines in `synthetic` and `published` modes.
- Run `market` mode only through `python/validate_raw.py`, `scripts/run.py`, and `scripts/verify_outputs.py`, whose output is Internal, and run `scripts/make_figures.py`.
- Read Public and Internal files.

Debug data problems on fixtures first. If a problem occurs only on licensed data, describe it by file name, row index, column, and rule.

`AI_USE.md` MUST record the tools and access mode, their roles, what was checked and how, important rejected or corrected suggestions, the data boundary, and the residual limitations that remain after those checks.

---

## 16. Verification matrix

Reference values in V-01 and V-06 MUST come from `tests/reference_values.yaml`, with citation and table or page, entered by the owner. All MC-based rules use the fixed test seed `620100` unless stated otherwise. V-09, V-10, and V-11 run on the reported parameter set, or on the published set before a market fit exists. Unless a rule says otherwise, Monte Carlo checks use the production profile.

The report and README MUST state what the checks establish and what they do not. For example, V-03 and V-06 test the engine on known answers. They do not show that Heston is the right model for AMD.

### 16.1 CORE checks

| ID | Check | Pass rule | Origin |
|---|---|---|---|
| V-01 | Semi-analytic Heston European vs. published benchmark | Absolute difference `<= max(1e-6, half the last published digit)` | New |
| V-02 | Put–call parity; small-`xi` limit (`xi = 1e-4`, `rho = 0`, `v0 = theta`) vs. Black–Scholes | Parity residual `<= 1e-10`; BS difference `<= 1e-6` | New |
| V-03 | Heston European MC vs. analytic, `T` in `{0.25, 0.5, 1, 1.5, 2}` × 8 evenly spaced `K/S` in `[0.80, 1.25]`, `Δ = 1/400`, `200_000` paths | At least 38 of 40 options within 3 SE | Proposal V1 |
| V-04 | GBM reduction (`xi = 0`, `v0 = theta`): LSMC Bermudan vs. 5,000-step CRR tree on the same exercise dates, for `(sigma, K)` in `{(0.3, 1.0), (0.5, 1.0), (0.5, 0.9), (0.3, 1.1)}` | Difference `<= 3 SE` for each pair | New |
| V-05 | `M = 1` vs. semi-analytic European | Difference `<= 3 SE` | Proposal V4 |
| V-06 | Heston Bermudan vs. Clarke–Parrott American benchmark `B` at each published spot, `M` in `{25, 50, 100}`, `200_000` paths | At `M = 100`: `\|V - B\| <= max(0.01 B, 3 SE)` and `V - B <= 3 SE`. For each rise in `M`: `V_low - V_high <= 3 sqrt(SE_low^2 + SE_high^2)` | New |
| V-07 | Vectorized LSMC vs. independent scalar implementation in `tests/reference_impl.py` on `tests/fixtures/shared_normals.npz` | Relative difference `<= 1e-10` | Replaces proposal V7 for CORE |
| V-08 | Repeat run with same config, seed, commit, and environment, in a new `--results-root` | Numeric fields of `result.json` identical, excluding `wall_seconds` | New |
| V-09 | Basis degree 2 vs. 3, and 3 vs. 4, on identical paths | Pooled price difference `<= 2 SE` of the higher-degree run | Proposal risk table |
| V-10 | `n_sub = 4` vs. `8`, pooled | `\|V_4 - V_8\| <= 3 sqrt(SE_4^2 + SE_8^2)`; if not, `8` becomes the production setting before G1 | New |
| V-11 | All pairs of production seeds | `\|V_a - V_b\| <= 3 sqrt(SE_a^2 + SE_b^2)` | Proposal V8, corrected |
| V-12 | Calibration on synthetic surface with known parameters | RMS IV `<= 1e-4`; `v0, kappa, theta, xi` within 5% relative; `rho` within `0.05` absolute | Proposal V2, extended |
| V-13 | Calibration on market grid (`market` mode only) | RMS IV `<= 0.01` and stable under §6.6 | Proposal V3, defined |
| V-14 | Implied-volatility conversion | (a) Tree-inversion round trip recovers `sigma` within `1e-4`; (b) market mode: for near-the-money pairs with `\|K/F - 1\| <= 0.05`, at least 90% of call and put IVs agree within their bid–ask IV bands; (c) market mode: median `\|own IV - vendor IV\| <= 0.02` | New |
| V-15 | Output completeness | Every configured run is complete, all values finite, and manifest hashes match | Proposal V9 |
| V-16 | Governance scan | `git log --all --name-only -- data/` is empty; pre-commit hook active; no prohibited fields or secrets in tracked files or submission package | New |
| V-17 | Clean-clone reproduction | `smoke.sh` passes from a fresh clone within B3 | Feedback, rubric |
| V-18 | Traceability | Every evidence-map row resolves; `make_figures.py` regenerates all figures | Rubric |

### 16.2 EXT-A checks

| ID | Check | Pass rule | Origin |
|---|---|---|---|
| V-20 | Sample driver correlations vs. §4.4 | All entries within 3 SE of target | New |
| V-21 | Three identical assets with `R = ones` and `identical_assets_test_mode=True` vs. single asset | Difference `<= 3 SE` | Proposal V5 |
| V-22 | Ordering on same paths, using paired comparisons: worst-of Bermudan `>=` worst-of European; worst-of Bermudan `>=` each single-asset Bermudan; worst-of European `>=` each single-asset European | Each holds within 3 paired SE | Proposal V6, corrected |
| V-23 | V-09, V-10, and V-11 repeated for the three-asset contract | Same rules | Proposal V8 |
| V-24 | Calibration passes V-13 for all three assets | Same rule, or the affected asset's substitution is stated | Proposal scope |
| V-25 | `d = 1` call and put, both payoff types, vs. the single-asset engine on the shared fixture | Relative difference `<= 1e-10` | v1.2 scope |
| V-26 | `basket` on identical assets with `R = ones` and `identical_assets_test_mode=True` vs. single asset | Difference `<= 3 SE` | v1.2 scope |
| V-27 | European basket put–call parity on the same paths: `mean(C - P)` vs. `sum_i w_i exp(-q_i T) - K exp(-r T)` | Difference `<= 3` paired SE | v1.2 scope |
| V-28 | Bermudan `>=` European, for calls and puts of both payoff types, on the same paths | Each holds within 3 paired SE | v1.2 scope |

### 16.3 EXT-B check

| ID | Check | Pass rule | Origin |
|---|---|---|---|
| V-30 | C++ engine vs. Python on the shared fixture | Relative difference `<= 1e-10` | Proposal V7 |

---

## 17. Gates and fallbacks

### 17.1 Gates

| Gate | Date | Requires | On failure |
|---|---|---|---|
| G0 | Nov 3 | Reviewable repository; method and data boundary documented; smoke test passes from a clean clone | Record the gaps; revise the method or evidence plan |
| G1 | Nov 14 | V-01 to V-12 and V-15 pass; B1 and B2 pass; §5.2 resolved | EXT-A does not start |
| G2 | Nov 20 | CORE result frozen: T1 met; V-13, V-14, V-16, and V-18 resolved under §6.7 and §5.2 | Freeze the minimum defensible analysis |
| G3 | Nov 27 | Draft report and slides; figures regenerate; every claim maps to evidence | Remove unsupported claims |
| G4 | Dec 1 | Eight-minute presentation finishes within the assigned slot | Cut content, not the verification or limitations slides |
| G5 | Dec 3 | Route A package verified; all CORE checks recorded | Submit the verified minimal package before extras |
| G6 | Dec 4, 11:59 p.m. | Report and repository uploaded to D2L; exact uploaded files opened and checked | Re-upload before the deadline |

### 17.2 EXT-A sequence

EXT-A is optional. It proceeds from `d = 2` to `d = 3` only after G1. If V-20 to V-24 do not pass by Nov 20, EXT-A is dropped from the claims. The CORE question and results stand.

Basket payoffs, calls, and values of `d` other than 3 need V-25 to V-28 to pass before any claim is made. Market data covers NVDA, AMD, and INTC only, so other values of `d` use synthetic or published inputs.

### 17.3 Fallback rules

| Trigger | Action |
|---|---|
| OD-1 is unresolved or negative | `published` mode becomes the reported result; revise the question and claims to match |
| Calibration fails | Apply §6.7 |
| V-09 fails | Add basis terms or features; report any residual bias |
| V-10 fails at `n_sub = 8` | Report discretization error as unresolved; do not claim a converged price |
| B1 or B2 fails | Follow §9.4; reduce paths only under §9.2 |
| Restricted data leak | Stop licensed-data work. If it reached Git history, contact the instructor immediately |

---

## 18. Definition of done

CORE is complete when all of the following hold:

- [ ] All §16.1 checks are recorded in `results/verification_table.csv`, with no unexplained `FAIL` or `NOT_RUN`.
- [ ] The production result has T1 met and a resolved or explicitly unresolved T2.
- [ ] The final price, European value, premium, uncertainty, and data mode appear consistently in report, slides, and README.
- [ ] Every principal figure and table has an evidence-map row and regenerates from committed Internal outputs.
- [ ] Limitations cover at least LSMC low bias, discretization bias, American-to-European conversion, flat `r`, Heston's lack of jumps and earnings effects, single-snapshot calibration, and any ladder step used.
- [ ] Data availability, authority, AI use, and residual AI limitations are disclosed.
- [ ] Both §8.3 baselines are reported, and the report states what the checks do and do not establish.
- [ ] `v1.0-final` exists, and the Route A package passes §13.
- [ ] No extension is claimed unless its checks passed.

---

## 19. Open decisions

Each item needs an owner decision. Defaults apply until changed through §1.3.

| ID | Priority | Decision | Default in this spec |
|---|---|---|---|
| OD-1 | Resolved 2026-11-08 | The iVolatility licence allows every use in `DATA_CONTRACT.md` provided the raw data is not published, and the owner holds written confirmation that derived results may be used for this project (`DATA_CONTRACT.md` §2.2). | Authority recorded; `market` runs stay refused until `configs/governance.yaml` sets `market_use_authorized: true` (§5.2) |
| OD-2 | High | Confirm AMD as the CORE asset. It pays no dividend, so `q = 0` holds and call quotes are effectively European. | AMD |
| OD-3 | Resolved 2026-11-08 | Treasury 1-Year CMT yields from the official table: 4.48% (2026-10-13), 4.53% (2026-10-14), 4.49% (2026-10-15). The proposal's fixed `r = 5%` is withdrawn. | Values in `configs/market_inputs.yaml` |
| OD-4 | High | Compare `DATA_CONTRACT.md` §1.2 with the actual iVolatility header. Confirm that `stock_price_for_iv` is synchronous with the quotes. | Contract schema, unverified |
| OD-5 | High | Choose the published parameter set and the literature `kappa` for §6.7, and enter both from their source papers. | Empty until the owner transcribes it |
| OD-6 | Medium | Confirm contract terms `K = 1.00`, `T = 1.00`, `M = 50`, and primary date `2026-10-14`. If any fit or price has already been viewed, record that in §22. | Values in §4.1 and §5.3 |
| OD-7 | Medium | The proposal barred repository write access. The owner now allows agent writes, commits, and pushes under `DATA_CONTRACT.md` §3.3. Disclose this change in `AI_USE.md` and the report. | Agent writes, commits, and pushes allowed |
| OD-8 | Medium | Raw-file hashes were removed from the data contract. Traceability currently relies on file name and row count. Hashes are not values and would strengthen the provenance evidence. | No raw-file hashes |
| OD-9 | Low | Choose the code licence for `LICENSE`. | Unset |
| OD-10 | EXT-A only | Confirm `q_INTC = 0` and NVDA's annualized declared dividend on the snapshot date. | §4.3 |

---

## 20. Traceability

### 20.1 Proposal decisions superseded

| Proposal | This spec | Reason |
|---|---|---|
| Three-asset worst-of is the primary target | Single-asset CORE; three-asset is conditional | Instructor feedback |
| C++ pricing engine | Python engine; C++ behind a measured gate | Instructor feedback |
| OpenMP tier and HPC mention | Removed | Instructor feedback |
| Commit raw market snapshot; publish later | Never committed or submitted | Feedback; `DATA_CONTRACT.md` |
| Fitted surface vs. market quotes figure | Synthetic surface plus aggregate market statistics | `DATA_CONTRACT.md` |
| `r = 5%`, all `q = 0` | Treasury-derived `r`; asset-specific `q` | Market-consistent inputs; NVDA now pays a material dividend |
| Calibrate directly to BS IV from listed quotes | OTM quotes, de-Americanized IVs | Listed single-stock options are American-style |
| Correlations and quotes on the same window | Stated gap in windows | Massive data ends in July 2026 |
| Calibration-grid robustness check (E7, stretch) | Removed | Stretch work; day-to-day stability covers robustness for CORE |
| Same seeds "agree within 1 SE" | `3 sqrt(SE_a^2 + SE_b^2)` | A 1-SE rule on two independent estimates would fail about half the time even for a correct engine |
| Worst-of European `>=` each single-asset Bermudan | Corrected ordering in V-22 | False in general; identical assets are a counterexample, and it contradicts proposal V5 |

### 20.2 Verification ID crosswalk

| Proposal Table 1 | This spec |
|---|---|
| V1 | V-03 |
| V2 | V-12 |
| V3 | V-13 |
| V4 | V-05 |
| V5 | V-21 |
| V6 | V-22 |
| V7 | V-07 (CORE), V-30 (EXT-B) |
| V8 | V-11, V-23 |
| V9 | V-15 |

Proposal Tables 2, 3, A2, and A3 and its evidence list cite V10–V14, which Table 1 does not define. Those citations are retired. Thread invariance was removed with OpenMP.

### 20.3 Instructor feedback coverage

| Feedback item | Addressed in |
|---|---|
| Single-asset study is the committed result | §2.1 |
| Stay in Python unless a measured bottleneck justifies C++ | §9.4 |
| Cut OpenMP | §2.3 |
| Attempt three assets only after accuracy and runtime gates | §2.2, §17 |
| Provider and retention, commit, and publication terms | §5.2, §5.3, OD-1, `DATA_CONTRACT.md` |
| One Heston parameterization, including initial variance | §4.2 |
| Variance discretization | §7.1 |
| Full cross-asset shock correlation | §4.4 |
| LSMC basis and regression controls | §8 |
| Paths, steps, repetitions, MC error target, runtime | §9 |
| Pass rules fixed before viewing main results | §1.3, §16 |
| Residual AI limitation was absent | §15, §18 |
| Minimum precision rule for reducing paths | §9.2 |
| Revise question and claims if three assets are not completed | §2.2, §17.2 |

---

## 21. References

Citations follow the proposal unless marked.

- Albrecher, H., Mayer, P., Schoutens, W., & Tistaert, J. (2007). The little Heston trap. *Wilmott Magazine*, January, 83–92.
- Burkovska, E., et al. (2018). Calibration to American options: numerical investigation of the de-Americanization method. *Quantitative Finance*. Added here; verify the full citation before use.
- Clarke, N., & Parrott, K. (1999). Multigrid for American option pricing with stochastic volatility. *Applied Mathematical Finance*, 6(3), 177–195.
- Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). Option pricing: A simplified approach. *Journal of Financial Economics*, 7(3), 229–263.
- Fang, F., & Oosterlee, C. W. (2008). A novel pricing method for European options based on Fourier-cosine series expansions. *SIAM Journal on Scientific Computing*, 31(2), 826–848.
- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer.
- Heston, S. L. (1993). A closed-form solution for options with stochastic volatility with applications to bond and currency options. *The Review of Financial Studies*, 6(2), 327–343.
- Longstaff, F. A., & Schwartz, E. S. (2001). Valuing American options by simulation: A simple least-squares approach. *The Review of Financial Studies*, 14(1), 113–147.
- Lord, R., Koekkoek, R., & van Dijk, D. (2010). A comparison of biased simulation schemes for stochastic volatility models. *Quantitative Finance*, 10(2), 177–194.

---

## 22. Change log

| Version | Date | Change | Reason | Main result already viewed? |
|---|---|---|---|---|
| 1.0 | 2026-11-03 | Initial spec | Consolidates proposal, instructor feedback, and data contract v0.2. Pass rules were spot-checked on a throwaway prototype using test parameters, not market data | Owner to confirm (OD-6) |
| 1.1 | 2026-11-08 | §10.1 and §10.2 use plain functions and dictionaries instead of dataclasses; references renamed to `DATA_CONTRACT.md` | Owner request: the owner must be able to read and explain the code. API representation only; no pass rule, tolerance, seed, filter, or default changed | Not applicable: no main result is affected by this change |
| 1.2 | 2026-11-08 | Optional n-asset EXT-A with worst-of and basket calls and puts (§2.2, §4.1, §8.2, §10.1, §16.2, §17.2); visualization plan (§12.9); agent commits, pushes, and market runs through committed scripts (§5.2, §15); OD-1 and OD-3 resolved | Owner instructions. No pass rule, tolerance, seed, filter, or default changed | No result under this spec had been viewed. Market fits and prices viewed on 2026-10-18 predate v1.0 (`AI_USE.md` Entries 5 to 11; OD-6) |
