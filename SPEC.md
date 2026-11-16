# SPEC.md — Heston LSMC Pricer: Out-of-Sample Accuracy on GOOG Options

| | |
|---|---|
| **Project** | ENSF 620 Fall 2026 final project (individual) |
| **Owner** | Jared Crebo |
| **Spec version** | 1.13 (2026-11-15) |
| **Companion document** | `DATA_CONTRACT.md` v0.9, whose §3.3 matches §15 of this spec |
| **Audience** | Engineers and AI coding agents writing implementation code and automated tests |

The key words MUST, MUST NOT, SHOULD, and MAY are normative. Requirements are binding until changed through §1.3.

---

## 1. Purpose, status, and precedence

### 1.1 What this document is

This spec defines what must be true of the pricing system and how each requirement is verified. It carries project decisions across chat sessions. It records requirements, not implementation status: nothing here asserts that any component already exists or passes.

### 1.2 Precedence

When documents conflict, the higher item wins:

1. ENSF 620 course documents and the instructor's written direction, including the proposal feedback.
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

### 2.1 Question and required core (CORE)

**Primary question.** How accurately does a Heston model price options contracts within its calibrated sample data and outside its calibrated sample data?

**How the project answers it.**

1. Calibrate the five Heston parameters to GOOG's option surface of 2026-10-13 (§6).
2. Price every contract in the test population (§2.2) as an American option with the LSMC engine, with Monte Carlo uncertainty bounds on each price (§8.5).
3. Compare the model prices with the market's bid, ask, and mid, and report aggregate accuracy (§8.5). The same test is run with the published 2015 Google parameters as a comparison model.

CORE is the committed result. Its required components are:

- Heston semi-analytic European pricer, used by calibration.
- Quote filtering, implied-volatility conversion, and single-asset calibration.
- Python/NumPy Monte Carlo simulator.
- Two-pass Least-Squares Monte Carlo (LSMC) pricer for puts and calls.
- Out-of-sample test script (§8.5).
- Raw-data validation (§11), verification suite, run provenance, and reviewer smoke path.

CORE MUST also run with published parameters and synthetic fixtures so that it does not depend on licensed data (§5.1).

Earlier versions of this spec asked other questions about other tickers. That history is kept in §22 and MUST be disclosed in the report. Since v1.13 the spec covers GOOG only.

### 2.2 Test population

A contract is in the test population if all of the following hold:

- It is a GOOG option in `data/goog_IV_Oct13.csv` with date 2026-10-13.
- It passes filters F1 to F8 (§6.2). Among them: 21 to 511 days to expiry, strike between 0.70 and 1.30 times the forward price `F`, out of the money, bid–ask spread at most 50% of mid, open interest at least 10, and a valid implied-volatility root.
- It traded that day: `volume > 0`.
- It is not in the calibration grid (§6.4). V-19 checks this.

The strike window is measured against the forward price, as filter F4 has always done. The test set was defined and viewed under that rule, so §1.3 does not allow changing it to spot now. In-the-money contracts are removed by F5 and are outside the claims. Calibration and test use the same snapshot, so this is a test across contracts on one day, not a test over time.

### 2.3 Out of scope

The system MUST NOT implement these within this project:

- C++ or other compiled pricing kernels. The Python/NumPy engine is the only engine.
- Contracts on more than one asset.
- Live data feeds or re-downloading data.
- P&L simulation, delta hedging, backtesting, trading signals, or paper trading.
- Stochastic rates, jumps, or discrete dividends in the simulated dynamics.
- Duality-based upper bounds.

### 2.4 Permitted claims

- The headline result is the out-of-sample accuracy of model prices for the test population on 2026-10-13 (§8.5), stated with its uncertainty and with the calibration status of the parameters used.
- Each model price is a **low-biased LSMC estimate** with a Monte Carlo standard error. It MUST NOT be described as a market price, a tradable quote, or a trading signal.
- A parameter set rejected under §6.7 MUST be labelled REJECTED wherever its results appear, together with its RMS implied-volatility residual and the 0.01 limit.

---

## 3. Conventions

| Item | Convention |
|---|---|
| Floating point | IEEE-754 `float64` throughout |
| Spot normalization | Pricing uses S(0) = 1. All reported values are fractions of initial spot. |
| Market quantities for comparison | Bid, ask, and mid divided by `stock_price_for_iv`, so they share the model's units |
| Time | Years, ACT/365 from snapshot date to expiration |
| Rates and yields | Decimal, continuously compounded |
| Volatility and variance | σ is annualized decimal volatility; ν is instantaneous variance |
| Standard error | `SE = s / sqrt(N)`, where `s` is the sample standard deviation with `ddof=1` over `N` independent evaluation paths |
| Paired SE | SE of the path-by-path difference on the same evaluation paths |
| Pooled result over R seeds | `mean(V_r)`, with `SE = sqrt(sum(SE_r^2)) / R` |
| 95% Monte Carlo interval | `[V - 1.96 SE, V + 1.96 SE]` on the pooled estimate. It covers simulation noise only, not LSMC low bias or discretization bias |
| Absolute prices | Absolute market prices MUST NOT appear in `results/`, logs, figures, or submitted files (§15). The published Google prices of 2015-02-02 in §8.4 are the only exception |

---

## 4. Contract and model

### 4.1 Contracts

With spot normalized to S(0) = 1 and normalized strike `K`, the payoff at exercise date `t_k` is:

```text
put:  g(t_k) = max(K - S(t_k), 0)
call: g(t_k) = max(S(t_k) - K, 0)
```

The project uses two kinds of contract:

| Use | Kind | Normalized strike `K` | Maturity `T` | Exercise dates `M` | Sub-steps `n_sub` |
|---|---|---|---|---|---|
| Reference contract: verification, sweeps, baselines | put | `1.00` | `1.00` year | `50` | `4` |
| Test contract (§8.5) | `call_put` of the quote | `strike / stock_price_for_iv` | ACT/365 from 2026-10-13 to expiration | `ceil(252 T)` | `1` |

- The exercise grid is `t_k = k T / M`, `k = 1..M`. There is no exercise at `t = 0`, and `t_M = T`.
- Test contracts approximate American exercise with about one exercise date per trading day. The gap to continuous exercise is a stated limitation.
- The corresponding European contract uses the same `K` and `T` and may be exercised only at `T`.

### 4.2 Risk-neutral dynamics

```text
dS  = (r - q) S dt + sqrt(nu) S dW^S
dnu = kappa (theta - nu) dt + xi sqrt(nu) dW^nu
d<W^S, W^nu> = rho dt
```

Parameter vector:

```text
phi = (nu0, kappa, theta, xi, rho)
```

`nu0` is **fitted**, not fixed. `r` and `q` are constants.

### 4.3 Rate and dividend inputs

These values live in `configs/market_inputs*.yaml`.

- **Rate.** Let `y` be the U.S. Treasury 1-Year CMT par yield on the snapshot date as a decimal. Then `r = 2 ln(1 + y / 2)`. For 2026-10-13, `y = 4.48%`.
- **Term structure.** The same flat `r` is used for implied-volatility conversion, calibration, and pricing. Ignoring the term structure is a stated limitation.
- **Dividend yield.** `q_GOOG = 0.0026` for 2026-10-13, from Alphabet's declared $0.22 quarterly dividend and a GOOG price near $345. It is approximate, which is a stated limitation.
- **Published parameters in the test.** When the 2015 parameters are used in §8.5, `r` and `q` are the 2026-10-13 values. Only the Heston parameters come from 2015.
- **2015 comparison (§8.4).** `r = 0.15%` and `q = 0`, as in the source paper.
- **Discrete dividends.** They are approximated by the continuous yield in both the tree and the simulation.

---

## 5. Input data and modes

### 5.1 Data modes

Every run MUST declare one mode.

| Mode | Parameter source | Purpose | Licensed data read |
|---|---|---|---|
| `synthetic` | Known parameters and `tests/fixtures/` | Tests, CI, reviewer smoke path | None |
| `published` | `configs/published_params.yaml` | Comparison model and known-answer checks | None |
| `market` | `results/calibration_goog/calibration.json` | Calibrated result and out-of-sample test | Read only by committed scripts (§15) |

Each published parameter set MUST include its citation and table or page. Every reported artifact MUST name its mode. A `published` result MUST NOT be described as calibrated to current GOOG options; its parameters were fitted to Google options on 2015-02-02.

### 5.2 Authority gate for market mode

`configs/governance.yaml` contains:

```yaml
market_use_authorized: false
basis: ""
recorded_on: null
```

- `market` runs MUST fail with `E_GOV_UNAUTHORIZED` while `market_use_authorized` is `false`.
- It may be set to `true` only by the owner, or by an agent on the owner's recorded instruction. `basis` must then state where the authority is documented.

### 5.3 Data sources

Schemas, classes, and paths are defined by `DATA_CONTRACT.md`; this spec does not duplicate them.

| Dataset | Used by | Notes |
|---|---|---|
| iVolatility Raw IV for GOOG, 2026-10-13 (`data/goog_IV_Oct13.csv`) | Calibration and the test population | Restricted. The only GOOG option date held, so it is the snapshot date |
| Massive minute bars for GOOG and GOOGL, November 2011 to July 2026 (`data/nvda_amd_intc_goog_minute/bars/`) | Raw-data validation only (§11) | Restricted. Not used by the reported analysis. **Pricing MUST NOT read Massive data** |
| U.S. Treasury CMT yields | `r` | Public |
| Published parameters and benchmarks | Comparison model and checks | Public |
| Synthetic fixtures | Tests, CI, smoke path | Public |

Files for other tickers stay on disk under `DATA_CONTRACT.md`; no reported run reads them.

---

## 6. Calibration

### 6.1 Pipeline

```text
validate raw data -> filter quotes -> convert to implied vol -> select grid
                  -> multi-start least squares -> stability check -> calibration.json
```

### 6.2 Quote filters

Filters are applied in this order, and each drop count is recorded. `S` is `stock_price_for_iv`, `mid = (bid + ask) / 2`, and `F = S exp((r - q) T)`.

| ID | Keep a row only if |
|---|---|
| F1 | It matches symbol GOOG and snapshot date 2026-10-13 |
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
3. If step 2 produced no contract point (defined below), add the surviving quote nearest the reference-contract strike at the selected expiry nearest the reference-contract maturity.

The grid MUST satisfy all of the following, or `E_SPARSE_GRID` is raised:

- At least 20 points.
- At least 3 expiries.
- A **contract point**: at least one point with tenor in `[0.70, 1.30]` and `|K/S - K_c| <= 0.025`, where `K_c` is the reference-contract strike in §4.1. The contract strike is a fraction of spot, not of the forward.

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
| 2 | Otherwise | Fix `kappa` to the cited value recorded in `configs/published_params.yaml` and refit four parameters; state it as an assumption. Accept it if it is stable and `RMS <= 0.01` |
| 3 | Step 2 still fails | No fit is accepted. The fit with the lowest RMS from steps 1 and 2 is recorded as the **rejected fit**, with its RMS, its stability result, and the 0.01 limit. It is still used for the out-of-sample test (§8.5) and the reference contract, and it is labelled REJECTED wherever its results appear (§2.4) |
| 4 | `E_SPARSE_GRID`, or no converged start | No market-calibrated result exists; only `published` mode is reported |

The 0.01 limit is unchanged. Step 3 was written after the GOOG fit's RMS of 0.01049 had been viewed, which §22 records and the report MUST disclose. Step 3 sets no "close enough" threshold, because a threshold chosen after seeing 0.01049 would be tuned to that value.

### 6.8 Synthetic surface fixture

`tests/fixtures/synthetic_raw_iv.csv` follows the `DATA_CONTRACT.md` schema, with these rules:

- The symbol is `SYN1`. `validate_raw.py --source fixtures` accepts `SYN1` to `SYN3` in place of real tickers.
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
Z^nu = rho Z^S + sqrt(1 - rho^2) Z^perp
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

The simulator returns `(S, nu)` at `t_0..t_M` only, with shape `(n_paths, M + 1, 1)`.

---

## 8. LSMC pricing

### 8.1 Algorithm

Use two independent path sets: regression set A and evaluation set B, each with `n_paths` paths. `g` is the put or call payoff of §4.1.

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

The variables are `(S/S0, nu_plus)` with total degree 3 by default. Basis size `p` is 6 at degree 2, 10 at degree 3, and 15 at degree 4. Variance MUST be part of the regression state.

### 8.3 Baselines for the reference contract

The reference contract reports two baselines:

1. **European put.** Both the Monte Carlo `V_E` and the semi-analytic price are reported. The premium uses the paired Monte Carlo difference, so simulation noise and discretization error are shared by both legs.
2. **Constant-volatility Bermudan.** `crr_price` with `n_steps = 50 M` and exercise at the same dates. `sigma` is the Black–Scholes implied volatility of the Heston European put at the contract `(K, T)`. It uses model output only, so it stays Internal. The gap to the Heston Bermudan is what stochastic volatility adds beyond matching the European level.

### 8.4 Engine check: Google puts of 2015-02-02

This check uses a put from the date and options the published parameters were fitted to. Prices come from Burkovska et al. (arXiv:1611.06452, Appendix A, Table 10), stored in `configs/google_2015_american_puts.csv`.

| Term | Value |
|---|---|
| Option | American put on Google, strike $525 |
| Valuation date | 2015-02-02 |
| Maturity | `T = 0.9507` years |
| Spot | $523.755 (`K/S = 1.00237707`) |
| Published market price | $46.05 |
| Rate and dividend | `r = 0.15%`, `q = 0`, as in the source paper |
| Exercise | `M = 240` daily dates with `n_sub = 1`, approximating American exercise |
| Parameters | Published Google set, fitted to these prices |

`scripts/compare_to_market.py` compares that contract and also prices all 401 published puts with `M = 50`, writing `results/market_comparison.json` and `results/market_comparison_2015.csv`. There is no pass rule. This comparison is **in-sample** and MUST be described that way. It shows that the engine reproduces a published fit; it does not answer §2.1.

### 8.5 Out-of-sample test on 2026-10-13 (headline result)

`scripts/test_goog_market.py` runs this test and writes `results/goog_market_test.json` (§12.6).

**Parameter sets.**

- (a) The 2026-10-13 calibration (§6, `configs/calibrate_goog.yaml`): the accepted fit, or the rejected fit under §6.7 step 3, labelled REJECTED.
- (b) The published 2015 Google set, as a comparison model. Its parameters are eleven years old, so it shows the cost of a stale calibration.

**Pricing.**

- Each contract in the test population (§2.2) is priced with `price_bermudan` as an American option, using the test-contract terms of §4.1.
- Profile: `n_paths = 100_000` per set, the five production seeds, basis degree 3.
- Paths MAY be simulated once per expiration and seed, and shared by every contract with that expiration. Their Monte Carlo errors are then correlated, which is stated.
- For each contract the script records the pooled price `V`, its `SE`, the 95% Monte Carlo interval (§3), the Monte Carlo European value, the premium, and the LSMC diagnostics.
- These per-contract values are written only to `data/derived/goog_test_prices.parquet` and are Restricted. Together with the contract identifiers, they would show which contracts traded and how each compares with its quote.

**Accuracy metrics**, for each parameter set, using market quantities normalized as in §3:

| Metric | Definition |
|---|---|
| Median relative price error | median of `|V - mid| / mid` |
| Inside the spread | share of contracts with `bid <= V <= ask` |
| Mid inside the Monte Carlo interval | share of contracts whose `mid` lies in `[V - 1.96 SE, V + 1.96 SE]` |
| Implied-volatility error | invert `V` with `american_implied_vol` (500 steps, as in §6.3); RMS and median of `|model IV - market IV|`, where market IV is the §6.3 value. Inversion failures are counted |

**Breakdowns.** Every metric is also reported by maturity (under 3 months, 3–6 months, 6–12 months, over 12 months), by moneyness `K/F` (0.70–0.90, 0.90–1.10, 1.10–1.30), and by kind (call or put). Only groups with `n >= 5` are reported.

**Uncertainty on the metrics.** A bootstrap resampling expirations rather than contracts, because contracts with the same expiration are correlated. 2000 resamples, seed `620200`, 95% percentile intervals.

**Diagnostic.** For parameter set (a), the metrics are also reported for the v1.12 shortcut, in which the model's European implied volatility is carried through the 500-step American tree. This shows whether pricing with LSMC changes accuracy.

There is no pass rule; the test is descriptive. Its stated limitations are:

- the same snapshot is used for calibration and test, and test contracts sit next to grid points, so the result is optimistic;
- LSMC low bias;
- daily exercise approximates American exercise;
- calibration targets are de-Americanized;
- `q` is approximate;
- in-the-money contracts are excluded.

### 8.6 Engine behaviour outside the source paper's range

`scripts/test_goog_out_of_range.py` prices GOOG puts with the published parameters at maturities of 0.08, 3, and 5 years and strikes of 0.4 to 1.8 times spot, outside the paper's data (about 0.48 to 1.68 times spot, 0.20 to 1.97 years). It writes `results/goog_out_of_range.json`. No market prices exist for these contracts, so each result is checked only for behaviour: the Monte Carlo European price within 3 SE of the semi-analytic price, the Bermudan price not below the European price, and not below intrinsic value, each within 3 SE. These results are model extrapolation and MUST NOT be described as validated against a market.

---

## 9. Precision, workload, and runtime

### 9.1 Run profiles

| Setting | Smoke | Production | Test pricing (§8.5) |
|---|---|---|---|
| `n_paths` per set | `20_000` | `100_000` | `100_000` |
| `M` | `50` | `50` | `ceil(252 T)` per contract |
| `n_sub` | `4` | `4` | `1` |
| Basis degree | 3 | 3 | 3 |
| Master seeds | `[620001]` | `[620001, 620002, 620003, 620004, 620005]` | Same as production |

One production seed on the reference contract performs 8 × 10⁷ state updates (two path sets, two state variables).

### 9.2 Precision targets

| ID | Applies to | Target | If missed |
|---|---|---|---|
| T1 | Reference contract, pooled | `SE(V_B) / V_B <= 0.005` | Raise `n_paths` within the runtime budget; the headline is not reported as meeting T1 otherwise |
| T2 | Reference contract, pooled | `SE(Premium) / Premium <= 0.10` | Raise `n_paths` within the budget; if still missed, set `premium_resolved = false` and report the 95% interval without a point claim |
| T3 | Test pricing, parameter set (a) | For at least 90% of test contracts, `1.96 SE <= 0.25 × (ask - bid) / S`: the Monte Carlo half-width is at most half the market's half-spread | Raise `n_paths` to `200_000` within B4; if still missed, report the share and mark T3 unmet |

T3 keeps Monte Carlo noise small next to the spread, so it cannot drive the inside-the-spread metric. A premium whose 95% interval includes zero is a valid finding and is reported that way.

**Path-reduction rule.** `n_paths` may be reduced for computing reasons only if T1 (reference contract) and T3 (test pricing) still hold. `n_paths` MUST NOT go below `20_000`.

### 9.3 Required sweeps

Sweeps use the reference contract with parameter set (a) and vary one factor from the production settings. Each configuration uses all production seeds.

| Factor | Values |
|---|---|
| `n_paths` | `20_000`, `50_000`, `100_000`, `200_000` |
| `n_sub` | `1`, `2`, `4`, `8` |
| `M`, at fixed `Δ = 1/200` | `10`, `25`, `50`, `100` |
| Basis degree | `2`, `3`, `4` |
| `r` | Base value, `+0.01`, `-0.01` |

The `M` sweep is a contract-convergence study, not a numerical convergence test of the 50-date contract.

### 9.4 Runtime budgets

| ID | Budget |
|---|---|
| B1 | One production pricing of the reference contract at one seed: `<= 30 s` median wall time over 5 repeats, single process |
| B2 | Full required sweep: `<= 6 h` |
| B3 | Reviewer smoke path from a clean clone: `<= 10 min` |
| B4 | Full test pricing (§8.5), both parameter sets, all seeds: `<= 2 h` |

The local baseline environment is recorded automatically in each manifest. Meeting these budgets in a single Python process is the computing rationale for the engine, and the report states the measured times.

---

## 10. Python API contract

The installable package is `hestonpricer`, located at `python/hestonpricer/`. Public names and semantics below are binding. Private helpers are not.

### 10.1 Data structures

The package uses plain functions and plain dictionaries. It MUST NOT define dataclasses or other classes for model inputs or results. The only classes in the package are the exceptions in §14.1, because Python exceptions must be classes.

The dictionaries below are binding: keys and meanings are part of the public API. Where a key has a default, a caller may leave it out.

```python
# Heston parameters. Functions take a list with one dict (single asset).
params = {"v0": 0.04, "kappa": 1.5, "theta": 0.04, "xi": 0.5, "rho": -0.7}

# Rate and dividend yield.
market = {"r": 0.04, "q": [0.0]}          # q is a one-entry list

# The contract. Defaults are the values shown.
contract = {"strike": 1.0, "maturity": 1.0, "n_exercise": 50, "kind": "put"}   # kind: "put" or "call"

# Simulation settings. n_paths is required; the other defaults are the values shown.
sim = {
    "n_paths": 100_000,
    "n_substeps": 4,
    "basis_degree": 3,
    "master_seed": 620001,
}

# Returned by simulate(). Arrays have shape (n_paths, M + 1, 1).
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
# params is a one-element list; corr is [[1.0]]. normals injects a shared fixture; rng is ignored when normals is given.

# lsmc.py
def fit_policy(paths, contract, market, basis_degree): ...          # returns a policy dict
def evaluate_policy(policy, paths, contract, market): ...           # returns a price_result dict
def price_bermudan(params, corr, market, contract, sim): ...        # returns a price_result dict

# surface.py
def filter_quotes(raw, cfg): ...      # raw is a DataFrame, cfg a dict; returns (DataFrame, dict of counts)
def select_grid(quotes, cfg): ...     # returns a DataFrame

# calibrate.py
def calibrate_asset(grid, r, q, cfg): ...                 # returns a dict with the §12.5 fields
```

### 10.3 Preconditions

Violations raise `InputError` (§14.1).

| Function | Requirement |
|---|---|
| `heston_european` | `s0, strike, maturity > 0`; `xi > 0`; `\|rho\| < 1`; `v0, kappa, theta > 0` |
| `simulate` | `xi >= 0`; `\|rho\| <= 1`; `kappa >= 0`; `v0, theta >= 0`; `corr == [[1.0]]`; `len(params) == len(market["q"]) == 1` |
| `fit_policy` | `basis_degree in {1, 2, 3, 4}` |
| All pricing functions | All numeric inputs finite; `contract["kind"]` is `"put"` or `"call"` |

---

## 11. CLI and workflow

```bash
python python/validate_raw.py --source {ivolatility|massive|fixtures}
python scripts/validate_goog_data.py [--iv-file <path>] [--minute-dir <path>]
python python/make_fixtures.py
python scripts/inspect_schema.py <path-under-data>
python scripts/run.py calibrate --config configs/<name>.yaml
python scripts/run.py price     --config configs/<name>.yaml [--results-root <dir>]
python scripts/run.py sweep     --config configs/<name>.yaml
python scripts/run.py compare   --config configs/<name>.yaml
python scripts/test_goog_market.py
python scripts/compare_to_market.py
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

- `inspect_schema.py` prints only file names, column names, dtypes, row counts, and null counts.
- `validate_goog_data.py` checks `data/goog_IV_Oct13.csv` and the GOOG rows of the Massive minute bars against every rule in `DATA_CONTRACT.md` §1.1 and §1.2. It writes `results/validation_goog.json` and prints only file and partition names, row counts, rule names, violation counts, row indices, and aggregate counts. It never prints a data value.
- `make_figures.py` MUST NOT read anything under `data/`.

`smoke.sh` MUST, using only Public material:

1. Regenerate the synthetic fixtures.
2. Calibrate to the synthetic surface.
3. Price the reference contract with the smoke profile, using the fixture's known parameters rather than the fitted ones, so that the comparison does not depend on solver differences across machines.
4. Compare the price with `results/smoke/expected.json` at relative tolerance `1e-8`.
5. Run the `fast` test set (§13) and return non-zero on any failure.

---

## 12. Artifact schemas

### 12.1 Run configuration

```yaml
schema_version: 1
run_name: core_goog_reference
reported: true
data_mode: market             # synthetic | published | market
assets: [GOOG]
snapshot_date: 2026-10-13
market_inputs: configs/market_inputs_goog_oct13.yaml
params_source: results/calibration_goog/calibration.json
contract: {strike: 1.0, maturity: 1.0, n_exercise: 50, kind: put}
simulation: {n_paths: 100000, n_substeps: 4, basis_degree: 3}
seeds: [620001, 620002, 620003, 620004, 620005]
```

Unknown keys MUST raise `E_CONFIG`. The config and market-inputs file names above are the required names for the reference-contract run under parameter set (a).

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
  "spec_version": "1.13",
  "data_contract_version": "0.9",
  "run_id": "core_goog_reference-1a2b3c4d",
  "git_commit": "<full hash>",
  "git_dirty": false,
  "command": "python scripts/run.py price --config configs/core_goog_reference.yaml",
  "config_sha256": "<hash>",
  "data_mode": "market",
  "seeds": [620001, 620002, 620003, 620004, 620005],
  "environment": {
    "python": "3.x.y",
    "packages": {"numpy": "x.y.z", "scipy": "x.y.z", "pandas": "x.y.z"},
    "os": "...", "cpu": "...", "cores": 0, "ram_gb": 0
  },
  "inputs": [
    {"dataset": "ivolatility_raw_iv", "tickers": ["GOOG"], "snapshot_date": "2026-10-13",
     "file_name": "goog_IV_Oct13.csv", "row_count": 0}
  ],
  "outputs": [{"path": "result.json", "sha256": "<hash>"}],
  "started_at": "<UTC ISO-8601>",
  "finished_at": "<UTC ISO-8601>",
  "wall_seconds": 0.0
}
```

Input records contain identifiers and counts only, never data values. Output hashes apply to Public and Internal files only.

### 12.4 `result.json` (reference contract)

```json
{
  "schema_version": 1,
  "run_id": "core_goog_reference-1a2b3c4d",
  "data_mode": "market",
  "parameter_status": "rejected",
  "units": "fraction_of_initial_spot",
  "assets": ["GOOG"],
  "contract": {"strike": 1.0, "maturity": 1.0, "n_exercise": 50, "kind": "put"},
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
    "feller_ratio": 0.0
  },
  "precision": {"T1": true, "T2": true}
}
```

`parameter_status` is `accepted`, `rejected`, or `published`.

### 12.5 `results/calibration_goog/calibration.json`

```json
{
  "schema_version": 1,
  "data_mode": "market",
  "snapshot_date": "2026-10-13",
  "r": 0.0,
  "assets": {
    "GOOG": {
      "q": 0.0026,
      "ladder_step": 1,
      "params": {"v0": 0.0, "kappa": 0.0, "theta": 0.0, "xi": 0.0, "rho": 0.0},
      "fixed_params": [],
      "rejected_fit": null,
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

At ladder step 3, `params` is `null` and `rejected_fit` holds `{"from_step", "params", "fixed_params", "rms_iv_residual", "stable", "limit": 0.01}`. `residual_buckets` MUST contain only buckets with `n >= 5`. Per-contract residuals are prohibited.

### 12.6 `results/goog_market_test.json`

```json
{
  "schema_version": 1,
  "spec_version": "1.13",
  "symbol": "GOOG",
  "snapshot_date": "2026-10-13",
  "units": "fraction_of_initial_spot",
  "test_set": {
    "n_after_filters": 0, "n_in_grid_removed": 0, "n_zero_volume_removed": 0,
    "n_test": 0, "n_calls": 0, "n_puts": 0, "n_expirations": 0, "grid_overlap": 0
  },
  "pricing": {"engine": "lsmc", "n_paths": 100000, "seeds": [], "n_substeps": 1,
              "exercise": "ceil(252 T) dates", "basis_degree": 3},
  "parameter_sets": {
    "calibrated_2026_10_13": {"status": "rejected", "rms_iv_residual": 0.0, "limit": 0.01},
    "published_2015": {"status": "published", "source": "Burkovska et al., Table 8"}
  },
  "results": {
    "<parameter set>": {
      "overall": {"n": 0, "median_rel_price_error": 0.0, "share_inside_spread": 0.0,
                  "share_mid_inside_mc_ci95": 0.0, "rms_iv_error": 0.0,
                  "median_abs_iv_error": 0.0, "n_iv_inversion_failed": 0,
                  "ci95": {"<metric>": [0.0, 0.0]}},
      "by_maturity": [{"bucket": "...", "n": 5, "<metric>": 0.0}],
      "by_moneyness": [{"bucket": "...", "n": 5, "<metric>": 0.0}],
      "by_kind": [{"bucket": "call", "n": 5, "<metric>": 0.0}]
    }
  },
  "tree_shortcut": {"overall": {}, "by_maturity": []},
  "precision": {"T3_share": 0.0, "T3_met": true, "median_rel_se": 0.0,
                "n_paths_used": 100000}
}
```

Only aggregates appear. Every group has `n >= 5`.

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
| Out-of-sample accuracy with uncertainty, by group | `results/goog_market_test.json` |
| Verification table | `results/verification_table.csv` |
| Raw-data validation report | `results/validation_goog.json` |
| Convergence plots in paths, sub-steps, exercise dates, and basis degree | §9.3 sweeps |
| Fitted surface versus known surface | Synthetic fixture only |
| Market fit quality | Aggregate statistics from `calibration.json`; no market data points plotted |
| Reference-contract price, European value, and premium with uncertainty | `result.json` |
| Seed robustness table | `per_seed` |
| Test and CI logs, run manifests | CI and `results/runs/` |

Every principal figure or table MUST state:

- Its conclusion, units, sample or workload, repetitions, uncertainty, exclusions, source run, and generating code.
- A readable encoding that does not rely on colour alone.

### 12.9 Visualization plan for the report and presentation

This is the plan for the figures in the report and the eight-minute presentation. It is a requirement for G3, not a statement that any figure exists. Each figure MUST follow §12.8 and have an evidence-map row. All are generated by `scripts/make_figures.py`, which MUST NOT read anything under `data/`.

| ID | Figure | Source | Class | Report | Slides |
|---|---|---|---|---|---|
| VZ-1b | PyVista 3D surface of the GOOG 2015 Heston model: the published Google parameters fitted on 2015-02-02 | Parameters in the comparison summary | Public | Yes | Optional |
| VZ-1e | PyVista 3D surface of the 2026-10-13 GOOG model, labelled REJECTED at ladder step 3 | `results/calibration_goog/calibration.json` | Internal | Yes | Yes |
| VZ-2 | Monte Carlo European vs. semi-analytic prices over the V-03 grid, with the difference in SE | Comparison run | Internal | Yes | Yes |
| VZ-3 | LSMC Bermudan vs. CRR tree on the V-04 cases | Comparison run | Public | Yes | Yes |
| VZ-4 | Heston Bermudan, European, and constant-volatility Bermudan across strikes, parameter set (a) | Comparison run | Internal | Yes | No |
| VZ-5 | Convergence panels in paths, sub-steps, exercise dates, and basis degree | §9.3 sweeps | Internal | Yes | One panel |
| VZ-6 | Reference-contract price, European value, and premium with 95% intervals | `result.json` | Internal | Yes | No |
| VZ-7 | Fitted vs. known surface on the synthetic fixture | `results/smoke/calibration.json` | Public | Yes | No |
| VZ-8 | Seed robustness table and aggregate fit-quality table | `per_seed`; `residual_buckets` | Internal | Yes | No |
| VZ-9 | Simulated prices vs. the 401 published Google puts of 2015-02-02, labelled in-sample | `results/market_comparison_2015.csv` | Public | Yes | Optional |
| VZ-10 | Out-of-sample implied-volatility error by maturity and by moneyness, parameter sets (a) and (b), with bootstrap 95% intervals | `results/goog_market_test.json` | Internal | Yes | Yes |
| VZ-11 | Share of model prices inside the spread, and share of mids inside the Monte Carlo interval, by maturity and moneyness | `results/goog_market_test.json` | Internal | Yes | Yes |
| VZ-12 | Monte Carlo half-width relative to market half-spread (T3), as a binned histogram with every bin `n >= 5` | `results/goog_market_test.json` | Internal | Yes | No |
| VZ-13 | LSMC pricing vs. the tree shortcut, accuracy by maturity | `results/goog_market_test.json` | Internal | Yes | No |

**VZ-1 requirements.**

- Evaluate `sigma_Heston` with `model_implied_vols` on a dense regular grid, using the stated parameters only. VZ-1e covers `K/F` in `[0.70, 1.30]` and tenor in `[0.06, 1.40]` years, the test population's range. VZ-1b covers `K/F` in `[0.80, 1.20]` and tenor in `[0.20, 1.97]`, the source paper's range, with `r = 0.15%` and `q = 0`.
- VZ-1e MUST carry the word REJECTED and the failed limit in the figure itself if the fit is from ladder step 3.
- VZ-1b MUST state that its parameters date from 2015-02-02 and that it is not a surface for today's market.
- Show `K/F`, tenor, and implied volatility as labelled axes, with height encoding implied volatility and a single-hue colour ramp with a labelled scale.
- Render off-screen with PyVista to a PNG for the report and slides.
- It MUST NOT overlay market implied-volatility points, quotes, or per-contract residuals (`DATA_CONTRACT.md` §4.3).

**Comparison runs.** VZ-2, VZ-3, and VZ-4 use `scripts/run.py compare`, which writes Internal summaries under `results/comparisons/`. Numerical-versus-analytical figures test the engine on known answers and do not show that Heston fits GOOG (§16).

**Presentation order.** Question and test population; data boundary; pipeline; VZ-3 and VZ-2 as engine checks; VZ-1e; VZ-10 and VZ-11 as the result; limitations.

**Implementation.** The report's conclusions and narrative are written by the owner.

---

## 13. Repository contract

```text
.
├── SPEC.md  README.md  AI_USE.md  LICENSE  DATA_CONTRACT.md
├── pyproject.toml  <pinned lock file>
├── .gitignore  .claude/settings.json  .github/workflows/
├── configs/        governance.yaml, market_inputs*.yaml, published_params.yaml, one YAML per reported run
├── python/         validate_raw.py, make_fixtures.py, hestonpricer/
├── scripts/        run.py, verify_outputs.py, make_figures.py, inspect_schema.py, validate_goog_data.py,
│                   test_goog_market.py, smoke.sh, check_no_data.sh
├── tests/          test_*.py, reference_values.yaml, reference_impl.py, fixtures/
├── results/        calibration_goog/, runs/, figures/, smoke/, goog_market_test.json,
│                   validation_goog.json, evidence_map.csv, verification_table.csv
├── prototype/      exploratory notebooks with outputs cleared; not part of the assessed workflow
└── data/           Git-ignored; Restricted; never in a commit or submission
```

**Tests.** Each check in §16 MUST have a test whose name starts with its ID, such as `test_V03_european_mc_grid`. Statistical tests use fixed seeds. Market-dependent checks are skipped in CI and report `NOT_RUN`, not `PASS`.

Tests use one of two pytest markers:

- `fast`: V-01, V-02, V-05, V-07, V-08, V-12, part (a) of V-14, and the synthetic-file tests of `validate_goog_data.py`, using the smoke profile where paths are needed. These run within B3.
- `slow`: every other check. The owner or an agent runs these, and `verify_outputs.py` records them from completed runs.

**CI.** Every push runs linting, the `fast` test set, `smoke.sh`, and `check_no_data.sh`. CI has no market data and no data-provider secrets.

**README.** It MUST cover the question, scope, test population, data boundary, setup, inputs, run and verify commands, outputs, limitations, and exactly what a reviewer can and cannot reproduce.

**Reviewer boundary.** A reviewer can reproduce the full pipeline on Public fixtures and `published` mode. A reviewer cannot rerun `market` mode or the out-of-sample test. For those results, the reviewer inspects committed code, configs, manifests, Internal outputs, and evidence maps.

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
| `E_SPARSE_GRID` | §6.4 requirements not met | Apply ladder step 4 | 3 |
| `E_CALIB_FAILED` | No converged start | Apply ladder step 2, then steps 3 and 4 | 3 |
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
| No in-the-money paths at a date, or fewer than `n_min` | Skip exercise at that date; count it. Short, far out-of-the-money test contracts may skip most dates |
| Rank-deficient regression | Use the minimum-norm solution; count it; do not raise |
| All values zero, so SE is 0 | A relative target is undefined: report it unmet; comparison checks use `max(SE, 1e-12)` |
| Quote `mid` differs from `mean_price` | Use computed `mid`; count rows differing by more than `1e-6` |
| No implied-volatility root | Drop under F8; count it |
| Model implied volatility not invertible during fit | Residual `1.0`; count it |
| LSMC test price cannot be inverted to an implied volatility | Count it; exclude it from the implied-volatility metrics only |
| `maturity` or `strike` passed as an integer | Cast to `float64` at the API boundary. Time and payoff arrays MUST NOT have an integer dtype, because `np.full(n, 1)` silently truncates later exercise times to 0 and removes discounting |
| Re-run with a completed `run_id` | Do not overwrite; exit with a clear message, unless `--results-root` names a new location |
| Run interrupted | Treat the run directory as incomplete; V-15 fails until re-run |

---

## 15. Agent and governance rules

`DATA_CONTRACT.md` governs data handling. These rules apply to every AI agent on the project.

**Licensed-data rule.** Any script committed to the repository MAY read `data/`, provided that:

- its output (terminal, logs, and files outside `data/`) is aggregate and Internal under `DATA_CONTRACT.md` §4, and it never prints or logs a row, a quote, an IV point, or a per-contract result;
- the owner has reviewed the script, or the change to it, before an agent runs it on licensed data.

An agent MAY run such scripts and see their aggregate output. An agent MUST NOT read `data/` itself.

An agent MUST NOT:

- Read, print, plot, paste, sample, or summarize at row level any value under `data/`, by any means: file tools, `cat`, `head`, `zcat`, DataFrame previews, uncommitted or ad hoc scripts, or one-liners. Structure is available only through `scripts/inspect_schema.py`.
- Set `market_use_authorized` or write `configs/governance.yaml` without the owner's recorded instruction.
- Hand-edit `results/**`. Those files are written only by pipeline scripts.
- Generate, recall, or fill in reference values in `tests/reference_values.yaml` or `configs/published_params.yaml`. The owner transcribes them from the cited source and records the table or page. Two exceptions, both on the owner's instruction of 2026-11-08: the agent entered the set in `configs/published_params.yaml` from Burkovska et al. (arXiv:1611.06452, Table 8), and the V-01 and V-06 values in `tests/reference_values.yaml`. The owner remains responsible for checking them against the sources.
- Write the report, slides, or claims. Reported results and figures come only from pipeline scripts.
- Change a pass rule, threshold, seed, or data filter (§1.3).
- Access credentials or commit anything under `data/`.
- Create misleading or artificial Git history.

An agent MAY:

- Write code, tests, fixtures, and documentation. Owner review may follow the commit, except for the pre-run review of data-reading scripts above.
- Commit and push when the owner asks, but only if `data/` is listed in `.gitignore` and `scripts/check_no_data.sh` passes immediately before the push.
- Run tests and pipelines in `synthetic` and `published` modes.
- Run committed, reviewed scripts that read `data/` under the licensed-data rule, and run `scripts/make_figures.py`.
- Read Public and Internal files.

Debug data problems on fixtures first. If a problem occurs only on licensed data, describe it by file name, row index, column, and rule.

`AI_USE.md` MUST record the tools and access mode, their roles, what was checked and how, important rejected or corrected suggestions, the data boundary, and the residual limitations that remain after those checks.

---

## 16. Verification matrix

Reference values in V-01 and V-06 MUST come from `tests/reference_values.yaml`, with citation and table or page. All MC-based rules use the fixed test seed `620100` unless stated otherwise. V-09, V-10, and V-11 run on the reference contract with parameter set (a). Unless a rule says otherwise, Monte Carlo checks use the production profile.

The report and README MUST state what the checks establish and what they do not. For example, V-03 and V-06 test the engine on known answers. They do not show that Heston is the right model for GOOG.

| ID | Check | Pass rule | Origin |
|---|---|---|---|
| V-00 | Raw-data validation: `scripts/validate_goog_data.py` on the GOOG iVolatility file and GOOG minute bars | Zero violations of every `DATA_CONTRACT.md` rule, apart from exceptions recorded in that contract | v1.13 |
| V-01 | Semi-analytic Heston European vs. published benchmark | Absolute difference `<= max(1e-6, half the last published digit)` | New |
| V-02 | Put–call parity; small-`xi` limit (`xi = 1e-4`, `rho = 0`, `v0 = theta`) vs. Black–Scholes | Parity residual `<= 1e-10`; BS difference `<= 1e-6` | New |
| V-03 | Heston European MC vs. analytic, `T` in `{0.25, 0.5, 1, 1.5, 2}` × 8 evenly spaced `K/S` in `[0.80, 1.25]`, `Δ = 1/400`, `200_000` paths | At least 38 of 40 options within 3 SE | Proposal V1 |
| V-04 | GBM reduction (`xi = 0`, `v0 = theta`): LSMC Bermudan vs. 5,000-step CRR tree on the same exercise dates, for `(sigma, K)` in `{(0.3, 1.0), (0.5, 1.0), (0.5, 0.9), (0.3, 1.1)}`, for puts and for calls with `q = 0.0026` | Difference `<= 3 SE` for each case | New; calls added v1.13 |
| V-05 | `M = 1` vs. semi-analytic European | Difference `<= 3 SE` | Proposal V4 |
| V-06 | Heston Bermudan vs. Ikonen–Toivanen American benchmark `B` at each published spot, `M` in `{25, 50, 100}`, `200_000` paths | At `M = 100`: `\|V - B\| <= max(0.01 B, 3 SE)` and `V - B <= 3 SE`. For each rise in `M`: `V_low - V_high <= 3 sqrt(SE_low^2 + SE_high^2)` | New |
| V-07 | Vectorized LSMC vs. independent scalar implementation in `tests/reference_impl.py` on `tests/fixtures/shared_normals.npz` | Relative difference `<= 1e-10` | Proposal V7 |
| V-08 | Repeat run with same config, seed, commit, and environment, in a new `--results-root` | Numeric fields of `result.json` identical, excluding `wall_seconds` | New |
| V-09 | Basis degree 2 vs. 3, and 3 vs. 4, on identical paths | Pooled price difference `<= 2 SE` of the higher-degree run | Proposal risk table |
| V-10 | `n_sub = 4` vs. `8`, pooled | `\|V_4 - V_8\| <= 3 sqrt(SE_4^2 + SE_8^2)`; if not, `8` becomes the production setting | New |
| V-11 | All pairs of production seeds | `\|V_a - V_b\| <= 3 sqrt(SE_a^2 + SE_b^2)` | Proposal V8, corrected |
| V-12 | Calibration on synthetic surface with known parameters | RMS IV `<= 1e-4`; `v0, kappa, theta, xi` within 5% relative; `rho` within `0.05` absolute | Proposal V2, extended |
| V-13 | Calibration on the GOOG 2026-10-13 grid (`market` mode) | RMS IV `<= 0.01` and stable under §6.6 | Proposal V3; GOOG since v1.13 |
| V-14 | Implied-volatility conversion | (a) Tree-inversion round trip recovers `sigma` within `1e-4`; (b) GOOG file: for near-the-money pairs with `\|K/F - 1\| <= 0.05`, at least 90% of call and put IVs agree within their bid–ask IV bands; (c) GOOG file: median `\|own IV - vendor IV\| <= 0.02` | New; GOOG since v1.13 |
| V-15 | Output completeness | Every configured run is complete, all values finite, and manifest hashes match | Proposal V9 |
| V-16 | Governance scan | `git log --all --name-only -- data/` is empty; pre-commit hook active; no prohibited fields or secrets in tracked files or submission package | New |
| V-17 | Clean-clone reproduction | `smoke.sh` passes from a fresh clone within B3 | Feedback, rubric |
| V-18 | Traceability | Every evidence-map row resolves; `make_figures.py` regenerates all figures | Rubric |
| V-19 | Test set disjoint from the calibration grid | No `option_symbol` appears in both; both counts recorded | v1.13 |

---

## 17. Gates and fallbacks

### 17.1 Gates

| Gate | Date | Requires | On failure |
|---|---|---|---|
| G0 | Nov 3 | Reviewable repository; method and data boundary documented; smoke test passes from a clean clone | Record the gaps; revise the method or evidence plan |
| G1 | Nov 14 | V-01 to V-12 and V-15 pass; B1 and B2 pass; §5.2 resolved | Freeze the engine; no new engine features |
| G2 | Nov 20 | V-00, V-04 with calls, and V-19 pass; out-of-sample test (§8.5) run and frozen; V-13 and V-14 recorded; T3 resolved | Freeze the minimum defensible analysis |
| G3 | Nov 27 | Draft report and slides; figures regenerate; every claim maps to evidence | Remove unsupported claims |
| G4 | Dec 1 | Eight-minute presentation finishes within the assigned slot | Cut content, not the verification or limitations slides |
| G5 | Dec 3 | Route A package verified; all CORE checks recorded | Submit the verified minimal package before extras |
| G6 | Dec 4, 11:59 p.m. | Report and repository uploaded to D2L; exact uploaded files opened and checked | Re-upload before the deadline |

### 17.2 Fallback rules

| Trigger | Action |
|---|---|
| OD-1 is unresolved or negative | `published` mode becomes the reported result; revise the question and claims to match |
| Calibration fails | Apply §6.7 |
| V-00 finds violations | Stop licensed-data runs until each violation is explained; record any accepted exception in `DATA_CONTRACT.md` with a §22 row |
| V-09 fails | Add basis terms or features; report any residual bias |
| V-10 fails at `n_sub = 8` | Report discretization error as unresolved; do not claim a converged price |
| B4 fails | Reduce paths only under §9.2; otherwise share paths across a contract's expiration (§8.5) |
| Restricted data leak | Stop licensed-data work. If it reached Git history, contact the instructor immediately |

---

## 18. Definition of done

CORE is complete when all of the following hold:

- [ ] All §16 checks are recorded in `results/verification_table.csv`, with no unexplained `FAIL` or `NOT_RUN`.
- [ ] The out-of-sample test (§8.5) is complete for both parameter sets, with bootstrap intervals and a resolved T3.
- [ ] The reference-contract result has T1 met and a resolved or explicitly unresolved T2.
- [ ] The headline accuracy figures, their uncertainty, the test population, and the parameter status appear consistently in report, slides, and README.
- [ ] Every principal figure and table has an evidence-map row and regenerates from committed Internal outputs.
- [ ] Limitations cover at least: same-day test with neighbouring contracts, LSMC low bias, discretization bias, daily exercise as an approximation to American exercise, de-Americanized calibration targets, flat `r`, approximate `q`, Heston's lack of jumps and earnings effects, a single snapshot, excluded in-the-money contracts, and the ladder step used.
- [ ] Data availability, authority, AI use, and residual AI limitations are disclosed, including the §22 rows marked as viewed.
- [ ] Both §8.3 baselines are reported, and the report states what the checks do and do not establish.
- [ ] `v1.0-final` exists, and the Route A package passes §13.

---

## 19. Open decisions

Each item needs an owner decision. Defaults apply until changed through §1.3.

| ID | Priority | Decision | Default in this spec |
|---|---|---|---|
| OD-1 | Resolved 2026-11-08 | The iVolatility licence allows every use in `DATA_CONTRACT.md` provided the raw data is not published, and the owner holds written confirmation that derived results may be used for this project (`DATA_CONTRACT.md` §2.2) | `market_use_authorized: true` on the owner's instruction |
| OD-2 | Resolved 2026-11-13 | The project uses GOOG only | GOOG |
| OD-3 | Resolved 2026-11-08 | Treasury 1-Year CMT yields from the official table: 4.48% on 2026-10-13. The proposal's fixed `r = 5%` is withdrawn | Values in `configs/market_inputs*.yaml` |
| OD-4 | High | Compare `DATA_CONTRACT.md` §1.2 with the actual iVolatility header, and confirm that `stock_price_for_iv` is synchronous with the quotes. The test's normalized quantities depend on it. `validate_goog_data.py` reports aggregate counts that inform this | Contract schema, unverified |
| OD-5 | Resolved 2026-11-08 | The published parameter set is the Google 2015-02-02 set in Burkovska et al. (arXiv:1611.06452, Table 8, de-Americanization with closed-form solution), and its `kappa` is used for §6.7 step 2 | Values in `configs/published_params.yaml` |
| OD-6 | Medium | Confirm the reference-contract terms `K = 1.00`, `T = 1.00`, `M = 50` | Values in §4.1 |
| OD-7 | Medium | The proposal barred repository write access. The owner now allows agent writes, commits, and pushes, and the running of reviewed data-reading scripts (§15). Disclose this change in `AI_USE.md` and the report | Allowed under §15 |
| OD-8 | Medium | Raw-file hashes were removed from the data contract. Traceability currently relies on file name and row count. Hashes are not values and would strengthen the provenance evidence | No raw-file hashes |
| OD-9 | Low | Choose the code licence for `LICENSE` | Unset |
| OD-12 | Medium | Whether to add a constant-volatility baseline model to the out-of-sample test, so Heston is compared with something simpler than itself, not only with stale parameters | Not included |

---

## 20. Traceability

### 20.1 Proposal decisions superseded

| Proposal | This spec | Reason |
|---|---|---|
| Three-asset worst-of is the primary target | Single-asset GOOG study; multi-asset removed | Instructor feedback; owner decision (v1.11, v1.13) |
| C++ pricing engine | Removed; Python only | Instructor feedback; owner decision (v1.13) |
| OpenMP tier and HPC mention | Removed | Instructor feedback |
| Commit raw market snapshot; publish later | Never committed or submitted | Feedback; `DATA_CONTRACT.md` |
| Fitted surface vs. market quotes figure | Synthetic surface plus aggregate market statistics | `DATA_CONTRACT.md` |
| `r = 5%`, all `q = 0` | Treasury-derived `r`; GOOG dividend yield | Market-consistent inputs |
| Calibrate directly to BS IV from listed quotes | OTM quotes, de-Americanized IVs | Listed single-stock options are American-style |
| Price one calibrated contract | Out-of-sample accuracy over held-out GOOG contracts | Owner decision (v1.13) |
| Calibration-grid robustness check (E7, stretch) | Removed | Stretch work |
| Same seeds "agree within 1 SE" | `3 sqrt(SE_a^2 + SE_b^2)` | A 1-SE rule on two independent estimates would fail about half the time even for a correct engine |

### 20.2 Verification ID crosswalk

| Proposal Table 1 | This spec |
|---|---|
| V1 | V-03 |
| V2 | V-12 |
| V3 | V-13 |
| V4 | V-05 |
| V5, V6 | Removed with the multi-asset scope |
| V7 | V-07 |
| V8 | V-11 |
| V9 | V-15 |

Proposal Tables 2, 3, A2, and A3 and its evidence list cite V10–V14, which Table 1 does not define. Those citations are retired.

### 20.3 Instructor feedback coverage

| Feedback item | Addressed in |
|---|---|
| Single-asset study is the committed result | §2.1 |
| Stay in Python unless a measured bottleneck justifies C++ | §2.3, §9.4 |
| Cut OpenMP | §2.3 |
| Attempt three assets only after accuracy and runtime gates | Not pursued (§2.3) |
| Provider and retention, commit, and publication terms | §5.2, §5.3, OD-1, `DATA_CONTRACT.md` |
| One Heston parameterization, including initial variance | §4.2 |
| Variance discretization | §7.1 |
| LSMC basis and regression controls | §8 |
| Paths, steps, repetitions, MC error target, runtime | §9 |
| Pass rules fixed before viewing main results | §1.3, §16, §22 |
| Residual AI limitation was absent | §15, §18 |
| Minimum precision rule for reducing paths | §9.2 |
| Revise question and claims if three assets are not completed | §2.1 |

---

## 21. References

Citations follow the proposal unless marked.

- Albrecher, H., Mayer, P., Schoutens, W., & Tistaert, J. (2007). The little Heston trap. *Wilmott Magazine*, January, 83–92.
- Burkovska, O., Glau, K., Mahlstedt, M., & Wohlmuth, B. Model reduction for calibration of American options. arXiv:1611.06452. Source of the published parameter set and the ladder-step-2 `kappa`.
- Burkovska, E., et al. (2018). Calibration to American options: numerical investigation of the de-Americanization method. *Quantitative Finance*. Added here; verify the full citation before use.
- Chan, T. L. Singular Fourier-Padé series expansion of European option prices. arXiv:1706.06709. Table 10 reproduces the Fang and Oosterlee Heston values used in V-01.
- Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). Option pricing: A simplified approach. *Journal of Financial Economics*, 7(3), 229–263.
- Fang, F., & Oosterlee, C. W. (2008). A novel pricing method for European options based on Fourier-cosine series expansions. *SIAM Journal on Scientific Computing*, 31(2), 826–848.
- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer.
- Haentjens, T., & in 't Hout, K. J. (2015). ADI schemes for pricing American options under the Heston model. *Applied Mathematical Finance*. arXiv:1309.0110. Table 2 tabulates the V-06 values of Ikonen and Toivanen.
- Heston, S. L. (1993). A closed-form solution for options with stochastic volatility with applications to bond and currency options. *The Review of Financial Studies*, 6(2), 327–343.
- Longstaff, F. A., & Schwartz, E. S. (2001). Valuing American options by simulation: A simple least-squares approach. *The Review of Financial Studies*, 14(1), 113–147.
- Lord, R., Koekkoek, R., & van Dijk, D. (2010). A comparison of biased simulation schemes for stochastic volatility models. *Quantitative Finance*, 10(2), 177–194.

---

## 22. Change log

Rows before 1.13 describe earlier scopes, including the AMD, NVDA, and INTC work, and are kept unchanged as the record that §1.3 requires.

| Version | Date | Change | Reason | Main result already viewed? |
|---|---|---|---|---|
| 1.0 | 2026-11-03 | Initial spec | Consolidates proposal, instructor feedback, and data contract v0.2. Pass rules were spot-checked on a throwaway prototype using test parameters, not market data | Owner to confirm (OD-6) |
| 1.1 | 2026-11-08 | §10.1 and §10.2 use plain functions and dictionaries instead of dataclasses; references renamed to `DATA_CONTRACT.md` | Owner request: the owner must be able to read and explain the code. API representation only; no pass rule, tolerance, seed, filter, or default changed | Not applicable: no main result is affected by this change |
| 1.2 | 2026-11-08 | Optional n-asset EXT-A with worst-of and basket calls and puts (§2.2, §4.1, §8.2, §10.1, §16.2, §17.2); visualization plan (§12.9); agent commits, pushes, and market runs through committed scripts (§5.2, §15); OD-1 and OD-3 resolved | Owner instructions. No pass rule, tolerance, seed, filter, or default changed | No result under this spec had been viewed. Market fits and prices viewed on 2026-10-18 predate v1.0 (`AI_USE.md` Entries 5 to 11; OD-6) |
| 1.3 | 2026-11-08 | Follows `DATA_CONTRACT.md` v0.4, which relaxes the raw-data rule on `expiration` from `> date` to `>= date` | Owner decision after validation flagged 518 rows. Disclosed because the rule changed in response to a failed check. Same-day expiries are normal listed contracts, and filter F3 is unchanged, so the quotes used for calibration are unchanged | No market-calibrated result had been produced under this spec. Earlier fits are noted in the 1.2 row |
| 1.4 | 2026-11-08 | OD-5 resolved; §15 exception for the agent entering the published set on the owner's instruction; reference added to §21 | The full five-parameter market fit missed the RMS limit (ladder step 3), and no cited `kappa` existed. The 0.01 limit and all rules are unchanged | Yes. The rejected fit's residual and parameters were viewed before this `kappa` was chosen; it MUST be disclosed in the report (§1.3). One value was chosen, from one source, before any refit |
| 1.5 | 2026-11-08 | §12.9 VZ-1 split into a rejected-fit AMD surface and a published Google surface; `run.py compare` added (§11) | Owner instruction. The market fit ended at ladder step 3, so no accepted AMD surface exists. No pass rule, tolerance, seed, filter, or default changed | Yes. Figures are generated from results already viewed and do not change them |
| 1.6 | 2026-11-08 | Primary question revised to GOOG under the published parameters (§2.1, §17.3); Massive dataset replaced to add GOOG and GOOGL; four-ticker correlations allowed before G1 (§5.4); `q_GOOG = 0`; OD-2 and OD-11 | Owner instruction, after the AMD fit failed acceptance. No pass rule, tolerance, seed, filter, or default changed | Yes. The AMD fit and the published-parameter price had been viewed; this MUST be disclosed in the report (§1.3) |
| 1.7 | 2026-11-08 | V-06 benchmark changed from Clarke and Parrott to Ikonen and Toivanen; V-01 and V-06 values entered by the agent on the owner's instruction (§15); market comparison contract (§8.4, §3); one named out-of-hours bar omitted (`DATA_CONTRACT.md` §1.1) | Clarke and Parrott's prices at spots 10 to 12 are 2% to 11% above four later independent studies, so a correct engine could fail V-06 against them. Decided by the owner before V-06 had been run. No pass rule, tolerance, seed, or filter changed | V-06 had not been run. Other results had been viewed; none is affected by this change |
| 1.8 | 2026-11-08 | Market comparison (§8.4) uses `q = 0.0031` from Alphabet's declared dividend | Owner instruction. No pass rule, tolerance, seed, or filter changed | Yes. The `q = 0` comparison had been viewed; the report MUST disclose this |
| 1.9 | 2026-11-08 | VZ-1b (§12.9) defined as the GOOG 2015 Heston model surface over the source paper's maturity range | Owner instruction. No pass rule, tolerance, seed, or filter changed | Yes. The figure is drawn from parameters already viewed and does not change them |
| 1.10 | 2026-11-08 | Feb 2015 Google put is the official market comparison (§8.4), replacing the 2026 contract, whose Cboe quote was removed; exploratory NVDA, AMD, and INTC runs (§8.5); VZ-1c, VZ-1d, VZ-9 | Owner instruction. Cboe's terms require written consent to store or publish its materials. No pass rule, tolerance, seed, or filter changed | Yes. The 2015 scratch comparison and the 2026 gap had been viewed; the report MUST disclose this and describe §8.4 as in-sample |
| 1.11 | 2026-11-13 | Project restricted to GOOG; EXT-A and OD-11 closed; out-of-range GOOG tests (§8.6) | Owner instruction. No pass rule, tolerance, seed, or filter changed | Yes. The NVDA, AMD, and INTC results had been viewed; the report MUST disclose that this work was stopped and why |
| 1.12 | 2026-11-13 | GOOG market test on 2026-10-13 (§8.7); GOOG iVolatility file added (`DATA_CONTRACT.md` v0.8); optional config key `raw_iv_file` | Owner purchased GOOG Raw IV for 2026-10-13. No pass rule, tolerance, seed, or filter changed | No GOOG fit or market comparison had been viewed when this was written |
| 1.13 | 2026-11-15 | Primary question changed to out-of-sample accuracy on GOOG (§2.1); test population defined (§2.2); headline is the accuracy figures (§2.4); test contracts priced with LSMC as American options with Monte Carlo intervals, precision target T3, budget B4, breakdowns, and an expiration-level bootstrap (§8.5, §9); §6.7 step 3 evaluates the rejected fit and labels it; GOOG uses 2026-10-13 only, and the calibration-date sweep is removed; V-13 and V-14 apply to GOOG; V-00, V-19, and call cases in V-04 added; reference contract kept for verification and sweeps (§4.1); C++ (EXT-B, V-30), multi-asset (EXT-A, V-20 to V-28), correlation estimation, and requirements for NVDA, AMD, and INTC removed; §15 lets any reviewed, committed script read `data/` if its output is aggregate, while agents never read `data/` themselves; precedence item 1 restored (§1.2); `validate_goog_data.py` added (§11) | Owner instruction. Filters F1 to F8, grid rules, the test-set definition, the 0.01 limit, and seed lists are unchanged | Yes. The GOOG 2026-10-13 fit (RMS 0.01049) and the v1.12 market-test aggregates, including RMS by maturity, had been viewed. §6.7 step 3 was written after the fit's RMS was seen; the report MUST disclose this. The LSMC test prices, moneyness and kind breakdowns, bootstrap intervals, and T3 had not been computed when this was written |
| 1.13 | 2026-11-15 | Header and §12.3 refer to `DATA_CONTRACT.md` v0.9, whose §3.3 now matches §15: any agent may run any committed script on licensed data once the owner has reviewed it | Owner instruction. Documentation only; no pass rule, tolerance, seed, filter, or default changed. The spec version stays 1.13 | Not applicable: no result is affected |
| 1.13 | 2026-11-16 | Primary question (§2.1) reworded: accuracy on contracts within and outside the calibrated sample | Owner instruction. The rest of the spec, including the test population, filters, pass rules, seeds, and the 0.01 limit, is unchanged | Yes, as for the first 1.13 row: the GOOG fit's RMS and the v1.12 tree-shortcut aggregates had been viewed. No LSMC test price had been computed |
