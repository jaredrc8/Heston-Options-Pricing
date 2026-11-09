# AI Use Log

This file records every material use of AI tools in this project, as committed to in Section E and Table A3 of the ENSF 620 project proposal. Entries are added as the work happens and are summarized as a disclosure in the final report.

## Tools

| Tool | Model | Access mode |
|---|---|---|
| Claude Code (VS Code extension) | Claude Opus 5.5 (`claude-opus-5-5`) | Reads the repository and writes files into the working tree when the author asks. Since 2026-11-08 it also commits and pushes when asked, under `DATA_CONTRACT.md` §3.3. |
| Claude (claude.ai chat) | — | Chat only (planned in Table A3; no use logged yet). |

## Repository access (change from the Table A3 plan)

Table A3 lists *repository write access* as out of bounds and limits the tool to *suggestions in chat, which I read, modify, and test myself*. On 2026-10-06 the author granted Claude Code write access to the working tree. Until 2026-11-08 only the author committed and pushed to Git. On 2026-11-08 the author allowed Claude Code to commit and push when asked, provided `data/` is in `.gitignore` and `scripts/check_no_data.sh` passes immediately before the push (`DATA_CONTRACT.md` §3.3). Owner review of AI-written files may follow the commit, and the review checkboxes in each entry show what has been reviewed.

**Author action:** update Table A3 in the final report to match.

---

## Log

### Entry 1 — 2026-10-06 — LSMC pricing engine

**Tool:** Claude Code, Claude Opus 5.5.

**Author's request:** Start a new Python file containing the Least-Squares Monte Carlo pricing engine, built robustly. Record all AI use in `AI_USE.md`. No C++ unless the computational cost requires it.

**AI context:** the proposal PDF and the existing `prototype.ipynb`.

**Files written by the AI:**
- `python/heston_lsmc.py` (new)
- `AI_USE.md` (this file, new)

**What the AI produced in `python/heston_lsmc.py`:**
- `HestonParams` and `HestonModel`: per-asset parameters, the d-asset model, the correlation set-up, and input validation.
- `simulate_paths`: multi-asset Heston simulation. It accepts either a seeded generator or a pre-generated array of normals; the normals array is the hook for the shared fixture in V7.
- `WorstOfPut`: the contract payoff.
- `PolynomialBasis` and `ContinuationFit`: the regression basis for the continuation value.
- `fit_policy` and `apply_policy`: Longstaff–Schwartz backward induction, and forward application of the fitted exercise policy.
- `price_bermudan`: a wrapper that fits the policy on one set of paths and prices on an independent set. It reports the Bermudan price, the European price on the same paths, the early-exercise premium, each with a standard error, plus diagnostics.

**Design decisions made by the AI** (for the author to review and either adopt or change):
1. **Discretization.** Full-truncation Euler for the variance and log-Euler for the price (Lord et al., 2010, ref. [11]). Each exercise interval has `substeps` equal steps. The Feller condition is reported but not enforced.
2. **Correlation construction.** W^v_i = ρ_i W^S_i + √(1−ρ_i²) W^⊥_i, with each W^⊥_i independent of everything else. This reproduces ρ_i and R_ij exactly, and implies the cross-asset variance correlations ρ_iρ_jR_ij, which the proposal does not specify. A full 2d×2d driver correlation can be passed instead, for example for V5.
3. **Factorization.** Cholesky when the correlation matrix is positive definite; pivoted Cholesky when it is only semi-definite.
4. **Regression basis.** Default features are the performances S_i/S_i0, the variances max(v_i, 0) and the worst performance min_i S_i/S_i0. The basis is every monomial up to total degree 2. Features are standardized before the fit. The solver is SciPy `lstsq` with the rank-revealing `gelsy` driver and a rank cutoff of 1e-10.
5. **Regression sample.** Only in-the-money paths are used. A date is skipped (the holder continues) when it has fewer than 2 in-the-money paths per basis term.
6. **Estimator.** Two independent path sets, spawned from one master seed with `SeedSequence.spawn`. The reported price is out-of-sample and low-biased. The in-sample Longstaff–Schwartz value is reported only as a diagnostic.
7. **Early-exercise premium.** Computed path by path against the European payoff on the same pricing paths, which gives it its own standard error.
8. **Payoff convention.** (K − min_i S_i/S_i0)^+ per unit notional, as in Section C of the proposal.

**AI errors found and corrected during the session** (all found by the AI's own checks):
1. A default argument `PolynomialBasis()` in a function signature ran at import time, before the validation helpers existed, and raised `NameError`. Changed the default to `None`.
2. **Rank cutoff in the regression.** SciPy's default cutoff (machine epsilon) overestimated the rank of designs with exactly duplicated columns. This happens with identical assets. The coefficients grew to about 3e9 and the fitted values were off by about 2e-4, so 1,436 of 50,000 paths exercised on a different date than in the single-asset run. Setting the cutoff to 1e-10 made the tiled three-identical-asset price equal the single-asset price exactly.
3. **Semi-definite correlation.** The first version factored semi-definite correlation matrices with an eigendecomposition. Identical assets then differed by up to 3e-8, because √max(v, 0) amplifies round-off near v = 0. Replaced it with pivoted Cholesky; the difference fell to about 5e-15.

**Checks the AI ran.** These ran in a scratch directory and are **not** committed. They are **not** Table 1 verification evidence, and they produce no reported results. Every reference value came from a closed form or a published source, not from the AI:
- Heston with vol-of-vol near zero, compared with the Black–Scholes European put: z = −1.0.
- The same near-zero vol-of-vol limit with 50 exercise dates, compared with the Longstaff & Schwartz (2001) Table 1 case (S=36, K=40, σ=0.2, r=0.06, T=1; finite-difference American value 4.478): Bermudan 4.467 ± 0.009.
- The Fang & Oosterlee (2008) Heston European benchmark, which violates Feller (reference 5.785155): z = +0.05 with 50 steps and +1.27 with 200 steps.
- Three identical assets on tiled paths, compared with the single-asset price: identical.
- Pre-generated normals reproduce the generator-driven paths bit for bit.
- The same seed reproduces the same price.
- Invalid inputs are rejected.
- Runtime for 3 assets, 10^5 paths per set, 50 dates, 4 sub-steps: about 4 s with degree 2 and 8 s with degree 3, on the author's laptop with illustrative parameters. This is below the 30 s trigger in Table 3.

**Suggestions made but not acted on** (the author's decision):
- In `prototype.ipynb`, `np.random.default_rng(123456)` creates a generator that is never used. All draws come from the unseeded global `np.random`, so the notebook results cannot be reproduced.
- In the prototype, `heston_monte_carlo` refuses to simulate when the Feller condition fails. Calibrated parameters often fail Feller, as the Fang & Oosterlee set does. The new engine does not refuse.
- **V8 pass rule.** "Seeds agree within 1 SE". Two independent estimates differ with standard deviation √2·SE, so a correct engine passes this only about 52% of the time.

**Author review status:**
- [ ] `python/heston_lsmc.py` read and understood
- [ ] Design decisions 1–8 accepted, changed, or rejected (record which)
- [ ] Table 1 checks written and run by the author
- [ ] Table A3 updated in the final report (write access granted 2026-10-06; see above)

### Entry 2 — 2026-10-06 — Heston closed-form pricer and calibration

**Tool:** Claude Code, Claude Opus 5.5.

**Author's request:** Write the Heston closed-form pricer and the calibration code. The author also clarified that Claude Code has repository write access but may not push to Git.

**Files written by the AI:**
- `python/heston_closed_form.py` (new)
- `python/heston_calibration.py` (new)
- `AI_USE.md` (updated: the repository-access section and this entry)

**What the AI produced:**
- `heston_closed_form.py`:
  - `heston_charfn`, the characteristic function of ln(S_T/F);
  - `heston_price`, European call and put prices;
  - `heston_implied_vol`, Black–Scholes implied vols of Heston prices;
  - `black_scholes_price`, `black_scholes_vega`, and `implied_vol`, a vectorized implied-vol inversion for converting market mid-quotes.
- `heston_calibration.py`:
  - `VolQuotes`, one asset's quote snapshot;
  - `calibrate_heston`, which minimizes Eq. (4) of the proposal from several starting points, with optional fixed parameters;
  - `CalibrationResult` and `StartOutcome`, which return the best fit, the residuals, and every start.

**Design decisions made by the AI** (for the author to review and either adopt or change):
1. **Pricing formula.** Lewis's (2001) single-integral formula, with characteristic function φ, instead of Heston's original two-integral form, the COS method, or Carr–Madan. Calls and puts share one integral, so put–call parity holds to round-off. Lewis (2001) is not in the proposal's reference list.
2. **Characteristic function.** The "little Heston trap" form [1], rearranged in two places so nothing cancels as ξ → 0: β − d is computed as −ξ²a/(β + d), and the logarithm uses a complex log1p that is accurate for small arguments.
3. **Quadrature.**
   - Rule: composite 32-point Gauss–Legendre.
   - Panels: widths double from 0.5 near the origin, capped at about four oscillation periods of e^{−iuk} or at U/4, whichever is smaller.
   - Truncation: the point U is the first point on a 1, 2, 4, …, 2²⁴ grid beyond which |φ(u − i/2)|/u stays below 1e-13.
4. **Implied-vol inversion.**
   - Inverts the out-of-the-money option at each strike.
   - Uses Newton's method on the log of the normalized Black price, falling back to bisection whenever a step leaves the bracket.
   - Searches σ√T up to 10, and returns NaN when no vol reproduces the price.
5. **Optimizer.** SciPy `least_squares` (`trf`, `x_scale="jac"`, ftol/xtol/gtol 1e-12, two-point finite-difference Jacobian).
6. **Starting points.** 8 by default: one at v0 = θ = (median market vol)², κ = 1.5, ξ = 0.5, ρ = −0.5, and 7 random draws around it from a seeded generator. Every start's outcome is returned, so stability across starts can be judged.
7. **Bounds.** v0 and θ in [1e-4, 2], κ in [1e-3, 20], ξ in [1e-3, 5], ρ in [−0.999, 0.999]. The proposal specifies only positivity and |ρ| < 1, so these limits are the AI's choice. The Feller condition is not imposed.
8. **Weights.** Every quote has weight 1 by default (Eq. (4) as written), and optional per-quote weights are supported. Any parameter can be held fixed (the κ fallback in Table 3).
9. **Far-wing quotes.** Inside the objective, each model implied vol is held at no less than the vol of a normalized price of 1e-12. That is the closed form's round-off floor, so the objective stays finite and continuous. Market quotes with a normalized price below 1e-10 (about 1e-8 dollars on a 100 spot) are rejected with an error rather than fitted.

**AI errors found and corrected during the session** (all found by the AI's own checks):
1. `implied_vol` did not broadcast `prices` together with the strike and maturity arrays, so a scalar strike with a vector of prices crashed. Fixed.
2. **Discontinuous fallback.** The first version replaced a NaN model implied vol with the price difference divided by the market vega. Market vegas are tiny far out of the money, so this residual was discontinuous: it jumped from 0.02 to −1.7. It also dominated the objective: one unresolvable quote shifted the fit on otherwise exact synthetic quotes by up to 0.016 in implied vol. Replaced with the floor and rejection described in decision 9.

**Findings while checking** (no code change needed):
- **Fang & Oosterlee (2008) Heston benchmark.** The published value is 5.785155450. The pricer gives 5.7851554344, and an independent textbook characteristic function integrated with adaptive `quad` gives the same value to 11 digits. The 1.6e-8 gap is therefore in the published figure.
- **Black–Scholes limit (ξ = 1e-7).** The gap to Black–Scholes is 7e-14 when ρ = 0, but grows linearly with ξ when ρ ≠ 0. That is Heston's genuine first-order ρξ correction, not an error.
- **Accuracy floor.** Out-of-the-money prices carry an absolute round-off of about 1e-16·DF·√(FK) whatever quadrature is used. Implied vols are accurate to about 1e-10 within 4 standard deviations of the forward, degrade beyond about 5, and become NaN once the price is lost. This is documented in the module docstring.

**Checks the AI ran.** These ran in a scratch directory and are **not** committed. They are **not** Table 1 verification evidence; V1 and V2 remain the author's to write. Every reference value came from a published source or from an independent implementation:
- **Normalization:** φ(0) = φ(−i) = 1 for three parameter sets at three maturities.
- **Independent integration:** against a textbook characteristic function integrated with adaptive `quad` (Heston's two-integral form), over 60 cases: 3 parameter sets (one violating Feller, one with ξ = 1.2), maturities 0.05 to 10, strikes 50 to 200. Maximum difference 3.5e-11.
- **Put–call parity:** 2.8e-14.
- **Implied-vol round trip:** on 852 Black–Scholes quotes, every recovered vol reprices to within 6e-14 of the input. Unreachable prices return NaN.
- **Monte Carlo engine:** European puts from `heston_lsmc` (200,000 paths, 200 steps) against the closed form give z = +0.78, +0.33 and +0.45.
- **Speed:** 0.6 ms to price and invert a 40-quote surface; about 0.3 s per 8-start calibration.
- **Synthetic surfaces** (5 maturities × 8 strikes) from three known parameter sets, including the Feller-violating one: parameters recovered with implied-vol RMS ≤ 2e-15, and all 8 starts reached the best cost.
- **Noise:** with 0.002 implied-vol noise, the fit RMS is 0.0018 and every start reaches the same cost.
- **Fixed κ:** at the true value, recovery is exact; at a wrong value the RMS is 3.5e-3.
- **Hard starts:** two starts far from the answer converge, with RMS 2.7e-13.
- **Far wing:** the floor is continuous, and unresolvable market quotes are rejected.
- **Invalid inputs** are rejected.

**Suggestions made but not acted on** (the author's decision):
- **Module-level example in `heston_lsmc.py`.** The author added an example at the end of the file. It runs on every import, including when `heston_closed_form` and `heston_calibration` import `HestonParams`, so each import costs about 4 s and prints a line. Wrapping it in `if __name__ == "__main__":` keeps it runnable as a script without that cost.
- **Reference list.** Add Lewis (2001), "A simple option formula for general jump-diffusion and other exponential Lévy processes". The author should verify the publication details.
- **Data cleaning.** The calibration expects clean implied vols. Converting mid-quotes with `implied_vol` returns NaN for quotes that violate no-arbitrage bounds; the data step should drop those, along with zero-bid, crossed, or stale quotes, before building `VolQuotes`.
- **Correlation estimation.** Estimating R_ij from trailing log-returns is not written yet.

**Author review status:**
- [ ] `python/heston_closed_form.py` and `python/heston_calibration.py` read and understood
- [ ] Design decisions 1–9 accepted, changed, or rejected (record which)
- [ ] V1 and V2 written and run by the author

### Entry 3 — 2026-10-18 — Rewrite of all three modules in a simpler style

**Tool:** Claude Code, Claude Opus 5.5.

**Author's request:** "These python scripts are too complicated for me to understand. Use a more simple coding style." The author had moved the files from `python/` to `claude_python/`.

**Files rewritten by the AI** (each file fully replaced):
- `claude_python/heston_lsmc.py`
- `claude_python/heston_closed_form.py`
- `claude_python/heston_calibration.py`

`AI_USE.md` gained this entry. The previous versions were never committed; Entries 1 and 2 describe them.

**What changed in style:**
- **Size:** 1,287 lines became 598.
- **Structure:** plain functions with explicit arguments. Each asset's Heston parameters are a dictionary `{"v0", "kappa", "theta", "xi", "rho"}`, and results are returned as dictionaries.
- **Removed:** dataclasses, input-validation layers, private helper functions and `__all__`.
- **Examples:** each file's example now runs only under `if __name__ == "__main__":`. This also resolves the import problem noted in Entry 2.

**Kept** (the same method, written more simply):
- **Simulation:** full-truncation Euler for the variance and log-Euler for the price.
- **Pricing:**
  - two independent path sets, one to fit the exercise rule and one to price;
  - the European price on the same paths, with a standard error for the premium;
  - regression on in-the-money paths only, using each asset's performance, its variance and the worst performance, with standardized features;
  - a rank cutoff of 1e-10, and at least 2 paths per basis term.
- **Closed form:** Lewis's formula with the little-Heston-trap characteristic function, 32-point Gauss–Legendre, and implied vols taken from out-of-the-money options.
- **Calibration:** `least_squares` from 8 starting points with bounds, with optional fixed parameters.

**Removed or simplified, and what each change means** (the author's decision whether to keep it or restore the old behaviour):
1. **Payoff.** The engine prices only the worst-of put; the general payoff argument is gone. Single-asset prices are per unit of S0.
2. **Correlation matrix.** The full 2d×2d driver-correlation input and support for semi-definite matrices are gone, so the asset correlation matrix must be positive definite. V5 can instead be run by repeating one asset's simulated paths three times (`np.repeat`).
3. **Pre-generated normals.** This input is gone. It was only needed for the C++ comparison (V7).
4. **Random numbers.** They now come from one generator in sequence (training paths, then pricing paths) instead of `SeedSequence.spawn`. The same seed still gives the same result, but the draws, and therefore the exact numbers, differ from the old version.
5. **Output.** Exercise probabilities and the run settings are no longer returned. The output is the price, the European price and the premium, each with its standard error, plus the in-sample price.
6. **Regression solver.** `numpy.linalg.lstsq` (SVD) replaces SciPy's `gelsy`, with the same 1e-10 cutoff.
7. **Characteristic function.** The textbook form replaces the cancellation-free rearrangement, so the lower bound on ξ in calibration rose from 1e-3 to 1e-2. At ξ = 0.01 the two forms agree to 1.4e-12.
8. **Quadrature.** Equal panels of width 4 replace the adaptive panels, and the pricer handles one maturity per call. This is about 10 times slower: 5.4 ms per 40-quote surface instead of 0.6 ms, and 2.4–3.8 s per calibration instead of 0.3 s.
9. **Round-off floor.** It rose from about 1e-14 × S0 to about 1e-13 × S0. Implied vols of options priced below about 1e-9 × S0 are inaccurate; listed quotes are far above that.
10. **Implied-vol inversion.** Bisection (60 halvings) replaces the safeguarded Newton method, and `black_scholes_vega` was removed.
11. **Calibration inputs and outputs.** There are no quote weights and no user-supplied starting points, and the `VolQuotes` and `CalibrationResult` classes became arrays in and a dictionary out.
12. **Far-wing handling.** A model implied vol that can't be computed (NaN), or a trial point where the pricer raises an error, now counts as zero vol, a large error. This replaces the continuous floor of Entry 2. Market quotes too small to price are no longer rejected automatically, so the author must remove tiny-price quotes before calibrating; the module docstring says so.
13. **Truncation cap.** The search for the truncation point stops at u = 2¹⁵. Beyond that the pricer raises an error, which happens only at extreme corners of the bounds: variance near 1e-4 together with ξ = 5 or |ρ| ≈ 1.

**AI error found and corrected during the rewrite:** the first draft of the new calibration crashed if the optimizer tried pathological parameters, because the pricer raised an error there. Testing the corners of the parameter bounds exposed it: 16 of 144 corner cases raised. It is fixed by treating such trial points as a very bad fit.

**Checks the AI ran.** These ran in a scratch directory and are **not** committed. They are **not** Table 1 evidence. The old versions served as references; they had been verified against independent methods in Entries 1 and 2.
- **Closed form:**
  - New and old prices agree to 3.7e-13 over 324 cases.
  - Put–call parity holds to 2.8e-14.
  - The Fang & Oosterlee case gives 5.785155434, unchanged.
  - The implied-vol round trip reprices all 852 quotes to within 5.7e-14.
  - New and old implied vols agree to 1.2e-10 within 2 standard deviations of the forward and 1.7e-7 within 3.
- **Calibration:**
  - Three synthetic parameter sets are recovered with RMS ≤ 9e-16 and parameters accurate to 1e-13; all 8 starts agree.
  - With 0.002 vol noise, the RMS is 0.0018 and all starts reach the same cost.
  - Holding κ at its true value recovers the rest exactly; at a wrong value the RMS is 3.5e-3.
  - Starts placed in pathological corners no longer crash.
- **LSMC:**
  - The new simulator matches the old over 20 seeds: European worst-of put 0.11339 ± 0.00012 against 0.11343 ± 0.00011.
  - Longstaff–Schwartz benchmark in the near-zero vol-of-vol limit, over 10 seeds: 4.4653 ± 0.0039 (old 4.4683 ± 0.0017; finite-difference American value 4.478).
  - European puts against the closed form give z = −0.44, −0.85 and −1.02.
  - Three tiled identical assets reproduce the single-asset price exactly, and the same seed reproduces the same result.
  - A 3-asset pricing (10⁵ paths per set, 50 dates, 4 sub-steps) takes 3.8 s with degree 2 and 9.2 s with degree 3.

**Author review status:**
- [ ] All three `claude_python/` modules read and understood
- [ ] Simplifications 1–13 accepted, or the old behaviour requested back (record which)
- [ ] Market-data step removes tiny-price quotes before calibration (simplification 12)

### Entry 4 — 2026-10-18 — Fix the data path in `src/correlation.py`

**Tool:** Claude Code, Claude Opus 5.5.

**Author's request:** `correlation.py` (written by the author) failed when run from another folder.

**Cause:** the default data root `"../data/nvda_amd_intc_minute"` was relative to the folder the script was run from, not to the script, so DuckDB found no Parquet files.

**Change:** `PACKAGE_ROOT` is now built from the script's own location (`Path(__file__).resolve().parent.parent / "data" / "nvda_amd_intc_minute"`). That is a 3-line edit; no other code changed.

**Check:** ran the script from `/tmp`; it loaded 3,704 daily sessions and printed the correlation tables. The printed correlation values were not reviewed by the AI and are not reported results.

### Entry 5 — 2026-10-18 — Calibration to the 2026-10-15 option snapshot

**Tool:** Claude Code, Claude Opus 5.5.

**Author's request:** Use `heston_calibration.py` and `data/nvda_amd_intc_IV_Oct13-15.csv` to calibrate Heston parameters for a 1-year contract.

**Files written by the AI:**
- `src/calibrate_market.py` (new)
- `results/heston_params.csv` (output of running it)

**Quote selection (AI's choices, for the author to review):**
- latest date in the file only (2026-10-15);
- maturities 0.08–1.3 years, which bracket T = 1;
- out-of-the-money options only;
- strikes 0.7–1.3 × spot;
- bid > 0 and iv > 0;
- the provider's `iv` column, treating American out-of-the-money vols as European;
- r = 5% and q = 0, as in the proposal.

**Outcome:** with all five parameters free, κ ends at its upper bound of 20 for all three tickers, with ξ between 3.5 and 4.3 and RMS implied-vol error 0.0092 (NVDA), 0.0102 (AMD) and 0.0097 (INTC). All 8 starts reach the same cost. Holding κ at 1, 2 or 5 raises the RMS to between 0.011 and 0.018. These numbers come from an AI-run calibration; the author should rerun the script before reporting them.

### Entry 6 — 2026-10-18 — Quote window and fixed κ for the market calibration, moved into `main.py`

**Tool:** Claude Code, Claude Opus 5.5.

**Author's request:** Narrow the quote window until κ is reasonable and the fit is stable. Apply it in `main.py` using `heston_calibration.py`, and save each ticker's parameters as an array or DataFrame.

**Files changed by the AI:**
- `src/main.py`: replaced the synthetic calibration block with the market calibration; the correlation block is unchanged.
- `results/heston_params.csv`: regenerated.

`src/calibrate_market.py` from Entry 5 still uses the old window and is now superseded.

**Findings from the AI's scan of quote windows:**
- Minimum maturity from 0.25 to 0.6 years, maximum from 1.3 to 2.3 years, all parameters free.
- No window gave a reasonable κ for all three tickers. κ moved between its bounds: 20, or about 0.02 with θ at its bound of 2.
- The objective is flat along a κ–θ–ξ ridge, so κ is not identified by these surfaces.

**Choice (proposal Table 3 fallback, for the author to review):**
- Maturities 0.4–1.3 years, which bracket T = 1 and leave out the steep short-dated term structure.
- κ fixed at 2.0. The author should cite a literature source for this value.
- In this window, κ = 1, 2 or 3 all give RMS error between 0.0050 and 0.0064, and the free-κ fit is at most 0.0007 better.

**Result:**

| | RMS implied-vol error | Max implied-vol error |
|---|---|---|
| NVDA | 0.0059 | 0.016 |
| AMD | 0.0055 | 0.016 |
| INTC | 0.0051 | 0.015 |

All 8 starts agree for each ticker. These numbers come from an AI-run calibration; the author should rerun `main.py` before reporting them.

### Entry 7 — 2026-10-18 — Market calibration moved back into `calibrate_market.py`

**Tool:** Claude Code, Claude Opus 5.5.

**Author's request:** Keep the market calibration in `calibrate_market.py`, called by one function from `main.py`, which writes a CSV of parameters per asset. `main.py` then reads the CSV. If the IV data cannot be used (license), the CSV is filled by hand with literature values. Set it up like the correlation step.

**Files changed by the AI:**
- `src/calibrate_market.py`: rewritten. It now uses the Entry 6 settings (maturities 0.4–1.3 years, κ fixed at 2.0), and a single `main()` writes `results/heston_params.csv` and returns the DataFrame.
- `src/main.py`: the calibration block was replaced by a `USE_MARKET_DATA` switch. When it is `True`, `main.py` calls `calibrate_market.main()`. Either way it then reads the CSV, keeping only the columns spot, v0, kappa, theta, xi and rho.

**Check:** ran `main.py` from `/tmp`. The parameters are the same as in Entry 6.

### Entry 8 — 2026-10-18 — Plot of simulated Heston paths in `main.py`

**Tool:** Claude Code, Claude Opus 5.5.

**Author's request:** Use `heston_lsmc.py` to simulate and plot each asset's paths in one figure with 3 subplots, with a fixed seed so the result repeats.

**Files changed by the AI:**
- `src/main.py`: new section "Simulated Heston Paths". It calls `lsmc.simulate_heston_paths` with:
  - spot prices and parameters from `heston_params.csv`;
  - the asset correlation matrix from `correlation.main()`, reordered to the CSV's ticker order;
  - r = 5% and q = 0;
  - T = 1 year in 252 daily steps;
  - 20 paths per asset;
  - `np.random.default_rng(2026)`.

  It plots one subplot per asset, with a dashed line at S0, and saves the figure to `results/simulated_paths.png` before `plt.show()`.

**Check:** ran `main.py` twice from `/tmp`; both runs saved byte-identical figures (same MD5).

### Entry 9 — 2026-10-18 — Single-asset NVDA Bermudan put in `main.py`

**Tool:** Claude Code, Claude Opus 5.5.

**Author's request:** Use `heston_lsmc` to price NVDA with LSMC from inputs of strike, expiry and current price, using the calibrated parameters.

**Files changed by the AI:**
- `src/main.py`: new section "NVDA Bermudan Put (LSMC)".

**What the section does:**
- **Inputs:** current price S0 (from the CSV), strike K (default at the money), expiry T (default 1 year), 52 weekly exercise dates, and 10⁵ paths per set.
- **Pricing:** calls `lsmc.price_bermudan_worst_of_put` with one asset and correlation `[[1]]`, passing the strike as K/S0 and converting the result to dollars by multiplying by S0. It uses the section's `SEED`, r = 5% and q = 0.
- **Check:** also prints the closed-form European put from `closed_form.heston_price`.

**Check result:** the simulated European put (26.978 ± 0.122) agrees with the closed form (26.997), z = −0.15. These numbers come from an AI run; the author should rerun `main.py` before reporting them.

### Entry 10 — 2026-10-18 — Plot of the NVDA put price as t approaches T

**Tool:** Claude Code, Claude Opus 5.5.

**Author's request:** Plot the fair price as t approaches T, with daily exercise dates so the Bermudan approximates an American.

**Files changed by the AI:**
- `src/main.py`: new section "NVDA Put Price as t -> T". It reprices the same put at 26 values of elapsed time t (one every two weeks), holding S0 fixed.

**Settings for each point:**
- time to expiry τ = T − t, with round(252τ) daily exercise dates;
- 50,000 paths per set, 2 sub-steps;
- the same `SEED` at every point, so the curve is smooth;
- the closed-form European put plotted alongside, with a 95% band (±1.96 standard errors) around the Bermudan curve and a line at the intrinsic value.

The figure is saved to `results/nvda_put_price_vs_time.png`. The whole section takes about 17 s.

**Check:** the early-exercise premium shrinks toward expiry, and the Bermudan curve stays above the European curve. These results come from an AI run; the author should rerun `main.py` before reporting them.

### Entry 11 — 2026-10-18 — NVDA pricing and price-curve sections merged

**Tool:** Claude Code, Claude Opus 5.5.

**Author's request:** Don't rerun the analysis to make the plot; run LSMC once in the NVDA section, store the data, and plot it there.

**Clarification:** one LSMC run only prices t = 0, so a curve at fixed S0 needs a repricing for each t. The AI asked how to proceed, and the author chose to keep the repricing.

**Files changed by the AI:**
- `src/main.py`: the NVDA section is now a single loop over 26 values of elapsed time. It stores the Bermudan, European Monte Carlo, closed-form European and premium prices in arrays. The t = 0 entry is the reported price, and the plot is drawn from the stored arrays.

All points use daily exercise, 100,000 paths and 2 sub-steps. This replaces the author's partial merge, which had broken indentation; the separate curve section and the duplicate timing block were removed.

**Check:** ran `main.py` from `/tmp` in about 33 s. At t = 0 the simulated European put (26.948 ± 0.121) agrees with the closed form (26.997). These numbers come from an AI run; the author should rerun `main.py` before reporting them.

### Entry 12 — 2026-11-08 — Spec v1.1 and implementation of the SPEC.md layout

**Tool:** Claude Code, Claude Opus 5.5. Reviewed writes to the working tree (OD-7). The AI made no commits.

**Author's request:** Use the layout in `SPEC.md` and the data contract `DATA_CONTRACT.md`. Edit `SPEC.md` to use plain functions and dictionaries instead of dataclasses, then follow `SPEC.md` to finish the project.

**Files changed by the AI:**
- `SPEC.md`: §10.1 and §10.2 rewritten for plain dictionaries; references renamed to `DATA_CONTRACT.md`; version 1.1 and a §22 row. No pass rule, tolerance, seed, filter or default changed.
- Protections: `.gitignore`, `.claude/settings.json`, `scripts/inspect_schema.py`, `scripts/check_no_data.sh`, and the pre-commit hook link.
- Package: `python/hestonpricer/` (`errors`, `analytic`, `iv`, `tree`, `simulate`, `lsmc`, `surface`, `calibrate`, `correlation`), `python/validate_raw.py`, `python/make_fixtures.py`.
- Pipeline: `scripts/run.py`, `verify_outputs.py`, `make_figures.py`, `smoke.sh`; `configs/`; `pyproject.toml`; `requirements.lock`; `.github/workflows/ci.yml`; `README.md`.
- Tests: `tests/test_checks.py`, `tests/test_units.py`, `tests/reference_impl.py`, generated `tests/fixtures/`.
- Moved, content unchanged: `src/` to `prototype/src/`, `prototype.ipynb` to `prototype/`. The notebook's saved outputs were cleared, as contract §3.1 requires; the originals remain in Git history.
- `results/smoke/expected.json`, written by `verify_outputs.py --check-smoke --bless`, not by hand.

**Left for the author, as SPEC.md §15 requires.** The AI did not write `configs/governance.yaml`, did not fill in any value in `tests/reference_values.yaml` or `configs/published_params.yaml` (both are empty templates), did not enter the Treasury yield, did not run `market` mode, and did not write the report, slides, reported figures or claims.

**Data boundary in this session.** The AI touched `data/` only by running `scripts/inspect_schema.py`, which printed file names, column names, dtypes, row counts and null counts. No data value was read or printed. Validation on licensed files was not run; that is an owner action.

**Design decisions made by the AI** (for the author to review and either adopt or change):
1. **Raw header mapping** in `surface.py`, inferred from the header names only: `adjusted close` to `stock_price` and `price` to `mean_price` are the two to confirm (OD-4). `is_settlement` and `*` are not in the contract and are dropped.
2. **Massive validation** reads the Parquet package on disk, because the contract's daily gzip CSV format does not match what is stored.
3. **Synthetic parameters** `v0 = 0.09, kappa = 2.0, theta = 0.10, xi = 0.6, rho = -0.6` and yield 4%. These are made up and describe no real stock.
4. **Residual buckets:** three tenor groups and three moneyness groups.
5. **V-14(b) "agree":** the call and put bid–ask IV bands overlap.
6. **Scope gate:** LSMC raises an error for more than one asset, because EXT-A may not start before G1. `simulate` and `estimate_correlation` are multi-asset because §10 makes them public API.
7. **Extra error code** `E_RUN_EXISTS` (exit 3) for the "do not overwrite a completed run" rule in §14.3.

**AI errors found and corrected during the session:**
1. An editable `pip install` silently did not work here, because macOS marked the install's `.pth` file as hidden and Python skips hidden `.pth` files. Scripts and tests now add `python/` to the import path themselves.
2. The first version of the price figure clipped its right axis label; found by viewing the figure.

**Checks the AI ran.** All on synthetic or test inputs. They are not reported results. The verification table was written to a scratch location, and `results/verification_table.csv` remains for the author to generate from a committed tree.
- Fast set: 39 passed, 1 skipped (V-01, no owner-entered reference values). `smoke.sh` passes in about 18 s.
- All checks: 9 PASS (V-02, V-03, V-04, V-05, V-07, V-08, V-12, V-16, B1), 0 FAIL, 17 NOT_RUN with reasons. V-16 ran before these files were committed, so it must be rerun.
- B1 median 0.88 s over 5 repeats on this laptop, against a 30 s budget.
- An informal pass of V-09, V-10 and V-11 on the made-up synthetic parameters found no problem. It is not evidence for those checks, which need the reported parameter set.

**Findings for the author:**
- Entries 5 to 11 show that market fits and prices were viewed on 2026-10-18, before spec v1.0. The §22 row for v1.0 and OD-6 ask the author to confirm this.
- Files in `results/` from before the contract (`heston_params.csv` and two PNG figures) show spot levels, so they are now Git-ignored. They must not be submitted.

**Residual limitations after these checks.** The AI wrote both the engine and most of its tests, so a shared misunderstanding of the spec could pass both. V-01 and V-06 are the checks against outside published numbers, and they have not run. The market-data path has run only on fixtures with the same schema, never on licensed data.

**Author review status:**
- [ ] `python/hestonpricer/` and `tests/` read and understood
- [ ] Decisions 1–7 accepted, changed, or rejected (record which)
- [ ] OD-4 header mapping confirmed against the iVolatility documentation
- [ ] OD-7 disclosed in the report

### Entry 13 — 2026-11-08 — Authorization, spec v1.2, commits, and the market pipeline run

**Tool:** Claude Code, Claude Opus 5.5.

**Author's request:** Allow Claude to push when `data/` is in `.gitignore`. Enter the 4.53% Treasury rate for 2026-10-14. Record the iVolatility licence position in the data contract. Run the pipeline. Add a visualization plan to `SPEC.md`, including a PyVista surface and numerical-versus-analytical price comparisons. Add an optional call or put on n correlated assets to the scope, keeping the current code.

**Files changed by the AI:**
- `DATA_CONTRACT.md` v0.3: §2.2 licence and derived-data confirmation; new §3.3 for agent commits, pushes and pipeline runs. Classification is unchanged.
- `SPEC.md` v1.2: optional n-asset EXT-A (spec only, no code); §12.9 visualization plan (spec only, no code); §5.2 and §15 permissions; OD-1, OD-3 and OD-7; §22 row. No pass rule, tolerance, seed, filter or default changed.
- `configs/market_inputs.yaml`: yields of 4.48%, 4.53% and 4.49% for 2026-10-13, 14 and 15. The author gave 4.53%; the AI read the other two from the U.S. Treasury's Daily Par Yield Curve table, which also shows 4.53% for 2026-10-14.
- `configs/core_published_production.yaml`: `reported: false`, because market is the reported mode and no published parameters exist.
- `scripts/run.py` and `tests/test_units.py`: version numbers only.
- `README.md`: wording for the changes above.

**Not done, by the author's choice or the spec's rules:** published parameters and the V-01 and V-06 benchmarks remain empty, and the AI did not fill them in. The report, slides and claims remain the author's.

**Blocked:** the AI's attempt to create `configs/governance.yaml` was denied by the Claude Code permission system. The AI did not write the file by another route, and the author then instructed it directly to create the file with text the author approved, which it did on 2026-11-08. Calibration, pricing and sweeps on licensed data have not run.

**Checks the AI ran from commit `b6206f9`.** 10 PASS (V-02, V-03, V-04, V-05, V-07, V-08, V-12, V-16, V-17, B1), 0 FAIL, 16 NOT_RUN with reasons. The table was written to a scratch location, because it should be regenerated after the market runs.

**Validation of the iVolatility file**, run only through `python/validate_raw.py`, which reports rule names and counts: 41,256 rows, 22 rules, and one rule violated. `expiration_after_date` fails on 518 rows. The file also has 518 rows without a valid IV; the report cannot show whether these are the same rows. `DATA_CONTRACT.md` §1.5 stops the pipeline on this violation, so the AI stopped there and changed no rule, filter or raw file. The decision is the author's. The author then chose to relax the rule to `expiration >= date` with the reasoning recorded in `DATA_CONTRACT.md` v0.4 and `SPEC.md` §22; filter F3 is unchanged.

**Market calibration, run through `scripts/run.py` on 2026-11-08 after contract v0.4.** Validation passed with 0 violations. The result is failure-ladder step 3 on all three dates: no market-calibrated result.
- Primary date 2026-10-14: 48 grid points, 6 expiries, all 8 starts converged to the same fit, and RMS IV residual 0.01049 against the limit of 0.01. Neighbouring dates: 0.01029 and 0.01163.
- `kappa` sits at or near its upper bound (20.0 on the primary date), with `xi` about 4.4 and `rho` about -0.04, consistent with Entries 5 and 6. The residual is largest at tenors under 0.5 years.
- Ladder step 2 could not be tried, because no cited `kappa` has been entered. `scripts/run.py price` refused with `E_CALIB_FAILED`, so no market price, sweep or figure exists.
- The AI changed `calibrate_with_ladder` to keep the rejected fit under `rejected_full_fit`, for diagnosis only. Acceptance rules did not change. The 0.01 limit was not changed after this result was seen (§1.3).
- V-14 (b) and (c) inputs: 93 of 94 near-the-money pairs agree; median |own IV - vendor IV| is 0.0012.

**Author review status:**
- [ ] Spec v1.2 and contract v0.3 read and accepted
- [ ] `configs/governance.yaml` created

### Entry 14 — 2026-11-08 — Literature search for published Heston parameters

**Tool:** Claude Code, Claude Opus 5.5.

**Author's request:** Search for `kappa` values for AMD, NVDA and INTC, and preferably a full parameter set for each.

**Outcome:** the web search found no published Heston calibration for AMD, NVDA or INTC. The nearest source is Burkovska et al., arXiv:1611.06452, Section 6.4 and Table 8: American puts on Google, 2015-02-02, including a de-Americanization fit. Its parameters were read from an automated extract of the HTML version, so the author must check them against the paper.

**Boundary:** no value was entered in `configs/published_params.yaml` (SPEC.md §15), and no calibration was rerun. Paywalled journals were not searched in full.

### Entry 15 — 2026-11-08 — Published set entered; published mode becomes the reported result

**Tool:** Claude Code, Claude Opus 5.5.

**Author's request:** Implement the recommended parameters as the published data and continue working.

**What the AI did:**
- Entered the set in `configs/published_params.yaml` from Burkovska et al., arXiv:1611.06452, Section 6.4, Table 8, row J_CF, after re-reading the table. This used a §15 exception recorded in `SPEC.md` v1.4 on the author's instruction. The AI read the values from an automated extract of the paper's HTML, so the author still needs to check them against the source.
- Reran the market calibration with that `kappa`. Step 2 also failed the RMS limit on all three dates, so the result is ladder step 3.
- Set `core_published_production` as the reported run and `core_amd_production` as not reported (§6.7, §17.3), then ran pricing, the sweep, verification and figures.

**Correction to an earlier AI statement:** the AI told the author this paper was already cited in `SPEC.md` §21. It is a different paper by the same group; the correct reference has been added.

**Fixed-`kappa` fit (rejected), RMS IV residual against the limit of 0.01:** 0.01457 on 2026-10-14, 0.01237 on 2026-10-13, and 0.01603 on 2026-10-15. All 8 starts agreed on each date.

**AI errors found and corrected:**
1. `git_dirty` in `scripts/run.py` stripped the leading space of the first `git status` line, so a changed file under `results/` counted as a code change and a reported run was wrongly refused. Fixed, with a regression test.
2. `make_figures.py` would have added an evidence row for a market fit that does not exist at ladder step 3. Fixed.
3. `calibrate_with_ladder` discarded the rejected fits. It now keeps them apart from `params`, for diagnosis only. Acceptance rules are unchanged.

**Disclosure required (§1.3):** the rejected full fit was viewed before the `kappa` was chosen. One value from one source was chosen, before any refit.

**Limit on the result:** the reported prices use parameters fitted to Google options on 2015-02-02, and §17.3 requires the question and claims to be revised. They must not be described as calibrated to AMD.

**Author review status:**
- [ ] Table 8 values checked against the paper
- [ ] Question and claims revised in the report (§17.3)

### Entry 16 — 2026-11-08 — Section 12.9 figures

**Tool:** Claude Code, Claude Opus 5.5.

**Author's request:** Plot the rejected full fit, clearly labelled as rejected, and plot the published Google parameters. Also try Google as a fourth stock, assuming correlated stocks share Heston parameters.

**What the AI did:** added `scripts/run.py compare`, PyVista to the lock file, and the §12.9 figures in `scripts/make_figures.py`. `SPEC.md` v1.5 splits VZ-1 into VZ-1a (rejected AMD fit) and VZ-1b (published Google parameters). The figures were generated by the pipeline; the AI wrote no claims about them. Pricing code and acceptance rules did not change.

**AI errors found and corrected:**
1. The first 3D surfaces showed rescaled axis numbers (strike/forward from 2.0 to 3.0); found by viewing the figure. The geometry is now drawn in a unit box and the axes are labelled with true values.
2. A draft assigned colours in the wrong array order, and this was fixed before any figure used it. Colours are now read back from the plotted points.
3. Commit `f15f228` was pushed with one line-length lint error, because a pipe hid the lint exit code. Fixed in the next commit.

**Not done:** the Google four-stock analysis. It needs Google price data, which is not in the licensed datasets, and EXT-A remains closed by gate G1. The AI raised this with the author, along with its concern about the shared-parameter assumption.

### Entry 17 — 2026-11-08 — Google data, four-ticker correlation code, and GOOG as the pricing target

**Tool:** Claude Code, Claude Opus 5.5.

**Author's request:** Read the new `nvda_amd_intc_goog_minute` dataset instead of the old one, include Google in `DATA_CONTRACT.md` and `SPEC.md`, find the correlation between the four tickers, and build the pricer for GOOG only until NVDA, AMD and INTC can be calibrated properly.

**Author's decisions:** `q_GOOG = 0`, disclosed as a limitation; the dataset comes from Massive under the same licence. GOOGL is validated but left out of the correlations (AI's reading of "4 tickers").

**What the AI did:**
- `DATA_CONTRACT.md` v0.5 and `SPEC.md` v1.6: added Google, revised the primary question to GOOG, recorded the Parquet format and the dataset replacement. No pass rule, tolerance, seed, filter or default changed.
- `python/validate_raw.py` reads the new folder and validates each month's part files together.
- Added split adjustment, session closes and daily returns to `hestonpricer/correlation.py`, plus `scripts/estimate_correlations.py` and `configs/splits.yaml`.
- Replaced the published run with `core_goog_production`, and ran pricing, the sweep, comparisons and figures.

**Data boundary:** the AI saw `data/` only through `scripts/inspect_schema.py` and `python/validate_raw.py`. A shell command that listed `data/` directly was declined and was not repeated.

**Validation of the Massive dataset:** 177 months and 9,953,349 rows. One rule was violated once: `window_start_in_session_hours`, in `year=2018/month=7`, row index 2090. `DATA_CONTRACT.md` §1.5 stops the pipeline on a violation, so `estimate_correlations.py` was **not run** on licensed data and `results/estimates.json` does not exist. The decision is the author's.

**Check on the GOOG run:** its pooled values are identical to the earlier published run, as expected from the same parameters, seeds and rate.

**AI errors found and corrected:** a check of the smoke result in a shell command used bash syntax under zsh, so the first commit attempt did not happen. Nothing incorrect was committed.

**Sources for `configs/splits.yaml`:** the 2022 GOOG split is widely reported. Public sources disagree on the exact date and ratio of the 2014 Class C issue, so that entry is approximate and is noted as such in the file.

**Author review status:**
- [ ] Decision on the out-of-hours bar
- [ ] GOOG question and `q = 0` limitation reflected in the report

### Entry 18 — 2026-11-08 — Benchmarks, correlations, and a GOOG market comparison

**Tool:** Claude Code, Claude Opus 5.5.

**Author's request:** Omit the one out-of-hours bar. Find published option prices for V-01 and V-06. Answer how much more IV data calibration would need, and whether shared parameters are meaningful. Select a real contract to price for comparison, and keep GOOG as a single-asset put until NVDA, AMD and INTC can be calibrated.

**What the AI did:**
- Omitted the named bar in `python/validate_raw.py` and recorded it in `DATA_CONTRACT.md` v0.6. Validation then passed: 177 months, 9,953,348 rows, 0 violations.
- Ran `scripts/estimate_correlations.py` on licensed data, writing `results/estimates.json` (aggregates only).
- Entered the V-01 and V-06 reference values on the author's instruction (`SPEC.md` §15 exception), read through automated extracts of the papers.
- Chose `GOOG271217P00280000` as the market-comparison contract, priced it, and wrote `results/market_comparison.json`.

**Benchmark choice:** the spec named Clarke and Parrott for V-06, but their prices at spots 10 to 12 are 2% to 11% above four later studies. The AI raised this before V-06 ran, and the author chose Ikonen and Toivanen. V-01 uses calls, because no published Heston put table was found. Neither benchmark uses Google's parameters, so they test the engine and not the GOOG parameter set.

**Market comparison, model against the public quote:** model $20.29 ± $0.04 against bid $31.50 and ask $36.50, which is 40% below the mid of $34.00. Implied volatility is 23.1% for the model and 35.3% for the mid. There is no pass rule. The gap reflects at least the Feb 2015 parameters, and it does not show that the engine is right or wrong.

**Data boundary:** the AI read `data/` only through committed scripts. The Cboe quote is public, cited reference data (`DATA_CONTRACT.md` §2.4). The AI did not check Cboe's terms of use beyond storing this one cited quote.

**Checks:** the outcome of each is in `results/verification_table.csv`. V-01 and V-06 passed on their first run. V-16 first failed because `results/market_comparison.json` had a `spot` key holding the public closing price, which §8.4 allows. The AI did not loosen the scan; it removed that field, which duplicates `configs/market_quote_goog.yaml`. The file still holds public dollar prices under other names, as §8.4 permits.

**Author review status:**
- [ ] V-01 and V-06 values checked against the papers
- [ ] Market comparison reviewed for the report

### Entry 19 — 2026-11-08 — Cboe terms of use, and the dividend

**Tool:** Claude Code, Claude Opus 5.5.

**Author's request:** Check the Cboe terms of use and try calibrating GOOG to the current surface. Include the dividend in the pricing.

**Cboe terms (cboe.com/terms, Section 2):** one copy of the Materials may be viewed, printed and downloaded for personal, non-commercial use. Copying, storing in an electronic retrieval system, creating a derivative work, publishing, distributing, or otherwise using the Materials needs Cboe's prior written consent. The terms do not address delayed quotes specifically, and the AI could not read Cboe's Market Data Policies PDF. This is the AI's reading of the page and is not legal advice.

**Outcome:** the AI did not download the Cboe chain and did not calibrate to it, because a calibration used in a report would need that consent.

**AI error disclosed:** in Entry 18 the AI stored and pushed one Cboe quote (`configs/market_quote_goog.yaml`, `results/market_comparison.json`, `SPEC.md` §8.4) and labelled it Public in `DATA_CONTRACT.md` §2.4 before reading these terms. That label may be wrong. The author must decide what to do. Until then the AI has not pushed anything further, and the updated `results/market_comparison.json` is left uncommitted.

**Dividend:** `q = 0.0031`, from Alphabet's declared $0.22 quarterly dividend. The run is `goog_market_contract-90258c0b`. The model price moved from $20.29 to $20.51, and the mid quote is $34.00, so the model is 39.7% below the mid.

**Author review status:**
- [ ] Decision on the stored Cboe quote
- [ ] Source of GOOG option data for calibration
