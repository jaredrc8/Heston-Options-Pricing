# AI Use Log

This file records every material use of AI tools in this project, as committed to in Section E and Table A3 of the ENSF 620 project proposal. Entries are added as the work happens and are summarized as a disclosure in the final report.

## Tools

| Tool | Model | Access mode |
|---|---|---|
| Claude Code (VS Code extension) | Claude Opus 5.5 (`claude-opus-5-5`) | Reads the repository and writes files into the working tree when the author asks. Commits nothing. |
| Claude (claude.ai chat) | — | Chat only (planned in Table A3; no use logged yet). |

## Repository access (change from the Table A3 plan)

Table A3 lists *repository write access* as out of bounds and limits the tool to *suggestions in chat, which I read, modify, and test myself*. On 2026-10-06 the author granted Claude Code write access to the working tree. **Only the author commits and pushes to Git.** AI-written files stay uncommitted until the author has reviewed them.

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
