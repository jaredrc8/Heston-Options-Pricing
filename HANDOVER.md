# Handover: Heston LSMC Pricer, out-of-sample accuracy on GOOG (ENSF 620)

State of the project on 2026-11-15, for whoever picks it up next, whether a new AI session or the owner. Read this first, then `SPEC.md` (**v1.13**, written by the owner on 2026-11-15) and `DATA_CONTRACT.md` (v0.9), which govern the work. `AI_USE.md` holds the detailed log (Entries 1–28).

Owner: Jared Crebo. Next gate G2 is 2026-11-20; final submission is due 2026-12-04.

**Main point for a new session:** spec v1.13 changed the question. Most of the engine is done, but the headline result it now asks for, LSMC prices of held-out GOOG options compared with the market, has **not been computed**. Section 5 lists what v1.13 requires that the code does not yet do.

---

## 1. Where things stand

- **The question (spec §2.1, reworded 2026-11-16):** how accurately does a Heston model price options contracts within its calibrated sample data and outside it?
  - **Within** means the 2026-10-13 calibration grid.
  - **Outside** means the held-out test population of §2.2.
  - The rest of the spec is unchanged.
- **How it is answered:**
  - Calibrate to GOOG's 2026-10-13 surface (iVolatility, `data/goog_IV_Oct13.csv`).
  - Price every contract in the test population as an American option with the LSMC engine, with Monte Carlo intervals.
  - Compare with bid, ask and mid, and report aggregate accuracy with bootstrap intervals.
  - Run the same test with the published 2015 Google parameters as a comparison model.
- **Test population (spec §2.2):** GOOG options on 2026-10-13 that pass filters F1–F8 and traded that day (`volume > 0`), excluding the calibration grid. Calls and puts both appear, because F5 keeps the out-of-the-money side.
- **Calibration status:**
  - The 2026-10-13 fit ended at ladder step 3, meaning no fit was accepted.
  - Full fit: RMS implied-vol error 0.01049, against the 0.01 limit. With `kappa` fixed at the cited 1.4554: 0.01235.
  - Under v1.13 §6.7 step 3, the lowest-RMS fit (the full fit) is still used for the test and the reference contract, and must be labelled **REJECTED**, with its RMS and the 0.01 limit, wherever it appears.
- **What exists:**
  - The engine and all checks under v1.12.
  - A v1.12 market test that priced held-out contracts through the CRR tree using the model's European implied vol. Under v1.13 that becomes only the "tree shortcut" diagnostic.
  - The in-sample 2015 engine check (§8.4).
  - The out-of-range tests (§8.6).
- **Removed by v1.13:** NVDA, AMD and INTC requirements; multi-asset (EXT-A, V-20 to V-28); C++ (EXT-B, V-30); correlation estimation; the calibration-date sweep. GOOG uses 2026-10-13 only.
- **Not started:** the v1.13 test pricing, report, slides, `LICENSE`, and the `v1.0-final` tag.

---

## 2. Rules any agent must follow

These come from `SPEC.md` §15 (v1.13), `DATA_CONTRACT.md` §3–4, and standing owner instructions.

- **An agent never reads `data/` itself.** That means no file tools, `cat`, `head`, DataFrame previews, one-liners or ad hoc scripts. Structure is available only through `python scripts/inspect_schema.py <path>`.
- **Licensed-data scripts.** `DATA_CONTRACT.md` v0.9 §3.3 and spec §15 let any agent run any committed script on licensed data, provided two conditions hold:
  - its output is aggregate and Internal, never a row, quote, IV point or per-contract result;
  - **the owner has reviewed the script, or the change to it, as committed.** A changed script needs a new review before its next run on licensed data.

  So the rewritten `scripts/test_goog_market.py`, the new `scripts/validate_goog_data.py` and the changed `run.py calibrate` path each wait for owner review.
- **Disclosure for `AI_USE.md` and the report.** Before v0.9, `estimate_correlations.py`, `compare_calibration_methods.py` and `test_goog_market.py` were run on licensed data at the owner's request. At the time, §3.3 named only three scripts. Contract §3.3 now records this.
- **Per-contract test values** (prices, SEs, intervals) go only to `data/derived/goog_test_prices.parquet`, which is Restricted. Only aggregates with every group `n >= 5` may appear in `results/`.
- **No absolute market prices** in `results/`, logs, figures or submitted files. The only exception is the published Google prices of 2015-02-02 (spec §3).
- **No hand edits to `results/**`;** only pipeline scripts write there.
- **Commits and pushes** only when the owner asks, and only if `data/` is in `.gitignore` and `bash scripts/check_no_data.sh` passes immediately before the push. Work directly on `main`. **Do not add a `Co-Authored-By` line** to commit messages (owner's instruction, 2026-11-16); AI use is recorded in `AI_USE.md`.
- **Never change a pass rule, tolerance, seed, filter or bound to make a check pass** (spec §1.3). A failed check is a recorded result. Any spec change needs a new §22 row and, if a result had already been viewed, a disclosure.
- **Reference values and published parameters:** an agent must not generate or fill them in. The two owner-instructed exceptions are in spec §15.
- **The report, slides and claims** are written by the owner, not the agent.
- **Every task gets an entry in `AI_USE.md`:** what was done, decisions, errors found, and what was not done.
- **Owner preferences:**
  - Plain functions and dicts, in the notebook style; no dataclasses or validation layers. Spec §10.1 now requires this.
  - Explain project IDs (G2, V-19, T3, OD-4) in plain words whenever you mention them.
  - Keep claims honest: label in-sample and rejected results as such.
- **Cboe data:** Cboe's terms require written consent to store or publish its data, so do not download or store it. One quote was removed on 2026-11-08 but remains in earlier Git history. The owner did not ask for the history to be rewritten.

---

## 3. Environment quirks

- **Python:** use `.venv/bin/python`. There is no `python` on the PATH, and the version is 3.14.
- **No editable install:** macOS marks the install's `.pth` file as hidden, so an editable `pip install` doesn't work. Every script adds `python/` to `sys.path` itself, and pytest uses `pythonpath = ["python"]`.
- **zsh, not bash:** `PIPESTATUS` doesn't work, so check exit codes directly. Piping into `| tail` hides a failing exit code; this once let a lint error through.
- **Smoke path:** `bash scripts/smoke.sh` needs `PYTHON=.venv/bin/python`.
- **Lint:** `.venv/bin/ruff check python scripts tests` (line length 130).
- **Tests:**
  - Fast tests: `.venv/bin/python -m pytest -m fast -q`, about 15 s.
  - `scripts/verify_outputs.py` takes about 2 minutes and writes `results/verification_table.csv`. V-17 (clean-clone smoke) runs only when the working tree is clean.
- **Reported runs** (`reported: true`) refuse a dirty tree. Changes under `results/` don't count.

---

## 4. Repository map

```text
SPEC.md, DATA_CONTRACT.md, README.md, AI_USE.md, HANDOVER.md
configs/   calibrate_goog.yaml            GOOG calibration, 2026-10-13 (raw_iv_file key)
           market_inputs_goog_oct13.yaml  y = 4.48%, q_GOOG = 0.0026 (v1.13 required name)
           published_params.yaml          Google 2015 set (Burkovska Table 8, row J_CF), cited kappa 1.4554
           goog_2015_contract.yaml        §8.4 engine check: $525 put, 2015-02-02, M = 240
           market_inputs_goog_2015.yaml   r = 0.15%, q = 0
           google_2015_american_puts.csv  401 published Google puts (paper's Appendix A, Table 10)
           governance.yaml                market_use_authorized: true (owner's instruction)
           smoke.yaml
           core_goog_production.yaml      v1.12 reported run (published params, Oct 14); not the v1.13 reference run
           market_inputs.yaml, splits.yaml, calibrate_{nvda,intc}.yaml, explore_{nvda,amd,intc}.yaml,
           core_amd_production.yaml       out of scope since v1.13; kept as a record
python/hestonpricer/   analytic, iv, tree, simulate, lsmc, surface, calibrate, errors
                       (correlation.py is out of scope since v1.13)
python/validate_raw.py, python/make_fixtures.py
scripts/   run.py (calibrate, price, sweep, compare), verify_outputs.py, make_figures.py, smoke.sh,
           check_no_data.sh, inspect_schema.py, test_goog_market.py, compare_to_market.py,
           test_goog_out_of_range.py
           estimate_correlations.py, compare_calibration_methods.py   out of scope; kept as a record
           validate_goog_data.py          REQUIRED by v1.13; does not exist yet
tests/     test_checks.py (one test per check), test_units.py, reference_impl.py,
           reference_values.yaml (V-01, V-06), fixtures/
results/   Internal outputs only; written by scripts, never by hand
data/      Git-ignored and Restricted: goog_IV_Oct13.csv (the only GOOG option date),
           nvda_amd_intc_goog_minute/ (Massive bars; validation only, pricing must not read it),
           nvda_amd_intc_IV_Oct13-15.csv and nvda_amd_intc_minute/ (not read by any reported run)
prototype/ the owner's earlier code and notebook, not part of the assessed work
```

`results/heston_params.csv` and two PNGs in `results/` predate the contract and show spot levels. They are Git-ignored and must never be committed or submitted.

---

## 5. What v1.13 requires that the code does not yet do

The list is roughly in dependency order. Items marked **(data)** read licensed data, so the owner must review the committed script before an agent runs it (contract §3.3).

**Progress (2026-11-16, `AI_USE.md` Entries 30–31):**
- **Part A is done**, including the VZ-10 to VZ-13 figures, tested on synthetic output only.
- **Reviewed `run.py`, run on GOOG (items 14, 18, 19):**
  - The recalibration reproduces RMS 0.01049 at ladder step 3, now in the v1.13 schema.
  - The reference contract under parameter set (a), labelled REJECTED: Bermudan 0.11677 ± 0.00019, premium 0.00377 (95% interval 0.00355–0.00399). T1 and T2 are met.
  - The comparison and sweeps are done (18 configurations in 77 s).
- **Part B scripts are written and committed:** `validate_goog_data.py` (item 12) and the rewritten `test_goog_market.py` (item 13).
  - `test_goog_market.py` also prices the calibration grid, for the reworded question about accuracy within and outside the calibrated sample.
  - **Neither has run on licensed data.** Both wait for owner review (contract §3.3).
- **Expected runtime:** about 10–20 minutes with `--workers 8`, double that at 200,000 paths. This is estimated from a production-size fixture run (148 contracts, 347 s); budget B4 is 2 hours.
- **V-04 still FAILS on one call case;** this needs an owner decision (section 10).

**Update 2026-11-16 (`AI_USE.md` Entry 32):** both part B scripts were approved and run.
- **V-00:** 0 violations.
- **Accuracy test:** 464 s, with B4 and T3 met (98.7%).
- **Calibrated (REJECTED) set:** 55% inside the spread out of sample and 56% in sample; median |IV error| 0.0095 out of sample and 0.0062 in sample.
- **Published 2015 set:** 0% inside the spread.
- **Figures:** regenerated.
- **Remaining:** `verify_outputs.py` from a clean tree, which waits on the prototype notebooks (contract §3.1, outputs cleared), and the V-04 decision.

**A. Changes that don't touch licensed data (an agent can do these now)**

1. **Version constants.** `scripts/run.py` has `SPEC_VERSION = "1.12"`; set it to `"1.13"`. Do the same for `spec_version` in the `goog_market_test.json` schema (§12.3, §12.6).
2. **Config keys (§12.1).**
   - The v1.13 config has no `tier` key, but `run.py` requires `tier` and checks `tier == "CORE"`. Drop it from `CONFIG_KEYS` and from every config.
   - `contract` gains `kind: put|call`.
   - Unknown keys must still raise `E_CONFIG`.
3. **Calls in the LSMC engine (§2.1, §4.1, §10.1).**
   - `lsmc.py` and `simulate.py` have no `kind`; payoffs are puts only.
   - Add `contract["kind"]` with the call payoff `max(S - K, 0)`.
   - Extend V-04 (LSMC against a binomial tree with constant volatility) to calls with `q = 0.0026`, for the same four `(sigma, K)` cases.
4. **Single asset only (§10.3).** `simulate` still accepts `d` assets with an eigendecomposed correlation matrix. v1.13 requires `corr == [[1.0]]` and `len(params) == len(market["q"]) == 1`. Simplify it and its precondition checks.
5. **Rejected-fit schema (§6.7, §12.5).**
   - `calibrate.py` writes `rejected_full_fit` and `rejected_fixed_kappa_fit`. v1.13 wants a single `rejected_fit` holding `{"from_step", "params", "fixed_params", "rms_iv_residual", "stable", "limit": 0.01}`, taken from the lowest-RMS fit across steps 1 and 2.
   - Update the readers in `run.py`, `make_figures.py` and `test_goog_market.py`.
6. **Reported runs may use the rejected fit (§6.7 step 3).**
   - `run.py` currently refuses `allow_rejected_fit` when `reported: true`. v1.13 requires the reference run to use the rejected fit and set `parameter_status: "rejected"`.
   - Add `configs/core_goog_reference.yaml` exactly as §12.1 shows: market mode, snapshot 2026-10-13, `params_source: results/calibration_goog/calibration.json`.
   - Decide what happens to `core_goog_production.yaml`: set it unreported or remove it.
7. **Calibration-date sweep removed.** Delete the `ALL_DATES` loops in `run.py` (around lines 418 and 458).
8. **Integer inputs (§14.3).** `simulate.py` casts `strike` and `maturity` to float. Add a test that integer `maturity` and `strike` give the same price as floats through `price_bermudan`.
9. **Tests (§13, §16).**
   - Remove `test_V20`–`test_V24` and `test_V30`.
   - Add `test_V00…`, `test_V19…` and the V-04 call cases.
   - Point V-13 and V-14(b, c) at the GOOG file.
   - The `fast` marker must include the synthetic-file tests of `validate_goog_data.py`.
10. **Figures (§12.9).**
    - Add VZ-1e: the 2026-10-13 surface, labelled REJECTED with its RMS and the 0.01 limit inside the figure, over K/F 0.70–1.30 and tenor 0.06–1.40.
    - Add VZ-10 to VZ-13 from `goog_market_test.json`.
    - VZ-4 and VZ-5 use parameter set (a).
    - Drop VZ-1a, VZ-1c and VZ-1d (AMD, NVDA, INTC) from the evidence map, and repoint the V-13 row from `/assets/AMD/...` to GOOG.
11. **README (§13):** rewrite it for the v1.13 question, test population, data boundary, and what a reviewer can and cannot reproduce.

**B. New or rewritten licensed-data scripts (written by an agent, reviewed by the owner, then run)**

12. **`scripts/validate_goog_data.py` (V-00, §11) (data).**
    - Checks `data/goog_IV_Oct13.csv` and the GOOG rows of the Massive bars against contract §1.1 and §1.2.
    - Writes `results/validation_goog.json` and prints counts, rule names and row indices only.
    - Needs synthetic-file tests on fixtures.
13. **Rewrite `scripts/test_goog_market.py` to §8.5 (data).** The current script prices with the tree shortcut and writes the v1.12 schema. v1.13 requires:
    - **Pricing:** each test contract priced with `price_bermudan` as an American option. Use `M = ceil(252 T)`, `n_sub = 1`, 100,000 paths per set, the five production seeds and degree 3. Paths may be shared per expiration and seed.
    - **Per-contract output:** `V`, `SE`, the 95% interval, the Monte Carlo European value, the premium and the LSMC diagnostics, written to `data/derived/goog_test_prices.parquet` only.
    - **Metrics:** median relative error against mid, share inside the spread, share of mids inside the Monte Carlo interval, and implied-vol error, inverting `V` with `american_implied_vol` (500 steps) and counting failures.
    - **Breakdowns** by maturity, K/F (0.70–0.90, 0.90–1.10, 1.10–1.30) and kind, keeping only groups with `n >= 5`.
    - **Bootstrap** that resamples expirations: 2000 resamples, seed 620200, 95% percentile intervals.
    - **Tree-shortcut diagnostic:** the current method, kept for parameter set (a).
    - **T3:** for at least 90% of contracts, the Monte Carlo half-width is at most half the market's half-spread. If missed, rerun at 200,000 paths.
    - **V-19:** test set and grid disjoint by `option_symbol`. The current script excludes by `(expiration, strike)`.
    - **Output schema** exactly as §12.6.
    - **Runtime:** budget B4 is 2 hours for both parameter sets and all seeds. It hasn't been measured; per-contract regressions over up to about 350 exercise dates will dominate. Time one expiration first.
14. **`run.py calibrate` changes (data):** after item 5, rerun the GOOG calibration so `results/calibration_goog/calibration.json` matches §12.5.

**C. Runs, after A and B, in this order**

15. `validate_goog_data.py` (V-00).
16. `run.py calibrate --config configs/calibrate_goog.yaml --results-root results/calibration_goog`.
17. `test_goog_market.py` (the headline, plus T3 and B4).
18. `run.py price --config configs/core_goog_reference.yaml`: precision targets T1 (pooled SE within 0.5% of the price) and T2 (premium SE within 10% of the premium).
19. `run.py sweep` on the reference contract with parameter set (a) (§9.3; budget B2 is 6 hours).
20. `make_figures.py`, then `verify_outputs.py` from a clean, committed tree.

---

## 6. Key results so far

All prices are fractions of the initial stock price unless stated otherwise.

**2026-10-13 calibration** (`results/calibration_goog/calibration.json`, uncommitted): ladder step 3. Full fit RMS 0.01049; fixed-`kappa` fit 0.01235. The RMS by maturity (shown in the v1.12 market test) was 0.020 under 3 months, 0.012 at 3–6 months, 0.006 at 6–12 months and 0.010 over a year. Heston misfits short tenors, and `kappa` is poorly identified from one day.

**v1.12 market test** (`results/goog_market_test.json`, uncommitted; the v1.12 schema). **Tree shortcut, not LSMC.** Under v1.13 this is only the diagnostic of §8.5. 299 held-out contracts that traded:

| | Rejected 2026-10-13 fit | Published 2015 parameters |
|---|---|---|
| Median implied-vol error | 0.8 vol points | about 11.4 points (11.7 too low on average) |
| Model price inside bid–ask | 58% | 0% |
| Median price error vs mid | 4.5% | 57% |

These aggregates were viewed before v1.13 was written (spec §22, row 1.13). The LSMC prices, moneyness and kind breakdowns, bootstrap intervals and T3 had not been computed.

**§8.4 engine check, Feb 2, 2015** (in-sample: the parameters were fitted to these prices):
- The $525 put with T = 0.9507: model $45.25 ± $0.09 against the published $46.05 (-1.7%).
- All 401 published puts: median gap $0.40, largest $1.16; 248 within 2% and 291 within 5%.

**§8.6 out-of-range tests** (published parameters; maturities of 1 month, 3 years and 5 years; strikes 0.4–1.8 times spot):
- 15 of 17 pass the behaviour checks. The two flags are LSMC low bias.
- One flag also exposes a check that is too strict. There is no exercise at t = 0, so the lower bound should be the value of exercising at the first date, not intrinsic value.
- v1.13 §8.6 still says "intrinsic", and the check was not changed after viewing.

**v1.12 reference run, kept as a record** (`core_goog_production`, published parameters, Oct 14, q = 0): Bermudan 0.07334 ± 0.00014; European 0.06916; premium 0.00420 (95% interval 0.00399–0.00442); constant-vol Bermudan 0.07359. **v1.13 replaces this** with the reference contract priced under the rejected 2026-10-13 fit.

**Out-of-scope work, kept as a record:**
- **Fits on 2026-10-14:**
  - NVDA: RMS 0.0082, `kappa` at its bound.
  - INTC: RMS 0.0092, bounds hit.
  - AMD: rejected, RMS 0.0105.
- **Correlations:** four-ticker correlations computed.
- **Calibration method comparison:** the paper's method gave worse implied-vol fits (about 0.016). See `AI_USE.md` Entries 17–25.

---

## 7. Verification status

The committed `results/verification_table.csv` predates v1.12 and v1.13. Its last result was 19 PASS, 1 FAIL, 6 NOT_RUN, with V-13 failing on the AMD fit and the multi-asset and C++ checks not run. The v1.13 list:

| ID | What it checks | Status for v1.13 |
|---|---|---|
| V-00 | Raw-data validation of the GOOG IV file and GOOG minute bars | `validate_goog_data.py` written, with synthetic-file tests passing. Not run on GOOG data: it awaits owner review |
| V-01 | Semi-analytic European against published prices (Fang & Oosterlee, via Chan) | PASS (v1.12 table) |
| V-02 | Put–call parity; Black–Scholes limit when vol-of-vol is tiny | PASS |
| V-03 | Monte Carlo European against semi-analytic, 40 options | PASS |
| V-04 | LSMC against a binomial tree with constant volatility | Puts PASS. **Calls FAIL on one case:** σ 0.5, K 0.9 is 3.16 SE from the tree. The other three call cases are 1.24–2.02 SE. Diagnosis: LSMC low bias. Early exercise is worthless here (q 0.26%, r 3.96%), and the Monte Carlo European matches Black–Scholes within 0.4 SE, but regression error triggers some early exercises. Not a payoff or path defect |
| V-05 | One exercise date gives the European price | PASS |
| V-06 | Bermudan against published American puts (Ikonen & Toivanen) | PASS |
| V-07 | Vectorized LSMC against an independent scalar version | PASS |
| V-08 | The same run repeated gives identical numbers | PASS |
| V-09 | Price stable across basis degrees | Inputs now exist (parameter set (a)); run in `verify_outputs.py` |
| V-10 | Price stable from 4 to 8 sub-steps | Same as V-09 |
| V-11 | Production seeds agree | Same as V-09 |
| V-12 | Calibration recovers known parameters on synthetic data | PASS |
| V-13 | GOOG 2026-10-13 calibration meets RMS ≤ 0.01 and is stable | **FAIL**, recorded: ladder step 3, rejected fit RMS 0.01049 |
| V-14 | Implied-vol conversion: (a) tree round trip; (b, c) checks on the GOOG file | PASS: 77 of 85 near-the-money pairs agree (0.906, rule ≥ 0.90); median own-vs-vendor gap 0.0008 (rule ≤ 0.02) |
| V-15 | Configured runs complete, finite, with matching hashes | `core_goog_reference` has now run; rerun the check |
| V-16 | No licensed data or secrets in Git | PASS |
| V-17 | Smoke test passes from a fresh clone | PASS; rerun |
| V-18 | Every figure and table traces to a result file | Rerun after the new figures |
| V-19 | Test set and calibration grid share no `option_symbol` | Implemented; reads `test_set.grid_overlap` once `test_goog_market.py` has run |
| B1, B2 | Runtime: one pricing ≤ 30 s (about 0.85 s measured); full sweep ≤ 6 h (72 s measured) | PASS under v1.12 |
| B3, B4 | Smoke ≤ 10 min from a clean clone; full test pricing ≤ 2 h | B3 passes via V-17; B4 is estimated well inside the budget and recorded by `test_goog_market.py` |
| T1, T2, T3 | Precision targets for the reference price, its premium, and the test pricing | T1 and T2 met under set (a); T3 is computed by `test_goog_market.py` (rerun at 200,000 paths if missed) |

**Gates (spec §17.1):**
- **G1** (Nov 14) needs V-01 to V-12 and V-15, B1, B2, and the governance gate. All passed in the last table, before V-04's calls were added.
- **G2** (Nov 20) needs V-00, V-04 with calls and V-19 to pass; the out-of-sample test run and frozen; V-13 and V-14 recorded; T3 resolved. **All of section 5 sits on the path to G2.**

---

## 8. How the project got here

Spec changes are recorded in `SPEC.md` §22; the full detail is in `AI_USE.md`.

1. Built the spec layout, plain-dict API (v1.1), engine, tests, CI and data protections. The old code moved to `prototype/`.
2. The owner authorized market use and agent commits and pushes (v1.2; contract §3.3), and the Treasury yields were entered.
3. Validation flagged 518 same-day expiries; the owner relaxed the rule to `expiration >= date` (v1.3).
4. The AMD calibration failed the limit. The owner chose the Google 2015 published set and its cited `kappa` (v1.4), and the published run became the reported result.
5. Added the visualization plan and PyVista surfaces, Google data, and four-ticker correlations; the question moved to GOOG (v1.5, v1.6, v1.9).
6. V-01 and V-06 benchmarks were found and passed. V-06 switched to Ikonen & Toivanen before it was run (v1.7).
7. A Cboe quote showed the 2015 parameters are far too cheap today. Cboe's terms ruled out keeping it, and the Feb 2015 Google put became the official comparison (v1.8–v1.10).
8. Tried NVDA and INTC and tested the paper's calibration method. The owner restricted the project to GOOG (v1.11).
9. The owner bought GOOG Raw IV for 2026-10-13. The fit ended at ladder step 3, and the tree-shortcut market test was run (v1.12).
10. **v1.13 (owner, 2026-11-15):**
    - The question became out-of-sample accuracy on GOOG, with LSMC test pricing, bootstrap intervals and T3.
    - Ladder step 3 now evaluates the rejected fit and labels it.
    - V-00 and V-19 were added, and V-04 gained calls.
    - Multi-asset, C++, correlation and other-ticker requirements were removed.
    - §15 was widened to reviewed, committed scripts.

---

## 9. Git state at handover

- **Committed locally on `main`, not pushed:**
  - `a92753d`, the GOOG market test (spec v1.12, contract v0.8);
  - the commit holding spec v1.13, contract v0.9, this handover, the GOOG calibration and the v1.12 market-test results.

  `origin/main` is at `ebd457a`. Push only when the owner asks.
- **Part A and the part B scripts** are committed locally; see `git log` and `AI_USE.md` Entry 31.
- **Next:** once the owner has reviewed them, run `validate_goog_data.py`, then `test_goog_market.py --workers 8`, then `make_figures.py` and `verify_outputs.py` from a clean tree.

---

## 10. Open items and decisions for the owner

1. **Review the licensed-data scripts** before an agent runs them: the new `validate_goog_data.py`, the rewritten `test_goog_market.py`, and the `run.py` calibrate and price changes.
   - The `run.py` changes are committed.
   - A `market` calibration now writes to the folder named by `params_source`, which is `results/calibration_goog/`.
   - A `market` price run reads from that same folder. It previously read `results/calibration.json`, which is the old AMD fit.
2. **V-04 call failure** (σ 0.5, K 0.9, 3.16 SE against the 3 SE limit). §1.3 forbids loosening the rule to make it pass. G2, the Nov 20 gate, requires V-04 with calls to pass. The options:
   - (a) Record the FAIL and disclose it as LSMC low bias.
   - (b) Change the engine's regression for calls and rerun. This is a real engine change, not a tolerance change, but it was found after viewing, so it needs disclosure.
   - (c) Change the rule through a §22 row disclosed as made after viewing. For example, a one-sided low-bias allowance like V-06's.
3. **Open decisions in spec §19:**
   - **OD-4:** confirm the iVolatility header against contract §1.2, and that `stock_price_for_iv` is synchronous with the quotes. The test's normalized prices depend on it.
   - **OD-6:** confirm the reference contract (K = 1, T = 1, M = 50).
   - **OD-8:** raw-file hashes.
   - **OD-9:** licence.
   - **OD-12:** whether to add a constant-volatility baseline to the out-of-sample test.
4. **Fate of out-of-scope files:** keep as a record or remove. Covers `correlation.py`, `estimate_correlations.py`, `compare_calibration_methods.py`, the NVDA, AMD and INTC configs, and their results.
5. **The too-strict §8.6 check:** the lower bound should be the value of exercising at the first date, not intrinsic value. Any change needs a §22 row disclosing it was found after viewing.
6. **Check transcribed values against the papers:** Table 8, Table 10, and the V-01 and V-06 values were read through automated extracts.
7. **A recent day of GOOG Raw IV** (recommended earlier for an out-of-time test) is outside v1.13, which uses 2026-10-13 only. Adding it would need a spec change.
8. **Report and slides** (owner). Required disclosures and limitations are listed in spec §18. They include:
   - the same-day test on neighbouring contracts;
   - the REJECTED label;
   - §6.7 step 3 written after the RMS was seen;
   - the §22 rows marked as viewed;
   - AI use, including agent commits and pushes (OD-7) and the scripts run beyond contract §3.3's list.
9. **Release:** `LICENSE`, the `v1.0-final` tag, and the Route A package verified as spec §13 describes.

---

10. **The in-sample addition to `test_goog_market.py`.** For the reworded question, the script also prices the calibration grid with the same metrics, under `in_sample_results` and `in_sample_set`. §8.5 and §12.6 don't describe this yet; the owner may want to add it through a §22 row.
11. **The uncommitted §2.3 edit in `SPEC.md`.** It removes "C++" and "contracts on more than one asset" from the out-of-scope list, and has no §22 row yet. §10.3 still requires one asset, and the engine enforces that since part A. The owner's `prototype/` changes are also uncommitted.

## 11. Useful commands

```bash
PY=.venv/bin/python
PYTHON=$PY bash scripts/smoke.sh                     # reviewer smoke path
$PY -m pytest -m fast -q                             # fast tests
.venv/bin/ruff check python scripts tests            # lint
$PY scripts/verify_outputs.py                        # all checks, writes results/verification_table.csv
$PY scripts/run.py calibrate --config configs/calibrate_goog.yaml --results-root results/calibration_goog
$PY scripts/run.py price --config configs/core_goog_reference.yaml     # once the config exists
$PY scripts/test_goog_market.py                      # headline test, aggregates only (after rewrite and review)
$PY scripts/compare_to_market.py                     # §8.4 Feb 2015 engine check
$PY scripts/make_figures.py                          # figures and evidence map
bash scripts/check_no_data.sh                        # run before every push
```
