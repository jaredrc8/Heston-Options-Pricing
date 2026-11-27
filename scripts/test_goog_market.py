"""
Accuracy of the Heston LSMC engine on GOOG options of 2026-10-13, within and outside the
calibrated sample (SPEC.md 2.1 and 8.5).

    python scripts/test_goog_market.py [--n-paths 100000] [--workers 1]

Populations:
  out of sample  the test population of SPEC.md 2.2: contracts after filters F1 to F8 that
                 traded that day and are not in the calibration grid (the 8.5 headline);
  in sample      the calibration grid itself, the contracts the fit was made to.

Parameter sets: (a) the 2026-10-13 calibration (labelled REJECTED at ladder step 3) and
(b) the published 2015 Google set. Every contract is priced with LSMC as an American option:
M = ceil(252 T) exercise dates, one sub-step, the production seeds, basis degree 3. Paths are
simulated once per expiration and seed and shared by the contracts of that expiration, so each
contract's price equals price_bermudan on that contract alone; their errors are correlated.

Per-contract values go only to data/derived/goog_test_prices.parquet (Restricted). The JSON
output holds aggregates only, and every group has n >= 5 (DATA_CONTRACT.md section 4.3).
"""

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import numpy as np
import pandas as pd
import yaml

from hestonpricer import surface
from hestonpricer.calibrate import model_implied_vols
from hestonpricer.errors import EXIT_CODES, HestonError
from hestonpricer.iv import american_implied_vol
from hestonpricer.lsmc import evaluate_policy, fit_policy
from hestonpricer.simulate import simulate
from hestonpricer.tree import crr_price

SPEC_VERSION = "1.13"
PRODUCTION_SEEDS = [620001, 620002, 620003, 620004, 620005]
BASIS_DEGREE = 3
BOOTSTRAP_SEED = 620200
BUDGET_B4_SECONDS = 7200
NAMES = ("v0", "kappa", "theta", "xi", "rho")
MATURITY_BUCKETS = [("under 3 months", 0.0, 0.25), ("3-6 months", 0.25, 0.5),
                    ("6-12 months", 0.5, 1.0), ("over 12 months", 1.0, 99.0)]
MONEYNESS_BUCKETS = [("0.70-0.90", 0.70, 0.90), ("0.90-1.10", 0.90, 1.10), ("1.10-1.30", 1.10, 1.30 + 1e-12)]
METRICS = ("median_rel_price_error", "share_inside_spread", "share_mid_inside_mc_ci95",
           "rms_iv_error", "median_abs_iv_error")

DEFAULTS = {
    "raw_file": ROOT / "data" / "goog_IV_Oct13.csv",
    "calibration": ROOT / "results" / "goog" / "calibration.json",
    "market_inputs": ROOT / "configs" / "inputs" / "market_inputs_goog_oct13.yaml",
    "output": ROOT / "results" / "goog" / "market_test.json",
    "prices_out": ROOT / "data" / "derived" / "goog_test_prices.parquet",
}


def check_governance():
    path = ROOT / "configs" / "governance.yaml"
    if not (path.exists() and yaml.safe_load(path.read_text()).get("market_use_authorized") is True):
        raise HestonError("E_GOV_UNAUTHORIZED", "market_use_authorized is not true")


# ----------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------

def load_parameter_sets(calibration_file, symbol):
    """(a) the snapshot calibration, accepted or REJECTED; (b) the published Google set."""
    document = json.loads(Path(calibration_file).read_text())
    entry = document["assets"][symbol]
    if entry["ladder_step"] in (1, 2):
        fitted = {"params": entry["params"], "status": "accepted", "ladder_step": entry["ladder_step"],
                  "rms_iv_residual": entry["rms_iv_residual"], "limit": 0.01}
    elif entry["ladder_step"] == 3:
        rejected = entry["rejected_fit"]
        fitted = {"params": rejected["params"], "status": "rejected", "ladder_step": 3,
                  "rms_iv_residual": rejected["rms_iv_residual"], "limit": rejected["limit"],
                  "from_step": rejected["from_step"]}
    else:
        raise HestonError("E_CALIB_FAILED", "ladder step 4: no calibrated parameter set (SPEC.md 6.7)")

    published = yaml.safe_load((ROOT / "configs" / "inputs" / "published_params.yaml").read_text())["sets"]["GOOG"]
    return {
        "calibrated_2026_10_13": fitted,
        "published_2015": {"params": {name: float(published[name]) for name in NAMES}, "status": "published",
                           "source": f"{published['citation']}, {published['table_or_page']}"},
    }


def contract_table(rows, population, r, q):
    """The columns pricing and scoring need. Market quantities are divided by the stock price (SPEC.md 3)."""
    spot = rows["spot"].to_numpy(float)
    return pd.DataFrame({
        "option_symbol": rows["option_symbol"].astype(str).to_numpy(),
        "population": population,
        "expiration": rows["expiration"].to_numpy(),
        "tenor": rows["tenor"].to_numpy(float),
        "kind": np.where(rows["call_put"].to_numpy() == "C", "call", "put"),
        "style": rows["style"].to_numpy(),
        "strike": rows["strike"].to_numpy(float) / spot,
        "k_over_f": rows["strike"].to_numpy(float) / (spot * np.exp((r - q) * rows["tenor"].to_numpy(float))),
        "bid": rows["bid"].to_numpy(float) / spot,
        "ask": rows["ask"].to_numpy(float) / spot,
        "mid": rows["mid"].to_numpy(float) / spot,
        "market_iv": rows["own_iv"].to_numpy(float),
    })


def build_populations(raw, symbol, date, r, q):
    quotes, _ = surface.filter_quotes(raw, {"symbol": symbol, "snapshot_date": date, "r": r, "q": q})
    grid = surface.select_grid(quotes, {"contract_strike": 1.0, "contract_maturity": 1.0})
    in_grid = quotes["option_symbol"].isin(set(grid["option_symbol"]))
    held_out = quotes[~in_grid]
    test = held_out[held_out["volume"] > 0]

    counts = {
        "n_after_filters": int(len(quotes)), "n_in_grid_removed": int(in_grid.sum()),
        "n_zero_volume_removed": int((held_out["volume"] <= 0).sum()), "n_test": int(len(test)),
        "n_calls": int((test["call_put"] == "C").sum()), "n_puts": int((test["call_put"] == "P").sum()),
        "n_expirations": int(test["expiration"].nunique()),
        "grid_overlap": len(set(test["option_symbol"]) & set(grid["option_symbol"])),
    }
    in_sample_counts = {"n": int(len(grid)), "n_calls": int((grid["call_put"] == "C").sum()),
                        "n_puts": int((grid["call_put"] == "P").sum()),
                        "n_expirations": int(grid["expiration"].nunique())}
    contracts = pd.concat([contract_table(test, "out_of_sample", r, q),
                           contract_table(grid, "in_sample", r, q)], ignore_index=True)
    return contracts, counts, in_sample_counts


# ----------------------------------------------------------------------
# Pricing
# ----------------------------------------------------------------------

def price_expiration(job):
    """
    LSMC prices for every contract of one expiration under one parameter set, pooled over seeds.
    Each seed simulates regression set A and evaluation set B once, from independent child
    streams as in price_bermudan, and every contract of this expiration uses those paths.
    """
    params, r, q, tenor, contracts, n_paths, seeds = job
    market = {"r": r, "q": [q]}
    n_exercise = math.ceil(252 * tenor)
    base = {"strike": 1.0, "maturity": tenor, "n_exercise": n_exercise}
    sim = {"n_paths": n_paths, "n_substeps": 1, "basis_degree": BASIS_DEGREE}
    terms = [{"strike": c["strike"], "maturity": tenor, "n_exercise": n_exercise, "kind": c["kind"]} for c in contracts]

    per_seed = []
    for seed in seeds:
        child_a, child_b = np.random.SeedSequence(seed).spawn(2)
        paths = simulate([params], [[1.0]], market, base, sim, np.random.Generator(np.random.PCG64(child_a)))
        policies = [fit_policy(paths, contract, market, BASIS_DEGREE) for contract in terms]
        del paths
        paths = simulate([params], [[1.0]], market, base, sim, np.random.Generator(np.random.PCG64(child_b)))
        per_seed.append([evaluate_policy(policy, paths, contract, market) for policy, contract in zip(policies, terms)])
        del paths

    rows = []
    n_seeds = len(seeds)
    for i, contract in enumerate(contracts):
        outcomes = [seed_results[i] for seed_results in per_seed]

        def pooled(name, se_name):
            value = float(np.mean([o[name] for o in outcomes]))
            se = float(np.sqrt(np.sum([o[se_name] ** 2 for o in outcomes])) / n_seeds)
            return value, se

        V, se = pooled("bermudan", "se_bermudan")
        european, se_european = pooled("european", "se_european")
        premium, se_premium = pooled("premium", "se_premium")
        rows.append({"index": contract["index"], "n_exercise": n_exercise, "V": V, "se": se,
                     "ci_low": V - 1.96 * se, "ci_high": V + 1.96 * se,
                     "european": european, "se_european": se_european, "premium": premium, "se_premium": se_premium,
                     "skipped_dates": int(sum(o["skipped_dates"] for o in outcomes)),
                     "rank_deficient_dates": int(sum(o["rank_deficient_dates"] for o in outcomes)),
                     "min_itm_count": int(min(o["min_itm_count"] for o in outcomes))})
    return rows


def price_contracts(params, contracts, r, q, n_paths, seeds, workers):
    """Pooled LSMC price for every contract, in the order of `contracts`."""
    jobs = []
    for expiration, group in contracts.groupby("expiration", sort=True):
        tenor = float(group["tenor"].iloc[0])
        members = [{"index": int(i), "strike": float(row.strike), "kind": row.kind} for i, row in group.iterrows()]
        jobs.append((params, r, q, tenor, members, n_paths, seeds))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(price_expiration, jobs))
    else:
        results = [price_expiration(job) for job in jobs]
    priced = pd.DataFrame([row for rows in results for row in rows]).set_index("index").sort_index()
    return contracts.join(priced)


def add_model_iv(priced, r, q):
    """Invert each LSMC price with the 500-step American tree, as in SPEC.md 6.3. Failures stay NaN."""
    priced = priced.copy()
    priced["model_iv"] = [american_implied_vol(row.kind, row.V, 1.0, row.strike, row.tenor, r, q)
                          for row in priced.itertuples()]
    return priced


def tree_shortcut(params, contracts, r, q):
    """The v1.12 method, kept as a diagnostic: the model's European IV carried through the American tree."""
    sigma = model_implied_vols(params, contracts["strike"].to_numpy(float), contracts["tenor"].to_numpy(float), r, q)
    prices = []
    for row, s in zip(contracts.itertuples(), sigma):
        if not np.isfinite(s):
            prices.append(np.nan)
            continue
        steps = None if row.style == "A" else [500]
        prices.append(crr_price(row.kind, 1.0, row.strike, row.tenor, s, r, q, 500, exercise_steps=steps))
    return contracts.assign(V=prices, se=np.nan, model_iv=sigma)


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------

def arrays(rows):
    """The columns scoring needs, as plain arrays, so resampling is fast."""
    return {"V": rows["V"].to_numpy(float), "se": rows["se"].to_numpy(float), "mid": rows["mid"].to_numpy(float),
            "bid": rows["bid"].to_numpy(float), "ask": rows["ask"].to_numpy(float),
            "iv_error": rows["model_iv"].to_numpy(float) - rows["market_iv"].to_numpy(float)}


def score(a, idx):
    """Aggregate accuracy of the rows at positions idx (SPEC.md 8.5)."""
    V, mid, se, iv_error = a["V"][idx], a["mid"][idx], a["se"][idx], a["iv_error"][idx]
    ok = np.isfinite(iv_error)
    has_se = len(se) > 0 and bool(np.isfinite(se).all())
    return {
        "n": int(len(idx)),
        "median_rel_price_error": float(np.median(np.abs(V - mid) / mid)),
        "share_inside_spread": float(np.mean((a["bid"][idx] <= V) & (V <= a["ask"][idx]))),
        "share_mid_inside_mc_ci95": float(np.mean(np.abs(mid - V) <= 1.96 * se)) if has_se else None,
        "rms_iv_error": float(np.sqrt(np.mean(iv_error[ok] ** 2))) if ok.any() else None,
        "median_abs_iv_error": float(np.median(np.abs(iv_error[ok]))) if ok.any() else None,
        "mean_iv_error": float(np.mean(iv_error[ok])) if ok.any() else None,
        "n_iv_inversion_failed": int((~ok).sum()),
    }


def bootstrap(a, expirations, idx, n_resamples):
    """
    95% percentile intervals that resample expirations, since contracts of one expiration are
    correlated (SPEC.md 8.5). Each group restarts the generator at the same seed.
    """
    if not n_resamples:
        return {}
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    blocks = [idx[expirations[idx] == value] for value in np.unique(expirations[idx])]
    draws = {name: [] for name in METRICS}
    for _ in range(n_resamples):
        picked = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), size=len(blocks))])
        sample = score(a, picked)
        for name in METRICS:
            if sample[name] is not None:
                draws[name].append(sample[name])
    return {name: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
            for name, values in draws.items() if values}


def summarize(rows, n_resamples):
    """Overall and grouped metrics, each with bootstrap intervals; groups need n >= 5."""
    rows = rows.reset_index(drop=True)
    a = arrays(rows)
    expirations = rows["expiration"].to_numpy()

    def entry(mask, bucket=None):
        idx = np.flatnonzero(mask)
        result = {"bucket": bucket} if bucket else {}
        return {**result, **score(a, idx), "ci95": bootstrap(a, expirations, idx, n_resamples)}

    def grouped(column, buckets):
        values = rows[column].to_numpy(float)
        masks = [(name, (values >= low) & (values < high)) for name, low, high in buckets]
        return [entry(mask, name) for name, mask in masks if mask.sum() >= 5]

    kinds = rows["kind"].to_numpy()
    return {"overall": entry(np.ones(len(rows), dtype=bool)),
            "by_maturity": grouped("tenor", MATURITY_BUCKETS),
            "by_moneyness": grouped("k_over_f", MONEYNESS_BUCKETS),
            "by_kind": [entry(kinds == kind, kind) for kind in ("call", "put") if (kinds == kind).sum() >= 5]}


def histogram_min5(values, edges):
    """Counts per bin, with neighbouring bins merged until every bin has n >= 5 (SPEC.md 12.6)."""
    # An open last edge is closed at the largest value, so the JSON stays finite
    edges = [e if np.isfinite(e) else max(float(np.max(values)), edges[i - 1]) for i, e in enumerate(edges)]
    counts, _ = np.histogram(values, bins=edges)
    bins, low, n = [], edges[0], 0
    for i, count in enumerate(counts):
        n += int(count)
        if n >= 5:
            bins.append({"low": float(low), "high": float(edges[i + 1]), "n": n})
            low, n = edges[i + 1], 0
    if n and bins:
        bins[-1] = {**bins[-1], "high": float(edges[-1]), "n": bins[-1]["n"] + n}
    return bins


def precision_t3(rows, n_paths):
    """T3: Monte Carlo half-width at most half the market's half-spread for at least 90% of contracts."""
    # Ratio of the Monte Carlo half-width to the market half-spread; T3 asks for at most 0.5
    ratio = (1.96 * rows["se"]) / (0.5 * (rows["ask"] - rows["bid"]))
    share = float((ratio <= 0.5).mean())
    edges = [0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, np.inf]
    return {"T3_share": share, "T3_met": bool(share >= 0.90),
            "median_rel_se": float(np.median(rows["se"] / rows["V"])), "n_paths_used": n_paths,
            "half_width_to_half_spread": {"median": float(np.median(ratio)), "t3_limit": 0.5,
                                          "bins": histogram_min5(ratio.to_numpy(float), edges)}}


# ----------------------------------------------------------------------

def run(args):
    begin = time.perf_counter()
    check_governance()
    inputs = yaml.safe_load(Path(args.market_inputs).read_text())
    y = {str(k): v for k, v in inputs["treasury_1y_cmt_yield"].items()}[args.date]
    r, q = 2.0 * math.log(1.0 + float(y) / 2.0), float(inputs["dividend_yield"][args.symbol])
    sets = load_parameter_sets(args.calibration, args.symbol)

    raw = surface.load_raw_iv(args.raw_file)
    contracts, test_counts, in_sample_counts = build_populations(raw, args.symbol, args.date, r, q)
    del raw

    results, in_sample_results, priced_tables, timings = {}, {}, [], {}
    for name, entry in sets.items():
        start = time.perf_counter()
        priced = add_model_iv(price_contracts(entry["params"], contracts, r, q, args.n_paths, args.seeds, args.workers), r, q)
        timings[name] = time.perf_counter() - start
        results[name] = summarize(priced[priced["population"] == "out_of_sample"], args.bootstrap)
        in_sample_results[name] = summarize(priced[priced["population"] == "in_sample"], args.bootstrap)
        priced_tables.append(priced.assign(parameter_set=name))
        if name == "calibrated_2026_10_13":
            precision = precision_t3(priced[priced["population"] == "out_of_sample"], args.n_paths)
        print(f"{name}: priced {len(priced)} contracts in {timings[name]:.0f} s")

    fitted = sets["calibrated_2026_10_13"]
    test_rows = contracts[contracts["population"] == "out_of_sample"]
    shortcut = tree_shortcut(fitted["params"], test_rows, r, q)
    shortcut_summary = summarize(shortcut, 0)

    # Per-contract values are Restricted: they go under data/ only
    prices_out = Path(args.prices_out)
    prices_out.parent.mkdir(parents=True, exist_ok=True)
    keep = ["option_symbol", "population", "parameter_set", "expiration", "kind", "strike", "tenor", "n_exercise",
            "V", "se", "ci_low", "ci_high", "european", "se_european", "premium", "se_premium", "model_iv",
            "skipped_dates", "rank_deficient_dates", "min_itm_count"]
    pd.concat(priced_tables)[keep].to_parquet(prices_out)

    total = time.perf_counter() - begin
    label = {key: value for key, value in fitted.items() if key != "params"}
    document = {
        "schema_version": 1, "spec_version": SPEC_VERSION, "symbol": args.symbol, "snapshot_date": args.date,
        "units": "fraction_of_initial_spot", "r": r, "q": q,
        "note": ("Aggregates only. out_of_sample (results) is the SPEC.md 8.5 test population; in_sample_results "
                 "prices the calibration grid the fit was made to. Same snapshot for both."),
        "test_set": test_counts,
        "in_sample_set": in_sample_counts,
        "pricing": {"engine": "lsmc", "n_paths": args.n_paths, "seeds": args.seeds, "n_substeps": 1,
                    "exercise": "ceil(252 T) dates", "basis_degree": BASIS_DEGREE,
                    "paths_shared_within_expiration": True, "bootstrap_resamples": args.bootstrap,
                    "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_unit": "expiration"},
        "parameter_sets": {
            "calibrated_2026_10_13": {**label, **({"label": "REJECTED"} if fitted["status"] == "rejected" else {})},
            "published_2015": {"status": "published", "source": sets["published_2015"]["source"]},
        },
        "results": results,
        "in_sample_results": in_sample_results,
        "tree_shortcut": {"parameter_set": "calibrated_2026_10_13", "overall": shortcut_summary["overall"],
                          "by_maturity": shortcut_summary["by_maturity"]},
        "precision": precision,
        "runtime": {"wall_seconds": total, "pricing_seconds": timings, "budget_b4_seconds": BUDGET_B4_SECONDS,
                    "b4_met": bool(sum(timings.values()) <= BUDGET_B4_SECONDS), "workers": args.workers},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2))

    for name in sets:
        tag = " (REJECTED fit)" if name == "calibrated_2026_10_13" and fitted["status"] == "rejected" else ""
        for population, table in (("out of sample", results), ("in sample", in_sample_results)):
            o = table[name]["overall"]
            iv = "n/a" if o["median_abs_iv_error"] is None else f"{o['median_abs_iv_error']:.4f}"
            print(f"{name}{tag}, {population}: n {o['n']}, inside spread {o['share_inside_spread']:.3f}, "
                  f"median rel error {o['median_rel_price_error']:.4f}, median |IV error| {iv}")
    print(f"T3 share {precision['T3_share']:.3f} (met {precision['T3_met']}); total {total:.0f} s")
    if not precision["T3_met"] and args.n_paths < 200_000:
        print("T3 missed: rerun with --n-paths 200000 (SPEC.md 9.2)")
    print(f"written: {output.name}")


def main():
    parser = argparse.ArgumentParser(description="GOOG accuracy within and outside the calibrated sample")
    parser.add_argument("--raw-file", default=str(DEFAULTS["raw_file"]))
    parser.add_argument("--calibration", default=str(DEFAULTS["calibration"]))
    parser.add_argument("--market-inputs", default=str(DEFAULTS["market_inputs"]))
    parser.add_argument("--symbol", default="GOOG")
    parser.add_argument("--date", default="2026-10-13")
    parser.add_argument("--n-paths", type=int, default=100_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=PRODUCTION_SEEDS)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", default=str(DEFAULTS["output"]))
    parser.add_argument("--prices-out", default=str(DEFAULTS["prices_out"]))
    args = parser.parse_args()
    if args.n_paths < 20_000 and Path(args.output).resolve() == DEFAULTS["output"].resolve():
        print("E_CONFIG: n_paths below 20,000 is not allowed for the reported output (SPEC.md 9.2)", file=sys.stderr)
        return 2
    try:
        run(args)
    except HestonError as error:
        print(str(error), file=sys.stderr)
        return EXIT_CODES.get(error.code, 1)
    except Exception as error:
        # Sanitized: pandas messages can echo row contents (SPEC.md 14.2)
        print(f"internal error: {type(error).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
