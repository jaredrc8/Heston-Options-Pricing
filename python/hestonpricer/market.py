"""
Pricing many contracts at once and scoring them against market quotes (SPEC.md 8.4 to 8.6).

Used by scripts/market_test.py (model against market data) and by `run.py grid` (behaviour
checks on a strike x maturity grid). Plain functions on pandas DataFrames.

A contract table has one row per contract with at least: expiration (the grouping key; contracts
with the same key share simulated paths), tenor (years), kind ("put" or "call") and strike
(fraction of spot). Market columns (bid, ask, mid as fractions of spot; market_iv) are needed
only for scoring.
"""

import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from hestonpricer import surface
from hestonpricer.calibrate import model_implied_vols
from hestonpricer.iv import american_implied_vol
from hestonpricer.lsmc import evaluate_policy, fit_policy
from hestonpricer.simulate import simulate
from hestonpricer.tree import crr_price

BASIS_DEGREE = 3
BOOTSTRAP_SEED = 620200
MATURITY_BUCKETS = [("under 3 months", 0.0, 0.25), ("3-6 months", 0.25, 0.5),
                    ("6-12 months", 0.5, 1.0), ("over 12 months", 1.0, 99.0)]
MONEYNESS_BUCKETS = [("0.70-0.90", 0.70, 0.90), ("0.90-1.10", 0.90, 1.10), ("1.10-1.30", 1.10, 1.30 + 1e-12)]
METRICS = ("median_rel_price_error", "share_inside_spread", "share_mid_inside_mc_ci95",
           "rms_iv_error", "median_abs_iv_error")


# ----------------------------------------------------------------------
# Contract tables
# ----------------------------------------------------------------------

def quote_table(rows, population, r, q):
    """Contracts from filtered iVolatility quotes. Market quantities are divided by the stock price (SPEC.md 3)."""
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


def ivolatility_populations(raw, symbol, date, r, q):
    """
    The two populations of SPEC.md 2.2: the calibration grid (in sample) and the contracts after
    filters F1 to F8 that traded that day and are not in the grid (out of sample).
    """
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
    contracts = pd.concat([quote_table(test, "out_of_sample", r, q),
                           quote_table(grid, "in_sample", r, q)], ignore_index=True)
    return contracts, counts, in_sample_counts


def price_table_contracts(rows, spot, kind):
    """
    Contracts from a table of published prices (columns maturity, strike, market_price, in currency).
    The maturity is the grouping key. Only the mid is known, so bid, ask and market IV are missing.
    """
    maturity = rows["maturity"].to_numpy(float)
    return pd.DataFrame({
        "option_symbol": [f"{m:g}-{k:g}" for m, k in zip(maturity, rows["strike"])],
        "population": "table",
        "expiration": maturity,
        "tenor": maturity,
        "kind": kind,
        "style": "A",
        "strike": rows["strike"].to_numpy(float) / spot,
        "k_over_f": np.nan,
        "bid": np.nan, "ask": np.nan,
        "mid": rows["market_price"].to_numpy(float) / spot,
        "market_iv": np.nan,
    })


def grid_contracts(cases, kind):
    """Contracts for a strike x maturity grid: cases is a list of {maturity, strikes}."""
    rows = [{"expiration": float(case["maturity"]), "tenor": float(case["maturity"]), "kind": kind,
             "strike": float(strike)} for case in cases for strike in case["strikes"]]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Pricing
# ----------------------------------------------------------------------

def exercise_terms(tenor, exercise):
    """
    (n_exercise, n_substeps) for one maturity. "daily": M = ceil(252 T) dates and one sub-step
    (SPEC.md 4.1 test contracts). An integer n: n dates, with sub-steps of about one trading day.
    """
    if exercise == "daily":
        return math.ceil(252 * tenor), 1
    n_exercise = int(exercise)
    return n_exercise, max(1, round(252 * tenor / n_exercise))


def price_expiration(job):
    """
    LSMC prices for every contract of one expiration under one parameter set, pooled over seeds.
    Each seed simulates regression set A and evaluation set B once, from independent child
    streams as in price_bermudan, and every contract of this expiration uses those paths.
    """
    params, r, q, tenor, contracts, n_paths, seeds, exercise = job
    market = {"r": r, "q": [q]}
    n_exercise, n_substeps = exercise_terms(tenor, exercise)
    base = {"strike": 1.0, "maturity": tenor, "n_exercise": n_exercise}
    sim = {"n_paths": n_paths, "n_substeps": n_substeps, "basis_degree": BASIS_DEGREE}
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


def price_contracts(params, contracts, r, q, n_paths, seeds, workers=1, exercise="daily"):
    """
    Pooled LSMC price for every contract, in the order of `contracts`. With workers > 1, the
    expirations are priced in separate processes; the numbers do not depend on workers.
    """
    jobs = []
    for _, group in contracts.groupby("expiration", sort=True):
        tenor = float(group["tenor"].iloc[0])
        members = [{"index": int(i), "strike": float(row.strike), "kind": row.kind} for i, row in group.iterrows()]
        jobs.append((params, r, q, tenor, members, n_paths, seeds, exercise))
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
# Scoring against bid, ask and mid
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
