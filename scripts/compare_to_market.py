"""
Compare the simulated Google put with the published market price of 2015-02-02 (SPEC.md 8.4),
and price all 401 published puts.

    python scripts/compare_to_market.py [--result <result.json>] [--output-dir <dir>] [--n-paths N]

The parameters were fitted to these same prices, so the comparison is in-sample.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import numpy as np
import yaml

from hestonpricer.iv import american_implied_vol
from hestonpricer.lsmc import evaluate_policy, fit_policy
from hestonpricer.simulate import simulate

TABLE = ROOT / "configs" / "google_2015_american_puts.csv"
INPUTS = ROOT / "configs" / "market_inputs_goog_2015.yaml"
PARAMS = ROOT / "configs" / "published_params.yaml"
SOURCE = "Burkovska, Glau, Mahlstedt and Wohlmuth, arXiv:1611.06452, Appendix A, Table 10"
SPOT = 523.755
OFFICIAL_MATURITY = 0.9507
OFFICIAL_STRIKE = 525.0
SEED = 620001
N_EXERCISE = 50


def price_all(rows, params, market, n_paths):
    """Price each published put. One simulation per maturity is reused across its strikes."""
    priced = []
    for maturity in sorted({row["maturity"] for row in rows}):
        n_substeps = max(1, round(252 * maturity / N_EXERCISE))
        base = {"strike": 1.0, "maturity": maturity, "n_exercise": N_EXERCISE}
        sim = {"n_paths": n_paths, "n_substeps": n_substeps}
        child_a, child_b = np.random.SeedSequence(SEED).spawn(2)
        paths_a = simulate([params], [[1.0]], market, base, sim, np.random.Generator(np.random.PCG64(child_a)))
        paths_b = simulate([params], [[1.0]], market, base, sim, np.random.Generator(np.random.PCG64(child_b)))
        for row in rows:
            if row["maturity"] != maturity:
                continue
            contract = {**base, "strike": row["strike"] / SPOT}
            outcome = evaluate_policy(fit_policy(paths_a, contract, market, 3), paths_b, contract, market)
            priced.append({**row, "model_price": outcome["bermudan"] * SPOT, "se": outcome["se_bermudan"] * SPOT})
    return priced


def newest_result():
    results = sorted(ROOT.glob("results/runs/goog_2015_contract-*/result.json"), key=lambda p: p.stat().st_mtime)
    if not results:
        raise SystemExit("no goog_2015_contract run found: run scripts/run.py price first")
    return results[-1]


def main():
    parser = argparse.ArgumentParser(description="Compare with the published Feb 2015 Google puts")
    parser.add_argument("--result", default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "results"))
    parser.add_argument("--table", default=str(TABLE))
    parser.add_argument("--n-paths", type=int, default=100_000)
    args = parser.parse_args()

    result = json.loads(Path(args.result).read_text() if args.result else newest_result().read_text())
    params = {k: float(v) for k, v in yaml.safe_load(PARAMS.read_text())["sets"]["GOOG"].items()
              if k in ("v0", "kappa", "theta", "xi", "rho")}
    y = yaml.safe_load(INPUTS.read_text())["treasury_1y_cmt_yield"]
    r = 2.0 * math.log(1.0 + float(next(iter(y.values()))) / 2.0)
    q = 0.0

    with open(args.table, newline="") as file:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(file)]
    official_rows = [row for row in rows if row["maturity"] == OFFICIAL_MATURITY and row["strike"] == OFFICIAL_STRIKE]
    if len(official_rows) != 1:
        raise SystemExit("the official contract is not in the table")
    market_price = official_rows[0]["market_price"]

    def implied(price):
        return american_implied_vol("put", price / SPOT, 1.0, OFFICIAL_STRIKE / SPOT, OFFICIAL_MATURITY, r, q)

    model = result["pooled"]["bermudan"] * SPOT
    priced = price_all(rows, params, {"r": r, "q": [q]}, args.n_paths)
    gaps = np.array([row["model_price"] - row["market_price"] for row in priced])
    relative = np.array([abs(row["model_price"] / row["market_price"] - 1) for row in priced])

    by_maturity = []
    for maturity in sorted({row["maturity"] for row in priced}):
        part = [row["model_price"] - row["market_price"] for row in priced if row["maturity"] == maturity]
        by_maturity.append({"maturity": maturity, "n": len(part), "rms_usd": float(np.sqrt(np.mean(np.square(part))))})

    document = {
        "schema_version": 1,
        "source": SOURCE,
        "note": "In-sample: the published parameters were fitted to these same prices.",
        "run_id": result["run_id"],
        "data_mode": result["data_mode"],
        "official_contract": {
            "valuation_date": "2015-02-02", "style": "American put", "underlying_usd": SPOT, "strike_usd": OFFICIAL_STRIKE,
            "maturity_years": OFFICIAL_MATURITY, "r": r, "q": q,
            "market_usd": market_price, "model_usd": model, "se_usd": result["pooled"]["se_bermudan"] * SPOT,
            "european_analytic_usd": result["pooled"]["european_analytic"] * SPOT,
            "model_minus_market_usd": model - market_price,
            "model_minus_market_percent": 100.0 * (model - market_price) / market_price,
            "market_implied_vol": implied(market_price), "model_implied_vol": implied(model),
        },
        "all_puts": {
            "n": len(priced), "n_paths": args.n_paths, "n_exercise": N_EXERCISE, "seed": SEED,
            "rms_usd": float(np.sqrt(np.mean(gaps**2))), "median_abs_usd": float(np.median(np.abs(gaps))),
            "max_abs_usd": float(np.max(np.abs(gaps))), "median_abs_percent": float(100.0 * np.median(relative)),
            "within_2_percent": int((relative <= 0.02).sum()), "within_5_percent": int((relative <= 0.05).sum()),
            "by_maturity": by_maturity,
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "market_comparison.json").write_text(json.dumps(document, indent=2))
    with open(output_dir / "market_comparison_2015.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["maturity", "strike", "market_price", "model_price", "se"])
        writer.writeheader()
        writer.writerows(priced)
    print(f"official put: model {model:.2f} USD against market {market_price:.2f} USD; {len(priced)} puts priced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
