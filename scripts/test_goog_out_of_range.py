"""
Price GOOG puts outside the strike and maturity range of the source paper (SPEC.md 8.6).

    python scripts/test_goog_out_of_range.py [--output <file>]

The paper's data covers strikes of about 0.48 to 1.68 times spot and maturities of
0.20 to 1.97 years. No market prices exist here, so each contract is checked only for
correct behaviour: the Monte Carlo European price agrees with the semi-analytic price,
the Bermudan price is not below the European price, and it is not below intrinsic value.
"""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import numpy as np
import yaml

from hestonpricer.analytic import heston_european
from hestonpricer.iv import bs_implied_vol
from hestonpricer.lsmc import evaluate_policy, fit_policy
from hestonpricer.simulate import simulate

SEEDS = [620001, 620002, 620003, 620004, 620005]
N_PATHS = 100_000
N_EXERCISE = 50
# Maturity -> strikes as fractions of spot. 0.9507 is inside the paper's range; its strikes are not.
CASES = {0.08: [0.4, 0.7, 1.0, 1.3, 1.8], 0.9507: [0.4, 1.8], 3.0: [0.4, 0.7, 1.0, 1.3, 1.8], 5.0: [0.4, 0.7, 1.0, 1.3, 1.8]}


def clean(value):
    """JSON has no NaN, so a missing number is written as null."""
    return None if isinstance(value, float) and not math.isfinite(value) else value


def main():
    parser = argparse.ArgumentParser(description="GOOG puts outside the paper's range")
    parser.add_argument("--output", default=str(ROOT / "results" / "goog" / "out_of_range.json"))
    args = parser.parse_args()

    published = yaml.safe_load((ROOT / "configs" / "inputs" / "published_params.yaml").read_text())["sets"]["GOOG"]
    params = {name: float(published[name]) for name in ("v0", "kappa", "theta", "xi", "rho")}
    y = yaml.safe_load((ROOT / "configs" / "inputs" / "market_inputs_goog_2015.yaml").read_text())["treasury_1y_cmt_yield"]
    r = 2.0 * math.log(1.0 + float(next(iter(y.values()))) / 2.0)
    q = 0.0
    market = {"r": r, "q": [q]}

    rows = []
    for maturity, strikes in CASES.items():
        base = {"strike": 1.0, "maturity": maturity, "n_exercise": N_EXERCISE}
        sim = {"n_paths": N_PATHS, "n_substeps": max(1, round(252 * maturity / N_EXERCISE))}
        per_strike = {k: [] for k in strikes}
        # One simulation per seed and maturity, reused for every strike
        for seed in SEEDS:
            child_a, child_b = np.random.SeedSequence(seed).spawn(2)
            paths_a = simulate([params], [[1.0]], market, base, sim, np.random.Generator(np.random.PCG64(child_a)))
            paths_b = simulate([params], [[1.0]], market, base, sim, np.random.Generator(np.random.PCG64(child_b)))
            for strike in strikes:
                contract = {**base, "strike": strike}
                per_strike[strike].append(evaluate_policy(fit_policy(paths_a, contract, market, 3), paths_b, contract, market))

        for strike in strikes:
            outcomes = per_strike[strike]

            def pooled(name):
                return float(np.mean([o[name] for o in outcomes]))

            def pooled_se(name):
                return float(np.sqrt(np.sum([o[name] ** 2 for o in outcomes])) / len(outcomes))

            bermudan, se_b = pooled("bermudan"), pooled_se("se_bermudan")
            european, se_e = pooled("european"), pooled_se("se_european")
            premium, se_p = pooled("premium"), pooled_se("se_premium")
            analytic = heston_european("put", 1.0, strike, maturity, params, r, q)
            intrinsic = max(strike - 1.0, 0.0)
            rows.append({
                "maturity": maturity, "strike_over_spot": strike,
                "bermudan": bermudan, "se_bermudan": se_b, "european_mc": european, "se_european": se_e,
                "european_analytic": analytic, "premium": premium, "se_premium": se_p,
                "model_implied_vol": clean(bs_implied_vol("put", analytic, 1.0, strike, maturity, r, q)),
                "checks": {
                    "european_within_3_se": abs(european - analytic) <= 3 * max(se_e, 1e-12),
                    "bermudan_not_below_european": premium >= -3 * max(se_p, 1e-12),
                    "bermudan_not_below_intrinsic": bermudan >= intrinsic - 3 * max(se_b, 1e-12),
                },
            })

    document = {"schema_version": 1, "data_mode": "published", "asset": "GOOG",
                "note": "Model extrapolation with the published 2015 parameters; no market prices exist for these contracts.",
                "r": r, "q": q, "n_paths": N_PATHS, "seeds": SEEDS, "n_exercise": N_EXERCISE,
                "units": "fraction_of_initial_spot", "rows": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(document, indent=2))
    passed = sum(all(row["checks"].values()) for row in rows)
    print(f"{passed} of {len(rows)} contracts pass all behaviour checks; written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
