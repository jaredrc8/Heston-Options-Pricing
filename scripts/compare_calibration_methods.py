"""
Diagnostic: calibrate with the method of Burkovska et al. (arXiv:1611.06452, objective J_CF, eq. 5.2).

    python scripts/compare_calibration_methods.py --source google
    python scripts/compare_calibration_methods.py --source market

The method de-Americanizes each put with a binomial tree, then fits Heston closed-form European
prices in price space, with uniform weights, the paper's bounds and starting point, and a
constraint 2 kappa theta >= xi^2. It is a diagnostic only: it changes no spec rule and gives no
reported parameters. Output holds aggregates only, never a quote or per-contract residual.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize

from hestonpricer import surface
from hestonpricer.analytic import heston_european_many
from hestonpricer.calibrate import model_implied_vols
from hestonpricer.errors import HestonError
from hestonpricer.iv import american_implied_vol
from hestonpricer.tree import crr_price

NAMES = ["xi", "rho", "theta", "kappa", "v0"]                 # the paper's order, with gamma = theta
START = [0.6005, -0.6815, 0.4867, 2.02, 0.4961]               # paper, Table 8, initial guess
BOUNDS = [(0.1, 0.9), (-0.95, 0.3), (0.01, 0.5), (0.1, 5.0), (1e-5, 1.0)]   # paper, Section 6
TREE_STEPS = 500
SNAPSHOT = "2026-10-14"
TICKERS = ["AMD", "NVDA", "INTC"]
RAW_FILE = ROOT / "data" / "nvda_amd_intc_IV_Oct13-15.csv"
GOOGLE_TABLE = ROOT / "configs" / "google_2015_american_puts.csv"
GOOGLE_SPOT = 523.755


def de_americanize(k_over_s, tenors, prices, r, q):
    """European-equivalent put prices from a binomial tree matched to each American price."""
    keep, european = [], []
    for k, tenor, price in zip(k_over_s, tenors, prices):
        sigma = american_implied_vol("put", price, 1.0, k, tenor, r, q, n_steps=TREE_STEPS)
        if np.isfinite(sigma):
            keep.append(True)
            european.append(crr_price("put", 1.0, k, tenor, sigma, r, q, TREE_STEPS, exercise_steps=[TREE_STEPS]))
        else:
            keep.append(False)
    return np.array(keep, dtype=bool), np.array(european)


def fit_prices(k_over_s, tenors, targets, r, q):
    """Minimize the mean squared price error from the paper's starting point."""
    def to_params(x):
        return dict(zip(NAMES, (float(v) for v in x)))

    def objective(x):
        params = to_params(x)
        model = np.zeros(len(targets))
        try:
            for tenor in np.unique(tenors):
                same = tenors == tenor
                model[same] = heston_european_many(np.zeros(same.sum(), dtype=bool), 1.0, k_over_s[same], tenor, params, r, q)
        except HestonError:
            return 1e3
        return 1e4 * float(np.mean((model - targets) ** 2))

    constraint = {"type": "ineq", "fun": lambda x: 2.0 * x[3] * x[2] - x[0] ** 2}
    fit = minimize(objective, START, method="SLSQP", bounds=BOUNDS, constraints=[constraint],
                   options={"maxiter": 500, "ftol": 1e-12})
    params = to_params(fit.x)
    return params, {
        "converged": bool(fit.success),
        "price_rms_fraction_of_spot": float(np.sqrt(objective(fit.x) / 1e4)),
        "at_bound": [name for name, value, (low, high) in zip(NAMES, fit.x, BOUNDS)
                     if value <= low + 1e-6 or value >= high - 1e-6],
    }


def run_google():
    """Check the implementation against the paper's own result (Table 8, row J_CF)."""
    with open(GOOGLE_TABLE, newline="") as file:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(file)]
    k_over_s = np.array([row["strike"] / GOOGLE_SPOT for row in rows])
    tenors = np.array([row["maturity"] for row in rows])
    prices = np.array([row["market_price"] / GOOGLE_SPOT for row in rows])
    above = prices > np.maximum(k_over_s - 1.0, 0.0) * 1.01

    keep, european = de_americanize(k_over_s[above], tenors[above], prices[above], 0.0015, 0.0)
    params, info = fit_prices(k_over_s[above][keep], tenors[above][keep], european, 0.0015, 0.0)
    return {"data": "Google puts of 2015-02-02", "puts_used": int(keep.sum()), "params": params, **info,
            "paper_table_8": {"xi": 0.3927, "rho": -0.6518, "theta": 0.0580, "kappa": 1.4554, "v0": 0.0546}}


def paper_quotes(raw, symbol, r, q):
    """Puts selected as the paper does: positive bids, mid above intrinsic by 1%, maturities within two years."""
    keep = (raw["symbol"] == symbol) & (raw["date"] == pd.Timestamp(SNAPSHOT)) & (raw["call_put"] == "P")
    quotes = surface.add_quote_columns(raw[keep], r, q)
    quotes = quotes[quotes["bid"] > 0]
    quotes = quotes[quotes["days"].between(21, 730)]
    intrinsic = (quotes["strike"] - quotes["spot"]).clip(lower=0.0)
    return quotes[quotes["mid"] > intrinsic * 1.01]


def run_market():
    path = ROOT / "configs" / "governance.yaml"
    if not (path.exists() and yaml.safe_load(path.read_text()).get("market_use_authorized") is True):
        raise HestonError("E_GOV_UNAUTHORIZED", "market_use_authorized is not true")
    inputs = yaml.safe_load((ROOT / "configs" / "market_inputs.yaml").read_text())
    y = {str(k): v for k, v in inputs["treasury_1y_cmt_yield"].items()}[SNAPSHOT]
    r = 2.0 * np.log(1.0 + float(y) / 2.0)
    raw = surface.load_raw_iv(RAW_FILE)

    output = []
    for symbol in TICKERS:
        q = float(inputs["dividend_yield"][symbol])
        cfg = {"symbol": symbol, "snapshot_date": SNAPSHOT, "r": r, "q": q}
        quotes = paper_quotes(raw, symbol, r, q)
        k_over_s = (quotes["strike"] / quotes["spot"]).to_numpy()
        tenors = quotes["tenor"].to_numpy()
        keep, european = de_americanize(k_over_s, tenors, (quotes["mid"] / quotes["spot"]).to_numpy(), r, q)
        params, info = fit_prices(k_over_s[keep], tenors[keep], european, r, q)

        # The same accuracy measure as the spec, on the same 48-point grid
        filtered, _ = surface.filter_quotes(raw, cfg)
        grid = surface.select_grid(filtered, {"contract_strike": 1.0, "contract_maturity": 1.0})
        model_iv = model_implied_vols(params, grid["k_over_s"].to_numpy(float), grid["tenor"].to_numpy(float), r, q)
        errors = grid["iv"].to_numpy(float) - model_iv
        output.append({"asset": symbol, "snapshot_date": SNAPSHOT, "puts_used": int(keep.sum()), "params": params, **info,
                       "feller_ratio": 2 * params["kappa"] * params["theta"] / params["xi"] ** 2,
                       "iv_rms_on_spec_grid": float(np.sqrt(np.nanmean(errors**2)))})
    return output


def main():
    parser = argparse.ArgumentParser(description="Diagnostic calibration with the paper's method")
    parser.add_argument("--source", choices=["google", "market"], required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    try:
        document = run_google() if args.source == "google" else {"results": run_market()}
    except HestonError as error:
        print(str(error), file=sys.stderr)
        return 4 if error.code.startswith("E_GOV") else 2
    except Exception as error:
        # Sanitized: pandas messages can echo row contents (SPEC.md 14.2)
        print(f"internal error: {type(error).__name__}", file=sys.stderr)
        return 1
    path = Path(args.output) if args.output else ROOT / "results" / f"method_comparison_{args.source}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2))
    print(f"written: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
