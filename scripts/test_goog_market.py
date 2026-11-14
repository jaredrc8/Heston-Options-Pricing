"""
GOOG against the market on 2026-10-13 (SPEC.md 8.7).

    python scripts/test_goog_market.py

Prices every GOOG option that passes filters F1 to F8, is not in the calibration grid, and traded
that day, and compares it with its quote. Output holds aggregates only: no quote, price, or
per-contract residual leaves this script (DATA_CONTRACT.md section 4.3).
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import numpy as np
import yaml

from hestonpricer import surface
from hestonpricer.calibrate import model_implied_vols
from hestonpricer.errors import HestonError
from hestonpricer.tree import crr_price

RAW_FILE = ROOT / "data" / "goog_IV_Oct13.csv"
CALIBRATION = ROOT / "results" / "calibration_goog" / "calibration.json"
INPUTS = ROOT / "configs" / "market_inputs_goog_oct13.yaml"
OUTPUT = ROOT / "results" / "goog_market_test.json"
DATE = "2026-10-13"
NAMES = ("v0", "kappa", "theta", "xi", "rho")


def check_governance():
    path = ROOT / "configs" / "governance.yaml"
    if not (path.exists() and yaml.safe_load(path.read_text()).get("market_use_authorized") is True):
        raise HestonError("E_GOV_UNAUTHORIZED", "market_use_authorized is not true")


def score(params, test, r, q):
    """Aggregate errors of one parameter set on the test contracts."""
    model_iv = model_implied_vols(params, test["k_over_s"].to_numpy(float), test["tenor"].to_numpy(float), r, q)
    iv_error = model_iv - test["own_iv"].to_numpy(float)

    inside, relative = [], []
    for (_, row), sigma in zip(test.iterrows(), model_iv):
        if not np.isfinite(sigma):
            continue
        kind = "call" if row["call_put"] == "C" else "put"
        # American contracts go through the same tree as the conversion in SPEC.md 6.3
        steps = None if row["style"] == "A" else [500]
        price = crr_price(kind, row["spot"], row["strike"], row["tenor"], sigma, r, q, 500, exercise_steps=steps)
        inside.append(row["bid"] <= price <= row["ask"])
        relative.append(abs(price / row["mid"] - 1.0))

    finite = iv_error[np.isfinite(iv_error)]
    tenors = test["tenor"].to_numpy(float)
    buckets = []
    for name, low, high in (("T<0.25", 0.0, 0.25), ("0.25<=T<0.5", 0.25, 0.5), ("0.5<=T<1", 0.5, 1.0), ("T>=1", 1.0, 99.0)):
        mask = (tenors >= low) & (tenors < high) & np.isfinite(iv_error)
        if mask.sum() >= 5:
            rms = float(np.sqrt(np.mean(iv_error[mask] ** 2)))
            buckets.append({"tenor_bucket": name, "n": int(mask.sum()), "rms_iv_error": rms})
    return {
        "n_priced": len(inside),
        "rms_iv_error": float(np.sqrt(np.mean(finite**2))),
        "median_abs_iv_error": float(np.median(np.abs(finite))),
        "mean_iv_error": float(np.mean(finite)),
        "share_inside_bid_ask": float(np.mean(inside)),
        "median_abs_price_error_percent": float(100.0 * np.median(relative)),
        "share_within_5_percent_of_mid": float(np.mean(np.array(relative) <= 0.05)),
        "by_tenor": buckets,
    }


def main():
    try:
        check_governance()
        inputs = yaml.safe_load(INPUTS.read_text())
        y = {str(k): v for k, v in inputs["treasury_1y_cmt_yield"].items()}[DATE]
        r, q = 2.0 * math.log(1.0 + float(y) / 2.0), float(inputs["dividend_yield"]["GOOG"])

        entry = json.loads(CALIBRATION.read_text())["assets"]["GOOG"]
        if entry.get("params"):
            fitted, status = entry["params"], f"accepted, ladder step {entry['ladder_step']}"
        else:
            fitted, status = entry["rejected_full_fit"]["params"], "REJECTED full fit (ladder step 3)"
        published = yaml.safe_load((ROOT / "configs" / "published_params.yaml").read_text())["sets"]["GOOG"]
        published = {name: float(published[name]) for name in NAMES}

        raw = surface.load_raw_iv(RAW_FILE)
        quotes, _ = surface.filter_quotes(raw, {"symbol": "GOOG", "snapshot_date": DATE, "r": r, "q": q})
        grid = surface.select_grid(quotes, {"contract_strike": 1.0, "contract_maturity": 1.0})
        used = set(zip(grid["expiration"], grid["strike"]))
        quotes = quotes.assign(k_over_s=quotes["strike"] / quotes["spot"])
        held_out = quotes[[(e, k) not in used for e, k in zip(quotes["expiration"], quotes["strike"])]]
        traded = held_out[held_out["volume"] > 0]

        document = {
            "schema_version": 1, "asset": "GOOG", "date": DATE, "r": r, "q": q,
            "note": "Same-day, out-of-sample across contracts. Aggregates only.",
            "fitted_parameter_status": status,
            "contracts": {"after_filters": int(len(quotes)), "in_calibration_grid": int(len(grid)),
                          "held_out": int(len(held_out)), "held_out_and_traded": int(len(traded))},
            "calibrated_on_2026_10_13": score(fitted, traded, r, q),
            "published_2015_parameters": score(published, traded, r, q),
        }
    except HestonError as error:
        print(str(error), file=sys.stderr)
        return 4 if error.code.startswith("E_GOV") else 2
    except Exception as error:
        # Sanitized: pandas messages can echo row contents (SPEC.md 14.2)
        print(f"internal error: {type(error).__name__}", file=sys.stderr)
        return 1
    OUTPUT.write_text(json.dumps(document, indent=2))
    print(f"written: {OUTPUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
