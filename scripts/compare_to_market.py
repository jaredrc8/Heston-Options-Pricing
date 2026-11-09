"""
Compare the priced GOOG put with its public market quote (SPEC.md 8.4).

    python scripts/compare_to_market.py [--result <result.json>] [--output <file>]

There is no pass rule. The comparison is written to results/market_comparison.json.
"""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import yaml

from hestonpricer.iv import american_implied_vol

QUOTE_FILE = ROOT / "configs" / "market_quote_goog.yaml"
MARKET_INPUTS = ROOT / "configs" / "market_inputs_goog_contract.yaml"


def newest_result():
    """The newest completed run of the market contract."""
    results = sorted(ROOT.glob("results/runs/goog_market_contract-*/result.json"), key=lambda p: p.stat().st_mtime)
    if not results:
        raise SystemExit("no goog_market_contract run found: run scripts/run.py price first")
    return results[-1]


def main():
    parser = argparse.ArgumentParser(description="Compare the GOOG put with its market quote")
    parser.add_argument("--result", default=None)
    parser.add_argument("--output", default=str(ROOT / "results" / "market_comparison.json"))
    args = parser.parse_args()

    result = json.loads(Path(args.result).read_text() if args.result else newest_result().read_text())
    quote = yaml.safe_load(QUOTE_FILE.read_text())
    inputs = yaml.safe_load(MARKET_INPUTS.read_text())
    yields = {str(date): value for date, value in inputs["treasury_1y_cmt_yield"].items()}

    spot = float(quote["spot"])
    strike_fraction = float(quote["strike"]) / spot
    maturity = float(result["contract"]["maturity"])
    r = 2.0 * math.log(1.0 + float(yields[str(quote["valuation_date"])]) / 2.0)
    q = float(inputs["dividend_yield"]["GOOG"])

    if abs(float(result["contract"]["strike"]) - strike_fraction) > 1e-6:
        raise SystemExit("the run's strike does not match the quote")

    def implied(price_usd):
        return american_implied_vol("put", price_usd / spot, 1.0, strike_fraction, maturity, r, q)

    pooled = result["pooled"]
    bid, ask = float(quote["bid"]), float(quote["ask"])
    mid = 0.5 * (bid + ask)
    model = pooled["bermudan"] * spot

    document = {
        "schema_version": 1,
        "run_id": result["run_id"],
        "data_mode": result["data_mode"],
        "note": "Published Google parameters from 2015-02-02; not a calibration to this market.",
        "quote_file": QUOTE_FILE.name,
        "contract": {key: str(quote[key]) if key in ("valuation_date", "expiry") else quote[key]
                     for key in ("option", "style", "valuation_date", "expiry", "strike")},
        "r": r,
        "q": q,
        "market": {"bid_usd": bid, "ask_usd": ask, "mid_usd": mid, "mid_fraction_of_spot": mid / spot,
                   "implied_vol_bid": implied(bid), "implied_vol_mid": implied(mid), "implied_vol_ask": implied(ask),
                   "vendor_implied_vol": quote["vendor_iv"], "source": quote["source"]},
        "model": {"bermudan_usd": model, "se_usd": pooled["se_bermudan"] * spot,
                  "bermudan_fraction_of_spot": pooled["bermudan"],
                  "european_analytic_usd": pooled["european_analytic"] * spot, "implied_vol": implied(model)},
        "difference": {"model_minus_mid_usd": model - mid, "model_minus_mid_percent_of_mid": 100.0 * (model - mid) / mid,
                       "model_inside_bid_ask": bid <= model <= ask},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2))
    print(f"model {model:.2f} USD against market bid {bid:.2f} / ask {ask:.2f}; written to {output.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
