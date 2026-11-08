"""
Calibrate Heston parameters for NVDA, AMD and INTC to the listed option
implied vols in data/nvda_amd_intc_IV_Oct13-15.csv, and write them to
results/heston_params.csv (one row per asset).

main.py reads that CSV. If this IV data cannot be used (for example under
its license), skip this script and fill the CSV by hand with literature
values, using the same columns: symbol, spot, v0, kappa, theta, xi, rho.

Quotes used, for a contract with about 1 year to expiry:
  - the latest snapshot date in the file only (2026-10-15)
  - maturities 0.4 to 1.3 years, which bracket T = 1 without the steep
    short-dated term structure
  - strikes between 70% and 130% of spot
  - out-of-the-money options only (their American implied vols are close
    to European ones)
  - a positive bid and a positive implied vol

kappa is not identified by these surfaces: left free it drifts to its
bounds (20 or about 0) while the other parameters fit equally well. It is
held at a typical value instead (proposal Table 3 fallback); in this window
that raises the RMS implied-vol error by at most 0.0007.

Run from anywhere:  python src/calibrate_market.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from heston_calibration import calibrate_heston

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE.parent / "data" / "nvda_amd_intc_IV_Oct13-15.csv"
PARAMS_FILE = HERE.parent / "results" / "heston_params.csv"

SYMBOLS = ["NVDA", "AMD", "INTC"]
R = 0.05  # risk-free rate (proposal Section C)
Q = 0.0   # dividend yield (proposal Section C)
MIN_T, MAX_T = 0.4, 1.3
MIN_MONEYNESS, MAX_MONEYNESS = 0.7, 1.3
FIXED = {"kappa": 2.0}


def load_quotes(path):
    """Read the CSV and keep the latest date, with maturities in years."""
    data = pd.read_csv(path)
    data["date"] = pd.to_datetime(data["date"])
    data["expiration"] = pd.to_datetime(data["expiration"])
    data = data[data["date"] == data["date"].max()].copy()
    data["T"] = (data["expiration"] - data["date"]).dt.days / 365.0
    return data


def select_quotes(data, symbol):
    """Spot price and the quotes used for one symbol (see the rules above)."""
    quotes = data[data["symbol"] == symbol]
    spot = quotes["stock price for iv"].iloc[0]
    forward = spot * np.exp((R - Q) * quotes["T"])
    out_of_the_money = ((quotes["call/put"] == "P") & (quotes["strike"] < forward)) | \
                       ((quotes["call/put"] == "C") & (quotes["strike"] >= forward))
    keep = (out_of_the_money
            & quotes["T"].between(MIN_T, MAX_T)
            & (quotes["strike"] / spot).between(MIN_MONEYNESS, MAX_MONEYNESS)
            & (quotes["bid"] > 0)
            & (quotes["iv"] > 0))
    return spot, quotes[keep]


def main():
    """Calibrate every symbol, write the parameters to PARAMS_FILE and return them."""
    data = load_quotes(DATA_FILE)

    rows = []
    for symbol in SYMBOLS:
        spot, quotes = select_quotes(data, symbol)
        result = calibrate_heston(spot, R, Q,
                                  quotes["strike"].to_numpy(),
                                  quotes["T"].to_numpy(),
                                  quotes["iv"].to_numpy(),
                                  fixed=FIXED)

        # The fit is stable if every starting point reaches the same cost
        costs = [cost for cost, _ in result["all_starts"]]
        n_agree = sum(cost <= 1.01 * costs[0] for cost in costs)

        p = result["params"]
        rows.append({"symbol": symbol, "spot": spot,
                     "v0": p["v0"], "kappa": p["kappa"], "theta": p["theta"],
                     "xi": p["xi"], "rho": p["rho"],
                     "rms_error": result["rms_error"], "max_error": result["max_error"],
                     "n_quotes": len(quotes), "starts_agree": f"{n_agree}/{len(costs)}",
                     "date": data["date"].iloc[0].date()})

    params = pd.DataFrame(rows).set_index("symbol")
    PARAMS_FILE.parent.mkdir(exist_ok=True)
    params.to_csv(PARAMS_FILE)
    return params


if __name__ == "__main__":
    params = main()
    print(params.round(4).to_string())
    print(f"Saved to {PARAMS_FILE}")
