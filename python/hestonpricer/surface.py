"""
Quote loading, filtering and grid selection (SPEC.md sections 6.2 to 6.4).

Everything here handles Restricted data when run on licensed files, so nothing
in this module prints or logs row values.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from hestonpricer.errors import HestonError
from hestonpricer.iv import american_implied_vol, bs_implied_vol

# Raw iVolatility header -> canonical name (DATA_CONTRACT.md section 1.2).
# Headers not listed here ('is_settlement', '*') are not in the contract and are dropped.
RAW_IV_HEADERS = {
    "adjusted close": "stock_price",
    "option symbol": "option_symbol",
    "call/put": "call_put",
    "price": "mean_price",
    "open interest": "open_interest",
    "stock price for iv": "stock_price_for_iv",
}

IV_COLUMNS = ["symbol", "exchange", "date", "stock_price", "option_symbol", "expiration",
              "strike", "call_put", "style", "bid", "ask", "mean_price", "iv", "volume",
              "open_interest", "stock_price_for_iv", "delta", "gamma", "vega", "theta", "rho"]

TARGET_TENORS = [0.10, 0.25, 0.50, 0.75, 1.00, 1.25]
TARGET_MONEYNESS = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20]


def load_raw_iv(path):
    """Read a raw iVolatility or fixture CSV and return the canonical columns."""
    try:
        raw = pd.read_csv(path, low_memory=False).rename(columns=RAW_IV_HEADERS)
        missing = [c for c in IV_COLUMNS if c not in raw.columns]
        if missing:
            raise HestonError("E_SCHEMA", f"{Path(path).name}: missing columns {missing}")
        raw = raw[IV_COLUMNS].copy()
        raw["date"] = pd.to_datetime(raw["date"])
        raw["expiration"] = pd.to_datetime(raw["expiration"])
    except HestonError:
        raise
    except Exception as error:
        # Sanitized: pandas messages can echo row contents (SPEC.md 14.2)
        raise HestonError("E_SCHEMA", f"{Path(path).name}: rule=readable_file "
                                      f"error={type(error).__name__}") from None
    return raw


def add_quote_columns(quotes, r, q):
    """Add mid, days to expiry, tenor, spot and forward."""
    quotes = quotes.copy()
    quotes["mid"] = (quotes["bid"] + quotes["ask"]) / 2
    quotes["days"] = (quotes["expiration"] - quotes["date"]).dt.days
    quotes["tenor"] = quotes["days"] / 365.0
    quotes["spot"] = quotes["stock_price_for_iv"]
    quotes["forward"] = quotes["spot"] * np.exp((r - q) * quotes["tenor"])
    return quotes


def own_implied_vol(row, price, r, q):
    """Implied vol of one quote at the given price: American tree, or Black-Scholes for E."""
    kind = "call" if row["call_put"] == "C" else "put"
    if row["style"] == "A":
        return american_implied_vol(kind, price, row["spot"], row["strike"], row["tenor"], r, q)
    return bs_implied_vol(kind, price, row["spot"], row["strike"], row["tenor"], r, q)


def filter_quotes(raw, cfg):
    """
    Apply filters F1 to F8 in order. cfg has keys symbol, snapshot_date, r, q.

    Returns (quotes, counts). counts holds the rows dropped by each filter and
    mid_mismatch, the rows where mid differs from mean_price by more than 1e-6.
    In the returned quotes, own_iv is the section 6.3 value and iv is the vendor's.
    """
    r, q = float(cfg["r"]), float(cfg["q"])
    counts = {}

    # F1: requested symbol and snapshot date
    keep = (raw["symbol"] == cfg["symbol"]) & (raw["date"] == pd.Timestamp(cfg["snapshot_date"]))
    counts["F1"] = int((~keep).sum())
    quotes = add_quote_columns(raw[keep], r, q)
    counts["mid_mismatch"] = int(((quotes["mid"] - quotes["mean_price"]).abs() > 1e-6).sum())

    # F2: valid vendor IV and a real two-sided quote
    keep = (quotes["iv"] > 0) & np.isfinite(quotes["iv"]) & (quotes["bid"] > 0) & \
           (quotes["ask"] >= quotes["bid"])
    counts["F2"] = int((~keep).sum())
    quotes = quotes[keep]

    # F3: days to expiry
    keep = quotes["days"].between(21, 511)
    counts["F3"] = int((~keep).sum())
    quotes = quotes[keep]

    # F4: moneyness against the forward
    keep = (quotes["strike"] / quotes["forward"]).between(0.70, 1.30)
    counts["F4"] = int((~keep).sum())
    quotes = quotes[keep]

    # F5: out of the money
    keep = ((quotes["call_put"] == "C") & (quotes["strike"] >= quotes["forward"])) | \
           ((quotes["call_put"] == "P") & (quotes["strike"] < quotes["forward"]))
    counts["F5"] = int((~keep).sum())
    quotes = quotes[keep]

    # F6: bid-ask spread relative to mid
    keep = (quotes["ask"] - quotes["bid"]) / quotes["mid"] <= 0.50
    counts["F6"] = int((~keep).sum())
    quotes = quotes[keep]

    # F7: open interest
    keep = quotes["open_interest"] >= 10
    counts["F7"] = int((~keep).sum())
    quotes = quotes[keep]

    # F8: mid above intrinsic value, and an implied-volatility root exists
    own_iv = []
    for _, row in quotes.iterrows():
        sign = 1.0 if row["call_put"] == "C" else -1.0
        intrinsic = max(sign * (row["spot"] - row["strike"]), 0.0)
        own_iv.append(own_implied_vol(row, row["mid"], r, q) if row["mid"] > intrinsic
                      else float("nan"))
    quotes = quotes.assign(own_iv=np.array(own_iv, dtype=float))
    keep = np.isfinite(quotes["own_iv"])
    counts["F8"] = int((~keep).sum())
    quotes = quotes[keep].reset_index(drop=True)

    return quotes, counts


def nearest_quote(candidates, target):
    """The row whose k_over_f is nearest target, within 0.025. Ties go to the lower strike."""
    distance = (candidates["k_over_f"] - target).abs()
    near = candidates[distance <= 0.025]
    if len(near) == 0:
        return None
    order = pd.DataFrame({"distance": (near["k_over_f"] - target).abs().round(12),
                          "strike": near["strike"]}).sort_values(["distance", "strike"])
    return near.loc[order.index[0]]


def has_contract_point(grid, contract_strike):
    """A point with tenor in [0.70, 1.30] and |K/S - K_c| <= 0.025."""
    return bool(((grid["tenor"].between(0.70, 1.30)) &
                 ((grid["k_over_s"] - contract_strike).abs() <= 0.025)).any())


def select_grid(quotes, cfg):
    """
    Deterministic grid selection. cfg has keys contract_strike and contract_maturity.
    The iv column of the result holds the section 6.3 value, never the vendor value.
    Raises E_SPARSE_GRID if the grid does not meet the section 6.4 requirements.
    """
    quotes = quotes.copy()
    quotes["k_over_f"] = quotes["strike"] / quotes["forward"]
    quotes["k_over_s"] = quotes["strike"] / quotes["spot"]
    empty = quotes.iloc[0:0].assign(iv=quotes["own_iv"].iloc[0:0])
    if len(quotes) == 0:
        raise HestonError("E_SPARSE_GRID", "no quotes survive the filters")

    # 1. The surviving expiry nearest each target tenor
    expiries = quotes[["expiration", "tenor"]].drop_duplicates().sort_values("tenor")
    selected = []
    for target in TARGET_TENORS:
        nearest = expiries.iloc[(expiries["tenor"] - target).abs().to_numpy().argmin()]
        if nearest["expiration"] not in selected:
            selected.append(nearest["expiration"])

    # 2. The nearest strike to each target K/F within each expiry
    rows = []
    for expiry in selected:
        candidates = quotes[quotes["expiration"] == expiry]
        for target in TARGET_MONEYNESS:
            row = nearest_quote(candidates, target)
            if row is not None:
                rows.append(row)
    grid = pd.DataFrame(rows) if rows else empty
    grid = grid.drop_duplicates(subset=["expiration", "strike"])

    # 3. If there is no contract point, add the quote nearest the contract strike
    #    at the selected expiry nearest the contract maturity.
    contract_strike = float(cfg["contract_strike"])
    if not has_contract_point(grid, contract_strike):
        chosen = quotes[quotes["expiration"].isin(selected)]
        expiry_tenor = chosen["tenor"].iloc[
            (chosen["tenor"] - float(cfg["contract_maturity"])).abs().to_numpy().argmin()]
        at_expiry = chosen[chosen["tenor"] == expiry_tenor]
        position = (at_expiry["k_over_s"] - contract_strike).abs().to_numpy().argmin()
        grid = pd.concat([grid, at_expiry.iloc[[position]]])
        grid = grid.drop_duplicates(subset=["expiration", "strike"])

    if len(grid) < 20:
        raise HestonError("E_SPARSE_GRID", f"grid has {len(grid)} points; 20 are required")
    if grid["expiration"].nunique() < 3:
        raise HestonError("E_SPARSE_GRID", "grid has fewer than 3 expiries")
    if not has_contract_point(grid, contract_strike):
        raise HestonError("E_SPARSE_GRID", "grid has no contract point")

    grid = grid.assign(iv=grid["own_iv"])
    return grid.sort_values(["expiration", "strike"]).reset_index(drop=True)


def iv_conversion_checks(raw, quotes, cfg):
    """
    Aggregate inputs for V-14 (b) and (c), in market mode. Only counts and
    summary statistics are returned, so the result is Internal.

    (b) For call/put pairs with |K/F - 1| <= 0.05, do their bid-ask IV bands overlap?
    (c) The median absolute difference between own IV and vendor IV.
    """
    r, q = float(cfg["r"]), float(cfg["q"])
    keep = (raw["symbol"] == cfg["symbol"]) & (raw["date"] == pd.Timestamp(cfg["snapshot_date"]))
    near = add_quote_columns(raw[keep], r, q)
    near = near[((near["strike"] / near["forward"]) - 1).abs() <= 0.05]
    near = near[(near["bid"] > 0) & near["days"].between(21, 511)]

    def band(row):
        return own_implied_vol(row, row["bid"], r, q), own_implied_vol(row, row["ask"], r, q)

    n_pairs = 0
    n_agree = 0
    for _, group in near.groupby(["expiration", "strike"]):
        calls = group[group["call_put"] == "C"]
        puts = group[group["call_put"] == "P"]
        if len(calls) != 1 or len(puts) != 1:
            continue
        call_low, call_high = band(calls.iloc[0])
        put_low, put_high = band(puts.iloc[0])
        if not np.all(np.isfinite([call_low, call_high, put_low, put_high])):
            continue
        n_pairs += 1
        if call_low <= put_high and put_low <= call_high:
            n_agree += 1

    return {
        "near_money_pairs": n_pairs,
        "pairs_with_overlapping_bands": n_agree,
        "median_abs_own_minus_vendor_iv": float((quotes["own_iv"] - quotes["iv"]).abs().median()),
    }
