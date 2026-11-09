"""
Daily log returns and their correlation (SPEC.md section 5.4).

The first functions work on Restricted bars, so they must never print or log values.
"""

import numpy as np
import pandas as pd

from hestonpricer.errors import HestonError


def last_regular_session_closes(bars):
    """
    The last close in 09:30-16:00 ET for each ticker and session.
    bars has columns ticker, date, local_time (New York time, no time zone) and close.
    Returns columns ticker, date, close.
    """
    minutes = bars["local_time"].dt.hour * 60 + bars["local_time"].dt.minute
    regular = bars[minutes.between(570, 959)]
    closes = regular.sort_values("local_time").groupby(["ticker", "date"], as_index=False).last()
    return closes[["ticker", "date", "close"]]


def count_large_minute_moves(bars, limit=0.5):
    """How many one-minute log returns exceed limit in size. Bars not 60 s apart are ignored."""
    ordered = bars.sort_values(["ticker", "local_time"])
    gap = ordered.groupby("ticker")["local_time"].diff()
    moves = np.log(ordered["close"]).groupby(ordered["ticker"]).diff()
    return int(((gap == pd.Timedelta(seconds=60)) & (moves.abs() > limit)).sum())


def adjust_for_splits(closes, splits):
    """
    Divide prices before each split by its ratio.
    splits is {ticker: [{"date": ..., "ratio": ...}, ...]}.
    """
    closes = closes.copy()
    dates = pd.to_datetime(closes["date"])
    for ticker, events in splits.items():
        for event in events:
            before = (closes["ticker"] == ticker) & (dates < pd.Timestamp(event["date"]))
            closes.loc[before, "close"] = closes.loc[before, "close"] / float(event["ratio"])
    return closes


def daily_log_returns(closes):
    """Close-to-close log returns. Returns columns ticker, date, log_return."""
    ordered = closes.sort_values(["ticker", "date"]).copy()
    ordered["log_return"] = np.log(ordered["close"]).groupby(ordered["ticker"]).diff()
    return ordered.dropna(subset=["log_return"])[["ticker", "date", "log_return"]].reset_index(drop=True)


def estimate_correlation(log_returns, window=252):
    """
    Pearson correlation over the trailing `window` days.

    log_returns has columns ticker, date, log_return. Tickers are inner-joined on
    date. Returns only aggregates, as section 12.6 requires: no returns or prices.
    """
    table = log_returns.pivot(index="date", columns="ticker", values="log_return")
    table = table.dropna().sort_index().tail(window)
    matrix = table.corr().to_numpy()

    valid = (np.allclose(matrix, matrix.T)
             and np.allclose(np.diag(matrix), 1.0)
             and np.all(np.abs(matrix) <= 1.0 + 1e-12)
             and np.min(np.linalg.eigvalsh(matrix)) >= -1e-10)
    if not valid:
        raise HestonError("E_CORR_NOT_PSD", "correlation matrix failed the section 5.4 checks")

    return {
        "tickers": list(table.columns),
        "window": window,
        "n_obs": int(len(table)),
        "first_date": str(table.index[0])[:10],
        "last_date": str(table.index[-1])[:10],
        "realized_vol": {t: float(table[t].std(ddof=1) * np.sqrt(252)) for t in table.columns},
        "matrix": matrix.tolist(),
    }
