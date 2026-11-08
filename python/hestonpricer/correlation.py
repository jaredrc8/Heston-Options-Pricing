"""Correlation of daily log returns (SPEC.md section 5.4). Used only by EXT-A."""

import numpy as np

from hestonpricer.errors import HestonError


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
