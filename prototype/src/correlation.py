#!/usr/bin/env python3
"""Return correlation between tickers using the Massive minute-bar Parquet package.

Correlates *log returns* (not prices: trending price series give spurious
correlations). For each ticker pair it reports Pearson and Spearman
correlation, the textbook p-value / Fisher-z CI (which assumes i.i.d. returns),
and a cluster-bootstrap CI that resamples whole days/months and so stays valid
under volatility clustering. It also prints a per-year breakdown and an
Engle-Granger cointegration test on daily log prices.

Examples:
    python correlation.py                         # NVDA AMD INTC, daily
    python correlation.py --freq 5min --start 2023-01-01
    python correlation.py --tickers NVDA AMD INTC SPY --freq 30min
"""
import argparse
import itertools
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import coint

# Use the extract bundled next to this script if present, else the full package.
# Both paths are built from this file's location, so the script works from any folder.
HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent / "data" / "nvda_amd_intc_minute"
DEFAULT_ROOT = str(HERE) if (HERE / "bars").is_dir() else str(PACKAGE_ROOT)

# Prices are unadjusted. Split (effective session date, ratio) — prices before
# the date are divided by the ratio. Verify against the data for new tickers:
# a split shows up as a daily log return near -ln(ratio).
SPLITS = {
    "NVDA": [("2021-07-20", 4), ("2024-06-10", 10)],
}


def load_bars(root, tickers, freq, start, end):
    """Last regular-session close per (ticker, session, bucket)."""
    daily = freq == "1d"
    bucket = "session_date" if daily else f"time_bucket(INTERVAL '{pd.Timedelta(freq)}', exchange_local_timestamp)"
    sql = f"""
        SELECT ticker, session_date, {bucket} AS bucket,
               arg_max(CAST(close AS DOUBLE), timestamp_utc) AS close
        FROM read_parquet('{root}/bars/year=*/month=*/*.parquet', hive_partitioning=true)
        WHERE ticker IN ({','.join('?' * len(tickers))})
          AND year BETWEEN ? AND ?
          AND session_date BETWEEN ? AND ?
          AND CAST(exchange_local_timestamp AS TIME) >= TIME '09:30'
          AND CAST(exchange_local_timestamp AS TIME) <  TIME '16:00'
        GROUP BY ALL
    """
    params = [*tickers, start.year, end.year, start.date(), end.date()]
    df = duckdb.connect().execute(sql, params).df()
    missing = set(tickers) - set(df["ticker"])
    if missing:
        raise SystemExit(f"No regular-session bars for: {sorted(missing)}")
    return df.pivot_table(index=["session_date", "bucket"], columns="ticker", values="close")[tickers]


def split_adjust(prices):
    prices = prices.copy()
    dates = pd.to_datetime(prices.index.get_level_values("session_date"))
    for ticker, splits in SPLITS.items():
        if ticker in prices:
            for date, ratio in splits:
                prices.loc[dates < pd.Timestamp(date), ticker] /= ratio
    return prices


def log_returns(prices, daily):
    logp = np.log(prices)
    if daily:
        return logp.diff().droplevel("bucket")
    # Intraday: carry the last price forward within a session (bars are missing
    # when nothing traded) and drop the overnight gap.
    logp = logp.groupby(level="session_date").ffill()
    return logp.groupby(level="session_date").diff()


def cluster_bootstrap_ci(x, y, clusters, n_boot=2000, alpha=0.05, seed=0):
    """Percentile CI for Pearson r, resampling whole clusters with replacement."""
    g = pd.DataFrame({"x": x, "y": y, "xx": x * x, "yy": y * y, "xy": x * y, "n": 1.0})
    sums = g.groupby(clusters).sum().to_numpy()  # per-cluster sufficient stats
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    sx, sy, sxx, syy, sxy, n = (sums[idx, k].sum(axis=1) for k in range(6))
    r = (sxy - sx * sy / n) / np.sqrt((sxx - sx**2 / n) * (syy - sy**2 / n))
    return np.quantile(r, [alpha / 2, 1 - alpha / 2])


def fisher_ci(r, n, alpha=0.05):
    z, se = math.atanh(r), 1 / math.sqrt(n - 3)
    q = stats.norm.ppf(1 - alpha / 2)
    return math.tanh(z - q * se), math.tanh(z + q * se)


def pair_stats(returns, a, b, daily):
    pair = returns[[a, b]].dropna()
    x, y = pair[a].to_numpy(), pair[b].to_numpy()
    dates = pd.to_datetime(pair.index.get_level_values("session_date"))
    # Clusters: calendar month for daily data, session day for intraday data.
    clusters = dates.to_period("M").astype(str) if daily else dates.astype(str)
    pr = stats.pearsonr(x, y)
    sr = stats.spearmanr(x, y)
    return {
        "pair": f"{a}-{b}",
        "n": len(x),
        "pearson_r": pr.statistic,
        "p_value": pr.pvalue,
        "fisher_95ci": fisher_ci(pr.statistic, len(x)),
        "bootstrap_95ci": tuple(cluster_bootstrap_ci(x, y, clusters)),
        "spearman_rho": sr.statistic,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=["NVDA", "AMD", "INTC"])
    ap.add_argument("--freq", default="1d", help="1d, or an intraday bucket such as 1min, 5min, 30min")
    ap.add_argument("--start", default="2011-11-01")
    ap.add_argument("--end", default="2026-07-31")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    args = ap.parse_args()

    tickers = [t.upper() for t in args.tickers]
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    daily = args.freq == "1d"

    prices = split_adjust(load_bars(args.root, tickers, args.freq, start, end))
    returns = log_returns(prices, daily)

    jumps = returns[returns.abs().gt(math.log(2)).any(axis=1)]
    if len(jumps):
        print("WARNING: |log return| > ln 2 — possible unhandled split:\n", jumps, "\n")

    # print(f"Tickers {tickers}  freq={args.freq}  {start.date()} .. {end.date()}  "
    #       f"({returns.index.get_level_values('session_date').nunique()} sessions)\n")

    # print("Pearson correlation matrix of log returns:")
    # print(returns.corr().round(3).to_string(), "\n")

    pairs = list(itertools.combinations(tickers, 2))
    table = pd.DataFrame([pair_stats(returns, a, b, daily) for a, b in pairs]).set_index("pair")
    for col in ("fisher_95ci", "bootstrap_95ci"):
        table[col] = table[col].map(lambda ci: f"[{ci[0]:.3f}, {ci[1]:.3f}]")
    # with pd.option_context("display.float_format", "{:.4g}".format, "display.width", 200):
    #     print("Pair statistics (H0: correlation = 0):")
    #     print(table.to_string(), "\n")

    years = pd.to_datetime(returns.index.get_level_values("session_date")).year
    by_year = pd.DataFrame({f"{a}-{b}": returns[[a, b]].groupby(years).corr().xs(a, level=1)[b]
                            for a, b in pairs})
    # print("Pearson r by year:")
    # print(by_year.round(3).to_string(), "\n")

    # Cointegration of daily log prices: do the price levels share a long-run trend?
    if daily:
        logp = np.log(prices.droplevel("bucket"))
        # print("Engle-Granger cointegration on daily log prices (H0: no cointegration):")
        for a, b in pairs:
            both = logp[[a, b]].dropna()
            _, p, _ = coint(both[a], both[b])
            # print(f"  {a}-{b}: p = {p:.4f}")
    return returns.corr()


if __name__ == "__main__":
    main()
