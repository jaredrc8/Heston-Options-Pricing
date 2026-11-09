"""
Estimate the correlation of daily log returns for NVDA, AMD, INTC and GOOG (SPEC.md 5.4).

    python scripts/estimate_correlations.py [--source {market|fixtures}] [--results-root <dir>]

In market mode, Restricted closes and returns go to data/derived/, and only aggregates
go to results/estimates.json. Nothing here prints or logs a price or a return.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from hestonpricer import correlation
from hestonpricer.errors import EXIT_CODES, HestonError

DATASET = "nvda_amd_intc_goog_minute"
BARS = ROOT / "data" / DATASET / "bars"
DERIVED = ROOT / "data" / "derived"
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_minute_bars.csv.gz"
MARKET_TICKERS = ["NVDA", "AMD", "INTC", "GOOG"]
FIXTURE_TICKERS = ["SYN1", "SYN2", "SYN3"]
WINDOWS = [126, 252, 504]
PRIMARY_WINDOW = 252
MOVE_LIMIT = 0.5


def check_governance():
    """Licensed data is read only when authorized (SPEC.md 5.2)."""
    path = ROOT / "configs" / "governance.yaml"
    authorized = path.exists() and yaml.safe_load(path.read_text()).get("market_use_authorized") is True
    if not authorized:
        raise HestonError("E_GOV_UNAUTHORIZED", "market_use_authorized is not true in configs/governance.yaml")


def read_month(folder):
    """One month of bars for the four tickers, with columns ticker, date, local_time, close."""
    columns = ["ticker", "session_date", "exchange_local_timestamp", "close"]
    table = pa.concat_tables([pq.ParquetFile(part).read(columns=columns) for part in sorted(folder.glob("*.parquet"))])
    table = table.filter(pc.is_in(table["ticker"], value_set=pa.array(MARKET_TICKERS)))
    table = table.set_column(3, "close", table["close"].cast(pa.float64(), safe=False))
    return table.to_pandas().rename(columns={"session_date": "date", "exchange_local_timestamp": "local_time"})


def market_closes():
    """Split-adjusted closes, reading one month at a time so the data fits in memory."""
    months = sorted({path.parent for path in BARS.rglob("*.parquet")})
    if not months:
        raise HestonError("E_SCHEMA", "no Massive bar files found")

    parts = []
    large_minute_moves = 0
    for month in months:
        bars = read_month(month)
        large_minute_moves += correlation.count_large_minute_moves(bars, MOVE_LIMIT)
        parts.append(correlation.last_regular_session_closes(bars))

    splits = yaml.safe_load((ROOT / "configs" / "splits.yaml").read_text())["splits"]
    closes = correlation.adjust_for_splits(pd.concat(parts, ignore_index=True), splits)
    return closes, large_minute_moves, len(months)


def fixture_closes():
    """The same steps on the synthetic bars, which need no split adjustment."""
    bars = pd.read_csv(FIXTURE)
    local = pd.to_datetime(bars["window_start"], unit="ns", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    bars = bars.assign(date=local.dt.date, local_time=local)[["ticker", "date", "local_time", "close"]]
    return correlation.last_regular_session_closes(bars), correlation.count_large_minute_moves(bars, MOVE_LIMIT), 1


def run(source, results_root):
    market = source == "market"
    if market:
        check_governance()
    closes, large_minute_moves, n_months = market_closes() if market else fixture_closes()

    if large_minute_moves:
        raise HestonError("E_SCHEMA", f"rule=one_minute_move_above_limit count={large_minute_moves}")
    returns = correlation.daily_log_returns(closes)
    large_daily_moves = int((returns["log_return"].abs() > MOVE_LIMIT).sum())
    if large_daily_moves:
        raise HestonError("E_SCHEMA", f"rule=daily_move_above_limit_after_split_adjustment count={large_daily_moves}")

    if market:
        DERIVED.mkdir(parents=True, exist_ok=True)
        closes.to_parquet(DERIVED / "session_close.parquet", index=False)
        returns.to_parquet(DERIVED / "log_returns.parquet", index=False)

    document = {
        "schema_version": 1,
        "data_mode": source,
        "dataset": DATASET if market else FIXTURE.name,
        "tickers": MARKET_TICKERS if market else FIXTURE_TICKERS,
        "primary_window": PRIMARY_WINDOW,
        "months_read": n_months,
        "windows": [correlation.estimate_correlation(returns, window) for window in WINDOWS],
    }
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "estimates.json").write_text(json.dumps(document, indent=2))
    print(f"estimates written ({source}): windows {WINDOWS}, months read {n_months}")


def main():
    parser = argparse.ArgumentParser(description="Estimate daily-return correlations")
    parser.add_argument("--source", choices=["market", "fixtures"], default="market")
    parser.add_argument("--results-root", default=str(ROOT / "results"))
    args = parser.parse_args()
    try:
        run(args.source, Path(args.results_root))
    except HestonError as error:
        print(str(error), file=sys.stderr)
        return EXIT_CODES.get(error.code, 1)
    except Exception as error:
        # Sanitized: pandas and pyarrow messages can echo row contents (SPEC.md 14.2)
        print(f"internal error: {type(error).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
