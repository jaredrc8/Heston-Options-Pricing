"""
Check the raw files against every rule in DATA_CONTRACT.md sections 1.1 and 1.2.

    python python/validate_raw.py --source {ivolatility|massive|fixtures}

The report holds only file names, row counts, rule names, violation counts and
row indices. It never holds a data value. Any violation stops the pipeline (exit 2).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from hestonpricer.errors import EXIT_CODES, HestonError
from hestonpricer.surface import load_raw_iv

REAL_TICKERS = {"NVDA", "AMD", "INTC"}
FIXTURE_TICKERS = {"SYN1", "SYN2", "SYN3"}
IV_DATES = {"2026-10-13", "2026-10-14", "2026-10-15"}
BAR_COLUMNS = ["ticker", "volume", "open", "close", "high", "low", "window_start", "transactions"]
MAX_INDICES = 1000    # keeps the report small; the violation count is always complete

IV_FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_raw_iv.csv"
BAR_FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_minute_bars.csv.gz"
IV_MARKET = ROOT / "data" / "nvda_amd_intc_IV_Oct13-15.csv"
BAR_MARKET = ROOT / "data" / "nvda_amd_intc_minute" / "bars"

# Column names in the Massive Parquet package -> names in the contract
BAR_PARQUET_NAMES = {"source_window_start_ns": "window_start"}


def rule_entry(name, violated):
    """One report line. violated is a boolean array, one entry per row."""
    violated = np.asarray(violated, dtype=bool)
    indices = np.flatnonzero(violated)
    return {"rule": name, "violations": int(len(indices)),
            "row_indices": [int(i) for i in indices[:MAX_INDICES]]}


def table_rule(name, ok):
    """A rule about the whole table: one violation if it fails, with no row index."""
    return {"rule": name, "violations": 0 if ok else 1, "row_indices": []}


def check_iv_table(df, tickers):
    """Every rule in DATA_CONTRACT.md section 1.2."""
    is_call = df["call_put"] == "C"
    is_put = df["call_put"] == "P"
    date_text = df["date"].dt.strftime("%Y-%m-%d")

    rules = [
        rule_entry("symbol_allowed", ~df["symbol"].isin(tickers)),
        rule_entry("exchange_non_empty", df["exchange"].isna() | (df["exchange"].astype(str) == "")),
        rule_entry("date_allowed", ~date_text.isin(IV_DATES)),
        rule_entry("stock_price_positive", ~(df["stock_price"] > 0)),
        rule_entry("option_symbol_non_empty", df["option_symbol"].isna() | (df["option_symbol"].astype(str) == "")),
        rule_entry("expiration_after_date", ~(df["expiration"] > df["date"])),
        rule_entry("strike_positive", ~(df["strike"] > 0)),
        rule_entry("call_put_allowed", ~df["call_put"].isin(["C", "P"])),
        rule_entry("style_allowed", ~df["style"].isin(["A", "E"])),
        rule_entry("bid_non_negative", ~(df["bid"] >= 0)),
        rule_entry("ask_at_least_bid", ~(df["ask"] >= df["bid"])),
        rule_entry("mean_price_within_quote", ~((df["bid"] <= df["mean_price"]) & (df["mean_price"] <= df["ask"]))),
        rule_entry("volume_non_negative", ~(df["volume"] >= 0)),
        rule_entry("open_interest_non_negative", ~(df["open_interest"] >= 0)),
        rule_entry("stock_price_for_iv_positive", ~(df["stock_price_for_iv"] > 0)),
        rule_entry("delta_range", ~((is_call & df["delta"].between(0, 1)) | (is_put & df["delta"].between(-1, 0)))),
        rule_entry("gamma_non_negative", ~(df["gamma"] >= 0)),
        rule_entry("vega_non_negative", ~(df["vega"] >= 0)),
        rule_entry("theta_finite", ~np.isfinite(df["theta"])),
        rule_entry("rho_finite", ~np.isfinite(df["rho"])),
        rule_entry("unique_symbol_date_option", df.duplicated(["symbol", "date", "option_symbol"], keep=False)),
    ]

    # Every symbol has records on all three dates
    dates_per_symbol = date_text.groupby(df["symbol"]).nunique()
    rules.append(table_rule("every_symbol_on_all_dates", bool((dates_per_symbol == 3).all())))

    # Not a violation: rows with no IV are excluded later, and counted here.
    no_valid_iv = int((~(df["iv"] > 0)).sum())
    return rules, {"rows_without_valid_iv": no_valid_iv}


def check_bar_table(df, tickers):
    """Every rule in DATA_CONTRACT.md section 1.1."""
    stamps = pd.to_datetime(df["window_start"], unit="ns", utc=True)
    local = stamps.dt.tz_convert("America/New_York")
    minutes = local.dt.hour * 60 + local.dt.minute
    low_end = pd.Timestamp("2011-11-01", tz="UTC").value
    high_end = pd.Timestamp("2026-08-01", tz="UTC").value

    return [
        rule_entry("ticker_allowed", ~df["ticker"].isin(tickers)),
        rule_entry("volume_non_negative", ~(df["volume"] >= 0)),
        rule_entry("open_positive", ~(df["open"] > 0)),
        rule_entry("close_positive", ~(df["close"] > 0)),
        rule_entry("high_consistent", ~((df["high"] > 0) & (df["high"] >= df[["open", "close"]].max(axis=1)))),
        rule_entry("low_consistent", ~((df["low"] > 0) & (df["low"] <= df[["open", "close"]].min(axis=1)))),
        rule_entry("window_start_on_minute", df["window_start"] % 60_000_000_000 != 0),
        rule_entry("window_start_in_range", ~((df["window_start"] >= low_end) & (df["window_start"] < high_end))),
        rule_entry("window_start_in_session_hours", ~minutes.between(240, 1200)),
        rule_entry("transactions_non_negative", ~(df["transactions"] >= 0)),
        rule_entry("unique_ticker_window", df.duplicated(["ticker", "window_start"], keep=False)),
        rule_entry("no_nulls", df[BAR_COLUMNS].isna().any(axis=1)),
    ]


def read_bars(path):
    """Read one bar file. Parquet columns are renamed and decimals become float64."""
    if path.name.endswith(".parquet"):
        table = pq.read_table(path).to_pandas().rename(columns=BAR_PARQUET_NAMES)
        table = table[BAR_COLUMNS].copy()
        for column in ("volume", "open", "close", "high", "low"):
            table[column] = table[column].astype(float)
        return table, True

    table = pd.read_csv(path)
    return table, list(table.columns) == BAR_COLUMNS


def validate_iv(path, tickers):
    df = load_raw_iv(path)
    rules, notes = check_iv_table(df, tickers)
    return {"file": path.name, "rows": int(len(df)), "rules": rules, **notes}


def validate_bars(path, tickers):
    df, columns_ok = read_bars(path)
    rules = [table_rule("columns_exact", columns_ok)]
    if columns_ok:
        rules += check_bar_table(df, tickers)
    return {"file": path.name, "rows": int(len(df)), "rules": rules}


def sanitized(function, path, tickers):
    """Pandas errors can echo row contents, so only the error type is kept (SPEC.md 14.2)."""
    try:
        return function(path, tickers)
    except HestonError:
        raise
    except Exception as error:
        raise HestonError("E_SCHEMA", f"{path.name}: rule=readable_file error={type(error).__name__}") from None


def main():
    parser = argparse.ArgumentParser(description="Validate raw data against the data contract")
    parser.add_argument("--source", required=True, choices=["ivolatility", "massive", "fixtures"])
    parser.add_argument("--report", default=None, help="report path (default results/validation_<source>.json)")
    args = parser.parse_args()

    try:
        if args.source == "fixtures":
            files = [sanitized(validate_iv, IV_FIXTURE, FIXTURE_TICKERS),
                     sanitized(validate_bars, BAR_FIXTURE, FIXTURE_TICKERS)]
        elif args.source == "ivolatility":
            files = [sanitized(validate_iv, IV_MARKET, REAL_TICKERS)]
        else:
            paths = sorted(BAR_MARKET.rglob("*.parquet"))
            if not paths:
                raise HestonError("E_SCHEMA", "no Massive bar files found")
            files = [sanitized(validate_bars, path, REAL_TICKERS) for path in paths]
    except HestonError as error:
        print(str(error), file=sys.stderr)
        return EXIT_CODES[error.code]

    report_path = Path(args.report) if args.report else ROOT / "results" / f"validation_{args.source}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"source": args.source, "files": files}, indent=2))

    total = sum(rule["violations"] for item in files for rule in item["rules"])
    print(f"{args.source}: {len(files)} file(s), {sum(i['rows'] for i in files)} rows, {total} violation(s)")
    print(f"report: {report_path}")
    if total:
        print("E_SCHEMA: data contract violated; see the report", file=sys.stderr)
        return EXIT_CODES["E_SCHEMA"]
    return 0


if __name__ == "__main__":
    sys.exit(main())
