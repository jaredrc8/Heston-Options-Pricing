"""
Raw-data validation for GOOG (SPEC.md section 11, check V-00).

    python scripts/validate_goog_data.py [--iv-file <path>] [--minute-dir <path>] [--report <path>]

Checks data/goog_IV_Oct13.csv against every rule in DATA_CONTRACT.md section 1.2, and the GOOG
rows of the Massive minute bars against every rule in section 1.1. The rules themselves live in
python/validate_raw.py, so both scripts apply the same contract.

Writes results/validation/goog.json. Output holds only file and partition names, row counts,
rule names, violation counts, row indices and aggregate counts. It never holds a data value.
Bar row indices are positions in the whole month table, the convention of the contract's
named exclusions.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import validate_raw
from hestonpricer.errors import EXIT_CODES, HestonError

IV_FILE = ROOT / "data" / "goog_IV_Oct13.csv"
MINUTE_DIR = ROOT / "data" / "nvda_amd_intc_goog_minute" / "bars"
REPORT = ROOT / "results" / "validation" / "goog.json"
SYMBOL = "GOOG"
DATE = "2026-10-13"


def validate_iv_file(path):
    """Section 1.2 rules on the GOOG option file, which covers 2026-10-13 only."""
    df = validate_raw.load_raw_iv(path)
    rules, notes = validate_raw.check_iv_table(df, {SYMBOL}, {DATE})
    return {"file": path.name, "rows": int(len(df)), "rules": rules, **notes}


def validate_month(month):
    """Section 1.1 rules on the GOOG rows of one month folder."""
    df, columns_ok = validate_raw.read_bars(month)
    label = f"{month.parent.name}/{month.name}"
    excluded = validate_raw.EXCLUDED_BARS.get(label, [])
    goog = df.drop(index=excluded)
    goog = goog[goog["ticker"] == SYMBOL]
    rules = [validate_raw.table_rule("columns_exact", columns_ok)]
    if columns_ok and len(goog):
        positions = goog.index.to_numpy()          # month-table positions, for the report
        for rule in validate_raw.check_bar_table(goog.reset_index(drop=True), {SYMBOL}):
            rule["row_indices"] = [int(positions[i]) for i in rule["row_indices"]]
            rules.append(rule)
    return {"file": label, "rows": int(len(goog)), "excluded_rows": excluded, "rules": rules}


def sanitized(function, path):
    """Pandas errors can echo row contents, so only the error type is kept (SPEC.md 14.2)."""
    try:
        return function(path)
    except HestonError:
        raise
    except Exception as error:
        raise HestonError("E_SCHEMA", f"{path.name}: rule=readable_file error={type(error).__name__}") from None


def main():
    parser = argparse.ArgumentParser(description="Validate the GOOG raw data against the data contract")
    parser.add_argument("--iv-file", default=str(IV_FILE))
    parser.add_argument("--minute-dir", default=str(MINUTE_DIR))
    parser.add_argument("--report", default=str(REPORT))
    args = parser.parse_args()

    try:
        iv_report = sanitized(validate_iv_file, Path(args.iv_file))
        months = sorted({path.parent for path in Path(args.minute_dir).rglob("*.parquet")})
        if not months:
            raise HestonError("E_SCHEMA", "no minute-bar Parquet files found")
        bar_reports = [sanitized(validate_month, month) for month in months]
    except HestonError as error:
        print(str(error), file=sys.stderr)
        return EXIT_CODES[error.code]

    files = [iv_report] + bar_reports
    total = sum(rule["violations"] for item in files for rule in item["rules"])
    by_rule = {}
    for item in files:
        for rule in item["rules"]:
            by_rule[rule["rule"]] = by_rule.get(rule["rule"], 0) + rule["violations"]
    report = {
        "schema_version": 1, "symbol": SYMBOL, "option_date": DATE,
        "iv_file": iv_report["file"], "iv_rows": iv_report["rows"],
        "bar_months": len(bar_reports), "bar_rows": sum(item["rows"] for item in bar_reports),
        "months_without_goog_rows": [item["file"] for item in bar_reports if item["rows"] == 0],
        "total_violations": total,
        "violations_by_rule": {name: count for name, count in sorted(by_rule.items()) if count},
        "files": files,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    print(f"GOOG: option file {iv_report['rows']} rows; {len(bar_reports)} bar months, "
          f"{report['bar_rows']} GOOG bar rows; {total} violation(s)")
    print(f"report: {report_path.name}")
    if total:
        print("E_SCHEMA: data contract violated; see the report", file=sys.stderr)
        return EXIT_CODES["E_SCHEMA"]
    return 0


if __name__ == "__main__":
    sys.exit(main())
