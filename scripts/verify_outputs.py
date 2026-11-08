"""
Record the verification checks in results/verification_table.csv (SPEC.md 12.7).

    python scripts/verify_outputs.py                run every check and write the table
    python scripts/verify_outputs.py --fast-only    run the fast set only
    python scripts/verify_outputs.py --output PATH  write the table somewhere else
    python scripts/verify_outputs.py --check-smoke DIR
                                                    compare a smoke run with expected.json
    python scripts/verify_outputs.py --check-smoke DIR --bless
                                                    record that smoke run as expected.json

A check that could not run is NOT_RUN with its reason, never PASS.
"""

import argparse
import csv
import datetime
import json
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ElementTree
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPECTED = ROOT / "results" / "smoke" / "expected.json"
SMOKE_TOLERANCE = 1e-8
COLUMNS = ["check_id", "status", "measured", "threshold", "run_id", "commit", "evaluated_at"]

# Checks that gate later work: a FAIL here gives exit code 3 (E_CHECK_FAILED)
TEST_NAME = re.compile(r"^test_(V|B)(\d+)")


def git_commit():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() or "nocommit"


def check_smoke(run_root, bless):
    """Compare (or record) the pooled prices of the single smoke run in run_root."""
    results = sorted(Path(run_root).glob("runs/*/result.json"))
    if len(results) != 1:
        print(f"E_CHECK_FAILED: expected one smoke run, found {len(results)}", file=sys.stderr)
        return 3
    pooled = json.loads(results[0].read_text())["pooled"]
    names = ("bermudan", "european_mc", "premium")
    measured = {name: pooled[name] for name in names}

    if bless:
        EXPECTED.parent.mkdir(parents=True, exist_ok=True)
        EXPECTED.write_text(json.dumps({"relative_tolerance": SMOKE_TOLERANCE, **measured}, indent=2))
        print(f"recorded {EXPECTED}")
        return 0

    if not EXPECTED.exists():
        print("E_CHECK_FAILED: results/smoke/expected.json is missing", file=sys.stderr)
        return 3
    expected = json.loads(EXPECTED.read_text())
    worst = max(abs(measured[name] - expected[name]) / abs(expected[name]) for name in names)
    if worst > SMOKE_TOLERANCE:
        print(f"E_CHECK_FAILED: smoke differs from expected, relative {worst:.3e}", file=sys.stderr)
        return 3
    print(f"smoke matches expected.json (largest relative difference {worst:.3e})")
    return 0


def run_pytest(fast_only):
    """Run the tests and return one row per test as (check_id, status, measured, threshold)."""
    with tempfile.TemporaryDirectory() as tmp:
        xml_path = Path(tmp) / "junit.xml"
        command = [sys.executable, "-m", "pytest", "-q", f"--junitxml={xml_path}", "-p", "no:cacheprovider"]
        if fast_only:
            command += ["-m", "fast"]
        subprocess.run(command, cwd=ROOT)
        tree = ElementTree.parse(xml_path)

    rows = []
    for case in tree.getroot().iter("testcase"):
        match = TEST_NAME.match(case.get("name"))
        if not match:
            continue
        check_id = f"{match.group(1)}-{match.group(2)}" if match.group(1) == "V" else match.group(1) + match.group(2)
        properties = {p.get("name"): p.get("value") for p in case.iter("property")}
        measured = properties.get("measured", "")
        skipped = case.find("skipped")
        if case.find("failure") is not None or case.find("error") is not None:
            status = "FAIL"
        elif skipped is not None:
            status = "NOT_RUN"
            measured = skipped.get("message", "skipped")
        else:
            status = "PASS"
        rows.append((check_id, status, measured, properties.get("threshold", "")))
    return rows


def combine(rows):
    """Merge several tests of one check: any FAIL fails, then any NOT_RUN means NOT_RUN."""
    merged = {}
    for check_id, status, measured, threshold in rows:
        merged.setdefault(check_id, []).append((status, measured, threshold))

    output = []
    for check_id in sorted(merged):
        parts = merged[check_id]
        statuses = [part[0] for part in parts]
        status = "FAIL" if "FAIL" in statuses else "NOT_RUN" if "NOT_RUN" in statuses else "PASS"
        measured = "; ".join(part[1] for part in parts if part[1])
        threshold = "; ".join(part[2] for part in parts if part[2])
        output.append((check_id, status, measured, threshold))
    return output


def main():
    parser = argparse.ArgumentParser(description="Record verification checks")
    parser.add_argument("--fast-only", action="store_true")
    parser.add_argument("--output", default=str(ROOT / "results" / "verification_table.csv"))
    parser.add_argument("--check-smoke", default=None)
    parser.add_argument("--bless", action="store_true")
    args = parser.parse_args()

    if args.check_smoke:
        return check_smoke(args.check_smoke, args.bless)

    table = combine(run_pytest(args.fast_only))
    commit = git_commit()
    evaluated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(COLUMNS)
        for check_id, status, measured, threshold in table:
            writer.writerow([check_id, status, measured, threshold, "", commit, evaluated_at])

    for status in ("PASS", "FAIL", "NOT_RUN"):
        print(f"{status}: {sum(row[1] == status for row in table)}")
    print(f"table: {output}")
    return 3 if any(row[1] == "FAIL" for row in table) else 0


if __name__ == "__main__":
    sys.exit(main())
