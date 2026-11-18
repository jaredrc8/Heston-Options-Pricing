"""
Record the verification checks in results/verification_table.csv (SPEC.md 12.7).

    python scripts/verify_outputs.py                run every check and write the table
    python scripts/verify_outputs.py --fast-only    run the fast set only
    python scripts/verify_outputs.py --output PATH  write the table somewhere else
    python scripts/verify_outputs.py --show-pytest  also show pytest's own output (for debugging)
    python scripts/verify_outputs.py --check-smoke DIR
                                                    compare a smoke run with expected.json
    python scripts/verify_outputs.py --check-smoke DIR --bless
                                                    record that smoke run as expected.json

A check that could not run is NOT_RUN with its reason, never PASS.

The report states what each check tests, what was measured, and its pass rule. A FAIL is a
recorded result of the check, not a crash. Tests that are not numbered checks are unit tests
of the code: if one of those fails, the report says so separately, because that is a code error.
"""

import argparse
import csv
import datetime
import json
import re
import subprocess
import sys
import tempfile
import textwrap
import xml.etree.ElementTree as ElementTree
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPECTED = ROOT / "results" / "smoke" / "expected.json"
SMOKE_TOLERANCE = 1e-8
COLUMNS = ["check_id", "status", "measured", "threshold", "run_id", "commit", "evaluated_at"]

# Checks that gate later work: a FAIL here gives exit code 3 (E_CHECK_FAILED)
TEST_NAME = re.compile(r"^test_(V|B)(\d+)")

# What each check tests, in plain words (SPEC.md 9.4 and 16)
DESCRIPTIONS = {
    "B1": "Runtime: one production pricing of the reference contract takes at most 30 s",
    "B2": "Runtime: the full convergence sweep takes at most 6 hours",
    "V-00": "Raw GOOG data (option file and minute bars) obeys every data-contract rule",
    "V-01": "Semi-analytic European price matches a published benchmark (Fang and Oosterlee, via Chan)",
    "V-02": "Put-call parity holds, and the model reduces to Black-Scholes when vol-of-vol is tiny",
    "V-03": "Monte Carlo European prices agree with the semi-analytic price on 40 options",
    "V-04": "LSMC agrees with a binomial tree when volatility is constant, for puts and calls",
    "V-05": "With one exercise date, the LSMC price equals the European price",
    "V-06": "LSMC matches published American put prices (Ikonen and Toivanen)",
    "V-07": "The vectorized LSMC matches an independent scalar implementation",
    "V-08": "Repeating a run with the same seed gives identical numbers",
    "V-09": "The price is stable when the regression basis degree changes",
    "V-10": "The price is stable when sub-steps go from 4 to 8",
    "V-11": "The five production seeds agree with each other",
    "V-12": "Calibration recovers known parameters from a synthetic surface",
    "V-13": "The GOOG 2026-10-13 calibration meets the 0.01 accuracy limit and is stable",
    "V-14": "Implied-volatility conversion is correct (round trip, and checks on the GOOG file)",
    "V-15": "Every reported run is complete, finite, and matches its recorded hashes",
    "V-16": "No licensed data or secrets are in Git, and the pre-commit hook is active",
    "V-17": "The smoke test passes from a fresh clone within 10 minutes",
    "V-18": "Every figure and table traces to a result file, and the figures regenerate",
    "V-19": "The out-of-sample test contracts share no contract with the calibration grid",
}

# Recorded failures the owner has reviewed and accepted (HANDOVER.md, AI_USE.md). Still FAIL.
ACCEPTED_FAILURES = {
    "V-04": "Accepted 2026-11-16: LSMC's known low bias on a call where early exercise is never worth "
            "anything; the paths and payoff are correct. Not a code error.",
    "V-13": "A finding about the model: Heston misfits the GOOG surface, mostly at short maturities. "
            "The fit is used and labelled REJECTED (SPEC.md 6.7 step 3). Not a code error.",
}


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


def failure_reason(case):
    """The assertion message of a failed test, without the traceback."""
    node = case.find("failure")
    if node is None:
        node = case.find("error")
    message = (node.get("message") or "").strip() if node is not None else ""
    return message.splitlines()[0] if message else "failed"


def run_pytest(fast_only, show_pytest):
    """
    Run the tests. Returns (rows, code_errors): one row per numbered check as
    (check_id, status, measured, threshold), and the names of failed unit tests.
    """
    with tempfile.TemporaryDirectory() as tmp:
        xml_path = Path(tmp) / "junit.xml"
        command = [sys.executable, "-m", "pytest", "-q", f"--junitxml={xml_path}", "-p", "no:cacheprovider"]
        if fast_only:
            command += ["-m", "fast"]
        if not show_pytest:
            command += ["--tb=no"]
        subprocess.run(command, cwd=ROOT, capture_output=not show_pytest)
        tree = ElementTree.parse(xml_path)

    rows, code_errors = [], []
    for case in tree.getroot().iter("testcase"):
        failed = case.find("failure") is not None or case.find("error") is not None
        match = TEST_NAME.match(case.get("name"))
        if not match:
            if failed:
                code_errors.append(f"{case.get('classname')}.{case.get('name')}: {failure_reason(case)}")
            continue
        check_id = f"{match.group(1)}-{match.group(2)}" if match.group(1) == "V" else match.group(1) + match.group(2)
        properties = {p.get("name"): p.get("value") for p in case.iter("property")}
        measured = properties.get("measured", "")
        skipped = case.find("skipped")
        if failed:
            status = "FAIL"
            measured = measured or failure_reason(case)
        elif skipped is not None:
            status = "NOT_RUN"
            measured = skipped.get("message", "skipped")
        else:
            status = "PASS"
        rows.append((check_id, status, measured, properties.get("threshold", "")))
    return rows, code_errors


def print_report(table, code_errors, output):
    """A readable report: what each check tests, what was measured, and its pass rule."""
    def wrapped(label, text):
        return textwrap.fill(text, width=100, initial_indent=f"      {label:<9}", subsequent_indent=" " * 15)

    print("\nVERIFICATION REPORT (SPEC.md section 16)")
    print("=" * 100)
    for status, heading in (("PASS", "Passed"), ("FAIL", "Failed (recorded results of the check)"),
                            ("NOT_RUN", "Not run")):
        rows = [row for row in table if row[1] == status]
        if not rows:
            continue
        print(f"\n{heading}: {len(rows)}")
        for check_id, _, measured, threshold in rows:
            print(f"  {status:<7} {check_id:<5} {DESCRIPTIONS.get(check_id, '')}")
            if status != "PASS" or measured:
                print(wrapped("Result:", measured or "-"))
            if status != "NOT_RUN":
                print(wrapped("Rule:", threshold or "-"))
            if status == "FAIL":
                if check_id in ACCEPTED_FAILURES:
                    print(wrapped("Note:", ACCEPTED_FAILURES[check_id]))

    print("\n" + "=" * 100)
    counts = ", ".join(f"{sum(row[1] == s for row in table)} {s}" for s in ("PASS", "FAIL", "NOT_RUN"))
    print(f"Checks: {counts}. Table written to {output}")
    if code_errors:
        print(f"\nCODE ERRORS: {len(code_errors)} unit test(s) failed. These are not check results; "
              "rerun with --show-pytest to see them:")
        for line in code_errors:
            print(textwrap.fill(line, width=100, initial_indent="  - ", subsequent_indent="    "))
    else:
        print("Unit tests of the code: all passed (no code errors).")
    if any(row[1] == "FAIL" for row in table):
        print("Exit code 3 means at least one check recorded FAIL (SPEC.md 14.1), not that the code crashed.")


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
    parser.add_argument("--show-pytest", action="store_true", help="also show pytest's own output")
    args = parser.parse_args()

    if args.check_smoke:
        return check_smoke(args.check_smoke, args.bless)

    print("Running the verification checks" + (" (fast set)" if args.fast_only else " (about 2-3 minutes)") + "...",
          flush=True)
    rows, code_errors = run_pytest(args.fast_only, args.show_pytest)
    table = combine(rows)
    commit = git_commit()
    evaluated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(COLUMNS)
        for check_id, status, measured, threshold in table:
            writer.writerow([check_id, status, measured, threshold, "", commit, evaluated_at])

    print_report(table, code_errors, output)
    if code_errors:
        return 1
    return 3 if any(row[1] == "FAIL" for row in table) else 0


if __name__ == "__main__":
    sys.exit(main())
