#!/usr/bin/env bash
# Reviewer smoke path (SPEC.md section 11). Uses Public material only.
# Set PYTHON to choose the interpreter, for example PYTHON=.venv/bin/python.
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "== 1. Regenerate the synthetic fixtures"
"$PY" python/make_fixtures.py
"$PY" python/validate_raw.py --source fixtures --report "$TMP/validation_fixtures.json"

echo "== 2. Calibrate to the synthetic surface"
"$PY" scripts/run.py calibrate --config configs/smoke.yaml --results-root "$TMP"

echo "== 3. Price the CORE contract with the fixture's known parameters"
"$PY" scripts/run.py price --config configs/smoke.yaml --results-root "$TMP"

echo "== 4. Compare with results/smoke/expected.json"
"$PY" scripts/verify_outputs.py --check-smoke "$TMP"

echo "== 5. Fast test set"
"$PY" -m pytest -m fast -q

echo "smoke: OK"
