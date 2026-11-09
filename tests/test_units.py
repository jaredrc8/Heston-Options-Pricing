"""Unit tests for the edge cases, errors and data-boundary rules of SPEC.md sections 10 and 14."""

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from hestonpricer import surface
from hestonpricer.analytic import heston_european
from hestonpricer.calibrate import calibrate_with_ladder, is_stable
from hestonpricer.correlation import estimate_correlation
from hestonpricer.errors import HestonError, InputError
from hestonpricer.lsmc import fit_policy, price_bermudan
from hestonpricer.simulate import simulate
from hestonpricer.tree import crr_price

pytestmark = pytest.mark.fast

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
PARAMS = {"v0": 0.09, "kappa": 2.0, "theta": 0.10, "xi": 0.6, "rho": -0.6}
MARKET = {"r": 0.04, "q": [0.0]}
CONTRACT = {"strike": 1.0, "maturity": 1.0, "n_exercise": 10}
SIM = {"n_paths": 4000, "n_substeps": 2, "master_seed": 1}


def load_run_module():
    spec = importlib.util.spec_from_file_location("run_module", ROOT / "scripts" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def smoke_config(**changes):
    config = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text())
    config.update(changes)
    return config


# --- preconditions and edge cases (SPEC.md 10.3 and 14.3) ---

def test_heston_european_rejects_zero_xi():
    with pytest.raises(InputError):
        heston_european("put", 1.0, 1.0, 1.0, {**PARAMS, "xi": 0.0}, 0.04, 0.0)


def test_heston_european_rejects_non_finite_input():
    with pytest.raises(InputError):
        heston_european("put", 1.0, float("nan"), 1.0, PARAMS, 0.04, 0.0)


def test_fit_policy_rejects_bad_degree():
    paths = {"S": np.ones((10, 3, 1)), "nu": np.ones((10, 3, 1))}
    with pytest.raises(InputError):
        fit_policy(paths, {"n_exercise": 2}, MARKET, 7)


def test_integer_maturity_and_strike_are_cast():
    as_float = price_bermudan([PARAMS], [[1.0]], MARKET, CONTRACT, SIM)
    as_int = price_bermudan([PARAMS], [[1.0]], MARKET, {"strike": 1, "maturity": 1, "n_exercise": 10}, SIM)
    assert as_int["bermudan"] == as_float["bermudan"]


def test_one_exercise_date_has_exactly_zero_premium():
    result = price_bermudan([PARAMS], [[1.0]], MARKET, {**CONTRACT, "n_exercise": 1}, SIM)
    assert result["bermudan"] == result["european"]
    assert result["premium"] == 0.0 and result["se_premium"] == 0.0


def test_zero_xi_drops_the_variance_feature_and_prices():
    params = {**PARAMS, "xi": 0.0, "v0": 0.09, "theta": 0.09}
    result = price_bermudan([params], [[1.0]], MARKET, CONTRACT, SIM)
    assert math.isfinite(result["bermudan"])


@pytest.mark.parametrize("rho", [-1.0, 1.0])
def test_perfect_correlation_runs(rho):
    result = price_bermudan([{**PARAMS, "rho": rho}], [[1.0]], MARKET, CONTRACT, SIM)
    assert math.isfinite(result["bermudan"])


def test_feller_violation_does_not_block_pricing():
    params = {"v0": 0.04, "kappa": 0.5, "theta": 0.04, "xi": 1.0, "rho": -0.7}
    assert 2 * params["kappa"] * params["theta"] < params["xi"] ** 2
    assert math.isfinite(price_bermudan([params], [[1.0]], MARKET, CONTRACT, SIM)["bermudan"])


def test_dates_with_too_few_itm_paths_are_skipped():
    result = price_bermudan([PARAMS], [[1.0]], MARKET, {**CONTRACT, "strike": 0.05}, SIM)
    assert result["skipped_dates"] == 9
    assert result["bermudan"] == result["european"]


def test_singular_correlation_is_valid_for_the_simulator():
    corr = np.ones((2, 2))
    market = {"r": 0.04, "q": [0.0, 0.0]}
    rng = np.random.Generator(np.random.PCG64(3))
    paths = simulate([PARAMS, PARAMS], corr, market, CONTRACT, SIM, rng)
    assert paths["S"].shape == (4000, 11, 2)


def test_paths_store_negative_variance_but_full_truncation_keeps_price_finite():
    params = {"v0": 0.01, "kappa": 0.5, "theta": 0.01, "xi": 2.0, "rho": -0.7}
    rng = np.random.Generator(np.random.PCG64(5))
    paths = simulate([params], [[1.0]], MARKET, CONTRACT, SIM, rng)
    assert paths["nu"].min() < 0
    assert np.all(np.isfinite(paths["S"]))


def test_regression_and_evaluation_streams_are_independent():
    child_a, child_b = np.random.SeedSequence(620001).spawn(2)
    a = np.random.Generator(np.random.PCG64(child_a)).standard_normal(5)
    b = np.random.Generator(np.random.PCG64(child_b)).standard_normal(5)
    assert not np.array_equal(a, b)


def test_bermudan_tree_lies_between_european_and_american():
    european = crr_price("put", 1.0, 1.0, 1.0, 0.3, 0.05, 0.0, 500, exercise_steps=[500])
    bermudan = crr_price("put", 1.0, 1.0, 1.0, 0.3, 0.05, 0.0, 500, exercise_steps=list(range(50, 501, 50)))
    american = crr_price("put", 1.0, 1.0, 1.0, 0.3, 0.05, 0.0, 500)
    assert european < bermudan < american


# --- quotes and grid (SPEC.md 6.2 to 6.4) ---

def fixture_quotes():
    raw = surface.load_raw_iv(FIXTURES / "synthetic_raw_iv.csv")
    r = 2.0 * math.log(1.02)
    return raw, surface.filter_quotes(raw, {"symbol": "SYN1", "snapshot_date": "2026-10-14", "r": r, "q": 0.0})


def test_filter_counts_add_up():
    raw, (quotes, counts) = fixture_quotes()
    assert sum(counts[f"F{i}"] for i in range(1, 9)) + len(quotes) == len(raw)


def test_filtered_quotes_are_out_of_the_money():
    _, (quotes, _) = fixture_quotes()
    calls = quotes[quotes["call_put"] == "C"]
    puts = quotes[quotes["call_put"] == "P"]
    assert (calls["strike"] >= calls["forward"]).all() and (puts["strike"] < puts["forward"]).all()


def test_grid_uses_own_iv_and_meets_requirements():
    _, (quotes, _) = fixture_quotes()
    tampered = quotes.assign(iv=0.999)     # the vendor value must never reach the grid
    grid = surface.select_grid(tampered, {"contract_strike": 1.0, "contract_maturity": 1.0})
    assert len(grid) >= 20 and grid["expiration"].nunique() >= 3
    assert not (grid["iv"] == 0.999).any()


def test_sparse_grid_raises():
    _, (quotes, _) = fixture_quotes()
    with pytest.raises(HestonError) as caught:
        surface.select_grid(quotes.head(5), {"contract_strike": 1.0, "contract_maturity": 1.0})
    assert caught.value.code == "E_SPARSE_GRID"


def test_nearest_quote_breaks_ties_toward_lower_strike():
    candidates = pd.DataFrame({"k_over_f": [0.98, 1.02], "strike": [98.0, 102.0]})
    assert surface.nearest_quote(candidates, 1.0)["strike"] == 98.0


def test_stability_needs_three_agreeing_starts():
    good = [{"params": {"v0": 0.1}}] * 3
    assert is_stable(good, ["v0"])
    assert not is_stable(good[:2], ["v0"])
    assert not is_stable([{"params": {"v0": v}} for v in (0.1, 0.1, 0.2)], ["v0"])


def test_ladder_reports_step_3_without_a_cited_kappa(monkeypatch):
    import hestonpricer.calibrate as calibrate

    def unusable(grid, r, q, cfg):
        return {"stable": False, "rms_iv_residual": 1.0}

    monkeypatch.setattr(calibrate, "calibrate_asset", unusable)
    assert calibrate_with_ladder(None, 0.04, 0.0, None)["ladder_step"] == 3


# --- correlation (EXT-A API only) ---

def test_correlation_matrix_is_valid_on_the_fixture():
    bars = pd.read_csv(FIXTURES / "synthetic_minute_bars.csv.gz")
    bars["date"] = pd.to_datetime(bars["window_start"]).dt.date
    # Last bar of each day for each ticker, then daily log returns
    closes = bars.sort_values("window_start").groupby(["ticker", "date"], as_index=False).last()
    closes["log_return"] = closes.groupby("ticker")["close"].transform(lambda c: np.log(c).diff())
    result = estimate_correlation(closes.dropna(subset=["log_return"]), window=252)
    assert result["n_obs"] == 252
    assert all("close" not in key for key in result)


# --- pipeline, governance and data boundary (SPEC.md 12, 14 and 15) ---

def test_unknown_config_key_exits_2(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text(yaml.safe_dump(smoke_config(unexpected=1)))
    done = subprocess.run([sys.executable, "scripts/run.py", "price", "--config", str(config),
                           "--results-root", str(tmp_path)], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 2 and "E_CONFIG" in done.stderr


def test_market_mode_is_refused_without_authorization(tmp_path, monkeypatch):
    run = load_run_module()
    (tmp_path / "configs").mkdir()
    monkeypatch.setattr(run, "ROOT", tmp_path)   # no governance.yaml here: unauthorized
    with pytest.raises(HestonError) as caught:
        run.check_governance({"data_mode": "market"})
    assert caught.value.code == "E_GOV_UNAUTHORIZED"


def test_market_mode_is_refused_when_flag_is_false(tmp_path, monkeypatch):
    run = load_run_module()
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "governance.yaml").write_text("market_use_authorized: false\n")
    monkeypatch.setattr(run, "ROOT", tmp_path)
    with pytest.raises(HestonError) as caught:
        run.check_governance({"data_mode": "market"})
    assert caught.value.code == "E_GOV_UNAUTHORIZED"


def test_reported_run_refuses_a_dirty_tree(tmp_path, monkeypatch):
    run = load_run_module()
    monkeypatch.setattr(run, "git_dirty", lambda: True)
    with pytest.raises(HestonError) as caught:
        run.price_run(smoke_config(reported=True), None, tmp_path, "test")
    assert caught.value.code == "E_DIRTY_TREE"


@pytest.mark.parametrize("status, dirty", [(" M results/calibration.json\n", False),
                                           (" M results/a.json\n M python/b.py\n", True),
                                           (" M python/b.py\n", True),
                                           ("", False)])
def test_only_changes_outside_results_make_the_tree_dirty(monkeypatch, status, dirty):
    run = load_run_module()

    def fake_git(*args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout=status, stderr="")

    monkeypatch.setattr(run.subprocess, "run", fake_git)
    assert run.git_dirty() is dirty


def test_completed_run_is_not_overwritten(tmp_path):
    arguments = [sys.executable, "scripts/run.py", "price", "--config", "configs/smoke.yaml", "--results-root", str(tmp_path)]
    assert subprocess.run(arguments, cwd=ROOT, capture_output=True).returncode == 0
    second = subprocess.run(arguments, cwd=ROOT, capture_output=True, text=True)
    assert second.returncode == 3 and "E_RUN_EXISTS" in second.stderr


def test_nonfinite_output_is_rejected():
    run = load_run_module()
    with pytest.raises(HestonError) as caught:
        run.assert_all_finite({"a": [1.0, float("nan")]})
    assert caught.value.code == "E_NONFINITE"


def test_run_writes_the_required_files_and_fields(tmp_path):
    subprocess.run([sys.executable, "scripts/run.py", "price", "--config", "configs/smoke.yaml",
                    "--results-root", str(tmp_path)], cwd=ROOT, check=True, capture_output=True)
    run_dir = next(tmp_path.glob("runs/*"))
    assert {p.name for p in run_dir.iterdir()} == {"config.yaml", "manifest.json", "result.json", "run.log"}
    result = json.loads((run_dir / "result.json").read_text())
    assert result["units"] == "fraction_of_initial_spot" and result["data_mode"] == "synthetic"
    assert len(result["diagnostics"]["exercise_fraction_by_date"]) == 50
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["spec_version"] == "1.5" and manifest["data_contract_version"] == "0.4"


def test_compare_writes_each_comparison(tmp_path):
    subprocess.run([sys.executable, "scripts/run.py", "compare", "--config", "configs/smoke.yaml",
                    "--results-root", str(tmp_path)], cwd=ROOT, check=True, capture_output=True)
    document = json.loads(next(tmp_path.glob("comparisons/*.json")).read_text())
    assert len(document["mc_vs_analytic"]) == 40
    assert len(document["lsmc_vs_tree"]) == 4
    assert len(document["strike_profile"]) == 5


def test_validator_accepts_the_fixtures(tmp_path):
    report = tmp_path / "report.json"
    done = subprocess.run([sys.executable, "python/validate_raw.py", "--source", "fixtures", "--report", str(report)],
                          cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_validator_reports_indices_but_never_values():
    sys.path.insert(0, str(ROOT / "python"))
    import validate_raw

    raw = surface.load_raw_iv(FIXTURES / "synthetic_raw_iv.csv")
    raw.loc[7, "ask"] = raw.loc[7, "bid"] - 1.0
    rules, _ = validate_raw.check_iv_table(raw, {"SYN1"})
    broken = [rule for rule in rules if rule["rule"] == "ask_at_least_bid"][0]
    assert broken["violations"] == 1 and broken["row_indices"] == [7]
    assert set(broken) == {"rule", "violations", "row_indices"}


def test_inspect_schema_prints_structure_only():
    done = subprocess.run([sys.executable, "scripts/inspect_schema.py", "tests/fixtures/synthetic_raw_iv.csv"],
                          cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0
    assert "SYN1-" not in done.stdout and "'bid'" in done.stdout
    assert "99." not in done.stdout


def test_unreadable_raw_file_gives_a_sanitized_error(tmp_path):
    path = tmp_path / "broken.csv"
    path.write_bytes(b"\x00\x01secret-row-value")
    with pytest.raises(HestonError) as caught:
        surface.load_raw_iv(path)
    assert "secret-row-value" not in str(caught.value)
    assert caught.value.__suppress_context__ or caught.value.code == "E_SCHEMA"
