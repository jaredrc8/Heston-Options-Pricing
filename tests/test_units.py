"""Unit tests for the edge cases, errors and data-boundary rules of SPEC.md sections 10 and 14."""

import csv
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
from hestonpricer.errors import HestonError, InputError
from hestonpricer import market
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


def test_simulator_is_single_asset():
    rng = np.random.Generator(np.random.PCG64(3))
    assert simulate([PARAMS], [[1.0]], MARKET, CONTRACT, SIM, rng)["S"].shape == (4000, 11, 1)
    with pytest.raises(InputError):
        simulate([PARAMS, PARAMS], np.eye(2), {"r": 0.04, "q": [0.0, 0.0]}, CONTRACT, SIM, rng)
    with pytest.raises(InputError):
        simulate([PARAMS], [[0.5]], MARKET, CONTRACT, SIM, rng)


def test_contract_kind_must_be_put_or_call():
    with pytest.raises(InputError):
        price_bermudan([PARAMS], [[1.0]], MARKET, {**CONTRACT, "kind": "straddle"}, SIM)


def test_call_and_put_share_paths_and_satisfy_parity_in_the_european_leg():
    """Same seed, same paths: C_E - P_E = S0 e^{-qT} - K e^{-rT} up to Monte Carlo error."""
    call = price_bermudan([PARAMS], [[1.0]], MARKET, {**CONTRACT, "kind": "call"}, SIM)
    put = price_bermudan([PARAMS], [[1.0]], MARKET, CONTRACT, SIM)
    parity = 1.0 - math.exp(-MARKET["r"])
    assert abs(call["european"] - put["european"] - parity) <= 4 * math.hypot(call["se_european"], put["se_european"])
    assert call["bermudan"] >= call["european"] - 3 * call["se_premium"]


def test_call_without_dividend_has_no_early_exercise_premium_beyond_noise():
    call = price_bermudan([PARAMS], [[1.0]], MARKET, {**CONTRACT, "kind": "call"}, SIM)
    assert abs(call["premium"]) <= 3 * max(call["se_premium"], 1e-12)


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
        return {"params": {"kappa": 1.0}, "fixed_params": [], "stable": False, "rms_iv_residual": 1.0}

    monkeypatch.setattr(calibrate, "calibrate_asset", unusable)
    result = calibrate_with_ladder(None, 0.04, 0.0, None)
    assert result["ladder_step"] == 3 and result["params"] is None


def test_ladder_step_3_keeps_the_lowest_rms_fit_as_the_rejected_fit(monkeypatch):
    import hestonpricer.calibrate as calibrate

    def two_fits(grid, r, q, cfg):
        rms = 0.0105 if not cfg["fixed"] else 0.0124
        return {"params": {"kappa": 5.0 if not cfg["fixed"] else 1.4554}, "fixed_params": sorted(cfg["fixed"]),
                "stable": True, "rms_iv_residual": rms}

    monkeypatch.setattr(calibrate, "calibrate_asset", two_fits)
    result = calibrate_with_ladder(None, 0.04, 0.0, 1.4554)
    rejected = result["rejected_fit"]
    assert result["ladder_step"] == 3 and result["params"] is None
    assert rejected["from_step"] == 1 and rejected["rms_iv_residual"] == 0.0105 and rejected["limit"] == 0.01
    assert [a["step"] for a in result["ladder_attempts"]] == [1, 2]


def test_ladder_step_4_when_no_start_converges(monkeypatch):
    import hestonpricer.calibrate as calibrate

    def never(grid, r, q, cfg):
        raise HestonError("E_CALIB_FAILED", "no converged start")

    monkeypatch.setattr(calibrate, "calibrate_asset", never)
    result = calibrate_with_ladder(None, 0.04, 0.0, 1.4554)
    assert result["ladder_step"] == 4 and result["rejected_fit"] is None


def write_config(path, content):
    path.write_text(yaml.safe_dump(content, sort_keys=False))
    return path


def table_config(tmp_path, table, highlight=None):
    content = {"schema_version": 1, "name": "test_table", "source": "table", "table": str(table),
               "citation": "test table", "symbol": "GOOG", "date": "2015-02-02", "spot": 523.755, "kind": "put",
               "market_inputs": "configs/market_inputs_goog_2015.yaml",
               "parameter_sets": [{"name": "published_2015", "params_source": "configs/published_params.yaml", "set": "GOOG"}],
               "exercise": {"n_exercise": 50}, "n_paths": 2000, "seeds": [620001],
               "outputs": {"summary": str(tmp_path / "summary.json"), "per_contract": str(tmp_path / "rows.csv")}}
    if highlight:
        content["highlight"] = highlight
    return write_config(tmp_path / "table.yaml", content)


def test_table_comparison_prices_every_published_contract(tmp_path):
    table = tmp_path / "table.csv"
    table.write_text("maturity,strike,market_price\n0.9507,525.0,46.05\n0.2,500.0,9.0\n0.9507,500.0,34.6\n")
    done = subprocess.run([sys.executable, "scripts/market_test.py", "--config", str(table_config(tmp_path, table))],
                          cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    document = json.loads((tmp_path / "summary.json").read_text())
    assert document["all_puts"]["n"] == 3 and "official_contract" not in document
    rows = list(csv.DictReader(open(tmp_path / "rows.csv")))
    assert [float(r["maturity"]) for r in rows] == [0.2, 0.9507, 0.9507]       # grouped by maturity, table order within
    assert [float(r["strike"]) for r in rows] == [500.0, 525.0, 500.0]


def test_highlight_uses_the_newest_reported_run_not_an_exploratory_one(tmp_path):
    table = tmp_path / "table.csv"
    table.write_text("maturity,strike,market_price\n0.9507,525.0,46.05\n")
    highlight = {"strike": 525.0, "maturity": 0.9507, "reference_run": "goog_2015_contract"}
    done = subprocess.run([sys.executable, "scripts/market_test.py", "--config", str(table_config(tmp_path, table, highlight))],
                          cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    document = json.loads((tmp_path / "summary.json").read_text())
    official = document["official_contract"]
    reported = [p for p in ROOT.glob("results/runs/goog_2015_contract-*/config.yaml")
                if yaml.safe_load(p.read_text()).get("reported") is True]
    assert document["run_id"] in {p.parent.name for p in reported}
    assert official["market_usd"] == 46.05


def test_licensed_source_keeps_per_contract_output_under_data_derived(tmp_path):
    content = {"schema_version": 1, "name": "x", "source": "ivolatility", "raw_file": "data/not_a_real_file.csv",
               "symbol": "GOOG", "date": "2026-10-13", "market_inputs": "configs/market_inputs_goog_oct13.yaml",
               "parameter_sets": [{"name": "p", "params_source": "configs/published_params.yaml", "set": "GOOG"}],
               "exercise": "daily", "n_paths": 1000, "seeds": [1], "bootstrap": 0,
               "outputs": {"summary": str(tmp_path / "s.json"), "per_contract": str(tmp_path / "rows.parquet")}}
    done = subprocess.run([sys.executable, "scripts/market_test.py", "--config", str(write_config(tmp_path / "c.yaml", content))],
                          cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 2 and "data/derived" in done.stderr
    assert not (tmp_path / "rows.parquet").exists()


def test_grid_command_checks_each_contract(tmp_path):
    config = yaml.safe_load((ROOT / "configs" / "grid_goog_out_of_range.yaml").read_text())
    config["simulation"]["n_paths"] = 2000
    config["seeds"] = [1]
    config["grid"]["cases"] = [{"maturity": 0.5, "strikes": [0.8, 1.0]}, {"maturity": 2.0, "strikes": [1.2]}]
    path = write_config(tmp_path / "grid.yaml", config)
    done = subprocess.run([sys.executable, "scripts/run.py", "grid", "--config", str(path), "--output", str(tmp_path / "g.json")],
                          cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    rows = json.loads((tmp_path / "g.json").read_text())["rows"]
    assert [(r["maturity"], r["strike_over_spot"]) for r in rows] == [(0.5, 0.8), (0.5, 1.0), (2.0, 1.2)]
    assert all(set(r["checks"]) == {"european_within_3_se", "bermudan_not_below_european", "bermudan_not_below_intrinsic"}
               for r in rows)


def test_market_inputs_can_set_a_published_assets_dividend(tmp_path):
    run = load_run_module()
    inputs = tmp_path / "inputs.yaml"
    inputs.write_text("dividend_yield:\n  GOOG: 0.0031\n")
    config = {"data_mode": "published", "assets": ["GOOG"], "params_source": "configs/published_params.yaml",
              "market_inputs": str(inputs)}
    _, q, _, status = run.load_params(config, ROOT / "results")
    assert q == 0.0031 and status == "published"


def rejected_calibration(tmp_path, ladder_step=3):
    fit = {"v0": 0.4, "kappa": 20.0, "theta": 0.3, "xi": 4.0, "rho": -0.1}
    rejected = {"from_step": 1, "params": fit, "fixed_params": [], "rms_iv_residual": 0.0105,
                "stable": True, "limit": 0.01} if ladder_step == 3 else None
    (tmp_path / "calibration.json").write_text(json.dumps(
        {"inputs": [], "assets": {"GOOG": {"ladder_step": ladder_step, "q": 0.0026, "params": None,
                                           "rejected_fit": rejected}}}))
    return {"data_mode": "market", "assets": ["GOOG"], "params_source": "results/calibration_goog/calibration.json"}, fit


def test_rejected_fit_is_used_and_labelled(tmp_path):
    run = load_run_module()
    config, fit = rejected_calibration(tmp_path)
    params, q, _, status = run.load_params(config, tmp_path)
    assert params == fit and q == 0.0026 and status == "rejected"
    label = run.rejected_label(tmp_path, config)
    assert label["label"] == "REJECTED" and label["rms_iv_residual"] == 0.0105 and label["limit"] == 0.01


def test_ladder_step_4_has_nothing_to_price(tmp_path):
    run = load_run_module()
    config, _ = rejected_calibration(tmp_path, ladder_step=4)
    with pytest.raises(HestonError) as caught:
        run.load_params(config, tmp_path)
    assert caught.value.code == "E_CALIB_FAILED"


def test_removed_config_keys_are_unknown(tmp_path):
    run = load_run_module()
    for key, value in (("tier", "CORE"), ("allow_rejected_fit", True)):
        path = tmp_path / f"{key}.yaml"
        path.write_text(yaml.safe_dump(smoke_config(**{key: value})))
        with pytest.raises(HestonError) as caught:
            run.load_config(path)
        assert caught.value.code == "E_CONFIG"


def test_reference_config_matches_the_spec():
    run = load_run_module()
    cfg = run.load_config(ROOT / "configs" / "core_goog_reference.yaml")
    assert cfg["data_mode"] == "market" and cfg["snapshot_date"] == "2026-10-13" and cfg["reported"] is True
    assert cfg["params_source"] == "results/calibration_goog/calibration.json"
    assert cfg["contract"] == {"strike": 1.0, "maturity": 1.0, "n_exercise": 50, "kind": "put"}


def test_named_bad_bar_is_the_only_exclusion():
    sys.path.insert(0, str(ROOT / "python"))
    import validate_raw

    assert validate_raw.EXCLUDED_BARS == {"year=2018/month=7": [2090]}


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
    assert manifest["spec_version"] == "1.13" and manifest["data_contract_version"] == "0.10"
    assert result["parameter_status"] == "synthetic" and result["contract"]["kind"] == "put"


def test_compare_writes_each_comparison(tmp_path):
    subprocess.run([sys.executable, "scripts/run.py", "compare", "--config", "configs/smoke.yaml",
                    "--results-root", str(tmp_path)], cwd=ROOT, check=True, capture_output=True)
    document = json.loads(next(tmp_path.glob("comparisons/*.json")).read_text())
    assert len(document["mc_vs_analytic"]) == 40
    assert len(document["lsmc_vs_tree"]) == 8
    assert {row["kind"] for row in document["lsmc_vs_tree"]} == {"put", "call"}
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


def goog_files(tmp_path, bad_strike_row=None):
    """Synthetic GOOG inputs for validate_goog_data.py, built from the Public fixtures."""
    iv = pd.read_csv(FIXTURES / "synthetic_raw_iv.csv")
    iv = iv[(iv["symbol"] == "SYN1") & (iv["date"] == "2026-10-13")].assign(symbol="GOOG").reset_index(drop=True)
    if bad_strike_row is not None:
        iv.loc[bad_strike_row, "strike"] = -123.456
    iv_file = tmp_path / "goog_iv.csv"
    iv.to_csv(iv_file, index=False)

    bars = pd.read_csv(FIXTURES / "synthetic_minute_bars.csv.gz")
    bars["ticker"] = bars["ticker"].astype(str).replace({"SYN1": "GOOG"})     # other tickers stay and are ignored
    month = tmp_path / "bars" / "year=2020" / "month=1"
    month.mkdir(parents=True)
    bars.rename(columns={"window_start": "source_window_start_ns"}).to_parquet(month / "part-0.parquet")
    return iv_file, tmp_path / "bars"


def run_goog_validation(tmp_path, iv_file, minute_dir):
    report = tmp_path / "validation_goog.json"
    done = subprocess.run([sys.executable, "scripts/validate_goog_data.py", "--iv-file", str(iv_file),
                           "--minute-dir", str(minute_dir), "--report", str(report)],
                          cwd=ROOT, capture_output=True, text=True)
    return done, json.loads(report.read_text()) if report.exists() else None


def test_goog_validation_passes_on_synthetic_goog_files(tmp_path):
    done, report = run_goog_validation(tmp_path, *goog_files(tmp_path))
    assert done.returncode == 0, done.stderr
    assert report["total_violations"] == 0 and report["bar_months"] == 1
    assert report["bar_rows"] == 1280      # GOOG rows only; SYN2 and SYN3 are ignored


def test_goog_validation_reports_indices_but_never_values(tmp_path):
    done, report = run_goog_validation(tmp_path, *goog_files(tmp_path, bad_strike_row=3))
    assert done.returncode == 2
    strike_rule = next(r for r in report["files"][0]["rules"] if r["rule"] == "strike_positive")
    assert strike_rule["violations"] == 1 and strike_rule["row_indices"] == [3]
    assert "123.456" not in done.stdout + done.stderr + json.dumps(report)


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


# --- market comparison (SPEC.md 8.4, 8.5) ---

def test_shared_paths_give_the_same_price_as_pricing_each_contract_alone():
    tenor = 0.3
    contracts = [{"index": 0, "strike": 0.95, "kind": "put"}, {"index": 1, "strike": 1.08, "kind": "call"}]
    rows = market.price_expiration((PARAMS, 0.04, 0.0026, tenor, contracts, 3000, [11, 12], "daily"))
    n_exercise = math.ceil(252 * tenor)
    for contract, row in zip(contracts, rows):
        alone = [price_bermudan([PARAMS], [[1.0]], {"r": 0.04, "q": [0.0026]},
                                {"strike": contract["strike"], "maturity": tenor, "n_exercise": n_exercise,
                                 "kind": contract["kind"]},
                                {"n_paths": 3000, "n_substeps": 1, "basis_degree": 3, "master_seed": seed})
                 for seed in (11, 12)]
        assert row["n_exercise"] == n_exercise
        assert row["V"] == pytest.approx(np.mean([a["bermudan"] for a in alone]), rel=1e-12)
        assert row["se"] == pytest.approx(np.sqrt(sum(a["se_bermudan"] ** 2 for a in alone)) / 2, rel=1e-12)


def test_exercise_terms_daily_and_fixed():
    assert market.exercise_terms(0.5, "daily") == (126, 1)
    assert market.exercise_terms(0.9507, 50) == (50, 5)       # about one sub-step per trading day
    assert market.exercise_terms(0.08, 50) == (50, 1)


def test_published_table_becomes_a_contract_table():
    table = pd.DataFrame({"maturity": [0.5, 1.0], "strike": [100.0, 120.0], "market_price": [5.0, 18.0]})
    contracts = market.price_table_contracts(table, 100.0, "put")
    assert list(contracts["strike"]) == [1.0, 1.2] and list(contracts["mid"]) == [0.05, 0.18]
    assert list(contracts["expiration"]) == [0.5, 1.0] and (contracts["kind"] == "put").all()


def test_market_test_output_is_aggregate_with_groups_of_five(tmp_path):
    out, prices = tmp_path / "out.json", tmp_path / "prices.parquet"
    content = {"schema_version": 1, "name": "syn", "source": "ivolatility", "raw_file": "tests/fixtures/synthetic_raw_iv.csv",
               "symbol": "SYN1", "date": "2026-10-14", "market_inputs": "tests/fixtures/synthetic_market_inputs.yaml",
               "parameter_sets": [{"name": "calibrated_2026_10_13", "params_source": "results/smoke/calibration.json"},
                                  {"name": "published_2015", "params_source": "configs/published_params.yaml", "set": "GOOG"}],
               "exercise": "daily", "n_paths": 1000, "seeds": [1], "bootstrap": 20,
               "outputs": {"summary": str(out), "per_contract": str(prices)}}
    done = subprocess.run([sys.executable, "scripts/market_test.py", "--config", str(write_config(tmp_path / "c.yaml", content))],
                          cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    document = json.loads(out.read_text())
    assert document["test_set"]["grid_overlap"] == 0
    assert set(document["results"]) == set(document["in_sample_results"]) == {"calibrated_2026_10_13", "published_2015"}
    for table in (document["results"], document["in_sample_results"]):
        for summary in table.values():
            for key in ("by_maturity", "by_moneyness", "by_kind"):
                assert all(group["n"] >= 5 for group in summary[key])
    text = out.read_text()
    assert "option_symbol" not in text and "SYN1-" not in text     # no contract identifiers
    assert "NaN" not in text and "Infinity" not in text             # strict JSON
    assert len(pd.read_parquet(prices)) == 2 * (document["test_set"]["n_test"] + document["in_sample_set"]["n"])


def test_histogram_bins_are_merged_until_each_has_five():
    values = np.array([0.1] * 7 + [0.3] * 2 + [0.6] * 4 + [3.0] * 1)
    bins = market.histogram_min5(values, [0.0, 0.25, 0.5, 1.0, np.inf])
    assert all(b["n"] >= 5 for b in bins) and sum(b["n"] for b in bins) == len(values)
    assert all(math.isfinite(b["high"]) for b in bins)
