"""
The verification matrix of SPEC.md section 16. Each check has a test named after it.

A check that cannot run yet is skipped with its reason, and verify_outputs.py records
that as NOT_RUN, never PASS. Monte Carlo checks use the fixed test seed 620100 unless
the spec says otherwise.
"""

import csv
import hashlib
import itertools
import json
import math
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from hestonpricer import surface
from hestonpricer.analytic import heston_european
from hestonpricer.calibrate import calibrate_asset
from hestonpricer.iv import american_implied_vol, bs_price
from hestonpricer.lsmc import evaluate_policy, fit_policy, price_bermudan
from hestonpricer.simulate import simulate
from hestonpricer.tree import crr_price
from reference_impl import scalar_lsmc

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
TEST_SEED = 620100
PRODUCTION_SEEDS = [620001, 620002, 620003, 620004, 620005]
PRODUCTION = {"n_paths": 100_000, "n_substeps": 4, "basis_degree": 3}

SYN_PARAMS = yaml.safe_load((FIXTURES / "synthetic_params.yaml").read_text())["assets"]["SYN1"]
SYN_YIELD = yaml.safe_load((FIXTURES / "synthetic_market_inputs.yaml").read_text())[
    "treasury_1y_cmt_yield"]["2026-10-14"]
SYN_R = 2.0 * math.log(1.0 + SYN_YIELD / 2.0)
SYN_MARKET = {"r": SYN_R, "q": [0.0]}
CONTRACT = {"strike": 1.0, "maturity": 1.0, "n_exercise": 50}


def reference_section(name):
    """A section of the owner-entered reference file, or skip if it is still empty."""
    section = yaml.safe_load((ROOT / "tests" / "reference_values.yaml").read_text())[name]
    if not section["citation"] or not section["table_or_page"]:
        pytest.skip(f"{name} reference values not yet transcribed by the owner")
    return section


def reported_inputs():
    """The reported parameter set: the market fit if one exists, otherwise the published set."""
    calibration = ROOT / "results" / "calibration.json"
    if calibration.exists():
        document = json.loads(calibration.read_text())
        entry = document["assets"]["AMD"]
        if entry["ladder_step"] < 3:
            return entry["params"], {"r": document["r"], "q": [entry["q"]]}

    published = yaml.safe_load((ROOT / "configs" / "published_params.yaml").read_text())
    entry = published["sets"]["GOOG"]
    if any(entry[name] is None for name in ("v0", "kappa", "theta", "xi", "rho", "q")):
        pytest.skip("no reported parameter set: OD-5 published parameters not entered")
    inputs = yaml.safe_load((ROOT / "configs" / "market_inputs.yaml").read_text())
    yields = {str(k): v for k, v in inputs["treasury_1y_cmt_yield"].items()}
    if yields.get("2026-10-14") is None:
        pytest.skip("no rate: OD-3 Treasury yield not entered")
    params = {name: float(entry[name]) for name in ("v0", "kappa", "theta", "xi", "rho")}
    return params, {"r": 2.0 * math.log(1.0 + float(yields["2026-10-14"]) / 2.0), "q": [float(entry["q"])]}


def pooled_price(params, market, contract, settings, seeds):
    """Pooled Bermudan price and standard error over several seeds (SPEC.md section 3)."""
    results = [price_bermudan([params], [[1.0]], market, contract, {**settings, "master_seed": s}) for s in seeds]
    value = float(np.mean([r["bermudan"] for r in results]))
    se = float(np.sqrt(np.sum([r["se_bermudan"] ** 2 for r in results])) / len(seeds))
    return value, se, results


def run_price(results_root, config="configs/smoke.yaml"):
    return subprocess.run([sys.executable, "scripts/run.py", "price", "--config", config,
                           "--results-root", str(results_root)], cwd=ROOT, capture_output=True, text=True)


# ----------------------------------------------------------------------
# CORE checks
# ----------------------------------------------------------------------

@pytest.mark.fast
def test_V01_published_benchmark(record_property):
    section = reference_section("V-01")
    if not section["cases"]:
        pytest.skip("V-01 has no benchmark cases entered")
    worst = 0.0
    for case in section["cases"]:
        params = {name: case[name] for name in ("v0", "kappa", "theta", "xi", "rho")}
        value = heston_european(case["kind"], case["s0"], case["strike"], case["maturity"], params, case["r"], case["q"])
        difference = abs(value - case["value"])
        worst = max(worst, difference)
        assert difference <= max(1e-6, 0.5 * case["last_digit"])
    record_property("measured", f"largest absolute difference {worst:.3e}")
    record_property("threshold", "max(1e-6, half the last published digit)")


@pytest.mark.fast
def test_V02_parity_and_black_scholes_limit(record_property):
    worst_parity = 0.0
    for strike, maturity in itertools.product([0.7, 0.9, 1.0, 1.1, 1.3], [0.1, 0.5, 1.0, 2.0]):
        call = heston_european("call", 1.0, strike, maturity, SYN_PARAMS, SYN_R, 0.01)
        put = heston_european("put", 1.0, strike, maturity, SYN_PARAMS, SYN_R, 0.01)
        parity = call - put - (math.exp(-0.01 * maturity) - strike * math.exp(-SYN_R * maturity))
        worst_parity = max(worst_parity, abs(parity))

    sigma = 0.25
    small = {"v0": sigma**2, "kappa": 1.5, "theta": sigma**2, "xi": 1e-4, "rho": 0.0}
    worst_bs = 0.0
    for strike, maturity in itertools.product([0.8, 1.0, 1.2], [0.25, 1.0, 2.0]):
        heston = heston_european("put", 1.0, strike, maturity, small, SYN_R, 0.0)
        worst_bs = max(worst_bs, abs(heston - float(bs_price("put", 1.0, strike, maturity, sigma, SYN_R, 0.0))))

    record_property("measured", f"parity {worst_parity:.2e}; Black-Scholes difference {worst_bs:.2e}")
    record_property("threshold", "parity <= 1e-10; Black-Scholes <= 1e-6")
    assert worst_parity <= 1e-10
    assert worst_bs <= 1e-6


@pytest.mark.slow
def test_V03_european_mc_grid(record_property):
    strikes = np.linspace(0.80, 1.25, 8)
    n_inside = 0
    for maturity in [0.25, 0.5, 1.0, 1.5, 2.0]:
        contract = {"strike": 1.0, "maturity": maturity, "n_exercise": 1}
        sim = {"n_paths": 200_000, "n_substeps": int(round(maturity * 400))}
        rng = np.random.Generator(np.random.PCG64(TEST_SEED))
        paths = simulate([SYN_PARAMS], [[1.0]], SYN_MARKET, contract, sim, rng)
        terminal = paths["S"][:, 1, 0]
        for strike in strikes:
            discounted = math.exp(-SYN_R * maturity) * np.maximum(strike - terminal, 0.0)
            se = discounted.std(ddof=1) / math.sqrt(len(discounted))
            exact = heston_european("put", 1.0, strike, maturity, SYN_PARAMS, SYN_R, 0.0)
            n_inside += abs(discounted.mean() - exact) <= 3 * max(se, 1e-12)
    record_property("measured", f"{n_inside} of 40 options within 3 SE")
    record_property("threshold", "at least 38 of 40")
    assert n_inside >= 38


@pytest.mark.slow
def test_V04_gbm_reduction_vs_tree(record_property):
    measured = []
    for sigma, strike in [(0.3, 1.0), (0.5, 1.0), (0.5, 0.9), (0.3, 1.1)]:
        params = {"v0": sigma**2, "kappa": 1.0, "theta": sigma**2, "xi": 0.0, "rho": 0.0}
        contract = {"strike": strike, "maturity": 1.0, "n_exercise": 50}
        result = price_bermudan([params], [[1.0]], SYN_MARKET, contract, {**PRODUCTION, "master_seed": TEST_SEED})
        tree = crr_price("put", 1.0, strike, 1.0, sigma, SYN_R, 0.0, 5000, exercise_steps=[100 * k for k in range(1, 51)])
        z = abs(result["bermudan"] - tree) / max(result["se_bermudan"], 1e-12)
        measured.append(z)
        assert z <= 3
    record_property("measured", "differences in SE: " + ", ".join(f"{z:.2f}" for z in measured))
    record_property("threshold", "<= 3 SE for each pair")


@pytest.mark.fast
def test_V05_one_exercise_date_is_european(record_property):
    contract = {"strike": 1.0, "maturity": 1.0, "n_exercise": 1}
    result = price_bermudan([SYN_PARAMS], [[1.0]], SYN_MARKET, contract,
                            {"n_paths": 20_000, "n_substeps": 200, "master_seed": TEST_SEED})
    exact = heston_european("put", 1.0, 1.0, 1.0, SYN_PARAMS, SYN_R, 0.0)
    z = abs(result["bermudan"] - exact) / max(result["se_bermudan"], 1e-12)
    record_property("measured", f"{z:.2f} SE")
    record_property("threshold", "<= 3 SE")
    assert z <= 3
    assert result["premium"] == 0.0 and result["se_premium"] == 0.0


@pytest.mark.slow
def test_V06_american_benchmark(record_property):
    section = reference_section("V-06")
    if not section["benchmarks"]:
        pytest.skip("V-06 has no benchmark spots entered")
    params = {name: float(section[name]) for name in ("v0", "kappa", "theta", "xi", "rho")}
    market = {"r": float(section["r"]), "q": [float(section["q"])]}
    worst = 0.0
    for point in section["benchmarks"]:
        spot, benchmark = float(point["spot"]), float(point["value"])
        previous = None
        for M in [25, 50, 100]:
            # Prices are in fractions of spot, so the strike is scaled and the price scaled back
            contract = {"strike": float(section["strike"]) / spot, "maturity": float(section["maturity"]), "n_exercise": M}
            result = price_bermudan([params], [[1.0]], market, contract,
                                    {**PRODUCTION, "n_paths": 200_000, "master_seed": TEST_SEED})
            value, se = result["bermudan"] * spot, result["se_bermudan"] * spot
            if previous is not None:
                assert previous[0] - value <= 3 * math.sqrt(previous[1] ** 2 + se**2)
            previous = (value, se)
        assert abs(value - benchmark) <= max(0.01 * benchmark, 3 * se)
        assert value - benchmark <= 3 * se
        worst = max(worst, abs(value - benchmark))
    record_property("measured", f"largest |V - B| at M = 100: {worst:.4f}")
    record_property("threshold", "max(0.01 B, 3 SE); V - B <= 3 SE; monotone in M within 3 SE")


@pytest.mark.fast
def test_V07_scalar_reference_implementation(record_property):
    data = np.load(FIXTURES / "shared_normals.npz")
    n = data["Z_indep"].shape[0] // 2
    set_a = {"Z_indep": data["Z_indep"][:n], "Z_perp": data["Z_perp"][:n]}
    set_b = {"Z_indep": data["Z_indep"][n:], "Z_perp": data["Z_perp"][n:]}
    contract = {"strike": 1.0, "maturity": 1.0, "n_exercise": 8}
    sim = {"n_paths": n, "n_substeps": 2}

    rng = np.random.Generator(np.random.PCG64(TEST_SEED))   # ignored: normals are injected
    policy = fit_policy(simulate([SYN_PARAMS], [[1.0]], SYN_MARKET, contract, sim, rng, normals=set_a),
                        contract, SYN_MARKET, 3)
    result = evaluate_policy(policy, simulate([SYN_PARAMS], [[1.0]], SYN_MARKET, contract, sim, rng, normals=set_b),
                             contract, SYN_MARKET)
    bermudan, european = scalar_lsmc(set_a, set_b, SYN_PARAMS, SYN_R, 0.0, 1.0, 1.0, 8, 2, 3)

    difference = max(abs(result["bermudan"] / bermudan - 1), abs(result["european"] / european - 1))
    record_property("measured", f"relative difference {difference:.2e}")
    record_property("threshold", "<= 1e-10")
    assert difference <= 1e-10


@pytest.mark.fast
def test_V08_repeat_run_is_identical(tmp_path, record_property):
    first, second = tmp_path / "first", tmp_path / "second"
    assert run_price(first).returncode == 0
    assert run_price(second).returncode == 0

    def numbers(root):
        result = json.loads(next(root.glob("runs/*/result.json")).read_text())
        for row in result["per_seed"]:
            del row["wall_seconds"]
        return result

    assert numbers(first) == numbers(second)
    record_property("measured", "result.json identical apart from wall_seconds")
    record_property("threshold", "identical")


@pytest.mark.slow
def test_V09_basis_degree(record_property):
    params, market = reported_inputs()
    prices = {degree: pooled_price(params, market, CONTRACT, {**PRODUCTION, "basis_degree": degree}, PRODUCTION_SEEDS)
              for degree in (2, 3, 4)}
    first = abs(prices[2][0] - prices[3][0]) / prices[3][1]
    second = abs(prices[3][0] - prices[4][0]) / prices[4][1]
    record_property("measured", f"2 vs 3: {first:.2f} SE; 3 vs 4: {second:.2f} SE")
    record_property("threshold", "<= 2 SE of the higher-degree run")
    assert first <= 2 and second <= 2


@pytest.mark.slow
def test_V10_substeps(record_property):
    params, market = reported_inputs()
    v4, se4, _ = pooled_price(params, market, CONTRACT, {**PRODUCTION, "n_substeps": 4}, PRODUCTION_SEEDS)
    v8, se8, _ = pooled_price(params, market, CONTRACT, {**PRODUCTION, "n_substeps": 8}, PRODUCTION_SEEDS)
    z = abs(v4 - v8) / math.sqrt(se4**2 + se8**2)
    record_property("measured", f"{z:.2f} combined SE")
    record_property("threshold", "<= 3 sqrt(SE_4^2 + SE_8^2)")
    assert z <= 3


@pytest.mark.slow
def test_V11_production_seeds_agree(record_property):
    params, market = reported_inputs()
    _, _, results = pooled_price(params, market, CONTRACT, PRODUCTION, PRODUCTION_SEEDS)
    worst = 0.0
    for a, b in itertools.combinations(results, 2):
        worst = max(worst, abs(a["bermudan"] - b["bermudan"]) / math.sqrt(a["se_bermudan"] ** 2 + b["se_bermudan"] ** 2))
    record_property("measured", f"largest pair difference {worst:.2f} combined SE")
    record_property("threshold", "<= 3 sqrt(SE_a^2 + SE_b^2)")
    assert worst <= 3


@pytest.mark.fast
def test_V12_calibration_recovers_known_parameters(record_property):
    raw = surface.load_raw_iv(FIXTURES / "synthetic_raw_iv.csv")
    quotes, _ = surface.filter_quotes(raw, {"symbol": "SYN1", "snapshot_date": "2026-10-14", "r": SYN_R, "q": 0.0})
    grid = surface.select_grid(quotes, {"contract_strike": 1.0, "contract_maturity": 1.0})
    fit = calibrate_asset(grid, SYN_R, 0.0, {})
    errors = {name: abs(fit["params"][name] / SYN_PARAMS[name] - 1) for name in ("v0", "kappa", "theta", "xi")}
    rho_error = abs(fit["params"]["rho"] - SYN_PARAMS["rho"])
    record_property("measured", f"RMS {fit['rms_iv_residual']:.2e}; largest relative parameter error "
                                f"{max(errors.values()):.2e}; rho error {rho_error:.2e}")
    record_property("threshold", "RMS <= 1e-4; 5% relative; rho within 0.05")
    assert fit["rms_iv_residual"] <= 1e-4
    assert max(errors.values()) <= 0.05
    assert rho_error <= 0.05


def market_calibration_or_skip():
    path = ROOT / "results" / "calibration.json"
    if not path.exists():
        pytest.skip("market mode only: no results/calibration.json (owner-run; OD-1 unresolved)")
    document = json.loads(path.read_text())
    if document["data_mode"] != "market":
        pytest.skip("market mode only")
    return document


@pytest.mark.slow
def test_V13_market_calibration(record_property):
    entry = market_calibration_or_skip()["assets"]["AMD"]
    if entry["ladder_step"] == 3:
        pytest.fail("failure ladder step 3: no market-calibrated result")
    record_property("measured", f"RMS {entry['rms_iv_residual']:.4f}; stable {entry['stable']}; "
                                f"ladder step {entry['ladder_step']}")
    record_property("threshold", "RMS <= 0.01 and stable")
    assert entry["rms_iv_residual"] <= 0.01 and entry["stable"]


@pytest.mark.fast
def test_V14a_tree_inversion_round_trip(record_property):
    worst = 0.0
    for kind, strike, maturity, sigma in itertools.product(
            ["put", "call"], [0.85, 1.0, 1.15], [0.25, 1.0], [0.2, 0.5, 1.0]):
        price = crr_price(kind, 1.0, strike, maturity, sigma, SYN_R, 0.0, 500)
        worst = max(worst, abs(american_implied_vol(kind, price, 1.0, strike, maturity, SYN_R, 0.0) - sigma))
    record_property("measured", f"largest error {worst:.2e}")
    record_property("threshold", "<= 1e-4")
    assert worst <= 1e-4


@pytest.mark.slow
def test_V14bc_market_iv_conversion(record_property):
    checks = market_calibration_or_skip()["assets"]["AMD"].get("iv_checks")
    if not checks or checks["near_money_pairs"] == 0:
        pytest.skip("no near-the-money call/put pairs were available")
    fraction = checks["pairs_with_overlapping_bands"] / checks["near_money_pairs"]
    record_property("measured", f"{fraction:.3f} of pairs agree; median |own - vendor| "
                                f"{checks['median_abs_own_minus_vendor_iv']:.4f}")
    record_property("threshold", ">= 0.90 of pairs; median <= 0.02")
    assert fraction >= 0.90
    assert checks["median_abs_own_minus_vendor_iv"] <= 0.02


@pytest.mark.slow
def test_V15_output_completeness(record_property):
    reported = []
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        config = yaml.safe_load(path.read_text())
        if isinstance(config, dict) and config.get("reported") is True:
            reported.append(config["run_name"])

    found = {name: sorted((ROOT / "results" / "runs").glob(f"{name}-*")) for name in reported}
    if not any(found.values()):
        pytest.skip("no reported run has been completed yet")

    n_checked = 0
    for directories in found.values():
        for directory in directories:
            assert (directory / "manifest.json").exists(), f"incomplete run {directory.name}"
            manifest = json.loads((directory / "manifest.json").read_text())
            for output in manifest["outputs"]:
                digest = hashlib.sha256((directory / output["path"]).read_bytes()).hexdigest()
                assert digest == output["sha256"], f"hash mismatch in {directory.name}"
            json.loads((directory / "result.json").read_text(), parse_constant=pytest.fail)
            n_checked += 1
    record_property("measured", f"{n_checked} run(s) complete, finite and hashes match")
    record_property("threshold", "every configured run")


@pytest.mark.slow
def test_V16_governance_scan(record_property):
    history = subprocess.run(["git", "log", "--all", "--name-only", "--pretty=format:", "--", "data/"],
                             cwd=ROOT, capture_output=True, text=True).stdout.strip()
    assert history == "", "data/ paths found in Git history"
    assert (ROOT / ".git" / "hooks" / "pre-commit").exists(), "pre-commit hook is not installed"

    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    forbidden_keys = re.compile(r'"(bid|ask|mean_price|stock_price|stock_price_for_iv|log_return|spot)"\s*:')
    secret = re.compile(r"(api[_-]?key|secret[_-]?key|password)\s*[:=]\s*['\"]?\w{12,}|ghp_\w{20,}", re.IGNORECASE)
    n_scanned = 0
    for name in tracked:
        path = ROOT / name
        if not path.is_file() or name.startswith(("tests/", "prototype/")) or path.suffix in (".pyc", ".pdf", ".png"):
            continue
        text = path.read_text(errors="ignore")
        n_scanned += 1
        assert not secret.search(text), f"possible secret in {name}"
        if name.startswith("results/") and name.endswith(".json"):
            assert not forbidden_keys.search(text), f"prohibited field in {name}"
    record_property("measured", f"{n_scanned} tracked files scanned; no data/ paths in history; hook active")
    record_property("threshold", "no findings")


@pytest.mark.slow
def test_V17_clean_clone_smoke(tmp_path, record_property):
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
    if status.strip():
        pytest.skip("working tree has uncommitted changes; run V-17 from a committed state")
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(ROOT), str(clone)], check=True)
    begin = time.perf_counter()
    done = subprocess.run(["bash", "scripts/smoke.sh"], cwd=clone, env={"PYTHON": sys.executable, "PATH": "/usr/bin:/bin"},
                          capture_output=True, text=True)
    wall = time.perf_counter() - begin
    record_property("measured", f"exit {done.returncode}; {wall:.0f} s")
    record_property("threshold", "exit 0 within 600 s")
    assert done.returncode == 0
    assert wall <= 600


@pytest.mark.slow
def test_V18_traceability(tmp_path, record_property):
    evidence = ROOT / "results" / "evidence_map.csv"
    if not evidence.exists():
        pytest.skip("no evidence map yet: run make_figures.py after the reported runs")
    with open(evidence, newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        document = json.loads((ROOT / "results" / row["result_path"]).read_text())
        for part in row["json_pointer"].strip("/").split("/"):
            document = document[int(part)] if isinstance(document, list) else document[part]
    done = subprocess.run([sys.executable, "scripts/make_figures.py"], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0
    record_property("measured", f"{len(rows)} evidence rows resolve; figures regenerate")
    record_property("threshold", "every row resolves")


# ----------------------------------------------------------------------
# Runtime budgets
# ----------------------------------------------------------------------

@pytest.mark.slow
def test_B1_one_production_pricing(record_property):
    timings = []
    for _ in range(5):
        begin = time.perf_counter()
        price_bermudan([SYN_PARAMS], [[1.0]], SYN_MARKET, CONTRACT, {**PRODUCTION, "master_seed": 620001})
        timings.append(time.perf_counter() - begin)
    median = statistics.median(timings)
    record_property("measured", f"median {median:.2f} s over 5 repeats")
    record_property("threshold", "<= 30 s")
    assert median <= 30


@pytest.mark.slow
def test_B2_full_sweep(record_property):
    summaries = sorted((ROOT / "results" / "sweeps").glob("core_*production-*.json"), key=lambda p: p.stat().st_mtime)
    if not summaries:
        pytest.skip("no reported sweep has been completed yet")
    total = json.loads(summaries[-1].read_text())["wall_seconds_total"]
    record_property("measured", f"{total:.0f} s")
    record_property("threshold", "<= 21600 s")
    assert total <= 21600


# ----------------------------------------------------------------------
# Extensions: closed until their gates pass (SPEC.md 2.2 and 17)
# ----------------------------------------------------------------------

EXT_A_REASON = "EXT-A has not started: GOOG stays single-asset until NVDA, AMD and INTC are calibrated (OD-11)"


@pytest.mark.slow
def test_V20_driver_correlations():
    pytest.skip(EXT_A_REASON)


@pytest.mark.slow
def test_V21_identical_assets():
    pytest.skip(EXT_A_REASON)


@pytest.mark.slow
def test_V22_orderings():
    pytest.skip(EXT_A_REASON)


@pytest.mark.slow
def test_V23_three_asset_stability():
    pytest.skip(EXT_A_REASON)


@pytest.mark.slow
def test_V24_three_asset_calibration():
    pytest.skip(EXT_A_REASON)


@pytest.mark.slow
def test_V30_cpp_engine():
    pytest.skip("EXT-B is closed: the C++ gate in SPEC.md 9.4 has not passed")
