"""
The pipeline commands (SPEC.md sections 11 and 12).

    python scripts/run.py calibrate --config configs/<name>.yaml [--results-root <dir>]
    python scripts/run.py price     --config configs/<name>.yaml [--results-root <dir>]
    python scripts/run.py sweep     --config configs/<name>.yaml [--results-root <dir>]
    python scripts/run.py compare   --config configs/<name>.yaml [--results-root <dir>]
    python scripts/run.py grid      --config configs/<name>.yaml [--workers N] [--output <file>]

Output never contains a market price, quote or row of Restricted data.
"""

import argparse
import copy
import datetime
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import numpy as np
import pandas as pd
import scipy
import yaml

from hestonpricer import market as market_pricing
from hestonpricer import surface
from hestonpricer.analytic import heston_european
from hestonpricer.calibrate import calibrate_with_ladder
from hestonpricer.errors import EXIT_CODES, HestonError
from hestonpricer.iv import bs_implied_vol
from hestonpricer.lsmc import price_bermudan
from hestonpricer.simulate import simulate
from hestonpricer.tree import crr_price

SPEC_VERSION = "1.13"
DATA_CONTRACT_VERSION = "0.10"
RAW_IV_MARKET = ROOT / "data" / "nvda_amd_intc_IV_Oct13-15.csv"
RAW_IV_SYNTHETIC = ROOT / "tests" / "fixtures" / "synthetic_raw_iv.csv"
GRID_FILE = ROOT / "data" / "derived" / "calibration_grid.parquet"
COMPARISON_SEED = 620100
V04_CASES = [(0.3, 1.0), (0.5, 1.0), (0.5, 0.9), (0.3, 1.1)]     # (sigma, K), SPEC.md 16
V04_CALL_Q = 0.0026

CONFIG_KEYS = {"schema_version", "run_name", "reported", "data_mode", "assets", "snapshot_date",
               "market_inputs", "params_source", "contract", "simulation", "seeds"}
OPTIONAL_KEYS = {"raw_iv_file",           # the licensed iVolatility file under data/ to calibrate to
                 "grid"}                  # strike x maturity cases for `run.py grid`
CONTRACT_KEYS = {"strike", "maturity", "n_exercise", "kind"}
SIMULATION_KEYS = {"n_paths", "n_substeps", "basis_degree"}

# The sweeps of SPEC.md section 9.3. "M" keeps the time step fixed at 1/200.
SWEEPS = {
    "n_paths": [20_000, 50_000, 100_000, 200_000],
    "n_substeps": [1, 2, 4, 8],
    "M": [10, 25, 50, 100],
    "basis_degree": [2, 3, 4],
    "r_shift": [0.0, 0.01, -0.01],
}

# Tracebacks can echo row values, so they are shown only once we know the run
# does not touch licensed data.
state = {"show_traceback": False}


# ----------------------------------------------------------------------
# Configuration and inputs
# ----------------------------------------------------------------------

def load_config(path):
    """Read a run configuration. Unknown or missing keys raise E_CONFIG."""
    try:
        cfg = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as error:
        raise HestonError("E_CONFIG", f"cannot read {path}: {type(error).__name__}") from None

    unknown = set(cfg) - CONFIG_KEYS - OPTIONAL_KEYS
    missing = CONFIG_KEYS - set(cfg)
    if unknown or missing:
        raise HestonError("E_CONFIG", f"unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    if set(cfg["contract"]) - CONTRACT_KEYS or set(cfg["simulation"]) - SIMULATION_KEYS:
        raise HestonError("E_CONFIG", "unknown key inside contract or simulation")
    if cfg["data_mode"] not in ("synthetic", "published", "market"):
        raise HestonError("E_CONFIG", "data_mode must be synthetic, published or market")
    if len(cfg["assets"]) != 1:
        raise HestonError("E_CONFIG", "contracts on more than one asset are out of scope (SPEC.md 2.3)")
    if cfg["contract"].get("kind", "put") not in ("put", "call"):
        raise HestonError("E_CONFIG", "contract kind must be put or call")

    cfg["contract"] = {"kind": "put", **cfg["contract"]}
    cfg["snapshot_date"] = str(cfg["snapshot_date"])
    state["show_traceback"] = cfg["data_mode"] != "market"
    return cfg


def check_governance(cfg):
    """Market runs stop here, before anything under data/ is read (SPEC.md 5.2)."""
    if cfg["data_mode"] != "market":
        return
    path = ROOT / "configs" / "governance.yaml"
    authorized = False
    if path.exists():
        authorized = yaml.safe_load(path.read_text()).get("market_use_authorized") is True
    if not authorized:
        raise HestonError("E_GOV_UNAUTHORIZED", "market_use_authorized is not true in configs/governance.yaml")


def load_rate(cfg, date=None):
    """r = 2 ln(1 + y / 2), from the 1-year Treasury par yield (SPEC.md 4.3)."""
    inputs = yaml.safe_load((ROOT / cfg["market_inputs"]).read_text())
    yields = {str(k): v for k, v in inputs["treasury_1y_cmt_yield"].items()}
    y = yields.get(date or cfg["snapshot_date"])
    if y is None:
        raise HestonError("E_CONFIG", f"no Treasury yield entered for {date or cfg['snapshot_date']} (OD-3)")
    return 2.0 * math.log(1.0 + float(y) / 2.0)


def load_dividend(cfg, asset):
    inputs = yaml.safe_load((ROOT / cfg["market_inputs"]).read_text())
    q = inputs.get("dividend_yield", {}).get(asset)
    if q is None:
        raise HestonError("E_CONFIG", f"no dividend yield entered for {asset}")
    return float(q)


def load_params(cfg, results_root):
    """
    Returns (params, q, inputs, status) for the run's single asset, according to its data mode.
    status is "accepted", "rejected", "published" or "synthetic" (SPEC.md 12.4). A rejected fit
    (ladder step 3) is used and labelled; at step 4 there is nothing to price (SPEC.md 6.7).
    """
    asset = cfg["assets"][0]
    source = ROOT / cfg["params_source"]

    if cfg["data_mode"] == "synthetic":
        params = yaml.safe_load(source.read_text())["assets"][asset]
        inputs = [{"dataset": "synthetic_params", "tickers": [asset], "file_name": source.name}]
        return params, load_dividend(cfg, asset), inputs, "synthetic"

    if cfg["data_mode"] == "published":
        entry = (yaml.safe_load(source.read_text()).get("sets") or {}).get(asset)
        names = ("v0", "kappa", "theta", "xi", "rho", "q", "citation", "table_or_page")
        if entry is None or any(entry.get(name) is None for name in names):
            raise HestonError("E_CONFIG", f"published set '{asset}' is not filled in (OD-5)")
        params = {name: float(entry[name]) for name in ("v0", "kappa", "theta", "xi", "rho")}
        inputs = [{"dataset": "published_params", "tickers": [asset], "file_name": source.name,
                   "citation": entry["citation"], "table_or_page": entry["table_or_page"]}]
        # A run's market_inputs file may set the dividend yield for this asset
        dividends = yaml.safe_load((ROOT / cfg["market_inputs"]).read_text()).get("dividend_yield", {})
        return params, float(dividends.get(asset, entry["q"])), inputs, "published"

    # Market mode: the fitted parameters in the Internal calibration summary named by params_source.
    # A redirected results root (tests, V-08) holds its own copy under the same file name.
    if Path(results_root).resolve() == (ROOT / "results").resolve():
        calibration_file = ROOT / cfg["params_source"]
    else:
        calibration_file = Path(results_root) / Path(cfg["params_source"]).name
    if not calibration_file.exists():
        raise HestonError("E_CONFIG", f"run `calibrate` first: {calibration_file.name} not found")
    calibration = json.loads(calibration_file.read_text())
    entry = calibration["assets"][asset]
    if entry["ladder_step"] == 3:
        return entry["rejected_fit"]["params"], entry["q"], calibration["inputs"], "rejected"
    if entry["ladder_step"] not in (1, 2):
        raise HestonError("E_CALIB_FAILED", "no market-calibrated result; only published mode is reported (SPEC.md 6.7)")
    return entry["params"], entry["q"], calibration["inputs"], "accepted"


def rejected_label(results_root, cfg):
    """The REJECTED label every result of a step-3 fit carries (SPEC.md 2.4)."""
    if Path(results_root).resolve() == (ROOT / "results").resolve():
        path = ROOT / cfg["params_source"]
    else:
        path = Path(results_root) / Path(cfg["params_source"]).name
    rejected = json.loads(path.read_text())["assets"][cfg["assets"][0]]["rejected_fit"]
    return {"label": "REJECTED", "rms_iv_residual": rejected["rms_iv_residual"], "limit": rejected["limit"],
            "from_step": rejected["from_step"]}


# ----------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------

def git_output(*arguments):
    result = subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, text=True)
    # rstrip only: `git status --porcelain` lines can begin with a meaningful space
    return result.stdout.rstrip() if result.returncode == 0 else ""


def git_commit():
    return git_output("rev-parse", "HEAD") or "nocommit"


def git_dirty():
    """True if anything outside results/ has changed."""
    lines = git_output("status", "--porcelain").splitlines()
    return any(not line[3:].strip('"').startswith("results/") for line in lines)


def environment():
    ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    return {"python": platform.python_version(),
            "packages": {"numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__},
            "os": platform.platform(), "cpu": platform.processor() or platform.machine(),
            "cores": os.cpu_count(), "ram_gb": round(ram_gb, 1)}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def assert_all_finite(value):
    """Raise E_NONFINITE if any number in a nested structure is NaN or infinite."""
    if isinstance(value, dict):
        for item in value.values():
            assert_all_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_all_finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise HestonError("E_NONFINITE", "a reported output is NaN or infinite")


# ----------------------------------------------------------------------
# price
# ----------------------------------------------------------------------

def pool_results(per_seed, params, r, q, contract):
    """Combine the seeds (SPEC.md section 3) and add the two baselines (SPEC.md 8.3)."""
    n_seeds = len(per_seed)

    def pooled_mean(name):
        return float(np.mean([row[name] for row in per_seed]))

    def pooled_se(name):
        return float(np.sqrt(np.sum([row[name] ** 2 for row in per_seed])) / n_seeds)

    K, T, M = float(contract["strike"]), float(contract["maturity"]), int(contract["n_exercise"])
    kind = contract.get("kind", "put")

    # European baseline. heston_european needs xi > 0, so nothing is reported at xi = 0.
    european_analytic = heston_european(kind, 1.0, K, T, params, r, q)

    # Constant-volatility Bermudan at the European-matching volatility
    sigma = bs_implied_vol(kind, european_analytic, 1.0, K, T, r, q)
    steps_per_date = 50
    baseline = crr_price(kind, 1.0, K, T, sigma, r, q, steps_per_date * M,
                         exercise_steps=[steps_per_date * k for k in range(1, M + 1)])

    bermudan, se_bermudan = pooled_mean("bermudan"), pooled_se("se_bermudan")
    premium, se_premium = pooled_mean("premium"), pooled_se("se_premium")

    # A relative target is undefined at zero, so it is reported as unmet (SPEC.md 14.3)
    t1 = bermudan > 0 and se_bermudan > 0 and se_bermudan / bermudan <= 0.005
    t2 = premium > 0 and se_premium > 0 and se_premium / premium <= 0.10

    pooled = {
        "bermudan": bermudan, "se_bermudan": se_bermudan,
        "european_mc": pooled_mean("european"), "se_european": pooled_se("se_european"),
        "european_analytic": european_analytic,
        "baseline_const_vol_bermudan": baseline,
        "premium": premium, "se_premium": se_premium,
        "premium_ci95": [premium - 1.96 * se_premium, premium + 1.96 * se_premium],
        "premium_resolved": bool(t2),
    }
    return pooled, {"T1": bool(t1), "T2": bool(t2)}


def price_run(cfg, config_source, results_root, command, r_shift=0.0, sweep_info=None):
    """Run one pricing over all seeds and write its run directory. Returns the run_id."""
    check_governance(cfg)
    if cfg["reported"] and git_dirty():
        raise HestonError("E_DIRTY_TREE", "commit your changes before a reported run")

    commit = git_commit()
    run_id = f"{cfg['run_name']}-{commit[:8]}"
    run_dir = Path(results_root) / "runs" / run_id
    if (run_dir / "manifest.json").exists():
        raise HestonError("E_RUN_EXISTS", f"{run_id} is complete; use --results-root for a new location")

    started = now()
    begin = time.perf_counter()
    params, q, inputs, status = load_params(cfg, results_root)
    r = load_rate(cfg) + r_shift
    market = {"r": r, "q": [q]}
    contract = cfg["contract"]
    simulation = cfg["simulation"]

    run_dir.mkdir(parents=True, exist_ok=True)
    log_lines = [f"run_id {run_id}", f"data_mode {cfg['data_mode']}", f"seeds {cfg['seeds']}"]

    per_seed = []
    outcomes = []
    for seed in cfg["seeds"]:
        seed_begin = time.perf_counter()
        sim = {**simulation, "master_seed": seed}
        outcome = price_bermudan([params], [[1.0]], market, contract, sim)
        wall = time.perf_counter() - seed_begin
        outcomes.append(outcome)
        per_seed.append({"seed": seed, **{name: outcome[name] for name in (
            "bermudan", "se_bermudan", "european", "se_european", "premium", "se_premium",
            "in_sample_bermudan")}, "wall_seconds": wall})
        log_lines.append(f"seed {seed} finished in {wall:.2f} s")

    pooled, precision = pool_results(per_seed, params, r, q, contract)
    feller = 2 * params["kappa"] * params["theta"] / params["xi"] ** 2

    result = {
        "schema_version": 1, "run_id": run_id, "data_mode": cfg["data_mode"],
        "parameter_status": status, "units": "fraction_of_initial_spot", "assets": cfg["assets"],
        "contract": contract, "simulation": simulation,
        "per_seed": per_seed, "pooled": pooled,
        "diagnostics": {
            "exercise_fraction_by_date": np.mean([o["exercise_fraction"] for o in outcomes], axis=0).tolist(),
            "min_itm_count": min(o["min_itm_count"] for o in outcomes),
            "skipped_dates": sum(o["skipped_dates"] for o in outcomes),
            "rank_deficient_dates": sum(o["rank_deficient_dates"] for o in outcomes),
            "feller_ratio": feller,
        },
        "precision": precision,
    }
    if sweep_info:
        result["sweep"] = sweep_info
    if status == "rejected":
        result["rejected_fit"] = rejected_label(results_root, cfg)
    assert_all_finite(result)  # no result.json if any value is NaN or infinite

    # The manifest is written last: a directory without one is an incomplete run.
    if isinstance(config_source, dict):
        (run_dir / "config.yaml").write_text(yaml.safe_dump(config_source, sort_keys=False))
    else:
        shutil.copyfile(config_source, run_dir / "config.yaml")
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    (run_dir / "run.log").write_text("\n".join(log_lines) + "\n")

    manifest = {
        "schema_version": 1, "spec_version": SPEC_VERSION, "data_contract_version": DATA_CONTRACT_VERSION,
        "run_id": run_id, "git_commit": commit, "git_dirty": git_dirty(), "command": command,
        "config_sha256": sha256(run_dir / "config.yaml"), "data_mode": cfg["data_mode"],
        "seeds": cfg["seeds"], "environment": environment(), "inputs": inputs,
        "outputs": [{"path": name, "sha256": sha256(run_dir / name)}
                    for name in ("result.json", "config.yaml", "run.log")],
        "started_at": started, "finished_at": now(), "wall_seconds": time.perf_counter() - begin,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    label = " REJECTED FIT" if status == "rejected" else ""
    print(f"{run_id}: bermudan {pooled['bermudan']:.6f} +/- {pooled['se_bermudan']:.6f} "
          f"(fraction of spot, mode {cfg['data_mode']}, parameters {status}{label})")
    return run_id


# ----------------------------------------------------------------------
# calibrate
# ----------------------------------------------------------------------

def calibrate_date(cfg, raw, date, fallback_kappa, inputs):
    """Filter, select the grid and fit the single asset on one date."""
    asset = cfg["assets"][0]
    r = load_rate(cfg, date)
    q = load_dividend(cfg, asset)
    contract = cfg["contract"]

    quotes, counts = surface.filter_quotes(raw, {"symbol": asset, "snapshot_date": date, "r": r, "q": q})
    filter_drops = {f"F{i}": counts[f"F{i}"] for i in range(1, 9)}
    entry = {"q": q, "filter_drops": filter_drops, "mid_mismatch_rows": counts["mid_mismatch"]}
    grid = None

    try:
        grid = surface.select_grid(quotes, {"contract_strike": contract["strike"],
                                            "contract_maturity": contract["maturity"]})
    except HestonError as error:
        if error.code != "E_SPARSE_GRID":
            raise
        entry.update({"ladder_step": 4, "params": None, "stable": False, "fixed_params": [],
                      "rejected_fit": None, "failure": error.code})

    if grid is not None:
        entry.update(calibrate_with_ladder(grid, r, q, fallback_kappa))
        if cfg["data_mode"] == "market" and len(quotes):
            config = {"symbol": asset, "snapshot_date": date, "r": r, "q": q}
            entry["iv_checks"] = surface.iv_conversion_checks(raw, quotes, config)

    document = {"schema_version": 1, "data_mode": cfg["data_mode"], "snapshot_date": date,
                "r": r, "inputs": inputs, "assets": {asset: entry}}
    return document, grid


def run_calibrate(cfg, results_root, root_given):
    check_governance(cfg)
    if cfg["data_mode"] == "published":
        raise HestonError("E_CONFIG", "published mode has nothing to calibrate")
    if cfg["reported"] and git_dirty():
        raise HestonError("E_DIRTY_TREE", "commit your changes before a reported run")

    market = cfg["data_mode"] == "market"
    raw_path = RAW_IV_MARKET if market else RAW_IV_SYNTHETIC
    if market and cfg.get("raw_iv_file"):
        raw_path = ROOT / cfg["raw_iv_file"]
    raw = surface.load_raw_iv(raw_path)
    inputs = [{"dataset": "ivolatility_raw_iv" if market else "synthetic_raw_iv",
               "tickers": cfg["assets"], "snapshot_date": cfg["snapshot_date"],
               "file_name": raw_path.name, "row_count": int(len(raw))}]

    fallback_kappa = None
    published = ROOT / "configs" / "published_params.yaml"
    if published.exists():
        fallback = yaml.safe_load(published.read_text()).get("kappa_fallback") or {}
        fallback_kappa = fallback.get("value")

    # Without --results-root, a market fit goes where params_source says (for GOOG,
    # results/calibration_goog/), and synthetic output is kept apart in results/smoke/
    if not root_given:
        results_root = ROOT / Path(cfg["params_source"]).parent if market else ROOT / "results" / "smoke"
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    document, grid = calibrate_date(cfg, raw, cfg["snapshot_date"], fallback_kappa, inputs)
    assert_all_finite(document)
    (results_root / "calibration.json").write_text(json.dumps(document, indent=2))

    # The Restricted grid is written only in market mode, only under data/, and only for
    # the default results location, so a redirected run cannot overwrite the reported grid
    if market and grid is not None and not root_given:
        GRID_FILE.parent.mkdir(parents=True, exist_ok=True)
        grid[["symbol", "date", "strike", "expiration", "call_put", "iv"]].to_parquet(GRID_FILE)

    entry = document["assets"][cfg["assets"][0]]
    print(f"calibration ({cfg['data_mode']}): ladder step {entry['ladder_step']}, "
          f"stable {entry['stable']}, RMS IV residual {entry.get('rms_iv_residual')}")


# ----------------------------------------------------------------------
# sweep
# ----------------------------------------------------------------------

def sweep_variants(cfg):
    """One (factor, value, config, r_shift) per sweep setting (SPEC.md 9.3)."""
    T = float(cfg["contract"]["maturity"])
    variants = []
    for factor, values in SWEEPS.items():
        for value in values:
            variant = copy.deepcopy(cfg)
            r_shift = 0.0
            if factor == "M":
                variant["contract"]["n_exercise"] = value
                variant["simulation"]["n_substeps"] = int(round(200 * T / value))
            elif factor == "r_shift":
                r_shift = value
            else:
                variant["simulation"][factor] = value
            variant["run_name"] = f"{cfg['run_name']}__{factor}_{value}"
            variants.append((factor, value, variant, r_shift))
    return variants


def run_sweep(cfg, results_root, command):
    begin = time.perf_counter()
    rows = []
    commit = git_commit()
    for factor, value, variant, r_shift in sweep_variants(cfg):
        run_id = f"{variant['run_name']}-{commit[:8]}"
        run_dir = Path(results_root) / "runs" / run_id
        if not (run_dir / "manifest.json").exists():     # finished variants are kept, not redone
            price_run(variant, variant, results_root, command, r_shift, {"factor": factor, "value": value})
        result = json.loads((run_dir / "result.json").read_text())
        pooled = result["pooled"]
        wall = json.loads((run_dir / "manifest.json").read_text())["wall_seconds"]
        rows.append({"factor": factor, "value": value, "run_id": run_id, "wall_seconds": wall,
                     "bermudan": pooled["bermudan"], "se_bermudan": pooled["se_bermudan"],
                     "premium": pooled["premium"], "se_premium": pooled["se_premium"]})

    summary = {"schema_version": 1, "run_name": cfg["run_name"], "data_mode": cfg["data_mode"],
               "parameter_status": result["parameter_status"], "rejected_fit": result.get("rejected_fit"),
               "commit": commit, "wall_seconds_this_invocation": time.perf_counter() - begin,
               "wall_seconds_total": sum(row["wall_seconds"] for row in rows), "budget_b2_seconds": 21600,
               "rows": rows}
    out = Path(results_root) / "sweeps"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{cfg['run_name']}-{commit[:8]}.json").write_text(json.dumps(summary, indent=2))
    print(f"sweep finished: {len(rows)} configurations in {summary['wall_seconds_total']:.0f} s")


# ----------------------------------------------------------------------
# compare
# ----------------------------------------------------------------------

def run_compare(cfg, results_root, command):
    """Numerical against analytical prices, for the figures in SPEC.md 12.9."""
    check_governance(cfg)
    if cfg["reported"] and git_dirty():
        raise HestonError("E_DIRTY_TREE", "commit your changes before a reported run")

    commit = git_commit()
    out = Path(results_root) / "comparisons" / f"{cfg['run_name']}-{commit[:8]}.json"
    if out.exists():
        raise HestonError("E_RUN_EXISTS", f"{out.name} exists; use --results-root for a new location")

    params, q, inputs, status = load_params(cfg, results_root)
    r = load_rate(cfg)
    market = {"r": r, "q": [q]}
    n_paths = int(cfg["simulation"]["n_paths"])
    base = cfg["contract"]

    # 1. Monte Carlo European against semi-analytic, on the V-03 grid
    mc_rows = []
    for maturity in [0.25, 0.5, 1.0, 1.5, 2.0]:
        contract = {"strike": 1.0, "maturity": maturity, "n_exercise": 1}
        sim = {"n_paths": 2 * n_paths, "n_substeps": int(round(maturity * 400))}
        rng = np.random.Generator(np.random.PCG64(COMPARISON_SEED))
        terminal = simulate([params], [[1.0]], market, contract, sim, rng)["S"][:, 1, 0]
        for strike in np.linspace(0.80, 1.25, 8):
            payoff = math.exp(-r * maturity) * np.maximum(strike - terminal, 0.0)
            mc, se = float(payoff.mean()), float(payoff.std(ddof=1) / math.sqrt(len(payoff)))
            analytic = heston_european("put", 1.0, strike, maturity, params, r, q)
            mc_rows.append({"maturity": maturity, "strike": float(strike), "mc": mc, "se": se,
                            "analytic": analytic, "z": (mc - analytic) / max(se, 1e-12)})

    # 2. LSMC against the CRR tree, on the V-04 cases (a test model, not the reported parameters).
    #    Puts use q = 0; calls use q = 0.0026, so early exercise of a call can matter.
    tree_rows = []
    for kind, q_case in (("put", 0.0), ("call", V04_CALL_Q)):
        for sigma, strike in V04_CASES:
            gbm = {"v0": sigma**2, "kappa": 1.0, "theta": sigma**2, "xi": 0.0, "rho": 0.0}
            contract = {"strike": strike, "maturity": 1.0, "n_exercise": 50, "kind": kind}
            outcome = price_bermudan([gbm], [[1.0]], {"r": r, "q": [q_case]}, contract,
                                     {**cfg["simulation"], "master_seed": COMPARISON_SEED})
            tree = crr_price(kind, 1.0, strike, 1.0, sigma, r, q_case, 5000,
                             exercise_steps=[100 * k for k in range(1, 51)])
            tree_rows.append({"kind": kind, "q": q_case, "sigma": sigma, "strike": strike,
                              "lsmc": outcome["bermudan"], "se": outcome["se_bermudan"], "tree": tree})

    # 3. Bermudan, European and constant-volatility Bermudan across strikes
    profile = []
    for strike in [0.80, 0.90, 1.00, 1.10, 1.20]:
        contract = {**base, "strike": strike}
        per_seed = [price_bermudan([params], [[1.0]], market, contract,
                                   {**cfg["simulation"], "master_seed": seed}) for seed in cfg["seeds"]]
        pooled, _ = pool_results(per_seed, params, r, q, contract)
        profile.append({"strike": strike, **pooled})

    document = {"schema_version": 1, "run_name": cfg["run_name"], "data_mode": cfg["data_mode"],
                "commit": commit, "command": command, "units": "fraction_of_initial_spot",
                "parameter_status": status,
                "rejected_fit": rejected_label(results_root, cfg) if status == "rejected" else None,
                "r": r, "q": q, "params": params, "inputs": inputs, "comparison_seed": COMPARISON_SEED,
                "seeds": cfg["seeds"], "n_paths": n_paths,
                "mc_vs_analytic": mc_rows, "lsmc_vs_tree": tree_rows, "strike_profile": profile}
    assert_all_finite(document)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2))
    print(f"comparison written: {out.name}")


# ----------------------------------------------------------------------
# grid
# ----------------------------------------------------------------------

def finite_or_none(value):
    """JSON has no NaN, so a missing number is written as null."""
    return None if isinstance(value, float) and not math.isfinite(value) else value


def run_grid(cfg, workers, output):
    """
    Price a strike x maturity grid (config key `grid`) with the run's parameters, and check each
    contract's behaviour (SPEC.md 8.6): the Monte Carlo European price agrees with the semi-analytic
    price, and the Bermudan price is not below the European price or intrinsic value, each within 3 SE.
    These are not market comparisons: no market prices are used.
    """
    if "grid" not in cfg:
        raise HestonError("E_CONFIG", "a grid run needs a `grid` block with `cases`")
    check_governance(cfg)
    params, q, _, status = load_params(cfg, ROOT / "results")
    r = load_rate(cfg)
    kind = cfg["contract"]["kind"]
    n_exercise = int(cfg["contract"]["n_exercise"])
    n_paths = int(cfg["simulation"]["n_paths"])

    contracts = market_pricing.grid_contracts(cfg["grid"]["cases"], kind)
    priced = market_pricing.price_contracts(params, contracts, r, q, n_paths, cfg["seeds"], workers, n_exercise)

    rows = []
    for row in priced.itertuples():
        analytic = heston_european(kind, 1.0, row.strike, row.tenor, params, r, q)
        intrinsic = max(row.strike - 1.0, 0.0) if kind == "put" else max(1.0 - row.strike, 0.0)
        rows.append({
            "maturity": row.tenor, "strike_over_spot": row.strike,
            "bermudan": row.V, "se_bermudan": row.se, "european_mc": row.european, "se_european": row.se_european,
            "european_analytic": analytic, "premium": row.premium, "se_premium": row.se_premium,
            "model_implied_vol": finite_or_none(bs_implied_vol(kind, analytic, 1.0, row.strike, row.tenor, r, q)),
            "checks": {
                "european_within_3_se": bool(abs(row.european - analytic) <= 3 * max(row.se_european, 1e-12)),
                "bermudan_not_below_european": bool(row.premium >= -3 * max(row.se_premium, 1e-12)),
                "bermudan_not_below_intrinsic": bool(row.V >= intrinsic - 3 * max(row.se, 1e-12)),
            },
        })

    document = {"schema_version": 1, "data_mode": cfg["data_mode"], "asset": cfg["assets"][0],
                "note": cfg["grid"].get("note", "Model behaviour on a strike x maturity grid; no market prices."),
                "r": r, "q": q, "n_paths": n_paths, "seeds": cfg["seeds"], "n_exercise": n_exercise,
                "units": "fraction_of_initial_spot", "rows": rows}
    if status == "rejected":
        document["rejected_fit"] = rejected_label(ROOT / "results", cfg)
    output = Path(output or ROOT / cfg["grid"].get("output", f"results/{cfg['run_name']}.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2))
    passed = sum(all(row["checks"].values()) for row in rows)
    print(f"{passed} of {len(rows)} contracts pass all behaviour checks; written to {output}")


# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Heston LSMC pipeline")
    parser.add_argument("command", choices=["calibrate", "price", "sweep", "compare", "grid"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--workers", type=int, default=1, help="grid only: price maturities in parallel")
    parser.add_argument("--output", default=None, help="grid only: output file (default: the config's grid.output)")
    args = parser.parse_args()

    command = "python scripts/run.py " + " ".join(sys.argv[1:])
    results_root = Path(args.results_root) if args.results_root else ROOT / "results"
    try:
        cfg = load_config(args.config)
        if args.command == "calibrate":
            run_calibrate(cfg, results_root, args.results_root is not None)
        elif args.command == "price":
            price_run(cfg, args.config, results_root, command)
        elif args.command == "compare":
            run_compare(cfg, results_root, command)
        elif args.command == "grid":
            run_grid(cfg, args.workers, args.output)
        else:
            run_sweep(cfg, results_root, command)
    except HestonError as error:
        print(str(error), file=sys.stderr)
        return EXIT_CODES.get(error.code, 1)
    except Exception as error:
        # Sanitized: on licensed data only the error type is shown (SPEC.md 14.2)
        if state["show_traceback"]:
            traceback.print_exc()
        print(f"internal error: {type(error).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
