"""
The pipeline commands (SPEC.md sections 11 and 12).

    python scripts/run.py calibrate --config configs/<name>.yaml [--results-root <dir>]
    python scripts/run.py price     --config configs/<name>.yaml [--results-root <dir>]
    python scripts/run.py sweep     --config configs/<name>.yaml [--results-root <dir>]

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

from hestonpricer import surface
from hestonpricer.analytic import heston_european
from hestonpricer.calibrate import calibrate_with_ladder
from hestonpricer.errors import EXIT_CODES, HestonError
from hestonpricer.iv import bs_implied_vol
from hestonpricer.lsmc import price_bermudan
from hestonpricer.tree import crr_price

SPEC_VERSION = "1.3"
DATA_CONTRACT_VERSION = "0.4"
RAW_IV_MARKET = ROOT / "data" / "nvda_amd_intc_IV_Oct13-15.csv"
RAW_IV_SYNTHETIC = ROOT / "tests" / "fixtures" / "synthetic_raw_iv.csv"
GRID_FILE = ROOT / "data" / "derived" / "calibration_grid.parquet"
ALL_DATES = ["2026-10-13", "2026-10-14", "2026-10-15"]

CONFIG_KEYS = {"schema_version", "run_name", "tier", "reported", "data_mode", "assets", "snapshot_date",
               "market_inputs", "params_source", "contract", "simulation", "seeds"}
CONTRACT_KEYS = {"strike", "maturity", "n_exercise"}
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

    unknown = set(cfg) - CONFIG_KEYS
    missing = CONFIG_KEYS - set(cfg)
    if unknown or missing:
        raise HestonError("E_CONFIG", f"unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    if set(cfg["contract"]) - CONTRACT_KEYS or set(cfg["simulation"]) - SIMULATION_KEYS:
        raise HestonError("E_CONFIG", "unknown key inside contract or simulation")
    if cfg["data_mode"] not in ("synthetic", "published", "market"):
        raise HestonError("E_CONFIG", "data_mode must be synthetic, published or market")
    if cfg["tier"] != "CORE" or len(cfg["assets"]) != 1:
        raise HestonError("E_CONFIG", "only the single-asset CORE tier is open (SPEC.md 2.2)")

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
    """Returns (params, q, inputs) for the run's single asset, according to its data mode."""
    asset = cfg["assets"][0]
    source = ROOT / cfg["params_source"]

    if cfg["data_mode"] == "synthetic":
        params = yaml.safe_load(source.read_text())["assets"][asset]
        inputs = [{"dataset": "synthetic_params", "tickers": [asset], "file_name": source.name}]
        return params, load_dividend(cfg, asset), inputs

    if cfg["data_mode"] == "published":
        entry = (yaml.safe_load(source.read_text()).get("sets") or {}).get(asset)
        names = ("v0", "kappa", "theta", "xi", "rho", "q", "citation", "table_or_page")
        if entry is None or any(entry.get(name) is None for name in names):
            raise HestonError("E_CONFIG", f"published set '{asset}' is not filled in (OD-5)")
        params = {name: float(entry[name]) for name in ("v0", "kappa", "theta", "xi", "rho")}
        inputs = [{"dataset": "published_params", "tickers": [asset], "file_name": source.name,
                   "citation": entry["citation"], "table_or_page": entry["table_or_page"]}]
        return params, float(entry["q"]), inputs

    # Market mode: the fitted parameters in the Internal calibration summary
    calibration_file = Path(results_root) / Path(cfg["params_source"]).name
    if not calibration_file.exists():
        raise HestonError("E_CONFIG", "run `calibrate` first: calibration.json not found")
    calibration = json.loads(calibration_file.read_text())
    entry = calibration["assets"][asset]
    if entry["ladder_step"] == 3:
        raise HestonError("E_CALIB_FAILED", "no market-calibrated result; use published mode (SPEC.md 6.7)")
    return entry["params"], entry["q"], calibration["inputs"]


# ----------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------

def git_output(*arguments):
    result = subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


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

    # European baseline. heston_european needs xi > 0, so nothing is reported at xi = 0.
    european_analytic = heston_european("put", 1.0, K, T, params, r, q)

    # Constant-volatility Bermudan at the European-matching volatility
    sigma = bs_implied_vol("put", european_analytic, 1.0, K, T, r, q)
    steps_per_date = 50
    baseline = crr_price("put", 1.0, K, T, sigma, r, q, steps_per_date * M,
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
    params, q, inputs = load_params(cfg, results_root)
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
        "schema_version": 1, "run_id": run_id, "tier": cfg["tier"], "data_mode": cfg["data_mode"],
        "units": "fraction_of_initial_spot", "assets": cfg["assets"],
        "contract": contract, "simulation": simulation,
        "per_seed": per_seed, "pooled": pooled,
        "diagnostics": {
            "exercise_fraction_by_date": np.mean([o["exercise_fraction"] for o in outcomes], axis=0).tolist(),
            "min_itm_count": min(o["min_itm_count"] for o in outcomes),
            "skipped_dates": sum(o["skipped_dates"] for o in outcomes),
            "rank_deficient_dates": sum(o["rank_deficient_dates"] for o in outcomes),
            "feller_ratio": [feller],
        },
        "precision": precision,
    }
    if sweep_info:
        result["sweep"] = sweep_info
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

    print(f"{run_id}: bermudan {pooled['bermudan']:.6f} +/- {pooled['se_bermudan']:.6f} "
          f"(fraction of spot, mode {cfg['data_mode']})")
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
        entry.update({"ladder_step": 3, "params": None, "stable": False, "fixed_params": [],
                      "failure": error.code})

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
    raw = surface.load_raw_iv(raw_path)
    inputs = [{"dataset": "ivolatility_raw_iv" if market else "synthetic_raw_iv",
               "tickers": cfg["assets"], "snapshot_date": cfg["snapshot_date"],
               "file_name": raw_path.name, "row_count": int(len(raw))}]

    fallback_kappa = None
    published = ROOT / "configs" / "published_params.yaml"
    if published.exists():
        fallback = yaml.safe_load(published.read_text()).get("kappa_fallback") or {}
        fallback_kappa = fallback.get("value")

    # Synthetic output is kept apart from the real calibration.json
    if not root_given and not market:
        results_root = ROOT / "results" / "smoke"
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    document, grid = calibrate_date(cfg, raw, cfg["snapshot_date"], fallback_kappa, inputs)
    assert_all_finite(document)
    (results_root / "calibration.json").write_text(json.dumps(document, indent=2))

    # The Restricted grid is written only in market mode, and only under data/
    if market and grid is not None:
        GRID_FILE.parent.mkdir(parents=True, exist_ok=True)
        grid[["symbol", "date", "strike", "expiration", "call_put", "iv"]].to_parquet(GRID_FILE)

    # The other two dates are for day-to-day stability only (SPEC.md 5.3)
    if market:
        for date in ALL_DATES:
            if date != cfg["snapshot_date"]:
                other, _ = calibrate_date(cfg, raw, date, fallback_kappa, inputs)
                assert_all_finite(other)
                (results_root / f"calibration_{date}.json").write_text(json.dumps(other, indent=2))

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


def calibration_date_variants(cfg, results_root):
    """Market mode only: price with the parameters fitted on each date."""
    variants = []
    if cfg["data_mode"] != "market":
        return variants
    for date in ALL_DATES:
        name = "calibration.json" if date == cfg["snapshot_date"] else f"calibration_{date}.json"
        if (Path(results_root) / name).exists():
            variant = copy.deepcopy(cfg)
            variant["params_source"] = f"results/{name}"
            variant["run_name"] = f"{cfg['run_name']}__calibration_date_{date}"
            variants.append(("calibration_date", date, variant, 0.0))
    return variants


def run_sweep(cfg, results_root, command):
    begin = time.perf_counter()
    rows = []
    commit = git_commit()
    for factor, value, variant, r_shift in sweep_variants(cfg) + calibration_date_variants(cfg, results_root):
        run_id = f"{variant['run_name']}-{commit[:8]}"
        run_dir = Path(results_root) / "runs" / run_id
        if not (run_dir / "manifest.json").exists():     # finished variants are kept, not redone
            price_run(variant, variant, results_root, command, r_shift, {"factor": factor, "value": value})
        pooled = json.loads((run_dir / "result.json").read_text())["pooled"]
        wall = json.loads((run_dir / "manifest.json").read_text())["wall_seconds"]
        rows.append({"factor": factor, "value": value, "run_id": run_id, "wall_seconds": wall,
                     "bermudan": pooled["bermudan"], "se_bermudan": pooled["se_bermudan"],
                     "premium": pooled["premium"], "se_premium": pooled["se_premium"]})

    summary = {"schema_version": 1, "run_name": cfg["run_name"], "data_mode": cfg["data_mode"],
               "commit": commit, "wall_seconds_this_invocation": time.perf_counter() - begin,
               "wall_seconds_total": sum(row["wall_seconds"] for row in rows), "budget_b2_seconds": 21600,
               "rows": rows}
    out = Path(results_root) / "sweeps"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{cfg['run_name']}-{commit[:8]}.json").write_text(json.dumps(summary, indent=2))
    print(f"sweep finished: {len(rows)} configurations in {summary['wall_seconds_total']:.0f} s")


# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Heston Bermudan put pipeline")
    parser.add_argument("command", choices=["calibrate", "price", "sweep"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-root", default=None)
    args = parser.parse_args()

    command = "python scripts/run.py " + " ".join(sys.argv[1:])
    results_root = Path(args.results_root) if args.results_root else ROOT / "results"
    try:
        cfg = load_config(args.config)
        if args.command == "calibrate":
            run_calibrate(cfg, results_root, args.results_root is not None)
        elif args.command == "price":
            price_run(cfg, args.config, results_root, command)
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
