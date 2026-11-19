"""
Compare Heston LSMC prices with market prices, for any asset, date and parameter sets
(SPEC.md 8.4 and 8.5). Everything about the case lives in the config.

    python scripts/market_test.py --config configs/market_test_<name>.yaml [--workers 8]
        [--n-paths N] [--seeds S ...] [--bootstrap B] [--output PATH] [--per-contract-output PATH]

Two kinds of market data (config key `source`):

  table        A Public CSV of published prices (maturity, strike, market_price) with one spot.
               Reports price gaps in currency for every contract (SPEC.md 8.4).
  ivolatility  A licensed iVolatility Raw IV file under data/. Filters F1 to F8 and the grid of
               SPEC.md 6 split the quotes into in-sample (the calibration grid) and out-of-sample
               (traded, not in the grid) contracts, scored against bid, ask and mid (SPEC.md 8.5).

Data boundary: when any input is under data/, per-contract values may be written only under
data/derived/, the summary holds aggregates only, every group has n >= 5, and the governance
gate must be open. The owner reviews this script before it is run on licensed data (contract 3.3).
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import numpy as np
import pandas as pd
import yaml

from hestonpricer import market, surface
from hestonpricer.errors import EXIT_CODES, HestonError
from hestonpricer.iv import american_implied_vol

SPEC_VERSION = "1.13"
BUDGET_B4_SECONDS = 7200
NAMES = ("v0", "kappa", "theta", "xi", "rho")
REQUIRED = {"schema_version", "name", "source", "symbol", "date", "market_inputs", "parameter_sets",
            "exercise", "n_paths", "seeds", "outputs"}
BY_SOURCE = {"table": {"table", "spot", "kind", "citation"}, "ivolatility": {"raw_file", "bootstrap"}}
OPTIONAL = {"highlight", "note"}


# ----------------------------------------------------------------------
# Configuration and inputs
# ----------------------------------------------------------------------

def under_data(path):
    return (ROOT / path).resolve().is_relative_to((ROOT / "data").resolve())


def load_config(path):
    """Read a market-test config. Unknown or missing keys raise E_CONFIG."""
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg.get("source") not in BY_SOURCE:
        raise HestonError("E_CONFIG", "source must be table or ivolatility")
    allowed = REQUIRED | BY_SOURCE[cfg["source"]] | OPTIONAL
    unknown, missing = set(cfg) - allowed, (REQUIRED | BY_SOURCE[cfg["source"]]) - set(cfg)
    if unknown or missing:
        raise HestonError("E_CONFIG", f"unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    if cfg["exercise"] != "daily" and not (isinstance(cfg["exercise"], dict) and set(cfg["exercise"]) == {"n_exercise"}):
        raise HestonError("E_CONFIG", "exercise must be 'daily' or {n_exercise: N}")
    if cfg["source"] == "table" and len(cfg["parameter_sets"]) != 1:
        raise HestonError("E_CONFIG", "a table comparison prices one parameter set")
    cfg["date"] = str(cfg["date"])
    cfg["exercise_mode"] = cfg["exercise"] if cfg["exercise"] == "daily" else int(cfg["exercise"]["n_exercise"])
    return cfg


def check_boundary(cfg, per_contract_path):
    """Licensed inputs: governance gate open, and per-contract output only under data/derived/."""
    inputs = [cfg.get("raw_file"), cfg.get("table")]
    if not any(path and under_data(path) for path in inputs):
        return False
    governance = ROOT / "configs" / "governance.yaml"
    if not (governance.exists() and yaml.safe_load(governance.read_text()).get("market_use_authorized") is True):
        raise HestonError("E_GOV_UNAUTHORIZED", "market_use_authorized is not true")
    if not Path(per_contract_path).resolve().is_relative_to((ROOT / "data" / "derived").resolve()):
        raise HestonError("E_CONFIG", "per-contract output from licensed data must be under data/derived/")
    return True


def rate_and_dividend(cfg):
    """r = 2 ln(1 + y / 2) from the 1-year Treasury yield on the date, and the asset's dividend yield."""
    inputs = yaml.safe_load((ROOT / cfg["market_inputs"]).read_text())
    y = {str(k): v for k, v in inputs["treasury_1y_cmt_yield"].items()}[cfg["date"]]
    return 2.0 * math.log(1.0 + float(y) / 2.0), float(inputs["dividend_yield"][cfg["symbol"]])


def load_parameter_set(entry, symbol):
    """A calibration.json (accepted, or REJECTED at ladder step 3) or a published set."""
    path = ROOT / entry["params_source"]
    if path.suffix == ".json":
        fit = json.loads(path.read_text())["assets"][symbol]
        if fit["ladder_step"] in (1, 2):
            return {"params": fit["params"], "status": "accepted", "ladder_step": fit["ladder_step"],
                    "rms_iv_residual": fit["rms_iv_residual"], "limit": 0.01}
        if fit["ladder_step"] == 3:
            rejected = fit["rejected_fit"]
            return {"params": rejected["params"], "status": "rejected", "ladder_step": 3,
                    "rms_iv_residual": rejected["rms_iv_residual"], "limit": rejected["limit"],
                    "from_step": rejected["from_step"]}
        raise HestonError("E_CALIB_FAILED", "ladder step 4: no calibrated parameter set (SPEC.md 6.7)")
    published = yaml.safe_load(path.read_text())["sets"][entry.get("set", symbol)]
    return {"params": {name: float(published[name]) for name in NAMES}, "status": "published",
            "source": f"{published['citation']}, {published['table_or_page']}"}


def describe(parameter_set):
    """What a summary says about a parameter set: everything but the parameters, plus REJECTED."""
    info = {key: value for key, value in parameter_set.items() if key != "params"}
    return {**info, "label": "REJECTED"} if parameter_set["status"] == "rejected" else info


# ----------------------------------------------------------------------
# Table source: published prices (SPEC.md 8.4)
# ----------------------------------------------------------------------

def run_table(cfg, args, r, q):
    spot, kind = float(cfg["spot"]), cfg["kind"]
    with open(ROOT / cfg["table"], newline="") as file:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(file)]
    table = pd.DataFrame(rows)
    entry = cfg["parameter_sets"][0]
    parameter_set = load_parameter_set(entry, cfg["symbol"])

    contracts = market.price_table_contracts(table, spot, kind)
    priced = market.price_contracts(parameter_set["params"], contracts, r, q, args.n_paths, args.seeds,
                                    args.workers, cfg["exercise_mode"])
    # Rows grouped by maturity, in table order within each maturity
    order = np.argsort(table["maturity"].to_numpy(float), kind="stable")
    out_rows = [{**rows[i], "model_price": float(priced["V"].iloc[i]) * spot, "se": float(priced["se"].iloc[i]) * spot}
                for i in order]

    gaps = np.array([row["model_price"] - row["market_price"] for row in out_rows])
    relative = np.array([abs(row["model_price"] / row["market_price"] - 1) for row in out_rows])
    by_maturity = []
    for maturity in sorted({row["maturity"] for row in out_rows}):
        part = [row["model_price"] - row["market_price"] for row in out_rows if row["maturity"] == maturity]
        by_maturity.append({"maturity": maturity, "n": len(part), "rms_usd": float(np.sqrt(np.mean(np.square(part))))})

    document = {"schema_version": 1, "source": cfg["citation"],
                "note": cfg.get("note", "In-sample: the published parameters were fitted to these same prices.")}
    highlight = cfg.get("highlight")
    if highlight:
        document.update(highlight_contract(cfg, highlight, rows, spot, kind, r, q))
    document[f"all_{kind}s"] = {
        "n": len(out_rows), "n_paths": args.n_paths, "n_exercise": cfg["exercise_mode"],
        "seed": args.seeds[0] if len(args.seeds) == 1 else args.seeds,
        "rms_usd": float(np.sqrt(np.mean(gaps**2))), "median_abs_usd": float(np.median(np.abs(gaps))),
        "max_abs_usd": float(np.max(np.abs(gaps))), "median_abs_percent": float(100.0 * np.median(relative)),
        "within_2_percent": int((relative <= 0.02).sum()), "within_5_percent": int((relative <= 0.05).sum()),
        "by_maturity": by_maturity,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2))
    per_contract = Path(args.per_contract_output)
    per_contract.parent.mkdir(parents=True, exist_ok=True)
    with open(per_contract, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["maturity", "strike", "market_price", "model_price", "se"])
        writer.writeheader()
        writer.writerows(out_rows)

    message = f"{len(out_rows)} {kind}s priced; median absolute gap {np.median(np.abs(gaps)):.2f} (currency)"
    if highlight:
        official = document["official_contract"]
        message = f"official {kind}: model {official['model_usd']:.2f} against market {official['market_usd']:.2f}; " + message
    print(message)
    print(f"written: {output.name}, {per_contract.name}")


def highlight_contract(cfg, highlight, rows, spot, kind, r, q):
    """One named contract, priced by its own reported `run.py price` run (for example 240 daily dates)."""
    strike, maturity = float(highlight["strike"]), float(highlight["maturity"])
    matches = [row for row in rows if row["maturity"] == maturity and row["strike"] == strike]
    if len(matches) != 1:
        raise HestonError("E_CONFIG", "the highlighted contract is not in the table")
    # The newest complete run whose saved config is a reported one; exploratory reruns of the same
    # run_name (reported: false) are ignored, whatever their file times
    runs = sorted((path for path in ROOT.glob(f"results/runs/{highlight['reference_run']}-*/result.json")
                   if (path.parent / "manifest.json").exists()
                   and yaml.safe_load((path.parent / "config.yaml").read_text()).get("reported") is True),
                  key=lambda path: json.loads((path.parent / "manifest.json").read_text())["finished_at"])
    if not runs:
        raise HestonError("E_CONFIG", f"no reported {highlight['reference_run']} run found: run scripts/run.py price first")
    result = json.loads(runs[-1].read_text())
    market_price, model = matches[0]["market_price"], result["pooled"]["bermudan"] * spot

    def implied(price):
        return american_implied_vol(kind, price / spot, 1.0, strike / spot, maturity, r, q)

    return {"run_id": result["run_id"], "data_mode": result["data_mode"], "official_contract": {
        "valuation_date": cfg["date"], "style": f"American {kind}", "underlying_usd": spot, "strike_usd": strike,
        "maturity_years": maturity, "r": r, "q": q,
        "market_usd": market_price, "model_usd": model, "se_usd": result["pooled"]["se_bermudan"] * spot,
        "european_analytic_usd": result["pooled"]["european_analytic"] * spot,
        "model_minus_market_usd": model - market_price,
        "model_minus_market_percent": 100.0 * (model - market_price) / market_price,
        "market_implied_vol": implied(market_price), "model_implied_vol": implied(model),
    }}


# ----------------------------------------------------------------------
# iVolatility source: quotes within and outside the calibrated sample (SPEC.md 8.5)
# ----------------------------------------------------------------------

def run_ivolatility(cfg, args, r, q):
    begin = time.perf_counter()
    sets = {entry["name"]: load_parameter_set(entry, cfg["symbol"]) for entry in cfg["parameter_sets"]}
    primary = cfg["parameter_sets"][0]["name"]

    raw = surface.load_raw_iv(ROOT / cfg["raw_file"])
    contracts, test_counts, in_sample_counts = market.ivolatility_populations(raw, cfg["symbol"], cfg["date"], r, q)
    del raw

    results, in_sample_results, priced_tables, timings = {}, {}, [], {}
    for name, entry in sets.items():
        start = time.perf_counter()
        priced = market.add_model_iv(market.price_contracts(entry["params"], contracts, r, q, args.n_paths, args.seeds,
                                                            args.workers, cfg["exercise_mode"]), r, q)
        timings[name] = time.perf_counter() - start
        results[name] = market.summarize(priced[priced["population"] == "out_of_sample"], args.bootstrap)
        in_sample_results[name] = market.summarize(priced[priced["population"] == "in_sample"], args.bootstrap)
        priced_tables.append(priced.assign(parameter_set=name))
        if name == primary:
            precision = market.precision_t3(priced[priced["population"] == "out_of_sample"], args.n_paths)
        print(f"{name}: priced {len(priced)} contracts in {timings[name]:.0f} s")

    test_rows = contracts[contracts["population"] == "out_of_sample"]
    shortcut = market.summarize(market.tree_shortcut(sets[primary]["params"], test_rows, r, q), 0)

    # Per-contract values are Restricted: under data/derived/ only (checked by check_boundary)
    per_contract = Path(args.per_contract_output)
    per_contract.parent.mkdir(parents=True, exist_ok=True)
    keep = ["option_symbol", "population", "parameter_set", "expiration", "kind", "strike", "tenor", "n_exercise",
            "V", "se", "ci_low", "ci_high", "european", "se_european", "premium", "se_premium", "model_iv",
            "skipped_dates", "rank_deficient_dates", "min_itm_count"]
    pd.concat(priced_tables)[keep].to_parquet(per_contract)

    total = time.perf_counter() - begin
    exercise = "ceil(252 T) dates" if cfg["exercise_mode"] == "daily" else f"{cfg['exercise_mode']} dates"
    document = {
        "schema_version": 1, "spec_version": SPEC_VERSION, "symbol": cfg["symbol"], "snapshot_date": cfg["date"],
        "units": "fraction_of_initial_spot", "r": r, "q": q,
        "note": cfg.get("note", "Aggregates only. out_of_sample (results) is the SPEC.md 8.5 test population; "
                                "in_sample_results prices the calibration grid the fit was made to. Same snapshot for both."),
        "test_set": test_counts,
        "in_sample_set": in_sample_counts,
        "pricing": {"engine": "lsmc", "n_paths": args.n_paths, "seeds": args.seeds,
                    "n_substeps": 1 if cfg["exercise_mode"] == "daily" else "about one per trading day",
                    "exercise": exercise, "basis_degree": market.BASIS_DEGREE,
                    "paths_shared_within_expiration": True, "bootstrap_resamples": args.bootstrap,
                    "bootstrap_seed": market.BOOTSTRAP_SEED, "bootstrap_unit": "expiration"},
        "parameter_sets": {name: describe(entry) for name, entry in sets.items()},
        "results": results,
        "in_sample_results": in_sample_results,
        "tree_shortcut": {"parameter_set": primary, "overall": shortcut["overall"],
                          "by_maturity": shortcut["by_maturity"]},
        "precision": precision,
        "runtime": {"wall_seconds": total, "pricing_seconds": timings, "budget_b4_seconds": BUDGET_B4_SECONDS,
                    "b4_met": bool(sum(timings.values()) <= BUDGET_B4_SECONDS), "workers": args.workers},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2))

    for name, entry in sets.items():
        tag = " (REJECTED fit)" if entry["status"] == "rejected" else ""
        for population, table in (("out of sample", results), ("in sample", in_sample_results)):
            o = table[name]["overall"]
            iv = "n/a" if o["median_abs_iv_error"] is None else f"{o['median_abs_iv_error']:.4f}"
            print(f"{name}{tag}, {population}: n {o['n']}, inside spread {o['share_inside_spread']:.3f}, "
                  f"median rel error {o['median_rel_price_error']:.4f}, median |IV error| {iv}")
    print(f"T3 share {precision['T3_share']:.3f} (met {precision['T3_met']}); total {total:.0f} s")
    if not precision["T3_met"] and args.n_paths < 200_000:
        print("T3 missed: rerun with --n-paths 200000 (SPEC.md 9.2)")
    print(f"written: {output.name}")


# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare Heston LSMC prices with market data named in a config")
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--n-paths", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--bootstrap", type=int, default=None)
    parser.add_argument("--output", default=None, help="summary path (default: the config's outputs.summary)")
    parser.add_argument("--per-contract-output", default=None, help="default: the config's outputs.per_contract")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        configured_output = (ROOT / cfg["outputs"]["summary"]).resolve()
        args.output = str(Path(args.output).resolve() if args.output else configured_output)
        args.per_contract_output = str(Path(args.per_contract_output).resolve() if args.per_contract_output
                                       else (ROOT / cfg["outputs"]["per_contract"]).resolve())
        args.n_paths = args.n_paths or int(cfg["n_paths"])
        args.seeds = args.seeds or [int(s) for s in cfg["seeds"]]
        args.bootstrap = int(cfg.get("bootstrap", 0)) if args.bootstrap is None else args.bootstrap
        # Reported outputs live under results/; they may not use fewer than 20,000 paths (SPEC.md 9.2)
        if args.n_paths < 20_000 and Path(args.output).is_relative_to((ROOT / "results").resolve()):
            raise HestonError("E_CONFIG", "n_paths below 20,000 is not allowed for an output under results/ (SPEC.md 9.2)")
        licensed = check_boundary(cfg, args.per_contract_output)
        r, q = rate_and_dividend(cfg)
        if cfg["source"] == "table":
            run_table(cfg, args, r, q)
        else:
            if not (ROOT / cfg["raw_file"]).exists():
                raise HestonError("E_CONFIG", f"{Path(cfg['raw_file']).name} is not present; this comparison "
                                              "needs the licensed data (DATA_CONTRACT.md)")
            run_ivolatility(cfg, args, r, q)
    except HestonError as error:
        print(str(error), file=sys.stderr)
        return EXIT_CODES.get(error.code, 1)
    except Exception as error:
        # Sanitized on licensed data: pandas messages can echo row contents (SPEC.md 14.2)
        if not locals().get("licensed"):
            raise
        print(f"internal error: {type(error).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
