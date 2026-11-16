"""
Make the figures and tables of SPEC.md section 12.8 from committed Public and
Internal outputs. It never reads anything under data/.

    python scripts/make_figures.py [--results-root <dir>] [--run-name <name>]

Each figure carries a footnote with its units, workload, repetitions, uncertainty,
exclusions, source run and generating code, and a heading computed from the numbers.
Each output gets a row in evidence_map.csv.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hestonpricer import surface
from hestonpricer.calibrate import model_implied_vols

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
CODE = "scripts/make_figures.py"
EVIDENCE_COLUMNS = ["claim_id", "location", "artifact_type", "run_id", "result_path", "json_pointer", "check_ids"]

SWEEP_PANELS = [("n_paths", "Paths per set"), ("n_substeps", "Sub-steps per exercise interval"),
                ("M", "Exercise dates M (at fixed step 1/200)"), ("basis_degree", "Regression basis degree")]


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 9, "text.color": INK, "axes.labelcolor": MUTED, "axes.edgecolor": GRID,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    })


def relative(path, results_root):
    """Evidence-map paths are relative to the results root."""
    return str(Path(path).relative_to(results_root))


def add_footnote(fig, text):
    fig.text(0.01, 0.005, text, ha="left", va="bottom", fontsize=7, color=MUTED, wrap=True)


def status_prefix(document):
    """The REJECTED label every result of a step-3 fit carries in its heading (SPEC.md 2.4)."""
    rejected = document.get("rejected_fit")
    if document.get("parameter_status") != "rejected" or not rejected:
        return ""
    return f"REJECTED FIT (RMS IV {rejected['rms_iv_residual']:.5f} > limit {rejected['limit']}). "


def find_run(results_root, run_name):
    """The newest completed run with this name, or None."""
    manifests = sorted(Path(results_root).glob(f"runs/{run_name}-*/manifest.json"),
                       key=lambda p: p.stat().st_mtime)
    return manifests[-1].parent if manifests else None


def default_run_name(results_root):
    for name in ("core_goog_reference", "core_goog_production", "core_smoke"):
        if find_run(results_root, name) is not None:
            return name
    return None


def price_figure(run_dir, results_root, out_dir):
    """Prices with 95% intervals, and the premium on its own axis."""
    result = json.loads((run_dir / "result.json").read_text())
    pooled = result["pooled"]
    labels = ["Heston Bermudan (LSMC)", "European (Monte Carlo)", "European (semi-analytic)",
              "Constant-vol Bermudan (tree)"]
    values = [pooled["bermudan"], pooled["european_mc"], pooled["european_analytic"],
              pooled["baseline_const_vol_bermudan"]]
    errors = [1.96 * pooled["se_bermudan"], 1.96 * pooled["se_european"], 0.0, 0.0]

    fig, (left, right) = plt.subplots(1, 2, figsize=(8.5, 3.4), width_ratios=[3, 2])
    y = np.arange(len(labels))[::-1]
    left.errorbar(values, y, xerr=errors, fmt="o", color=BLUE, markersize=6, capsize=3, linewidth=1.5)
    left.set_yticks(y, labels)
    left.set_xlabel("Value (fraction of initial spot)")
    right.errorbar([pooled["premium"]], [0], xerr=[1.96 * pooled["se_premium"]], fmt="s", color=ORANGE,
                   markersize=6, capsize=3, linewidth=1.5)
    right.axvline(0, color=MUTED, linewidth=1, linestyle="--")
    right.set_yticks([0], ["Early-exercise premium"])
    right.set_xlabel("Premium (fraction of spot)")

    fig.suptitle(f"{status_prefix(result)}Bermudan {pooled['bermudan']:.4f}, premium {pooled['premium']:.4f} "
                 f"(95% interval {pooled['premium_ci95'][0]:.4f} to {pooled['premium_ci95'][1]:.4f})",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    add_footnote(fig, f"Mode {result['data_mode']}. Model-implied, low-biased estimate; not a market price. "
                      f"{result['simulation']['n_paths']} paths per set, {len(result['per_seed'])} seed(s). "
                      f"Bars: 95% interval. Exclusions: none. Run {result['run_id']}. Code {CODE}.")
    path = out_dir / "price_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return [{"claim_id": "fig_price_summary", "location": "figure", "artifact_type": "figure",
             "run_id": result["run_id"], "result_path": relative(run_dir / "result.json", results_root),
             "json_pointer": "/pooled/bermudan", "check_ids": "V-03;V-04;V-11;V-15"}]


def convergence_figure(run_name, results_root, out_dir):
    """Small multiples: the price against each numerical setting, with 95% intervals."""
    summaries = sorted(Path(results_root).glob(f"sweeps/{run_name}-*.json"), key=lambda p: p.stat().st_mtime)
    if not summaries:
        print("no sweep summary found; skipping the convergence figure")
        return []
    summary = json.loads(summaries[-1].read_text())
    table = pd.DataFrame(summary["rows"])

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.6), sharey=True)
    spreads = []
    for ax, (factor, label) in zip(axes.ravel(), SWEEP_PANELS):
        rows = table[table["factor"] == factor].sort_values("value")
        ax.errorbar(rows["value"].astype(float), rows["bermudan"], yerr=1.96 * rows["se_bermudan"],
                    fmt="o-", color=BLUE, markersize=5, capsize=3, linewidth=1.5)
        if factor in ("n_paths", "M", "n_substeps"):
            ax.set_xscale("log")
            labels = [f"{int(v) // 1000}k" if factor == "n_paths" else str(int(v)) for v in rows["value"]]
            ax.set_xticks(rows["value"].astype(float), labels)
            ax.minorticks_off()
        else:
            ax.set_xticks(rows["value"].astype(float), [str(int(v)) for v in rows["value"]])
        ax.set_xlabel(label)
        spreads.append(rows["bermudan"].max() - rows["bermudan"].min())
    for ax in axes[:, 0]:
        ax.set_ylabel("Bermudan price (fraction of spot)")

    fig.suptitle(f"{status_prefix(summary)}Largest price range across one numerical setting: {max(spreads):.5f} of spot",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    add_footnote(fig, f"Mode {summary['data_mode']}. Each point pools all configured seeds. Bars: 95% interval. "
                      f"M sweep is a contract-convergence study. Sweep {summaries[-1].name}. Code {CODE}.")
    path = out_dir / "convergence.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return [{"claim_id": "fig_convergence", "location": "figure", "artifact_type": "figure",
             "run_id": summary["run_name"], "result_path": relative(summaries[-1], results_root),
             "json_pointer": "/rows/0/bermudan", "check_ids": "V-09;V-10"}]


def surface_figure(results_root, out_dir):
    """Fitted against known implied volatility. Synthetic fixture only (SPEC.md 12.8)."""
    calibration_file = Path(results_root) / "smoke" / "calibration.json"
    if not calibration_file.exists():
        calibration_file = Path(results_root) / "calibration.json"
    if not calibration_file.exists() or json.loads(calibration_file.read_text())["data_mode"] != "synthetic":
        print("no synthetic calibration found; skipping the surface figure")
        return []
    document = json.loads(calibration_file.read_text())
    entry = document["assets"]["SYN1"]
    r, q = document["r"], entry["q"]

    raw = surface.load_raw_iv(ROOT / "tests" / "fixtures" / "synthetic_raw_iv.csv")
    quotes, _ = surface.filter_quotes(raw, {"symbol": "SYN1", "snapshot_date": document["snapshot_date"], "r": r, "q": q})
    grid = surface.select_grid(quotes, {"contract_strike": 1.0, "contract_maturity": 1.0})
    tenors = sorted(grid["tenor"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(8.5, 5.0), sharex=True, sharey=True)
    for ax, tenor in zip(axes.ravel(), tenors):
        rows = grid[grid["tenor"] == tenor].sort_values("k_over_f")
        fitted = model_implied_vols(entry["params"], rows["k_over_s"].to_numpy(), rows["tenor"].to_numpy(), r, q)
        ax.plot(rows["k_over_f"], fitted, "-", color=BLUE, linewidth=2, label="Fitted Heston")
        ax.plot(rows["k_over_f"], rows["iv"], "o", color=ORANGE, markersize=5, markerfacecolor="none",
                markeredgewidth=1.5, label="Known surface")
        ax.set_title(f"Tenor {tenor:.2f} years", fontsize=9, loc="left", color=MUTED)
    axes[0, 0].legend(frameon=False)
    for ax in axes[1]:
        ax.set_xlabel("Strike / forward")
    for ax in axes[:, 0]:
        ax.set_ylabel("Implied volatility")

    fig.suptitle(f"Synthetic surface: RMS implied-vol residual {entry['rms_iv_residual']:.1e}, "
                 f"{entry['grid_size']} points", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    add_footnote(fig, f"Synthetic fixture with known parameters; no market data. One calibration, 8 starts. "
                      f"Exclusions: filters F1 to F8. Source {calibration_file.name}. Code {CODE}.")
    path = out_dir / "synthetic_surface.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return [{"claim_id": "fig_synthetic_surface", "location": "figure", "artifact_type": "figure",
             "run_id": "synthetic", "result_path": relative(calibration_file, results_root),
             "json_pointer": "/assets/SYN1/rms_iv_residual", "check_ids": "V-12"}]


def seed_table(run_dir, results_root, out_dir):
    """Seed robustness table (SPEC.md 12.8)."""
    result = json.loads((run_dir / "result.json").read_text())
    columns = ["seed", "bermudan", "se_bermudan", "european", "se_european", "premium", "se_premium"]
    pd.DataFrame(result["per_seed"])[columns].to_csv(out_dir / "seed_robustness.csv", index=False)
    return [{"claim_id": "table_seed_robustness", "location": "table", "artifact_type": "table",
             "run_id": result["run_id"], "result_path": relative(run_dir / "result.json", results_root),
             "json_pointer": "/per_seed/0/bermudan", "check_ids": "V-11"}]


def market_fit_table(results_root, out_dir):
    """Aggregate fit quality only: no market data point is ever plotted (SPEC.md 12.8)."""
    calibration_file = Path(results_root) / "calibration_goog" / "calibration.json"
    if not calibration_file.exists():
        return []
    document = json.loads(calibration_file.read_text())
    entry = document["assets"]["GOOG"]
    if document["data_mode"] != "market" or "rejected_fit" not in entry or entry["ladder_step"] == 4:
        return []    # no fit, or a summary from before the v1.13 schema
    status = "REJECTED" if entry["ladder_step"] == 3 else "accepted"
    rows = [{"asset": "GOOG", "fit_status": status, "ladder_step": entry["ladder_step"], **bucket}
            for bucket in entry.get("residual_buckets", [])]
    pd.DataFrame(rows).to_csv(out_dir / "market_fit_buckets.csv", index=False)
    return [{"claim_id": "table_market_fit", "location": "table", "artifact_type": "table",
             "run_id": "calibrate_goog", "result_path": relative(calibration_file, results_root),
             "json_pointer": "/assets/GOOG/rms_iv_residual", "check_ids": "V-13"}]


def newest_comparison(run_name, results_root):
    """The newest comparison summary for this run name, or None."""
    files = sorted(Path(results_root).glob(f"comparisons/{run_name}-*.json"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def render_surface(params, r, q, heading, footnote, path, tenor_low=0.10, tenor_high=1.25, kf_low=0.80, kf_high=1.20):
    """PyVista 3D surface of the model implied volatility. Model values only."""
    import pyvista as pv

    k_over_f = np.linspace(kf_low, kf_high, 41)
    tenors = np.linspace(tenor_low, tenor_high, 47)
    vols = np.zeros((len(tenors), len(k_over_f)))
    for j, tenor in enumerate(tenors):
        k_over_s = k_over_f * np.exp((r - q) * tenor)
        vols[j] = model_implied_vols(params, k_over_s, np.full(len(k_over_f), tenor), r, q)

    # The geometry is drawn in a unit box so the surface is readable; the axis labels
    # below show the true values, and the colour scale uses the true implied volatility.
    low, high = float(vols.min()), float(vols.max())
    x, y = np.meshgrid((k_over_f - kf_low) / (kf_high - kf_low), (tenors - tenor_low) / (tenor_high - tenor_low))
    grid = pv.StructuredGrid(x, y, 0.7 * (vols - low) / max(high - low, 1e-6))
    # Read the colour values back from the points, so they cannot be misordered against the geometry
    grid["Implied volatility"] = low + grid.points[:, 2] / 0.7 * (high - low)

    plotter = pv.Plotter(off_screen=True, window_size=(1500, 1050))
    plotter.set_background(SURFACE)
    plotter.add_mesh(grid, scalars="Implied volatility", cmap="Blues", clim=[low - 0.5 * (high - low), high],
                     show_edges=True, edge_color="#ffffff", line_width=0.4, lighting=False,
                     scalar_bar_args={"title": "Implied vol", "color": INK, "n_labels": 5, "fmt": "%.2f",
                                      "vertical": True, "position_x": 0.87, "position_y": 0.3})
    plotter.show_bounds(axes_ranges=[kf_low, kf_high, tenor_low, tenor_high, low, high],
                        xtitle="Strike / forward", ytitle="Tenor (years)",
                        ztitle="Implied vol", color=MUTED, font_size=12, grid="back", location="outer", n_xlabels=5,
                        n_ylabels=5, n_zlabels=4, fmt="%.2f")
    plotter.camera_position = [(1.85, -1.75, 1.35), (0.5, 0.5, 0.3), (0, 0, 1)]
    plotter.add_text(heading, position="upper_left", font_size=12, color=INK)
    plotter.add_text(footnote, position="lower_left", font_size=8, color=MUTED)
    plotter.screenshot(str(path))
    plotter.close()


def goog_surface_figure(results_root, out_dir):
    """VZ-1e: the 2026-10-13 GOOG model surface, labelled REJECTED at ladder step 3 (SPEC.md 12.9)."""
    calibration_file = Path(results_root) / "calibration_goog" / "calibration.json"
    if not calibration_file.exists():
        return []
    document = json.loads(calibration_file.read_text())
    entry = document["assets"]["GOOG"]
    if document["data_mode"] != "market" or "rejected_fit" not in entry or entry["ladder_step"] == 4:
        return []    # no fit, or a summary from before the v1.13 schema
    if entry["ladder_step"] == 3:
        fit = entry["rejected_fit"]
        p, pointer = fit["params"], "/assets/GOOG/rejected_fit/rms_iv_residual"
        heading = (f"REJECTED FIT: GOOG Heston surface, {document['snapshot_date']}\n"
                   f"RMS IV residual {fit['rms_iv_residual']:.5f} above the {fit['limit']} limit "
                   f"(ladder step 3, from step {fit['from_step']}).")
    else:
        fit, p, pointer = entry, entry["params"], "/assets/GOOG/rms_iv_residual"
        heading = (f"GOOG Heston surface, {document['snapshot_date']}\n"
                   f"Accepted at ladder step {entry['ladder_step']}: RMS IV residual {fit['rms_iv_residual']:.5f}.")
    footnote = (f"Model surface only; no market data points shown. K/F 0.70-1.30 and tenor 0.06-1.40 years, the test "
                f"population's range. v0 {p['v0']:.4f}, kappa {p['kappa']:.3f}, theta {p['theta']:.4f}, "
                f"xi {p['xi']:.3f}, rho {p['rho']:.3f}.\n"
                f"Fit: {entry['grid_size']} grid points, 8 starts; no uncertainty band. Exclusions: filters F1 to F8. "
                f"Source {relative(calibration_file, results_root)}. Code {CODE}.")
    render_surface(p, document["r"], entry["q"], heading, footnote, out_dir / "surface_goog_2026_10_13.png",
                   tenor_low=0.06, tenor_high=1.40, kf_low=0.70, kf_high=1.30)
    return [{"claim_id": "fig_surface_goog_2026_10_13", "location": "figure", "artifact_type": "figure",
             "run_id": "calibrate_goog", "result_path": relative(calibration_file, results_root),
             "json_pointer": pointer, "check_ids": "V-13"}]


def published_surface_figure(comparison_file, results_root, out_dir):
    """VZ-1b: the surface implied by the published Google parameters."""
    document = json.loads(comparison_file.read_text())
    if document["data_mode"] != "published":
        return []
    p = document["params"]
    source = document["inputs"][0]
    heading = ("GOOG 2015 HESTON MODEL: implied-volatility surface\n"
               "Parameters fitted to Google American puts on 2015-02-02. Not a surface for today's market.")
    footnote = (f"Model surface; no market data points shown. Tenors span the paper's maturities; r = 0.15%, q = 0. "
                f"v0 {p['v0']:.4f}, kappa {p['kappa']:.4f}, theta {p['theta']:.4f}, "
                f"xi {p['xi']:.4f}, rho {p['rho']:.4f}. No uncertainty band.\n"
                f"Source: {source['citation']}, {source['table_or_page']}. Code {CODE}.")
    path = out_dir / "surface_google_published.png"
    render_surface(p, 0.0015, 0.0, heading, footnote, path, tenor_low=0.20, tenor_high=1.97)
    return [{"claim_id": "fig_surface_google_published", "location": "figure", "artifact_type": "figure",
             "run_id": document["run_name"], "result_path": relative(comparison_file, results_root),
             "json_pointer": "/params/kappa", "check_ids": "V-02"}]


def comparison_figures(comparison_file, results_root, out_dir):
    """VZ-2, VZ-3 and VZ-4: numerical against analytical prices."""
    document = json.loads(comparison_file.read_text())
    mode = document["data_mode"]
    source = relative(comparison_file, results_root)
    rows = pd.DataFrame(document["mc_vs_analytic"])

    # VZ-2: Monte Carlo against semi-analytic European puts
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.5, 3.9))
    left.plot([0, rows["analytic"].max()], [0, rows["analytic"].max()], "--", color=MUTED, linewidth=1, label="Equal")
    left.plot(rows["analytic"], rows["mc"], "o", color=BLUE, markersize=4, label="40 options")
    left.set_xlabel("Semi-analytic price (fraction of spot)")
    left.set_ylabel("Monte Carlo price (fraction of spot)")
    left.legend(frameon=False)
    markers = ["o", "s", "^", "D", "v"]
    for marker, maturity in zip(markers, sorted(rows["maturity"].unique())):
        part = rows[rows["maturity"] == maturity]
        right.plot(part["strike"], part["z"], marker, color=BLUE, markersize=5, markerfacecolor="none", label=f"T = {maturity:g}")
    for level in (-3, 3):
        right.axhline(level, color=MUTED, linewidth=1, linestyle="--")
    right.set_xlabel("Strike (fraction of spot)")
    right.set_ylabel("(Monte Carlo - analytic) / SE")
    right.legend(frameon=False, ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 0.9))
    inside = int((rows["z"].abs() <= 3).sum())
    fig.suptitle(f"European puts: {inside} of {len(rows)} Monte Carlo prices lie within 3 SE of the semi-analytic price",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    add_footnote(fig, f"Mode {mode}. {2 * document['n_paths']} paths, step 1/400, seed {document['comparison_seed']}, "
                      f"same paths across strikes. Dashed lines: +/- 3 SE. Source {source}. Code {CODE}.")
    fig.savefig(out_dir / "compare_mc_vs_analytic.png", dpi=180)
    plt.close(fig)

    # VZ-3: LSMC against the binomial tree
    cases = pd.DataFrame(document["lsmc_vs_tree"])
    fig, ax = plt.subplots(figsize=(7.0, 1.4 + 0.3 * len(cases)))
    y = np.arange(len(cases))[::-1]
    ax.errorbar(cases["lsmc"], y, xerr=3 * cases["se"], fmt="o", color=BLUE, markersize=6, capsize=3, label="LSMC, +/- 3 SE")
    ax.plot(cases["tree"], y, "s", color=ORANGE, markersize=6, markerfacecolor="none", markeredgewidth=1.5, label="CRR tree")
    kinds = cases["kind"] if "kind" in cases else ["put"] * len(cases)
    ax.set_yticks(y, [f"{k}, sigma {a:g}, K {b:g}" for k, a, b in zip(kinds, cases["sigma"], cases["strike"])])
    ax.set_xlabel("Bermudan price (fraction of spot)")
    ax.legend(frameon=False, loc="lower right")
    worst = ((cases["lsmc"] - cases["tree"]).abs() / cases["se"]).max()
    fig.suptitle(f"Constant-volatility test: largest LSMC to tree gap is {worst:.2f} SE", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.09, 1, 0.97))
    add_footnote(fig, f"Test model (xi = 0), not the reported parameters; puts q = 0, calls q = 0.0026. "
                      f"{document['n_paths']} paths per set, 50 dates, "
                      f"seed {document['comparison_seed']}; tree 5,000 steps. Source {source}. Code {CODE}.")
    fig.savefig(out_dir / "compare_lsmc_vs_tree.png", dpi=180)
    plt.close(fig)

    # VZ-4: prices across strikes, and the gaps to the European price
    profile = pd.DataFrame(document["strike_profile"])
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.5, 3.7))
    left.errorbar(profile["strike"], profile["bermudan"], yerr=1.96 * profile["se_bermudan"], fmt="o-", color=BLUE,
                  markersize=5, capsize=2, linewidth=1.5, label="Heston Bermudan")
    left.plot(profile["strike"], profile["european_analytic"], "s--", color=ORANGE, markersize=5, linewidth=1.5,
              label="European (semi-analytic)")
    left.set_xlabel("Strike (fraction of spot)")
    left.set_ylabel("Put price (fraction of spot)")
    left.legend(frameon=False)
    right.errorbar(profile["strike"], profile["premium"], yerr=1.96 * profile["se_premium"], fmt="o-", color=BLUE,
                   markersize=5, capsize=2, linewidth=1.5, label="Heston premium (paired)")
    right.plot(profile["strike"], profile["baseline_const_vol_bermudan"] - profile["european_analytic"], "^:", color=ORANGE,
               markersize=5, linewidth=1.5, label="Constant-vol premium")
    right.set_xlabel("Strike (fraction of spot)")
    right.set_ylabel("Early-exercise premium (fraction of spot)")
    right.legend(frameon=False)
    largest = profile.loc[profile["premium"].idxmax()]
    fig.suptitle(f"{status_prefix(document)}Largest early-exercise premium: {largest['premium']:.4f} of spot "
                 f"at strike {largest['strike']:g}",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    add_footnote(fig, f"Mode {mode}. {document['n_paths']} paths per set, {len(document['seeds'])} seed(s), 50 dates. "
                      f"Bars: 95% interval. Source {source}. Code {CODE}.")
    fig.savefig(out_dir / "compare_strike_profile.png", dpi=180)
    plt.close(fig)

    return [{"claim_id": name, "location": "figure", "artifact_type": "figure", "run_id": document["run_name"],
             "result_path": source, "json_pointer": pointer, "check_ids": checks}
            for name, pointer, checks in (("fig_compare_mc_vs_analytic", "/mc_vs_analytic/0/z", "V-03"),
                                          ("fig_compare_lsmc_vs_tree", "/lsmc_vs_tree/0/lsmc", "V-04"),
                                          ("fig_compare_strike_profile", "/strike_profile/0/bermudan", "V-11"))]


def google_2015_figure(results_root, out_dir):
    """VZ-9: the model against all 401 published Google puts of 2015-02-02."""
    table_file = Path(results_root) / "market_comparison_2015.csv"
    summary_file = Path(results_root) / "market_comparison.json"
    if not table_file.exists() or not summary_file.exists():
        return []
    rows = pd.read_csv(table_file)
    summary = json.loads(summary_file.read_text())
    spot = summary["official_contract"]["underlying_usd"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(8.5, 3.9))
    top = rows["market_price"].max()
    left.plot([0, top], [0, top], "--", color=MUTED, linewidth=1, label="Equal")
    left.plot(rows["market_price"], rows["model_price"], "o", color=BLUE, markersize=3, label="401 published puts")
    left.set_xlabel("Published market price (USD)")
    left.set_ylabel("Simulated price (USD)")
    left.legend(frameon=False)
    for marker, maturity in zip(["o", "s", "^", "D", "v"], sorted(rows["maturity"].unique())):
        part = rows[rows["maturity"] == maturity]
        right.plot(part["strike"] / spot, part["model_price"] - part["market_price"], marker, color=BLUE, markersize=4,
                   markerfacecolor="none", label=f"T = {maturity:.2f}")
    right.axhline(0, color=MUTED, linewidth=1, linestyle="--")
    right.set_xlabel("Strike / spot")
    right.set_ylabel("Simulated - published (USD)")
    right.legend(frameon=False, ncol=2, fontsize=7)
    stats = summary["all_puts"]
    fig.suptitle(f"Google puts, 2015-02-02: median absolute gap {stats['median_abs_usd']:.2f} USD "
                 f"over {stats['n']} options (in-sample)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    add_footnote(fig, f"Parameters were fitted to these prices. {stats['n_paths']} paths per set, "
                      f"{stats['n_exercise']} exercise dates, seed {stats['seed']}. Source: {summary['source']}. Code {CODE}.")
    fig.savefig(out_dir / "compare_google_2015.png", dpi=180)
    plt.close(fig)
    return [{"claim_id": "fig_compare_google_2015", "location": "figure", "artifact_type": "figure",
             "run_id": summary["run_id"], "result_path": relative(summary_file, results_root),
             "json_pointer": "/all_puts/median_abs_usd", "check_ids": "V-06"}]


# ----------------------------------------------------------------------
# VZ-10 to VZ-13: accuracy within and outside the calibrated sample (SPEC.md 8.5, 12.9)
# ----------------------------------------------------------------------

SERIES = [("results", "calibrated_2026_10_13", "Calibrated 2026-10-13, out of sample", "o", BLUE, "full"),
          ("in_sample_results", "calibrated_2026_10_13", "Calibrated 2026-10-13, in sample", "s", BLUE, "none"),
          ("results", "published_2015", "Published 2015, out of sample", "^", ORANGE, "full")]


def market_test_title(document, text):
    """The heading, led by the REJECTED label when the calibrated set failed acceptance (SPEC.md 2.4)."""
    fitted = document["parameter_sets"]["calibrated_2026_10_13"]
    if fitted.get("status") != "rejected":
        return text
    return (f"Calibrated set is a REJECTED FIT (RMS IV {fitted['rms_iv_residual']:.5f} > limit {fitted['limit']}).\n"
            + text)


def market_test_footnote(document, source):
    pricing = document["pricing"]
    return (f"{document['symbol']} {document['snapshot_date']}. Out of sample: {document['test_set']['n_test']} traded "
            f"contracts not in the grid; in sample: the {document['in_sample_set']['n']}-point calibration grid. LSMC, "
            f"{pricing['n_paths']} paths per set, {len(pricing['seeds'])} seeds, daily exercise. Bars: 95% bootstrap "
            f"intervals over expirations ({pricing['bootstrap_resamples']} resamples). Groups with n < 5 omitted. "
            f"Aggregates only. Source {source}. Code {CODE}.")


def plot_grouped(ax, document, key, metric, label_fn=None):
    """One marker per series and group, with bootstrap intervals; shape and fill separate the series."""
    offsets = np.linspace(-0.2, 0.2, len(SERIES))
    labels = None
    for (table, name, label, marker, colour, fill), offset in zip(SERIES, offsets):
        groups = document[table][name][key]
        labels = labels or [g["bucket"] for g in groups]
        xs, ys, lows, highs = [], [], [], []
        for group in groups:
            if group.get(metric) is None or group["bucket"] not in labels:
                continue
            interval = group.get("ci95", {}).get(metric, [group[metric], group[metric]])
            xs.append(labels.index(group["bucket"]) + offset)
            ys.append(group[metric])
            lows.append(group[metric] - interval[0])
            highs.append(interval[1] - group[metric])
        ax.errorbar(xs, ys, yerr=[lows, highs], fmt=marker, color=colour, markerfacecolor=colour if fill == "full" else "none",
                    markersize=6, capsize=3, linewidth=1.2, label=label)
    ax.set_xticks(range(len(labels)), labels)
    return labels


def accuracy_figures(results_root, out_dir):
    path = Path(results_root) / "goog_market_test.json"
    if not path.exists():
        return []
    document = json.loads(path.read_text())
    if "in_sample_results" not in document:
        return []    # written before the v1.13 schema
    source = relative(path, results_root)
    evidence = []

    def row(claim, pointer, checks):
        return {"claim_id": claim, "location": "figure", "artifact_type": "figure", "run_id": "goog_market_test",
                "result_path": source, "json_pointer": pointer, "check_ids": checks}

    # VZ-10: implied-volatility error by maturity and by moneyness
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.0, 3.9), sharey=True)
    plot_grouped(left, document, "by_maturity", "median_abs_iv_error")
    plot_grouped(right, document, "by_moneyness", "median_abs_iv_error")
    left.set_yscale("log")
    left.set_ylabel("Median |model IV - market IV| (log scale)")
    left.set_xlabel("Maturity")
    right.set_xlabel("Strike / forward")
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.1))
    overall = document["results"]["calibrated_2026_10_13"]["overall"]["median_abs_iv_error"]
    fig.suptitle(market_test_title(document, f"Out of sample, the calibrated set's median implied-vol error is {overall:.4f}"),
                 fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.17, 1, 0.94))
    add_footnote(fig, market_test_footnote(document, source))
    fig.savefig(out_dir / "accuracy_iv_error.png", dpi=180)
    plt.close(fig)
    evidence.append(row("fig_accuracy_iv_error", "/results/calibrated_2026_10_13/overall/median_abs_iv_error", "V-19"))

    # VZ-11: model inside the spread, and mid inside the Monte Carlo interval
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0), sharey=True)
    for column, key, label in ((0, "by_maturity", "Maturity"), (1, "by_moneyness", "Strike / forward")):
        plot_grouped(axes[0, column], document, key, "share_inside_spread")
        plot_grouped(axes[1, column], document, key, "share_mid_inside_mc_ci95")
        axes[1, column].set_xlabel(label)
    axes[0, 0].set_ylabel("Share of model prices\ninside bid-ask")
    axes[1, 0].set_ylabel("Share of mids inside the\nMonte Carlo 95% interval")
    axes[0, 0].set_ylim(-0.03, 1.03)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.06))
    inside = document["results"]["calibrated_2026_10_13"]["overall"]["share_inside_spread"]
    fig.suptitle(market_test_title(document, f"Out of sample, {inside:.0%} of calibrated-set prices fall inside the "
                                             "bid-ask spread"), fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.11, 1, 0.94))
    add_footnote(fig, market_test_footnote(document, source))
    fig.savefig(out_dir / "accuracy_inside_spread.png", dpi=180)
    plt.close(fig)
    evidence.append(row("fig_accuracy_inside_spread", "/results/calibrated_2026_10_13/overall/share_inside_spread", "V-19"))

    # VZ-12: Monte Carlo half-width against the market half-spread (T3)
    precision = document["precision"]
    bins = precision["half_width_to_half_spread"]["bins"]
    # Equal-width bars labelled by range, so a wide merged bin is not drawn larger than it is
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    # Each bar is labelled as well as coloured: within the limit, outside it, or a merged bin spanning it
    sides = ["within" if b["high"] <= 0.5 else "outside" if b["low"] >= 0.5 else "spans 0.5" for b in bins]
    colours = {"within": BLUE, "outside": ORANGE, "spans 0.5": MUTED}
    ax.bar(range(len(bins)), [b["n"] for b in bins], color=[colours[s] for s in sides], edgecolor=SURFACE)
    for i, (b, side) in enumerate(zip(bins, sides)):
        ax.text(i, b["n"], side, ha="center", va="bottom", fontsize=7, color=MUTED)
    ax.set_xticks(range(len(bins)), [f"{b['low']:.3g}-{b['high']:.3g}" for b in bins], fontsize=7)
    ax.set_xlabel("Monte Carlo 95% half-width / market half-spread (T3 limit 0.5)")
    ax.set_ylabel("Contracts (bins merged to n >= 5)")
    verdict = "met" if precision["T3_met"] else "not met"
    fig.suptitle(market_test_title(document, f"T3 {verdict}: {precision['T3_share']:.0%} of contracts within the limit "
                                             "(needs 90%)"), fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 0.93))
    add_footnote(fig, f"Calibrated set, out of sample, {precision['n_paths_used']} paths per set, "
                      f"{len(document['pricing']['seeds'])} seeds. Source {source}. Code {CODE}.")
    fig.savefig(out_dir / "precision_t3.png", dpi=180)
    plt.close(fig)
    evidence.append({**row("fig_precision_t3", "/precision/T3_share", "T3"), "location": "figure"})

    # VZ-13: LSMC pricing against the tree shortcut, by maturity
    shortcut = {g["bucket"]: g for g in document["tree_shortcut"]["by_maturity"]}
    lsmc = {g["bucket"]: g for g in document["results"]["calibrated_2026_10_13"]["by_maturity"]}
    buckets = [b for b in lsmc if b in shortcut]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    x = np.arange(len(buckets))
    for ax, metric, label in ((left, "median_abs_iv_error", "Median |IV error|"),
                              (right, "share_inside_spread", "Share inside bid-ask")):
        ax.plot(x - 0.08, [lsmc[b][metric] for b in buckets], "o", color=BLUE, markersize=6, label="LSMC (reported)")
        ax.plot(x + 0.08, [shortcut[b][metric] for b in buckets], "D", color=ORANGE, markersize=6, markerfacecolor="none",
                markeredgewidth=1.5, label="Tree shortcut (diagnostic)")
        ax.set_xticks(x, buckets)
        ax.set_ylabel(label)
        ax.set_xlabel("Maturity")
    left.legend(frameon=False, fontsize=7)
    fig.suptitle(market_test_title(document, "Pricing with LSMC instead of the tree shortcut, calibrated set, "
                                             "out of sample"), fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    add_footnote(fig, "The shortcut carries the model's European IV through a 500-step American tree; no Monte Carlo "
                      f"interval. Source {source}. Code {CODE}.")
    fig.savefig(out_dir / "accuracy_lsmc_vs_tree.png", dpi=180)
    plt.close(fig)
    evidence.append(row("fig_accuracy_lsmc_vs_tree", "/tree_shortcut/overall/median_abs_iv_error", "V-19"))
    return evidence


def main():
    parser = argparse.ArgumentParser(description="Make figures from Public and Internal outputs")
    parser.add_argument("--results-root", default=str(ROOT / "results"))
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    out_dir = results_root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    style()

    evidence = []
    run_name = args.run_name or default_run_name(results_root)
    if run_name is None:
        print("no completed run found; only the synthetic surface can be drawn")
    else:
        run_dir = find_run(results_root, run_name)
        evidence += price_figure(run_dir, results_root, out_dir)
        evidence += seed_table(run_dir, results_root, out_dir)
        evidence += convergence_figure(run_name, results_root, out_dir)
        comparison_file = newest_comparison(run_name, results_root)
        if comparison_file is not None:
            evidence += comparison_figures(comparison_file, results_root, out_dir)
    # VZ-1b uses the published-mode comparison, which holds the published Google parameters
    published_comparison = newest_comparison("core_goog_production", results_root)
    if published_comparison is not None:
        evidence += published_surface_figure(published_comparison, results_root, out_dir)
    evidence += goog_surface_figure(results_root, out_dir)
    evidence += accuracy_figures(results_root, out_dir)
    evidence += google_2015_figure(results_root, out_dir)
    evidence += surface_figure(results_root, out_dir)
    evidence += market_fit_table(results_root, out_dir)

    with open(results_root / "evidence_map.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=EVIDENCE_COLUMNS)
        writer.writeheader()
        writer.writerows(evidence)
    print(f"{len(evidence)} output(s) written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
