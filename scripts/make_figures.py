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


def find_run(results_root, run_name):
    """The newest completed run with this name, or None."""
    manifests = sorted(Path(results_root).glob(f"runs/{run_name}-*/manifest.json"),
                       key=lambda p: p.stat().st_mtime)
    return manifests[-1].parent if manifests else None


def default_run_name(results_root):
    for name in ("core_amd_production", "core_goog_production", "core_smoke"):
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

    fig.suptitle(f"Bermudan {pooled['bermudan']:.4f}, premium {pooled['premium']:.4f} "
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

    fig.suptitle(f"Largest price range across one numerical setting: {max(spreads):.5f} of spot",
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
    calibration_file = Path(results_root) / "calibration.json"
    if not calibration_file.exists():
        return []
    document = json.loads(calibration_file.read_text())
    if document["data_mode"] != "market" or document["assets"]["AMD"]["ladder_step"] == 3:
        return []    # no accepted market fit to report
    rows = []
    for asset, entry in document["assets"].items():
        for bucket in entry.get("residual_buckets", []):
            rows.append({"asset": asset, **bucket})
    pd.DataFrame(rows).to_csv(out_dir / "market_fit_buckets.csv", index=False)
    return [{"claim_id": "table_market_fit", "location": "table", "artifact_type": "table",
             "run_id": "market", "result_path": relative(calibration_file, results_root),
             "json_pointer": "/assets/AMD/rms_iv_residual", "check_ids": "V-13"}]


def newest_comparison(run_name, results_root):
    """The newest comparison summary for this run name, or None."""
    files = sorted(Path(results_root).glob(f"comparisons/{run_name}-*.json"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def render_surface(params, r, q, heading, footnote, path):
    """PyVista 3D surface of the model implied volatility. Model values only."""
    import pyvista as pv

    k_over_f = np.linspace(0.80, 1.20, 41)
    tenors = np.linspace(0.10, 1.25, 47)
    vols = np.zeros((len(tenors), len(k_over_f)))
    for j, tenor in enumerate(tenors):
        k_over_s = k_over_f * np.exp((r - q) * tenor)
        vols[j] = model_implied_vols(params, k_over_s, np.full(len(k_over_f), tenor), r, q)

    # The geometry is drawn in a unit box so the surface is readable; the axis labels
    # below show the true values, and the colour scale uses the true implied volatility.
    low, high = float(vols.min()), float(vols.max())
    x, y = np.meshgrid((k_over_f - 0.80) / 0.40, (tenors - 0.10) / 1.15)
    grid = pv.StructuredGrid(x, y, 0.7 * (vols - low) / max(high - low, 1e-6))
    # Read the colour values back from the points, so they cannot be misordered against the geometry
    grid["Implied volatility"] = low + grid.points[:, 2] / 0.7 * (high - low)

    plotter = pv.Plotter(off_screen=True, window_size=(1500, 1050))
    plotter.set_background(SURFACE)
    plotter.add_mesh(grid, scalars="Implied volatility", cmap="Blues", clim=[low - 0.5 * (high - low), high],
                     show_edges=True, edge_color="#ffffff", line_width=0.4, lighting=False,
                     scalar_bar_args={"title": "Implied vol", "color": INK, "n_labels": 5, "fmt": "%.2f",
                                      "vertical": True, "position_x": 0.87, "position_y": 0.3})
    plotter.show_bounds(axes_ranges=[0.80, 1.20, 0.10, 1.25, low, high], xtitle="Strike / forward", ytitle="Tenor (years)",
                        ztitle="Implied vol", color=MUTED, font_size=12, grid="back", location="outer", n_xlabels=5,
                        n_ylabels=5, n_zlabels=4, fmt="%.2f")
    plotter.camera_position = [(1.85, -1.75, 1.35), (0.5, 0.5, 0.3), (0, 0, 1)]
    plotter.add_text(heading, position="upper_left", font_size=12, color=INK)
    plotter.add_text(footnote, position="lower_left", font_size=8, color=MUTED)
    plotter.screenshot(str(path))
    plotter.close()


def rejected_surface_figure(results_root, out_dir):
    """VZ-1a: the rejected full AMD fit. It is the only AMD surface, and it is labelled as rejected."""
    calibration_file = Path(results_root) / "calibration.json"
    if not calibration_file.exists():
        return []
    document = json.loads(calibration_file.read_text())
    entry = document["assets"]["AMD"]
    rejected = entry.get("rejected_full_fit")
    if document["data_mode"] != "market" or rejected is None:
        return []

    p = rejected["params"]
    at_bound = " (at its upper bound)" if p["kappa"] >= 19.999 else ""
    heading = (f"REJECTED FIT: AMD Heston surface, {document['snapshot_date']}\n"
               f"Failed acceptance: RMS IV residual {rejected['rms_iv_residual']:.4f} "
               "above the 0.01 limit. Not used for pricing.")
    footnote = (f"Model surface from rejected parameters; no market data points shown. "
                f"v0 {p['v0']:.4f}, kappa {p['kappa']:.2f}{at_bound}, "
                f"theta {p['theta']:.4f}, xi {p['xi']:.3f}, rho {p['rho']:.3f}.\n"
                f"Fit: {rejected['grid_size']} grid points, 8 starts; no uncertainty band. Exclusions: filters F1 to F8. "
                f"Source calibration.json. Code {CODE}.")
    path = out_dir / "surface_amd_rejected.png"
    render_surface(p, document["r"], entry["q"], heading, footnote, path)
    return [{"claim_id": "fig_surface_amd_rejected", "location": "figure", "artifact_type": "figure",
             "run_id": "market-rejected", "result_path": relative(calibration_file, results_root),
             "json_pointer": "/assets/AMD/rejected_full_fit/rms_iv_residual", "check_ids": "V-13"}]


def published_surface_figure(comparison_file, results_root, out_dir):
    """VZ-1b: the surface implied by the published Google parameters."""
    document = json.loads(comparison_file.read_text())
    if document["data_mode"] != "published":
        return []
    p = document["params"]
    source = document["inputs"][0]
    heading = ("PUBLISHED PARAMETERS: Google options, 2015-02-02\n"
               "Not calibrated to AMD, NVDA or INTC.")
    footnote = (f"Model surface; no market data points shown. v0 {p['v0']:.4f}, kappa {p['kappa']:.4f}, theta {p['theta']:.4f}, "
                f"xi {p['xi']:.4f}, rho {p['rho']:.4f}. No uncertainty band.\n"
                f"Source: {source['citation']}, {source['table_or_page']}. Code {CODE}.")
    path = out_dir / "surface_google_published.png"
    render_surface(p, document["r"], document["q"], heading, footnote, path)
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
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    y = np.arange(len(cases))[::-1]
    ax.errorbar(cases["lsmc"], y, xerr=3 * cases["se"], fmt="o", color=BLUE, markersize=6, capsize=3, label="LSMC, +/- 3 SE")
    ax.plot(cases["tree"], y, "s", color=ORANGE, markersize=6, markerfacecolor="none", markeredgewidth=1.5, label="CRR tree")
    ax.set_yticks(y, [f"sigma {a:g}, K {b:g}" for a, b in zip(cases["sigma"], cases["strike"])])
    ax.set_xlabel("Bermudan put (fraction of spot)")
    ax.legend(frameon=False, loc="lower right")
    worst = ((cases["lsmc"] - cases["tree"]).abs() / cases["se"]).max()
    fig.suptitle(f"Constant-volatility test: largest LSMC to tree gap is {worst:.2f} SE", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.09, 1, 0.97))
    add_footnote(fig, f"Test model (xi = 0), not the reported parameters. {document['n_paths']} paths per set, 50 dates, "
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
    fig.suptitle(f"Largest early-exercise premium: {largest['premium']:.4f} of spot at strike {largest['strike']:g}",
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
            evidence += published_surface_figure(comparison_file, results_root, out_dir)
    evidence += rejected_surface_figure(results_root, out_dir)
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
