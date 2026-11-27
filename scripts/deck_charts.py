"""
Deck-sized charts for the presentation slides (6.5 x 4.5 in). Aggregates only, plus the
public 2015 published prices.

    .venv/bin/python scripts/deck_charts.py [repo] [output folder]

Both arguments are optional: repo defaults to the repository root and the output folder to
results/figures/presentation/. Replace the images in the slides with the new PNGs.

Each chart is written three ways: .svg and .pdf (vector; use the .pdf with includegraphics
in LaTeX) and .png (Google Slides cannot import SVG or PDF).

    d_payoffs         slide 3, Options Contract
    d_paths           slide 7, Monte Carlo Simulations
    d_2015            slide 9, 2015 parameters on 2015 prices
    d_inside_spread   slide 10, share of model prices inside the bid-ask range, 2015 vs 2026
    d_price_gap       slide 10, median % gap between model price and mid, 2015 vs 2026
    d_maturity        slide 11, Where the 2026 model struggles
    d_maturity_nvda   the same chart for NVDA, INTC and AMD (2026-10-13 calibrations;
    d_maturity_intc     results/<ticker>/market_test.json from scripts/market_test.py)
    d_maturity_amd
    d_price_gap_tickers  median % gap between model price and mid, GOOG, NVDA, INTC, AMD
    d_surface         calibration slide: the 2026-10-13 GOOG model IV surface (model values only)
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
out = Path(sys.argv[2]) if len(sys.argv) > 2 else repo / "results" / "figures" / "presentation"
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(repo / "python"))
from hestonpricer.calibrate import model_implied_vols  # noqa: E402
mt = json.load(open(repo / "results/goog/market_test.json"))
pub = json.load(open(repo / "results/goog/market_comparison_2015.json"))
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "axes.titlesize": 8.5, "axes.titlecolor": "black",
    "text.color": "black", "legend.labelcolor": "black",
})
DPI = 300
res = mt["results"]
a, b = res["published_2015"]["overall"], res["calibrated_2026_10_13"]["overall"]


def save(fig, name):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=DPI)
    plt.close(fig)


def grid_y(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


# Payoffs: made-up example
s = np.linspace(70, 150, 400)
fig, axes = plt.subplots(1, 2, figsize=(5.9, 2.25), sharey=True)
for ax, pay, name, col in [(axes[0], np.maximum(s - 110, 0), "Call: the right to BUY at $110", BLUE),
                           (axes[1], np.maximum(110 - s, 0), "Put: the right to SELL at $110", ORANGE)]:
    ax.plot(s, pay, lw=2, color=col)
    ax.axvline(110, color=INK2, lw=0.8, ls="--")
    ax.text(111.5, 34, "K = $110", color="black", fontsize=7)
    ax.set_title(name, loc="left")
    ax.set_xlabel("Stock price at expiry ($)")
    grid_y(ax)
axes[0].set_ylabel("Payoff ($)")
fig.tight_layout()
save(fig, "d_payoffs")

# Simulated paths, published 2015 parameters, start = 1
p = {"v0": 0.0546, "kappa": 1.4554, "theta": 0.0580, "xi": 0.3927, "rho": -0.6518}
n_steps, n_paths, dt = 252, 12, 1 / 252
rng = np.random.default_rng(620)
S_ = np.ones((n_paths, n_steps + 1))
V = np.full((n_paths, n_steps + 1), p["v0"])
for t in range(n_steps):
    z1 = rng.standard_normal(n_paths)
    z2 = p["rho"] * z1 + np.sqrt(1 - p["rho"] ** 2) * rng.standard_normal(n_paths)
    v = np.maximum(V[:, t], 0)
    S_[:, t + 1] = S_[:, t] * np.exp((0.0015 - 0.5 * v) * dt + np.sqrt(v * dt) * z1)
    V[:, t + 1] = V[:, t] + p["kappa"] * (p["theta"] - v) * dt + p["xi"] * np.sqrt(v * dt) * z2
days = np.arange(n_steps + 1)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.25, 2.75), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
for i in range(n_paths):
    ax1.plot(days, S_[i], lw=0.9, color=BLUE, alpha=0.6)
    ax2.plot(days, np.sqrt(np.maximum(V[i], 0)) * 100, lw=0.9, color=ORANGE, alpha=0.6)
ax1.axhline(1, color=INK2, lw=0.7, ls="--")
ax1.set_ylabel("Price ($)")
ax2.set_ylabel("Volatility (%/yr)")
ax2.set_xlabel("Trading Days")
grid_y(ax1)
grid_y(ax2)
ax1.set_title("12 Simulated Stock Prices", loc="left")
fig.tight_layout(h_pad=0.4)
save(fig, "d_paths")

# 2015 engine check: published vs simulated, the 401 published puts (public)
rows = list(csv.DictReader(open(repo / "results/goog/market_comparison_2015.csv")))
mk = np.array([float(r["market_price"]) for r in rows])
md = np.array([float(r["model_price"]) for r in rows])
fig, ax = plt.subplots(figsize=(3.2, 2.9))
lim = max(mk.max(), md.max()) * 1.05
ax.plot([0, lim], [0, lim], color=INK2, lw=0.8, ls="--", alpha=0.4, label="Perfect match")
ax.scatter(mk, md, s=6, color=BLUE, edgecolor="white", linewidth=0.3, label=f"{len(rows)} published Google puts")
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel("Published Price ($)", color="black")
ax.set_ylabel("Current Model Price ($)", color="black")
ax.tick_params(colors="black")
ax.legend(frameon=False, loc="upper left", fontsize=7, labelcolor="black")
grid_y(ax)
fig.tight_layout()
save(fig, "d_2015")


# Headline, as two separate charts over the same 299 held-out contracts:
#   share_inside_spread    = share of contracts with bid <= model price <= ask
#   median_rel_price_error = median of |model price - mid| / mid, mid = (bid + ask) / 2
NOTE = "Comparison of outdated 2015 published parameters against\n2026 calibrated parameters on 2026 options contracts"


def bar_chart(name, title, vals, fmt, ymax, yticks):
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    x = np.arange(2)
    ax.bar(x, vals, width=0.6, color=[ORANGE, BLUE], edgecolor="white", linewidth=1)
    for i, v in enumerate(vals):
        ax.text(i, v + ymax * 0.02, fmt(v), ha="center", va="bottom", fontsize=11, weight="bold", color="black")
    ax.set_xticks(x, ["Published\nParameters", "Calibrated\nParameters"])
    ax.set_ylim(0, ymax)
    ax.set_yticks(yticks, [f"{t}%" for t in yticks])
    grid_y(ax)
    ax.set_title(title, loc="left")
    fig.text(0.5, 0.015, NOTE, ha="center", fontsize=5.5, color="black")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, name)


bar_chart("d_inside_spread", "Share of Options with Model Price\nMatching Market Bid-Ask Range",
          [0.0, b["share_inside_spread"] * 100],
          lambda v: f"{v:.0f}%", 100, [0, 25, 50, 75, 100])
bar_chart("d_price_gap", "Error of Model Prices vs Market Mid-Price",
          [a["median_rel_price_error"] * 100, b["median_rel_price_error"] * 100],
          lambda v: f"{v:.0f}%" if v >= 10 else f"{v:.1f}%", 100, [0, 25, 50, 75, 100])

# By maturity, calibrated set. One chart per ticker, each from its own market-test summary.
BUCKET_LABELS = {"under 3 months": "< 3\nmonths", "3-6 months": "3-6\nmonths",
                 "6-12 months": "6-12\nmonths", "over 12 months": "> 12\nmonths"}


def maturity_chart(name, rows, title):
    fig, ax = plt.subplots(figsize=(3.7, 2.75))
    x = np.arange(len(rows))
    vals = [r["share_inside_spread"] * 100 for r in rows]
    ax.bar(x, vals, width=0.62, color=BLUE, edgecolor="white", linewidth=1)
    for i, r in enumerate(rows):
        lo, hi = [v * 100 for v in r["ci95"]["share_inside_spread"]]
        ax.errorbar(i, vals[i], yerr=[[vals[i] - lo], [hi - vals[i]]], color=INK, capsize=3, lw=0.8)
        ax.text(i, hi + 2.5, f"{vals[i]:.0f}%", ha="center", va="bottom", fontsize=10, weight="bold", color="black")
        ax.text(i, 2, f"n={r['n']}", ha="center", va="bottom", fontsize=6.5, color="white")
    ax.set_xticks(x, [BUCKET_LABELS[r["bucket"]] for r in rows])
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    grid_y(ax)
    ax.set_xlabel("Time Until Expiry Date")
    ax.set_title(title, loc="left")
    fig.tight_layout()
    save(fig, name)


maturity_chart("d_maturity", res["calibrated_2026_10_13"]["by_maturity"],
               "Share of Options with Model Price Matching\nMarket Bid-Ask Range (2026 Parameters)")
for ticker in ("NVDA", "INTC", "AMD"):
    other = json.load(open(repo / f"results/{ticker.lower()}/market_test.json"))
    maturity_chart(f"d_maturity_{ticker.lower()}", other["results"]["calibrated_2026_10_13"]["by_maturity"],
                   f"Share of {ticker} Options with Model Price Matching\nMarket Bid-Ask Range (2026 Parameters)")

# Median |model price - mid| / mid on the held-out contracts, per ticker, 2026-10-13 calibrations
tickers = ["GOOG", "NVDA", "INTC", "AMD"]
gaps = [json.load(open(repo / f"results/{t.lower()}/market_test.json"))["results"]["calibrated_2026_10_13"]["overall"]
        ["median_rel_price_error"] * 100 for t in tickers]
fig, ax = plt.subplots(figsize=(3.7, 2.75))
x = np.arange(len(tickers))
ax.bar(x, gaps, width=0.62, color=BLUE, edgecolor="white", linewidth=1)
for i, v in enumerate(gaps):
    ax.text(i, v + 0.15, f"{v:.1f}%", ha="center", va="bottom", fontsize=10, weight="bold", color="black")
ax.set_xticks(x, tickers)
ax.set_ylim(0, 8)
ax.set_yticks([0, 2, 4, 6, 8], ["0%", "2%", "4%", "6%", "8%"])
grid_y(ax)
ax.set_title("Median Error of Model Prices vs\nMarket Mid-Price (2026 Parameters)", loc="left")
fig.tight_layout()
save(fig, "d_price_gap_tickers")

# Model implied-volatility surface from the 2026-10-13 GOOG fit. Model values only: market IV
# points are Restricted (DATA_CONTRACT.md section 4) and are never plotted.
document = json.load(open(repo / "results/goog/calibration.json"))
entry = document["assets"]["GOOG"]
if entry["ladder_step"] == 3:
    fit = entry["rejected_fit"]
else:
    fit = entry
params, r, q = fit["params"], document["r"], entry["q"]
ks = np.linspace(0.70, 1.30, 31)              # strike / spot S0
tenor = np.linspace(0.06, 1.40, 30)           # years
KS, TT = np.meshgrid(ks, tenor)
iv = model_implied_vols(params, KS.ravel(), TT.ravel(), r, q).reshape(KS.shape) * 100
fig = plt.figure(figsize=(3.9, 3.0))
ax = fig.add_subplot(projection="3d")
ax.plot_surface(KS, TT, iv, cmap="Blues", vmin=np.nanmin(iv) - 8, vmax=np.nanmax(iv),
                edgecolor="white", linewidth=0.15, antialiased=True)
ax.set_xlabel(r"Strike ($\times S_0$)", labelpad=-2)
ax.set_ylabel("Time to Expire (yr)", labelpad=-2)
ax.zaxis.set_rotate_label(False)
ax.set_zlabel(r"$\sigma$ (%/yr)", labelpad=-4, rotation=90)
ax.tick_params(pad=-2, labelsize=6.5)
ax.view_init(elev=22, azim=-128)
ax.set_box_aspect(None, zoom=0.92)
fig.suptitle(f"GOOG Heston Model IV Surface, {document['snapshot_date']}", fontsize=8, y=0.98)
fig.text(0.5, 0.875, "Model values only; no market quotes shown.", fontsize=6, color="black", ha="center")
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.93)
save(fig, "d_surface")
print("wrote 11 charts (svg, pdf, png) to", out)
