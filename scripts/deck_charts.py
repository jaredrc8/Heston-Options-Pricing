"""
Deck-sized charts for the presentation slides (6.5 x 4.5 in). Aggregates only, plus the
public 2015 published prices.

    .venv/bin/python scripts/deck_charts.py [repo] [output folder]

Both arguments are optional: repo defaults to the repository root and the output folder to
results/figures/presentation/. Replace the images in the slides with the new PNGs.

    d_payoffs.png   slide 3, Options Contract
    d_paths.png     slide 7, Monte Carlo Simulations
    d_2015.png      slide 9, 2015 parameters on 2015 prices
    d_headline.png  slide 10, 2015 vs 2026 parameters, today
    d_maturity.png  slide 11, Where the 2026 model struggles
    d_surface.png   calibration slide: the 2026-10-13 GOOG model IV surface (model values only)
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
mt = json.load(open(repo / "results/goog_market_test.json"))
pub = json.load(open(repo / "results/market_comparison.json"))
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "axes.titlesize": 8.5, "axes.titlecolor": INK,
})
DPI = 300
res = mt["results"]
a, b = res["published_2015"]["overall"], res["calibrated_2026_10_13"]["overall"]


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
    ax.text(111.5, 34, "K = $110", color=INK2, fontsize=7)
    ax.set_title(name, loc="left")
    ax.set_xlabel("Stock price at expiry ($)")
    grid_y(ax)
axes[0].set_ylabel("Payoff ($)")
fig.tight_layout()
fig.savefig(out / "d_payoffs.png", dpi=DPI)
plt.close(fig)

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
ax1.set_ylabel("Price (start = 1)")
ax2.set_ylabel("Swing (%/yr)")
ax2.set_xlabel("Trading days ahead")
grid_y(ax1)
grid_y(ax2)
ax1.set_title("12 simulated futures (of 100,000)", loc="left")
fig.tight_layout(h_pad=0.4)
fig.savefig(out / "d_paths.png", dpi=DPI)
plt.close(fig)

# 2015 engine check: published vs simulated, the 401 published puts (public)
rows = list(csv.DictReader(open(repo / "results/market_comparison_2015.csv")))
mk = np.array([float(r["market_price"]) for r in rows])
md = np.array([float(r["model_price"]) for r in rows])
fig, ax = plt.subplots(figsize=(3.2, 2.9))
lim = max(mk.max(), md.max()) * 1.05
ax.plot([0, lim], [0, lim], color=INK2, lw=0.8, ls="--", label="Perfect match")
ax.scatter(mk, md, s=6, color=BLUE, edgecolor="white", linewidth=0.3, label=f"{len(rows)} published Google puts")
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel("Published price, Feb 2015 ($)")
ax.set_ylabel("Our model's price ($)")
ax.legend(frameon=False, loc="upper left", fontsize=7)
grid_y(ax)
fig.tight_layout()
fig.savefig(out / "d_2015.png", dpi=DPI)
plt.close(fig)


# Headline: two small multiples, one scale each
def bar_panel(ax, title, vals, ci, fmt, ymax):
    x = np.arange(2)
    ax.bar(x, vals, width=0.6, color=[ORANGE, BLUE], edgecolor="white", linewidth=1)
    for i, (v, c) in enumerate(zip(vals, ci)):
        if c is not None:
            ax.errorbar(i, v, yerr=[[v - c[0]], [c[1] - v]], color=INK, capsize=3, lw=0.8)
        top = c[1] if c is not None else v
        ax.text(i, top + ymax * 0.03, fmt(v), ha="center", va="bottom", fontsize=12, weight="bold", color=INK)
    ax.set_xticks(x, ["Published\nParameters", "Calibrated\nParameters"])
    ax.set_ylim(0, ymax)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title(title, pad=6, loc="center")


def pc(d, k):
    return [v * 100 for v in d["ci95"][k]]


# Two panels over the same 299 held-out contracts:
#   share_inside_spread    = share of contracts with bid <= model price <= ask
#   median_rel_price_error = median of |model price - mid| / mid, mid = (bid + ask) / 2
fig, axes = plt.subplots(1, 2, figsize=(5.9, 2.3))
bar_panel(axes[0], "Share of Options with Model Price\nMatching Market Bid-Ask Range",
          [0.0, b["share_inside_spread"] * 100],
          [None, pc(b, "share_inside_spread")], lambda v: f"{v:.0f}%", 100)
bar_panel(axes[1], "Error of Model Prices vs Market Mid-Price",
          [a["median_rel_price_error"] * 100, b["median_rel_price_error"] * 100],
          [pc(a, "median_rel_price_error"), pc(b, "median_rel_price_error")],
          lambda v: f"{v:.0f}%" if v >= 10 else f"{v:.1f}%", 85)
fig.text(0.5, 0.015,
         "Comparison of outdated 2015 published parameters against 2026 calibrated parameters on 2026 options contracts",
         ha="center", fontsize=6.5, color=INK2)
fig.tight_layout(w_pad=2.0, rect=(0, 0.07, 1, 1))
fig.savefig(out / "d_headline.png", dpi=DPI)
plt.close(fig)

# By maturity, calibrated set
rows = res["calibrated_2026_10_13"]["by_maturity"]
fig, ax = plt.subplots(figsize=(3.7, 2.75))
x = np.arange(len(rows))
vals = [r["share_inside_spread"] * 100 for r in rows]
ax.bar(x, vals, width=0.62, color=BLUE, edgecolor="white", linewidth=1)
for i, r in enumerate(rows):
    lo, hi = pc(r, "share_inside_spread")
    ax.errorbar(i, vals[i], yerr=[[vals[i] - lo], [hi - vals[i]]], color=INK, capsize=3, lw=0.8)
    ax.text(i, hi + 2.5, f"{vals[i]:.0f}%", ha="center", va="bottom", fontsize=10, weight="bold", color=INK)
    ax.text(i, 2, f"n={r['n']}", ha="center", va="bottom", fontsize=6.5, color="white")
ax.set_xticks(x, ["< 3\nmonths", "3-6\nmonths", "6-12\nmonths", "> 12\nmonths"])
ax.set_ylim(0, 108)
ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
grid_y(ax)
ax.set_xlabel("Time Until Expiry Date")
ax.set_title("Share of Options with Model Price Matching\nMarket Bid-Ask Range (2026 Parameters)", loc="left")
fig.tight_layout()
fig.savefig(out / "d_maturity.png", dpi=DPI)
plt.close(fig)

# Model implied-volatility surface from the 2026-10-13 GOOG fit. Model values only: market IV
# points are Restricted (DATA_CONTRACT.md section 4) and are never plotted.
document = json.load(open(repo / "results/calibration_goog/calibration.json"))
entry = document["assets"]["GOOG"]
if entry["ladder_step"] == 3:
    fit = entry["rejected_fit"]
else:
    fit = entry
params, r, q = fit["params"], document["r"], entry["q"]
kf = np.linspace(0.70, 1.30, 31)              # strike / forward, the test population's range
tenor = np.linspace(0.06, 1.40, 30)           # years
KF, TT = np.meshgrid(kf, tenor)
k_over_s = (KF * np.exp((r - q) * TT)).ravel()
iv = model_implied_vols(params, k_over_s, TT.ravel(), r, q).reshape(KF.shape) * 100
fig = plt.figure(figsize=(3.9, 3.0))
ax = fig.add_subplot(projection="3d")
ax.plot_surface(KF, TT, iv, cmap="Blues", vmin=np.nanmin(iv) - 8, vmax=np.nanmax(iv),
                edgecolor="white", linewidth=0.15, antialiased=True)
ax.set_xlabel("Strike / forward", labelpad=-2)
ax.set_ylabel("Years to expiry", labelpad=-2)
ax.set_zlabel("Implied vol (%)", labelpad=-4)
ax.tick_params(pad=-2, labelsize=6.5)
ax.view_init(elev=22, azim=-128)
ax.set_box_aspect(None, zoom=0.92)
fig.suptitle(f"GOOG Heston Model IV Surface, {document['snapshot_date']}", fontsize=8, y=0.98)
fig.text(0.5, 0.875, "Model values only; no market quotes shown.", fontsize=6, color=INK2, ha="center")
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.93)
fig.savefig(out / "d_surface.png", dpi=DPI)
plt.close(fig)
print("wrote 6 charts to", out)
