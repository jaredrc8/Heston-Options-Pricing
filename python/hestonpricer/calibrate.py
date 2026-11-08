"""
Multi-start least-squares calibration in implied-volatility space (SPEC.md 6.5 to 6.7).

The grid is Restricted when it comes from licensed data. Everything returned here
is aggregated (parameters, counts, RMS by bucket), so the result is Internal.
No per-contract residual ever leaves this module.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc

from hestonpricer.analytic import heston_european_many
from hestonpricer.errors import HestonError
from hestonpricer.iv import bs_implied_vol_array

PARAM_NAMES = ["v0", "kappa", "theta", "xi", "rho"]
BOUNDS = {"v0": (1e-4, 4.0), "kappa": (0.01, 20.0), "theta": (1e-4, 4.0),
          "xi": (0.01, 5.0), "rho": (-0.999, 0.999)}
DEFAULT_CFG = {"seed": 620000, "n_starts": 8, "fixed": {}}


def model_implied_vols(params, k_over_s, tenors, r, q):
    """Heston implied vol at each point, with S = 1. nan where it cannot be inverted."""
    vols = np.full(len(k_over_s), np.nan)
    for tenor in np.unique(tenors):
        same = tenors == tenor
        strikes = k_over_s[same]
        forward = np.exp((r - q) * tenor)
        is_call = strikes >= forward    # calls for K >= F, puts otherwise
        try:
            prices = heston_european_many(is_call, 1.0, strikes, tenor, params, r, q)
        except HestonError:
            continue
        vols[same] = bs_implied_vol_array(is_call, prices, 1.0, strikes, tenor, r, q)
    return vols


def residual_buckets(k_over_f, tenors, residuals):
    """RMS residual by tenor and moneyness, keeping only buckets with n >= 5."""
    tenor_groups = [("T<0.5", tenors < 0.5),
                    ("0.5<=T<1", (tenors >= 0.5) & (tenors < 1.0)),
                    ("T>=1", tenors >= 1.0)]
    moneyness_groups = [("K/F<0.95", k_over_f < 0.95),
                        ("0.95<=K/F<=1.05", (k_over_f >= 0.95) & (k_over_f <= 1.05)),
                        ("K/F>1.05", k_over_f > 1.05)]
    buckets = []
    for tenor_name, tenor_mask in tenor_groups:
        for moneyness_name, moneyness_mask in moneyness_groups:
            mask = tenor_mask & moneyness_mask
            if mask.sum() >= 5:
                buckets.append({"tenor_bucket": tenor_name, "moneyness_bucket": moneyness_name,
                                "n": int(mask.sum()),
                                "rms": float(np.sqrt(np.mean(residuals[mask] ** 2)))})
    return buckets


def is_stable(accepted, free_names):
    """Stable if at least 3 accepted starts and each parameter spreads by at most 10%."""
    if len(accepted) < 3:
        return False
    for name in free_names:
        values = np.array([start["params"][name] for start in accepted])
        if (values.max() - values.min()) / abs(np.median(values)) > 0.10:
            return False
    return True


def calibrate_asset(grid, r, q, cfg):
    """
    Fit one asset. grid needs columns k_over_s, k_over_f, tenor, iv, expiration.
    cfg may have keys seed, n_starts and fixed (for example {"kappa": 2.0}).
    Returns a dictionary with the section 12.5 asset fields.
    Raises E_CALIB_FAILED if no start converges.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    fixed = cfg["fixed"]
    k_over_s = grid["k_over_s"].to_numpy(dtype=float)
    k_over_f = grid["k_over_f"].to_numpy(dtype=float)
    tenors = grid["tenor"].to_numpy(dtype=float)
    market_iv = grid["iv"].to_numpy(dtype=float)

    free_names = [name for name in PARAM_NAMES if name not in fixed]
    lower = np.array([BOUNDS[name][0] for name in free_names])
    upper = np.array([BOUNDS[name][1] for name in free_names])

    def to_params(x):
        params = dict(fixed)
        for name, value in zip(free_names, x):
            params[name] = float(value)
        return params

    uninvertible = {"events": 0}

    def residuals(x):
        model_iv = model_implied_vols(to_params(x), k_over_s, tenors, r, q)
        bad = ~np.isfinite(model_iv)
        uninvertible["events"] += int(bad.sum())
        return np.where(bad, 1.0, market_iv - model_iv)

    # Starting points: scrambled Sobol sequence scaled to the bounds
    sobol = qmc.Sobol(d=len(free_names), scramble=True, rng=cfg["seed"])
    starts = qmc.scale(sobol.random(cfg["n_starts"]), lower, upper)

    results = []
    for i, x0 in enumerate(starts):
        fit = least_squares(residuals, x0, bounds=(lower, upper), method="trf", x_scale="jac",
                            ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=200)
        results.append({"start": i, "converged": bool(fit.status > 0), "status": int(fit.status),
                        "rms_iv_residual": float(np.sqrt(np.mean(fit.fun**2))),
                        "params": to_params(fit.x)})

    converged = [item for item in results if item["converged"]]
    if not converged:
        raise HestonError("E_CALIB_FAILED", "no starting point converged")

    best = min(converged, key=lambda item: item["rms_iv_residual"])
    cutoff = max(1.05 * best["rms_iv_residual"], 1e-8)
    accepted = [item for item in converged if item["rms_iv_residual"] <= cutoff]

    params = best["params"]
    final_residuals = residuals(np.array([params[name] for name in free_names]))

    # Relative spread of each free parameter over the accepted starts (section 6.6)
    spread = {}
    for name in free_names:
        values = np.array([item["params"][name] for item in accepted])
        spread[name] = float((values.max() - values.min()) / abs(np.median(values)))

    return {
        "params": {name: params[name] for name in PARAM_NAMES},
        "fixed_params": sorted(fixed),
        "feller_ratio": 2 * params["kappa"] * params["theta"] / params["xi"] ** 2,
        "grid_size": int(len(grid)),
        "n_expiries": int(grid["expiration"].nunique()),
        "rms_iv_residual": best["rms_iv_residual"],
        "stable": is_stable(accepted, free_names),
        "n_accepted_starts": len(accepted),
        "accepted_spread": spread,
        "uninvertible_events": uninvertible["events"],
        "starts": [{key: item[key] for key in
                    ("start", "converged", "status", "rms_iv_residual")} for item in results],
        "residual_buckets": residual_buckets(k_over_f, tenors, final_residuals),
    }


def fit_acceptable(result):
    """A fit is usable if it is stable with RMS IV residual at most 0.01."""
    return result is not None and result["stable"] and result["rms_iv_residual"] <= 0.01


def calibrate_with_ladder(grid, r, q, fallback_kappa, cfg=None):
    """
    Apply the failure ladder (SPEC.md 6.7). fallback_kappa is the cited value from
    configs/published_params.yaml, or None if the owner has not entered it.
    Returns the fit dictionary with ladder_step added. At step 3 there are no params.
    """
    cfg = cfg or {}

    def attempt(fixed):
        try:
            return calibrate_asset(grid, r, q, {**cfg, "fixed": fixed})
        except HestonError as error:
            if error.code == "E_CALIB_FAILED":
                return None
            raise

    full_fit = attempt({})
    if fit_acceptable(full_fit):
        return {**full_fit, "ladder_step": 1}

    fixed_fit = None
    if fallback_kappa is not None:
        fixed_fit = attempt({"kappa": float(fallback_kappa)})
        if fit_acceptable(fixed_fit):
            return {**fixed_fit, "ladder_step": 2}

    # No accepted fit. The rejected fits are kept apart from "params", for diagnosis only.
    return {"ladder_step": 3, "params": None, "stable": False, "fixed_params": [],
            "rejected_full_fit": full_fit, "rejected_fixed_kappa_fit": fixed_fit}
