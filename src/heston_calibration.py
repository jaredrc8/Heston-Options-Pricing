"""
Calibrate one asset's Heston parameters to its implied-volatility surface
(proposal Eq. 4):

    minimize   sum over quotes of  (heston_iv(K, T; params) - market_iv(K, T))^2

using scipy.optimize.least_squares, from several starting points. If the
starting points end at different answers, the fit is unstable (a fallback
trigger in the proposal).

Before calibrating, remove market quotes with tiny prices (far out of the
money): their implied vols are unreliable in the market and in the pricer.
"""

import numpy as np
from scipy.optimize import least_squares

from heston_closed_form import heston_implied_vol

PARAM_NAMES = ["v0", "kappa", "theta", "xi", "rho"]

# (lower, upper) bound for each parameter
BOUNDS = {
    "v0": (1e-4, 2.0),
    "kappa": (1e-3, 20.0),
    "theta": (1e-4, 2.0),
    "xi": (1e-2, 5.0),
    "rho": (-0.999, 0.999),
}


def model_implied_vols(params, S0, strikes, maturities, r, q):
    """Heston implied vol at every quote. The pricer works one maturity at a time."""
    vols = np.zeros(len(strikes))
    for T in np.unique(maturities):
        same_T = maturities == T
        vols[same_T] = heston_implied_vol(params, S0, strikes[same_T], T, r, q)
    return vols


def starting_points(market_vols, n_starts, seed):
    """
    The first start is a typical guess: v0 = theta = (median market vol)^2.
    The others are random draws around it, so the fit is tried from
    different places.
    """
    var = np.median(market_vols) ** 2
    rng = np.random.default_rng(seed)
    starts = [{"v0": var, "kappa": 1.5, "theta": var, "xi": 0.5, "rho": -0.5}]
    for _ in range(n_starts - 1):
        starts.append({
            "v0": var * rng.uniform(0.5, 2.0),
            "kappa": rng.uniform(0.2, 8.0),
            "theta": var * rng.uniform(0.5, 2.0),
            "xi": rng.uniform(0.1, 2.0),
            "rho": rng.uniform(-0.9, 0.3),
        })
    return starts


def calibrate_heston(S0, r, q, strikes, maturities, market_vols, fixed=None, n_starts=8, seed=0):
    """
    Fit Heston parameters to market implied vols.

    strikes, maturities, market_vols : one entry per quote
    fixed    : parameters held constant, e.g. {"kappa": 1.5}
    n_starts : number of starting points for the optimizer
    seed     : seed for the random starting points

    Returns a dictionary:
      params      best-fit parameters
      rms_error   root-mean-square implied-vol error
      max_error   largest absolute implied-vol error
      model_vols  Heston implied vols at the quotes
      all_starts  (cost, params) for every starting point, best first
    """
    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    market_vols = np.asarray(market_vols, dtype=float)
    if fixed is None:
        fixed = {}

    # The optimizer only sees the free parameters, as an array x.
    free = [name for name in PARAM_NAMES if name not in fixed]
    lower = [BOUNDS[name][0] for name in free]
    upper = [BOUNDS[name][1] for name in free]

    def to_params(x):
        """Combine the free values x with the fixed values into one dictionary."""
        params = dict(fixed)
        for name, value in zip(free, x):
            params[name] = float(value)
        return params

    def residuals(x):
        try:
            model = model_implied_vols(to_params(x), S0, strikes, maturities, r, q)
        except ValueError:
            # Extreme trial parameters (e.g. tiny variance with huge xi) that
            # the pricer cannot handle: count every quote as zero vol.
            model = np.zeros(len(market_vols))
        # NaN means this trial's model price is too small to invert. Count it
        # as zero vol too: a large error that pushes the optimizer away.
        model = np.nan_to_num(model, nan=0.0)
        return model - market_vols

    all_starts = []
    for start in starting_points(market_vols, n_starts, seed):
        x0 = [np.clip(start[name], BOUNDS[name][0], BOUNDS[name][1]) for name in free]
        fit = least_squares(residuals, x0, bounds=(lower, upper), x_scale="jac",
                            ftol=1e-12, xtol=1e-12, gtol=1e-12)
        cost = np.sum(fit.fun**2)
        all_starts.append((cost, to_params(fit.x)))

    all_starts.sort(key=lambda item: item[0])
    best = all_starts[0][1]
    model_vols = model_implied_vols(best, S0, strikes, maturities, r, q)
    errors = model_vols - market_vols

    return {
        "params": best,
        "rms_error": np.sqrt(np.mean(errors**2)),
        "max_error": np.max(np.abs(errors)),
        "model_vols": model_vols,
        "all_starts": all_starts,
    }


if __name__ == "__main__":
    # Make a surface from known parameters, then check that calibration finds them
    true_params = {"v0": 0.04, "kappa": 2.0, "theta": 0.05, "xi": 0.6, "rho": -0.7}
    S0, r, q = 100.0, 0.05, 0.0
    strikes = np.tile([80.0, 90.0, 100.0, 110.0, 120.0], 3)
    maturities = np.repeat([0.25, 0.5, 1.0], 5)
    market_vols = model_implied_vols(true_params, S0, strikes, maturities, r, q)

    result = calibrate_heston(S0, r, q, strikes, maturities, market_vols)
    print("true:      ", true_params)
    print("calibrated:", {name: round(value, 6) for name, value in result["params"].items()})
    print("rms error: ", result["rms_error"])
