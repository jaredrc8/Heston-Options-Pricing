"""
Least-Squares Monte Carlo (LSMC) pricing of a Bermudan worst-of put under
the multi-asset Heston model.

Model (risk-neutral), for each asset i:
    dS_i = (r - q) S_i dt + sqrt(v_i) S_i dW^S_i
    dv_i = kappa_i (theta_i - v_i) dt + xi_i sqrt(v_i) dW^v_i
    corr(dW^S_i, dW^v_i) = rho_i        corr(dW^S_i, dW^S_j) = R_ij

The worst-of put pays  max(K - min_i S_i(t) / S_i(0), 0)  on the exercise
date t chosen by the holder.

The Heston parameters of one asset are kept in a dictionary, e.g.
    {"v0": 0.04, "kappa": 1.5, "theta": 0.04, "xi": 0.5, "rho": -0.7}

References: Longstaff & Schwartz (2001); Glasserman (2004);
Lord, Koekkoek & van Dijk (2010).
"""

from itertools import combinations_with_replacement

import numpy as np


# ----------------------------------------------------------------------
# 1. Simulating the assets
# ----------------------------------------------------------------------

def simulate_heston_paths(S0, params, corr, r, q, times, substeps, n_paths, rng):
    """
    Simulate correlated Heston assets.

    S0       : starting prices, one per asset
    params   : list of Heston parameter dictionaries, one per asset
    corr     : correlation matrix of the asset prices (d x d)
    r, q     : risk-free rate and dividend yield
    times    : 0 followed by the exercise dates, e.g. [0, 0.25, 0.5, 0.75, 1]
    substeps : number of time steps between two exercise dates
    n_paths  : number of paths
    rng      : random number generator, e.g. np.random.default_rng(42)

    Returns S and v, each of shape (len(times), n_paths, d): the prices and
    variances on every date in `times`.

    Each time step uses full-truncation Euler for the variance and log-Euler
    for the price. The variance can go slightly negative, so max(v, 0) is used
    wherever it enters a formula. This works even when the Feller condition
    2 kappa theta > xi^2 fails, which calibrated parameters often do.
    """
    S0 = np.asarray(S0, dtype=float)
    n_assets = len(S0)
    v0 = np.array([p["v0"] for p in params])
    kappa = np.array([p["kappa"] for p in params])
    theta = np.array([p["theta"] for p in params])
    xi = np.array([p["xi"] for p in params])
    rho = np.array([p["rho"] for p in params])

    # Cholesky factor L of the correlation matrix: if e holds independent
    # standard normals, then e @ L.T has correlation matrix corr.
    L = np.linalg.cholesky(corr)

    S = np.zeros((len(times), n_paths, n_assets))
    v = np.zeros((len(times), n_paths, n_assets))
    S[0] = S0
    v[0] = v0

    log_s = np.log(S[0])
    var = v[0].copy()

    for k in range(1, len(times)):
        dt = (times[k] - times[k - 1]) / substeps
        for _ in range(substeps):
            # Correlated shocks for the prices
            z_price = rng.standard_normal((n_paths, n_assets)) @ L.T
            # Shock for each variance: correlation rho_i with its own price
            z_var = rho * z_price + np.sqrt(1 - rho**2) * rng.standard_normal((n_paths, n_assets))

            v_pos = np.maximum(var, 0.0)
            log_s = log_s + (r - q - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * z_price
            var = var + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos * dt) * z_var

        S[k] = np.exp(log_s)
        v[k] = var

    return S, v


# ----------------------------------------------------------------------
# 2. The contract
# ----------------------------------------------------------------------

def worst_of_put_payoff(S, S0, K):
    """
    Payoff max(K - min_i S_i / S0_i, 0) of every path on one date.
    S has shape (n_paths, d).

    With a single asset, multiply a price by S0 to get the vanilla put
    with strike K * S0.
    """
    worst = np.min(S / S0, axis=1)
    return np.maximum(K - worst, 0.0)


# ----------------------------------------------------------------------
# 3. Regression for the continuation value
# ----------------------------------------------------------------------

def regression_features(S, v, S0):
    """
    The state used to estimate the continuation value: each asset's
    performance S_i / S0_i, each asset's variance, and (with more than one
    asset) the worst performance, which is what the payoff depends on.
    """
    x = S / S0
    features = [x, np.maximum(v, 0.0)]
    if x.shape[1] > 1:
        features.append(x.min(axis=1, keepdims=True))
    return np.hstack(features)


def polynomial_basis(features, degree):
    """
    A column of ones plus every product of up to `degree` features.
    For example, degree 2 with features (a, b) gives: 1, a, b, a*a, a*b, b*b.
    """
    n_rows, n_features = features.shape
    columns = [np.ones(n_rows)]
    for power in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n_features), power):
            columns.append(np.prod(features[:, list(combo)], axis=1))
    return np.column_stack(columns)


def fit_regression(features, y, degree):
    """
    Least-squares fit of y on the polynomial basis.

    Each feature is first standardized (minus its mean, divided by its
    standard deviation). This does not change the fitted values, but it keeps
    the least-squares problem well conditioned.

    Returns None if there are fewer than 2 data points per basis term,
    which is too few for a reliable fit.
    """
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1.0  # a constant feature: leave it unscaled

    X = polynomial_basis((features - mean) / std, degree)
    if X.shape[0] < 2 * X.shape[1]:
        return None

    # rcond=1e-10 drops basis columns that duplicate others (for example
    # when two assets are identical) instead of fitting round-off noise.
    coef = np.linalg.lstsq(X, y, rcond=1e-10)[0]
    return {"mean": mean, "std": std, "coef": coef}


def evaluate_regression(fit, features, degree):
    """Continuation value predicted by a fitted regression."""
    X = polynomial_basis((features - fit["mean"]) / fit["std"], degree)
    return X @ fit["coef"]


# ----------------------------------------------------------------------
# 4. Least-Squares Monte Carlo
# ----------------------------------------------------------------------

def lsmc_fit(S, v, times, S0, K, r, degree):
    """
    Longstaff-Schwartz backward induction: learn when to exercise.

    Starting at maturity and stepping back one exercise date at a time:
      1. discount each path's future cash flow back to this date
      2. on in-the-money paths, regress that cash flow on the current state;
         the fitted value is the estimated continuation value C
      3. exercise where the immediate payoff h is larger than C

    Returns
      fits            : the regression for each date (None = never exercise there)
      in_sample_price : the price on these same paths; usually slightly too
                        high, because the exercise rule was fitted to them
    """
    n_dates = len(times)
    fits = [None] * n_dates

    # cash[p] = the cash flow of path p, valued at the current date
    cash = worst_of_put_payoff(S[-1], S0, K)

    for k in range(n_dates - 2, 0, -1):
        cash = cash * np.exp(-r * (times[k + 1] - times[k]))

        h = worst_of_put_payoff(S[k], S0, K)
        itm = np.where(h > 0)[0]
        if len(itm) == 0:
            continue

        features = regression_features(S[k, itm], v[k, itm], S0)
        fit = fit_regression(features, cash[itm], degree)
        if fit is None:
            continue

        continuation = evaluate_regression(fit, features, degree)
        exercise = itm[h[itm] > continuation]
        cash[exercise] = h[exercise]
        fits[k] = fit

    cash = cash * np.exp(-r * times[1])
    return fits, cash.mean()


def lsmc_apply(S, v, times, S0, K, r, fits, degree):
    """
    Use fitted regressions on a new set of paths: each path is exercised on
    the first date where its payoff beats the estimated continuation value.

    Returns each path's payoff discounted to t = 0.
    """
    n_dates = len(times)
    n_paths = S.shape[1]
    discounted_payoff = np.zeros(n_paths)
    alive = np.ones(n_paths, dtype=bool)  # paths not exercised yet

    for k in range(1, n_dates):
        h = worst_of_put_payoff(S[k], S0, K)
        exercise = np.where(alive & (h > 0))[0]

        # Before maturity, only exercise where h beats the continuation value
        if k < n_dates - 1:
            if fits[k] is None or len(exercise) == 0:
                continue
            features = regression_features(S[k, exercise], v[k, exercise], S0)
            continuation = evaluate_regression(fits[k], features, degree)
            exercise = exercise[h[exercise] > continuation]

        discounted_payoff[exercise] = h[exercise] * np.exp(-r * times[k])
        alive[exercise] = False

    return discounted_payoff


def price_bermudan_worst_of_put(S0, params, corr, r, q, K, exercise_times,
                                n_paths, substeps=4, degree=2, seed=None):
    """
    Price the Bermudan worst-of put with LSMC.

    Two independent sets of paths are simulated. The exercise rule is fitted
    on the first set and the price is computed on the second, so the price is
    not biased upward by reusing the same paths (Glasserman 2004, Sec. 8.6).
    The European put (exercise at maturity only) is priced on the same second
    set, so the early-exercise premium = Bermudan - European is estimated
    path by path, with its own standard error.

    exercise_times : exercise dates, e.g. np.arange(1, 51) / 50 for 50 dates in a year
    seed           : the same seed always gives the same result

    Returns a dictionary of prices and standard errors.
    """
    S0 = np.asarray(S0, dtype=float)
    rng = np.random.default_rng(seed)
    times = np.concatenate(([0.0], exercise_times))

    # 1. Fit the exercise rule on a training set of paths
    S, v = simulate_heston_paths(S0, params, corr, r, q, times, substeps, n_paths, rng)
    fits, in_sample_price = lsmc_fit(S, v, times, S0, K, r, degree)

    # 2. Price on a new, independent set of paths
    S, v = simulate_heston_paths(S0, params, corr, r, q, times, substeps, n_paths, rng)
    bermudan = lsmc_apply(S, v, times, S0, K, r, fits, degree)
    european = worst_of_put_payoff(S[-1], S0, K) * np.exp(-r * times[-1])
    premium = bermudan - european

    return {
        "price": bermudan.mean(),
        "std_error": bermudan.std(ddof=1) / np.sqrt(n_paths),
        "european_price": european.mean(),
        "european_std_error": european.std(ddof=1) / np.sqrt(n_paths),
        "premium": premium.mean(),
        "premium_std_error": premium.std(ddof=1) / np.sqrt(n_paths),
        "in_sample_price": in_sample_price,
    }


if __name__ == "__main__":
    S0 = [100.0, 50.0, 80.0]
    params = [{"v0": 0.04, "kappa": 1.5, "theta": 0.04, "xi": 0.5, "rho": -0.7}] * 3
    corr = np.array([[1.0, 0.5, 0.4],
                     [0.5, 1.0, 0.3],
                     [0.4, 0.3, 1.0]])
    T = 1.0
    n_dates = 50
    exercise_times = np.arange(1, n_dates + 1) * T / n_dates

    result = price_bermudan_worst_of_put(S0, params, corr, r=0.05, q=0.0, K=1.0,
                                         exercise_times=exercise_times, n_paths=100_000,
                                         substeps=4, degree=2, seed=2026)
    for name, value in result.items():
        print(f"{name:20s} {value:.5f}")
