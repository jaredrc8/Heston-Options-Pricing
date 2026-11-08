"""
Two-pass Least-Squares Monte Carlo pricing of the Bermudan put (SPEC.md section 8).

The exercise rule is fitted on regression set A and the price is computed on an
independent evaluation set B, so the reported price is low-biased.
"""

from itertools import combinations_with_replacement
from math import comb

import numpy as np

from hestonpricer.errors import InputError, require
from hestonpricer.simulate import DEFAULT_SIM, read_contract, simulate


def payoff(S, strike):
    """max(K - min_i S_i(t) / S_i(0), 0). S has shape (n_paths, d) and S(0) = 1."""
    return np.maximum(strike - np.min(S, axis=1), 0.0)


def regression_features(S, nu):
    """The regression state (S/S0, nu_plus) for one asset, one row per path."""
    return np.column_stack([S[:, 0], np.maximum(nu[:, 0], 0.0)])


def polynomial_basis(x, degree):
    """A column of ones plus every product of up to `degree` columns of x."""
    n_rows, n_columns = x.shape
    columns = [np.ones(n_rows)]
    for power in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n_columns), power):
            column = np.ones(n_rows)
            for j in combo:
                column = column * x[:, j]
            columns.append(column)
    return np.column_stack(columns)


def continuation_value(date_fit, features, degree):
    """Estimated continuation value, using the scaling stored from set A."""
    keep = date_fit["keep"]
    x = (features[:, keep] - date_fit["mean"][keep]) / date_fit["std"][keep]
    return polynomial_basis(x, degree) @ date_fit["coef"]


def fit_policy(paths, contract, market, basis_degree):
    """Backward induction on set A. Returns a policy dictionary (SPEC.md 10.1)."""
    require(basis_degree in (1, 2, 3, 4), "basis_degree must be 1, 2, 3 or 4")
    S, nu = paths["S"], paths["nu"]
    # The three-asset basis belongs to EXT-A, which may start only after gate G1.
    if S.shape[2] != 1:
        raise InputError("LSMC is single-asset until EXT-A opens (SPEC.md 2.2)")

    strike, maturity, M = read_contract(contract)
    r = float(market["r"])
    times = np.arange(M + 1, dtype=float) * maturity / M
    n_paths = S.shape[0]

    # Smallest number of in-the-money paths for a fit: max(50, 10 p)
    n_features = 2
    p = comb(n_features + basis_degree, basis_degree)
    n_min = max(50, 10 * p)

    # Cash flow of each path and the time it is received
    cash = payoff(S[:, M], strike)
    tau = np.full(n_paths, maturity, dtype=float)

    dates = {}
    skipped = 0
    rank_deficient = 0
    min_itm = n_paths if M > 1 else 0

    for k in range(M - 1, 0, -1):
        g = payoff(S[:, k], strike)
        itm = np.flatnonzero(g > 0)
        min_itm = min(min_itm, len(itm))
        if len(itm) < n_min:
            skipped += 1
            continue

        # Standardize, and drop features that are constant (for example nu when xi = 0)
        features = regression_features(S[itm, k], nu[itm, k])
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        keep = std >= 1e-12
        x = (features[:, keep] - mean[keep]) / std[keep]
        X = polynomial_basis(x, basis_degree)

        # Regress the discounted future cash flow on the current state
        y = np.exp(-r * (tau[itm] - times[k])) * cash[itm]
        coef, _, rank, _ = np.linalg.lstsq(X, y, rcond=1e-10)
        if rank < X.shape[1]:
            rank_deficient += 1

        # Exercise where the payoff beats continuation. A tie continues.
        exercise = itm[g[itm] > X @ coef]
        cash[exercise] = g[exercise]
        tau[exercise] = times[k]

        dates[k] = {"coef": coef, "keep": keep, "mean": mean, "std": std}

    return {
        "dates": dates,
        "basis_degree": basis_degree,
        "in_sample_bermudan": float(np.mean(np.exp(-r * tau) * cash)),
        "skipped_dates": skipped,
        "rank_deficient_dates": rank_deficient,
        "min_itm_count": int(min_itm),
    }


def evaluate_policy(policy, paths, contract, market):
    """Apply a fitted policy to set B. Returns a price_result dictionary (SPEC.md 10.1)."""
    S, nu = paths["S"], paths["nu"]
    strike, maturity, M = read_contract(contract)
    r = float(market["r"])
    times = np.arange(M + 1, dtype=float) * maturity / M
    n_paths = S.shape[0]

    g_paid = np.zeros(n_paths)
    tau = np.full(n_paths, maturity, dtype=float)
    alive = np.ones(n_paths, dtype=bool)
    exercise_count = np.zeros(M)

    for k in range(1, M):
        if k not in policy["dates"]:
            continue
        g = payoff(S[:, k], strike)
        candidates = np.flatnonzero(alive & (g > 0))
        if len(candidates) == 0:
            continue
        features = regression_features(S[candidates, k], nu[candidates, k])
        continuation = continuation_value(policy["dates"][k], features, policy["basis_degree"])
        exercise = candidates[g[candidates] > continuation]
        g_paid[exercise] = g[exercise]
        tau[exercise] = times[k]
        alive[exercise] = False
        exercise_count[k - 1] = len(exercise)

    # Paths still alive receive the payoff at maturity
    g_final = payoff(S[:, M], strike)
    g_paid[alive] = g_final[alive]
    exercise_count[M - 1] = np.sum(alive & (g_final > 0))

    bermudan = np.exp(-r * tau) * g_paid
    european = np.exp(-maturity * r) * g_final
    premium = bermudan - european
    root_n = np.sqrt(n_paths)

    return {
        "bermudan": float(bermudan.mean()),
        "se_bermudan": float(bermudan.std(ddof=1) / root_n),
        "european": float(european.mean()),
        "se_european": float(european.std(ddof=1) / root_n),
        "premium": float(premium.mean()),
        "se_premium": float(premium.std(ddof=1) / root_n),
        "in_sample_bermudan": policy["in_sample_bermudan"],
        "exercise_fraction": [float(c) / n_paths for c in exercise_count],
        "skipped_dates": policy["skipped_dates"],
        "rank_deficient_dates": policy["rank_deficient_dates"],
        "min_itm_count": policy["min_itm_count"],
        "n_paths": n_paths,
    }


def price_bermudan(params, corr, market, contract, sim):
    """Fit on one independent stream and price on another. Returns a price_result."""
    settings = {**DEFAULT_SIM, **sim}
    child_a, child_b = np.random.SeedSequence(settings["master_seed"]).spawn(2)

    paths_a = simulate(params, corr, market, contract, settings,
                       np.random.Generator(np.random.PCG64(child_a)))
    policy = fit_policy(paths_a, contract, market, settings["basis_degree"])
    del paths_a  # free the memory before simulating the second set

    paths_b = simulate(params, corr, market, contract, settings,
                       np.random.Generator(np.random.PCG64(child_b)))
    return evaluate_policy(policy, paths_b, contract, market)
