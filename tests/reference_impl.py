"""
An independent scalar LSMC for V-07. It is written path by path with plain Python
floats, shares no code with hestonpricer, and lists the polynomial terms in a
different order. Only numpy.linalg.lstsq is common to both, as SPEC.md 8.1 requires it.
"""

import math

import numpy as np


def simulate_one_path(z_indep, z_perp, params, r, q, dt, n_substeps, n_exercise):
    """Returns lists S and nu at the exercise dates, starting at S = 1."""
    log_s = 0.0
    var = params["v0"]
    prices = [1.0]
    variances = [var]
    rho_perp = math.sqrt(1.0 - params["rho"] ** 2)
    step = 0
    for _ in range(n_exercise):
        for _ in range(n_substeps):
            z_s = float(z_indep[step])
            z_nu = params["rho"] * z_s + rho_perp * float(z_perp[step])
            step += 1
            var_plus = max(var, 0.0)
            root = math.sqrt(var_plus * dt)
            log_s += (r - q - 0.5 * var_plus) * dt + root * z_s
            var += params["kappa"] * (params["theta"] - var_plus) * dt + params["xi"] * root * z_nu
        prices.append(math.exp(log_s))
        variances.append(var)
    return prices, variances


def basis_row(a, b, degree):
    """All monomials a^i b^j with i + j <= degree."""
    row = []
    for j in range(degree + 1):
        for i in range(degree + 1 - j):
            row.append(a**i * b**j)
    return row


def scaling(values):
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(variance)


def scalar_lsmc(normals_a, normals_b, params, r, q, strike, maturity, n_exercise, n_substeps, degree):
    """Returns (bermudan, european) on set B, with the rule fitted on set A."""
    dt = maturity / (n_exercise * n_substeps)
    n_a = normals_a["Z_indep"].shape[0]
    n_b = normals_b["Z_indep"].shape[0]
    n_min = max(50, 10 * (degree + 1) * (degree + 2) // 2)

    paths_a = [simulate_one_path(normals_a["Z_indep"][i, :, 0], normals_a["Z_perp"][i, :, 0],
                                 params, r, q, dt, n_substeps, n_exercise) for i in range(n_a)]
    paths_b = [simulate_one_path(normals_b["Z_indep"][i, :, 0], normals_b["Z_perp"][i, :, 0],
                                 params, r, q, dt, n_substeps, n_exercise) for i in range(n_b)]

    times = [k * maturity / n_exercise for k in range(n_exercise + 1)]
    cash = [max(strike - path[0][n_exercise], 0.0) for path in paths_a]
    tau = [maturity] * n_a
    rules = {}

    for k in range(n_exercise - 1, 0, -1):
        itm = [i for i in range(n_a) if strike - paths_a[i][0][k] > 0.0]
        if len(itm) < n_min:
            continue
        spots = [paths_a[i][0][k] for i in itm]
        variances = [max(paths_a[i][1][k], 0.0) for i in itm]
        s_mean, s_std = scaling(spots)
        v_mean, v_std = scaling(variances)

        X = [basis_row((s - s_mean) / s_std, (v - v_mean) / v_std, degree) for s, v in zip(spots, variances)]
        y = [math.exp(-r * (tau[i] - times[k])) * cash[i] for i in itm]
        coef = np.linalg.lstsq(np.array(X), np.array(y), rcond=1e-10)[0]

        for row, i in zip(X, itm):
            continuation = sum(float(c) * x for c, x in zip(coef, row))
            payoff = strike - paths_a[i][0][k]
            if payoff > continuation:
                cash[i] = payoff
                tau[i] = times[k]
        rules[k] = (coef, s_mean, s_std, v_mean, v_std)

    total_bermudan = 0.0
    total_european = 0.0
    for prices, variances in paths_b:
        paid = None
        for k in range(1, n_exercise):
            payoff = strike - prices[k]
            if k in rules and payoff > 0.0:
                coef, s_mean, s_std, v_mean, v_std = rules[k]
                row = basis_row((prices[k] - s_mean) / s_std, (max(variances[k], 0.0) - v_mean) / v_std, degree)
                if payoff > sum(float(c) * x for c, x in zip(coef, row)):
                    paid = math.exp(-r * times[k]) * payoff
                    break
        final = math.exp(-r * maturity) * max(strike - prices[n_exercise], 0.0)
        total_bermudan += final if paid is None else paid
        total_european += final

    return total_bermudan / n_b, total_european / n_b
