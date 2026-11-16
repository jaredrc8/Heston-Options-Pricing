"""
Simulation of one Heston asset (SPEC.md section 7).

Each time step uses full-truncation Euler for the variance and log-Euler for
the price. The variance may go negative in storage; only nu_plus = max(nu, 0)
is used in formulas. There is no Feller check.
"""

import numpy as np

from hestonpricer.errors import require, require_finite

DEFAULT_CONTRACT = {"strike": 1.0, "maturity": 1.0, "n_exercise": 50, "kind": "put"}
DEFAULT_SIM = {"n_substeps": 4, "basis_degree": 3, "master_seed": 620001}


def read_contract(contract):
    """Fill in defaults. Numbers are cast so no integer array can be created later (SPEC.md 14.3)."""
    contract = {**DEFAULT_CONTRACT, **contract}
    strike = float(contract["strike"])
    maturity = float(contract["maturity"])
    n_exercise = int(contract["n_exercise"])
    require_finite(strike=strike, maturity=maturity)
    require(strike > 0 and maturity > 0, "strike and maturity must be positive")
    require(n_exercise >= 1, "n_exercise must be at least 1")
    require(contract["kind"] in ("put", "call"), "contract['kind'] must be 'put' or 'call'")
    return strike, maturity, n_exercise


def contract_kind(contract):
    return {**DEFAULT_CONTRACT, **contract}["kind"]


def simulate(params, corr, market, contract, sim, rng, normals=None):
    """
    Simulate the asset and return {"S": ..., "nu": ...} at t_0, ..., t_M.
    Each array has shape (n_paths, M + 1, 1), and S starts at 1.

    params is a one-element list and corr is [[1.0]] (SPEC.md 10.3).
    normals, if given, is {"Z_indep": ..., "Z_perp": ...}, each with shape
    (n_paths, M * n_substeps, 1). rng is then ignored.
    """
    require(len(params) == 1 and len(market["q"]) == 1, "one asset only: params and market['q'] need one entry")
    require(np.array_equal(np.asarray(corr, dtype=float), [[1.0]]), "corr must be [[1.0]]")
    p = params[0]
    require_finite(**p)
    require(p["xi"] >= 0 and p["kappa"] >= 0, "xi and kappa must be non-negative")
    require(p["v0"] >= 0 and p["theta"] >= 0, "v0 and theta must be non-negative")
    require(abs(p["rho"]) <= 1, "|rho| must be at most 1")

    _, maturity, n_exercise = read_contract(contract)
    settings = {**DEFAULT_SIM, **sim}
    n_paths = int(settings["n_paths"])
    n_substeps = int(settings["n_substeps"])
    require(n_paths >= 1 and n_substeps >= 1, "n_paths and n_substeps must be at least 1")
    dt = maturity / (n_exercise * n_substeps)

    r = float(market["r"])
    q = float(market["q"][0])
    v0, kappa, theta = float(p["v0"]), float(p["kappa"]), float(p["theta"])
    xi, rho = float(p["xi"]), float(p["rho"])
    rho_perp = np.sqrt(1.0 - rho**2)

    S = np.zeros((n_paths, n_exercise + 1, 1))
    nu = np.zeros((n_paths, n_exercise + 1, 1))
    S[:, 0] = 1.0
    nu[:, 0] = v0

    log_s = np.zeros((n_paths, 1))
    var = np.full((n_paths, 1), v0)
    step = 0

    for k in range(1, n_exercise + 1):
        for _ in range(n_substeps):
            if normals is not None:
                z_price = normals["Z_indep"][:, step]
                z_perp = normals["Z_perp"][:, step]
            else:
                z_price = rng.standard_normal((n_paths, 1))
                z_perp = rng.standard_normal((n_paths, 1))
            step += 1

            z_var = rho * z_price + rho_perp * z_perp

            var_plus = np.maximum(var, 0.0)
            root = np.sqrt(var_plus * dt)
            log_s = log_s + (r - q - 0.5 * var_plus) * dt + root * z_price
            var = var + kappa * (theta - var_plus) * dt + xi * root * z_var

        S[:, k] = np.exp(log_s)
        nu[:, k] = var

    return {"S": S, "nu": nu}
