"""
Simulation of correlated Heston assets (SPEC.md section 7).

Each time step uses full-truncation Euler for the variance and log-Euler for
the price. The variance may go negative in storage; only nu_plus = max(nu, 0)
is used in formulas. There is no Feller check.
"""

import numpy as np

from hestonpricer.errors import require, require_finite

DEFAULT_CONTRACT = {"strike": 1.0, "maturity": 1.0, "n_exercise": 50}
DEFAULT_SIM = {"n_substeps": 4, "basis_degree": 3, "master_seed": 620001,
               "identical_assets_test_mode": False}


def read_contract(contract):
    """Fill in defaults. Numbers are cast so no integer array can be created later."""
    contract = {**DEFAULT_CONTRACT, **contract}
    strike = float(contract["strike"])
    maturity = float(contract["maturity"])
    n_exercise = int(contract["n_exercise"])
    require_finite(strike=strike, maturity=maturity)
    require(strike > 0 and maturity > 0, "strike and maturity must be positive")
    require(n_exercise >= 1, "n_exercise must be at least 1")
    return strike, maturity, n_exercise


def check_correlation(corr, d):
    """Check the correlation matrix (SPEC.md 10.3). Singular matrices are valid."""
    require(corr.shape == (d, d), "corr must be d x d")
    require(np.all(np.isfinite(corr)), "corr must be finite")
    require(np.allclose(corr, corr.T, atol=1e-12), "corr must be symmetric")
    require(np.allclose(np.diag(corr), 1.0, atol=1e-12), "corr must have a unit diagonal")
    require(np.min(np.linalg.eigvalsh(corr)) >= -1e-10, "corr must be positive semi-definite")


def simulate(params, corr, market, contract, sim, rng, normals=None):
    """
    Simulate the assets and return {"S": ..., "nu": ...} at t_0, ..., t_M.
    Each array has shape (n_paths, M + 1, d), and S starts at 1.

    normals, if given, is {"Z_indep": ..., "Z_perp": ...}, each with shape
    (n_paths, M * n_substeps, d). rng is then ignored.
    """
    d = len(params)
    corr = np.asarray(corr, dtype=float)
    require(len(market["q"]) == d, "market['q'] needs one entry per asset")
    check_correlation(corr, d)
    for p in params:
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
    q = np.array(market["q"], dtype=float)
    v0 = np.array([p["v0"] for p in params], dtype=float)
    kappa = np.array([p["kappa"] for p in params], dtype=float)
    theta = np.array([p["theta"] for p in params], dtype=float)
    xi = np.array([p["xi"] for p in params], dtype=float)
    rho = np.array([p["rho"] for p in params], dtype=float)
    rho_perp = np.sqrt(1.0 - rho**2)

    # Symmetric eigendecomposition, so a singular corr is fine: if e has independent
    # standard normals, e @ A.T has correlation matrix corr.
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    A = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))

    S = np.zeros((n_paths, n_exercise + 1, d))
    nu = np.zeros((n_paths, n_exercise + 1, d))
    S[:, 0] = 1.0
    nu[:, 0] = v0

    log_s = np.zeros((n_paths, d))
    var = np.tile(v0, (n_paths, 1))
    step = 0

    for k in range(1, n_exercise + 1):
        for _ in range(n_substeps):
            if normals is not None:
                z_indep = normals["Z_indep"][:, step]
                z_perp = normals["Z_perp"][:, step]
            else:
                z_indep = rng.standard_normal((n_paths, d))
                if settings["identical_assets_test_mode"]:
                    z_perp = np.tile(rng.standard_normal((n_paths, 1)), (1, d))
                else:
                    z_perp = rng.standard_normal((n_paths, d))
            step += 1

            z_price = z_indep @ A.T
            z_var = rho * z_price + rho_perp * z_perp

            var_plus = np.maximum(var, 0.0)
            root = np.sqrt(var_plus * dt)
            log_s = log_s + (r - q - 0.5 * var_plus) * dt + root * z_price
            var = var + kappa * (theta - var_plus) * dt + xi * root * z_var

        S[:, k] = np.exp(log_s)
        nu[:, k] = var

    return {"S": S, "nu": nu}
