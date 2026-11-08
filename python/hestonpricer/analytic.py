"""
Semi-analytic European option prices under Heston (Lewis 2001 formula).

With forward F = S0 exp((r - q) T), discount DF = exp(-r T) and k = ln(K / F):

    call = DF * (F - sqrt(F K) / pi * I)
    put  = DF * (K - sqrt(F K) / pi * I)
    I    = integral from 0 to infinity of Re[exp(-i u k) phi(u - i/2)] / (u^2 + 1/4) du

phi is the characteristic function of ln(S_T / F), in the "little Heston trap"
form of Albrecher et al. (2007), which has no jumps in the complex logarithm.
"""

import numpy as np

from hestonpricer.errors import HestonError, require, require_finite

# Gauss-Legendre points and weights on [-1, 1]
GL_X, GL_W = np.polynomial.legendre.leggauss(32)


def heston_charfn(u, T, params):
    """E[exp(i u ln(S_T / F))] at maturity T. u can be a complex array."""
    v0, kappa, theta = params["v0"], params["kappa"], params["theta"]
    xi, rho = params["xi"], params["rho"]

    a = 1j * u + u**2
    beta = kappa - rho * xi * 1j * u
    d = np.sqrt(beta**2 + xi**2 * a)
    # beta - d written as -xi^2 a / (beta + d), so that nothing cancels when xi is small
    beta_minus_d = -xi**2 * a / (beta + d)
    g = beta_minus_d / (beta + d)
    e = np.exp(-d * T)

    C = kappa * theta * (-a / (beta + d) * T
                         - 2 / xi**2 * np.log((1 - g * e) / (1 - g)))
    D = -a / (beta + d) * (1 - e) / (1 - g * e)
    return np.exp(C + D * v0)


def lewis_integral(k, T, params):
    """The integral I, for an array k of log-moneyness values at one maturity T."""
    # Upper limit U: the part of the integral beyond U is at most |phi(U - i/2)| / U.
    U = 1.0
    while abs(heston_charfn(U - 0.5j, T, params)) / U > 1e-13:
        U = 2 * U
        if U > 2**15:
            raise HestonError("E_NONFINITE", "characteristic function decays too slowly")

    # 32-point Gauss-Legendre on panels of width at most 4
    n_panels = int(np.ceil(U / 4))
    edges = np.linspace(0.0, U, n_panels + 1)
    middle = (edges[:-1] + edges[1:]) / 2
    half_width = (edges[1:] - edges[:-1]) / 2
    u = (middle[:, None] + half_width[:, None] * GL_X).ravel()
    weights = (half_width[:, None] * GL_W).ravel()

    phi = heston_charfn(u - 0.5j, T, params)
    k = np.asarray(k, dtype=float)
    integrand = np.real(np.exp(-1j * k[:, None] * u) * phi) / (u**2 + 0.25)
    return integrand @ weights


def heston_european_many(is_call, s0, strikes, maturity, params, r, q):
    """Prices at one maturity for an array of strikes. is_call is a boolean array."""
    strikes = np.asarray(strikes, dtype=float)
    forward = s0 * np.exp((r - q) * maturity)
    integral = lewis_integral(np.log(strikes / forward), maturity, params)
    common = np.sqrt(forward * strikes) / np.pi * integral
    return np.exp(-r * maturity) * np.where(is_call, forward - common, strikes - common)


def heston_european(kind, s0, strike, maturity, params, r, q):
    """European call or put price under Heston. kind is "call" or "put"."""
    s0, strike, maturity, r, q = float(s0), float(strike), float(maturity), float(r), float(q)
    require_finite(s0=s0, strike=strike, maturity=maturity, r=r, q=q, **params)
    require(kind in ("call", "put"), "kind must be 'call' or 'put'")
    require(s0 > 0 and strike > 0 and maturity > 0, "s0, strike and maturity must be positive")
    require(params["xi"] > 0, "xi must be positive in heston_european")
    require(abs(params["rho"]) < 1, "|rho| must be below 1 in heston_european")
    require(params["v0"] > 0 and params["kappa"] > 0 and params["theta"] > 0,
            "v0, kappa and theta must be positive")
    is_call = np.array([kind == "call"])
    return float(heston_european_many(is_call, s0, [strike], maturity, params, r, q)[0])
