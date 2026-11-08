"""
Closed-form European option prices under the Heston model, and
Black-Scholes implied volatility.

Heston prices use Lewis's (2001) formula. With forward F = S0 exp((r - q) T),
discount factor DF = exp(-r T) and log-moneyness k = ln(K / F):

    call = DF * (F - sqrt(F K) / pi * I)
    put  = DF * (K - sqrt(F K) / pi * I)
    I    = integral from 0 to infinity of  Re[exp(-i u k) phi(u - i/2)] / (u^2 + 1/4)  du

phi is the characteristic function of ln(S_T / F), written in the
"little Heston trap" form of Albrecher et al. (2007), which avoids jumps
in the complex logarithm at long maturities.

The Heston parameters are a dictionary, as in heston_lsmc.py:
    {"v0": 0.04, "kappa": 1.5, "theta": 0.04, "xi": 0.5, "rho": -0.7}
"""

import numpy as np
from scipy.special import ndtr  # standard normal CDF N(x), faster than scipy.stats.norm.cdf


# ----------------------------------------------------------------------
# Heston
# ----------------------------------------------------------------------

def heston_charfn(u, T, params):
    """
    Characteristic function E[exp(i u ln(S_T / F))] of the Heston model at
    maturity T. u can be a complex number or an array.
    """
    v0, kappa, theta = params["v0"], params["kappa"], params["theta"]
    xi, rho = params["xi"], params["rho"]

    beta = kappa - rho * xi * 1j * u
    d = np.sqrt(beta**2 + xi**2 * (1j * u + u**2))
    g = (beta - d) / (beta + d)
    e = np.exp(-d * T)

    C = kappa * theta / xi**2 * ((beta - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
    D = (beta - d) / xi**2 * (1 - e) / (1 - g * e)
    return np.exp(C + D * v0)


def lewis_integral(k, T, params):
    """
    The integral I of the Lewis formula, for an array of log-moneyness
    values k = ln(K / F) at one maturity T.
    """
    # 1. Upper limit U. The integrand is at most |phi(u - i/2)| / u^2, so the
    #    part of the integral beyond U is at most |phi(U - i/2)| / U.
    #    Double U until that is negligible.
    U = 1.0
    while abs(heston_charfn(U - 0.5j, T, params)) / U > 1e-13:
        U = 2 * U
        if U > 2**15:
            raise ValueError("the characteristic function decays too slowly; "
                             "check the parameters and maturity")

    # 2. Gauss-Legendre quadrature with 32 points on each panel of width at
    #    most 4. leggauss gives the points x and weights w on [-1, 1].
    x, w = np.polynomial.legendre.leggauss(32)
    n_panels = int(np.ceil(U / 4))
    edges = np.linspace(0, U, n_panels + 1)
    u_points = []
    u_weights = []
    for a, b in zip(edges[:-1], edges[1:]):
        u_points.append((a + b) / 2 + (b - a) / 2 * x)
        u_weights.append((b - a) / 2 * w)
    u = np.concatenate(u_points)
    weights = np.concatenate(u_weights)

    # 3. Integrand at every point, for every k at once (one row per k)
    phi = heston_charfn(u - 0.5j, T, params)
    k_column = np.reshape(k, (-1, 1))
    integrand = np.real(np.exp(-1j * k_column * u) * phi) / (u**2 + 0.25)

    return np.reshape(integrand @ weights, np.shape(k))


def heston_price(params, S0, K, T, r, q=0.0, call=True):
    """
    European option prices for one maturity T. K can be an array of
    strikes; call is True (call), False (put), or an array of both.
    """
    K = np.asarray(K, dtype=float)
    F = S0 * np.exp((r - q) * T)
    I = lewis_integral(np.log(K / F), T, params)
    common = np.sqrt(F * K) / np.pi * I
    return np.exp(-r * T) * np.where(call, F - common, K - common)


def heston_implied_vol(params, S0, K, T, r, q=0.0):
    """
    Black-Scholes implied volatilities of Heston prices for one maturity T.

    At each strike it uses the out-of-the-money option (a put below the
    forward, a call above it). Its price is small but accurate, while the
    in-the-money price would be mostly intrinsic value. Both give the same
    implied vol by put-call parity.

    Heston prices carry a round-off error of roughly 1e-13 * S0, so the
    implied vols of options cheaper than about 1e-9 * S0 (very far out of
    the money) are inaccurate or NaN. Listed quotes are far above this.
    """
    K = np.asarray(K, dtype=float)
    F = S0 * np.exp((r - q) * T)
    call = K >= F
    price = heston_price(params, S0, K, T, r, q, call)
    return implied_vol(price, S0, K, T, r, q, call)


# ----------------------------------------------------------------------
# Black-Scholes
# ----------------------------------------------------------------------

def black_scholes_price(S0, K, T, sigma, r, q=0.0, call=True):
    """Black-Scholes price with dividend yield q. Works on arrays."""
    F = S0 * np.exp((r - q) * T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = np.exp(-r * T) * (F * ndtr(d1) - K * ndtr(d2))
    put_price = np.exp(-r * T) * (K * ndtr(-d2) - F * ndtr(-d1))
    return np.where(call, call_price, put_price)


def implied_vol(price, S0, K, T, r, q=0.0, call=True, low=1e-4, high=5.0):
    """
    Black-Scholes implied volatility by bisection. Works on arrays.

    The option price rises with volatility, so we keep an interval
    [low, high] that contains the answer and halve it 60 times
    (its width ends near 5 / 2^60, about 4e-18).

    Returns NaN where no volatility between `low` and `high` gives the price
    (for example a price below intrinsic value).
    """
    price, K = np.broadcast_arrays(np.asarray(price, dtype=float), np.asarray(K, dtype=float))
    lo = np.full(price.shape, low)
    hi = np.full(price.shape, high)

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        too_high = black_scholes_price(S0, K, T, mid, r, q, call) > price
        hi = np.where(too_high, mid, hi)
        lo = np.where(too_high, lo, mid)
    vol = 0.5 * (lo + hi)

    reachable = ((price > black_scholes_price(S0, K, T, low, r, q, call)) &
                 (price < black_scholes_price(S0, K, T, high, r, q, call)))
    return np.where(reachable, vol, np.nan)


if __name__ == "__main__":
    # Fang & Oosterlee (2008) test case: published call price 5.785155450
    params = {"v0": 0.0175, "kappa": 1.5768, "theta": 0.0398, "xi": 0.5751, "rho": -0.5711}
    print("call price:", heston_price(params, S0=100.0, K=100.0, T=1.0, r=0.0))
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    print("implied vols:", heston_implied_vol(params, 100.0, strikes, 1.0, r=0.0))
