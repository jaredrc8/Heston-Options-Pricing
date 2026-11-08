"""Black-Scholes prices and implied volatility, European and American (SPEC.md 6.3)."""

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

from hestonpricer.tree import crr_price

# Search range for implied volatility (filter F8)
IV_LOW = 0.01
IV_HIGH = 5.0


def bs_price(kind, s0, strike, maturity, sigma, r, q):
    """Black-Scholes price with continuous dividend yield q. Works on arrays."""
    maturity = np.asarray(maturity, dtype=float)
    forward = s0 * np.exp((r - q) * maturity)
    root_t = sigma * np.sqrt(maturity)
    d1 = (np.log(forward / strike) + 0.5 * root_t**2) / root_t
    d2 = d1 - root_t
    discount = np.exp(-r * maturity)
    if kind == "call":
        return discount * (forward * ndtr(d1) - strike * ndtr(d2))
    return discount * (strike * ndtr(-d2) - forward * ndtr(-d1))


def solve_vol(price_function, price, xtol):
    """Find sigma in [IV_LOW, IV_HIGH] with price_function(sigma) = price, or nan."""
    def difference(sigma):
        return price_function(sigma) - price

    if difference(IV_LOW) * difference(IV_HIGH) > 0:
        return float("nan")
    return float(brentq(difference, IV_LOW, IV_HIGH, xtol=xtol))


def bs_implied_vol(kind, price, s0, strike, maturity, r, q):
    """Black-Scholes implied volatility. Returns nan if there is no root."""
    def pricer(sigma):
        return float(bs_price(kind, s0, strike, maturity, sigma, r, q))

    return solve_vol(pricer, price, xtol=1e-12)


def american_implied_vol(kind, price, s0, strike, maturity, r, q, n_steps=500):
    """Volatility at which a CRR American price equals price, or nan if no root."""
    def pricer(sigma):
        return crr_price(kind, s0, strike, maturity, sigma, r, q, n_steps)

    return solve_vol(pricer, price, xtol=1e-6)


def bs_implied_vol_array(is_call, prices, s0, strikes, maturity, r, q):
    """
    Implied volatilities for arrays of options at one maturity, by bisection.
    Used inside calibration, where a Python loop would be too slow.
    Returns nan where no volatility in [IV_LOW, IV_HIGH] reproduces the price.
    """
    def price_at(sigma):
        call = bs_price("call", s0, strikes, maturity, sigma, r, q)
        put = bs_price("put", s0, strikes, maturity, sigma, r, q)
        return np.where(is_call, call, put)

    low = np.full(len(prices), IV_LOW)
    high = np.full(len(prices), IV_HIGH)
    for _ in range(60):
        middle = 0.5 * (low + high)
        too_high = price_at(middle) > prices
        high = np.where(too_high, middle, high)
        low = np.where(too_high, low, middle)

    reachable = (prices > price_at(np.full(len(prices), IV_LOW))) & \
                (prices < price_at(np.full(len(prices), IV_HIGH)))
    return np.where(reachable, 0.5 * (low + high), np.nan)
