"""Cox-Ross-Rubinstein binomial tree for European, Bermudan and American options."""

import numpy as np

from hestonpricer.errors import require, require_finite


def crr_price(kind, s0, strike, maturity, sigma, r, q, n_steps, exercise_steps=None):
    """
    Option price on a CRR tree.

    exercise_steps=None : American, exercise allowed at every step.
    exercise_steps=[..] : Bermudan, exercise allowed only at those step numbers.
    The final step n_steps is always a payoff date.
    """
    s0, strike, maturity, sigma = float(s0), float(strike), float(maturity), float(sigma)
    r, q = float(r), float(q)
    require_finite(s0=s0, strike=strike, maturity=maturity, sigma=sigma, r=r, q=q)
    require(kind in ("call", "put"), "kind must be 'call' or 'put'")
    require(s0 > 0 and strike > 0 and maturity > 0 and sigma > 0,
            "s0, strike, maturity and sigma must be positive")
    require(n_steps >= 1, "n_steps must be at least 1")

    sign = 1.0 if kind == "call" else -1.0
    dt = maturity / n_steps
    up = np.exp(sigma * np.sqrt(dt))
    down = 1.0 / up
    discount = np.exp(-r * dt)
    # Very small sigma can push p outside [0, 1]; clipping keeps the tree stable.
    p = float(np.clip((np.exp((r - q) * dt) - down) / (up - down), 0.0, 1.0))

    if exercise_steps is None:
        can_exercise = np.ones(n_steps + 1, dtype=bool)
    else:
        can_exercise = np.zeros(n_steps + 1, dtype=bool)
        can_exercise[list(exercise_steps)] = True

    # Stock prices at the last step, with j up moves out of n_steps
    stock = s0 * up ** (2.0 * np.arange(n_steps + 1) - n_steps)
    values = np.maximum(sign * (stock - strike), 0.0)

    # Step back through the tree
    for step in range(n_steps - 1, -1, -1):
        values = discount * (p * values[1:] + (1.0 - p) * values[:-1])
        if can_exercise[step]:
            stock = s0 * up ** (2.0 * np.arange(step + 1) - step)
            values = np.maximum(values, sign * (stock - strike))

    return float(values[0])
