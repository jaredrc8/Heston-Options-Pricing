"""
Generate the synthetic fixtures in tests/fixtures/ from a fixed seed
(DATA_CONTRACT.md section 1.4 and SPEC.md section 6.8). Everything here is
fictitious, so the output is Public.

    python python/make_fixtures.py
"""

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from hestonpricer.analytic import heston_european
from hestonpricer.iv import bs_implied_vol

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
SEED = 620200

# Known parameters of the synthetic surface. These are made up for testing; they are
# not taken from any paper and describe no real stock.
SYN_PARAMS = {"v0": 0.09, "kappa": 2.0, "theta": 0.10, "xi": 0.6, "rho": -0.6}
SYN_YIELD = 0.04      # the "Treasury yield" of the synthetic world
SYN_Q = 0.0
SYN_SPOT = 100.0
DATES = ["2026-10-13", "2026-10-14", "2026-10-15"]
EXPIRIES = ["2026-11-19", "2027-01-13", "2027-04-14", "2027-07-14", "2027-10-14", "2028-01-12"]
STRIKES = np.arange(70.0, 140.1, 2.5)


def black_scholes_greeks(kind, s, k, t, sigma, r, q):
    """Delta, gamma, vega, theta and rho, so that the fixture matches the schema."""
    from scipy.special import ndtr
    root_t = sigma * np.sqrt(t)
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / root_t
    d2 = d1 - root_t
    pdf = np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)
    gamma = np.exp(-q * t) * pdf / (s * root_t)
    vega = s * np.exp(-q * t) * pdf * np.sqrt(t)
    if kind == "call":
        delta = np.exp(-q * t) * ndtr(d1)
        rho = k * t * np.exp(-r * t) * ndtr(d2)
        theta = -s * np.exp(-q * t) * pdf * sigma / (2 * np.sqrt(t)) - r * k * np.exp(-r * t) * ndtr(d2)
    else:
        delta = np.exp(-q * t) * (ndtr(d1) - 1)
        rho = -k * t * np.exp(-r * t) * ndtr(-d2)
        theta = -s * np.exp(-q * t) * pdf * sigma / (2 * np.sqrt(t)) + r * k * np.exp(-r * t) * ndtr(-d2)
    return delta, gamma, vega, theta, rho


def make_raw_iv():
    """Exact Heston European quotes with no noise, on all three dates (SPEC.md 6.8)."""
    r = 2 * np.log(1 + SYN_YIELD / 2)
    rows = []
    for date in DATES:
        for expiry in EXPIRIES:
            days = (pd.Timestamp(expiry) - pd.Timestamp(date)).days
            tenor = days / 365.0
            for strike in STRIKES:
                for code, kind in (("C", "call"), ("P", "put")):
                    mid = heston_european(kind, SYN_SPOT, strike, tenor, SYN_PARAMS, r, SYN_Q)
                    iv = bs_implied_vol(kind, mid, SYN_SPOT, strike, tenor, r, SYN_Q)
                    if not np.isfinite(iv):
                        continue    # too deep in or out of the money to invert
                    delta, gamma, vega, theta, rho = black_scholes_greeks(
                        kind, SYN_SPOT, strike, tenor, iv, r, SYN_Q)
                    rows.append({
                        "symbol": "SYN1", "exchange": "SYNX", "date": date,
                        "stock_price": SYN_SPOT,
                        "option_symbol": f"SYN1-{expiry}-{strike:.1f}-{code}",
                        "expiration": expiry, "strike": strike, "call_put": code, "style": "E",
                        "bid": mid * 0.99, "ask": mid * 1.01, "mean_price": mid, "iv": iv,
                        "volume": 100.0, "open_interest": 500.0, "stock_price_for_iv": SYN_SPOT,
                        "delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho,
                    })
    pd.DataFrame(rows).to_csv(FIXTURES / "synthetic_raw_iv.csv", index=False)
    return len(rows)


def make_minute_bars(rng):
    """Four bars a day for three fictitious tickers, with correlated returns."""
    days = pd.bdate_range("2025-01-02", periods=320)
    bar_times = ["07:00", "09:30", "12:00", "15:59"]    # 07:00 is pre-market, on purpose
    corr = np.array([[1.0, 0.6, 0.4], [0.6, 1.0, 0.3], [0.4, 0.3, 1.0]])
    chol = np.linalg.cholesky(corr)
    daily_vol = 0.02
    log_price = np.log([50.0, 80.0, 120.0])

    rows = []
    for day in days:
        shocks = chol @ rng.standard_normal(3) * daily_vol
        previous = log_price.copy()
        log_price = log_price + shocks
        for b, bar in enumerate(bar_times):
            stamp = pd.Timestamp(f"{day.date()} {bar}", tz="America/New_York")
            window_start = int(stamp.tz_convert("UTC").value)
            fraction = (b + 1) / len(bar_times)
            close = np.exp(previous + fraction * shocks)
            open_ = close * np.exp(0.0002 * rng.standard_normal(3))
            for i, ticker in enumerate(["SYN1", "SYN2", "SYN3"]):
                rows.append({
                    "ticker": ticker, "volume": float(rng.integers(100, 10_000)),
                    "open": open_[i], "close": close[i],
                    "high": max(open_[i], close[i]) * 1.001,
                    "low": min(open_[i], close[i]) * 0.999,
                    "window_start": window_start,
                    "transactions": int(rng.integers(1, 200)),
                })

    # mtime=0 makes the gzip file byte-identical on every run
    with open(FIXTURES / "synthetic_minute_bars.csv.gz", "wb") as raw_file:
        with gzip.GzipFile(fileobj=raw_file, mode="wb", mtime=0) as gz:
            gz.write(pd.DataFrame(rows).to_csv(index=False).encode())
    return len(rows)


def make_shared_normals(rng):
    """Normals for V-07, the comparison with the independent scalar LSMC."""
    n_paths, n_steps, d = 1000, 16, 1
    np.savez_compressed(FIXTURES / "shared_normals.npz",
                        Z_indep=rng.standard_normal((n_paths, n_steps, d)),
                        Z_perp=rng.standard_normal((n_paths, n_steps, d)))


def write_yaml_files():
    """Known parameters and market inputs of the synthetic world."""
    (FIXTURES / "synthetic_params.yaml").write_text(yaml.safe_dump(
        {"assets": {"SYN1": SYN_PARAMS}}, sort_keys=False))
    (FIXTURES / "synthetic_market_inputs.yaml").write_text(yaml.safe_dump(
        {"treasury_1y_cmt_yield": {date: SYN_YIELD for date in DATES},
         "dividend_yield": {"SYN1": SYN_Q}}, sort_keys=False))


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    rng = np.random.Generator(np.random.PCG64(SEED))
    write_yaml_files()
    n_iv = make_raw_iv()
    n_bars = make_minute_bars(rng)
    make_shared_normals(rng)
    print(f"fixtures written: {n_iv} option rows, {n_bars} minute bars")


if __name__ == "__main__":
    main()
