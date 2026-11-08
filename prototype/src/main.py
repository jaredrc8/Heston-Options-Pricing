import heston_closed_form as closed_form
import heston_lsmc as lsmc
import correlation
import calibrate_market

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import pandas as pd
import time


if __name__ == "__main__":

    start_time = time.perf_counter()

    # Run the correlation analysis script
    # This section of code is not necessary for the pricing model,
    # but it proves the correlation between the three assets for further analysis
    correlation_matrix = correlation.main()
    print("======== Correlation Matrix ========")
    print(correlation_matrix)
    print("====================================")

    end_time = time.perf_counter()
    print(f"Executed: {end_time - start_time:.2f} seconds\n")
    start_time = time.perf_counter()

    print("================= Heston Parameters =================")
    # Heston parameters for each asset come from results/heston_params.csv
    # (columns: symbol, spot, v0, kappa, theta, xi, rho).
    USE_MARKET_DATA = False
    if USE_MARKET_DATA:
        calibrate_market.main()

    heston_params = pd.read_csv(calibrate_market.PARAMS_FILE, index_col="symbol")
    print(heston_params[["spot", "v0", "kappa", "theta", "xi", "rho"]].round(4))
    print("=====================================================")

    end_time = time.perf_counter()
    print(f"Executed: {end_time - start_time:.2f} seconds\n")
    start_time = time.perf_counter()

    print("============ Simulated Heston Paths ============")
    SEED = 149032        # random number generator seed
    r, q = 0.05, 0.0     # risk-free rate and dividend yield
    T = 1.0              # 1-year 
    n_steps = 252        # one step per trading day
    n_plot_paths = 20    # paths drawn per asset
    print(f"RNG Seed                    {SEED}")
    print(f"Risk-free rate              {r}")
    print(f"Dividend yield              {q}")
    print(f"Simulation time (years)     {T}")
    print(f"Steps                       {n_steps}")
    print(f"Simulations per asset       {n_plot_paths}")

    symbols = list(heston_params.index)
    S0 = heston_params["spot"].to_numpy()
    params = [heston_params.loc[s, ["v0", "kappa", "theta", "xi", "rho"]].to_dict() for s in symbols]
    # Correlation of the asset prices, in the same order as the parameter file
    corr = correlation_matrix.loc[symbols, symbols].to_numpy()

    # Store every day, so each "exercise date" here is just a plotting point
    times = np.linspace(0.0, T, n_steps + 1)
    rng = np.random.default_rng(SEED)
    S, v = lsmc.simulate_heston_paths(S0, params, corr, r, q, times, 1, n_plot_paths, rng)

    fig, axes = plt.subplots(1, len(symbols), figsize=(15, 4.5))
    for i, symbol in enumerate(symbols):
        axes[i].plot(times, S[:, :, i], linewidth=0.8)
        axes[i].axhline(S0[i], color="black", linestyle="--", linewidth=1)
        axes[i].set_title(f"{symbol} (S0 = {S0[i]:.2f})")
        axes[i].set_xlabel("Time (years)")
        axes[i].set_ylabel("Price ($)")
    fig.suptitle(f"Simulated Heston paths ({n_plot_paths} per asset, seed {SEED})")
    fig.tight_layout()
    figure_file = calibrate_market.PARAMS_FILE.parent / "simulated_paths.png"
    fig.savefig(figure_file, dpi=150)
    print(f"Saved figure to {figure_file}")
    print("================================================")

    end_time = time.perf_counter()
    print(f"Executed: {end_time - start_time:.2f} seconds\n")
    start_time = time.perf_counter()

    print("============ NVDA Bermudan Put (LSMC) ============")
    # Inputs
    symbol = "NVDA"
    S0_nvda = heston_params.loc[symbol, "spot"]   # current price ($)
    K = S0_nvda                                   # strike price ($), here at the money
    T_option = 1.0                                # time to expiry (years)
    days_per_year = 252                           # daily exercise dates (~American)
    n_paths = 1000000                             # paths per set (fit set + pricing set)
    n_points = 26                                 # points on the price curve, one every two weeks
    nvda_params = heston_params.loc[symbol, ["v0", "kappa", "theta", "xi", "rho"]].to_dict()

    # Price the put with time t already elapsed, holding today's price S0
    # fixed, i.e. the same contract with T - t left to expiry. The first
    # point (t = 0) is today's fair price. The same seed at every point
    # keeps the curve smooth.
    # heston_lsmc prices the put on performance, max(K/S0 - S/S0, 0), so the
    # strike is passed as K/S0 and the results are multiplied by S0 for dollars.
    time_elapsed = np.linspace(0.0, T_option, n_points, endpoint=False)
    bermudan_price = np.zeros(n_points)
    bermudan_error = np.zeros(n_points)
    european_mc = np.zeros(n_points)
    european_mc_error = np.zeros(n_points)
    european_exact = np.zeros(n_points)
    premium = np.zeros(n_points)
    premium_error = np.zeros(n_points)

    for i, t in enumerate(time_elapsed):
        tau = T_option - t                                  # time to expiry
        n_dates = max(1, int(round(days_per_year * tau)))   # daily exercise dates
        exercise_times = np.arange(1, n_dates + 1) * tau / n_dates
        result = lsmc.price_bermudan_worst_of_put([S0_nvda], [nvda_params], np.eye(1), r, q,
                                                  K / S0_nvda, exercise_times, n_paths,
                                                  substeps=2, degree=2, seed=SEED)
        bermudan_price[i] = S0_nvda * result["price"]
        bermudan_error[i] = S0_nvda * result["std_error"]
        european_mc[i] = S0_nvda * result["european_price"]
        european_mc_error[i] = S0_nvda * result["european_std_error"]
        premium[i] = S0_nvda * result["premium"]
        premium_error[i] = S0_nvda * result["premium_std_error"]
        # Closed-form European put, as a check on the simulated European price
        european_exact[i] = float(closed_form.heston_price(nvda_params, S0_nvda, K, tau, r, q, call=False))

    # Today's fair price is the first point (t = 0)
    print(f"Current price S0            {S0_nvda:.2f}")
    print(f"Strike K                    {K:.2f}")
    print(f"Expiry T (years)            {T_option}")
    print(f"Exercise dates              {days_per_year * T_option:.0f} (daily)")
    print(f"Paths per set               {n_paths}")
    print(f"Bermudan put price          {bermudan_price[0]:.4f} +/- {bermudan_error[0]:.4f}")
    print(f"European put price (MC)     {european_mc[0]:.4f} +/- {european_mc_error[0]:.4f}")
    print(f"European put (closed form)  {european_exact[0]:.4f}")
    print(f"Early-exercise premium      {premium[0]:.4f} +/- {premium_error[0]:.4f}")

    # Plot the stored prices as t approaches T
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time_elapsed, bermudan_price, label="Bermudan, daily exercise (~American)")
    ax.fill_between(time_elapsed, bermudan_price - 1.96 * bermudan_error,
                    bermudan_price + 1.96 * bermudan_error, alpha=0.3, label="95% confidence band")
    ax.plot(time_elapsed, european_exact, linestyle="--", label="European (closed form)")
    ax.axhline(max(K - S0_nvda, 0.0), color="black", linewidth=1, label="Intrinsic value")
    ax.set_xlabel("Time elapsed t (years)")
    ax.set_ylabel("Put price ($)")
    ax.set_title(f"{symbol} put, S0 = {S0_nvda:.2f}, K = {K:.2f}, T = {T_option} (seed {SEED})")
    ax.legend()
    fig.tight_layout()
    figure_file = calibrate_market.PARAMS_FILE.parent / "nvda_put_price_vs_time.png"
    fig.savefig(figure_file, dpi=150)
    print(f"Saved figure to {figure_file}")
    print("==================================================")

    end_time = time.perf_counter()
    print(f"Executed: {end_time - start_time:.2f} seconds\n")