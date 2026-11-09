# Data Contract — ENSF 620 Final Project

**Project:** Pricing a Bermudan put under calibrated Heston stochastic volatility
**Owner:** Jared Crebo
**Contract version:** 0.6 · **Last updated:** 2026-11-08

This contract governs two licensed datasets and everything derived from them:

1. **Massive** minute-resolution U.S. stock bars for NVDA, AMD, INTC, GOOG, and GOOGL, used for return and correlation estimation.
2. **iVolatility** Raw IV option-contract data for NVDA, AMD, and INTC, used for Heston calibration. This dataset has no Google options.

It defines what the data looks like (Schema), where it came from and under what terms (Provenance), which files an AI agent may touch (Allowed Paths), and what may appear verbatim in prompts, logs, the repository, the report, and the presentation (Classification).

Where this contract conflicts with the September 29 proposal, **this contract wins**. In particular, the proposal planned to commit the raw market-data snapshot to Git; that is not permitted.

---

## 1. Schema

### 1.1 Massive — Stocks Minute Aggregates

A Parquet package at `data/nvda_amd_intc_goog_minute/bars/`, partitioned by year and month, with one or more part files per month. Each row is one one-minute OHLCV bar for one ticker. At ingest, `source_window_start_ns` is mapped to `window_start` and decimal columns are cast to float64; the package's other columns are not used. Files are stored unmodified; filtering and cleaning happen only in the derived layer.

| Column | Type | Allowed range / rule | Notes |
|---|---|---|---|
| `ticker` | string | One of `NVDA`, `AMD`, `INTC`, `GOOG`, `GOOGL` | Rows for other tickers are dropped at ingest and never stored in derived data. GOOGL is validated but not used in derived data, because it is a share class of the same company as GOOG. |
| `volume` | number (float64) | `>= 0` | Shares traded in the window. |
| `open` | number (float64) | `> 0` | USD. |
| `close` | number (float64) | `> 0` | USD. |
| `high` | number (float64) | `> 0`; `high >= max(open, close)` | USD. |
| `low` | number (float64) | `> 0`; `low <= min(open, close)` | USD. |
| `window_start` | integer (int64) | Unix **nanoseconds**, UTC; divisible by `60_000_000_000`; between 2011-11-01 and 2026-07-31 | Start of the one-minute window. |
| `transactions` | integer (int64) | `>= 0` | Trade count in the window. |

**Table-level rules**

- After the ingest mapping, the table has exactly the columns `ticker, volume, open, close, high, low, window_start, transactions`. CSV fixtures must also have them in this order.
- `(ticker, window_start)` is unique within each month, across all of that month's part files.
- No nulls in any column.
- `window_start`, converted to `America/New_York`, falls between 04:00 and 20:00 (pre-market, regular session, and after-hours). The regular-session filter (09:30–16:00 ET) is applied in the derived layer, not to the raw files. One bar breaks this rule and is omitted by the owner's decision of 2026-11-08: `year=2018/month=7`, row index 2090. It stays in the raw files, and any other bar outside these hours is still a violation.
- Minutes with no trades have **no row**; missing minutes are expected and are not errors.
- Prices are treated as **unadjusted** for splits and dividends. NVDA's and GOOG's stock splits inside the 2011–2026 window, listed in `configs/splits.yaml`, must be adjusted for in the derived layer before returns are computed; an unexplained one-minute log return larger than 0.5 in absolute value stops the pipeline.

### 1.2 iVolatility — Raw IV (individual option contracts)

End-of-day records, one row per listed option contract per trading day, as delivered by iVolatility. Raw header strings are mapped to the canonical names below at ingest; the raw files themselves are not modified.

| Column (canonical) | Type | Allowed range / rule | Notes |
|---|---|---|---|
| `symbol` | string | One of `NVDA`, `AMD`, `INTC` | Underlying. |
| `exchange` | string | Non-empty | Listing venue code as delivered. |
| `date` | date | 2026-10-13, 2026-10-14, or 2026-10-15 | Trading date of the record. |
| `stock_price` | float64 | `> 0` | Underlying close (USD). |
| `option_symbol` | string | Non-empty; unique within `(symbol, date)` | Contract identifier. |
| `expiration` | date | `>= date` | Contract expiry. A contract may expire on its trade date: same-day expiries are normal listed contracts, so `> date` was too strict. They cannot reach calibration, because filter F3 in `SPEC.md` §6.2 requires 21 to 511 days to expiry. An expiry before the trade date is still a violation. |
| `strike` | float64 | `> 0` | USD. |
| `call_put` | string | `C` or `P` | |
| `style` | string | `A` (American) or `E` (European) | U.S. single-stock options are expected to be `A`. |
| `bid` | float64 | `>= 0` | USD. |
| `ask` | float64 | `>= bid` | USD. |
| `mean_price` | float64 | `bid <= mean_price <= ask` | Mid-quote. |
| `iv` | float64 | `> 0` when present; missing or non-positive means "no IV computed" | Decimal annualised implied volatility. Rows without a valid IV are excluded from calibration and counted in the validation report. |
| `volume` | float64 | `>= 0` | Contracts traded. |
| `open_interest` | float64 | `>= 0` | |
| `stock_price_for_iv` | float64 | `> 0` | Underlying price iVolatility used in the IV computation. |
| `delta` | float64 | Calls in `[0, 1]`; puts in `[-1, 0]` | |
| `gamma` | float64 | `>= 0` | |
| `vega` | float64 | `>= 0` | |
| `theta` | float64 | Any finite value | |
| `rho` | float64 | Any finite value | |

**Table-level rules**

- `(symbol, date, option_symbol)` is unique.
- Every symbol has records on all three dates.
- Calibration grid selection (strikes, maturities, liquidity filters) is defined in code and applied in the derived layer; the raw files are never filtered in place.

### 1.3 Derived datasets

| Dataset | Path | Contents | Class |
|---|---|---|---|
| Session closes | `data/derived/session_close.parquet` | `ticker`, `date` (ET), split-adjusted `close` | **Restricted** (lossless subset of raw prices) |
| Log returns | `data/derived/log_returns.parquet` | `ticker`, `date`, `log_return` | **Restricted** (prices are recoverable from returns plus one price) |
| Calibration grid | `data/derived/calibration_grid.parquet` | Selected `(symbol, date, strike, expiration, call_put, iv)` rows | **Restricted** (subset of raw iVolatility data) |
| Estimation summary | `results/estimates.json` | Per-ticker sample size and realised volatility; cross-asset correlation matrix (entries in [−1, 1], symmetric, unit diagonal, positive semi-definite) | **Internal** |
| Calibration summary | `results/calibration.json` | Per-ticker fitted Heston parameters, grid size, RMS IV residual, convergence status per starting point | **Internal** |

`results/estimates.json` and `results/calibration.json` are the only data-derived artifacts that leave the restricted boundary.

### 1.4 Synthetic fixtures

| Fixture | Schema | Class |
|---|---|---|
| `tests/fixtures/synthetic_minute_bars.csv.gz` | §1.1, fictitious tickers `SYN1`–`SYN3` | **Public** |
| `tests/fixtures/synthetic_raw_iv.csv` | §1.2, generated from known Heston parameters | **Public** |

Both are generated by `python/make_fixtures.py` from a fixed seed. A reviewer runs the full pipeline on them and checks the known answers (including calibration parameter recovery) without access to licensed data. They live under `tests/`, not `data/`, because everything under `data/` is excluded from Git.

### 1.5 Validation

`python/validate_raw.py` checks every rule in §1.1 and §1.2 and writes a report containing only: file name, row count, rule name, number of violating rows, and violating row **indices**. It never prints row contents (see §4.3). Any violation of a uniqueness, range, or price-consistency rule stops the pipeline.

---

## 2. Provenance

### 2.1 Massive

| Field | Value |
|---|---|
| Provider / steward | Massive (massive.com), formerly Polygon.io |
| Product | Stocks — Minute Aggregates |
| Tickers | NVDA, AMD, INTC, GOOG, GOOGL |
| Date range | November 2011 – July 2026 |
| How it was obtained | Downloaded by the project owner using a personal Massive account |
| Licence / terms | Massive Terms of Service. Permitted: non-commercial analysis by the account holder. **Not permitted: distribution to any other person**, in raw or reconstructable form. |

### 2.2 iVolatility

| Field | Value |
|---|---|
| Provider / steward | iVolatility (ivolatility.com) |
| Product | Raw IV — individual option contracts with prices, implied volatility, and Greeks |
| Tickers | NVDA, AMD, INTC |
| Date range | 2026-10-13 to 2026-10-15 (three trading days) |
| How it was obtained | Downloaded by the project owner under an iVolatility Individual license |
| Licence / terms | iVolatility Individual license, held by the project owner. The licence allows every use listed in this contract, provided the downloaded raw data is **not published or distributed to any other person**, in raw or reconstructable form. |
| Use of derived data | The owner holds written confirmation that anything derived from this data may be used for this project. Recorded 2026-11-08 on the owner's statement; the confirmation itself is held by the owner. |

### 2.3 Change control

Both datasets are fixed. They will never be updated, re-downloaded, extended, or modified, apart from this one recorded exception: on 2026-11-08 the owner replaced the Massive package with `nvda_amd_intc_goog_minute`, which adds GOOG and GOOGL. It comes from Massive under the same terms. The earlier folder `nvda_amd_intc_minute` stays on disk and is no longer read. Raw files are set read-only after download, and all cleaning happens in the derived layer (§1.3), which is regenerated from the unchanged raw files by code.

### 2.4 Public market reference

`configs/market_quote_goog.yaml` records one delayed public quote and closing price for the listed GOOG put used in `SPEC.md` §8.4, with its source and time. It is not licensed data under §2.1 or §2.2. It is **Public**, cited reference data, and it may appear with its absolute prices. No other market quote is stored in the repository.

---

## 3. Allowed Paths

"Agent" means any AI tool used on this project: Claude in the claude.ai chat interface and Claude Code in VS Code.

### 3.1 Path permissions

| Path | Contents | Class | Agent read | Agent write | Git |
|---|---|---|---|---|---|
| `data/**` | Raw Massive and iVolatility files; all derived Restricted datasets | Restricted | **Headers and structure only** (column names, types, row counts) — never values | **No** | Ignored |
| `results/estimates.json`, `results/calibration.json` | Aggregated estimates and fitted parameters | Internal | Yes | No (written only by pipeline scripts) | Committed |
| `results/**` (other) | Run logs, summary CSVs, figures | Internal | Yes | No (written only by pipeline scripts) | Committed (small files only) |
| `tests/fixtures/**` | Synthetic fixtures | Public | Yes | Yes | Committed |
| `python/**`, `tests/**` | Source code and tests | Public | Yes | Yes (see §3.3) | Committed |
| `prototype/**` | Exploratory notebooks | Internal | Yes | Yes | Committed with **outputs cleared** |
| `README.md`, `AI_USE.md`, `DATA_CONTRACT.md`, configs | Documentation | Public | Yes | Yes (see §3.3) | Committed |
| `.env`, any credentials file | Secrets | Secret | **No** | **No** | Ignored / outside repo |

### 3.2 Enforcement

`.gitignore`:

```gitignore
data/
.env
```

Claude Code project settings (`.claude/settings.json`). The agent cannot open, edit, or create anything under `data/`. It learns headers and data structure only through `scripts/inspect_schema.py`, which prints file names, column names, dtypes, row counts, and null counts, and never prints a data value.

```json
{
  "permissions": {
    "allow": [
      "Bash(python scripts/inspect_schema.py:*)"
    ],
    "deny": [
      "Read(./data/**)",
      "Edit(./data/**)",
      "Write(./data/**)",
      "Read(./.env)"
    ]
  }
}
```

**Known gap:** file-tool deny rules do not stop a shell command (`cat`, `head`, `zcat`, a Python one-liner) from reading the same files. The owner reviews every agent-proposed shell command touching `data/` before approving it. A pre-commit hook (`scripts/check_no_data.sh`) also rejects any commit containing a path under `data/`, which catches `git add -f`.

### 3.3 Agent commits, pushes, and pipeline runs

Granted by the owner on 2026-11-08.

- **Commits and pushes.** An agent may commit and push when the owner asks, but only if `data/` is listed in `.gitignore` and `scripts/check_no_data.sh` passes immediately before the push. Owner review of agent-written files may follow the commit.
- **Licensed-data pipeline.** An agent may run the licensed-data pipeline only through the committed scripts `python/validate_raw.py`, `scripts/run.py`, and `scripts/verify_outputs.py`. Those scripts read the files; the agent sees only their output, which is Internal.
- **Unchanged.** An agent still may not read, print, or sample any value under `data/`, by these scripts or any other means. Classification in §4 is unchanged: raw data, quotes, and market IV points stay Restricted.

---

## 4. Classification

### 4.1 Levels

| Level | Definition | Examples in this project |
|---|---|---|
| **Public** | Can be shared with anyone, including in a public repository. | Source code, synthetic fixtures, configs, this contract, published Heston parameters from cited papers. |
| **Internal** | Derived from licensed data but aggregated so that the licensed data cannot be reconstructed. May be shared with the instructor and graders as part of the graded work, and may appear in AI prompts and logs. | Correlation matrix, realised volatility, fitted Heston parameters, aggregate RMS fit residuals, row counts, validation pass/fail counts, final prices and standard errors. |
| **Restricted** | Licensed Massive or iVolatility data, or anything from which it can be reconstructed. Stays on the owner's laptop only. | Raw minute bars, option quotes and IVs, session closes, return series, the calibration grid, any figure that plots these values. |
| **Secret** | Credentials. | Massive and iVolatility account credentials, API keys, GitHub tokens. |

### 4.2 What may appear where

**No raw data goes to an AI tool, the instructor, or a grader.**

| Destination | Public | Internal | Restricted | Secret |
|---|---|---|---|---|
| AI prompts (claude.ai, Claude Code) | Yes | Yes | **No** — not even a few rows | **No** |
| Instructor, TAs, graders | Yes | Yes | **No** | **No** |
| Terminal output / run logs in `results/` | Yes | Yes | **No** | **No** |
| Git history (any commit, any branch) | Yes | Yes | **No** | **No** |
| D2L submission ZIP / bundle | Yes | Yes | **No** | **No** |
| Report and slides | Yes | Yes | **No** | **No** |
| Owner's laptop, under `data/` | Yes | Yes | Yes | No (credentials stay outside the repo) |

The D2L submission is built from Public and Internal material only. A reviewer reproduces the pipeline on the synthetic fixtures (§1.4); the market-calibrated result is reported through `results/estimates.json` and `results/calibration.json`.

### 4.3 Verbatim rules for prompts and logs

- **Allowed verbatim:** ticker symbols, date ranges, file names, column names, dtypes, row counts, rule names, violation counts, row indices, aggregated statistics, model outputs.
- **Never verbatim:** any row of a Restricted dataset; any price, quote, or IV tied to a specific ticker and timestamp or contract; DataFrame previews (`head`, `tail`, `sample`, printing a DataFrame); stack traces or pandas error messages that echo row contents.
- **Debugging rule:** to get AI help with a data bug, reproduce it on a synthetic fixture first and share that. If it only occurs on real data, describe it by file name, row index, column, and rule name.
- **Logging rule:** pipeline code catches exceptions raised while reading Restricted data and re-raises a message containing only the file name, row index, and rule name.
- **Figures:** may show Internal quantities (correlation heatmaps, fitted parameters, convergence plots, price estimates with error bars, synthetic-fixture surfaces). No figure may plot Restricted values — market prices, market IV points, or per-contract fit residuals — since any of these lets the licensed data be reconstructed. Fit quality on market data is reported as aggregate statistics.

### 4.4 Stop conditions

Work on licensed data stops, and the project falls back to the synthetic fixtures and published Heston parameters, if:

1. Restricted data is found in a Git commit, an AI prompt, a log, or a submission file. If it reached Git history, contact the instructor immediately; deleting it in a later commit does not remove it from history.
2. Either licence ends, or its terms change in a way that removes the right to use the data for this project.

---

## 5. Sign-off

| Item | Status |
|---|---|
| Raw files set read-only (§2.3) | ☐ |
| `.gitignore`, Claude Code settings, `scripts/inspect_schema.py`, and pre-commit hook in place (§3.2) | ☐ |
| Synthetic fixtures generated and validated against §1.1 and §1.2 | ☐ |
| Proposal's "commit the snapshot" plan withdrawn in README and report | ☐ |
