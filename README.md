# NSE Quant

**A risk-adjusted equity valuation and portfolio analytics platform for the Kenyan stock market.**

NSE Quant is a small Streamlit web app and Python library for studying the Nairobi Securities
Exchange. It covers market overview, company valuation, risk, cross-sectional factors and portfolio
construction, plus a set of reproducible research notes generated from the same code.

## Features

| Page | What it shows |
|---|---|
| **Market overview** | Market proxy index (liquidity- or equal-weighted), sector performance, rolling volatility, market breadth (advance–decline line, % above 200-day average), correlation heat map, net foreign flows |
| **Company analysis** | Price history, risk profile, financial statements and ratios (profitability, growth, leverage, multiples), DCF with sensitivity grid, Gordon and two-stage dividend discount models |
| **Risk** | Volatility, OLS and Dimson beta, VaR and CVaR (historical, Gaussian, Cornish–Fisher, EVT peaks-over-threshold), drawdowns, historical-scenario and beta stress tests |
| **Quant factors** | Momentum, low volatility, size, value and quality as long–short tercile portfolios in a liquidity-screened universe, with t-stats |
| **Portfolio** | Long-only efficient frontier with Ledoit–Wolf shrinkage, max-Sharpe and min-variance portfolios, factor exposures, block-bootstrap Monte Carlo |
| **Research** | Research notes generated from the data (see [`research/`](research/)) |
| **Data quality** | Every correction the cleaning pipeline made to the raw export, raw vs. clean price charts |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # fish: source .venv/bin/activate.fish
pip install -r requirements.txt
streamlit run app.py
```

Regenerate the research notes (takes a few minutes; the tail-risk backtest refits an EVT model across the sample):

```bash
python -m research.build
```

Run the tests:

```bash
pytest
```

## Data

| File | Contents |
|---|---|
| `data/raw/*.csv` | Daily broker exports: date, security, close (KES), volume, net foreign flow, % change. Every CSV in the folder is loaded and merged. The repo ships with a Rock Investment Bank export covering Jun 2017 – Jun 2026; add newer exports alongside it. |
| `data/sectors.csv` | NSE sector classification per security (editable). |
| `data/corporate_actions.csv` | Confirmed splits, bonus issues and consolidations, used to back-adjust prices. Ships empty. |
| `data/fundamentals.csv` | Annual financial statements per company. Ships as a template. Fill it from annual reports to unlock ratios, multiples, and the Value and Quality factors. |

### Cleaning

The raw export needs work before it can be used for returns:

* **Unquoted thousands separators** in the foreign-flow column mean the file cannot be read with a normal CSV
  parser. Each line is anchored on the `KES` price field instead.
* **Scrambled export days.** Days where more than 10% of stocks "jump" by more than 30% at once are dropped.
* **One-day spikes** (×100 decimal errors, zero prices) that revert within ten sessions are blanked.
* **Unexplained jumps.** Any other single-day move beyond ±30% is kept in prices but excluded from returns. The NSE
  caps most daily moves at about ±10%, so these are errors or corporate actions. Once a move is confirmed as a split
  or bonus in `corporate_actions.csv`, prices are back-adjusted and the return is restored.

The Data Quality page lists every correction.

### Known limitations

* **No official index series.** The export has no NASI or NSE 20 index, so the market is rebuilt from
  constituents. It is weighted by trailing traded value until shares-outstanding data is available.
* **Size factor.** Without shares outstanding, size is proxied by average traded value.
* **Thin trading.** Many counters trade on only a fraction of days. Returns are measured from the last traded price,
  betas use the Dimson lag correction, and factor tests use a liquidity-screened universe.
* **Small cross-section.** About 20 stocks pass the liquidity screen in a typical month, so factor results carry wide
  error bands.

## Project layout

```
app.py                  Streamlit dashboard
nsequant/
  data.py               parsing, cleaning, returns, market proxy, sectors
  risk.py               volatility, beta, VaR/CVaR, EVT, drawdown, stress tests
  factors.py            factor scores, liquid universe, long-short factor returns
  portfolio.py          shrinkage covariance, optimisation, efficient frontier, Monte Carlo
  valuation.py          ratios, multiples, DCF, dividend discount models
research/
  build.py              generates the research notes and figures
  *.md                  research notes
tests/                  pytest suite
data/                   raw exports and reference tables
```

## Disclaimer

For research and education only. Nothing here is investment advice.
