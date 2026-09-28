"""Regenerate the research papers and their figures from the current data.

    python -m research.build

Each paper is a Markdown file in research/ whose numbers are computed here,
so re-running after adding data (or corporate actions) updates every figure
and table.
"""
from __future__ import annotations

import warnings
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from nsequant import data, factors, risk

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = Path(__file__).parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
BLUE, ORANGE, AQUA, RED, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#8a8984"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 9, "lines.linewidth": 1.6})


def newey_west_t(x: pd.Series, lags: int = 6) -> float:
    x = x.dropna().to_numpy()
    n, e = len(x), x - x.mean()
    s = e @ e / n
    for k in range(1, lags + 1):
        s += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return x.mean() / np.sqrt(s / n)


def fmt_table(df: pd.DataFrame, fmts: dict) -> str:
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [fmts.get(c, "{}").format(v) if pd.notna(v) else "—" for c, v in row.items()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- momentum
def momentum_paper() -> str:
    mp = factors.monthly_prices()
    mr = factors.monthly_returns()
    uni = factors.liquid_universe()
    mkt = data.index_level("liquidity").resample("ME").last().pct_change()

    def long_short(lookback: int, skip: int = 1, groups: int = 3):
        sig = mp.shift(skip) / mp.shift(lookback) - 1
        sig = sig.reindex(columns=uni.columns).where(uni.reindex(sig.index).fillna(False).astype(bool))
        fwd = mr.shift(-1).reindex(columns=sig.columns)
        rows = {}
        for d in sig.index:
            x = pd.concat([sig.loc[d], fwd.loc[d]], axis=1, keys=["s", "r"]).dropna()
            if len(x) < 6:
                continue
            g = pd.qcut(x["s"].rank(method="first"), groups, labels=False)
            rows[d + pd.offsets.MonthEnd(1)] = {"Winners": x["r"][g == groups - 1].mean(),
                                                 "Losers": x["r"][g == 0].mean(), "N": len(x)}
        out = pd.DataFrame(rows).T
        out["WML"] = out["Winners"] - out["Losers"]
        return out

    specs = {"12-1": (12, 1), "6-1": (6, 1), "3-1": (3, 1), "12-0 (no skip)": (12, 0), "1-0 (reversal)": (1, 0)}
    res = {k: long_short(*v) for k, v in specs.items()}
    main = res["12-1"]

    tbl = pd.DataFrame({k: {
        "Months": v["WML"].count(),
        "Mean WML / month": v["WML"].mean(),
        "Annualised": (1 + v["WML"].mean()) ** 12 - 1,
        "t (OLS)": v["WML"].mean() / (v["WML"].std() / np.sqrt(v["WML"].count())),
        "t (Newey–West)": newey_west_t(v["WML"]),
        "Hit rate": (v["WML"] > 0).mean(),
    } for k, v in res.items()}).T
    tbl.index.name = "Formation"

    # CAPM alpha of WML
    d = pd.concat([main["WML"], mkt.rename("M")], axis=1, sort=True).dropna()
    b = np.polyfit(d["M"], d["WML"], 1)
    resid = d["WML"] - np.polyval(b, d["M"])
    se_a = resid.std(ddof=2) * np.sqrt(1 / len(d) + d["M"].mean() ** 2 / ((d["M"] - d["M"].mean()) ** 2).sum())

    # Sub-periods
    halves = {"2018–2021": main.loc[:"2021", "WML"], "2022–2026": main.loc["2022":, "WML"]}
    sub = pd.DataFrame({k: {"Months": v.count(), "Mean / month": v.mean(), "t (Newey–West)": newey_west_t(v)}
                        for k, v in halves.items()}).T
    sub.index.name = "Period"

    # Turnover of the winner leg (share of names replaced each month)
    sig = mp.shift(1) / mp.shift(12) - 1
    sig = sig.reindex(columns=uni.columns).where(uni.reindex(sig.index).fillna(False).astype(bool))
    tops = [set(r.dropna().nlargest(max(2, r.count() // 3)).index) for _, r in sig.iterrows() if r.count() >= 6]
    turnover = np.mean([1 - len(a & b) / len(b) for a, b in zip(tops[:-1], tops[1:]) if b])

    fig, ax = plt.subplots(figsize=(7, 3.2))
    for col, c in [("Winners", BLUE), ("Losers", RED)]:
        ax.plot((1 + main[col].fillna(0)).cumprod(), color=c, label=f"{col} (top/bottom tercile)")
    ax.plot((1 + main["WML"].fillna(0)).cumprod(), color=GREY, ls="--", label="Winners − Losers")
    ax.set_ylabel("Growth of KES 1")
    ax.legend(frameon=False)
    ax.set_title("12-1 momentum terciles, liquid NSE universe")
    fig.tight_layout()
    fig.savefig(FIG / "momentum_cumulative.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.bar(main.index, main["WML"], width=20, color=[BLUE if v >= 0 else RED for v in main["WML"]])
    ax.set_ylabel("Monthly WML return")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Monthly winners-minus-losers return (12-1)")
    fig.tight_layout()
    fig.savefig(FIG / "momentum_monthly.png")
    plt.close(fig)

    m = tbl.loc["12-1"]
    verdict = ("positive and statistically significant" if abs(m["t (Newey–West)"]) >= 2 and m["Mean WML / month"] > 0
               else "positive but not statistically significant" if m["Mean WML / month"] > 0
               else "negative")
    return f"""# Does Momentum Work on the Nairobi Securities Exchange?

*NSE Quant research note · generated {date.today():%d %B %Y} from daily prices {main.index.min():%b %Y}–{main.index.max():%b %Y}*

## Abstract

We test the cross-sectional momentum effect of Jegadeesh & Titman (1993) on NSE-listed equities.
Each month we rank liquid stocks by their past 12-month return, skipping the most recent month, and
hold an equal-weighted portfolio of the top tercile against the bottom tercile. Over
{int(m['Months'])} months the winners-minus-losers (WML) portfolio earned **{m['Mean WML / month']:.2%} per month**
({m['Annualised']:.1%} annualised) with a Newey–West t-statistic of **{m['t (Newey–West)']:.2f}** — {verdict}.
The shortest horizon is the strongest pattern in the data. Last month's losers beat last month's winners by
{-tbl.loc['1-0 (reversal)', 'Mean WML / month']:.2%} a month (Newey–West t = {tbl.loc['1-0 (reversal)', 't (Newey–West)']:.2f}),
consistent with stale prices and bid–ask bounce. Our reading is that the NSE shows at most a weak intermediate-horizon momentum
premium. Too few stocks trade, and the history is too short, to distinguish it from zero with confidence.

## 1. Data and universe

Daily closing prices, volumes and net foreign flows for {data.load_long()['name'].nunique()} NSE securities come from a broker
export. The export is noisy. The cleaning pipeline (`nsequant.data.clean_prices`) drops scrambled export days, blanks out
one-day price spikes that revert, and leaves out of returns any unexplained single-day move above 30%. That last rule
matters because the NSE caps most daily moves at ±10%, so a larger move is an error or a corporate action. Every correction is
listed on the dashboard's Data Quality page.

Momentum is only tradable where prices are real. Each month the universe is restricted to stocks with at least KES 1 million
of average daily traded value over the past quarter that also traded on at least half of those days. ETFs and REITs are excluded.
The universe holds a median of **{int(main['N'].median())} stocks** (range {int(main['N'].min())}–{int(main['N'].max())}), so
we use terciles rather than the deciles common in US studies.

## 2. Method

At each month-end *t* the signal is the return from *t−12* to *t−1*, skipping month *t* to avoid the short-term reversal effect.
Stocks are sorted into terciles on the signal. The next month's return is measured for the equal-weighted top (Winners) and
bottom (Losers) groups. WML is their difference. Standard errors use Newey–West with six lags, to allow for the
serial correlation that overlapping formation windows and stale prices create.

## 3. Results

{fmt_table(tbl, {"Months": "{:.0f}", "Mean WML / month": "{:.2%}", "Annualised": "{:.1%}", "t (OLS)": "{:.2f}", "t (Newey–West)": "{:.2f}", "Hit rate": "{:.0%}"})}

![Cumulative returns of winner and loser terciles](figures/momentum_cumulative.png)

The 12-1 WML portfolio has a CAPM beta of **{b[0]:.2f}** against the liquidity-weighted market proxy and a monthly
alpha of **{b[1]:.2%}** (t = {b[1] / se_a:.2f}). The winner leg replaces about **{turnover:.0%}** of its names each month.

![Monthly WML returns](figures/momentum_monthly.png)

### Robustness: sub-periods

{fmt_table(sub, {"Months": "{:.0f}", "Mean / month": "{:.2%}", "t (Newey–West)": "{:.2f}"})}

## 4. Discussion

* **Statistical power is the binding constraint.** With about twenty names per month, each tercile holds six or seven stocks,
  so idiosyncratic noise dominates. At the observed WML volatility of {main['WML'].std():.1%} a month, a true premium of 1% a
  month would need about {(2 * main['WML'].std() / 0.01) ** 2 / 12:.0f} years of data to reach t = 2.
* **Short-horizon reversal.** The 1-0 row shows how last month's return predicts the next one. On the NSE, bid–ask bounce and
  stale prices push this towards reversal, which is why the skip month matters.
* **Implementability.** Short-selling is not practically available on the NSE, so the long-only Winners leg is the relevant
  portfolio for a local investor. At the observed turnover, round-trip costs of roughly 2–3% (brokerage, levies and spread
  on illiquid names) would absorb much of any premium.
* **Survivorship.** The export includes suspended and delisted counters (e.g. Mumias, ARM Cement, Uchumi), so the universe
  is not survivorship-filtered. Their stale prices are excluded by the liquidity screen, not dropped from the sample.

## 5. Conclusion

Momentum on the NSE is {verdict} over this sample. The evidence is not strong enough to trade on without
further out-of-sample data, and costs are likely to consume much of the gross premium. Extending the sample back before 2017 and
adding weekly-rebalanced variants are the obvious next steps.

## References

* Jegadeesh, N. & Titman, S. (1993). Returns to buying winners and selling losers. *Journal of Finance*, 48(1).
* Asness, C., Moskowitz, T. & Pedersen, L. (2013). Value and momentum everywhere. *Journal of Finance*, 68(3).
* Newey, W. & West, K. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3).

*For research and education only; not investment advice.*
"""


# --------------------------------------------------------------------------- tail risk
def kupiec(exceed: np.ndarray, p: float) -> tuple[float, float]:
    n, x = len(exceed), int(exceed.sum())
    if x in (0, n):
        return np.nan, np.nan
    lr = -2 * (np.log((1 - p) ** (n - x) * p**x) - np.log((1 - x / n) ** (n - x) * (x / n) ** x))
    return lr, 1 - stats.chi2.cdf(lr, 1)


def tail_paper() -> str:
    r = data.returns()
    uni = factors.liquid_universe()
    liquid = [c for c in uni.columns[uni.iloc[-24:].mean() > 0.5] if c in r.columns]
    port = r[liquid].mean(axis=1, skipna=True).dropna()
    mkt = data.index_level("liquidity").pct_change().dropna()
    series = {"Market proxy (liquidity-weighted)": mkt, f"Equal-weighted liquid portfolio ({len(liquid)} stocks)": port}

    rows, fits = [], {}
    for name, x in series.items():
        e = risk.evt_tail(x, 0.99)
        e995 = risk.evt_tail(x, 0.995)
        fits[name] = e
        rows.append({"Series": name, "Skew": stats.skew(x), "Excess kurtosis": stats.kurtosis(x), "GPD ξ": e["xi"],
                     "VaR99 Gaussian": risk.var_parametric(x, 0.99), "VaR99 Historical": risk.var_historical(x, 0.99),
                     "VaR99 EVT": e["var"], "ES99 Gaussian": risk.cvar_parametric(x, 0.99),
                     "ES99 Historical": risk.cvar_historical(x, 0.99), "ES99 EVT": e["es"],
                     "VaR99.5 EVT": e995["var"]})
    t1 = pd.DataFrame(rows).set_index("Series")

    # Rolling one-day-ahead VaR backtest, 500-day window.
    W, P = 500, 0.01
    x = mkt
    bt = {"Gaussian": [], "Historical": [], "EVT": []}
    for i in range(W, len(x)):
        w = x.iloc[i - W:i]
        bt["Gaussian"].append(risk.var_parametric(w, 1 - P))
        bt["Historical"].append(risk.var_historical(w, 1 - P))
        # The GPD fit is the slow step; refit weekly and carry the estimate between refits.
        if (i - W) % 5 == 0:
            evt_var = risk.evt_tail(w, 1 - P, 0.90)["var"]
        bt["EVT"].append(evt_var)
    realised = -x.iloc[W:].to_numpy()
    bt_rows = {}
    for k, v in bt.items():
        ex = realised > np.array(v)
        lr, pv = kupiec(ex, P)
        bt_rows[k] = {"Days": len(ex), "Expected breaches": len(ex) * P, "Actual breaches": int(ex.sum()),
                      "Breach rate": ex.mean(), "Kupiec LR": lr, "p-value": pv}
    t2 = pd.DataFrame(bt_rows).T
    t2.index.name = "Model"

    # Figures: tail QQ vs normal, and rolling VaR vs realised
    fig, ax = plt.subplots(figsize=(7, 3.0))
    loss = np.sort(-mkt.to_numpy())
    q = stats.norm.ppf((np.arange(1, len(loss) + 1) - 0.5) / len(loss))
    ax.scatter(q, (loss - loss.mean()) / loss.std(), s=6, color=BLUE)
    lim = [q.min(), q.max()]
    ax.plot(lim, lim, color=GREY, ls="--", label="Normal")
    ax.set_xlabel("Normal quantile")
    ax.set_ylabel("Standardised loss quantile")
    ax.set_title("Market proxy daily losses vs the normal distribution")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "tail_qq.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.0))
    idx = x.index[W:]
    ax.plot(idx, -realised, color=GREY, lw=0.6, label="Daily return")
    ax.plot(idx, -np.array(bt["Gaussian"]), color=ORANGE, label="Gaussian VaR 99%")
    ax.plot(idx, -np.array(bt["EVT"]), color=BLUE, label="EVT VaR 99%")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Rolling 500-day one-day VaR backtest, market proxy")
    ax.legend(frameon=False, ncol=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "tail_backtest.png")
    plt.close(fig)

    pct2 = "{:.2%}"
    m = t1.iloc[0]
    under = m["ES99 EVT"] / m["ES99 Gaussian"] - 1
    return f"""# Estimating Tail Risk in Kenyan Equity Portfolios Using Extreme-Value Methods

*NSE Quant research note · generated {date.today():%d %B %Y} from daily prices {mkt.index.min():%b %Y}–{mkt.index.max():%b %Y}*

## Abstract

Kenyan equity returns are strongly non-normal. Daily losses on a liquidity-weighted NSE market proxy have excess kurtosis of
**{m['Excess kurtosis']:.1f}**. A Generalised Pareto fit to the loss tail gives a shape parameter ξ = **{m['GPD ξ']:.2f}**,
which implies a heavy, power-law tail. At the 99% level, extreme-value (EVT) expected shortfall exceeds the Gaussian estimate
by **{under:.0%}**. In a rolling out-of-sample backtest the Gaussian 99% VaR is breached on {t2.loc['Gaussian', 'Breach rate']:.2%} of days
(Kupiec p = {t2.loc['Gaussian', 'p-value']:.3f}). EVT and historical simulation breach on {t2.loc['EVT', 'Breach rate']:.2%} and
{t2.loc['Historical', 'Breach rate']:.2%} of days respectively.

## 1. Data

The market proxy is built from daily constituent returns weighted by trailing 60-day traded value, because the export carries no
NASI series. The second series is an equal-weighted portfolio of the {len(liquid)} stocks that passed the liquidity screen in most
of the last two years. Prices go through the cleaning pipeline described on the dashboard's Data Quality page. That step matters
for tail estimation: a single uncorrected ×100 decimal error would dominate any extreme-value fit.

## 2. Method

Following McNeil & Frey (2000), we take the losses above a high threshold *u* (the 95th percentile of losses) and fit a
Generalised Pareto Distribution by maximum likelihood. The fitted shape ξ and scale σ give VaR and expected shortfall
at confidence *q*:

VaR_q = u + (σ/ξ)·[((n/N_u)(1−q))^(−ξ) − 1],  ES_q = (VaR_q + σ − ξu)/(1 − ξ)

We compare these with Gaussian (mean–variance) and historical-simulation estimates.

## 3. Full-sample tail estimates (1-day)

{fmt_table(t1, {"Skew": "{:.2f}", "Excess kurtosis": "{:.1f}", "GPD ξ": "{:.2f}", **{c: pct2 for c in t1.columns if c.startswith(("VaR", "ES"))}})}

![QQ plot of losses against the normal](figures/tail_qq.png)

Points above the dashed line in the upper right are losses larger than a normal distribution would produce. This is the fat
tail the GPD captures.

## 4. Out-of-sample backtest

Each day we estimate 99% VaR from the previous 500 trading days (EVT here uses the 90th-percentile threshold so the window
has enough exceedances, and is refitted weekly) and count how often the next day's loss exceeds it. The Kupiec (1995) test asks whether the breach rate
is consistent with the nominal 1%. A p-value below 0.05 rejects the model.

{fmt_table(t2, {"Days": "{:.0f}", "Expected breaches": "{:.1f}", "Actual breaches": "{:.0f}", "Breach rate": "{:.2%}", "Kupiec LR": "{:.2f}", "p-value": "{:.3f}"})}

![Rolling VaR backtest](figures/tail_backtest.png)

## 5. Discussion

* **Thin trading smooths daily returns.** Stale prices spread a real shock over several days, which *reduces* measured daily
  tail risk. Multi-day losses are therefore larger than √h scaling of the 1-day VaR suggests. The dashboard's Monte Carlo uses a
  block bootstrap for this reason.
* **Regimes.** The largest losses cluster in the 2020 COVID sell-off and in the 2022–2024 currency and rate stress. A
  conditional (GARCH-filtered) EVT model would respond faster to these regimes. That is the natural next step.
* **Small samples in the tail.** With about {fits[next(iter(fits))]['n_exceed']} exceedances, ξ is estimated with wide
  error bands. Reported EVT figures should be read as ranges, not points.

## 6. Conclusion

Gaussian risk models materially understate tail losses on the NSE. At the 99% level, EVT and historical simulation perform
about equally out of sample. Both sit near the edge of acceptance, which suggests a conditional-volatility model is still needed.
EVT's advantage lies further out in the tail: unlike historical simulation, it can extrapolate beyond the worst observed day
(e.g. the 99.5% VaR above). For that reason the dashboard reports it next to the standard VaR and CVaR figures.

## References

* McNeil, A. & Frey, R. (2000). Estimation of tail-related risk measures for heteroscedastic financial time series: an extreme value approach. *Journal of Empirical Finance*, 7(3–4).
* Kupiec, P. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*, 3(2).
* Embrechts, P., Klüppelberg, C. & Mikosch, T. (1997). *Modelling Extremal Events for Insurance and Finance*. Springer.

*For research and education only; not investment advice.*
"""


PAPERS = {
    "01_momentum_nse.md": momentum_paper,
    "02_tail_risk_evt.md": tail_paper,
}

if __name__ == "__main__":
    for fname, fn in PAPERS.items():
        (OUT / fname).write_text(fn(), encoding="utf-8")
        print("wrote", OUT / fname)
