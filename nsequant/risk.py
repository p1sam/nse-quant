"""Risk measures for daily return series.

All functions take simple daily returns and report annualised figures using
252 trading days unless stated otherwise. VaR and CVaR are returned as positive
loss fractions (0.03 = a 3% loss).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TDAYS = 252


def ann_vol(r: pd.Series | pd.DataFrame):
    return r.std() * np.sqrt(TDAYS)


def ann_return(r: pd.Series | pd.DataFrame):
    """Geometric annualised return, using the observations actually present."""
    growth = (1 + r.fillna(0)).prod()
    years = r.count() / TDAYS
    return growth ** (1 / years) - 1


def rolling_vol(r: pd.Series | pd.DataFrame, window: int = 63):
    return r.rolling(window, min_periods=window // 2).std() * np.sqrt(TDAYS)


def beta(r: pd.Series, m: pd.Series) -> float:
    d = pd.concat([r, m], axis=1).dropna()
    if len(d) < 30:
        return np.nan
    c = np.cov(d.iloc[:, 0], d.iloc[:, 1])
    return c[0, 1] / c[1, 1]


def dimson_beta(r: pd.Series, m: pd.Series, lags: int = 2) -> float:
    """Sum of slopes on the contemporaneous and lagged market returns.

    Thinly traded stocks react to market news with a delay, which pushes plain
    OLS betas towards zero. Adding lagged market terms (Dimson, 1979) corrects
    for most of that.
    """
    X = pd.concat({f"m{k}": m.shift(k) for k in range(lags + 1)}, axis=1)
    d = pd.concat([r.rename("r"), X], axis=1).dropna()
    if len(d) < 60:
        return np.nan
    A = np.column_stack([np.ones(len(d)), d.drop(columns="r").to_numpy()])
    coef, *_ = np.linalg.lstsq(A, d["r"].to_numpy(), rcond=None)
    return float(coef[1:].sum())


def var_historical(r: pd.Series, level: float = 0.95) -> float:
    return float(-np.nanquantile(r, 1 - level))


def cvar_historical(r: pd.Series, level: float = 0.95) -> float:
    r = r.dropna()
    q = np.quantile(r, 1 - level)
    return float(-r[r <= q].mean())


def var_parametric(r: pd.Series, level: float = 0.95) -> float:
    return float(-(r.mean() + stats.norm.ppf(1 - level) * r.std()))


def cvar_parametric(r: pd.Series, level: float = 0.95) -> float:
    a = 1 - level
    return float(-(r.mean() - r.std() * stats.norm.pdf(stats.norm.ppf(a)) / a))


def var_cornish_fisher(r: pd.Series, level: float = 0.95) -> float:
    """Normal VaR corrected for skewness and excess kurtosis."""
    r = r.dropna()
    z = stats.norm.ppf(1 - level)
    s, k = stats.skew(r), stats.kurtosis(r)
    zcf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
    return float(-(r.mean() + zcf * r.std()))


def evt_tail(r: pd.Series, level: float = 0.99, threshold_q: float = 0.95) -> dict:
    """Peaks-over-threshold VaR / ES from a Generalised Pareto fit to losses.

    Losses above the threshold_q quantile are fitted with a GPD; the tail
    estimator of McNeil & Frey (2000) then gives VaR and ES at `level`.
    """
    loss = -r.dropna().to_numpy()
    u = np.quantile(loss, threshold_q)
    exc = loss[loss > u] - u
    xi, _, sigma = stats.genpareto.fit(exc, floc=0)
    n, nu = len(loss), len(exc)
    var = u + sigma / xi * (((n / nu) * (1 - level)) ** (-xi) - 1) if xi != 0 else u - sigma * np.log((n / nu) * (1 - level))
    es = (var + sigma - xi * u) / (1 - xi) if xi < 1 else np.inf
    return {"xi": xi, "sigma": sigma, "threshold": u, "n_exceed": nu, "var": var, "es": es}


def drawdown(r: pd.Series) -> pd.Series:
    wealth = (1 + r.fillna(0)).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown(r: pd.Series) -> float:
    return float(drawdown(r).min())


def sharpe(r: pd.Series, rf: float = 0.0) -> float:
    ex = r - rf / TDAYS
    return float(ex.mean() / ex.std() * np.sqrt(TDAYS))


def sortino(r: pd.Series, rf: float = 0.0) -> float:
    ex = r - rf / TDAYS
    down = ex[ex < 0].std()
    return float(ex.mean() / down * np.sqrt(TDAYS))


# Historical windows used for stress testing, chosen from market events that
# fall inside the price history.
SCENARIOS = {
    "COVID-19 crash (Feb–Apr 2020)": ("2020-02-20", "2020-04-30"),
    "Interest-rate-cap repeal & 2019 banking sell-off (Oct–Dec 2019)": ("2019-10-01", "2019-12-31"),
    "Eurobond / FX stress (Jan–Jul 2022)": ("2022-01-01", "2022-07-31"),
    "Shilling sell-off & rate hikes (Jun 2023–Jan 2024)": ("2023-06-01", "2024-01-31"),
    "Finance Bill protests (Jun–Jul 2024)": ("2024-06-18", "2024-07-31"),
}


def historical_stress(r: pd.Series) -> pd.DataFrame:
    rows = []
    for name, (a, b) in SCENARIOS.items():
        w = r.loc[a:b]
        if w.count() < 5:
            continue
        rows.append({"scenario": name, "return": (1 + w.fillna(0)).prod() - 1, "max drawdown": max_drawdown(w)})
    return pd.DataFrame(rows)


def beta_stress(betas: pd.Series, weights: pd.Series, shocks=(-0.10, -0.20, -0.30)) -> pd.DataFrame:
    """Portfolio P&L if the market falls by each shock and stocks move with beta."""
    b = float((betas.reindex(weights.index).fillna(1.0) * weights).sum())
    return pd.DataFrame({"market shock": shocks, "portfolio beta": b, "expected portfolio move": [b * s for s in shocks]})


def summary(r: pd.Series, m: pd.Series | None = None, rf: float = 0.0) -> dict:
    out = {
        "Annual return": ann_return(r),
        "Annual volatility": ann_vol(r),
        "Sharpe": sharpe(r, rf),
        "Sortino": sortino(r, rf),
        "Max drawdown": max_drawdown(r),
        "VaR 95% (1d, hist)": var_historical(r),
        "CVaR 95% (1d, hist)": cvar_historical(r),
        "VaR 99% (1d, hist)": var_historical(r, 0.99),
        "Skew": float(stats.skew(r.dropna())),
        "Excess kurtosis": float(stats.kurtosis(r.dropna())),
    }
    if m is not None:
        out["Beta (OLS)"] = beta(r, m)
        out["Beta (Dimson)"] = dimson_beta(r, m)
    return out
