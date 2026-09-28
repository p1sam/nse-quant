"""Cross-sectional equity factors and long-short factor portfolios.

Price-based factors (momentum, low volatility, size/liquidity) come straight
from the price history. Value and quality need data/fundamentals.csv and are
skipped for companies without it. Scores are cross-sectional z-scores each
month-end, signed so that a high score means "the factor likes this stock".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data, valuation

FACTORS = ["Momentum", "Low volatility", "Size", "Value", "Quality"]


def monthly_prices() -> pd.DataFrame:
    px, _ = data.clean_prices()
    return px.ffill(limit=21).resample("ME").last()


def monthly_returns() -> pd.DataFrame:
    """Month-end to month-end returns from daily returns (so excluded jumps stay excluded)."""
    r = data.returns()
    m = (1 + r.fillna(0)).resample("ME").prod() - 1
    return m.where(r.resample("ME").count() > 0)


def traded_value(window: int = 63) -> pd.DataFrame:
    px, _ = data.clean_prices()
    vol = data.pivot("volume").reindex_like(px)
    tv = (px.ffill() * vol).rolling(window, min_periods=window // 3).mean()
    return tv.resample("ME").last()


def liquid_universe(min_traded_value: float = 1e6, min_trading_share: float = 0.5) -> pd.DataFrame:
    """Boolean month x stock mask: averages >= KES 1m/day and traded on >= half of the last 63 days."""
    px, _ = data.clean_prices()
    vol = data.pivot("volume").reindex_like(px)
    traded = (vol > 0).rolling(63, min_periods=21).mean().resample("ME").last()
    ok = (traded_value() >= min_traded_value) & (traded >= min_trading_share)
    # ETFs and REITs are not equities for factor purposes.
    keep = [c for c in ok.columns if data.board_of(c) not in ("ETF", "REIT")]
    return ok[keep]


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)


def scores() -> dict[str, pd.DataFrame]:
    mp = monthly_prices()
    r = data.returns()
    out = {
        # 12-1 momentum: past 12 months skipping the most recent month (short-term reversal).
        "Momentum": mp.shift(1) / mp.shift(12) - 1,
        "Low volatility": -(r.rolling(252, min_periods=126).std().resample("ME").last()),
        # No shares-outstanding data in the price export, so traded value stands in
        # for size. Small = high score, as in the classic SMB factor.
        "Size": -np.log(traded_value().replace(0, np.nan)),
    }
    f = valuation.load_fundamentals()
    if len(f):
        rat = valuation.ratios(f)
        # Assume annual reports are public ~4 months after fiscal year-end.
        rat["avail"] = pd.to_datetime(rat["fiscal_year"].astype(int).astype(str) + "-12-31") + pd.offsets.MonthEnd(4)
        bvps = rat.pivot_table(index="avail", columns="name", values="BVPS")
        eps = rat.pivot_table(index="avail", columns="name", values="EPS")
        roe = rat.pivot_table(index="avail", columns="name", values="ROE")
        idx = mp.index
        bvps, eps, roe = (x.reindex(idx.union(x.index)).ffill().reindex(idx) for x in (bvps, eps, roe))
        cols = mp.columns.intersection(bvps.columns)
        out["Value"] = zscore(bvps[cols] / mp[cols]).add(zscore(eps[cols] / mp[cols]), fill_value=0) / 2
        out["Quality"] = roe[cols]
    return out


def factor_returns(n_groups: int = 3, min_names: int = 6) -> pd.DataFrame:
    """Monthly long-short returns: top group minus bottom group, equal weighted.

    Terciles rather than quintiles because the liquid NSE universe is often
    fewer than 20 names. Months with fewer than min_names eligible stocks are skipped.
    """
    mr = monthly_returns()
    uni = liquid_universe()
    out = {}
    for name, s in scores().items():
        s = s.reindex(index=mr.index, columns=uni.columns).where(uni.reindex(mr.index).fillna(False).astype(bool))
        fwd = mr.shift(-1).reindex(columns=s.columns)
        rets = {}
        for d in s.index:
            x = pd.concat([s.loc[d], fwd.loc[d]], axis=1, keys=["s", "r"]).dropna()
            if len(x) < min_names:
                continue
            g = pd.qcut(x["s"].rank(method="first"), n_groups, labels=False)
            rets[d + pd.offsets.MonthEnd(1)] = x["r"][g == n_groups - 1].mean() - x["r"][g == 0].mean()
        out[name] = pd.Series(rets, dtype=float)
    return pd.DataFrame(out).sort_index()


def factor_stats(fr: pd.DataFrame) -> pd.DataFrame:
    n = fr.count()
    mean = fr.mean()
    se = fr.std() / np.sqrt(n)
    return pd.DataFrame({
        "Months": n,
        "Mean monthly": mean,
        "Annualised": (1 + mean) ** 12 - 1,
        "Volatility (ann.)": fr.std() * np.sqrt(12),
        "Sharpe (ann.)": mean / fr.std() * np.sqrt(12),
        "t-stat": mean / se,
        "Hit rate": (fr > 0).sum() / n,
    })


def latest_exposures(weights: pd.Series) -> pd.Series:
    """Weighted-average z-score of a portfolio on each factor at the latest month."""
    out = {}
    for name, s in scores().items():
        z = zscore(s).iloc[-1].reindex(weights.index)
        ok = z.notna()
        out[name] = float((z[ok] * weights[ok]).sum() / weights[ok].sum()) if ok.any() else np.nan
    return pd.Series(out)


def regression_exposures(port_monthly: pd.Series, fr: pd.DataFrame) -> pd.DataFrame:
    """OLS of portfolio excess-of-mean monthly returns on factor returns (time-series betas)."""
    d = pd.concat([port_monthly.rename("p"), fr], axis=1).dropna(axis=1, thresh=24).dropna()
    if len(d) < 24:
        return pd.DataFrame()
    X = np.column_stack([np.ones(len(d)), d.drop(columns="p").to_numpy()])
    y = d["p"].to_numpy()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    s2 = resid @ resid / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    names = ["Alpha (monthly)", *d.columns.drop("p")]
    return pd.DataFrame({"coef": coef, "t-stat": coef / se}, index=names)
