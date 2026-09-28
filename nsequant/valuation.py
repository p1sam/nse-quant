"""Fundamental valuation: DCF, dividend discount models, multiples and ratios.

Fundamentals come from data/fundamentals.csv (one row per company per fiscal
year, in KES millions except per-share fields). The file ships as a template;
fill it from company annual reports.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FUND_FILE = Path(__file__).resolve().parent.parent / "data" / "fundamentals.csv"

COLUMNS = [
    "name", "fiscal_year", "shares_out_m", "revenue", "ebit", "net_income", "total_assets",
    "total_equity", "total_debt", "cash", "operating_cf", "capex", "dividend_ps", "tax_rate",
]


def load_fundamentals() -> pd.DataFrame:
    if not FUND_FILE.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(FUND_FILE, comment="#").dropna(subset=["name", "fiscal_year"])


def ratios(f: pd.DataFrame, price: pd.Series | None = None) -> pd.DataFrame:
    """Profitability, growth, leverage and (if prices are given) valuation multiples."""
    f = f.sort_values(["name", "fiscal_year"]).copy()
    g = f.groupby("name")
    out = pd.DataFrame({"name": f["name"], "fiscal_year": f["fiscal_year"]})
    out["ROE"] = f["net_income"] / f["total_equity"]
    out["ROA"] = f["net_income"] / f["total_assets"]
    out["EBIT margin"] = f["ebit"] / f["revenue"]
    out["Net margin"] = f["net_income"] / f["revenue"]
    out["Revenue growth"] = g["revenue"].pct_change(fill_method=None)
    out["EPS growth"] = g["net_income"].pct_change(fill_method=None)
    out["Debt/Equity"] = f["total_debt"] / f["total_equity"]
    out["Equity/Assets"] = f["total_equity"] / f["total_assets"]
    out["EPS"] = f["net_income"] / f["shares_out_m"]
    out["BVPS"] = f["total_equity"] / f["shares_out_m"]
    out["FCF"] = f["operating_cf"] - f["capex"]
    out["DPS"] = f["dividend_ps"]
    out["Payout"] = f["dividend_ps"] / out["EPS"]
    if price is not None:
        px = f["name"].map(price)
        mcap = px * f["shares_out_m"]
        out["Price"] = px
        out["Market cap (KES m)"] = mcap
        out["P/E"] = px / out["EPS"]
        out["P/B"] = px / out["BVPS"]
        out["Dividend yield"] = f["dividend_ps"] / px
        out["EV/EBIT"] = (mcap + f["total_debt"] - f["cash"]) / f["ebit"]
        out["FCF yield"] = out["FCF"] / mcap
    return out.reset_index(drop=True)


def wacc(cost_equity: float, cost_debt: float, tax: float, equity_value: float, debt_value: float) -> float:
    v = equity_value + debt_value
    return cost_equity * equity_value / v + cost_debt * (1 - tax) * debt_value / v


def capm_cost_of_equity(rf: float, beta: float, erp: float) -> float:
    return rf + beta * erp


def dcf(fcf0: float, growth: list[float], terminal_growth: float, discount: float,
        net_debt: float = 0.0, shares: float = 1.0) -> dict:
    """Free-cash-flow DCF with explicit growth years then a Gordon terminal value."""
    if discount <= terminal_growth:
        raise ValueError("discount rate must exceed terminal growth")
    flows, f = [], fcf0
    for g in growth:
        f *= 1 + g
        flows.append(f)
    t = np.arange(1, len(flows) + 1)
    pv = np.array(flows) / (1 + discount) ** t
    tv = flows[-1] * (1 + terminal_growth) / (discount - terminal_growth)
    pv_tv = tv / (1 + discount) ** len(flows)
    ev = pv.sum() + pv_tv
    equity = ev - net_debt
    return {"flows": flows, "pv_flows": pv.tolist(), "terminal_value": tv, "pv_terminal": pv_tv,
            "enterprise_value": ev, "equity_value": equity, "per_share": equity / shares,
            "terminal_share": pv_tv / ev}


def gordon(d0: float, g: float, r: float) -> float:
    if r <= g:
        raise ValueError("required return must exceed growth")
    return d0 * (1 + g) / (r - g)


def ddm_two_stage(d0: float, g1: float, years: int, g2: float, r: float) -> float:
    d = d0
    pv = 0.0
    for t in range(1, years + 1):
        d *= 1 + g1
        pv += d / (1 + r) ** t
    return pv + gordon(d, g2, r) / (1 + r) ** years


def sensitivity(fn, rows: dict, cols: dict) -> pd.DataFrame:
    """Grid of fn(**{row_key: r, col_key: c}) for a two-way sensitivity table."""
    (rk, rv), (ck, cv) = next(iter(rows.items())), next(iter(cols.items()))
    grid = [[_safe(fn, **{rk: r, ck: c}) for c in cv] for r in rv]
    return pd.DataFrame(grid, index=pd.Index(rv, name=rk), columns=pd.Index(cv, name=ck))


def _safe(fn, **kw):
    try:
        return fn(**kw)
    except ValueError:
        return np.nan
