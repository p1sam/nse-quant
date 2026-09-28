"""NSE Quant — Streamlit dashboard.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from nsequant import data, factors, portfolio as pf, risk, valuation

warnings.filterwarnings("ignore", category=RuntimeWarning)
st.set_page_config(page_title="NSE Quant", layout="wide")

# Categorical order and diverging pair from a colour-vision-deficiency-validated palette.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
DIVERGING = [[0, "#e34948"], [0.5, "#f0efec"], [1, "#2a78d6"]]
SEQUENTIAL = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
pio.templates["nse"] = go.layout.Template(layout=dict(colorway=SERIES, hovermode="x unified",
                                                      margin=dict(l=10, r=10, t=40, b=10)))
pio.templates.default = "plotly+nse"
pct = "{:.1%}".format

RESEARCH_DIR = Path(__file__).parent / "research"


# ---------------------------------------------------------------- cached data
@st.cache_data(show_spinner="Loading and cleaning NSE prices…")
def load():
    px_, issues = data.clean_prices()
    return px_, issues, data.returns(), data.pivot("volume"), data.pivot("foreign_flow")


@st.cache_data
def market(weights: str) -> pd.Series:
    return data.index_level(weights)


@st.cache_data(show_spinner="Building factor portfolios…")
def factor_data():
    return factors.factor_returns(), factors.scores(), factors.liquid_universe()


@st.cache_data
def sector_map() -> pd.Series:
    return data.sectors()


prices, issues, rets, volume, fflow = load()
sectors = sector_map()

with st.sidebar:
    st.title("NSE Quant")
    st.caption(f"Data: {prices.index.min():%d %b %Y} – {prices.index.max():%d %b %Y} · {prices.shape[1]} securities")
    rf = st.number_input("Risk-free rate (91-day T-bill, % p.a.)", 0.0, 30.0, 9.0, 0.25,
                         help="Used for Sharpe ratios and the CAPM cost of equity.") / 100
    index_kind = st.radio("Market proxy", ["liquidity", "equal"], format_func=lambda k: {
        "liquidity": "Liquidity-weighted", "equal": "Equal-weighted"}[k],
        help="The export has no NASI series, so the market is rebuilt from constituents.")

mkt_level = market(index_kind)
mkt = mkt_level.pct_change().dropna()


def window_picker(key: str, default: str = "3Y") -> pd.Timestamp:
    opts = {"6M": 126, "1Y": 252, "3Y": 756, "5Y": 1260, "All": None}
    choice = st.segmented_control("Window", list(opts), default=default, key=key) or default
    n = opts[choice]
    return prices.index[-n] if n and n < len(prices) else prices.index[0]


def liquid_names() -> list[str]:
    _, _, uni = factor_data()
    return [c for c in uni.columns[uni.iloc[-1]] if c in rets.columns]


# ---------------------------------------------------------------- pages
def page_market():
    st.header("Market overview")
    start = window_picker("mkt")
    lvl = mkt_level.loc[start:]
    r = mkt.loc[start:]
    c = st.columns(5)
    c[0].metric("Market return (window)", pct(lvl.iloc[-1] / lvl.iloc[0] - 1))
    c[1].metric("Annualised volatility", pct(risk.ann_vol(r)))
    c[2].metric("Max drawdown", pct(risk.max_drawdown(r)))
    c[3].metric("1-day VaR 95%", pct(risk.var_historical(r)))
    c[4].metric("Sharpe", f"{risk.sharpe(r, rf):.2f}")

    both = pd.DataFrame({"Liquidity-weighted": market("liquidity"), "Equal-weighted": market("equal")}).loc[start:]
    both = both / both.iloc[0] * 100
    st.plotly_chart(px.line(both, title="Market proxy (rebased to 100)", labels={"value": "Index", "variable": ""}),
                    use_container_width=True)

    left, right = st.columns(2)
    with left:
        sr = rets.loc[start:].T.groupby(sectors).mean().T
        sr = sr.drop(columns=["Exchange Traded Fund", "Other"], errors="ignore")
        perf = ((1 + sr.fillna(0)).prod() - 1).sort_values()
        fig = px.bar(perf, orientation="h", title="Sector performance (equal-weighted)",
                     labels={"value": "Return", "index": ""})
        fig.update_traces(marker_color=[SERIES[0] if v >= 0 else SERIES[7] for v in perf], hovertemplate="%{x:.1%}")
        fig.update_layout(showlegend=False, xaxis_tickformat=".0%", hovermode="closest")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        rv = risk.rolling_vol(mkt, 63).loc[start:]
        fig = px.area(rv, title="Market volatility (63-day rolling, annualised)", labels={"value": "Volatility"})
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Market breadth")
    px_f = prices.ffill(limit=5)
    d = px_f.pct_change(fill_method=None).where(volume.reindex_like(prices).fillna(0) > 0)
    adv, dec = (d > 0).sum(axis=1), (d < 0).sum(axis=1)
    above = (px_f > px_f.rolling(200, min_periods=150).mean()).sum(axis=1) / px_f.rolling(200, min_periods=150).mean().notna().sum(axis=1)
    b1, b2 = st.columns(2)
    with b1:
        ad = (adv - dec).cumsum().loc[start:]
        st.plotly_chart(px.line(ad, title="Advance–decline line (cumulative)", labels={"value": "Net advancers"})
                        .update_layout(showlegend=False), use_container_width=True)
    with b2:
        st.plotly_chart(px.line(above.loc[start:], title="Share of stocks above 200-day average",
                                labels={"value": "Share"}).update_layout(showlegend=False, yaxis_tickformat=".0%"),
                        use_container_width=True)

    st.subheader("Correlations — liquid names")
    names = liquid_names()
    corr = rets.loc[start:, names].corr(min_periods=60)
    short = [n.split(" ")[0] if len(n) > 14 else n for n in names]
    fig = go.Figure(go.Heatmap(z=corr.values, x=short, y=short, zmin=-1, zmax=1, colorscale=DIVERGING,
                               hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>"))
    fig.update_layout(height=620, hovermode="closest")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Daily-return correlations are biased towards zero for thinly traded stocks; weekly returns "
               "or Dimson betas give a truer picture of co-movement.")

    flow = fflow.reindex_like(prices).loc[start:].sum(axis=1).resample("ME").sum() / 1e6
    fig = px.bar(flow, title="Net foreign flow, all securities (KES m per month)", labels={"value": "KES m"})
    fig.update_traces(marker_color=[SERIES[0] if v >= 0 else SERIES[7] for v in flow])
    fig.update_layout(showlegend=False, hovermode="closest")
    st.plotly_chart(fig, use_container_width=True)


def page_company():
    st.header("Company analysis")
    names = sorted(c for c in rets.columns)
    name = st.selectbox("Company", names, index=names.index("Safaricom Ltd") if "Safaricom Ltd" in names else 0)
    st.caption(f"Sector: {sectors.get(name, 'Other')} · Board: {data.board_of(name)}")
    start = window_picker("co", "All")
    p = prices[name].loc[start:].dropna()
    r = rets[name].loc[start:]
    last = p.iloc[-1]

    s = risk.summary(r.dropna(), mkt, rf)
    c = st.columns(6)
    c[0].metric("Last price (KES)", f"{last:,.2f}")
    c[1].metric("Return p.a.", pct(s["Annual return"]))
    c[2].metric("Volatility p.a.", pct(s["Annual volatility"]))
    c[3].metric("Dimson beta", f"{s['Beta (Dimson)']:.2f}")
    c[4].metric("Max drawdown", pct(s["Max drawdown"]))
    traded = (volume[name].loc[start:] > 0).mean()
    c[5].metric("Days traded", pct(traded))

    fig = px.line(p, title=f"{name} — close (KES)", labels={"value": "KES"})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    fund = valuation.load_fundamentals()
    fund = fund[fund["name"] == name]
    st.subheader("Financial statements & ratios")
    if fund.empty:
        st.info("No fundamentals for this company yet. Add rows to `data/fundamentals.csv` from its annual "
                "reports to unlock statements, multiples, profitability, growth and leverage here — "
                "and the Value and Quality factors.")
        latest = None
    else:
        st.dataframe(fund.set_index("fiscal_year").drop(columns="name").T, use_container_width=True)
        rat = valuation.ratios(fund, prices.ffill().iloc[-1]).set_index("fiscal_year").drop(columns="name")
        st.dataframe(rat.T.style.format("{:.3g}"), use_container_width=True)
        latest = fund.sort_values("fiscal_year").iloc[-1]

    st.subheader("Valuation")
    beta_ = s["Beta (Dimson)"] if np.isfinite(s["Beta (Dimson)"]) else 1.0
    t1, t2 = st.tabs(["Discounted cash flow", "Dividend discount"])
    with t1:
        a, b, cc = st.columns(3)
        shares = a.number_input("Shares outstanding (m)", 0.0, value=float(latest["shares_out_m"]) if latest is not None else 1000.0)
        fcf0 = a.number_input("Base free cash flow (KES m)",
                              value=float(latest["operating_cf"] - latest["capex"]) if latest is not None else 10_000.0)
        net_debt = a.number_input("Net debt (KES m)",
                                  value=float(latest["total_debt"] - latest["cash"]) if latest is not None else 0.0)
        erp = b.number_input("Equity risk premium (%)", 0.0, 20.0, 7.0, 0.5) / 100
        beta_in = b.number_input("Beta", 0.0, 3.0, round(float(beta_), 2), 0.05)
        ke = valuation.capm_cost_of_equity(rf, beta_in, erp)
        disc = b.number_input("Discount rate (%)", 1.0, 40.0, round(ke * 100, 2), 0.25,
                              help=f"CAPM cost of equity = {ke:.2%}. Replace with a WACC if valuing the firm.") / 100
        g1 = cc.number_input("Growth, years 1–5 (%)", -20.0, 50.0, 8.0, 0.5) / 100
        gt = cc.number_input("Terminal growth (%)", -5.0, 15.0, 5.0, 0.25) / 100
        try:
            res = valuation.dcf(fcf0, [g1] * 5, gt, disc, net_debt, shares)
            m = st.columns(4)
            m[0].metric("Intrinsic value / share", f"KES {res['per_share']:,.2f}")
            m[1].metric("Market price", f"KES {last:,.2f}")
            m[2].metric("Upside", pct(res["per_share"] / last - 1))
            m[3].metric("Terminal value share of EV", pct(res["terminal_share"]))
            st.caption("Sensitivity — value per share by discount rate (rows) and terminal growth (columns)")
            grid = valuation.sensitivity(
                lambda discount, terminal_growth: valuation.dcf(fcf0, [g1] * 5, terminal_growth, discount, net_debt, shares)["per_share"],
                {"discount": [round(disc + d, 4) for d in (-0.02, -0.01, 0, 0.01, 0.02)]},
                {"terminal_growth": [round(gt + d, 4) for d in (-0.01, -0.005, 0, 0.005, 0.01)]})
            st.dataframe(grid.style.format("{:,.1f}").background_gradient(cmap="Blues"), use_container_width=True)
        except ValueError as e:
            st.error(str(e))
    with t2:
        a, b = st.columns(2)
        d0 = a.number_input("Last annual dividend per share (KES)", 0.0,
                            value=float(latest["dividend_ps"]) if latest is not None else 1.0)
        r_req = a.number_input("Required return (%)", 1.0, 40.0, round(ke * 100, 2), 0.25) / 100
        gh = b.number_input("High growth (%)", -20.0, 50.0, 8.0, 0.5) / 100
        yrs = b.number_input("High-growth years", 1, 20, 5)
        gl = b.number_input("Long-run growth (%)", -5.0, 15.0, 5.0, 0.25) / 100
        try:
            gv = valuation.gordon(d0, gl, r_req)
            tv = valuation.ddm_two_stage(d0, gh, int(yrs), gl, r_req)
            m = st.columns(3)
            m[0].metric("Gordon growth value", f"KES {gv:,.2f}", pct(gv / last - 1))
            m[1].metric("Two-stage DDM value", f"KES {tv:,.2f}", pct(tv / last - 1))
            m[2].metric("Implied dividend yield", pct(d0 / last))
        except ValueError as e:
            st.error(str(e))


def page_risk():
    st.header("Risk")
    default = [n for n in ["Safaricom Ltd", "Equity Group Holdings Ltd", "KCB Group Ltd", "East African Breweries Ltd"]
               if n in rets.columns]
    picks = st.multiselect("Securities (equal-weighted if more than one)", sorted(rets.columns), default)
    if not picks:
        st.stop()
    start = window_picker("risk", "5Y")
    r = rets.loc[start:, picks].mean(axis=1, skipna=True).dropna()
    m = mkt.loc[start:]
    level = st.select_slider("Confidence level", [0.90, 0.95, 0.975, 0.99], 0.95, format_func=pct)

    evt = risk.evt_tail(r, level if level >= 0.95 else 0.95)
    tbl = pd.DataFrame({
        "VaR": [risk.var_historical(r, level), risk.var_parametric(r, level), risk.var_cornish_fisher(r, level),
                evt["var"] if level >= 0.95 else np.nan],
        "CVaR / ES": [risk.cvar_historical(r, level), risk.cvar_parametric(r, level), np.nan,
                      evt["es"] if level >= 0.95 else np.nan],
    }, index=["Historical", "Gaussian", "Cornish–Fisher", "EVT (GPD peaks-over-threshold)"])
    c1, c2 = st.columns([2, 3])
    with c1:
        st.subheader(f"1-day loss at {level:.1%}")
        st.dataframe(tbl.style.format(lambda v: "—" if pd.isna(v) else f"{v:.2%}"), use_container_width=True)
        st.caption(f"GPD tail index ξ = {evt['xi']:.2f} ({evt['n_exceed']} exceedances). ξ > 0 means a fat, "
                   "power-law tail; Gaussian VaR then understates extreme losses.")
        s = risk.summary(r, m, rf)
        st.dataframe(pd.Series(s, name="").to_frame().style.format("{:.3f}"), use_container_width=True)
    with c2:
        fig = px.histogram(r, nbins=120, title="Daily return distribution", labels={"value": "Daily return"})
        fig.add_vline(-risk.var_historical(r, level), line_dash="dash", line_color=SERIES[7],
                      annotation_text=f"Hist. VaR {level:.0%}")
        fig.update_layout(showlegend=False, xaxis_tickformat=".0%", hovermode="closest")
        st.plotly_chart(fig, use_container_width=True)

    a, b = st.columns(2)
    with a:
        fig = px.area(risk.drawdown(r), title="Drawdown", labels={"value": "Drawdown"})
        fig.update_traces(line_color=SERIES[7])
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig = px.line(risk.rolling_vol(r, 63), title="Rolling 63-day volatility (annualised)", labels={"value": "Vol"})
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Stress tests")
    full = rets[picks].mean(axis=1, skipna=True)
    h = risk.historical_stress(full)
    h["market"] = risk.historical_stress(mkt)["return"]
    st.dataframe(h.set_index("scenario").style.format("{:.1%}"), use_container_width=True)
    betas = pd.Series({n: risk.dimson_beta(rets[n], mkt) for n in picks})
    st.dataframe(risk.beta_stress(betas, pd.Series(1 / len(picks), index=picks)).style.format(
        {"market shock": "{:.0%}", "portfolio beta": "{:.2f}", "expected portfolio move": "{:.1%}"}),
        use_container_width=True, hide_index=True)
    st.caption("Beta shocks use Dimson betas; the historical windows replay the realised path of each episode.")


def page_factors():
    st.header("Quant factors")
    fr, sc, uni = factor_data()
    missing = [f for f in factors.FACTORS if f not in fr.columns]
    if missing:
        st.info(f"{', '.join(missing)} need `data/fundamentals.csv` (book value, earnings, ROE). "
                "Size uses average traded value as a proxy until shares outstanding are added.")
    st.caption(f"Long top-tercile / short bottom-tercile, equal-weighted, rebalanced monthly within the liquid "
               f"universe (≥ KES 1m average daily value, traded ≥ 50% of days) — currently "
               f"{int(uni.iloc[-1].sum())} names.")
    cum = (1 + fr.fillna(0)).cumprod()
    st.plotly_chart(px.line(cum, title="Cumulative long–short factor returns", labels={"value": "Growth of 1", "variable": ""}),
                    use_container_width=True)
    stats_ = factors.factor_stats(fr)
    st.dataframe(stats_.style.format({"Months": "{:.0f}", "Mean monthly": "{:.2%}", "Annualised": "{:.1%}",
                                      "Volatility (ann.)": "{:.1%}", "Sharpe (ann.)": "{:.2f}", "t-stat": "{:.2f}",
                                      "Hit rate": "{:.0%}"}), use_container_width=True)
    st.caption("|t| > 2 is the usual bar for significance. With under ten years of data and ~20 liquid names, "
               "treat these as indicative.")

    st.subheader("Latest factor scores (z-scores, liquid universe)")
    names = uni.columns[uni.iloc[-1]]
    z = pd.DataFrame({k: factors.zscore(v.reindex(columns=names)).iloc[-1] for k, v in sc.items()}).dropna(how="all")
    st.dataframe(z.sort_values(z.columns[0], ascending=False).style.format("{:+.2f}")
                 .background_gradient(cmap="RdBu", vmin=-2.5, vmax=2.5), use_container_width=True)


def page_portfolio():
    st.header("Portfolio")
    liquid = liquid_names()
    picks = st.multiselect("Universe", sorted(rets.columns), liquid[:12])
    if len(picks) < 2:
        st.warning("Pick at least two securities.")
        st.stop()
    c = st.columns(3)
    start = c[0].selectbox("Estimation window", ["3Y", "5Y", "All"], 1)
    start = prices.index[-{"3Y": 756, "5Y": 1260}.get(start, len(prices))]
    wmax = c[1].slider("Max weight per stock", 0.05, 1.0, 0.25, 0.05)
    shrink = c[2].toggle("Ledoit–Wolf shrinkage", True)

    r = rets.loc[start:, picks].fillna(0)
    mu, cov = pf.estimate(r, shrink)
    ef = pf.efficient_frontier(mu, cov, 25, wmax)
    w_ms = pf.max_sharpe(mu, cov, rf, wmax)
    w_mv = pf.min_variance(mu, cov, wmax)
    w_eq = np.full(len(picks), 1 / len(picks))
    ports = {"Max Sharpe": w_ms, "Min variance": w_mv, "Equal weight": w_eq}

    fig = go.Figure()
    fig.add_scatter(x=ef["volatility"], y=ef["return"], mode="lines", name="Efficient frontier",
                    line=dict(color=SERIES[0], width=2), hovertemplate="σ %{x:.1%} · μ %{y:.1%}<extra></extra>")
    vols = np.sqrt(np.diag(cov))
    fig.add_scatter(x=vols, y=mu, mode="markers", name="Stocks", text=picks,
                    marker=dict(color=SERIES[2], size=9, line=dict(color="white", width=2)),
                    hovertemplate="%{text}<br>σ %{x:.1%} · μ %{y:.1%}<extra></extra>")
    for i, (k, w) in enumerate(ports.items()):
        s = pf.port_stats(w, mu, cov, rf)
        fig.add_scatter(x=[s["volatility"]], y=[s["return"]], mode="markers+text", name=k, text=[k],
                        textposition="top center", marker=dict(size=14, color=SERIES[[1, 6, 3][i]], symbol="diamond",
                                                                line=dict(color="white", width=2)))
    fig.update_layout(title="Efficient frontier (annualised, long-only)", xaxis_title="Volatility",
                      yaxis_title="Expected return", xaxis_tickformat=".0%", yaxis_tickformat=".0%", hovermode="closest",
                      height=520)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Expected returns are historical means — the noisiest input in portfolio optimisation. "
               "Read the frontier as a description of the past, not a forecast.")

    stats_tbl = pd.DataFrame({k: pf.port_stats(w, mu, cov, rf) for k, w in ports.items()}).T
    stats_tbl.columns = ["Expected return", "Volatility", "Sharpe"]
    wt = pd.DataFrame(ports, index=picks)
    a, b = st.columns([2, 3])
    a.dataframe(stats_tbl.style.format({"Expected return": "{:.1%}", "Volatility": "{:.1%}", "Sharpe": "{:.2f}"}),
                use_container_width=True)
    b.dataframe(wt[wt.max(axis=1) > 0.001].style.format("{:.1%}"), use_container_width=True)

    choice = st.radio("Analyse", list(ports), horizontal=True)
    w = pd.Series(ports[choice], index=picks)
    port_r = rets.loc[start:, picks].fillna(0) @ w

    st.subheader("Factor exposures")
    fr, _, _ = factor_data()
    e1, e2 = st.columns(2)
    with e1:
        ex = factors.latest_exposures(w)
        fig = px.bar(ex, title="Characteristic exposure (weighted z-score)", labels={"value": "z", "index": ""})
        fig.update_traces(marker_color=[SERIES[0] if v >= 0 else SERIES[7] for v in ex])
        fig.update_layout(showlegend=False, hovermode="closest")
        st.plotly_chart(fig, use_container_width=True)
    with e2:
        pm = (1 + port_r).resample("ME").prod() - 1
        mm = (1 + mkt).resample("ME").prod() - 1
        reg = factors.regression_exposures(pm, pd.concat([mm.rename("Market"), fr], axis=1))
        st.caption("Time-series regression of monthly returns on market and factor returns")
        st.dataframe(reg.style.format("{:.3f}"), use_container_width=True)

    st.subheader("Monte Carlo simulation")
    m1, m2, m3 = st.columns(3)
    horizon = m1.selectbox("Horizon", [63, 126, 252, 504], 2, format_func=lambda d: f"{d // 21} months")
    n = m2.selectbox("Paths", [500, 1000, 2000], 1)
    amount = m3.number_input("Initial investment (KES)", 1_000.0, value=1_000_000.0, step=100_000.0)
    paths = pf.monte_carlo(port_r, horizon, n) * amount
    q = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    x = np.arange(horizon + 1)
    fig = go.Figure()
    fig.add_scatter(x=np.r_[x, x[::-1]], y=np.r_[q[4], q[0][::-1]], fill="toself", fillcolor="rgba(42,120,214,0.15)",
                    line=dict(width=0), name="5–95%", hoverinfo="skip")
    fig.add_scatter(x=np.r_[x, x[::-1]], y=np.r_[q[3], q[1][::-1]], fill="toself", fillcolor="rgba(42,120,214,0.30)",
                    line=dict(width=0), name="25–75%", hoverinfo="skip")
    fig.add_scatter(x=x, y=q[2], name="Median", line=dict(color=SERIES[0], width=2))
    fig.update_layout(title="Simulated portfolio value (block bootstrap of historical returns)",
                      xaxis_title="Trading days", yaxis_title="KES", hovermode="x")
    st.plotly_chart(fig, use_container_width=True)
    end = paths[:, -1]
    k = st.columns(4)
    k[0].metric("Median outcome", f"KES {np.median(end):,.0f}")
    k[1].metric("5th percentile", f"KES {np.percentile(end, 5):,.0f}")
    k[2].metric("Probability of loss", pct((end < amount).mean()))
    k[3].metric("Horizon CVaR 95%", pct(1 - end[end <= np.percentile(end, 5)].mean() / amount))


def page_research():
    st.header("Research")
    papers = sorted(RESEARCH_DIR.glob("*.md"))
    if not papers:
        st.info("No papers yet — run `python -m research.build` to generate them.")
        return
    titles = {p: p.read_text(encoding="utf-8").splitlines()[0].lstrip("# ") for p in papers}
    p = st.selectbox("Paper", papers, format_func=titles.get)
    text = p.read_text(encoding="utf-8")
    fig_dir = RESEARCH_DIR / "figures"
    # Papers reference figures as ![caption](figures/x.png); render those with st.image.
    for block in text.split("\n\n"):
        if block.startswith("![") and "](figures/" in block:
            cap, path = block[2:].split("](")
            st.image(str(fig_dir / Path(path.rstrip(")")).name), caption=cap.rstrip("]"))
        else:
            st.markdown(block)


def page_quality():
    st.header("Data quality")
    st.write("Every correction the cleaning pipeline made to the raw export. Unexplained jumps are excluded from "
             "returns until confirmed in `data/corporate_actions.csv` (then prices are back-adjusted).")
    counts = issues["issue"].str.replace(r"[+-]?\d+%|x[\d.]+", "", regex=True).str.strip().value_counts()
    st.dataframe(counts.rename("count"), use_container_width=True)
    kind = st.selectbox("Show", ["unexplained jump", "one-day spike", "scrambled export day", "corporate action", "all"])
    view = issues if kind == "all" else issues[issues["issue"].str.startswith(kind)]
    st.dataframe(view, use_container_width=True, hide_index=True)
    name = st.selectbox("Inspect raw vs clean", sorted(prices.columns))
    raw = data.pivot("price")[name].mask(lambda s: s <= 0)
    fig = go.Figure()
    fig.add_scatter(x=raw.index, y=raw, name="Raw export", line=dict(color=SERIES[1], width=1))
    fig.add_scatter(x=prices.index, y=prices[name], name="Cleaned", line=dict(color=SERIES[0], width=2))
    fig.update_layout(yaxis_type="log", title=f"{name} (log scale)")
    st.plotly_chart(fig, use_container_width=True)


nav = st.navigation([
    st.Page(page_market, title="Market overview", url_path="market", default=True),
    st.Page(page_company, title="Company analysis", url_path="company"),
    st.Page(page_risk, title="Risk", url_path="risk"),
    st.Page(page_factors, title="Quant factors", url_path="factors"),
    st.Page(page_portfolio, title="Portfolio", url_path="portfolio"),
    st.Page(page_research, title="Research", url_path="research"),
    st.Page(page_quality, title="Data quality", url_path="data-quality"),
])
nav.run()
st.sidebar.caption("For research and education. Not investment advice.")
