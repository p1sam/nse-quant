import numpy as np
import pandas as pd
import pytest
from scipy import stats

from nsequant import data, portfolio as pf, risk, valuation


# ---------------------------------------------------------------- parser
def test_parse_line_handles_unquoted_thousands_separators():
    line = "7/28/2023,Safaricom Ltd Ord 0.05,KES 16.60,208485900,-3,267,923,640.00,-3.77%"
    assert data.parse_line(line) == ("7/28/2023", "Safaricom Ltd Ord 0.05", 16.60, 208485900.0, -3267923640.0)


def test_parse_line_missing_price():
    d, name, price, vol, flow = data.parse_line("6/29/2026,Africa Mega Agricorp Plc Ord 5.00AIMS,-,0,0.00,-")
    assert name.startswith("Africa Mega") and np.isnan(price) and vol == 0 and flow == 0


@pytest.mark.parametrize("raw, clean", [
    ("KCB Group Ltd Ord 1.00", "KCB Group Ltd"),
    ("Kapchorua Tea Co. Ltd Ord Ord 5.00AIMS", "Kapchorua Tea Co. Ltd"),
    ("KenGen Co. Ltd  Ord. 2.50", "KenGen Co. Ltd"),
    ("Shri Krishana Overseas Plc 0.20", "Shri Krishana Overseas Plc"),
    ("Eaagads Ltd Ord 1.25 AIMS", "Eaagads Ltd"),
    ("ABSA New Gold ETF", "ABSA New Gold ETF"),
])
def test_clean_name(raw, clean):
    assert data.clean_name(raw) == clean


def test_despike_removes_reverting_spike_only():
    s = pd.Series([10, 10.2, 1020, 10.1, 10.3, 6.0, 6.1], dtype=float)
    out, hits = data._despike(s)
    assert hits == [2]
    assert out.iloc[5] == 6.0  # a permanent drop is not a spike


# ---------------------------------------------------------------- risk
@pytest.fixture
def normal_returns():
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0005, 0.01, 200_000))


def test_var_matches_normal_quantile(normal_returns):
    expect = -(0.0005 + stats.norm.ppf(0.01) * 0.01)
    assert risk.var_historical(normal_returns, 0.99) == pytest.approx(expect, rel=0.02)
    assert risk.var_parametric(normal_returns, 0.99) == pytest.approx(expect, rel=0.01)
    assert risk.var_cornish_fisher(normal_returns, 0.99) == pytest.approx(expect, rel=0.02)


def test_cvar_exceeds_var(normal_returns):
    assert risk.cvar_historical(normal_returns) > risk.var_historical(normal_returns)
    assert risk.cvar_parametric(normal_returns) == pytest.approx(risk.cvar_historical(normal_returns), rel=0.02)


def test_evt_recovers_pareto_tail():
    rng = np.random.default_rng(1)
    losses = stats.genpareto.rvs(0.3, scale=0.01, size=50_000, random_state=rng)
    fit = risk.evt_tail(pd.Series(-losses), 0.99, 0.9)
    assert fit["xi"] == pytest.approx(0.3, abs=0.06)
    assert fit["var"] == pytest.approx(stats.genpareto.ppf(0.99, 0.3, scale=0.01), rel=0.05)


def test_drawdown():
    r = pd.Series([0.1, -0.5, 0.2])
    assert risk.max_drawdown(r) == pytest.approx(-0.5)


def test_dimson_beta_recovers_lagged_response():
    rng = np.random.default_rng(2)
    m = pd.Series(rng.normal(0, 0.01, 5000))
    # Stock reacts half today, half tomorrow: true beta 1, OLS beta ~0.5.
    r = 0.5 * m + 0.5 * m.shift(1) + rng.normal(0, 0.002, 5000)
    assert risk.beta(r, m) == pytest.approx(0.5, abs=0.05)
    assert risk.dimson_beta(r, m) == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------- portfolio
def test_min_variance_two_uncorrelated_assets():
    mu = pd.Series([0.1, 0.1], index=["a", "b"])
    cov = pd.DataFrame(np.diag([0.04, 0.01]), index=mu.index, columns=mu.index)
    w = pf.min_variance(mu, cov)
    assert w == pytest.approx([0.2, 0.8], abs=1e-4)


def test_max_sharpe_respects_cap():
    mu = pd.Series([0.3, 0.1, 0.1], index=list("abc"))
    cov = pd.DataFrame(np.eye(3) * 0.04, index=mu.index, columns=mu.index)
    w = pf.max_sharpe(mu, cov, max_weight=0.5)
    assert w.sum() == pytest.approx(1) and w.max() <= 0.5 + 1e-6


def test_ledoit_wolf_is_psd():
    rng = np.random.default_rng(3)
    r = pd.DataFrame(rng.normal(0, 0.01, (40, 30)))
    assert np.linalg.eigvalsh(pf.ledoit_wolf(r)).min() > 0


def test_monte_carlo_shape_and_start():
    paths = pf.monte_carlo(pd.Series(np.full(500, 0.001)), horizon=10, n_paths=5)
    assert paths.shape == (5, 11)
    assert paths[:, 0] == pytest.approx(1) and paths[:, -1] == pytest.approx(1.001**10)


# ---------------------------------------------------------------- valuation
def test_gordon():
    assert valuation.gordon(1.0, 0.05, 0.10) == pytest.approx(21.0)


def test_two_stage_collapses_to_gordon():
    assert valuation.ddm_two_stage(1.0, 0.05, 5, 0.05, 0.10) == pytest.approx(valuation.gordon(1.0, 0.05, 0.10))


def test_dcf_zero_growth_is_perpetuity():
    res = valuation.dcf(100, [0, 0, 0], 0.0, 0.10, net_debt=0, shares=10)
    assert res["enterprise_value"] == pytest.approx(1000)
    assert res["per_share"] == pytest.approx(100)


def test_dcf_rejects_growth_above_discount():
    with pytest.raises(ValueError):
        valuation.dcf(100, [0.05], 0.12, 0.10)
