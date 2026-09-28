"""Mean-variance portfolio construction and Monte Carlo simulation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TDAYS = 252


def ledoit_wolf(r: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf (2004) shrinkage of the sample covariance towards a scaled identity.

    With ~20 liquid names and gappy histories the sample covariance is noisy;
    shrinking it keeps the optimiser from betting on estimation error.
    """
    X = r.dropna().to_numpy()
    X = X - X.mean(axis=0)
    n, p = X.shape
    S = X.T @ X / n
    mu = np.trace(S) / p
    F = mu * np.eye(p)
    d2 = np.sum((S - F) ** 2)
    b2 = sum(np.sum((np.outer(x, x) - S) ** 2) for x in X) / n**2
    delta = min(b2, d2) / d2 if d2 > 0 else 1.0
    return pd.DataFrame(delta * F + (1 - delta) * S, index=r.columns, columns=r.columns)


def estimate(r: pd.DataFrame, shrink: bool = True) -> tuple[pd.Series, pd.DataFrame]:
    """Annualised expected returns (historical mean) and covariance."""
    mu = r.mean() * TDAYS
    cov = (ledoit_wolf(r) if shrink else r.cov()) * TDAYS
    return mu, cov


def port_stats(w: np.ndarray, mu: pd.Series, cov: pd.DataFrame, rf: float = 0.0) -> dict:
    ret = float(w @ mu.to_numpy())
    vol = float(np.sqrt(w @ cov.to_numpy() @ w))
    return {"return": ret, "volatility": vol, "sharpe": (ret - rf) / vol if vol else np.nan}


def _solve(obj, n, max_weight, extra=()):
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}, *extra]
    res = minimize(obj, np.full(n, 1 / n), bounds=[(0, max_weight)] * n, constraints=cons, method="SLSQP",
                   options={"maxiter": 500, "ftol": 1e-12})
    return res.x


def min_variance(mu, cov, max_weight=1.0) -> np.ndarray:
    C = cov.to_numpy()
    return _solve(lambda w: w @ C @ w, len(mu), max_weight)


def max_sharpe(mu, cov, rf=0.0, max_weight=1.0) -> np.ndarray:
    C, m = cov.to_numpy(), mu.to_numpy()
    return _solve(lambda w: -(w @ m - rf) / np.sqrt(w @ C @ w), len(mu), max_weight)


def target_return(mu, cov, target, max_weight=1.0) -> np.ndarray:
    C, m = cov.to_numpy(), mu.to_numpy()
    return _solve(lambda w: w @ C @ w, len(mu), max_weight,
                  extra=[{"type": "eq", "fun": lambda w: w @ m - target}])


def efficient_frontier(mu, cov, points=30, max_weight=1.0) -> pd.DataFrame:
    """Long-only frontier from the minimum-variance portfolio up to the highest attainable return."""
    w0 = min_variance(mu, cov, max_weight)
    lo = float(w0 @ mu.to_numpy())
    # Highest return attainable under the weight cap: fill the best assets first.
    hi, left = 0.0, 1.0
    for v in mu.sort_values(ascending=False):
        take = min(max_weight, left)
        hi += take * v
        left -= take
        if left <= 1e-12:
            break
    rows = []
    for t in np.linspace(lo, hi, points):
        w = target_return(mu, cov, t, max_weight)
        rows.append({**port_stats(w, mu, cov), "weights": w})
    return pd.DataFrame(rows)


def monte_carlo(port_r: pd.Series, horizon: int = 252, n_paths: int = 2000, block: int = 20,
                seed: int = 7) -> np.ndarray:
    """Stationary block bootstrap of a portfolio's own daily returns.

    Resampling blocks rather than single days keeps volatility clustering and
    the autocorrelation that stale NSE prices create. Returns an array of shape
    (n_paths, horizon + 1) of wealth, starting at 1.
    """
    x = port_r.dropna().to_numpy()
    rng = np.random.default_rng(seed)
    n = len(x)
    out = np.empty((n_paths, horizon))
    for p in range(n_paths):
        idx = np.empty(horizon, dtype=int)
        i = 0
        while i < horizon:
            start = rng.integers(0, n)
            length = min(rng.geometric(1 / block), horizon - i)
            idx[i : i + length] = (start + np.arange(length)) % n
            i += length
        out[p] = x[idx]
    return np.hstack([np.ones((n_paths, 1)), np.cumprod(1 + out, axis=1)])
