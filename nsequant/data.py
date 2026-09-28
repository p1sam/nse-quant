"""Load and clean NSE daily price exports (Rock Investment Bank CSV format).

The export does not quote fields, and the Net Foreign Flow column uses
thousands separators ("-3,267,923,640.00"), so lines cannot be read with a
plain CSV parser. Each line is anchored on the price field instead: it is the
first field after the security name that starts with "KES " or is "-".
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
ACTIONS_FILE = RAW_DIR.parent / "corporate_actions.csv"

# NSE caps most daily moves at about +/-10%, so anything beyond this is an
# export error or a corporate action rather than trading.
JUMP = 0.30


def _num(s: str) -> float:
    s = s.strip()
    return np.nan if s in ("", "-") else float(s)


def parse_line(line: str) -> tuple[str, str, float, float, float]:
    p = line.split(",")
    k = next(i for i, x in enumerate(p) if i > 1 and (x.startswith("KES ") or x == "-"))
    return (
        p[0],
        ",".join(p[1:k]).strip(),
        _num(p[k].removeprefix("KES ")),
        _num(p[k + 1]),
        _num("".join(p[k + 2 : -1])),
    )


def clean_name(security: str) -> str:
    name = re.sub(r"\s+", " ", security).strip()
    name = re.sub(r"\s+ord\.?\s*[\d.]+\s*(aims|gems)?$", "", name, flags=re.I)
    name = re.sub(r"\s+(ord\.?\s*)+$", "", name, flags=re.I)
    name = re.sub(r"\s+[\d.]+\s*(aims|gems)?$", "", name, flags=re.I)
    return name.strip()


def board(security: str) -> str:
    s = security.upper().rstrip()
    if s.endswith("GEMS"):
        return "GEMS"
    if s.endswith("AIMS"):
        return "AIMS"
    if "ETF" in s:
        return "ETF"
    if "REIT" in s:
        return "REIT"
    return "MIMS"


def read_export(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("Date,")) + 1
    rows = [parse_line(ln) for ln in lines[start:] if ln.strip()]
    df = pd.DataFrame(rows, columns=["date", "security", "price", "volume", "foreign_flow"])
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    return df


@lru_cache(maxsize=1)
def load_long(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """All exports in raw_dir as one long table, one row per (date, ticker)."""
    frames = [read_export(p) for p in sorted(raw_dir.glob("*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    df["name"] = df["security"].map(clean_name)
    df["board"] = df["security"].map(board)
    # Some tickers appear under two spellings of the same name; keep the
    # row that actually carries a price.
    df = (
        df.sort_values("price", na_position="first")
        .drop_duplicates(["date", "name"], keep="last")
        .sort_values(["name", "date"])
    )
    # Drop calendar days where (almost) nothing is priced: weekends/holidays
    # that the export fills with "-".
    coverage = df.groupby("date")["price"].apply(lambda s: s.notna().mean())
    df = df[df["date"].isin(coverage[coverage > 0.2].index)]
    return df.reset_index(drop=True)


def pivot(field: str = "price") -> pd.DataFrame:
    return load_long().pivot(index="date", columns="name", values=field).sort_index()


def _bad_days(px: pd.DataFrame) -> pd.Index:
    """Dates where a large share of the market 'jumps' at once (scrambled export)."""
    r = px.ffill().pct_change(fill_method=None).abs()
    share = (r > JUMP).sum(axis=1) / px.notna().sum(axis=1).clip(lower=1)
    return share[share > 0.10].index


def _despike(s: pd.Series) -> tuple[pd.Series, list]:
    """Blank out prices that jump > JUMP and revert within 10 observations."""
    s = s.copy()
    flagged = []
    for _ in range(5):
        v = s.dropna()
        if len(v) < 3:
            break
        lr = np.log(v).diff()
        hits = []
        for i in np.flatnonzero(lr.abs().to_numpy() > np.log1p(JUMP)):
            for w in range(1, 11):
                if i + w < len(v) and abs(np.log(v.iloc[i + w] / v.iloc[i - 1])) < np.log(1.15):
                    hits.extend(v.index[i : i + w])
                    break
        if not hits:
            break
        s[hits] = np.nan
        flagged.extend(hits)
    return s, flagged


def load_actions() -> pd.DataFrame:
    """Confirmed splits / bonus issues. factor = new shares per old share
    (e.g. a 1-for-1 bonus has factor 2)."""
    if not ACTIONS_FILE.exists():
        return pd.DataFrame(columns=["name", "ex_date", "factor", "note"])
    a = pd.read_csv(ACTIONS_FILE, comment="#", parse_dates=["ex_date"])
    return a


@lru_cache(maxsize=1)
def clean_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adjusted, cleaned close prices plus a log of every correction made.

    Returns (prices, issues). Prices are back-adjusted for confirmed actions in
    data/corporate_actions.csv; unconfirmed one-way jumps > JUMP are left in the
    price series but reported in issues, and returns() drops them.
    """
    px = pivot("price").mask(lambda d: d <= 0)
    log = []
    for d in _bad_days(px):
        log.append({"date": d, "name": "(all)", "issue": "scrambled export day", "action": "row dropped"})
    px = px.drop(_bad_days(px))

    for name in px.columns:
        px[name], hits = _despike(px[name])
        log += [{"date": d, "name": name, "issue": "one-day spike", "action": "price blanked"} for d in hits]

    for a in load_actions().itertuples():
        if a.name in px.columns:
            px.loc[px.index < a.ex_date, a.name] /= a.factor
            log.append({"date": a.ex_date, "name": a.name, "issue": f"corporate action x{a.factor}", "action": "back-adjusted"})

    r = px.ffill().pct_change(fill_method=None)
    confirmed = set(zip(load_actions()["name"], load_actions()["ex_date"]))
    for (d, name), v in r.stack()[lambda s: s.abs() > JUMP].items():
        if (name, d) not in confirmed:
            log.append({"date": d, "name": name, "issue": f"unexplained jump {v:+.0%}",
                        "action": "return excluded; add to corporate_actions.csv if genuine"})
    issues = pd.DataFrame(log, columns=["date", "name", "issue", "action"]).sort_values("date")
    return px, issues.reset_index(drop=True)


def returns(min_obs: int = 250) -> pd.DataFrame:
    """Daily simple returns on cleaned prices.

    Non-trading days are not filled: a return is measured from the last traded
    price, which is how thinly traded NSE counters actually behave. Unexplained
    jumps > JUMP are dropped. Securities with fewer than min_obs returns are removed.
    """
    px, _ = clean_prices()
    r = px.ffill().pct_change(fill_method=None).where(px.notna())
    r = r.mask(r.abs() > JUMP)
    return r.loc[:, r.count() >= min_obs]


def index_level(weights: str = "equal") -> pd.Series:
    """Market proxy built from constituents (the export carries no NASI series).

    "equal": equal-weighted average of daily returns.
    "liquidity": weighted by trailing 60-day traded value (price x volume), a
    stand-in for free-float cap weighting until shares-outstanding data is added.
    """
    r = returns()
    r = r.drop(columns=[c for c in r.columns if board_of(c) in ("ETF",)], errors="ignore")
    if weights == "liquidity":
        px, _ = clean_prices()
        tv = (px * pivot("volume").reindex_like(px)).rolling(60, min_periods=20).mean().shift(1)
        w = tv[r.columns].where(r.notna())
        m = (r * w).sum(axis=1) / w.sum(axis=1)
    else:
        m = r.mean(axis=1)
    return 100 * (1 + m.fillna(0)).cumprod()


def board_of(name: str) -> str:
    return _boards().get(name, "MIMS")


@lru_cache(maxsize=1)
def _boards() -> dict[str, str]:
    return load_long().groupby("name")["board"].last().to_dict()


SECTORS_FILE = RAW_DIR.parent / "sectors.csv"


@lru_cache(maxsize=1)
def sectors() -> pd.Series:
    s = pd.read_csv(SECTORS_FILE, comment="#").set_index("name")["sector"]
    names = load_long()["name"].unique()
    return s.reindex(names).fillna("Other")
