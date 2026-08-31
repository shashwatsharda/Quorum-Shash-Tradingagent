"""Evidence partitioning -- the core idea of this project.

A standard multi-agent "investment committee" spawns several LLM personas,
hands them all the same context, and treats their agreement as signal. That
agreement is close to worthless: they are the same base model reading the same
evidence, so their errors are correlated. You get the *feeling* of a committee
without the statistical benefit of one.

Quorum enforces independence structurally instead of hoping for it:

  * TECHNICAL sees price-derived features only -- and the symbol is anonymised,
    so the model cannot fall back on what it "knows" about the company.
  * NARRATIVE sees headlines only, with no price or performance data at all,
    so it cannot rationalise a chart it has already seen.
  * REGIME sees cross-asset context only, and never the name in question.

No packet contains another packet's evidence. Members physically cannot
anchor on each other. That is the difference between a committee and a chorus.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .alpaca import Alpaca


# --------------------------------------------------------------------- helpers

def bars_to_frame(bars: list[dict]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame(columns=["t", "o", "h", "l", "c", "v"])
    df = pd.DataFrame(bars)
    df["t"] = pd.to_datetime(df["t"], format="mixed", utc=True)
    return df.sort_values("t").reset_index(drop=True)


def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return float(val) if pd.notna(val) else 50.0


def _atr_pct(df: pd.DataFrame, period: int = 14) -> float:
    """Average True Range as a percentage of price.

    ATR is the honest measure of 'how much does this thing normally move'.
    It matters here because every stop distance and every position size in
    risk.py is expressed in ATRs, not in arbitrary percentages.
    """
    prev_close = df["c"].shift(1)
    tr = pd.concat(
        [df["h"] - df["l"], (df["h"] - prev_close).abs(), (df["l"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    last = df["c"].iloc[-1]
    return float(atr / last * 100) if pd.notna(atr) and last else 2.0


def _safe_pct(series: pd.Series, n: int) -> float:
    if len(series) <= n:
        return 0.0
    return float((series.iloc[-1] / series.iloc[-1 - n] - 1) * 100)


def anonymise(symbol: str) -> str:
    """Stable pseudonym so the technical analyst reasons about the chart, not the brand.

    Same symbol always maps to the same alias within a run, so the debate stays
    coherent, but the model gets no free lunch from remembering that a given
    ticker is a beloved mega-cap.
    """
    h = hashlib.sha1(symbol.encode()).hexdigest()[:4].upper()
    return f"ASSET-{h}"


# ---------------------------------------------------------------- feature sets

def technical_features(df: pd.DataFrame) -> dict[str, Any]:
    """Everything a chart can tell you, reduced to numbers a model reads reliably.

    LLMs are poor at reading raw OHLCV rows -- they lose track of ordering and
    hallucinate levels. They are much better at reasoning over a handful of
    named, pre-computed facts. So we do the arithmetic in Python and let the
    model do what it is actually good at: weighing evidence in words.
    """
    if len(df) < 30:
        return {"insufficient_history": True, "bars": len(df)}

    close = df["c"]
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1] if len(df) >= 50 else float("nan")
    last = float(close.iloc[-1])
    hi20 = float(df["h"].tail(20).max())
    lo20 = float(df["l"].tail(20).min())
    vol = df["v"]
    vol_z = float((vol.iloc[-1] - vol.tail(20).mean()) / (vol.tail(20).std() or 1))
    rets = close.pct_change().dropna()
    realized_vol = float(rets.tail(20).std() * np.sqrt(252) * 100)

    return {
        "bars_available": int(len(df)),
        "return_1d_pct": round(_safe_pct(close, 1), 2),
        "return_5d_pct": round(_safe_pct(close, 5), 2),
        "return_20d_pct": round(_safe_pct(close, 20), 2),
        "price_vs_sma20_pct": round(float((last / sma20 - 1) * 100), 2) if pd.notna(sma20) else None,
        "price_vs_sma50_pct": round(float((last / sma50 - 1) * 100), 2) if pd.notna(sma50) else None,
        "sma20_above_sma50": bool(sma20 > sma50) if pd.notna(sma50) else None,
        "rsi_14": round(_rsi(close), 1),
        "atr_pct": round(_atr_pct(df), 2),
        "realized_vol_annualised_pct": round(realized_vol, 1),
        "pct_below_20d_high": round(float((last / hi20 - 1) * 100), 2),
        "pct_above_20d_low": round(float((last / lo20 - 1) * 100), 2),
        "volume_zscore_20d": round(vol_z, 2),
        "consecutive_up_days": int(_streak(close)),
    }


def _streak(close: pd.Series) -> int:
    d = close.diff().dropna()
    if d.empty:
        return 0
    sign = 1 if d.iloc[-1] > 0 else -1
    n = 0
    for x in reversed(d.tolist()):
        if (x > 0 and sign > 0) or (x < 0 and sign < 0):
            n += 1
        else:
            break
    return n * sign


_TICKER_RE = re.compile(r"\b[A-Z]{1,5}\b")


# Scrubbing the ticker alone is not enough -- headlines use the company name,
# and "Nvidia beats" leaks identity just as effectively as "NVDA beats".
NAME_ALIASES: dict[str, list[str]] = {
    "NVDA": ["nvidia"],
    "AMD": ["advanced micro devices"],
    "JPM": ["jpmorgan", "jp morgan", "chase"],
    "XOM": ["exxon", "exxonmobil", "exxon mobil"],
    "LLY": ["eli lilly", "lilly"],
    "COST": ["costco"],
    "BTC/USD": ["bitcoin", "btc"],
    "ETH/USD": ["ethereum", "ether", "eth"],
}


def scrub_headline(text: str, symbol: str) -> str:
    """Remove the ticker, the company name, and any bare price figures.

    Without this, the narrative analyst reads 'NVDA jumps 8%' and is really
    just doing technical analysis with extra steps -- which reintroduces
    exactly the correlation we are trying to design out.
    """
    base = symbol.split("/")[0]
    for alias in [base, *NAME_ALIASES.get(symbol, [])]:
        text = re.sub(rf"\b{re.escape(alias)}\b", "the company", text, flags=re.I)
    text = re.sub(r"[-+]?\d+(\.\d+)?\s?%", "a certain amount", text)
    # The magnitude suffix has to be swallowed too, or "$2.1B" leaves a stray
    # "B" behind and the packet leaks a hint about deal size.
    text = re.sub(
        r"\$\d+(?:,\d{3})*(?:\.\d+)?(?:\s?(?:billion|million|trillion|bn|mn)\b|[BMKT]\b)?",
        "an undisclosed sum",
        text,
    )
    return text


# -------------------------------------------------------------------- packets

@dataclass
class EvidenceBundle:
    """Three disjoint views of the same decision, plus the numbers nobody sees.

    `private` holds real prices for sizing and execution. No LLM ever receives
    it -- it exists so that the deterministic members and the risk officer can
    do arithmetic that must not be hallucinated.
    """

    symbol: str
    alias: str
    technical: dict[str, Any] = field(default_factory=dict)
    narrative: dict[str, Any] = field(default_factory=dict)
    regime: dict[str, Any] = field(default_factory=dict)
    private: dict[str, Any] = field(default_factory=dict)

    def packet_for(self, member: str) -> dict[str, Any]:
        return {
            "technical": self.technical,
            "narrative": self.narrative,
            "regime": self.regime,
        }.get(member, {})


def build_evidence(
    api: Alpaca,
    symbol: str,
    benchmark: str = "SPY",
    lookback_days: int = 200,
) -> EvidenceBundle:
    bars = api.bars(symbol, "1Day", lookback_days)
    df = bars_to_frame(bars)

    # --- technical packet: anonymised, price-derived only
    tech = {"asset": anonymise(symbol), **technical_features(df)}

    # --- narrative packet: headlines only, scrubbed of price and identity
    try:
        raw_news = api.news([symbol], limit=15, lookback_days=7)
    except Exception:
        raw_news = []
    headlines = [
        {
            "headline": scrub_headline(n.get("headline", ""), symbol),
            "source": n.get("source", ""),
            "age_hours": _age_hours(n.get("created_at", "")),
        }
        for n in raw_news
    ]
    narrative = {
        "asset": anonymise(symbol),
        "headline_count_7d": len(headlines),
        "headlines": headlines[:12],
        "note": "No price, return, or ticker information is available to you.",
    }

    # --- regime packet: market-wide only, never the name in question
    regime = build_regime(api, benchmark)

    private = {
        "last_close": float(df["c"].iloc[-1]) if len(df) else 0.0,
        "atr_pct": _atr_pct(df) if len(df) >= 15 else 2.0,
        "frame": df,
    }
    return EvidenceBundle(symbol, anonymise(symbol), tech, narrative, regime, private)


def _age_hours(ts: str) -> float | None:
    if not ts:
        return None
    try:
        t = pd.to_datetime(ts, utc=True, format="mixed")
        return round(float((pd.Timestamp.utcnow() - t).total_seconds() / 3600), 1)
    except Exception:
        return None


_REGIME_CACHE: dict[str, dict] = {}


def build_regime(api: Alpaca, benchmark: str = "SPY") -> dict[str, Any]:
    """Cross-asset context. Cached per run because it is identical for every symbol."""
    if benchmark in _REGIME_CACHE:
        return _REGIME_CACHE[benchmark]
    try:
        df = bars_to_frame(api.bars(benchmark, "1Day", 120))
        close = df["c"]
        out = {
            "market_proxy": "broad US equity index ETF",
            "index_return_5d_pct": round(_safe_pct(close, 5), 2),
            "index_return_20d_pct": round(_safe_pct(close, 20), 2),
            "index_above_50d_avg": bool(close.iloc[-1] > close.rolling(50).mean().iloc[-1]),
            "index_realized_vol_pct": round(
                float(close.pct_change().tail(20).std() * np.sqrt(252) * 100), 1
            ),
            "index_drawdown_from_60d_high_pct": round(
                float((close.iloc[-1] / close.tail(60).max() - 1) * 100), 2
            ),
            "note": "You do not know which individual asset is under discussion.",
        }
    except Exception as exc:  # a missing benchmark must not kill the run
        out = {"error": str(exc)[:120], "note": "Regime data unavailable; vote flat."}
    _REGIME_CACHE[benchmark] = out
    return out
