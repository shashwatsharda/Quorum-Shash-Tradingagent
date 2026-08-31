"""The seat that is not an LLM.

Every other member of this committee is a language model. That means every
other member shares a prior, a training set, and a failure mode. If the whole
committee is drawn from one distribution, then the committee's "consensus" is
just that distribution talking to itself in four voices.

So one seat is deterministic. It reads the same price history, applies a fixed
rulebook, and cannot be argued with, flattered, or prompt-injected. Its job is
not to be smarter than the models -- it usually isn't -- but to be *wrong in a
different direction*, which is the only thing that makes an ensemble worth
having.

The rulebook itself is deliberately boring and well-documented: cross-sectional
momentum with a trend filter and a volatility brake. Boring is the point. If
the LLMs consistently fail to beat a rulebook this simple, the calibration
leaderboard will say so out loud -- and that finding is more interesting than
a committee that always agrees.
"""

from __future__ import annotations

from typing import Any

from ..types import Vote


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def quant_vote(symbol: str, technical: dict[str, Any]) -> Vote:
    if technical.get("insufficient_history"):
        return Vote(
            member="quant",
            symbol=symbol,
            stance="flat",
            confidence=0.5,
            thesis="Insufficient price history to score. Abstaining.",
            evidence_slice="technical (deterministic)",
            is_llm=False,
        )

    score = 0.0
    reasons: list[str] = []

    # 1. Medium-term momentum -- the most robust anomaly in the literature,
    #    documented since Jegadeesh & Titman (1993) and still stubbornly alive.
    r20 = technical.get("return_20d_pct") or 0.0
    contrib = _clamp(r20 / 10.0, -1.0, 1.0)
    score += contrib
    reasons.append(f"20d momentum {r20:+.1f}% -> {contrib:+.2f}")

    # 2. Trend filter -- momentum's known weakness is buying into reversals.
    #    Requiring price above its own moving averages is the cheapest guard.
    above20 = technical.get("price_vs_sma20_pct")
    if above20 is not None:
        contrib = _clamp(above20 / 5.0, -0.8, 0.8)
        score += contrib
        reasons.append(f"price vs 20d avg {above20:+.1f}% -> {contrib:+.2f}")
    if technical.get("sma20_above_sma50") is True:
        score += 0.3
        reasons.append("20d avg above 50d avg -> +0.30")
    elif technical.get("sma20_above_sma50") is False:
        score -= 0.3
        reasons.append("20d avg below 50d avg -> -0.30")

    # 3. Exhaustion brake -- fade the last leg of a stretched move rather than
    #    chasing it. This is what stops pure momentum buying the top tick.
    rsi = technical.get("rsi_14", 50.0)
    if rsi > 75:
        score -= 0.5
        reasons.append(f"RSI {rsi:.0f} overbought -> -0.50")
    elif rsi < 25:
        score += 0.5
        reasons.append(f"RSI {rsi:.0f} oversold -> +0.50")

    streak = technical.get("consecutive_up_days", 0)
    if abs(streak) >= 5:
        score -= 0.25 * (1 if streak > 0 else -1)
        reasons.append(f"{abs(streak)} straight days one way -> mean-reversion tilt")

    # 4. Volatility brake -- identical edge in a wilder asset is a worse bet,
    #    because the same stop gets hit by noise. Shrink conviction, not size
    #    (size is risk.py's job, and mixing the two is how systems get confused).
    vol = technical.get("realized_vol_annualised_pct", 25.0) or 25.0
    if vol > 60:
        score *= 0.6
        reasons.append(f"realized vol {vol:.0f}% -> conviction cut 40%")
    elif vol > 40:
        score *= 0.8
        reasons.append(f"realized vol {vol:.0f}% -> conviction cut 20%")

    # Map an unbounded score to a probability. The 0.5 + score/8 mapping caps
    # the deterministic member at 78% confidence: a rulebook this simple has no
    # business ever claiming near-certainty, and capping it keeps its Brier
    # score honest.
    confidence = _clamp(0.5 + score / 8.0, 0.22, 0.78)
    if abs(score) < 0.4:
        stance, confidence = "flat", 0.5
    else:
        stance = "long" if score > 0 else "short"
        confidence = confidence if stance == "long" else 1 - confidence

    return Vote(
        member="quant",
        symbol=symbol,
        stance=stance,  # type: ignore[arg-type]
        confidence=round(confidence, 3),
        thesis=f"Rulebook score {score:+.2f}. " + "; ".join(reasons),
        key_evidence=reasons,
        disconfirming=(
            "This member has no knowledge of news, earnings, or macro events. "
            "It will happily buy a company on the morning it is being investigated."
        ),
        evidence_slice="technical (deterministic)",
        is_llm=False,
    )
