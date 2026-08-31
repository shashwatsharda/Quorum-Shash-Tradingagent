"""The three language-model seats.

Each gets a different system prompt AND a different, disjoint slice of evidence.
The second half of that sentence is what makes this different from the usual
persona-only committee: personas alone give you one model wearing hats, and
hats do not decorrelate errors.

Every analyst must state a probability and must name what would change its
mind. Forcing a disconfirming condition is a cheap, well-evidenced debiasing
device -- it is the same trick as a pre-mortem, and it gives the debate stage
something concrete to attack.
"""

from __future__ import annotations

import json
from typing import Any

from ..llm import LLM
from ..types import Vote

_OUTPUT_CONTRACT = """
Reply with ONLY a JSON object, no prose before or after:
{
  "stance": "long" | "short" | "flat",
  "confidence": <number between 0 and 1: your honest probability that this stance
                 is directionally right over the stated horizon. 0.5 means you
                 have no view. Do not inflate it; you are scored on calibration,
                 not on boldness.>,
  "thesis": "<two sentences maximum>",
  "key_evidence": ["<specific item from your packet>", "..."],
  "disconfirming": "<the single observation that would flip your view>"
}
""".strip()

_SHARED_RULES = """
Hard rules:
- Reason ONLY from the evidence packet you are given. You have deliberately been
  denied other information. If your packet cannot support a view, vote "flat"
  with confidence 0.5. Abstaining is a valid, respected outcome here.
- Never invent a data point that is not in your packet.
- You are one of four independent members. You cannot see the others and must
  not speculate about what they think.
- Your confidence is scored against reality afterwards using a Brier score.
  Systematic overconfidence will be visible and will cost you standing.
""".strip()

PROFILES: dict[str, dict[str, str]] = {
    "technical": {
        "slice": "technical",
        "system": """You are the price-action analyst on an investment committee.

You receive pre-computed features from a single asset's daily price history.
The asset is anonymised on purpose: you are being asked to read a chart, not to
recall opinions about a company. If you catch yourself guessing the ticker,
stop -- that guess is exactly the contamination this seat is designed to remove.

You care about: trend alignment, momentum persistence versus exhaustion,
volatility regime, and position within recent range. You do not care about
narrative, and you have no access to it.""",
    },
    "narrative": {
        "slice": "narrative",
        "system": """You are the narrative analyst on an investment committee.

You receive recent headlines only. Prices, returns, and the asset's identity
have been scrubbed, deliberately: your job is to judge whether the flow of news
is improving or deteriorating, not to explain a move you already saw.

You care about: the direction and freshness of news flow, whether items are
company-specific or ambient noise, and whether coverage is concentrated in a
short burst (which usually signals an event) or spread thin.

Be sceptical. Most headlines are noise. An empty or bland news slate is a
genuine reason to vote flat, not a reason to strain for a story.""",
    },
    "regime": {
        "slice": "regime",
        "system": """You are the market-regime analyst on an investment committee.

You receive broad-market context only, and you are never told which individual
asset is under discussion. You are effectively answering one question: is this
a market in which a new directional risk position deserves to be taken at all?

You care about: index trend, volatility level, and drawdown from recent highs.
In a high-volatility drawdown, the correct answer is usually "flat" regardless
of how attractive any single name looks -- correlations converge in stress and
single-name skill matters less than people think.""",
    },
}


def llm_vote(
    llm: LLM,
    member: str,
    symbol: str,
    packet: dict[str, Any],
    horizon_days: int = 5,
) -> Vote:
    profile = PROFILES[member]
    user = (
        f"Decision horizon: {horizon_days} trading days.\n\n"
        f"YOUR EVIDENCE PACKET (this is everything you get):\n"
        f"{json.dumps(packet, indent=2, default=str)}\n\n"
        f"{_SHARED_RULES}\n\n{_OUTPUT_CONTRACT}"
    )
    try:
        data = llm.ask_json(profile["system"], user, temperature=0.2)
    except Exception as exc:
        return Vote(
            member=member,
            symbol=symbol,
            stance="flat",
            confidence=0.5,
            thesis="Member unavailable; abstained.",
            evidence_slice=profile["slice"],
            error=str(exc)[:200],
        )

    if data.get("_parse_error"):
        return Vote(
            member=member,
            symbol=symbol,
            stance="flat",
            confidence=0.5,
            thesis="Member returned unparseable output; abstained.",
            evidence_slice=profile["slice"],
            error="parse_error",
        )

    stance = str(data.get("stance", "flat")).lower()
    if stance not in ("long", "short", "flat"):
        stance = "flat"
    try:
        conf = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5

    return Vote(
        member=member,
        symbol=symbol,
        stance=stance,  # type: ignore[arg-type]
        confidence=conf,
        thesis=str(data.get("thesis", ""))[:600],
        key_evidence=[str(x)[:200] for x in (data.get("key_evidence") or [])][:5],
        disconfirming=str(data.get("disconfirming", ""))[:300],
        evidence_slice=profile["slice"],
        is_llm=True,
    )
