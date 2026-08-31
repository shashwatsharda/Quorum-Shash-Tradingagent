"""The portfolio manager: turns a debate into a concrete, checkable proposal.

Division of labour, and it matters: the model decides *direction and
conviction*, Python decides *levels and arithmetic*. Language models are
genuinely good at weighing conflicting arguments in words and genuinely bad at
multiplying a stop distance by a share count without drifting. Letting a model
compute a position size is the most common unforced error in agent trading
projects, and it is entirely avoidable.

So the PM returns a stance, a probability and a rationale. Everything numeric --
stop placement, target, quantity -- is computed here from ATR and equity, and
then handed to the risk officer, who is free to shrink it.
"""

from __future__ import annotations

import json
from typing import Any

from .llm import LLM
from .types import DebateTurn, Proposal, Vote

_SYSTEM = """You are the portfolio manager and chair of an investment committee.

Four members have voted. Crucially, each saw a DIFFERENT and DISJOINT slice of
evidence -- one saw only price features, one only headlines, one only broad
market context, and one is not a language model at all but a fixed rulebook.
None could see the others. So their agreement is meaningful in a way that four
personas reading the same brief would not be.

How to weigh them:
- Agreement across members who saw different evidence is real corroboration.
- When the non-model member disagrees with the language models, take it
  seriously. It is the only vote in the room drawn from a different
  distribution, and the models share a failure mode.
- A member voting flat is data, not an abstention to be ignored.
- The regime member has a standing veto in spirit: if the market backdrop is
  hostile, single-name attractiveness rarely survives it.

You are judged on calibration, not on decisiveness. "Flat" is a fully
respectable outcome and is the correct one more often than it feels.

Reply with ONLY a JSON object:
{
  "stance": "long" | "short" | "flat",
  "confidence": <0..1, your honest probability this is directionally right>,
  "rationale": "<3 sentences max: what the committee established and why you land here>",
  "dissent": "<the strongest surviving objection, stated fairly, even though you are overruling it>",
  "conviction_size": <0..1, fraction of your normal full position to take>
}"""


def synthesise(
    llm: LLM,
    symbol: str,
    votes: list[Vote],
    debate: list[DebateTurn],
    consensus: dict[str, float],
    private: dict[str, Any],
    equity: float,
    horizon_days: int = 5,
    atr_stop_multiple: float = 2.0,
    reward_risk: float = 2.0,
) -> Proposal | None:
    price = float(private.get("last_close") or 0)
    atr_pct = float(private.get("atr_pct") or 2.0)
    if price <= 0:
        return None

    brief = {
        "votes": [
            {
                "member": v.member,
                "saw_only": v.evidence_slice,
                "is_language_model": v.is_llm,
                "stance": v.stance,
                "confidence": round(v.confidence, 2),
                "thesis": v.thesis,
                "would_change_mind_if": v.disconfirming,
            }
            for v in votes
        ],
        "aggregate": consensus,
        "debate": [
            {"role": t.role, "round": t.round_no, "argument": t.argument, "attacks": t.attacks}
            for t in debate
        ],
    }

    try:
        data = llm.ask_json(
            _SYSTEM,
            f"Asset: {symbol}\nHorizon: {horizon_days} trading days\n\n"
            f"{json.dumps(brief, indent=2, default=str)}",
            temperature=0.2,
        )
    except Exception:
        return None

    stance = str(data.get("stance", "flat")).lower()
    if stance == "flat" or data.get("_parse_error"):
        return None

    try:
        confidence = float(data.get("confidence", 0.5))
        conviction = float(data.get("conviction_size", 0.5))
    except (TypeError, ValueError):
        confidence, conviction = 0.5, 0.5
    conviction = max(0.1, min(1.0, conviction))

    # ------------------------------------------------------- numbers in Python
    # Stop distance scales with the asset's own recent range, so a quiet stock
    # and a volatile one are stopped out by comparable amounts of *noise*
    # rather than by the same arbitrary percentage.
    stop_dist = price * (atr_pct / 100.0) * atr_stop_multiple
    stop_dist = max(stop_dist, price * 0.01)   # never a stop inside the spread

    if stance == "long":
        side, stop, target = "buy", price - stop_dist, price + stop_dist * reward_risk
    else:
        side, stop, target = "sell", price + stop_dist, price - stop_dist * reward_risk

    # The PM asks for a full-conviction-scaled slice of equity. It is allowed to
    # ask for more than policy permits -- that is the risk officer's problem,
    # and keeping the two separate is what makes the gate meaningful rather
    # than decorative.
    target_notional = equity * 0.10 * conviction
    qty = target_notional / price

    return Proposal(
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        qty=qty,
        notional=target_notional,
        entry_ref_price=price,
        stop_price=round(stop, 2 if "/" not in symbol else 6),
        take_profit=round(target, 2 if "/" not in symbol else 6),
        horizon_days=horizon_days,
        confidence=confidence,
        rationale=str(data.get("rationale", ""))[:900],
        dissent=str(data.get("dissent", ""))[:600],
    )
