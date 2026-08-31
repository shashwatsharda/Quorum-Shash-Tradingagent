"""Shared data structures.

Everything that crosses a module boundary in Quorum is one of these.
Keeping them dumb and serialisable is what lets the audit log be a
faithful record rather than a summary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Literal

Stance = Literal["long", "short", "flat"]
GateAction = Literal["approve", "resize", "reject"]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Vote:
    """One committee member's opinion on one symbol.

    `confidence` is the member's stated probability that its stance is
    directionally right over the decision horizon. It is NOT a conviction
    adjective dressed up as a number -- calibration.py later scores every
    member on how honest this figure was.
    """

    member: str
    symbol: str
    stance: Stance
    confidence: float
    thesis: str
    key_evidence: list[str] = field(default_factory=list)
    disconfirming: str = ""
    evidence_slice: str = ""      # which partition this member was allowed to see
    is_llm: bool = True
    error: str | None = None

    def __post_init__(self) -> None:
        # A member that cannot express uncertainty cannot be scored.
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    @property
    def signed_confidence(self) -> float:
        """+p for long, -p for short, 0 for flat. Used for aggregation."""
        if self.stance == "long":
            return self.confidence
        if self.stance == "short":
            return -self.confidence
        return 0.0


@dataclass
class DebateTurn:
    role: Literal["bull", "bear"]
    round_no: int
    argument: str
    attacks: str = ""     # which member's claim this turn is challenging


@dataclass
class Proposal:
    """What the PM wants to do, before anyone checks whether it is allowed."""

    symbol: str
    side: Literal["buy", "sell"]
    qty: float
    notional: float
    entry_ref_price: float
    stop_price: float | None
    take_profit: float | None
    horizon_days: int
    confidence: float
    rationale: str
    dissent: str = ""     # the strongest surviving objection, recorded on purpose


@dataclass
class GateDecision:
    """The risk officer's ruling. Deterministic, and always explained."""

    action: GateAction
    original_qty: float
    final_qty: float
    reasons: list[str] = field(default_factory=list)
    breached_rules: list[str] = field(default_factory=list)

    @property
    def approved(self) -> bool:
        return self.action in ("approve", "resize") and self.final_qty > 0


@dataclass
class DecisionRecord:
    """One full committee cycle for one symbol, start to finish.

    This is the atom of the audit log. If it is not in here, it did not
    happen -- that is the whole point of writing it append-only.
    """

    ts: str
    symbol: str
    votes: list[Vote]
    debate: list[DebateTurn]
    proposal: Proposal | None
    gate: GateDecision | None
    executed_order_id: str | None = None
    outcome: dict[str, Any] | None = None   # filled in later by calibration.py
    run_id: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)
