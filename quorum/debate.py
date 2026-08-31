"""Cross-examination, then a decision.

The debate stage is where most multi-agent trading demos quietly go wrong. Left
unconstrained, two LLM personas converge within a round or two on whichever
position was argued more fluently, and the transcript looks like deliberation
while functioning as mutual reassurance.

Three constraints keep it useful here:

  1. **Fixed rounds.** Two, always. Debate until agreement is a recipe for
     manufactured consensus; the goal is to surface the strongest objection,
     not to reach a verdict.
  2. **Attack the evidence, not the conclusion.** Each turn must name the
     specific claim it is challenging. Vague rebuttals are worthless and this
     forces them to be concrete.
  3. **The dissent survives.** Whatever the bear says is carried forward into
     the proposal and written into the audit log even when the committee goes
     long. A decision record that only contains the winning argument is a
     press release, not a record.
"""

from __future__ import annotations

import json

from .llm import LLM
from .types import DebateTurn, Vote

ROUNDS = 2

_BULL_SYSTEM = """You are the bull advocate in an investment committee.

You did not gather any evidence yourself. You have the four members' votes and
must build the strongest honest case FOR taking a long position.

Rules:
- Cite specific claims from specific members by name.
- When you attack the bear case, attack the evidence behind it, not its tone.
- If the evidence genuinely does not support a long, say so plainly. Advocacy
  that ignores the evidence is worse than useless -- it corrupts the record.
- Maximum 120 words."""

_BEAR_SYSTEM = """You are the bear advocate in an investment committee.

You did not gather any evidence yourself. You have the four members' votes and
must build the strongest honest case AGAINST taking the position.

Your most valuable move is not pessimism -- it is finding where two members
agree for reasons that are not actually independent, or where a member's stated
confidence is not supported by the evidence it cited.

Rules:
- Cite specific claims from specific members by name.
- Name the single most likely way this trade loses money.
- Maximum 120 words."""

_CONTRACT = """
Reply with ONLY a JSON object:
{"argument": "<your case, max 120 words>", "attacks": "<the specific member and claim you are challenging>"}
""".strip()


def _votes_brief(votes: list[Vote]) -> str:
    return json.dumps(
        [
            {
                "member": v.member,
                "sees_only": v.evidence_slice,
                "stance": v.stance,
                "confidence": round(v.confidence, 2),
                "thesis": v.thesis,
                "evidence": v.key_evidence,
                "would_change_mind_if": v.disconfirming,
                "is_language_model": v.is_llm,
            }
            for v in votes
        ],
        indent=2,
    )


def run_debate(llm: LLM, symbol: str, votes: list[Vote]) -> list[DebateTurn]:
    transcript: list[DebateTurn] = []
    brief = _votes_brief(votes)

    for rnd in range(1, ROUNDS + 1):
        prior = "\n\n".join(f"[{t.role} r{t.round_no}] {t.argument}" for t in transcript)
        for role, system in (("bull", _BULL_SYSTEM), ("bear", _BEAR_SYSTEM)):
            user = (
                f"Asset under discussion: {symbol}\n\n"
                f"COMMITTEE VOTES (note: each member saw a different, disjoint slice "
                f"of evidence and could not see the others):\n{brief}\n\n"
                + (f"DEBATE SO FAR:\n{prior}\n\n" if prior else "")
                + f"Round {rnd} of {ROUNDS}.\n\n{_CONTRACT}"
            )
            try:
                data = llm.ask_json(system, user, temperature=0.5)
                transcript.append(
                    DebateTurn(
                        role=role,  # type: ignore[arg-type]
                        round_no=rnd,
                        argument=str(data.get("argument", ""))[:900],
                        attacks=str(data.get("attacks", ""))[:300],
                    )
                )
            except Exception as exc:
                transcript.append(
                    DebateTurn(role=role, round_no=rnd,  # type: ignore[arg-type]
                               argument=f"[unavailable: {str(exc)[:120]}]")
                )
            prior = "\n\n".join(f"[{t.role} r{t.round_no}] {t.argument}" for t in transcript)

    return transcript


def consensus_strength(votes: list[Vote]) -> dict[str, float]:
    """Aggregate the votes, and measure how much the agreement is worth.

    `agreement` alone is a trap: four members agreeing tells you nothing if
    they are all the same model reading the same data. Here the members saw
    disjoint evidence and one is not a model at all, so agreement is
    informative -- but only in proportion to how *dispersed* the members are.

    `independence_weighted` discounts the aggregate score by how much of the
    agreement comes from language models alone. When the deterministic member
    dissents from a unanimous LLM bloc, that dissent is worth more than one
    vote, because it is the only vote drawn from a different distribution.
    """
    if not votes:
        return {"score": 0.0, "net_conviction": 0.0, "independence_weighted": 0.0}

    signed = [v.signed_confidence for v in votes]
    score = sum(signed) / len(signed)

    directions = [1 if s > 0.05 else (-1 if s < -0.05 else 0) for s in signed]
    non_flat = [d for d in directions if d != 0]
    # Net-cancellation, not majority-fraction: opposing directions cancel out
    # before dividing, so a 2-1 split nets to 1/3 (weak), not the 66% a
    # majority vote would report.
    net_conviction = (
        abs(sum(non_flat)) / len(non_flat) if non_flat else 0.0
    )

    llm_dirs = [d for d, v in zip(directions, votes) if v.is_llm and d != 0]
    quant_dirs = [d for d, v in zip(directions, votes) if not v.is_llm and d != 0]
    penalty = 1.0
    if llm_dirs and quant_dirs and (sum(llm_dirs) > 0) != (sum(quant_dirs) > 0):
        penalty = 0.6      # the out-of-distribution member disagrees: discount hard
    elif not quant_dirs:
        penalty = 0.85     # no independent confirmation available

    return {
        "score": round(score, 4),
        "net_conviction": round(net_conviction, 3),
        "independence_weighted": round(score * net_conviction * penalty, 4),
        "penalty_applied": penalty,
    }
