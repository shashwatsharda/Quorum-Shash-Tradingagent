"""Scoring the committee members against reality.

This is the part almost no agent project has, and it is the most interesting
thing in the repo.

Every member states a probability, not an adjective. Once the horizon has
passed we know what actually happened, so we can ask the only question that
matters about a forecaster: **were its stated probabilities honest?**

The Brier score answers it. For a binary outcome it is simply the mean squared
error of the probability:

    Brier = mean( (p - o)^2 )        where o is 1 if the event happened, else 0

Intuition, because the formula hides it: guessing 0.5 on everything scores
0.25. That is the "I know nothing" baseline. Anything above 0.25 means the
member is worse than useless -- it is not merely uninformed, it is confidently
wrong, and you would do better inverting it. Below 0.25 means it is adding
information. The score punishes confident errors quadratically, which is
exactly the behaviour you want: being 95% sure and wrong should hurt far more
than being 55% sure and wrong.

Two members can also be compared on *agreement*: if two members' votes are
highly correlated, the second one is not adding a vote, it is adding an echo.
That is the empirical test of whether the evidence partition in evidence.py is
doing its job.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

BASELINE_BRIER = 0.25      # the score of always saying 50/50


def brier(prob: float, happened: bool) -> float:
    return (prob - (1.0 if happened else 0.0)) ** 2


def score_members(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-member Brier score over every resolved decision in the log."""
    acc: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "brier": 0.0, "hits": 0, "flat": 0, "conf": 0.0, "is_llm": True}
    )

    for rec in records:
        outcome = rec.get("outcome") or {}
        if "return_pct" not in outcome:
            continue
        went_up = float(outcome["return_pct"]) > 0

        for v in rec.get("votes", []):
            m = v.get("member", "?")
            stance = v.get("stance", "flat")
            conf = float(v.get("confidence", 0.5))
            a = acc[m]
            a["is_llm"] = bool(v.get("is_llm", True))
            if stance == "flat":
                a["flat"] += 1
                continue
            # Convert every stance into "probability the asset rises", so that
            # longs and shorts are scored on one common scale.
            p_up = conf if stance == "long" else 1.0 - conf
            a["n"] += 1
            a["brier"] += brier(p_up, went_up)
            a["conf"] += max(conf, 1 - conf)
            a["hits"] += int((stance == "long") == went_up)

    rows = []
    for member, a in acc.items():
        n = a["n"]
        rows.append(
            {
                "member": member,
                "is_llm": a["is_llm"],
                "resolved_votes": n,
                "abstentions": a["flat"],
                "brier": round(a["brier"] / n, 4) if n else None,
                "skill_vs_coinflip": round(BASELINE_BRIER - a["brier"] / n, 4) if n else None,
                "hit_rate": round(a["hits"] / n, 3) if n else None,
                "avg_confidence": round(a["conf"] / n, 3) if n else None,
                "overconfidence": (
                    round(a["conf"] / n - a["hits"] / n, 3) if n else None
                ),
            }
        )
    # Positive skill first; a member with no resolved votes sorts last.
    return sorted(rows, key=lambda r: (r["skill_vs_coinflip"] is None,
                                       -(r["skill_vs_coinflip"] or 0)))


def pairwise_agreement(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """How often do two members land on the same side?

    High agreement between two LLM members is the smoking gun for correlated
    error -- it means the evidence partition leaked, or that both are simply
    reciting the same prior. This table is how you *prove* the design works
    instead of asserting it in a slide.
    """
    pair: dict[tuple[str, str], list[int]] = defaultdict(list)
    for rec in records:
        votes = {v["member"]: v for v in rec.get("votes", [])}
        names = sorted(votes)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                sa, sb = votes[a]["stance"], votes[b]["stance"]
                if sa == "flat" or sb == "flat":
                    continue
                pair[(a, b)].append(int(sa == sb))
    return [
        {
            "pair": f"{a} / {b}",
            "co_votes": len(vals),
            "agreement_rate": round(sum(vals) / len(vals), 3) if vals else None,
        }
        for (a, b), vals in sorted(pair.items())
    ]


def committee_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    resolved = [r for r in records if (r.get("outcome") or {}).get("return_pct") is not None]
    gated = [r for r in records if r.get("gate")]
    actions = defaultdict(int)
    for r in gated:
        actions[(r["gate"] or {}).get("action", "?")] += 1
    wins = [
        r for r in resolved
        if (r.get("gate") or {}).get("final_qty", 0) > 0
        and float(r["outcome"]["return_pct"]) * (1 if r["proposal"]["side"] == "buy" else -1) > 0
    ]
    taken = [r for r in resolved if (r.get("gate") or {}).get("final_qty", 0) > 0]
    return {
        "decisions_logged": len(records),
        "resolved": len(resolved),
        "gate_approve": actions["approve"],
        "gate_resize": actions["resize"],
        "gate_reject": actions["reject"],
        "positions_taken": len(taken),
        "win_rate": round(len(wins) / len(taken), 3) if taken else None,
    }
