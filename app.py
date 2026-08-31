"""Streamlit dashboard.

    streamlit run app.py

Three tabs, in the order a judge should see them:
  1. Latest decision  -- the committee's reasoning, made legible
  2. Risk gate        -- what was blocked or resized, and why
  3. Calibration      -- which members are actually any good

Tab 3 is the one to linger on. Every trading-agent demo shows you a decision.
Almost none shows you whether its own components were worth listening to.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from quorum.audit import AuditLog
from quorum.calibration import committee_summary, pairwise_agreement, score_members

st.set_page_config(page_title="Quorum", page_icon="⚖️", layout="wide")

ROOT = Path(__file__).parent
cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
records = AuditLog(cfg.get("log_path", "data/decisions.jsonl")).read()

st.title("Quorum")
st.caption(
    "An investment committee whose members are forced to be independent — "
    "and a risk officer that is not a language model."
)

if not records:
    st.warning("No decisions logged yet. Run `python -m quorum.run scan --dry-run` first.")
    st.stop()

summary = committee_summary(records)
cols = st.columns(6)
for col, (label, key) in zip(
    cols,
    [("Decisions", "decisions_logged"), ("Resolved", "resolved"),
     ("Approved", "gate_approve"), ("Resized", "gate_resize"),
     ("Rejected", "gate_reject"), ("Win rate", "win_rate")],
):
    val = summary.get(key)
    col.metric(label, "—" if val is None else (f"{val:.0%}" if key == "win_rate" else val))

tab1, tab2, tab3 = st.tabs(["Committee", "Risk gate", "Calibration"])

# --------------------------------------------------------------- committee
with tab1:
    symbols = sorted({r["symbol"] for r in records})
    symbol = st.selectbox("Asset", symbols)
    latest = [r for r in records if r["symbol"] == symbol][-1]

    st.subheader(f"{symbol} — {latest['ts'][:19].replace('T', ' ')} UTC")

    votes = pd.DataFrame(latest["votes"])
    if not votes.empty:
        show = votes[["member", "evidence_slice", "is_llm", "stance", "confidence", "thesis"]]
        show = show.rename(columns={"evidence_slice": "saw only", "is_llm": "model?"})
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.caption(
            "Each member saw a **different, disjoint** slice of evidence and could not "
            "see the others. One is a fixed rulebook, not a model — it is the only vote "
            "in the room drawn from a different distribution."
        )

    with st.expander("Full theses and what would change each member's mind"):
        for v in latest["votes"]:
            st.markdown(
                f"**{v['member']}** ({'model' if v['is_llm'] else 'rulebook'}) — "
                f"*{v['stance']}, {v['confidence']:.0%}*  \n{v['thesis']}"
            )
            if v.get("disconfirming"):
                st.caption(f"Would change its mind if: {v['disconfirming']}")

    if latest.get("debate"):
        st.subheader("Cross-examination")
        for t in latest["debate"]:
            with st.chat_message("user" if t["role"] == "bull" else "assistant"):
                st.markdown(f"**{t['role'].title()}, round {t['round_no']}**  \n{t['argument']}")
                if t.get("attacks"):
                    st.caption(f"Challenging: {t['attacks']}")

    if latest.get("proposal"):
        p = latest["proposal"]
        st.subheader("Chair's proposal")
        st.markdown(f"**{p['side'].upper()} {p['qty']:.4f}** @ ~{p['entry_ref_price']:.2f} · "
                    f"stop {p['stop_price']} · confidence {p['confidence']:.0%}")
        st.write(p["rationale"])
        if p.get("dissent"):
            st.info(f"**Recorded dissent:** {p['dissent']}")
            st.caption(
                "The losing argument is kept on the record on purpose. A decision log "
                "containing only the winning case is a press release, not a record."
            )
    else:
        st.info("Committee landed flat. No proposal — which is a legitimate outcome.")

# --------------------------------------------------------------- risk gate
with tab2:
    rows = []
    for r in records:
        g = r.get("gate")
        if not g:
            continue
        rows.append({
            "time": r["ts"][:16].replace("T", " "),
            "symbol": r["symbol"],
            "action": g["action"],
            "requested": round(g["original_qty"], 4),
            "permitted": round(g["final_qty"], 4),
            "rules touched": ", ".join(g.get("breached_rules", [])) or "—",
            "reason": " ".join(g.get("reasons", []))[:160],
        })
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        blocked = df[df["action"] != "approve"]
        st.metric("Orders the gate changed or refused",
                  f"{len(blocked)} of {len(df)}")
        st.caption(
            "Improvements in signal quality are marginal and diminishing. Failures of "
            "constraint are unbounded. That asymmetry — not model quality — is why "
            "pre-trade risk controls are mandated (SEC Rule 15c3-5) and models are not."
        )
    else:
        st.info("No gated proposals yet.")

# -------------------------------------------------------------- calibration
with tab3:
    scores = score_members(records)
    if any(s["brier"] is not None for s in scores):
        df = pd.DataFrame(scores)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(
            "**Brier score** is the mean squared error of a stated probability. "
            "Always guessing 50/50 scores 0.25 — that is the know-nothing baseline. "
            "Below 0.25 the member is adding information; above it, the member is "
            "confidently wrong and you would do better inverting it. Confident errors "
            "are punished quadratically, which is the point."
        )
    else:
        st.info("No resolved decisions yet. Run `python -m quorum.run resolve` "
                "once the horizon has passed.")

    st.subheader("Pairwise agreement")
    pa = pairwise_agreement(records)
    if pa:
        st.dataframe(pd.DataFrame(pa), use_container_width=True, hide_index=True)
        st.caption(
            "This is the test of whether the evidence partition works. Two model seats "
            "agreeing most of the time would mean their errors are correlated and the "
            "'committee' is really one opinion counted three times."
        )

    with st.expander("Raw audit log (last 5 records)"):
        st.code(
            "\n".join(json.dumps(r, default=str)[:1200] for r in records[-5:]),
            language="json",
        )
