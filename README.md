# Quorum

**An investment committee whose members are structurally prevented from agreeing with each other — and a risk officer that is not a language model.**

Built for the [Alpaca AI Trading Agents Hackathon](https://lablab.ai/ai-hackathons/alpaca-ai-trading-agents-hackathon), 28 Aug – 4 Sep 2026.

---

## The problem with multi-agent trading committees

The standard pattern is well known: spawn several LLM personas — a technical analyst, a news analyst, a macro analyst — give them all the same context, let them debate, and treat their agreement as signal.

That agreement is close to worthless.

The appeal of a committee rests on **ensembling**: independent noisy estimates average out toward the truth. But these agents are not independent. They are the same base model, with the same training distribution, reading the same evidence. Their errors are correlated. What you get is the *feeling* of a committee without the statistical benefit of one — one opinion, counted four times, wearing four hats.

Worse, unconstrained LLM debate converges within a round or two on whichever position was argued more fluently. The transcript looks like deliberation and functions as mutual reassurance.

## What Quorum does differently

Three structural changes, each cheap to implement and each attacking a specific failure:

### 1. The evidence is partitioned, not shared

No member sees another member's data. Enforced in code, not in prompts.

| Member | Sees | Cannot see |
|---|---|---|
| `technical` | Pre-computed price features, **ticker anonymised** | News, macro, the asset's identity |
| `narrative` | Headlines, **scrubbed of ticker, company name and every price figure** | Any price, return, or performance data |
| `regime` | Broad-market context only | Which asset is even under discussion |
| `quant` | Price features | Everything else — and it is not a model |

Anonymising the ticker for the technical seat matters more than it sounds. Given the symbol, a model stops reading the chart and starts reciting what it remembers about the company — which reintroduces exactly the correlation the partition is designed to remove.

### 2. One seat is not a language model

`quorum/members/quant.py` is a fixed rulebook: cross-sectional momentum, a trend filter, an exhaustion brake and a volatility brake. It is deliberately boring and cannot be argued with, flattered, or prompt-injected.

Its job is not to be smarter than the models. It usually isn't. Its job is to be **wrong in a different direction**, which is the only property that makes an ensemble worth having. When it dissents from a unanimous model bloc, `consensus_strength()` discounts the aggregate by 40% — because that one dissent is the only vote in the room drawn from a different distribution.

### 3. The risk officer has a veto and no model

`quorum/risk.py` is deterministic Python. The portfolio manager proposes; this decides.

Long-Term Capital Management had two Nobel laureates on signal and no binding constraint on leverage. Knight Capital lost ~$440m in 45 minutes in 2012 because a bad deploy let an algorithm trade unsupervised. Neither was a failure of prediction. Both were failures of permission.

The asymmetry is the whole argument: **improvements in signal quality are marginal and diminishing; failures of constraint are unbounded.** That is why the industry's answer to those blow-ups was mandatory pre-trade risk controls (SEC Rule 15c3-5), not better models.

The gate enforces volatility-scaled position sizing, single-name and sector concentration caps, a gross exposure ceiling, a drawdown circuit breaker, an overtrading guard, a minimum-confidence floor and mandatory stops. Caps compose by taking the **tightest**, never by averaging. Every ruling is written to an append-only log with its reason.

It resizes wherever policy allows rather than rejecting outright — a gate that only ever says "no" gets switched off, and a policy that gets switched off is not a policy.

### 4. Every member is scored against reality

This is the part almost no agent project has.

Members state **probabilities**, not adjectives. Once the horizon passes, `quorum/calibration.py` scores each one with a Brier score — the mean squared error of the stated probability.

The intuition the formula hides: always guessing 50/50 scores **0.25**. That is the know-nothing baseline. Below it, a member is adding information. Above it, the member is not merely uninformed but *confidently wrong*, and you would do better inverting it. Confident errors are punished quadratically, which is exactly right.

The dashboard also reports **pairwise agreement** between members. That is the empirical test of whether the evidence partition actually worked: two model seats agreeing most of the time would prove their errors are correlated and the committee is a chorus after all.

---

## Architecture

```
                    ┌──────────────── evidence.py ────────────────┐
                    │  ONE asset, FOUR disjoint evidence packets  │
                    └──┬───────────┬───────────┬──────────────┬───┘
                       │           │           │              │
                  technical    narrative    regime          quant
                  (anonymised) (scrubbed)  (no name)     (RULEBOOK,
                     LLM          LLM         LLM         not a model)
                       │           │           │              │
                       └───────────┴─────┬─────┴──────────────┘
                                         │  4 votes, each with a probability
                                         ▼
                              debate.py — 2 fixed rounds
                              bull / bear must attack specific
                              claims; the dissent survives
                                         │
                                         ▼
                              pm.py — chair synthesises
                              model picks DIRECTION + CONVICTION
                              python computes LEVELS + SIZE
                                         │
                                         ▼
                    ┌────────── risk.py — deterministic ──────────┐
                    │  approve  ·  resize  ·  reject   + reasons  │
                    └────────────────────┬───────────────────────┘
                                         ▼
                          Alpaca paper account (alpaca.py)
                                         │
                                         ▼
                    audit.py — append-only JSONL decision log
                                         │
                                         ▼
                    calibration.py — Brier scores per member
```

---

## Quick start

```bash
git clone <your-repo> && cd quorum
pip install -r requirements.txt
cp .env.example .env        # add Alpaca PAPER keys + Anthropic key
```

```bash
python tests/test_risk.py                  # 10 tests, no keys needed
python -m quorum.run gate-demo NVDA        # watch the gate refuse a reckless order
python -m quorum.run scan --dry-run        # full cycle, no orders placed
python -m quorum.run scan                  # for real, on paper
python -m quorum.run resolve               # attach outcomes once the horizon passes
python -m quorum.run report                # calibration leaderboard
streamlit run app.py                       # dashboard
```

### Two practical notes

**Free-plan data.** Alpaca's free tier serves IEX in real time; SIP requires the `end` parameter to be at least 15 minutes old. `alpaca.py` stops its window 16 minutes back so queries return data instead of silently returning nothing.

**Time zones.** US market hours are 19:00–01:30 IST. Alpaca's crypto data is free and trades 24/7, so `BTC/USD` and `ETH/USD` are in the universe deliberately — they give a live, working demo at any hour. Crypto also exercises fractional sizing, and Alpaca does not accept bracket orders on crypto, so the stop is recorded in the audit log and managed by the runner instead.

---

## Alpaca MCP server

The same paper account is wired to [Alpaca's MCP server](https://github.com/alpacahq/alpaca-mcp-server) so the committee's positions and the audit log can be interrogated conversationally:

```json
{
  "mcpServers": {
    "alpaca": {
      "command": "uvx",
      "args": ["alpaca-mcp-server"],
      "env": {
        "ALPACA_API_KEY": "...",
        "ALPACA_SECRET_KEY": "...",
        "ALPACA_PAPER_TRADE": "true"
      }
    }
  }
}
```

---

## What this project does not claim

Honesty is cheaper than a backtest that flatters itself.

- **This is not a proven edge.** A week of paper trading over a handful of symbols cannot establish one, and any win rate shown in the dashboard is drawn from a sample far too small to mean anything.
- **The evidence partition reduces correlated error; it does not eliminate it.** All three model seats still share a base model and a training distribution. The pairwise-agreement table is there to show honestly how much correlation survives.
- **The quant seat is a simple rulebook, not a researched strategy.** It exists to be *differently* wrong, not to be right.
- **Brier scores need volume.** A dozen resolved decisions is an anecdote. The machinery is built so that with a few hundred, it would become evidence.

The claim this project does make is narrower and defensible: a committee of language models is only worth more than one language model if its members are structurally prevented from sharing evidence — and no committee should be allowed to size its own positions.

---

## Layout

```
quorum/
  alpaca.py        raw REST client — no SDK, so there is less to break
  llm.py           Anthropic Messages client with defensive JSON extraction
  evidence.py      the evidence partition (the core idea)
  members/
    llm_analysts.py    three model seats, three disjoint packets
    quant.py           the seat that is not a model
  debate.py        bounded cross-examination + independence-weighted consensus
  pm.py            chair: model picks direction, python computes size
  risk.py          deterministic gate — approve / resize / reject
  audit.py         append-only decision log
  calibration.py   Brier scores and pairwise agreement
  run.py           CLI orchestrator
app.py             Streamlit dashboard
tests/test_risk.py 10 tests on the one component that must never be wrong
```

Dependencies: `requests`, `pandas`, `numpy`, `PyYAML`, and `streamlit` for the dashboard. Nothing else. Every dependency is a way for a live demo to fail.
