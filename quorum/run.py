"""Orchestrator and CLI.

    python -m quorum.run scan                       # full committee cycle over the watchlist
    python -m quorum.run scan --dry-run              # decide and log, but place no orders
    python -m quorum.run gate-demo NVDA              # show the risk officer refusing a reckless order
    python -m quorum.run backfill --dates 20         # committee cycles against past dates, for calibration
    python -m quorum.run resolve                     # attach realised outcomes to past decisions
    python -m quorum.run report                      # calibration leaderboard in the terminal

Order of operations per symbol:
    evidence (partitioned) -> 4 independent votes -> 2 rounds of debate
    -> PM proposal -> risk gate -> execution -> audit log
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from .alpaca import Alpaca, AlpacaError
from .audit import AuditLog, new_run_id
from .calibration import committee_summary, pairwise_agreement, score_members
from .debate import consensus_strength, run_debate
from .evidence import bars_to_frame, build_evidence
from .llm import LLM
from .members import PROFILES, llm_vote, quant_vote
from .pm import synthesise
from .risk import RiskOfficer
from .types import DecisionRecord, Proposal, utcnow

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | None = None) -> dict:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with cfg_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Tiny .env reader so the project has no python-dotenv dependency."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# --------------------------------------------------------------------- scan

def scan(cfg: dict, dry_run: bool = False, symbols: list[str] | None = None) -> None:
    api = Alpaca(paper=cfg.get("paper", True), feed=cfg.get("feed", "iex"))
    llm = LLM(model=cfg.get("model", "claude-sonnet-4-5"))
    log = AuditLog(cfg.get("log_path", "data/decisions.jsonl"))
    officer = RiskOfficer(cfg.get("policy"), cfg.get("sectors", {}))

    account = api.account()
    positions = api.positions()
    equity = float(account["equity"])
    entries_today = log.entries_today()
    run_id = new_run_id()

    print(f"\nRun {run_id}   equity ${equity:,.0f}   "
          f"{len(positions)} open   {entries_today} entries today")
    print("=" * 72)

    for symbol in (symbols or cfg["universe"]):
        print(f"\n{symbol}")
        try:
            ev = build_evidence(api, symbol, cfg.get("benchmark", "SPY"),
                                cfg.get("lookback_days", 200))
        except AlpacaError as exc:
            print(f"  ! data unavailable: {exc}")
            continue

        # --- four independent votes on four disjoint evidence packets
        votes = [
            llm_vote(llm, name, symbol, ev.packet_for(PROFILES[name]["slice"]),
                     cfg.get("horizon_days", 5))
            for name in PROFILES
        ]
        votes.append(quant_vote(symbol, ev.technical))

        for v in votes:
            tag = "LLM" if v.is_llm else "RULE"
            print(f"  {v.member:<10} [{tag:<4}] {v.stance:<5} {v.confidence:.0%}  {v.thesis[:70]}")

        cons = consensus_strength(votes)
        print(f"  consensus: score {cons['score']:+.2f}  net_conviction {cons['net_conviction']:.0%}"
              f"  independence-weighted {cons['independence_weighted']:+.2f}")

        debate = run_debate(llm, symbol, votes) if cfg.get("debate", True) else []

        proposal, chair_note = synthesise(
            llm, symbol, votes, debate, cons, ev.private, equity,
            cfg.get("horizon_days", 5),
            cfg.get("atr_stop_multiple", 2.0),
            cfg.get("reward_risk", 2.0),
        )

        gate = None
        order_id = None
        if proposal is None:
            if chair_note is not None:
                print(f"  PM: FLAT ({chair_note.confidence:.0%} confidence) — {chair_note.rationale}")
                if chair_note.dissent:
                    print(f"      dissent: {chair_note.dissent}")
            else:
                print("  PM: no trade (no verdict reached — bad price data or a failed chair call)")
        else:
            print(f"  PM: {proposal.side.upper()} {proposal.qty:.4f} @ ~{proposal.entry_ref_price:.2f} "
                  f"stop {proposal.stop_price}  conf {proposal.confidence:.0%}")
            gate = officer.review(proposal, account, positions, entries_today)
            verdict = {"approve": "APPROVED", "resize": "RESIZED", "reject": "REJECTED"}[gate.action]
            print(f"  GATE: {verdict} -> {gate.final_qty:.4f}")
            for r in gate.reasons:
                print(f"        - {r}")

            if gate.approved and not dry_run:
                try:
                    order = api.submit_order(
                        symbol, gate.final_qty, proposal.side,
                        proposal.stop_price, proposal.take_profit,
                    )
                    order_id = order.get("id")
                    entries_today += 1
                    print(f"  ORDER: submitted {order_id}")
                except AlpacaError as exc:
                    print(f"  ORDER: rejected by broker: {exc}")
            elif gate.approved:
                print("  ORDER: suppressed (--dry-run)")

        log.append(DecisionRecord(
            ts=utcnow(), symbol=symbol, votes=votes, debate=debate,
            proposal=proposal, gate=gate, executed_order_id=order_id, run_id=run_id,
            chair_note=chair_note,
        ))

    print("\n" + "=" * 72)
    print(f"LLM usage: {llm.calls} calls, "
          f"{llm.input_tokens:,} in / {llm.output_tokens:,} out tokens")


# -------------------------------------------------------------------- backfill

def _select_backfill_dates(
    bench_bars, n_dates: int, step: int, horizon_days: int
) -> list[str]:
    """Pick `n_dates` past as-of dates, `step` trading days apart.

    Dates come from the benchmark's own daily bar calendar, so weekends and
    holidays are skipped for free -- there is never a bar for a day the
    market didn't trade, so a date that isn't in this list was never a
    candidate. The most recent `horizon_days + 1` bars are excluded from
    consideration: a backfilled decision is timestamped end-of-day (see
    `backfill()`), so resolve()'s `t >= t0` excludes the as_of bar itself --
    entry is the *next* bar's close, same as a live decision (timestamped
    mid-run, well after its own day's bar). That means scoring one needs one
    bar to reach entry plus `horizon_days` more to reach exit: `horizon_days
    + 1` trading days must have already elapsed after as_of, not just
    `horizon_days`, or picking from right up to today would log decisions
    that can't be resolved yet.
    """
    if bench_bars.empty:
        return []
    all_dates = bench_bars["t"].dt.strftime("%Y-%m-%d").tolist()
    eligible = all_dates[: -(horizon_days + 1)]
    if not eligible:
        return []
    # Most-recent-eligible first, every `step`-th, then back to chronological
    # order so the run (and the audit log) reads oldest -> newest.
    picked = list(reversed(eligible))[::step][:n_dates]
    return sorted(picked)


def backfill(
    cfg: dict, dates: int, step: int = 3, symbols: list[str] | None = None
) -> None:
    """Run the full committee against PAST dates, so calibration has more than
    a handful of live decisions to score.

    Every step here has to be lookahead-safe: build_evidence(as_of=...) caps
    bars, news, and headline age at that date's close (see alpaca.py and
    evidence.py), so a member evaluating date D reasons only over what
    actually existed by D. Nothing is ever submitted to the broker -- this is
    pure evaluation-against-history, not a shadow live run.

    SYNTHETIC ACCOUNT: the risk gate needs an account and open-position list,
    and there is no way to reconstruct what your account actually looked like
    on a past date -- so every backfilled date is gated against the same
    flat $100k / no-positions snapshot below. That means backfilled gate
    rulings (sizing, sector caps, gross exposure) reflect this one
    hypothetical book, not an account that evolved trade by trade the way a
    live run's does. Treat the committee's *direction and confidence* as the
    trustworthy signal from a backfill; treat the *gate's sizing* as
    illustrative only, not as what would actually have happened.
    """
    step = max(1, step)
    api = Alpaca(paper=cfg.get("paper", True), feed=cfg.get("feed", "iex"))
    llm = LLM(model=cfg.get("model", "claude-sonnet-4-5"))
    log = AuditLog(cfg.get("log_path", "data/decisions.jsonl"))
    officer = RiskOfficer(cfg.get("policy"), cfg.get("sectors", {}))
    horizon_days = cfg.get("horizon_days", 5)
    benchmark = cfg.get("benchmark", "SPY")
    universe = symbols or cfg["universe"]

    # Enough calendar days to cover `dates` decisions `step` trading days
    # apart, plus the horizon each one needs to already have elapsed, with
    # slack for weekends/holidays.
    lookback = max(400, (dates * step + horizon_days) * 2 + 30)
    bench_bars = bars_to_frame(api.bars(benchmark, "1Day", lookback))
    as_of_dates = _select_backfill_dates(bench_bars, dates, step, horizon_days)
    if not as_of_dates:
        print("No eligible backfill dates -- not enough benchmark history.")
        return
    if len(as_of_dates) < dates:
        print(f"Only {len(as_of_dates)} eligible dates available (asked for {dates}).")

    account = {"equity": 100_000.0, "last_equity": 100_000.0, "buying_power": 200_000.0}
    positions: list[dict] = []
    # Not a global overtrading count -- there's no real order history to
    # count. This tracks how many symbols *this backfill run itself* would
    # have gated approve/resize on the same as_of date, so the overtrading
    # guard still means something within a single backfill pass.
    entries_by_date: dict[str, int] = {}

    run_id = new_run_id()
    total = len(as_of_dates) * len(universe)
    done = 0
    print(f"\nBackfill {run_id}   {len(as_of_dates)} dates x {len(universe)} symbols "
          f"= {total} decisions   (synthetic flat book, no orders submitted)")
    print(f"Dates: {', '.join(as_of_dates)}")
    print("=" * 72)

    for as_of in as_of_dates:
        for symbol in universe:
            done += 1
            print(f"\n[{done}/{total}] {symbol}  as of {as_of}")
            try:
                ev = build_evidence(
                    api, symbol, benchmark, cfg.get("lookback_days", 200), as_of=as_of
                )
            except AlpacaError as exc:
                print(f"  ! data unavailable: {exc}")
                continue

            votes = [
                llm_vote(llm, name, symbol, ev.packet_for(PROFILES[name]["slice"]), horizon_days)
                for name in PROFILES
            ]
            votes.append(quant_vote(symbol, ev.technical))
            for v in votes:
                tag = "LLM" if v.is_llm else "RULE"
                print(f"  {v.member:<10} [{tag:<4}] {v.stance:<5} {v.confidence:.0%}  {v.thesis[:70]}")

            cons = consensus_strength(votes)
            debate = run_debate(llm, symbol, votes) if cfg.get("debate", True) else []

            proposal, chair_note = synthesise(
                llm, symbol, votes, debate, cons, ev.private, account["equity"],
                horizon_days, cfg.get("atr_stop_multiple", 2.0), cfg.get("reward_risk", 2.0),
            )

            gate = None
            if proposal is None:
                if chair_note is not None:
                    print(f"  PM: FLAT ({chair_note.confidence:.0%} confidence) — {chair_note.rationale}")
                else:
                    print("  PM: no trade (no verdict reached — bad price data or a failed chair call)")
            else:
                print(f"  PM: {proposal.side.upper()} {proposal.qty:.4f} @ ~{proposal.entry_ref_price:.2f} "
                      f"stop {proposal.stop_price}  conf {proposal.confidence:.0%}")
                entries_today = entries_by_date.get(as_of, 0)
                gate = officer.review(proposal, account, positions, entries_today)
                verdict = {"approve": "APPROVED", "resize": "RESIZED", "reject": "REJECTED"}[gate.action]
                print(f"  GATE: {verdict} -> {gate.final_qty:.4f}  (synthetic flat book, not a real account)")
                if gate.approved:
                    entries_by_date[as_of] = entries_today + 1
                print("  ORDER: not submitted (backfill never places orders)")

            # End-of-day on as_of, deliberately NOT that date's own bar
            # timestamp: resolve()'s `t >= t0` must exclude the as_of bar so
            # entry is the NEXT bar's close -- the same convention a live
            # decision gets for free by being timestamped mid-run, well after
            # its own day's bar. One entry-price convention across both
            # populations is what makes them comparable on one leaderboard.
            ts = f"{as_of}T23:59:59Z"

            log.append(DecisionRecord(
                ts=ts, symbol=symbol, votes=votes, debate=debate,
                proposal=proposal, gate=gate, executed_order_id=None, run_id=run_id,
                chair_note=chair_note,
            ))

            print(f"  cost so far: {llm.calls} calls, {llm.input_tokens:,} in / "
                  f"{llm.output_tokens:,} out tok  (~${llm.estimated_cost_usd:.3f})")

    print("\n" + "=" * 72)
    print(f"Backfill done: {done} decisions logged across {len(as_of_dates)} dates.")
    print(f"LLM usage: {llm.calls} calls, {llm.input_tokens:,} in / {llm.output_tokens:,} out tokens "
          f"(~${llm.estimated_cost_usd:.3f})")
    print("Run `python -m quorum.run resolve` to score these against realised outcomes.")


# ----------------------------------------------------------------- gate demo

def gate_demo(cfg: dict, symbol: str) -> None:
    """The 30-second moment for the demo video.

    A deliberately reckless proposal -- 60% of the account into one name, no
    stop -- is put to the risk officer, which refuses it in writing. Then the
    same trade with a stop is resized to what policy actually permits.
    """
    api = Alpaca(paper=cfg.get("paper", True), feed=cfg.get("feed", "iex"))
    officer = RiskOfficer(cfg.get("policy"), cfg.get("sectors", {}))
    account, positions = api.account(), api.positions()
    equity = float(account["equity"])
    price = api.latest_price(symbol)

    reckless = Proposal(
        symbol=symbol, side="buy", qty=(equity * 0.60) / price,
        notional=equity * 0.60, entry_ref_price=price, stop_price=None,
        take_profit=None, horizon_days=5, confidence=0.93,
        rationale="Extremely high conviction. Going big.",
    )
    print(f"\nPROPOSAL 1 — 60% of the account into {symbol}, no stop, 93% 'conviction'")
    _show(officer.review(reckless, account, positions))

    with_stop = Proposal(**{**reckless.__dict__, "stop_price": round(price * 0.94, 2)})
    print(f"\nPROPOSAL 2 — same trade, now with a stop 6% away")
    _show(officer.review(with_stop, account, positions))
    print("\nThe gate never argues and never negotiates. It sizes, or it refuses.\n")


def _show(d) -> None:
    print(f"  -> {d.action.upper()}   {d.original_qty:.4f} requested, {d.final_qty:.4f} permitted")
    for r in d.reasons:
        print(f"     - {r}")
    if d.breached_rules:
        print(f"     rules touched: {', '.join(d.breached_rules)}")


# ------------------------------------------------------------------- resolve

def resolve(cfg: dict) -> None:
    """Attach what actually happened, so calibration has something to score."""
    api = Alpaca(paper=cfg.get("paper", True), feed=cfg.get("feed", "iex"))
    log = AuditLog(cfg.get("log_path", "data/decisions.jsonl"))
    horizon = cfg.get("horizon_days", 5)
    records = log.read()

    import pandas as pd

    updates = {}
    price_cache: dict[str, pd.DataFrame] = {}
    for rec in records:
        if rec.get("outcome"):
            continue
        symbol, ts = rec["symbol"], rec["ts"]
        if symbol not in price_cache:
            price_cache[symbol] = bars_to_frame(api.bars(symbol, "1Day", 400))
        df = price_cache[symbol]
        if df.empty:
            continue
        t0 = pd.to_datetime(ts, utc=True, format="mixed")
        after = df[df["t"] >= t0]
        if len(after) <= horizon:
            continue          # not yet resolvable; leave it open
        entry = float(after["c"].iloc[0])
        exit_ = float(after["c"].iloc[horizon])
        updates[(rec.get("run_id", ""), symbol)] = {
            "entry_close": entry,
            "exit_close": exit_,
            "return_pct": round((exit_ / entry - 1) * 100, 3),
            "horizon_days": horizon,
        }
    n = log.resolve(updates)
    print(f"Resolved {n} decisions.")


# -------------------------------------------------------------------- report

def report(cfg: dict) -> None:
    log = AuditLog(cfg.get("log_path", "data/decisions.jsonl"))
    records = log.read()
    if not records:
        print("No decisions logged yet. Run `python -m quorum.run scan` first.")
        return

    print("\nCOMMITTEE SUMMARY")
    for k, v in committee_summary(records).items():
        print(f"  {k:<20} {v}")

    print("\nCALIBRATION LEADERBOARD  (Brier below 0.25 = better than a coin flip)")
    print(f"  {'member':<12}{'type':<7}{'n':>4}{'brier':>9}{'skill':>9}{'hit':>7}{'overconf':>10}")
    for r in score_members(records):
        if r["brier"] is None:
            print(f"  {r['member']:<12}{'LLM' if r['is_llm'] else 'RULE':<7}"
                  f"{r['resolved_votes']:>4}{'  (unresolved)':>35}")
            continue
        print(f"  {r['member']:<12}{'LLM' if r['is_llm'] else 'RULE':<7}"
              f"{r['resolved_votes']:>4}{r['brier']:>9.3f}{r['skill_vs_coinflip']:>+9.3f}"
              f"{r['hit_rate']:>7.0%}{r['overconfidence']:>+10.3f}")

    print("\nPAIRWISE AGREEMENT  (high agreement between two LLM seats = correlated error)")
    for r in pairwise_agreement(records):
        if r["agreement_rate"] is not None:
            print(f"  {r['pair']:<26} {r['co_votes']:>3} co-votes  {r['agreement_rate']:>6.0%}")


# ----------------------------------------------------------------------- cli

def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(prog="quorum")
    p.add_argument("command", choices=["scan", "gate-demo", "backfill", "resolve", "report"])
    p.add_argument("symbol", nargs="?", default=None)
    p.add_argument("--dry-run", action="store_true", help="decide and log, place no orders")
    p.add_argument("--config", default=None)
    p.add_argument("--dates", type=int, default=None,
                    help="backfill: how many past decision dates")
    p.add_argument("--step", type=int, default=3,
                    help="backfill: trading days between dates (default 3)")
    p.add_argument("--symbols", default=None,
                    help="backfill: comma-separated, default is the config universe")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    try:
        if args.command == "scan":
            scan(cfg, args.dry_run, [args.symbol] if args.symbol else None)
        elif args.command == "gate-demo":
            gate_demo(cfg, args.symbol or cfg["universe"][0])
        elif args.command == "backfill":
            if not args.dates:
                print("ERROR: backfill requires --dates N\n", file=sys.stderr)
                return 1
            symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else None
            backfill(cfg, args.dates, args.step, symbols)
        elif args.command == "resolve":
            resolve(cfg)
        else:
            report(cfg)
    except (AlpacaError, RuntimeError) as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
