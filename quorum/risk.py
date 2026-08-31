"""The risk officer: a committee seat with a veto and no language model.

Why this exists
---------------
Long-Term Capital Management had two Nobel laureates working on signal and no
binding constraint on leverage. Knight Capital lost roughly $440m in 45 minutes
in 2012 because a bad deployment let an algorithm trade unsupervised. Neither
failure was a failure of prediction. Both were failures of permission.

The asymmetry is the whole argument: improvements in signal quality are
marginal and diminishing, while failures of constraint are unbounded. That is
why the industry's answer to those blow-ups was not better models but mandatory
pre-trade risk controls (SEC Rule 15c3-5).

So this module is deliberately dumb, deliberately deterministic, and cannot be
talked out of anything. The portfolio manager proposes. This decides. Every
ruling is written down with its reason, whether or not anyone is watching.

Design note: it resizes rather than rejects wherever a rule allows it. Real
desks do the same, and for a good reason -- a gate that only ever says "no"
gets switched off, and a policy that gets switched off is not a policy.
"""

from __future__ import annotations

from typing import Any

from .types import GateDecision, Proposal


DEFAULT_POLICY: dict[str, Any] = {
    "risk_per_trade_pct": 0.75,      # % of equity risked between entry and stop
    # Note on the interaction between the next two lines: caps compose by taking
    # the tightest, so if the single-name cap is set too tight it silently
    # overrides risk-based sizing and every trade gets the same size regardless
    # of its volatility -- which defeats the point of measuring volatility at
    # all. At 0.75% risk and a 10% name cap, the risk rule binds whenever the
    # stop is more than 7.5% away, which is where you actually want it to bind.
    "max_position_pct": 10.0,        # % of equity in any single name
    "max_sector_pct": 20.0,          # % of equity in any one sector
    "max_gross_exposure_pct": 60.0,  # total invested as % of equity
    "max_drawdown_pct": 8.0,         # halt new entries beyond this
    "max_new_entries_per_day": 3,
    "min_confidence": 0.58,
    "require_stop_loss": True,
    "max_stop_distance_pct": 12.0,
    "min_order_notional": 10.0,
}


class RiskOfficer:
    def __init__(
        self,
        policy: dict[str, Any] | None = None,
        sector_map: dict[str, str] | None = None,
    ) -> None:
        self.policy = {**DEFAULT_POLICY, **(policy or {})}
        self.sector_map = sector_map or {}

    # ------------------------------------------------------------------ public

    def review(
        self,
        proposal: Proposal,
        account: dict[str, Any],
        positions: list[dict[str, Any]],
        entries_today: int = 0,
        peak_equity: float | None = None,
    ) -> GateDecision:
        p = self.policy
        reasons: list[str] = []
        breached: list[str] = []
        equity = float(account.get("equity", 0) or 0)
        original_qty = proposal.qty

        if equity <= 0:
            return GateDecision("reject", original_qty, 0.0,
                                ["Account equity is zero or unreadable."], ["equity"])

        # ---------------------------------------------------- hard blocks first
        # These are not negotiable and are checked before any sizing maths,
        # because a trade that must not happen should never get as far as
        # having a size computed for it.

        dd = self._drawdown_pct(account, peak_equity)
        if dd > p["max_drawdown_pct"]:
            breached.append("max_drawdown_pct")
            return GateDecision(
                "reject", original_qty, 0.0,
                [f"Portfolio drawdown {dd:.1f}% exceeds the {p['max_drawdown_pct']}% "
                 f"circuit breaker. No new entries until recovery."],
                breached,
            )

        if entries_today >= p["max_new_entries_per_day"]:
            breached.append("max_new_entries_per_day")
            return GateDecision(
                "reject", original_qty, 0.0,
                [f"Already opened {entries_today} positions today; daily cap is "
                 f"{p['max_new_entries_per_day']}. Overtrading guard."],
                breached,
            )

        if proposal.confidence < p["min_confidence"]:
            breached.append("min_confidence")
            return GateDecision(
                "reject", original_qty, 0.0,
                [f"Committee confidence {proposal.confidence:.0%} is below the "
                 f"{p['min_confidence']:.0%} threshold. Marginal edge, real costs."],
                breached,
            )

        if p["require_stop_loss"] and not proposal.stop_price:
            breached.append("require_stop_loss")
            return GateDecision(
                "reject", original_qty, 0.0,
                ["No stop loss attached. Every position must have a defined "
                 "maximum loss before it is opened, not after."],
                breached,
            )

        stop_dist_pct = self._stop_distance_pct(proposal)
        if proposal.stop_price and stop_dist_pct > p["max_stop_distance_pct"]:
            breached.append("max_stop_distance_pct")
            return GateDecision(
                "reject", original_qty, 0.0,
                [f"Stop sits {stop_dist_pct:.1f}% away, beyond the "
                 f"{p['max_stop_distance_pct']}% limit. A stop that wide is a "
                 f"hope, not a risk control."],
                breached,
            )

        # -------------------------------------------------------- then sizing
        # Everything below shrinks the order rather than killing it. The final
        # size is the *minimum* permitted by every independent cap -- caps
        # compose by taking the tightest, never by averaging.

        caps: list[tuple[str, float, str]] = []

        # 1. Risk-based size: how many units can we hold such that being stopped
        #    out costs no more than risk_per_trade_pct of equity? This is the
        #    only sizing rule that adapts to the asset's own volatility, which
        #    is why it comes first. A wide stop automatically buys less.
        if proposal.stop_price and proposal.entry_ref_price:
            per_unit_risk = abs(proposal.entry_ref_price - proposal.stop_price)
            if per_unit_risk > 0:
                budget = equity * p["risk_per_trade_pct"] / 100.0
                caps.append(("risk_per_trade_pct", budget / per_unit_risk,
                             f"risking {p['risk_per_trade_pct']}% of equity to the stop"))

        px = proposal.entry_ref_price or 0.0
        if px > 0:
            # 2. Single-name concentration.
            existing = self._exposure(positions, proposal.symbol)
            room = equity * p["max_position_pct"] / 100.0 - existing
            caps.append(("max_position_pct", max(room, 0.0) / px,
                         f"{p['max_position_pct']}% single-name cap"))

            # 3. Sector concentration -- the cap people skip, and the one that
            #    actually bites. Five uncorrelated 5% positions is a portfolio;
            #    five semiconductor names at 5% each is one 25% position wearing
            #    a disguise.
            sector = self.sector_map.get(proposal.symbol)
            if sector:
                sector_now = sum(
                    self._exposure(positions, s)
                    for s, sec in self.sector_map.items() if sec == sector
                )
                room = equity * p["max_sector_pct"] / 100.0 - sector_now
                caps.append(("max_sector_pct", max(room, 0.0) / px,
                             f"{p['max_sector_pct']}% cap on {sector}"))

            # 4. Gross exposure -- keeps dry powder and bounds a correlated hit.
            gross = sum(abs(float(q.get("market_value", 0) or 0)) for q in positions)
            room = equity * p["max_gross_exposure_pct"] / 100.0 - gross
            caps.append(("max_gross_exposure_pct", max(room, 0.0) / px,
                         f"{p['max_gross_exposure_pct']}% gross exposure cap"))

            # 5. Settled buying power -- the broker's opinion, which overrules ours.
            bp = float(account.get("buying_power", 0) or 0)
            caps.append(("buying_power", bp / px, "available buying power"))

        final_qty = original_qty
        for rule, cap, human in caps:
            if cap < final_qty:
                final_qty = max(cap, 0.0)
                breached.append(rule)
                reasons.append(f"Resized down by {human}.")

        final_qty = self._round_qty(final_qty, proposal.symbol)

        if final_qty * px < p["min_order_notional"]:
            return GateDecision(
                "reject", original_qty, 0.0,
                reasons + [f"Permitted size is below the "
                           f"${p['min_order_notional']:.0f} minimum. The limits "
                           f"leave no room for this trade."],
                breached or ["min_order_notional"],
            )

        if final_qty >= original_qty:
            return GateDecision(
                "approve", original_qty, self._round_qty(original_qty, proposal.symbol),
                ["Within all policy limits."], [],
            )

        reasons.insert(0, f"Proposed {original_qty:.4f} units, permitted {final_qty:.4f}.")
        return GateDecision("resize", original_qty, final_qty, reasons, breached)

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _exposure(positions: list[dict[str, Any]], symbol: str) -> float:
        return sum(
            abs(float(q.get("market_value", 0) or 0))
            for q in positions
            if q.get("symbol", "").replace("/", "") == symbol.replace("/", "")
        )

    @staticmethod
    def _stop_distance_pct(proposal: Proposal) -> float:
        if not proposal.stop_price or not proposal.entry_ref_price:
            return 0.0
        return abs(proposal.entry_ref_price - proposal.stop_price) / proposal.entry_ref_price * 100

    @staticmethod
    def _drawdown_pct(account: dict[str, Any], peak_equity: float | None) -> float:
        equity = float(account.get("equity", 0) or 0)
        peak = peak_equity or float(account.get("last_equity", equity) or equity)
        if not peak:
            return 0.0
        return max(0.0, (peak - equity) / peak * 100)

    @staticmethod
    def _round_qty(qty: float, symbol: str) -> float:
        # Crypto is fractionable to many decimals; US equities are safest whole.
        if "/" in symbol:
            return round(qty, 6)
        return float(int(qty))
