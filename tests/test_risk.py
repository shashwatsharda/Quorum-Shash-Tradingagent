"""Tests for the one component that must never be wrong.

The LLM members are allowed to be fuzzy. The gate is not. If it ever lets
through something the policy forbids, everything else in the project is
decoration.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quorum.risk import RiskOfficer          # noqa: E402
from quorum.types import Proposal            # noqa: E402

ACCOUNT = {"equity": 100_000, "last_equity": 100_000, "buying_power": 200_000}
SECTORS = {"NVDA": "semis", "AMD": "semis", "JPM": "financials"}


def proposal(**kw):
    base = dict(
        symbol="NVDA", side="buy", qty=100, notional=10_000,
        entry_ref_price=100.0, stop_price=95.0, take_profit=115.0,
        horizon_days=5, confidence=0.7, rationale="test",
    )
    base.update(kw)
    return Proposal(**base)


def test_tightest_cap_wins():
    # 0.75% of 100k = $750 of risk; $5 per unit to the stop => 150 units.
    # But the 10% single-name cap allows only $10,000 / $100 = 100 units.
    # Caps compose by taking the minimum, never by averaging.
    d = RiskOfficer(sector_map=SECTORS).review(proposal(qty=1000), ACCOUNT, [])
    assert d.action == "resize"
    assert d.final_qty == 100, d
    assert "max_position_pct" in d.breached_rules


def test_wide_stop_buys_less():
    # The point of volatility-based sizing: identical conviction, wider stop,
    # smaller position -- so the dollar loss if wrong is the same either way.
    tight = RiskOfficer().review(proposal(qty=1000, stop_price=99.0), ACCOUNT, [])
    wide = RiskOfficer().review(proposal(qty=1000, stop_price=92.0), ACCOUNT, [])
    assert tight.final_qty > wide.final_qty, (tight, wide)
    # And the wide-stop size must be the one set by the risk budget, not the cap.
    assert "risk_per_trade_pct" in wide.breached_rules


def test_missing_stop_is_rejected_outright():
    d = RiskOfficer().review(proposal(stop_price=None), ACCOUNT, [])
    assert d.action == "reject" and "require_stop_loss" in d.breached_rules


def test_drawdown_circuit_breaker_halts_new_entries():
    acct = {"equity": 90_000, "last_equity": 100_000, "buying_power": 180_000}
    d = RiskOfficer().review(proposal(), acct, [], peak_equity=100_000)
    assert d.action == "reject" and "max_drawdown_pct" in d.breached_rules


def test_low_confidence_is_rejected():
    d = RiskOfficer().review(proposal(confidence=0.51), ACCOUNT, [])
    assert d.action == "reject" and "min_confidence" in d.breached_rules


def test_sector_cap_counts_sibling_positions():
    # $19,000 already in AMD. The semis cap is 20% of 100k = $20,000,
    # leaving $1,000 => 10 units of a $100 stock, not the 50 the
    # single-name cap alone would have allowed.
    positions = [{"symbol": "AMD", "market_value": 19_000}]
    d = RiskOfficer(sector_map=SECTORS).review(proposal(qty=1000), ACCOUNT, positions)
    assert d.action == "resize"
    assert d.final_qty == 10, d
    assert "max_sector_pct" in d.breached_rules


def test_overtrading_guard():
    d = RiskOfficer().review(proposal(), ACCOUNT, [], entries_today=3)
    assert d.action == "reject" and "max_new_entries_per_day" in d.breached_rules


def test_clean_proposal_passes_untouched():
    d = RiskOfficer(sector_map=SECTORS).review(proposal(qty=40), ACCOUNT, [])
    assert d.action == "approve" and d.final_qty == 40 and d.approved


def test_no_room_left_is_a_rejection_not_a_dust_order():
    positions = [{"symbol": "NVDA", "market_value": 9_999}]
    d = RiskOfficer(sector_map=SECTORS).review(proposal(qty=1000), ACCOUNT, positions)
    assert d.action == "reject"


def test_crypto_keeps_fractional_size():
    d = RiskOfficer().review(
        proposal(symbol="BTC/USD", qty=10, entry_ref_price=60_000,
                 stop_price=57_000, notional=600_000),
        ACCOUNT, [],
    )
    assert 0 < d.final_qty < 1


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
                passed += 1
            except AssertionError as exc:
                print(f"  FAIL  {name}: {exc}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
