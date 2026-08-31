"""Tests for the Alpaca client's order-building logic.

No network calls: _post is monkeypatched to capture the request body instead
of sending it, so these run offline with fake keys.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quorum.alpaca import Alpaca             # noqa: E402


def alpaca_with_captured_post():
    """An Alpaca instance whose _post records its args instead of calling out."""
    api = Alpaca(key="test-key", secret="test-secret", paper=True)
    calls = []

    def fake_post(base, path, body):
        calls.append({"base": base, "path": path, "body": body})
        return {"id": "fake-order-id"}

    api._post = fake_post
    return api, calls


def test_stop_price_forces_gtc_for_equity():
    api, calls = alpaca_with_captured_post()
    api.submit_order("NVDA", 19, "buy", stop_price=215.26)
    body = calls[0]["body"]
    assert body["time_in_force"] == "gtc", body
    assert body["order_class"] == "bracket", body
    assert body["stop_loss"] == {"stop_price": "215.26"}, body


def test_explicit_day_is_overridden_when_stop_price_set():
    # The caller-supplied "day" must not survive once a stop is attached --
    # that's exactly the silent-expiry bug this behavior guards against.
    api, calls = alpaca_with_captured_post()
    api.submit_order("NVDA", 19, "buy", stop_price=215.26, time_in_force="day")
    body = calls[0]["body"]
    assert body["time_in_force"] == "gtc", body


def test_no_stop_price_keeps_day_default_for_equity():
    api, calls = alpaca_with_captured_post()
    api.submit_order("NVDA", 19, "buy")
    body = calls[0]["body"]
    assert body["time_in_force"] == "day", body
    assert "order_class" not in body, body


def test_crypto_stays_gtc_without_bracket():
    # Crypto never gets order_class="bracket" (Alpaca doesn't support it there),
    # but time_in_force is still "gtc" -- unrelated to the stop_price fix.
    api, calls = alpaca_with_captured_post()
    api.submit_order("BTC/USD", 0.1, "buy", stop_price=57_000)
    body = calls[0]["body"]
    assert body["time_in_force"] == "gtc", body
    assert "order_class" not in body, body


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
