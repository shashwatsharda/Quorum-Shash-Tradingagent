"""Tests for the Alpaca client's order-building logic.

No network calls: _post is monkeypatched to capture the request body instead
of sending it, so these run offline with fake keys.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests                              # noqa: E402

import quorum.alpaca as alpaca_module        # noqa: E402
from quorum.alpaca import Alpaca             # noqa: E402


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


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


def test_get_retries_transient_read_timeout_then_succeeds():
    api = Alpaca(key="test-key", secret="test-secret", paper=True)
    calls = {"n": 0}

    def flaky_get(url, params=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.exceptions.ReadTimeout("simulated transient timeout")
        return _FakeResponse(200, {"ok": True})

    api._s.get = flaky_get
    sleeps = []
    orig_sleep = alpaca_module.time.sleep
    alpaca_module.time.sleep = lambda s: sleeps.append(s)   # skip the real backoff delay
    try:
        result = api._get(api.trading_base, "/v2/account")
    finally:
        alpaca_module.time.sleep = orig_sleep

    assert result == {"ok": True}, result
    assert calls["n"] == 2, calls      # failed once, succeeded on retry
    assert len(sleeps) == 1, sleeps    # backed off exactly once


def test_get_gives_up_after_repeated_connection_errors():
    api = Alpaca(key="test-key", secret="test-secret", paper=True)
    calls = {"n": 0}

    def always_flaky_get(url, params=None, timeout=None):
        calls["n"] += 1
        raise requests.exceptions.ConnectionError("simulated dropped connection")

    api._s.get = always_flaky_get
    orig_sleep = alpaca_module.time.sleep
    alpaca_module.time.sleep = lambda s: None
    try:
        try:
            api._get(api.trading_base, "/v2/account")
            raised = False
        except alpaca_module.AlpacaError:
            raised = True
    finally:
        alpaca_module.time.sleep = orig_sleep

    assert raised, "expected AlpacaError once retries are exhausted"
    assert calls["n"] == 3, calls      # exactly 3 attempts, no more, no fewer


def test_post_retries_transient_connection_error_then_succeeds():
    api = Alpaca(key="test-key", secret="test-secret", paper=True)
    calls = {"n": 0}

    def flaky_post(url, json=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.exceptions.ChunkedEncodingError("simulated truncated read")
        return _FakeResponse(200, {"id": "order-1"})

    api._s.post = flaky_post
    orig_sleep = alpaca_module.time.sleep
    alpaca_module.time.sleep = lambda s: None
    try:
        result = api._post(api.trading_base, "/v2/orders", {"symbol": "NVDA"})
    finally:
        alpaca_module.time.sleep = orig_sleep

    assert result == {"id": "order-1"}, result
    assert calls["n"] == 2, calls


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
