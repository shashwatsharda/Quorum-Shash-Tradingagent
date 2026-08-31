"""A small, honest Alpaca REST client.

Deliberately not using alpaca-py. Three reasons:
  1. Fewer dependencies means fewer ways for a demo to die.
  2. Every call is visible here, so you can explain your own stack.
  3. The SDK's typed objects have to be unwrapped for the LLM anyway.

Docs: https://docs.alpaca.markets
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

PAPER_TRADING = "https://paper-api.alpaca.markets"
LIVE_TRADING = "https://api.alpaca.markets"
DATA = "https://data.alpaca.markets"


class AlpacaError(RuntimeError):
    pass


class Alpaca:
    def __init__(
        self,
        key: str | None = None,
        secret: str | None = None,
        paper: bool = True,
        feed: str = "iex",
        timeout: int = 20,
    ) -> None:
        self.key = key or os.environ.get("ALPACA_API_KEY", "")
        self.secret = secret or os.environ.get("ALPACA_SECRET_KEY", "")
        if not self.key or not self.secret:
            raise AlpacaError(
                "Missing ALPACA_API_KEY / ALPACA_SECRET_KEY. Copy .env.example to .env "
                "and fill in your paper-trading keys from https://app.alpaca.markets"
            )
        self.trading_base = PAPER_TRADING if paper else LIVE_TRADING
        self.paper = paper
        self.feed = feed          # 'iex' is what the free plan gets in real time
        self.timeout = timeout
        self._s = requests.Session()
        self._s.headers.update(
            {
                "APCA-API-KEY-ID": self.key,
                "APCA-API-SECRET-KEY": self.secret,
                "accept": "application/json",
            }
        )

    # ---------------------------------------------------------------- plumbing

    def _get(self, base: str, path: str, params: dict | None = None) -> dict:
        for attempt in range(3):
            r = self._s.get(f"{base}{path}", params=params, timeout=self.timeout)
            if r.status_code == 429:          # rate limited: back off, don't hammer
                time.sleep(2 ** attempt)
                continue
            if r.status_code >= 400:
                raise AlpacaError(f"GET {path} -> {r.status_code}: {r.text[:300]}")
            return r.json()
        raise AlpacaError(f"GET {path} rate-limited after 3 attempts")

    def _post(self, base: str, path: str, body: dict) -> dict:
        r = self._s.post(f"{base}{path}", json=body, timeout=self.timeout)
        if r.status_code >= 400:
            raise AlpacaError(f"POST {path} -> {r.status_code}: {r.text[:300]}")
        return r.json()

    # ----------------------------------------------------------------- account

    def account(self) -> dict:
        return self._get(self.trading_base, "/v2/account")

    def positions(self) -> list[dict]:
        return self._get(self.trading_base, "/v2/positions")  # type: ignore[return-value]

    def clock(self) -> dict:
        return self._get(self.trading_base, "/v2/clock")

    def portfolio_history(self, period: str = "1M", timeframe: str = "1D") -> dict:
        return self._get(
            self.trading_base,
            "/v2/account/portfolio/history",
            {"period": period, "timeframe": timeframe},
        )

    def asset(self, symbol: str) -> dict:
        return self._get(self.trading_base, f"/v2/assets/{symbol}")

    # -------------------------------------------------------------------- data

    @staticmethod
    def _is_crypto(symbol: str) -> bool:
        return "/" in symbol

    def bars(self, symbol: str, timeframe: str = "1Day", lookback_days: int = 200) -> list[dict]:
        """Daily/intraday OHLCV.

        Note the free-plan gotcha: for US equities, SIP data requires `end` to be
        at least 15 minutes old. We ask for the IEX feed and stop the window
        16 minutes back, so this works on a free account without silently
        returning an empty list. Crypto has no such restriction.
        """
        end = datetime.now(timezone.utc) - timedelta(minutes=16)
        start = end - timedelta(days=lookback_days)
        params: dict[str, Any] = {
            "symbols": symbol,
            "timeframe": timeframe,
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": 10000,
        }
        if self._is_crypto(symbol):
            payload = self._get(DATA, "/v1beta3/crypto/us/bars", params)
        else:
            params["feed"] = self.feed
            params["adjustment"] = "split"
            payload = self._get(DATA, "/v2/stocks/bars", params)
        return payload.get("bars", {}).get(symbol, []) or []

    def latest_price(self, symbol: str) -> float:
        """Last trade price, with a bar-close fallback so a demo never dies here."""
        try:
            if self._is_crypto(symbol):
                p = self._get(DATA, "/v1beta3/crypto/us/latest/trades", {"symbols": symbol})
                return float(p["trades"][symbol]["p"])
            p = self._get(
                DATA, "/v2/stocks/trades/latest", {"symbols": symbol, "feed": self.feed}
            )
            return float(p["trades"][symbol]["p"])
        except (AlpacaError, KeyError, TypeError):
            bars = self.bars(symbol, "1Day", 10)
            if not bars:
                raise AlpacaError(f"No price available for {symbol}")
            return float(bars[-1]["c"])

    def news(self, symbols: list[str], limit: int = 20, lookback_days: int = 5) -> list[dict]:
        start = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        payload = self._get(
            DATA,
            "/v1beta1/news",
            {
                "symbols": ",".join(symbols),
                "limit": limit,
                "start": start,
                "sort": "desc",
                "include_content": "false",
            },
        )
        return payload.get("news", []) or []

    # ------------------------------------------------------------------ orders

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        stop_price: float | None = None,
        take_profit: float | None = None,
        time_in_force: str = "day",
    ) -> dict:
        """Market entry, bracketed where the venue allows it.

        Crypto on Alpaca does not accept bracket orders, so the stop is
        recorded in the audit log and managed by the runner instead. That
        difference is worth saying out loud in your demo -- it is exactly
        the kind of venue detail that separates a real system from a toy.
        """
        # A bracket's legs inherit the entry's time_in_force. "day" lets them
        # silently expire unfilled at market close, leaving the position with
        # no stop at all -- so any stop-bearing equity order goes "gtc" too.
        is_bracket = bool(stop_price) and not self._is_crypto(symbol)
        body: dict[str, Any] = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": "market",
            "time_in_force": "gtc" if (self._is_crypto(symbol) or is_bracket) else time_in_force,
        }
        if is_bracket:
            body["order_class"] = "bracket"
            body["stop_loss"] = {"stop_price": str(round(stop_price, 2))}
            if take_profit:
                body["take_profit"] = {"limit_price": str(round(take_profit, 2))}
        return self._post(self.trading_base, "/v2/orders", body)
