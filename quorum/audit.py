"""Append-only decision log.

A governance rule that isn't recorded isn't a rule, it's a preference. The log
is JSONL because that format survives crashes gracefully -- a half-written last
line costs you one record, not the file.

Nothing in here ever updates a row in place except `resolve`, which fills in
what actually happened afterwards. That is by design: the log answers "what did
we decide, on what evidence, and were we right", and you cannot answer the
third part at the time you decide.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .types import DecisionRecord

DEFAULT_PATH = Path(os.environ.get("QUORUM_LOG", "data/decisions.jsonl"))


def new_run_id() -> str:
    return f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"


class AuditLog:
    def __init__(self, path: Path | str = DEFAULT_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: DecisionRecord) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record), default=str) + "\n")

    def read(self) -> list[dict[str, Any]]:
        return list(self.iter_records())

    def iter_records(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue      # tolerate one torn line rather than dying

    def resolve(self, updates: dict[tuple[str, str], dict[str, Any]]) -> int:
        """Attach realised outcomes, keyed by (run_id, symbol). Rewrites the file once."""
        records = self.read()
        n = 0
        for rec in records:
            key = (rec.get("run_id", ""), rec.get("symbol", ""))
            if key in updates and not rec.get("outcome"):
                rec["outcome"] = updates[key]
                n += 1
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, default=str) + "\n")
        tmp.replace(self.path)
        return n

    def entries_today(self) -> int:
        today = datetime.now(timezone.utc).date().isoformat()
        return sum(
            1
            for r in self.iter_records()
            if str(r.get("ts", "")).startswith(today)
            and r.get("executed_order_id") is not None
        )
