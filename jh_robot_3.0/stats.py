# stats.py
from __future__ import annotations
import os
import json
from dataclasses import dataclass, field
from typing import Dict, Any, DefaultDict
from collections import defaultdict
import time

@dataclass
class BehaviorStats:
    # 分桶统计：seen(0/1) + round_bucket(0..3)
    counts: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))
    bets: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))
    pks: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))
    folds: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))

    def _key(self, seen: bool, rnd: int) -> str:
        # round: 0-4,5-9,10-14,15-20
        b = 0 if rnd < 5 else 1 if rnd < 10 else 2 if rnd < 15 else 3
        return f"s{1 if seen else 0}_r{b}"

    def record(self, seen: bool, rnd: int, action_kind: str, bet_mul: int | None = None):
        k = self._key(seen, rnd)
        self.counts[k] += 1
        if action_kind == "BET":
            self.bets[k] += 1
            if bet_mul is not None:
                self.counts[f"{k}_mul_{bet_mul}"] += 1
        elif action_kind == "PK":
            self.pks[k] += 1
        elif action_kind == "FOLD":
            self.folds[k] += 1

    def snapshot(self) -> Dict[str, Any]:
        out = {}
        for k, n in self.counts.items():
            out[k] = n
        for k in list(self.counts.keys()):
            pass
        return {
            "counts": dict(self.counts),
            "bets": dict(self.bets),
            "pks": dict(self.pks),
            "folds": dict(self.folds),
        }

    def dump_jsonl(self, path: str, extra: Dict[str, Any]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rec = {"ts": int(time.time()), **extra, **self.snapshot()}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
