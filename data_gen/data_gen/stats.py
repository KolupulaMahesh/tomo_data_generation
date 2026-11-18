"""Failure and recovery statistics over a generated folder: which failures happen, which recoveries work, how often.

    python3 -m data_gen.stats out/test
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


def summarize(folder):
    folder = Path(folder)
    fail = collections.Counter()
    strat = collections.defaultdict(lambda: [0, 0])          # strategy -> [tries, successes]
    cats = collections.Counter()
    ok = collections.Counter()
    attempts = collections.Counter()
    terminal = collections.Counter()
    rows = []
    for ev in sorted(folder.glob("ep_*/events.json")):
        d = json.loads(ev.read_text())
        v = json.loads((ev.parent / "validation.json").read_text())
        m = d["meta"]
        cats[m["category"]] += 1
        ok[m["category"]] += v["success"]
        for e in d["events"]:
            if e["kind"] == "failure_detected" and not e.get("reassessed"):
                fail[e["failure_type"]] += 1
            elif e["kind"] == "recovery_verified":
                strat[e["strategy"]][0] += 1
                strat[e["strategy"]][1] += bool(e["success"])
                if e["success"]:
                    attempts[e["attempt"]] += 1
            elif e["kind"] == "terminal_failure":
                terminal[e["failure_type"]] += 1
        rows.append({"episode": ev.parent.name, "category": m["category"], "success": v["success"],
                     "failed_checks": v["failed_checks"]})
    out = {"totals": {"episodes": sum(cats.values()), "successful": sum(ok.values()),
                      "by_category": {c: {"episodes": n, "successful": ok[c]} for c, n in cats.items()}},
           "failures_detected": dict(fail.most_common()),
           "recovery_strategies": {s: {"tries": t, "succeeded": k, "rate": round(k / t, 3) if t else None}
                                   for s, (t, k) in sorted(strat.items())},
           "recovered_on_attempt": dict(sorted(attempts.items())),
           "terminal_failures": dict(terminal),
           "episodes": rows}
    (folder / "stats.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    print(json.dumps({k: v for k, v in summarize(sys.argv[1]).items() if k != "episodes"}, indent=1))
