#!/usr/bin/env python3
"""Deterministically generate the agentic needle-in-haystack fixture.

Run once to produce tasks/18-needle-haystack-agentic/workspace/haystack.log.
Seeded, so the fixture is reproducible and the embedded code is known. The judge
does NOT depend on this seed: it reads the log and extracts the unique code
itself. The one hard rule: no noise line may contain the pattern ERR-XXXX
(ERR- followed by exactly four uppercase hex digits); the embedded code is the
only such token in the file.
"""
import random
from pathlib import Path

NEEDLE = "ERR-7F3A"
N_LINES = 3200
SEED = 20261101

WORDS = ["billing", "auth", "queue", "cache", "router", "worker", "scheduler",
         "gateway", "session", "token", "invoice", "ledger", "index", "replica",
         "shard", "proxy", "worker", "connector", "buffer"]
STATES = ["ok", "retry", "flush", "drain", "commit", "rollback", "handoff"]


def main() -> None:
    rng = random.Random(SEED)
    lines = []
    seq = 0
    needle_line = rng.randint(N_LINES // 3, 2 * N_LINES // 3)
    for i in range(N_LINES):
        seq += 1
        rid = rng.randint(100000, 999999)
        ms = rng.randint(4, 4200)
        lvl = rng.choices(["INFO", "DEBUG", "WARN", "ERROR"], weights=[6, 2, 1, 1])[0]
        svc = rng.choice(WORDS)
        if i == needle_line:
            # The one line that carries the code. The word ERROR is safe: it is
            # not "ERR-" + 4 hex, so the judge regex cannot match it elsewhere.
            line = (f"[2026-10-30T09:{(i*7)%60:02d}:{(i*13)%60:02d}Z] ERROR "
                    f"req-{rid} svc={svc} fatal module code {NEEDLE} halted pipeline")
        elif lvl == "ERROR":
            line = (f"[2026-10-30T09:{(i*7)%60:02d}:{(i*13)%60:02d}Z] ERROR "
                    f"req-{rid} svc={svc} connection reset by peer")
        elif lvl == "WARN":
            line = (f"[2026-10-30T09:{(i*7)%60:02d}:{(i*13)%60:02d}Z] WARN "
                    f"req-{rid} svc={svc} slow response {ms}ms state={rng.choice(STATES)}")
        elif lvl == "DEBUG":
            line = (f"[2026-10-30T09:{(i*7)%60:02d}:{(i*13)%60:02d}Z] DEBUG "
                    f"req-{rid} svc={svc} cache {'hit' if rng.random() < 0.5 else 'miss'}")
        else:
            line = (f"[2026-10-30T09:{(i*7)%60:02d}:{(i*13)%60:02d}Z] INFO "
                    f"req-{rid} svc={svc} processed {rng.randint(1,40)} items in {ms}ms")
        lines.append(line)

    text = "\n".join(lines) + "\n"
    out = Path(__file__).resolve().parent / "workspace" / "haystack.log"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)

    import re
    found = sorted(set(re.findall(r"ERR-[0-9A-F]{4}", text)))
    assert found == [NEEDLE], f"expected exactly one code, got {found}"
    print(f"wrote {out}: {len(text)} bytes, {len(lines)} lines, codes={found}")


if __name__ == "__main__":
    main()
