# Task bundle hashes: which instrument a published row was measured on

A **task bundle** is the hash of everything a task brings: `prompt.md`, `task.json`,
`judge.py`, and every file under `workspace/`. The hash of all eighteen bundles goes into
each run's manifest as `reproducibility.task_bundle`, and every published record carries
its twelve-character short form as `task_bundle_sha256`.

Two rows carrying the same hash sat the same exam and may be compared. Two rows carrying
different hashes did not, and the publish surface test fails when a table presents more
than one without an entry here recording why.

## `d41a23888d49` (superseded, published 2026-11-01 13:54 and earlier)

Measured before 2026-11-01 19:06.

Rows: LFM3 1.7B (llama.cpp, q8kv, thinking unstated), Ornith 1.5 9B (llama.cpp,
configuration unrecorded), Ornith 1.5 9B (llama.cpp, q8kv, thinking unstated).

Frozen at `tasks/FROZEN.sha256`, then changed on 2026-11-01 by four repairs, none of which
was in service when these rows were measured:

| File | Changed | Why |
|---|---|---|
| `tasks/03-test-fix/judge.py` | 19:06 | quality-gate and trace grading work |
| `tasks/04-rename-safe/judge.py` | 19:06 | same |
| `tasks/05-memory-nodestructive/judge.py` | 19:06 | same |
| `tasks/07-error-recovery/workspace/raw.csv` | 20:12 | the fixture shipped bare integers, so the judge's own build wrote the correct answer and a seeded run scored full marks: the grader proof showed 7/7 on the fixture and it was exempted (`grader-proof-2026-11-01.txt`). Repaired, and the re-run scores 0/3 on the same fixture (`grader-proof-2026-11-01-t441ae086.txt`). |
| `tasks/07-error-recovery/prompt.md` | 20:15 | the preamble for trace grading, so the judge can read the rejected call |
| `tasks/10-multifile-refactor/workspace/app/pricing.py` | 20:12 | same contamination: `pricing.py` already carried the `Decimal ROUND_HALF_UP` fix and the fixture scored 1.00. Repaired; the re-run scores 0/4. |

Read as: the two contaminated fixtures were repaired, and three judges gained their
quality-gate and trace-grading paths. A row measured before those repairs was scored by a
judge that could hand two tasks away for free.

## `9fe550deb1c0` (current)

Measured from 2026-11-01 20:18 onward, against `tasks/FROZEN.sha256` digest
`486c8c94121ba5894ac47be716d9c7ed634d3492a86b2d414968693034b34434` (18 tasks, 105 files,
frozen 20:15:30).

Rows: Beam (default, thinking unstated), Ornith 1.5 122B (free, default, thinking
unstated).

Grader proof on this instrument: every task scores 0 on an empty workspace, 0 on its own
seeded fixture (`08` and `16` have no meaningful seed), full marks on the reference
solution, and a near-miss mutation fails every judge. See
`/mnt/agent_shared/hbench-robustness/grader-proof-2026-11-01-t441ae086.txt`.

## What this does not solve

The two rows measured before the repairs have no re-run on the current bundle. Comparing
them with a current-bundle row on the same table is a comparison of two instruments, and
the two tasks that moved were repaired for shipping the answer, so the old rows are, if
anything, generous to the models measured on them. Re-running either row on the current
bundle is the way to make the comparison honest; until then, read them as separate rows
that happen to share a table.
