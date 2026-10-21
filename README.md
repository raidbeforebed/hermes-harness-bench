# hbench

A small, reproducible benchmark for choosing a **default model for the Hermes Agent harness**.

It answers one question: which model actually finishes the work, judged on the files and state it leaves behind rather than on the reply it writes.

---

## Why this exists

Running ad-hoc prompts through an agent and comparing the transcripts gives unstable results. Three things cause the drift:

1. **The reply is not the work.** Models describe success without committing it. Harness-Bench (arXiv [2605.27922](https://arxiv.org/abs/2605.27922)) measured this artifact-commitment failure at 11.1% of all agent failures.
2. **State carries between runs.** An agent that writes memory or notes during one run changes the starting conditions of the next.
3. **Turn count is not constant across models.** Harness-Bench Table 2 found the Hermes harness averages **22.6 turns and 139.7K tokens per task**, against 5.0 to 12.1 turns for other harnesses on identical work. A model that looks competent in a six-turn prompt can fall apart at turn fifteen.

This harness addresses all three.

## What it measures

Twelve tasks, each graded by asserting over the resulting file tree.

| Task | Failure class | What a model must do |
|---|---|---|
| `01-json-extract` | Contract / format | Emit schema-valid JSON with correct arithmetic over a CSV |
| `02-inventory-reconcile` | Evidence / grounding | Reconcile two sources without inventing records |
| `03-test-fix` | Tool / recovery | Run a suite, read the failure, fix the implementation not the test |
| `04-rename-safe` | Artifact commitment | Actually perform renames; describing them fails |
| `05-memory-nodestructive` | Safety / restraint | Answer a question without mutating inputs |
| `06-multiturn-continuity` | State / continuation | Carry three dependent stages forward |
| `07-error-recovery` | Tool / recovery | Diagnose a failed pipeline and satisfy what it actually wants |
| `08-constraint-format` | Instruction following | Satisfy six simultaneous formatting constraints |
| `09-long-context-retrieval` | Long context | Find three codes buried at 12%, 48% and 91% of an 82 KB log |
| `10-multifile-refactor` | Code change | Fix a rounding bug across files without touching the tests |
| `11-correction-handling` | Instruction persistence | Honour a mid-prompt reversal of an earlier instruction |
| `12-long-chain` | Long horizon | 14 dependent steps over 12 files with shuffled column order |

Four of these are adapted from the roadmap in Bent-Solutions/hermes-bench section 15 (instruction following, tool-call error recovery, long-context retrieval, multi-file code change). Two are additions aimed at the long-horizon failure the 22.6-turn figure implies.

## Credits

This project stands on published work. See [CREDITS.md](CREDITS.md) for the full account of what came from where.

- **Harness-Bench** (Yao et al., Peking University, arXiv [2605.27922](https://arxiv.org/abs/2605.27922), code at [Qihoo360/harness-bench](https://github.com/Qihoo360/harness-bench)) supplies the failure taxonomy and every percentage quoted above. No tasks or graders were taken from it.
- **Bent-Solutions/hermes-bench** (MIT) supplies the idea for four tasks, from the category list in its section 15 roadmap. No code or content was taken; their grading model is string matching, this one asserts over the file tree.
- **Nous Research, Hermes Agent** (MIT) is the harness being measured. The CLI flags this runner depends on are theirs.
- **verkyyi/hermesbench** (MIT) was consulted for comparison and shaped the decision about what this project should not try to be.

## Design decisions worth knowing

**Grading inspects state, never prose.** Each `judge.py` receives the working directory after the agent exits and returns pass/fail per check. Nothing is scored by reading the model's answer.

**A non-zero exit scores zero.** A run whose CLI errored is not a result, even if the workspace happens to look correct. Without this guard an argv bug produced a fake 100%.

**Tasks are verified in both directions.** `tests/test_graders.py` asserts every grader scores 0 on an empty workspace and 1.00 on a reference solution. It has caught four real defects: checks that passed on a wiped directory, an arithmetic error in an expected value, a test suite that passed before any fix, and a hash comparison against a stale literal.

**Memory and user config injection are disabled** (`--ignore-rules --ignore-user-config`), so runs are comparable and personal memory does not leak into results.

**Sessions are tagged and removed.** Runs use `--source tool`, which keeps them out of the user's session lists, and each session row is deleted after grading.

**A hard ceiling per task.** `--max-turns` and `--run-budget` bound every run; exceeding them records a failure with exit code `-1`.

**No live network.** Task prompts state that all needed data is in the working directory, so a page changing cannot move a score.

## Install

Requires Python 3.10 or newer, a working `hermes` CLI on `PATH`, and `pytest` for task 03.

```bash
git clone https://github.com/raidbeforebed/hermes-harness-bench.git
cd hermes-harness-bench

python3 -m venv .venv
.venv/bin/pip install pytest
```

The runner is standard library only; no third-party packages are needed for it.

## Use

```bash
# list the tasks
.venv/bin/python bin/hbench.py list

# verify the graders before trusting any result
.venv/bin/python tests/test_graders.py

# run one model over the full suite
.venv/bin/python bin/hbench.py run \
    --model deepseek-v4.1-flash:1017 --provider ollama-cloud --tasks all

# compare several models, three repeats for variance
.venv/bin/python bin/hbench.py run \
    --model <model-a> --model <model-b> \
    --provider <provider> --tasks all --repeat 3

# read a finished run
.venv/bin/python bin/hbench.py report <run_id>
.venv/bin/python bin/hbench.py matrix <run_id>
```

Results land in `runs/<run_id>/`, one directory per model per task, each with a `result.json`.

### Testing local models

Local models need the Ollama context window raised first. On a card under 24 GiB
Ollama serves 4096 tokens by default, which is smaller than an agent system prompt,
so every task fails before the model does any work. See
[docs/local-models.md](docs/local-models.md) for the fix and for how to read local
results honestly.

```bash
python bin/hbench.py run --model <tag> --provider local-ollama --tasks all
```

## Results as published

Measured 2026-10-21, one repeat each, twelve tasks. Raw per-task records are in `results/`.

| Model | pass@1 | Mean score | Mean turns | Mean tokens | Mean wall |
|---|---|---|---|---|---|
| GLM-5.3 Flash | 12/12 | 1.00 | 9.8 | 15,720 | 66s |
| DeepSeek V4.1 Flash | 12/12 | 1.00 | 12.5 | 20,512 | 55s |
| Beam | 12/12 | 1.00 | 14.7 | 23,716 | 77s |
| MiniMax M3.1 Flash | 10/12 | 0.83 | 18.9 | 16,477 | 122s |

**Reading this honestly.** Three models are clean, so pass rate alone does not separate them; cost and turn count do. GLM-5.3 Flash used about a quarter fewer tokens than DeepSeek V4.1 Flash and a third fewer than Beam at the same 12/12.

MiniMax M3.1 Flash lost two tasks to thrashing rather than to capability. On `04-rename-safe`, which needs two file renames and a five-line manifest, it ran 52 turns and 68 tool calls before exhausting a 360-second budget. On `07-error-recovery` it ran 45 turns and 58 tool calls before exhausting 480 seconds. Its ten successes were clean, including a 26-turn multi-file refactor. This is the tool-recovery failure class Harness-Bench measured at 24.6%.

## Limits

- Twelve tasks at one repeat is a **screen, not a leaderboard**. Treat a single failure as a signal and rerun with `--repeat 3` before acting on it.
- The suite discriminates on **reliability under long-horizon work**, not on capability ceiling. Six tasks were clean for every model.
- Provider-specific settings (reasoning effort, thinking defaults, temperature) are not normalized here. Pin them per model if a comparison must be exact.
- Concurrent runs share one Hermes install. Relocating `HERMES_HOME` per run was tried and abandoned: a cold home triggers Hermes' dependency bootstrap, which re-execs and loses its command line under contention.

## Adding a task

Create `tasks/<your-task>/` with:

```
task.json      {"max_turns": 40, "run_budget": 420, "timeout": 480}
prompt.md      the instruction the agent receives
workspace/     the starting files, copied fresh for every run
judge.py       argv[1] is the working directory; print {"checks": [...]}
```

`judge.py` must print exactly:

```json
{"checks": [{"check": "name", "pass": true, "detail": "why"}]}
```

Then add a solver to `tests/test_graders.py` and confirm it reports 0 on an empty workspace and 1.00 on your reference solution. A task whose grader cannot be verified in both directions is not a task.

## Licence

MIT.
