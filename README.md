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
- **Qwen3.6 35B A3B** (Apache 2.0, Alibaba/Qwen) is the local model in the baseline. The GGUF build is [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF), quantised with Unsloth Dynamic 2.0.
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

`llama-server` works directly; confirm `n_ctx` is large enough and that the server
emits native tool calls. `Ollama` on a card under 24 GiB serves only 4096 tokens
by default, which is smaller than an agent system prompt, so every task fails
before the model does any work. Both cases, and how to read local results
honestly, are covered in [docs/local-models.md](docs/local-models.md).

```bash
# llama.cpp
python bin/hbench.py run --model <alias> --provider qwen-local --tasks all
```

## Results as published

Measured 2026-10-21, one repeat each, twelve tasks. Raw per-task records are in `results/`.

| Model | pass@1 | Mean score | Mean turns | Mean tokens | Mean wall |
|---|---|---|---|---|---|
| GLM-5.3 Flash | 12/12 | 1.00 | 9.8 | 15,720 | 66s |
| DeepSeek V4.1 Flash | 12/12 | 1.00 | 12.5 | 20,512 | 55s |
| Beam | 12/12 | 1.00 | 14.7 | 23,716 | 77s |
| MiniMax M3.1 Flash | 10/12 | 0.83 | 18.9 | 16,477 | 122s |

### Local baseline: Qwen3.6 35B A3B

Run 2026-10-21, **three repeats per task** (36 runs), against `llama-server` on a
GTX 1070 Ti with 8 GB of VRAM. Full hardware, launch flags and throughput in
[reference-hardware.md](docs/reference-hardware.md).

- Model: [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), Apache 2.0
- GGUF: [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF), file `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`

| pass@1 | Mean score | Mean turns | Mean tokens | Mean wall | Total |
|---|---|---|---|---|---|
| 31/36 (86.1%) | 0.95 | 12.3 | 7,218 | 106 s | 1.07 h |

**This is more interesting than the single-run result, because three repeats
separate variance from systematic failure.**

| Task | r1 | r2 | r3 | Reading |
|---|---|---|---|---|
| `01-json-extract` | 0.71 | 1.00 | 1.00 | **flaky**: arithmetic instability |
| `07-error-recovery` | 0.00 | 1.00 | 1.00 | **flaky**: sometimes does not finish the repair |
| `05-memory-nodestructive` | 0.83 | 0.83 | 0.83 | **deterministic failure**, same error every time |
| the other nine | 1.00 | 1.00 | 1.00 | stable |

Nine of twelve tasks passed every single time with zero variance. That is the
headline: on this suite the model's behaviour is mostly predictable, and where it
fails it fails either consistently or not at all, rather than randomly across the
board.

**`05-memory-nodestructive` is a systematic counting error.** In all three runs it
answered `entry_count: 2` for a file containing three entries, while computing
`alice_total: 40` correctly. It reads and sums the values right and then miscounts
the rows, every time. That is the clearest single finding here: a 3B-active-parameter
model is reliable at reading and transforming data and unreliable at counting it,
and no amount of context or retries changes that.

**`01-json-extract` is arithmetic instability.** Three runs produced 120.47, 125.48
and a correct 149.97. Same task, same file, three different sums. The correct answer
appears about a third of the time, so a single run either overstates or understates
this model on that task.

**`07-error-recovery` failed once by not doing the work.** In repeat 1 it wrote a
`clean.csv` but never fixed the input file the pipeline actually validates, so the
build stayed red. Repeats 2 and 3 diagnosed it correctly. This is the tool-recovery
failure class: knowing the fix and not applying it.

One run took **70 turns and 464 seconds** and still passed. Under an unscaled budget
it would have been killed and misreported as a failure. That is the case for
`--budget-scale` on slow hardware, and the case for not raising turn limits.

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
