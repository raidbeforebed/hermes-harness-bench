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

**The published surfaces are checked against each other.** `tests/test_publish.py` asserts that every model in `results/` is named in every chart and in the results table, and it fails if a chart drops one. It exists because Ornith 1.5 122B published at 36/36 and appeared in none of the four charts: the table builder fell back to the raw model id while the chart builder filtered against its own hardcoded list. Nothing errored. A model that is published in one place and missing from another is worse than one that is missing from both.

**Memory and user config injection are disabled** (`--ignore-rules --ignore-user-config`), so runs are comparable and personal memory does not leak into results.

**Sessions are tagged and archived.** Runs use `--source tool`, which keeps them out of
the user's session lists, and the run's session rows are archived (`archived=1`) after
grading. Deletion is refused by the session store's runtime ledger, so archiving is the
operation that actually runs.

**A hard ceiling per task.** `--max-turns` bounds the turns, and the runner's own
subprocess timeout bounds the wall clock; exceeding either records a failure with exit
code `-1`. The `--run-budget` flag was removed from the child command when `hermes chat`
became a gateway client, so every record now carries `budget_mode: runner_timeout`.

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

# verify that every published result reached every chart and the README table
python3 tests/test_publish.py

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

Every model below was run on the same twelve tasks with the same graders. Raw
per-task records are in `results/`, one file per task per repeat. This table is
generated from `results/` by `tools/publish.py`, so it cannot drift from the data.

<!-- RESULTS:START -->
| Model | Repeats | Runs passed | Tasks all-pass | Mean turns | Mean tokens | Mean wall | Total |
|---|---|---|---|---|---|---|---|
| DeepSeek V4.1 Flash | 3 | **36/36 (100%)** | 12/12 | 11.9 | 24,275 | 34 s | 21 min |
| GLM-5.3 Flash | 3 | **36/36 (100%)** | 12/12 | **10.6** | 16,781 | 56 s | 34 min |
| MiniMax M3.1 Flash | 3 | **36/36 (100%)** | 12/12 | 16.5 | 23,413 | 82 s | 49 min |
| Ornith 1.5 122B (free) | 3 | **36/36 (100%)** | 12/12 | 12.1 | 146,497 | 115 s | 69 min |
| Laguna S 2.1 (free) | 1 | **12/12 (100%)** | 12/12 | 19.4 | 241,814 | 162 s | 32 min |
| Beam | 3 | 34/36 (94.4%) | 11/12 | 13.5 | 19,895 | 64 s | 39 min |
| Gemma 4 (cloud) | 3 | 32/36 (88.9%) | 10/12 | 12.4 | 137,578 | **18 s** | 11 min |
| Qwen3.6 35B (local) | 3 | 31/36 (86.1%) | 9/12 | 12.2 | **7,218** | 107 s | 64 min |
<!-- RESULTS:END -->

Read the **Repeats** column before comparing rows. A one-repeat row is a screening
measurement, not a stable figure: the same model scored 0.80 and then 1.00 across two
single-repeat runs of different timeout, which is why the graded baseline is three
repeats. Charts and this table are generated from `results/` by `tools/publish.py`, so
they cannot drift from the data.

Two columns, because they answer different questions:

- **Runs passed** counts every execution the model attempted, 36 for a three-repeat row
  and 12 for a screening row. It is not `pass@1` in the usual sense, which would be a
  single first-attempt probability; this repository does not report that, because one
  attempt per task is exactly what proved unreliable here.
- **Tasks all-pass** counts how many of the twelve tasks succeeded on every attempt. That
  is the stability figure. A model can show a high runs-passed number while still
  having a task it fails two-thirds of the time; the Beam row is that case.

### At a glance

![Reliability by model: runs passed and tasks stable for every model](assets/passes.svg)

The solid bar is the share of the runs each model attempted. The faded bar is the share
of the twelve tasks that passed on **every** attempt. Both are drawn as a proportion of
their own denominator and both carry their counts, so a one-repeat screening row cannot
be read as a graded one. A gap between the two bars is the useful signal: Beam passes 34
of 36 runs yet has a task it fails two-thirds of the time, and Qwen's gap is wider still.

![Token cost per task by model](assets/tokens.svg)

![Turns per task by model](assets/turns.svg)

![Quality against token cost](assets/pareto.svg)

The local model sits far left on token cost and lower on reliability, which is the
trade-off it actually offers: 7,218 tokens a task, under a third of DeepSeek's and well
under half of GLM's, at about 86 per cent of their reliability.

Ornith sits well to the right of the three paid models it ties with. It reaches 36/36 on
146,497 tokens a task, six times DeepSeek's, and at 115 seconds mean it is the second
slowest row in the table after Laguna S. What the chart cannot show is price: Ornith was
free on the Nous Portal on the day it ran, so the tokens buy no dollars, only latency.

Charts are generated from `results/` by `tools/make_charts.py`, so they cannot drift
from the data. Re-run it after a benchmark:

```bash
python3 tools/make_charts.py
```

### Five of eight rows are perfect, so cost is the only differentiator

Four models passed all **36 of 36** runs with zero variance on every individual task and
zero timeouts, and a fifth, Laguna S 2.1, passed all 12 runs of its single-repeat
screening pass. When five of the eight rows are perfect, pass rate tells you almost
nothing and the useful question becomes what each one costs to be perfect:

- **DeepSeek V4.1 Flash is the fastest of the five perfect rows**, at 34 seconds mean
  and 21 minutes for the full suite. It is also the most token-hungry of the three paid
  ones at 24,275, roughly 45 percent more than GLM. Fast here means it takes more,
  shorter turns rather than fewer efficient ones.
- **GLM-5.3 Flash is the most efficient**, at 10.6 turns and 16,781 tokens, a third
  fewer tokens than DeepSeek and 28 percent fewer than MiniMax. It is 65 percent
  slower in wall time, which on a cloud model costs little and on this suite does not
  matter.
- **MiniMax M3.1 Flash is perfect but expensive on both axes**: most turns, nearly
  the most tokens, and the slowest paid model at 82 seconds mean. Its
  `07-error-recovery` runs averaged 370 seconds against 35 to 124 for everything else.
- **Ornith 1.5 122B costs no money and the most tokens**: free on the Nous Portal,
  perfect on all 36 runs, and 146,497 tokens a task, six times DeepSeek's and more than
  any other model graded three times. Its 115 second mean is the price of that spend, not
  of the model being weak.

If the question is which model to make the default, GLM-5.3 Flash is the answer on a
paid account: it ties on quality, uses the fewest tokens, and takes the fewest turns.
DeepSeek wins only if wall-clock latency matters more than token spend, and Ornith wins
only while it stays free and the queue does not.

### Ornith 1.5 122B: free, perfect, and the most token-hungry row that does it

36 of 36 runs, twelve of twelve tasks stable, zero failed checks out of 216, at 12.1 turns
and 115 seconds mean. On this suite a 122B model served free through the Nous Portal
matched the three paid models exactly. Laguna S was free and perfect too, but on a single
pass, so Ornith is the first free row here to hold up across three repeats.

It is also the least efficient way to be perfect. 146,497 tokens a task is six times
DeepSeek and 8.7 times GLM, and most of it goes on one task: `07-error-recovery` averages
**414 seconds and 408,608 tokens** across its three runs. Every model finds that task the
expensive one, and Ornith pays it in the currency it has, which is tokens rather than
dollars. Its `06-multiturn-continuity` and `01-json-extract` runs also sit above 140,000
tokens, against 64,000 for `11-correction-handling`, so the spread is task-driven rather
than a fixed per-call overhead.

**One correction is recorded in the history.** Nine of the 36 runs overlapped a runaway
local MusicGen process holding the machine at load 8.2, and recorded inflated wall, turn
and token figures. Those three tasks were re-run on a clean machine and the nine records
replaced: mean wall 130.7 to 115.3 seconds, turns 13.3 to 12.1, tokens 5.95M to 5.27M.
Scores did not move, and every judge was re-executed against the saved work directories
to confirm the pass logic had not. The result files in `results/` are the corrected ones,
which is why this row differs from any figure quoted before that re-run.

What that buys: if a free tier is acceptable, Ornith ties the paid models on reliability
and costs nothing but latency. It is not a default for interactive work at 115 seconds a
task, and a free pool can be withdrawn or rate-limited without notice, which the score
cannot warn you about.

### Beam is 94.4%, not 100%

Eleven tasks passed three times out of three. The single failure is
`07-error-recovery`, and it is a budget story rather than a wrong answer:

| Repeat | Result | What happened |
|---|---|---|
| r1 | 0.00 | hit the 540 s budget after 58 turns |
| r2 | 0.00 | hit the 540 s budget after 18 turns, slow individual calls |
| r3 | 1.00 | finished cleanly in 167 s, 35 turns |

Beam thrashes on this task and sometimes does not converge. Its earlier single run
scored 12/12, which was luck.

### MiniMax M3.1 Flash: 36/36, and why the earlier 10/12 was wrong

Zero variance on every task, zero timeouts. That overturns the first published result
of 10/12, and the correction matters more than the number.

The earlier failures were **timeouts on a run with a misconfigured harness**, not
model limits. With three clean repeats it never missed. Published once, that model
would have been written off as unreliable on long-horizon work. It is instead one of
the five perfect rows.

It is also the slowest paid row, and the most expensive in turns. It spends **370
seconds on average on `07-error-recovery`** while every other task costs 35 to 124
seconds. Reliable, not fast.

### Gemma 4: the fastest model, and one ambiguous field it reads the same way every time

32 of 36 runs, ten of twelve tasks stable, 0.982 mean score. It is the fastest model
measured on this suite: **18 seconds mean and 11 minutes for the whole thing**, half of
DeepSeek and a sixth of the local Qwen. Turns are unremarkable at 12.4, so the speed is
per-call latency, not fewer steps.

The token column is the outlier and needs reading carefully. It reports **137,578 mean,
5.7 times DeepSeek**, which looks like a cost disaster and is not necessarily one: the
figure is the harness's reported usage and it includes context re-sent on every turn. The
session ledger shows a mean of 118,835 cached-input tokens per session against 12,659 fresh,
so the number measures **context processed, not billed cost**. Whether that is expensive
depends on the provider's cache pricing, which this suite does not model. Do not read the
token column as a price column for this row.

Both imperfect tasks are shown below, and only one of them is variance:

| Task | Result | What happened |
|---|---|---|
| `05-memory-nodestructive` | 0.83 in all three repeats | answered `entry_count: 2` instead of 3, in every repeat |
| `06-multiturn-continuity` | 0.86 in r1 only | one averaged duration off by 2 ms (97.4 vs 99.4), r2 and r3 clean |

`05` is the interesting one: identical answer three times, so it is comprehension rather
than luck. The task asks for `alice_total`, `entry_count` and `owners` from a three-entry
file. Gemma returned the correct `alice_total` of 40 and the correct `owners` of
`["alice","bob"]`, which proves it read all three entries, then reported `entry_count` as
**2**, the count for the queried owner rather than the count of entries in the file. The
field name is genuinely ambiguous; the grader is not, and the six models that pass this
check all read it as 3.

Qwen fails the same check with the same answer: `entry_count: 2`, in all three of its
repeats. Two different architectures, the same reading of an ambiguous field, which says
more about the prompt than about either model. The two rows are only distinguishable at
all by reading the check detail, which is why the field is worth naming rather than
counting.

That is the reason to prefer the two-column reporting over a single pass rate. A bare
88.9% would not tell you the model is fast, correct on everything else, and reliably wrong
in the same way on one specific ambiguity, which is a prompt-design problem it would hit
every time rather than a reliability problem that averaging hides.

### Qwen3.6 35B A3B: 86.1% on a 2017 GPU

Full hardware, launch flags and throughput: [reference-hardware.md](docs/reference-hardware.md).
Model: [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), GGUF from
[unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF), `UD-Q4_K_M`, served by
`llama-server` with `--n-cpu-moe` on an 8 GB card at 18.6 tokens/second.

| Task | r1 | r2 | r3 | Reading |
|---|---|---|---|---|
| `01-json-extract` | 0.71 | 1.00 | 1.00 | **flaky**: three runs gave 120.47, 125.48, and a correct 149.97 |
| `07-error-recovery` | 0.00 | 1.00 | 1.00 | **flaky**: once wrote the output without fixing the input the pipeline validates |
| `05-memory-nodestructive` | 0.83 | 0.83 | 0.83 | **deterministic**: `entry_count: 2` for a three-entry file, every time |
| the other nine | 1.00 | 1.00 | 1.00 | stable, zero variance |

Nine of twelve passed every time. The two kinds of failure are worth separating:

**Task 05 is a systematic counting error.** It computes `alice_total: 40` correctly
and then miscounts the rows, in all three runs. A 3B-active-parameter model is
dependable at reading and transforming data and unreliable at counting it, and no
amount of context or retrying changes that.

**Task 01 is arithmetic instability.** Same file, three different sums, correct about
one time in three, so any single run either overstates or understates it.

One local run took **70 turns and 464 seconds** and still passed. Under the default
budget it would have been killed and misreported as a failure. That is why
`--budget-scale` exists, and why turn limits are deliberately not scaled with it.

### What three repeats actually bought

- **A model was nearly written off wrongly.** MiniMax's 10/12 became 36/36.
- **A lucky run was exposed.** Beam's 12/12 became 94.4%.
- **Two failure types were separated.** On the local model, task 05 looked like a
  one-off and turned out deterministic; task 01 looked deterministic and turned out
  flaky. One run cannot tell those apart.
- **A budget artefact was caught.** Beam's failures are timeouts, not wrong answers,
  which points at a different remedy.
- **A machine fault was caught, not published as a model result.** Nine Ornith runs
  overlapped a runaway local process. Re-running them on a clean machine moved the wall,
  turn and token figures and left the scores at 1.00, which is what tells you a slow box
  apart from a weak model. That correction is in the history, not quietly absorbed.

- **The top of the table survived, the middle did not.** GLM and DeepSeek both came
  out at 36/36, so their single-run 12/12 was real. MiniMax and Beam both moved. The
  models you would have ranked by a single run are the ones most likely to have been
  ranked wrongly, and you cannot predict which from the score.

The asymmetry is the argument: a flaky model scored once looks either better or
worse than it is, and the number alone never says which. Three repeats cost roughly
an hour per model and settled it.

## Limits

- Twelve tasks is a **screen, not a leaderboard**. It separates reliable from
  unreliable well; it does not finely rank models that are all reliable.
- **Free tiers move.** Laguna S 2.1 and Ornith 1.5 122B were free on the Nous Portal on
  the dates recorded in their result files. A free row can be withdrawn, rate-limited or
  re-priced without any of its numbers changing, so read the price from the provider, not
  from this table.
- The suite discriminates on **reliability under long-horizon work**, not on
  capability ceiling. Several tasks are clean for every model tested.
- **A timeout is a failure, not a verdict on quality.** Every Beam failure was a
  budget exhaustion on a task where it thrashes, not a wrong answer. On slower
  hardware, or with a different budget, that number moves. The budget is recorded in
  every run manifest for this reason.
- Provider-specific settings (reasoning effort, thinking defaults, temperature) are
  not normalized. Pin them per model if a comparison must be exact.
- Concurrent runs share one Hermes install. Relocating `HERMES_HOME` per run was tried
  and abandoned: a cold home triggers Hermes' dependency bootstrap, which re-execs and
  loses its command line under contention.

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
