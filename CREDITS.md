# Credits and provenance

This repository builds on published work and on the open-source Hermes ecosystem.
Nothing here was copied as source code from another project, but the ideas,
taxonomy and measurements below shaped the design and must be credited.

---

## 1. Harness-Bench, Peking University and Qiyuan Tech

**What this repository takes from it:** the failure taxonomy that the twelve tasks
are built around, and every percentage quoted in the README.

| Figure used here | Source |
|---|---|
| Contract / format 36.4% | Harness-Bench Table 3 |
| Tool / recovery 24.6% | Harness-Bench Table 3 |
| Evidence / grounding 14.6% | Harness-Bench Table 3 |
| Artifact commitment 11.1% | Harness-Bench Table 3 |
| State / continuation 9.3% | Harness-Bench Table 3 |
| Hermes harness 22.6 turns, 139.7K tokens | Harness-Bench Table 2 |
| 5,194 trajectories, 106 tasks (paper version) | Harness-Bench abstract |

**What this repository does not take:** no task content, no graders, no Docker
images, no oracles, no code. Harness-Bench ships 300 sandboxed tasks with hidden
container oracles; this repository has twelve tasks with local file assertions.
The two are not interchangeable, and this project does not claim to reproduce
Harness-Bench results.

Note that the Harness-Bench repository states no licence file. The paper is
cited, not the code, and no code was used.

Their requested citation:

```bibtex
@article{yao2026harnessbench,
  title   = {Harness-Bench: Measuring Harness Effects across Models in Realistic Agent Workflows},
  author  = {Yao, Yilun and Tan, Xinyu and Liu, Chao-Hsuan and Li, Yaoming and Wang, Zhengyang and Yu, Wenhan and Tan, Zhewen and Tian, Yuxuan and Zhao, Guangxiang and Sun, Lin and Zhang, Xiangzheng and Yang, Tong},
  journal = {arXiv preprint arXiv:2605.27922},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.27922}
}
```

- Paper: https://arxiv.org/abs/2605.27922
- Code: https://github.com/Qihoo360/harness-bench
- Site: https://www.harness-bench.ai/

---

## 2. Bent-Solutions/hermes-bench (MIT)

**What this repository takes from it:** the idea for four tasks. Section 15 of
their build specification lists ten further benchmark categories as future work;
four of those category descriptions became tasks here:

| Task here | Their section 15 category |
|---|---|
| `07-error-recovery` | 15.4 Tool Calling Accuracy, error recovery |
| `08-constraint-format` | 15.6 Instruction Following, format compliance |
| `09-long-context-retrieval` | 15.1 Long-Context Benchmarks, needle in a haystack |
| `10-multifile-refactor` | 15.2 Code Benchmarks, bug fixing and multi-file |

**What this repository does not take:** no code, no schema, no task text. Their
grading contract is `pass_keywords`, a JSON array of strings matched against the
model's reply. This project deliberately uses the opposite approach: assertions
over the resulting file tree. None of their class or module names appear here.

They are MIT licensed, and their README invites contributions. This repository is
not a fork and is not affiliated with it.

- Repo: https://github.com/Bent-Solutions/hermes-bench

---

## 3. Nous Research, Hermes Agent (MIT)

This benchmark runs the Hermes CLI. Everything it measures is a property of a
model running inside that harness, and the harness is the reason the measurements
are worth taking: the Hermes harness averages far more turns per task than its
peers, which is what makes long-horizon reliability the binding constraint.

The `--format stream-json`, `--source tool`, `--ignore-rules`,
`--ignore-user-config`, `--max-turns` and `--run-budget` flags are Hermes CLI
features. The runner is a wrapper around them, nothing more.

- Repo: https://github.com/NousResearch/hermes-agent
- Docs: https://hermes-agent.nousresearch.com/docs

---

## 4. verkyyi/hermesbench (MIT)

Consulted for comparison while deciding what to measure. That project evaluates
whole personal-agent configurations across 27 workflow recipes with redacted trace
inspection. It takes a different approach: it scores the configuration including
prompt and skills, where this repository holds the harness fixed and varies only
the model.

No code or content was taken. Credited because it shaped the decision about what
this project should not try to be.

- Site: https://verkyyi.github.io/hermesbench/
- Repo: https://github.com/verkyyi/hermesbench

---

## 5. Source material that informed the design

A research summary on agentic benchmarking was reviewed before this project
started. Most of its claims were verified against primary sources; three were not
accurate and are recorded here so nobody repeats the errors:

1. **Two different scores were conflated.** The document compared Hermes Bench
   results (GLM-5.3 71.8, DeepSeek V4.1 Flash 70.7, Claude Opus 5.5 76.7) as though
   they were Hermes Index results. The Index averages four suites and is far
   harsher: Opus 5.5 is 63.31, GLM-5.3 falls to 39.23, largely because
   TerminalBench Science collapses for most models.
2. **Harness-Bench task counts were stale.** The paper uses 106 tasks and 5,194
   trajectories; the current repository ships 274 general plus 26 multimodal.
3. **Bent-Solutions' Hermes integration was overstated.** Tool-calling accuracy,
   long context, RAG quality and code benchmarks are roadmap items, not shipped
   features.

Verified reference points, for anyone repeating this work:

- Nous Hermes Index: https://portal.nousresearch.com/bench
- Hermes Bench: 150 tasks across 25 categories, graded on files and state left behind
- Berkeley Function Calling Leaderboard (BFCL): https://gorilla.cs.berkeley.edu/leaderboard.html
  Tool-call accuracy scored by AST matching and executable verification. BFCL v4 adds
  agentic evaluation and multi-turn interaction. Model-level, harness-agnostic, so it
  screens a model before it ever meets a harness rather than replacing a harness test.

---

## Licence of this repository

MIT. See `LICENSE`.

The MIT licence covers the code, tasks and graders in this repository. It does not
grant any rights to the third-party projects credited above; consult their own
licences. The Harness-Bench paper is cited as academic work.
