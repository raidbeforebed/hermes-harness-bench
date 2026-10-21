# Reference hardware: what a local baseline actually costs

Published local results are only useful if the machine is described. Most "a 35B
runs locally" claims omit the spec, and the spec is what makes the number
reproducible or not.

This is the hardware behind the Qwen3.6 35B A3B baseline in the README. It is
deliberately old: a 2019 consumer CPU and a 2017 mid-range GPU. The point is to
show what a local model does on hardware someone already owns, not on a rented
A100.

## The machine

| Component | Spec | Age |
|---|---|---|
| CPU | AMD Ryzen 9 3900X, 12 cores / 24 threads, 3.8 GHz base | Launched 2019 |
| GPU | NVIDIA GTX 1070 Ti, 8 GB GDDR5, Pascal (sm_61) | Launched 2017 |
| RAM | 32 GB DDR4 | 2019 |
| Swap | 20 GB, with 16 GB zram | - |
| OS | Pop!_OS, kernel 7.2.6 | - |

Captured live during the benchmark run:

| Reading | Value |
|---|---|
| CPU package temperature | 42.9 C |
| GPU temperature | 52 to 56 C |
| GPU utilisation | 84 percent while generating |
| VRAM used | 7,101 of 8,192 MB |
| VRAM attributed to llama-server | 6,636 MB |
| llama-server resident memory | 19 GB |

That last row is the important one. **Most of this model is not on the GPU.**

## The model and how it is served

| Setting | Value |
|---|---|
| Model | Qwen3.6 35B A3B, Mixture of Experts, 3B active parameters |
| File | `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` |
| Quantisation | Q4_K_M, reported as `Q4_K - Medium` |
| Server | llama.cpp `b11669` (`build-next-20261020-v0.7.0`) |
| Launch flags | `-ngl 99 --n-cpu-moe ...` |
| Context | 98,304 tokens |
| Parallel slots | 1 |
| Sampling defaults | temperature 1.0, top-p 0.95, top-k 20, min-p 0.05 |
| Authentication | bearer token on the server |

### Why `--n-cpu-moe` matters more than the GPU does

A Mixture of Experts model keeps its parameters in expert layers and routes each
token to only a few of them. Running a 35B MoE in 8 GB of VRAM is only possible by
leaving the expert weights in system RAM and keeping the attention layers on the
card.

`--n-cpu-moe N` does exactly that for the first N layers, so the 19 GB resident
footprint above is mostly expert weights in RAM. Generation therefore has a
CPU-bound component that a larger GPU would remove entirely, which is why 18.6
tokens/second here says almost nothing about the same model on a 24 GB card.

**If you quote a local token rate, quote the offload strategy with it.** Without
the flag list, the number is not comparable to anything.

## Measured performance on this hardware

| Quantity | Value |
|---|---|
| Generation | 18.6 tokens/second |
| Prompt processing at 19k context | 405 tokens/second |
| Cold prefill of a 19k prompt | 47.5 s |
| KV cache reuse on repeat prefix | confirmed, ~19k tokens cached |
| Effective cost per agent turn | ~9 s |

Prefill is roughly 20 times faster than generation. That asymmetry is why a local
agent run is closer to cloud wall-time than the generation rate suggests: most of
the prompt is processed once and then cached, and only the new output tokens are
slow.

## Results on this hardware

Twelve tasks, `--budget-scale 3`:

| pass@1 | Mean score | Mean turns | Mean tokens | Mean wall | Total |
|---|---|---|---|---|---|
| 10/12 | 0.96 | 12.2 | 6,938 | 98 s | 20 min |

For comparison, the same twelve tasks on cloud models took 55 to 122 seconds per
task with no budget scaling at all. A 2017 GPU running a 4-bit 35B lands in the
same order of magnitude once the prompt is cached.

## What this hardware cannot do

Honest limits, because a reader will hit them:

- **8 GB of VRAM caps the context-plus-layers budget.** 96K context fits, but with
  only 1 parallel slot. Two concurrent requests would not.
- **Expert layers in RAM cost latency.** Anything requiring many short, fast turns
  feels slow interactively, whatever the token rate says.
- **Precision is a compromise.** Q4_K_M is what fits. A Q5 or Q6 of a smaller model
  may suit some workloads better than a Q4 of a larger one, and this suite is the
  way to find out on your own card rather than trusting someone else's benchmark.

## Reproducing

You do not need this exact hardware. You need:

1. `llama-server` with `n_ctx` at 32k or more, and tool calling verified
2. a provider entry in `config.yaml` with `base_url` pointing at it
3. `--budget-scale` set from your own measured tokens/second

See [local-models.md](local-models.md) for the checks.
