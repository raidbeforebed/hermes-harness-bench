# Running this benchmark against local models

Local models need one extra step that cloud models do not. Skip it and every task
fails before the model does any work.

## The problem

Ollama chooses its default context window from available VRAM. On a card under
24 GiB it serves **4096 tokens**, whatever the model itself supports:

> Ollama defaults to the following context lengths based on VRAM:
> under 24 GiB: 4k context; 24 to 48 GiB: 32k; 48 GiB and above: 256k.
>
> https://docs.ollama.com/context-length

An agent's system prompt plus tool schemas runs to tens of thousands of tokens.
At 4096 the run cannot even begin. The symptom is:

```
This conversation has grown too long for <model> to read, and Hermes couldn't
shrink it enough automatically.
```

The trap is that it fails *quietly upstream*. `GET /api/show` reports the model's
capability (for example 262144 for a model trained at 262k), so Hermes sizes its
budget against that number. The server then serves 4096 and truncates. On the
OpenAI-compatible endpoint (`/v1/chat/completions`) a per-request `num_ctx` or
`options.num_ctx` field is **ignored**, so the client cannot correct it.

Net effect: the client believes it has a large window, the server gives it a
small one, and the run dies with a confusing message.

## The fix

Set the context length on the Ollama server, not per request. On the machine
running Ollama:

```bash
sudo systemctl edit ollama
```

Add:

```ini
[Service]
Environment="OLLAMA_CONTEXT_LENGTH=32768"
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

Choose the value from VRAM. 32k is a sensible floor for agent work on an 8 GB
card. Going much higher costs KV cache and shrinks the batch that fits, which
matters when the model is being used for anything else at the same time.

Verify it took effect:

```bash
curl -s localhost:11434/api/ps | python3 -m json.tool | grep context_length
```

After a warm request the loaded model should report your value, not 4096.

## Register the provider with Hermes

A provider entry needs `base_url`. Without one, Hermes warns that the request is
not sent anywhere and would otherwise fall back to OpenRouter, which would make a
"local" test silently hit a cloud model:

```
provider 'ollama' has no endpoint configured, so the request is not sent
anywhere (it would otherwise fall back to OpenRouter)
```

In `config.yaml`:

```yaml
custom_providers:
  - name: local-ollama
    base_url: http://127.0.0.1:11434/v1
    api_key: ollama-local
    default_model: <your-model-tag>
    models: {}
    models_discovered: true
```

Then:

```bash
python bin/hbench.py run --model <your-model-tag> --provider local-ollama --tasks all
```

## Expect lower scores, and read them correctly

Do not compare a local 9B against a frontier cloud model on pass rate alone and
conclude the local model is broken. The comparison that matters is:

- **against its own previous run**, after changing quantisation, context or prompt
- **against other local models** of similar size
- **on turn count**, since long-horizon drift shows up there first

A small model may pass the short tasks and fail `07-error-recovery`,
`10-multifile-refactor` or `12-long-chain`. That is the tool-recovery and long
horizon failure classes, and it is information rather than noise.

## Budgets are sized for fast models

The default per-task budgets assume a cloud model answering in tens of seconds.
A local model on a Pascal-era card is far slower. Before running the full suite,
time one task and scale:

```bash
time python bin/hbench.py run --model <tag> --provider local-ollama --tasks 01-json-extract
```

If that task takes four minutes, the budgets in `tasks/*/task.json` will time out
the harder tasks. Raise `run_budget` and `timeout` for local runs rather than
concluding the model failed. Record that you did, because a raised budget is a
different measurement.

## A note on turn counting

`result.json` reads turn and tool counts from the session row in the Hermes
database. If a local run produces `turns: null`, the session was not written, and
that usually means the run died before its first successful model call, which
points back at the context problem above.
