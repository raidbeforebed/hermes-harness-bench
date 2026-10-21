# Running this benchmark against local models

Two serving stacks are covered. They fail differently, and one has a trap that
makes a "local" run silently hit a cloud model.

- **llama.cpp** (`llama-server`): works out of the box if `n_ctx` is large enough.
- **Ollama**: needs its context window raised, or every run dies before the model starts.

---

## llama.cpp

Verified against `llama-server` build `b11669-620d2225e`, serving a Qwen3.6 35B
A3B GGUF at 96K context on an 8 GB card.

### Check the served context first

```bash
curl -s http://127.0.0.1:8080/props -H "Authorization: Bearer <key>" \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['default_generation_settings']['n_ctx'])"
```

That prints what the server is actually serving. Confirm it is tens of thousands,
not 2048 or 4096. An agent system prompt plus tool schemas runs to roughly 20K
tokens before the model reads your task, so a small `n_ctx` fails even when the
model is capable.

### Check tool calling before a full run

Every task here needs tool calls. Confirm the server emits them natively:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer <key>" -H "Content-Type: application/json" \
  -d '{"model":"<alias>","messages":[{"role":"user","content":"Create hello.txt containing HI using the tool."}],
       "tools":[{"type":"function","function":{"name":"write_file","description":"Write a file",
       "parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},
       "required":["path","content"]}}}],"tool_choice":"auto","max_tokens":256}'
```

A healthy response has `"finish_reason": "tool_calls"` and a populated
`tool_calls` array with JSON arguments. If `tool_calls` is empty and the model
prints the call as prose, the chat template is not tool-aware, and the harness
will spin: the agent never sees a tool result and keeps retrying until its turn
budget expires.

### Register the provider

`base_url` is mandatory. Without it Hermes warns that the request is not sent
anywhere and would otherwise fall back to OpenRouter, which would make a local
test quietly hit a cloud model:

```
provider 'ollama' has no endpoint configured, so the request is not sent
anywhere (it would otherwise fall back to OpenRouter)
```

```yaml
custom_providers:
  - name: qwen-local
    base_url: http://127.0.0.1:8080/v1
    api_key: <your key>
    default_model: <model alias or gguf path>
    models: {}
    models_discovered: true
```

The `--model` value can be the GGUF path or an alias, whichever the server
reports in `/v1/models`.

### Run

```bash
python bin/hbench.py run --model <alias> --provider qwen-local --tasks all
```

### Budgets

A 35B A3B on an 8 GB card answered a simple file write in about 95 seconds and
`01-json-extract` in about 55. The harder tasks (`07-error-recovery`,
`10-multifile-refactor`, `12-long-chain`) will exceed the default `run_budget` on
slower hardware. Time one task first, then scale the budgets in
`tasks/*/task.json`. Record that you raised them, because a raised budget is a
different measurement.

---

## Ollama: the 4k context trap

Ollama chooses its default context from available VRAM. Under 24 GiB it serves
**4096 tokens**, whatever the model supports:

> Ollama defaults to the following context lengths based on VRAM: under 24 GiB:
> 4k context; 24 to 48 GiB: 32k; 48 GiB and above: 256k.
>
> https://docs.ollama.com/context-length

That is smaller than an agent system prompt, so every task fails before the model
does any work:

```
This conversation has grown too long for <model> to read, and Hermes couldn't
shrink it enough automatically.
```

The trap is that it fails upstream and misleadingly. `GET /api/show` reports the
model's training context (262144 for a 262k model), so the client sizes its budget
against that number while the server serves 4096 and truncates. On the
OpenAI-compatible endpoint a per-request `num_ctx` or `options.num_ctx` field is
**ignored**, so the client cannot correct it from the request side.

### Fix

```bash
sudo systemctl edit ollama
```

```ini
[Service]
Environment="OLLAMA_CONTEXT_LENGTH=32768"
```

```bash
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

Verify after a warm request:

```bash
curl -s localhost:11434/api/ps | python3 -m json.tool | grep context_length
```

Use 32k as a floor for agent work on an 8 GB card. Higher costs KV cache and
shrinks the batch that fits.

---

## Reading local results honestly

Do not compare a 9B or a 4-bit 35B against a frontier cloud model on pass rate
alone and conclude the local model is broken. The comparisons that mean something:

- **against its own previous run**, after changing quantisation, context or prompt
- **against other local models** of similar size
- **on turn count**, since long-horizon drift shows up there first

Expect failures in specific places. A local model that passes the short,
single-output tasks and fails `07-error-recovery`, `10-multifile-refactor` or
`12-long-chain` is telling you about tool recovery and long-horizon state, which
is information rather than noise.

A worked example from this setup: Qwen3.6 35B A3B passed five of seven checks on
`01-json-extract` and lost the other two by dropping one row from a four-row
filtered sum, producing 120.47 where the file says 149.97. That is a real
arithmetic and table-reading failure, correctly caught, and it is the kind of
error that never shows up in a chat transcript.

## If `turns` is null

`result.json` reads turn and tool counts from the session row in the Hermes
database. A null means no session row was written, which almost always means the
run died before its first successful model call. Work backwards to the server:
check `n_ctx`, then check tool calling.
