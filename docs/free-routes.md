# The Nous Portal free routes, as measured

The Portal catalogue is OpenAI-compatible at `https://inference-api.nousresearch.com/v1`. Two facts
make it worth re-reading rather than remembering: the free list rotates by delisting, so a queued run
fails as model-not-found rather than a billing surprise, and a zero price in the catalogue is not a
promise that the route will answer.

Snapshot taken 2026-11-19: **413 models listed, 5 with prompt and completion both zero**.

| Route | Answers | Tool calls | Queued for the eighteen-task suite |
| --- | --- | --- | --- |
| `qwen/qwen4-max:free` | yes | yes | yes |
| `meituan/longcat-2.5:free` | yes | yes | yes |
| `tencent/hy4-flash:free` | yes | yes | yes |
| `stepfun/step-3.7-flash:free` | **no** | n/a | no |
| `upstage/solar-mini4:free` | **no** | n/a | no |

## The two refusals

Both `stepfun/step-3.7-flash:free` and `upstage/solar-mini4:free` answer every request with HTTP 400
and this body:

```
{"status":400,"message":"This request is not valid. Check the request parameters.
 Additional info: missing tags"}
```

The field is not documented in the OpenAI-compatible surface the client uses. The shapes tried on
2026-11-19, all refused: `tags: ["hbench"]` (400, `missing user tag`), `user: "hermes"` (400,
`missing tags`), `metadata.tags` (400, `missing tags`), `tag` and `user_tag` (400, `missing tags`), and
three object shapes `[{type|key: ..., value: ...}]` (HTTP 500). A zero-priced row that refuses
inference with an undocumented parameter requirement is a catalogue entry, not a runnable route, so
neither is benchmarked here and neither is reported as a score.

## The reasoning route reads as an empty reply on a small budget

`tencent/hy4-flash:free` returns its thinking in a separate `reasoning` field. With `max_tokens: 20`
it spends the entire budget there and returns `content: null` with `finish_reason: length`; at 200
tokens the same prompt returns `content: "ok"` with `finish_reason: stop`. A `content: null` record on
this route is a budget fact, not a refusal, and is only a fault when `finish_reason` is `length`.

## Reading a free row in the results table

Free rows are screening measurements: one series, no vendor support, and the route can vanish without
notice. They are comparable to the paid rows on the instrument and the repeat count, and they are not
comparable on cost, because the price they were measured at is a promotion rather than a rate.
