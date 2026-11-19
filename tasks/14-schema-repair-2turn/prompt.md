You are recovering from a rejected tool call.

`turn1_rejected.json` shows what your agent sent in turn 1 and the schema error
the harness returned. `tool_schema.json` is the exact shape the call must take.

Write the CORRECTED tool call to `corrected_call.json`. Requirements:

- It must be a single JSON object (one tool call). Not a list, and not wrapped in
  a retry, attempt, or batch structure.
- It must satisfy `tool_schema.json` exactly: both required properties present,
  correct types, and NO properties the schema does not list.
- Use the SAME operation (`search_files`) and the SAME file path from the turn-1
  request: `/srv/payments/charge.py`.
- Search for the SAME pattern the user asked for: `def calculate_total`.
- Do not truncate, and do not add speculative fields.

Write only `corrected_call.json`. Do not print the call as your reply.
