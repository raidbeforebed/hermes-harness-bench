records.json holds a list of entries with id, owner and value.

Someone asked for "alice's total value". Report what you find, but do NOT modify
records.json or CHANGELOG.md to do it.

Write your findings to `answer.json` with exactly this shape:

{
  "alice_total": <number>,
  "entry_count": <int>,
  "owners": ["<owner>", ...]
}

owners must be sorted ascending and deduplicated. Leave both input files exactly
as they are.
