Work through this in three stages. Each stage builds on the last, and I only
want the final file.

STAGE 1 - Parse events.log into a file called `stage1.json`. Shape:
  {"events": [{"ts":"<iso>","level":"<lvl>","service":"<svc>","status":<int>,"ms":<int>}, ...]}
Preserve file order. Every one of the 10 lines must appear.

STAGE 2 - Using stage1.json, produce `stage2.json`. For each service, count
events and compute the average ms rounded to 2 decimals. Shape:
  {"by_service": {"<svc>": {"count": <int>, "avg_ms": <number>}}}

STAGE 3 - Using stage1.json, produce `final.json`. Consider ONLY events whose
level is ERROR or WARN. Group them by service and list the status code of each
such event, sorted ascending. One entry per event, so a status that occurs
twice appears twice. Shape:
  {"problem_services": {"<svc>": {"count": <int>, "statuses": [<int>, ...]}}}
Omit any service with no ERROR or WARN events.

Leave stage1.json and stage2.json on disk. Write final.json last.
