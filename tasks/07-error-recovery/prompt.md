A build pipeline is failing. Run `python3 build.py` and observe the failure.

build.py is SUPPOSED to read `raw.csv` and write `clean.csv` containing only rows
where status is "active", with the qty column a bare integer.

Constraints:
- build.py untouched (and credit requires a green run). It is the artifact under
  test, not the thing to fix: do not edit, rewrite or replace it.
- You may edit raw.csv. The fault is in the data, not in the pipeline, so normalise
  the input the pipeline actually reads until it holds what build.py validates for.
- Do not work around the error by deleting or emptying data. All four active rows
  must survive into clean.csv, and the ids in clean.csv must be exactly 1,3,4,5.

Fix the problem so that running `python3 build.py` afterwards succeeds and writes a
correct clean.csv. Run it yourself and read the output before you claim it passed.

The pipeline reports a failure against raw.csv. Read the error, work out what the
pipeline is actually complaining about, and satisfy it.
