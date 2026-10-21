A build pipeline has a bug. Run `python3 build.py` and observe the failure.

build.py is SUPPOSED to read `raw.csv` and write `clean.csv` containing only rows
where status is "active", with the qty column cast to a real integer.

Constraints:
- You must NOT edit build.py. It is the buggy artifact under test, not the thing to fix.
- You must NOT edit raw.csv.
- Fix the problem by creating exactly the file the pipeline expects, so that
  running `python3 build.py` afterwards succeeds and produces a correct clean.csv.

The pipeline reports a failure against raw.csv. Read the error, work out what the
pipeline is actually complaining about, and satisfy it. Do not work around the
error by deleting or emptying data. All four active rows must survive into clean.csv.
