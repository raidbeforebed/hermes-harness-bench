import json, subprocess, sys, os
from pathlib import Path
w = Path(sys.argv[1]); checks=[]
# Prefer the interpreter running this grader; fall back to sys.executable.
# pytest must be importable by PY (see README: pip install pytest in your venv).
PY = sys.executable

# The pipeline must run green from a clean invocation. This is the primary gate:
# nothing else can earn credit unless the build actually passes.
r = subprocess.run([PY,"build.py"], cwd=str(w), capture_output=True, text=True, timeout=120)
green = r.returncode==0
checks.append({"check":"pipeline runs green","pass": green,
               "detail": (r.stdout+r.stderr)[-300:]})

bp = w/"build.py"
checks.append({"check":"build.py untouched (and credit requires a green run)",
               "pass": green and bp.exists() and "def main" in bp.read_text(),
               "detail":"build.py missing/rewritten" if not bp.exists() else ("build still failing" if not green else "ok")})

c = w/"clean.csv"
exists = c.exists()
checks.append({"check":"clean.csv produced (requires green run)","pass": green and exists,
               "detail":"missing" if not exists else ("build not green" if not green else "ok")})
if exists and green:
    import csv
    with open(c) as fh:
        rows=list(csv.DictReader(fh))
    checks.append({"check":"4 active rows survive","pass": len(rows)==4, "detail":"%d rows"%len(rows)})
    checks.append({"check":"only active rows","pass": all((x.get("status") or "").strip()=="active" for x in rows),
                   "detail":str([x.get("status") for x in rows])})
    ints=all(str(x.get("qty","")).strip().isdigit() for x in rows)
    checks.append({"check":"qty is a bare integer","pass": ints, "detail":str([x.get("qty") for x in rows])})
    # the real intent: inactive row 2 must NOT be present, all four actives must be
    ids=sorted((x.get("id") or "").strip() for x in rows)
    checks.append({"check":"ids are exactly 1,3,4,5","pass": ids==["1","3","4","5"],"detail":str(ids)})
print(json.dumps({"checks":checks}))
