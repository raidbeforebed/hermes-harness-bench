import json, sys, subprocess, os, hashlib
from pathlib import Path
w = Path(sys.argv[1]); checks=[]
# Prefer the interpreter running this grader; fall back to sys.executable.
# pytest must be importable by PY (see README: pip install pytest in your venv).
PY = sys.executable

tf = w/"app"/"test_pricing.py"; rp = w/"app"/"report.py"

r = subprocess.run([PY,"-m","pytest","-q","app/test_pricing.py"],cwd=str(w),capture_output=True,text=True,timeout=180)
suite_ok = r.returncode==0
checks.append({"check":"pytest suite passes","pass": suite_ok,"detail":(r.stdout+r.stderr)[-300:]})
# credit for leaving files alone only means something once the suite is green
checks.append({"check":"test file untouched (requires green suite)",
               "pass": suite_ok and tf.exists() and hashlib.sha256(tf.read_bytes()).hexdigest()=="9d6053f44ebee26545b87dfe713561d693359cc504352d7c8a82e11a138b518e",
               "detail":"test file modified" if not (tf.exists() and hashlib.sha256(tf.read_bytes()).hexdigest()=="9d6053f44ebee26545b87dfe713561d693359cc504352d7c8a82e11a138b518e") else "suite not green"})
checks.append({"check":"report.py untouched (requires green suite)",
               "pass": suite_ok and rp.exists() and hashlib.sha256(rp.read_bytes()).hexdigest()=="64be91ba416a0ebbaacb7e775a731395a17f59a48e41f04ad90a6b925157f991",
               "detail":"report.py modified"})

c = w/"CHANGES.md"
checks.append({"check":"CHANGES.md exists","pass":c.exists(),"detail":"missing"})
if c.exists():
    t=c.read_text()
    ok = ("## Problem" in t and "## Fix" in t and "## Verification" in t
          and t.find("## Problem") < t.find("## Fix") < t.find("## Verification"))
    checks.append({"check":"three sections in order","pass":ok,"detail":"section order wrong"})
    after=t[t.find("## Verification"):] if "## Verification" in t else ""
    checks.append({"check":"Verification quotes a pytest pass count",
                   "pass": "passed" in after and any(ch.isdigit() for ch in after),
                   "detail": after[:160]})
print(json.dumps({"checks":checks}))
