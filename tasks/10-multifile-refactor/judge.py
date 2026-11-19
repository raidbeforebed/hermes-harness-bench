import json, sys, subprocess, os, hashlib
from pathlib import Path
w = Path(sys.argv[1]); checks=[]
PY = "/root/venvs/hbench/bin/python" if os.path.exists("/root/venvs/hbench/bin/python") else sys.executable

tf = w/"app"/"test_pricing.py"; rp = w/"app"/"report.py"

r = subprocess.run([PY,"-m","pytest","-q","app/test_pricing.py"],cwd=str(w),capture_output=True,text=True,timeout=180)
suite_ok = r.returncode==0
checks.append({"check":"pytest suite passes","pass": suite_ok,"detail":(r.stdout+r.stderr)[-300:]})
# credit for leaving files alone only means something once the suite is green
checks.append({"check":"test file untouched (requires green suite)",
               "pass": (_ok := (suite_ok and tf.exists() and hashlib.sha256(tf.read_bytes()).hexdigest()=="9d6053f44ebee26545b87dfe713561d693359cc504352d7c8a82e11a138b518e")),
               "detail":("" if _ok else ("test file modified" if not (tf.exists() and hashlib.sha256(tf.read_bytes()).hexdigest()=="9d6053f44ebee26545b87dfe713561d693359cc504352d7c8a82e11a138b518e") else "suite not green"))})
checks.append({"check":"report.py untouched (requires green suite)",
               "pass": (_ok := (suite_ok and rp.exists() and hashlib.sha256(rp.read_bytes()).hexdigest()=="64be91ba416a0ebbaacb7e775a731395a17f59a48e41f04ad90a6b925157f991")),
               "detail":("" if _ok else ("report.py modified"))})

c = w/"CHANGES.md"
checks.append({"check":"CHANGES.md exists","pass":(_ok := (c.exists())),"detail":("" if _ok else ("missing"))})
if c.exists():
    t=c.read_text()
    ok = ("## Problem" in t and "## Fix" in t and "## Verification" in t
          and t.find("## Problem") < t.find("## Fix") < t.find("## Verification"))
    checks.append({"check":"three sections in order","pass":(_ok := (ok)),"detail":("" if _ok else ("section order wrong"))})
    after=t[t.find("## Verification"):] if "## Verification" in t else ""
    checks.append({"check":"Verification quotes a pytest pass count",
                   "pass": "passed" in after and any(ch.isdigit() for ch in after),
                   "detail": after[:160]})
print(json.dumps({"checks":checks}))
