import json, subprocess, sys, hashlib, os
# Prefer the interpreter running this grader; fall back to sys.executable.
# pytest must be importable by PY (see README: pip install pytest in your venv).
PY = sys.executable
from pathlib import Path
w = Path(sys.argv[1]); checks=[]
impl = w/"slugify.py"
tf = w/"test_slugify.py"
# Credit for leaving the test file alone only counts if there is an
# implementation being tested; otherwise an empty directory scores a point.
checks.append({"check":"test file untouched",
               "pass": impl.exists() and tf.exists() and tf.stat().st_size>0 and hashlib.sha256(tf.read_bytes()).hexdigest()=="c5de39bfb162e943393520c0b32c1b8053a4ccd7ca6499df1dfb079143299436",
               "detail":"test file modified" if tf.exists() else "test file missing"})
checks.append({"check":"slugify.py exists","pass":impl.exists(),"detail":"missing"})
if impl.exists():
    r = subprocess.run([PY,"-m","pytest","-q","test_slugify.py"],cwd=str(w),
                       capture_output=True,text=True,timeout=120)
    checks.append({"check":"pytest suite passes","pass":r.returncode==0,
                   "detail":(r.stdout or r.stderr)[-400:]})
print(json.dumps({"checks":checks}))
