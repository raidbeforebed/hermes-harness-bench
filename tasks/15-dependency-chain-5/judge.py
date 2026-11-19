"""Grader for 15-dependency-chain-5.

Capability: propagating a single change transitively across a five-hop import
chain (a -> b -> c -> d -> e). The deepest function changes its return type; the
task is to update every caller's annotation so the chain stays type-consistent.
A small model typically changes the bottom module and leaves the callers stale
("type drift"), which runs fine at runtime but is wrong.

Two independent checks stand in for the SPEC's "type checker plus unit tests":
  * a runtime check imports the chain and asserts top() returns the string "42";
  * a static check parses each module and asserts every function's return
    annotation is `str` (this is what catches drift the runtime cannot see).
The workspace unit test must also still pass and must be left unedited.

Every check is gated on the runtime achievement, so an empty workspace scores 0.
Deterministic, code-based, no judge model.
"""
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

w = Path(sys.argv[1])
checks = []

# SHA-256 of the pristine test_chain.py: proves the unit test was not weakened.
PRISTINE_TEST_SHA256 = "13d04e21ea63f6b74bc470cb4efc84fbd1c006b863b409aef283ad29a513d16a"

# 1. Runtime: import the chain and confirm the top returns the string "42".
runtime_code = (
    "import a\n"
    "v = a.top()\n"
    "print('OK' if (v == '42' and isinstance(v, str)) else 'NO')\n"
)
try:
    r = subprocess.run([sys.executable, "-c", runtime_code], cwd=str(w),
                       capture_output=True, text=True, timeout=60)
    runtime_ok = (r.returncode == 0 and "OK" in r.stdout)
    runtime_detail = r.stdout.strip() + ((" | " + r.stderr.strip()[-160:]) if r.stderr.strip() else "")
except Exception as exc:  # noqa: BLE001
    runtime_ok = False
    runtime_detail = "runtime check errored: " + str(exc)[:160]

# 2. Static: every function's return annotation must be `str`.
def return_anno(path: Path, func: str):
    try:
        tree = ast.parse(path.read_text())
    except Exception:  # noqa: BLE001
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func:
            if node.returns is None:
                return None
            try:
                return ast.unparse(node.returns)
            except Exception:  # noqa: BLE001
                return "<unparseable>"
    return None


annos = {
    "e.load_value": return_anno(w / "e.py", "load_value"),
    "d.resolve": return_anno(w / "d.py", "resolve"),
    "c.compute": return_anno(w / "c.py", "compute"),
    "b.build": return_anno(w / "b.py", "build"),
    "a.top": return_anno(w / "a.py", "top"),
}
all_str = all(v == "str" for v in annos.values())

checks.append({
    "check": "chain runs and top() returns the string '42'",
    "pass": runtime_ok,
    "detail": "ok" if runtime_ok else runtime_detail,
})

checks.append({
    "check": "all five functions annotated -> str (no type drift)",
    "pass": runtime_ok and all_str,
    "detail": "annotations=" + str(annos),
})

test_p = w / "test_chain.py"
test_unchanged = test_p.exists() and hashlib.sha256(test_p.read_bytes()).hexdigest() == PRISTINE_TEST_SHA256
checks.append({
    "check": "test_chain.py left unedited",
    "pass": runtime_ok and test_unchanged,
    "detail": ("ok" if test_unchanged else "test_chain.py edited or missing"),
})

# 4. The workspace unit test itself must pass, run against a clean interpreter.
test_code = "import test_chain\nraises = test_chain.test_top_is_string_42()\n"
try:
    tr = subprocess.run([sys.executable, "-c", test_code], cwd=str(w),
                        capture_output=True, text=True, timeout=60)
    test_ok = (tr.returncode == 0)
    test_detail = (tr.stdout.strip() + " " + tr.stderr.strip())[-200:]
except Exception as exc:  # noqa: BLE001
    test_ok = False
    test_detail = "test run errored: " + str(exc)[:160]

checks.append({
    "check": "workspace unit test passes",
    "pass": runtime_ok and test_ok,
    "detail": "ok" if test_ok else test_detail,
})

print(json.dumps({"checks": checks}))
