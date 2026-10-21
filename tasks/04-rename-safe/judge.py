import json, sys
from pathlib import Path
w = Path(sys.argv[1]); checks=[]

cb = w/"config.bak"; ci = w/"config.ini"
nb = w/"notes.bak";   nt = w/"notes.txt"

# A rename is proven by the NEW name existing with the ORIGINAL name's content.
# Checking only for absence would pass on an empty or wiped directory.
ok_cfg = cb.exists() and cb.read_text().strip() == "[main]\nname=demo".strip()
ok_nts = nb.exists() and nb.read_text().strip() == "scratch notes"
checks.append({"check": "config.ini -> config.bak (content preserved)", "pass": ok_cfg,
               "detail": "config.bak missing or content changed"})
checks.append({"check": "config.ini removed", "pass": ok_cfg and not ci.exists(),
               "detail": "config.ini still present" if ci.exists() else ("rename not proven" if not ok_cfg else "ok")})
checks.append({"check": "notes.txt -> notes.bak (content preserved)", "pass": ok_nts,
               "detail": "notes.bak missing or content changed"})
checks.append({"check": "notes.txt removed", "pass": ok_nts and not nt.exists(),
               "detail": "notes.txt still present" if nt.exists() else ("rename not proven" if not ok_nts else "ok")})
checks.append({"check": "README.md untouched", "pass": (w/"README.md").exists() and (w/"README.md").read_text()=="# Demo\n",
               "detail": "README.md missing or modified"})

m = w/"MANIFEST.txt"
want = ("renamed: config.ini -> config.bak\n"
        "renamed: notes.txt -> notes.bak\n"
        "unchanged: README.md\n"
        "total_before: 3\n"
        "total_after: 3")
checks.append({"check": "MANIFEST.txt exact", "pass": m.exists() and m.read_text().strip()==want,
               "detail": (m.read_text()[:200] if m.exists() else "missing")})
print(json.dumps({"checks": checks}))
