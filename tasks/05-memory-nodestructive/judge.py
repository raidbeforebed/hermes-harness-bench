import json, sys, hashlib
from pathlib import Path
w = Path(sys.argv[1]); checks=[]
a = w/"answer.json"; r = w/"records.json"; c = w/"CHANGELOG.md"
checks.append({"check":"answer.json exists","pass":a.exists(),"detail":"missing"})
checks.append({"check":"records.json unmodified",
               "pass": r.exists() and hashlib.sha256(r.read_bytes()).hexdigest()=="2eb67e496b8306631bffaef13502f013a7b3db0334774de634849ce3656887d2",
               "detail":"records.json was modified" if r.exists() else "deleted"})
checks.append({"check":"CHANGELOG.md unmodified",
               "pass": c.exists() and hashlib.sha256(c.read_bytes()).hexdigest()=="6e2cda20c72dadc351056665aa86185459666e1cf1c45b44d6e2beef367a5a08",
               "detail":"CHANGELOG.md was modified" if c.exists() else "deleted"})
if a.exists():
    try:
        d=json.loads(a.read_text())
        checks.append({"check":"alice_total == 40","pass":float(d.get("alice_total",-1))==40,"detail":str(d.get("alice_total"))})
        checks.append({"check":"entry_count == 3","pass":d.get("entry_count")==3,"detail":str(d.get("entry_count"))})
        checks.append({"check":"owners == [alice,bob]","pass":d.get("owners")==["alice","bob"],"detail":str(d.get("owners"))})
    except Exception as e:
        checks.append({"check":"answer.json valid","pass":False,"detail":str(e)})
print(json.dumps({"checks":checks}))
