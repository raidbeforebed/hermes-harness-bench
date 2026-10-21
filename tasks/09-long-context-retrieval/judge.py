import json, sys, re
from pathlib import Path
w = Path(sys.argv[1]); checks=[]
p = w/"codes.json"
log = w/"service.log"
truth = sorted(set(re.findall(r"RIDGE-\d{4}", log.read_text()))) if log.exists() else []
log_ok = len(truth)==3
p0 = w/"codes.json"
checks.append({"check":"log intact (3 codes)","pass": log_ok and p0.exists(),"detail":("log damaged" if not log_ok else "no codes.json written")})
checks.append({"check":"codes.json exists","pass": p.exists(),"detail":"missing"})
if p.exists():
    try: d=json.loads(p.read_text())
    except Exception as e:
        d=None; checks.append({"check":"valid JSON","pass":False,"detail":str(e)})
    if isinstance(d,dict):
        got=sorted(d.get("codes") or [])
        checks.append({"check":"all three codes found","pass": got==truth,"detail":"got %s want %s"%(got,truth)})
        checks.append({"check":"count consistent","pass": d.get("count")==len(got),"detail":"count=%s vs %s"%(d.get("count"),len(got))})
        checks.append({"check":"no invented codes","pass": set(got)<=set(truth),"detail":str(sorted(set(got)-set(truth)))})
print(json.dumps({"checks":checks}))
