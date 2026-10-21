import json, sys
from pathlib import Path
w = Path(sys.argv[1]); checks=[]
def load(n):
    p=w/n
    if not p.exists(): return None
    try: return json.loads(p.read_text())
    except Exception: return None
s1=load("stage1.json"); s2=load("stage2.json"); f=load("final.json")
checks.append({"check":"stage1.json exists+valid","pass":s1 is not None,"detail":"missing/invalid"})
checks.append({"check":"stage2.json exists+valid","pass":s2 is not None,"detail":"missing/invalid"})
checks.append({"check":"final.json exists+valid","pass":f is not None,"detail":"missing/invalid"})
if isinstance(s1,dict):
    ev=s1.get("events") or []
    checks.append({"check":"stage1 has 10 events","pass":len(ev)==10,"detail":f"got {len(ev)}"})
    checks.append({"check":"stage1 field coverage",
                   "pass": all(all(k in e for k in ("ts","level","service","status","ms")) for e in ev) and len(ev)==10,
                   "detail":"missing fields" if len(ev)==10 else "count wrong"})
if isinstance(s2,dict):
    bs=s2.get("by_service") or {}
    exp={"api":{"count":5,"avg_ms":round((120+15+210+12+140)/5,2)},
         "db":{"count":3,"avg_ms":round((2400+3100+5200)/3,2)},
         "web":{"count":2,"avg_ms":round((95+88)/2,2)}}
    ok=set(bs)==set(exp)
    det=[]
    if ok:
        for k,v in exp.items():
            g=bs.get(k) or {}
            if g.get("count")!=v["count"] or abs(float(g.get("avg_ms",-1))-v["avg_ms"])>0.011:
                ok=False; det.append(f"{k}: got {g} want {v}")
    checks.append({"check":"stage2 by_service correct","pass":ok,"detail":"; ".join(det) or ("keys "+str(sorted(bs)) if det==[] and not ok else "ok")})
if isinstance(f,dict):
    ps=f.get("problem_services") or {}
    exp={"api":{"count":2,"statuses":[429,429]},"db":{"count":3,"statuses":[500,500,503]}}
    ok = set(ps)==set(exp)
    det=[]
    if ok:
        for k,v in exp.items():
            g=ps.get(k) or {}
            if g.get("count")!=v["count"] or sorted(g.get("statuses") or [])!=sorted(v["statuses"]):
                ok=False; det.append(f"{k}: got {g} want {v}")
    else:
        det.append("keys "+str(sorted(ps)))
    checks.append({"check":"final problem_services correct","pass":ok,"detail":"; ".join(det) or "ok"})
print(json.dumps({"checks":checks}))
