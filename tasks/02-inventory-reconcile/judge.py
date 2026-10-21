import json, sys
from pathlib import Path
w = Path(sys.argv[1]); checks=[]
p = w/"reconcile.json"
checks.append({"check":"reconcile.json exists","pass":p.exists(),"detail":"missing" if not p.exists() else "ok"})
d=None
if p.exists():
    try: d=json.loads(p.read_text()); checks.append({"check":"valid JSON","pass":True,"detail":"ok"})
    except Exception as e: checks.append({"check":"valid JSON","pass":False,"detail":str(e)})
if isinstance(d,dict):
    checks.append({"check":"only_in_a == ['A-300']","pass":d.get("only_in_a")==["A-300"],"detail":str(d.get("only_in_a"))})
    checks.append({"check":"only_in_b == ['C-900']","pass":d.get("only_in_b")==["C-900"],"detail":str(d.get("only_in_b"))})
    mism = d.get("quantity_mismatch") or []
    norm = sorted([{"sku":m.get("sku"),"a":m.get("a"),"b":m.get("b")} for m in mism], key=lambda x:str(x.get("sku")))
    checks.append({"check":"one mismatch only (A-200 4 vs 6)",
                   "pass": norm==[{"sku":"A-200","a":4,"b":6}],"detail":str(norm)})
    # grounding: nothing invented
    known={"A-100","A-200","A-300","B-100","B-200","C-900"}
    seen=set(d.get("only_in_a") or [])|set(d.get("only_in_b") or [])|{m.get("sku") for m in mism}
    checks.append({"check":"no invented SKUs","pass":seen<=known,"detail":str(sorted(seen-known))})
print(json.dumps({"checks":checks}))
