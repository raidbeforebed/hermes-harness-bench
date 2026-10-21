import json, sys, csv, glob, os, hashlib
from pathlib import Path
w = Path(sys.argv[1]); checks=[]

# ground truth, computed independently of what the model did
rev={}; region_rev={}; skipped=0; rowcount={}
for f in sorted(glob.glob(str(w/"sales_*.csv"))):
    mo=Path(f).stem.replace("sales_","")
    used=0
    with open(f) as fh:
        for row in csv.DictReader(fh):
            q=(row.get("qty") or "").strip()
            if q=="" or q is None:
                skipped+=1; continue
            try: qty=int(q)
            except Exception: skipped+=1; continue
            price=float(row["unit_price"])
            rev[mo]=round(rev.get(mo,0)+qty*price,2)
            region_rev[row["region"]]=round(region_rev.get(row["region"],0)+qty*price,2)
            used+=1
    rowcount[mo]=used
top=max(region_rev.items(), key=lambda kv: kv[1]) if region_rev else ("",0)

p=w/"summary.json"
checks.append({"check":"summary.json exists","pass":p.exists(),"detail":"missing"})
if p.exists():
    try: d=json.loads(p.read_text())
    except Exception as e:
        d=None; checks.append({"check":"valid JSON","pass":False,"detail":str(e)})
    if isinstance(d,dict):
        m=d.get("months") or {}
        checks.append({"check":"all 12 months present","pass":len(m)==12,"detail":"got %d"%len(m)})
        bad=[]
        for k,v in rev.items():
            g=m.get(k) or {}
            if abs(float(g.get("revenue",-1))-v)>0.011: bad.append("%s got %s want %s"%(k,g.get("revenue"),v))
        checks.append({"check":"per-month revenue correct","pass":not bad,"detail":"; ".join(bad)[:300] or "ok"})
        rowsok=all((m.get(k) or {}).get("rows_used")==rowcount[k] for k in rev)
        checks.append({"check":"rows_used correct per month","pass":rowsok,
                       "detail":str([ (k,(m.get(k) or {}).get("rows_used"),rowcount[k]) for k in list(rev)[:3]])})
        checks.append({"check":"skipped_rows correct","pass":d.get("skipped_rows")==skipped,
                       "detail":"got %s want %s"%(d.get("skipped_rows"),skipped)})
        tr=d.get("top_region") or {}
        checks.append({"check":"top_region correct","pass": tr.get("region")==top[0] and abs(float(tr.get("revenue",-1))-top[1])<0.011,
                       "detail":"got %s want %s"%(tr,top)})
print(json.dumps({"checks":checks}))
