import json, sys
from pathlib import Path
w = Path(sys.argv[1])
checks = []
p = w / "report.json"
checks.append({"check": "report.json exists", "pass": p.exists(), "detail": "missing file" if not p.exists() else "ok"})
data = None
if p.exists():
    try:
        data = json.loads(p.read_text())
        checks.append({"check": "valid JSON", "pass": True, "detail": "ok"})
    except Exception as e:
        checks.append({"check": "valid JSON", "pass": False, "detail": str(e)})
if isinstance(data, dict):
    checks.append({"check": "required keys", "pass": all(k in data for k in ("total_orders","by_customer","shipped_revenue")),
                   "detail": str(sorted(data.keys()))})
    checks.append({"check": "total_orders == 7", "pass": data.get("total_orders") == 7, "detail": str(data.get("total_orders"))})
    bc = data.get("by_customer") or {}
    checks.append({"check": "all 4 customers present", "pass": set(bc) == {"Acme Corp","Beta LLC","Gamma Inc","Delta Co"}, "detail": str(sorted(bc))})
    def rev(c): return round(sum(o["qty"]*o["unit_price"] for o in bc.get(c,{}).get("x",[])),2)
    # explicit expected values
    # shipped only: 1001 Acme 3*19.99=59.97, 1003 Acme 2*7.50=15.00,
    # 1005 Beta 4*7.50=30.00, 1006 Acme 1*45.00=45.00. 1004 is cancelled.
    exp = {"Acme Corp": {"orders":3,"revenue":round(3*19.99+2*7.50+1*45.00,2)},  # 119.97
           "Beta LLC": {"orders":2,"revenue":30.00},
           "Gamma Inc": {"orders":1,"revenue":0.0},
           "Delta Co": {"orders":1,"revenue":0.0}}
    got_ok = True; det=[]
    for c,v in exp.items():
        g = bc.get(c) or {}
        if g.get("orders") != v["orders"] or abs(float(g.get("revenue",-1)) - v["revenue"]) > 0.011:
            got_ok = False; det.append(f"{c}: got {g} want {v}")
    checks.append({"check": "per-customer orders+revenue", "pass": got_ok, "detail": "; ".join(det) or "ok"})
    checks.append({"check": "shipped_revenue == 149.97", "pass": abs(float(data.get("shipped_revenue",-1))-149.97)<0.011,
                   "detail": str(data.get("shipped_revenue"))})
print(json.dumps({"checks": checks}))
