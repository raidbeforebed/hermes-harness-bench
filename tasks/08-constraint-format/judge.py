import json, sys
from pathlib import Path
w = Path(sys.argv[1]); checks=[]
p = w/"format_check.txt"
checks.append({"check":"format_check.txt exists","pass":p.exists(),"detail":"missing"})
if p.exists():
    raw = p.read_text()
    lines = raw.split("\n")
    # strip a single trailing newline as "no trailing blank line"
    if lines and lines[-1]=="": lines=lines[:-1]
    checks.append({"check":"exactly 5 lines", "pass": len(lines)==5, "detail": f"{len(lines)} lines"})
    checks.append({"check":"every line starts with '- '", "pass": all(l.startswith("- ") for l in lines),
                   "detail": str([l[:12] for l in lines if not l.startswith("- ")])})
    tools=["grep","sed","awk","cut","sort","uniq","head","tail","wc","find","xargs","tr"]
    found=[]; 
    for l in lines:
        hit=[t for t in tools if t in l.lower()]
        found.append(hit)
    used=[h[0] for h in found if h]
    checks.append({"check":"each line names >=1 tool","pass": len(used)==5, "detail": f"{len(used)} of 5"})
    checks.append({"check":"no tool used twice","pass": len(used)==len(set(used)), "detail": str(used)})
    chars=sum(len(l) for l in lines)
    checks.append({"check":"<=220 chars excluding newlines","pass": chars<=220, "detail": f"{chars} chars"})
    # rule 5: no lowercase 'e' outside tool names
    ok=True; det=[]
    for l in lines:
        stripped=l
        for t in tools: stripped=stripped.replace(t,"")
        if "e" in stripped: ok=False; det.append(l[:60])
    checks.append({"check":"no stray lowercase 'e'","pass": ok, "detail": "; ".join(det)[:200] or "ok"})
    if len(lines)==5:
        lens=[len(l) for l in lines]
        checks.append({"check":"line 3 is longest (strictly)",
                       "pass": lens[2]==max(lens) and lens.count(max(lens))==1,
                       "detail": str(lens)})
    else:
        checks.append({"check":"line 3 is longest (strictly)","pass":False,"detail":"line count wrong"})
print(json.dumps({"checks":checks}))
