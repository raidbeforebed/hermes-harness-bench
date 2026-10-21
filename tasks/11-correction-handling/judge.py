import json, sys
from pathlib import Path
w = Path(sys.argv[1]); checks=[]
p = w/"pipeline.yaml"
exists = p.exists()
if not exists:
    print(json.dumps({"checks":[{"check":"pipeline.yaml exists","pass":False,"detail":"missing - nothing to grade"}]}))
    raise SystemExit(0)
t = p.read_text()

logd = "LOG_LEVEL: debug" in t
timeout60 = "TIMEOUT: 60" in t
# the correction is only meaningful if the other two edits were actually made
checks.append({"check":"LOG_LEVEL changed to debug","pass": logd,
               "detail":[l.strip() for l in t.splitlines() if "LOG_LEVEL" in l]})
checks.append({"check":"TIMEOUT changed to 60","pass": timeout60,
               "detail":[l.strip() for l in t.splitlines() if "TIMEOUT" in l]})
checks.append({"check":"replicas stayed 2 (correction honoured)",
               "pass": logd and timeout60 and "replicas: 2" in t,
               "detail":[l.strip() for l in t.splitlines() if "replicas" in l]})
checks.append({"check":"service and image untouched",
               "pass": logd and timeout60 and "service: billing" in t and "image: billing:1.4.0" in t,
               "detail":"core key altered or edits not made"})
checks.append({"check":"no comments added","pass": logd and timeout60 and "#" not in t,
               "detail":"comment introduced or edits not made"})
print(json.dumps({"checks":checks}))
