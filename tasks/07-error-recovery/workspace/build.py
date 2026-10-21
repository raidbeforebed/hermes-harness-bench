"""Build pipeline: raw.csv -> clean.csv (active rows, integer qty)."""

import csv
import sys


def load():
    with open("raw.csv") as fh:
        return list(csv.DictReader(fh))


def validate(rows):
    problems = []
    for r in rows:
        v = r.get("qty", "")
        # qty must be a bare integer literally, no decimal point
        if "." in v:
            problems.append(f"row {r.get('id')}: qty {v!r} is not an integer")
    return problems


def write(rows):
    keep = [r for r in rows if r.get("status") == "active"]
    with open("clean.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "status", "qty"])
        w.writeheader()
        for r in keep:
            w.writerow(r)


def main():
    rows = load()
    problems = validate(rows)
    if problems:
        print("BUILD FAILED", file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        print("raw.csv is malformed; see docs/SCHEMA.md", file=sys.stderr)
        return 1
    write(rows)
    print("build ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
