from app.report import summarise
result = summarise([{"id": "o1", "items": [{"price": 20.0, "qty": 1}]}], 0.05)
print("report result:", result)
assert result == [{"id": "o1", "total": 21.0}], f"Unexpected: {result}"
print("OK: report.py works without edits")
