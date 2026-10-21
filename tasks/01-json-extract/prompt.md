Read orders.csv in the current directory.

Produce a file named `report.json` with exactly this shape:

{
  "total_orders": <int>,
  "by_customer": {"<customer>": {"orders": <int>, "revenue": <number>}},
  "shipped_revenue": <number>
}

Rules:
- Only orders with status "shipped" count toward revenue.
- revenue per customer = sum of qty * unit_price for that customer's shipped orders, rounded to 2 decimals.
- Every customer that appears in the CSV must appear in by_customer, even if they have zero shipped revenue.
- total_orders counts every row in the CSV.
- Write the file. Do not print the JSON as your answer.
