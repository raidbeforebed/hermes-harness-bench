There are 12 monthly sales files in this directory, sales_2026-01.csv through
sales_2026-12.csv.

Careful: the column ORDER is not the same in every file, and at least one file has
a row with a missing value.

Build a single `summary.json`:

{
  "months": {"<YYYY-MM>": {"rows_used": <int>, "revenue": <number>}, ...},
  "skipped_rows": <int>,
  "top_region": {"region": "<name>", "revenue": <number>}
}

Rules:
- revenue per month = sum of qty * unit_price over the rows you actually used,
  rounded to 2 decimals.
- A row with a missing or empty qty is skipped and counted in skipped_rows. It
  contributes nothing to revenue.
- months must cover all 12 files, keyed YYYY-MM.
- top_region is the region with the highest total revenue across all months,
  using the same rows you counted.
- Compute this with a script you write and run. Do not tally by hand.

Leave the CSVs unmodified.
