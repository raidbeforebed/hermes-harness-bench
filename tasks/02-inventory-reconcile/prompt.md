Two inventory exports are in the working directory: system_a.csv and system_b.csv.
Each has columns sku,on_hand.

Write `reconcile.json` describing how they disagree. Use exactly this shape:

{
  "only_in_a": ["<sku>", ...],
  "only_in_b": ["<sku>", ...],
  "quantity_mismatch": [{"sku": "<sku>", "a": <int>, "b": <int>}, ...]
}

Sort every list by sku ascending. A sku present in both with the same on_hand is
not a mismatch and must not appear anywhere. Do not infer, guess or fill in any
sku that is not present in the files. Write the file.
