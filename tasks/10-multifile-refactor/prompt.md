This is a small Python project. Its test suite has a real failure caused by a
rounding rule that finance requires to be half-up.

Do all of the following:
1. Make the whole test suite pass by fixing `app/pricing.py`. Do not weaken or
   edit `app/test_pricing.py`.
2. Keep `app/report.py` working without editing it (it only consumes the API).
3. Write `CHANGES.md` with exactly these three sections, in this order, each
   headed with the literal text shown:

## Problem
## Fix
## Verification

Under "Problem" state in one sentence what was wrong. Under "Fix" state in one
sentence what you changed. Under "Verification" paste the last line of the
pytest summary, which must contain the pass count.

Run the tests to verify. Do not claim success without having run them.
