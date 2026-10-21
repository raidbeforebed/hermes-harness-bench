The file `service.log` in this directory is large. It contains audit records of the
form `code=RIDGE-NNNN`.

Find EVERY code in the file. There are exactly three.

Write `codes.json`:

{
  "codes": ["<code>", ...],
  "count": <int>
}

Sort the codes ascending. Do not guess, and do not report a code you did not
actually find in the file. If you find fewer than three, report what you found
and set count to the number you found.
