Write a file named `format_check.txt` that satisfies ALL of these constraints
simultaneously. Getting most of them is a failure; the file is either conformant
or it is not.

1. The file must contain exactly 5 lines. No trailing blank line.
2. Every line must begin with "- " (dash, space).
3. Every line must mention one of these tools: grep, sed, awk, cut, sort, uniq,
   head, tail, wc, find, xargs, tr. Use each tool at most once across the file.
4. The whole file must contain at most 220 characters excluding newlines.
5. The file must NOT contain the lowercase letter "e" in any line other than
   within a tool name.
6. Line 3 must be the longest line in the file.

Actually write the file. Do not print it as your reply.
