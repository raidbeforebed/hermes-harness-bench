You maintain a plain-text note file named `notes.md` in the current directory.
It does not exist yet; you create and edit it by performing this exact sequence
of operations, in order:

1. CREATE `notes.md` with a single header line: `# Notes`
2. APPEND the line `- alpha`
3. APPEND the line `- bravo`
4. REVERT the file to the state it was in immediately AFTER step 1. In other
   words: the header only, with both appended lines removed.
5. APPEND the line `- charlie`

After all five operations, leave `notes.md` on disk with its final content.

The final file must be exactly these three lines, in this order, ending with a
trailing newline:

    # Notes
    - alpha
    - charlie

Do not leave the file in any other state, and do not print the file as your
reply.
