`haystack.log` in the current directory is a large generated application log (~60K
tokens). Exactly one line in it carries a fatal error code of the form
`ERR-XXXX`, where `XXXX` is four uppercase hexadecimal digits.

Find that one error code and write it to `needle.txt`, with nothing else in the
file, just the code (a trailing newline is fine).

Do not guess. Do not report a code you did not actually find in the file. If you
cannot find it, say so rather than inventing one.
