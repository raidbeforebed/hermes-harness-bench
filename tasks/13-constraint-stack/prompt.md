This project has three modules: `main.py`, `utils.py`, `config.py`.

Add logging to the processing loop in `main.py`. `utils.py` already defines the
project logger `debug_log(message)`; call it as `utils.debug_log(...)`.

Every one of these constraints must hold at the same time. Getting most of them
is a failure.

1. You MUST call `utils.debug_log(...)` to log the work (one call per processed
   item is expected). Import the `utils` module so the call is written
   `utils.debug_log(...)`.
2. You MUST NOT import the standard `logging` module: no `import logging`, no
   `from logging ...`.
3. You MUST NOT use `print(...)` anywhere in `main.py`. Remove the existing print
   statement and log through `utils.debug_log` instead.
4. You MUST NOT edit `config.py` in any way.

Edit only `main.py`. Do not print the code as your reply; make the edit to the
file.
