"""Utility helpers for the billing service.

`debug_log` is the project's structured logger. Call it as `utils.debug_log(msg)`.
It appends a timestamped line to logs/app.log next to this module.
"""
import os
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_PATH = os.path.join(LOG_DIR, "app.log")


def debug_log(message):
    os.makedirs(LOG_DIR, exist_ok=True)
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    with open(LOG_PATH, "a") as fh:
        fh.write(line + "\n")
    return line
