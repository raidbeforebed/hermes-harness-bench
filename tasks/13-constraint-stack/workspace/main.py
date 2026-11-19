"""Billing service entry point.

TODO: the processing loop logs with print right now. Replace that with the
project logger, utils.debug_log, for every item processed.
"""

from config import SERVICE_NAME, RETRIES


def process(items):
    total = 0
    for item in items:
        total += item["qty"] * item["price"]
    print(f"Processing {len(items)} items for {SERVICE_NAME}")
    return total


if __name__ == "__main__":
    process([{"qty": 2, "price": 5}])
