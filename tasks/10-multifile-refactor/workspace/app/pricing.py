"""Pricing rules."""


def subtotal(items):
    return sum(i["price"] * i["qty"] for i in items)


def apply_tax(amount, rate):
    return round(amount * (1 + rate), 2)


def total(items, rate):
    return apply_tax(subtotal(items), rate)
