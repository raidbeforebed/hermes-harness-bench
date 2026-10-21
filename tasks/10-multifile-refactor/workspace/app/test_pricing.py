from app.pricing import apply_tax, subtotal, total


def test_subtotal():
    items = [{"price": 10.0, "qty": 2}, {"price": 5.5, "qty": 3}]
    assert subtotal(items) == 36.5


def test_tax_half_up_boundary():
    # 1.005 at 0% tax: Python's round() gives 1.0 (half-even on the float
    # representation), finance requires 1.01 (half-up).
    assert apply_tax(1.005, 0.0) == 1.01


def test_tax_half_up_boundary_2():
    assert apply_tax(2.675, 0.0) == 2.68


def test_tax_normal():
    assert apply_tax(100.0, 0.08) == 108.0


def test_total():
    items = [{"price": 20.0, "qty": 1}]
    assert total(items, 0.05) == 21.0
