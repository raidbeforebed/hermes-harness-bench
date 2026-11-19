"""Unit test: the top of the chain returns the propagated value."""
from a import top


def test_top_is_string_42():
    v = top()
    assert v == "42", f"expected '42', got {v!r}"
    assert isinstance(v, str), f"expected str, got {type(v).__name__}"
