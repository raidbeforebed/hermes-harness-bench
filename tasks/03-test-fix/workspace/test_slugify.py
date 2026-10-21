from slugify import slugify


def test_basic():
    assert slugify("Hello World") == "hello-world"


def test_collapse_runs():
    assert slugify("Hello   World") == "hello-world"


def test_strip_edges():
    assert slugify("  Hello World  ") == "hello-world"


def test_underscores():
    assert slugify("hello_world") == "hello-world"


def test_punctuation():
    assert slugify("Hello, World!") == "hello-world"


def test_empty():
    assert slugify("") == ""
