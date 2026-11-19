This is a five-module Python package with a strict import chain:

    a.py -> b.py -> c.py -> d.py -> e.py

Each function simply returns the result of the one below it, so the return type
is the same all the way up. Right now every function returns `int`.

Change the return type at the BOTTOM of the chain and propagate the change all
the way to the top:

1. In `e.py`, `load_value()` must return the STRING `"42"` instead of the integer
   42. Change its return annotation to `str` and return `"42"`.
2. Every caller up the chain (`d.resolve`, `c.compute`, `b.build`, `a.top`) must
   have its return annotation changed to `str` so the whole chain is
   type-consistent. The functions still just pass the value through.

Do NOT edit `test_chain.py`. It must pass unchanged.

Run the tests to verify. Do not claim success without having run them.
