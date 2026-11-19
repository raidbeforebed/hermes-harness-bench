"""Top-level entry point."""
from b import build


def top() -> int:
    return build()


if __name__ == "__main__":
    print(top())
