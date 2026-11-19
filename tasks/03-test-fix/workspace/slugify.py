"""Turn arbitrary strings into URL slugs."""


def slugify(text):
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in " -_":
            out.append("-")
    # Collapse runs of separators and strip leading/trailing separators.
    result = "".join(out).strip("-").replace("--", "-").replace("  ", " ")
    return result
