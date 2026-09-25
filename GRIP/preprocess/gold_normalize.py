import re


def _find_boxed_spans(s):
    """[(start, end, content), ...] for every \\boxed{...} in s, brace-matched
    so nested braces survive. end is the index of the closing '}'."""
    spans = []
    search_from = 0
    marker = "\\boxed{"
    while True:
        idx = s.find(marker, search_from)
        if idx == -1:
            break
        brace_start = idx + len(marker) - 1  # index of the opening '{'
        depth = 0
        end = None
        i = brace_start
        while i < len(s):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
            i += 1
        if end is None:
            search_from = idx + len(marker)  # unmatched brace, skip past it
            continue
        spans.append((idx, end, s[brace_start + 1:end]))
        search_from = end + 1
    return spans


def _unwrap_last_boxed(s):
    spans = _find_boxed_spans(s)
    return spans[-1][2] if spans else s


def _strip_math_delims(s):
    if len(s) >= 2 and s.startswith("$") and s.endswith("$"):
        return s[1:-1].strip()
    if len(s) >= 4 and s.startswith("\\(") and s.endswith("\\)"):
        return s[2:-2].strip()
    return s


def _strip_trailing_punct(s):
    if s and s[-1] in ".,":
        return s[:-1].strip()
    return s


def normalize_gold(raw):
    """
    Normalize one raw gold value into a list of acceptable answer strings.

    Order: stringify+strip -> unwrap last \\boxed{} (brace-matched) -> strip
    surrounding $...$ / \\(...\\) -> strip one trailing '.'/',' -> split on
    r'\\s+or\\s+' (case-insensitive).

    sympify("0 or 4") does not raise - "or" is a Python boolean operator, so
    it silently evaluates to 4 with no error anywhere. The split step exists
    to catch this before it reaches sympify/math_verify.

    Returns (answers, flagged). answers is never empty; flagged is True when
    normalization collapsed to nothing and the raw string was kept instead.
    """
    s = str(raw).strip()
    unboxed = _unwrap_last_boxed(s)
    delimited = _strip_math_delims(unboxed)
    punct_stripped = _strip_trailing_punct(delimited)

    parts = re.split(r"\s+or\s+", punct_stripped, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]

    if not parts:
        return [s], True
    return parts, False
