"""
LaTeX-aware identifier substitution shared by GS and DLM.

Boundary rule adapted from GAP's `mini_gap_math.py::replace_in_math`
(`(?<![a-zA-Z\\])tok(?![a-zA-Z])`; https://github.com/YurenHao0426/GAP,
CC BY 4.0), with three changes:

1. The lookahead also blocks a following digit, so `x` never matches inside
   a distinct identifier such as `x2`.
2. A token directly preceded by a digit gets `\\cdot ` inserted
   (`2x` -> `2 \\cdot lr4dr`, not `2lr4dr`).
3. A token directly followed by `^` or `_` is brace-wrapped
   (`x^2` -> `{lr4dr}^2`); otherwise the exponent would bind to the last
   character of the replacement only.

All tokens are replaced in a single regex pass over the original text
(longest token first), so no replacement can be re-matched by a later token.

Known limitation inherited from GAP: adjacent single-letter variables with
implicit multiplication (`xy`) never match, so they are left un-renamed.
"""

import re


def _boundary_pattern(alternation):
    return re.compile(rf'(?<![a-zA-Z\\])(?:{alternation})(?![a-zA-Z0-9])')


def apply_substitution(text, token_map):
    """Replace every token in `token_map` ({old: new}) in one pass over `text`."""
    if not token_map:
        return text

    sorted_tokens = sorted(token_map, key=len, reverse=True)
    pattern = _boundary_pattern('|'.join(re.escape(t) for t in sorted_tokens))

    def replacer(m):
        replacement = token_map[m.group(0)]
        prefix = ' \\cdot ' if m.start() > 0 and text[m.start() - 1].isdigit() else ''
        following = text[m.end()] if m.end() < len(text) else ''
        body = f'{{{replacement}}}' if following in ('^', '_') else replacement
        return f'{prefix}{body}'

    return pattern.sub(replacer, text)


def token_occurs(text, token):
    """True if apply_substitution would match `token` at least once in `text`."""
    return _boundary_pattern(re.escape(token)).search(text) is not None
