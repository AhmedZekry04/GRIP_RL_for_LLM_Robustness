"""
Named-quantity filters for GS/DLM.

Renaming a free variable (`x`, `n`) is surface-neutral; renaming a token
whose meaning comes from an outside convention (`radius`, `mean`,
`determinant`) or that carries a value (`(0, 0)`, `x = 0`, `10 cm`) changes
the problem itself. Groups whose actual GS or DLM rename map contains such a
token are dropped in two passes:

  v1 — any word of the token is in NAMED_QUANTITY_WORDS.
  v2 — the token is in CONFIRMED_DROPS, or matches one of the value
       patterns. CONFIRMED_DROPS is the outcome of manually reviewing every
       candidate token surfaced by `is_candidate` (6,757 tokens).
"""

import re

NAMED_QUANTITY_WORDS = {
    'radius', 'diameter', 'circumference', 'area', 'perimeter', 'volume',
    'mean', 'median', 'mode', 'variance', 'sum', 'product', 'average',
    'height', 'width', 'length', 'angle', 'slope', 'side', 'sides', 'base',
    'apothem', 'hypotenuse', 'circumradius', 'inradius', 'diagonal',
    'altitude', 'vertex', 'vertices', 'midpoint', 'edge', 'edges', 'face',
    'faces', 'distance', 'speed', 'rate', 'time', 'probability',
    'expectation', 'frequency',
}

# Exact tokens (case-insensitive) confirmed as named quantities in review.
CONFIRMED_DROPS = {
    # Named mathematical objects
    'matrix', 'matrices', 'determinant', 'trace', 'transition matrix',
    'characteristic polynomial', 'galois group', 'dimension',
    'ring homomorphisms', 'automorphisms', 'arithmetic sequence',
    'gf(2)', 'gf(64)',
    # Named geometric shapes and objects
    'square', 'squares', 'cube', 'cubes', 'hexagon', 'regular hexagon',
    'octagon', 'circle', 'circles', 'triangle', 'triangles', 'sphere',
    'unit sphere', 'unit disk', 'unit ball', 'plane', 'xy-plane',
    'paraboloid', 'cylinder', 'first octant', 'origin', '3x3 matrices',
    # Probability / statistics
    'expected value', 'expected number of coin tosses',
    'expected number of steps', 'expected number of coin flips',
    # Number theory
    'remainder', 'least common multiple', 'greatest common divisor',
    'primitive root', 'units digit', 'tens digit', 'first digit',
    'last digit', 'last two digits',
    # Named sets
    'positive integers', 'natural numbers', 'two positive integers',
    # Other named concepts
    'genus', 'temperature',
    # Time units
    'minutes', 'hours', 'seconds', 'days',
    # Number words mis-tagged as variables
    'five', 'four', 'three', 'eight', 'nine', 'seven',
}

# Value-bearing tokens mis-tagged as variables.
_VALUE_PATTERNS = [
    # Numeric coordinate tuples: (0, 0), (-1, 1), (1, 0, 0)
    ('coordinate_value', re.compile(r'^\(\s*-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?)+\s*\)$')),
    # Variable equals a literal: x=0, z = 1, x+y+z=1, z = i
    ('equation_value', re.compile(r'^[a-zA-Z][a-zA-Z0-9+\-\s]*\s*=\s*[-\di\s./]+$')),
    # Number with a unit: 20 minutes, 10 cm
    ('measurement_value', re.compile(r'^\d+\s*(?:minutes?|hours?|seconds?|days?|cm|mm|inches?|inch|ft|kg|g)$', re.IGNORECASE)),
    # Clock time: 2:00, 14:30
    ('clock_value', re.compile(r'^\d{1,2}:\d{2}$')),
]

# Candidate heuristic used to build the v2 review list.
_LATEX_CHARS = set('\\^{}_$[]|')
_SAFE_ABBREVIATIONS = {'sin', 'cos', 'tan', 'log', 'exp', 'max', 'min', 'sup', 'inf', 'lim', 'iff', 'gcd',
                       'lcm', 'mod', 'det', 'arg', 'sgn', 'var', 'cov', 'sec', 'csc', 'cot', 'abs'}


def v1_match(token):
    """The NAMED_QUANTITY_WORDS word contained in `token`, or None."""
    for word in token.lower().replace('-', ' ').split():
        if word in NAMED_QUANTITY_WORDS:
            return word
    return None


def v2_match(token):
    """The drop category of `token`, or None."""
    t = token.strip()
    if t.lower() in CONFIRMED_DROPS:
        return 'named_quantity'
    for category, pattern in _VALUE_PATTERNS:
        if pattern.match(t):
            return category
    return None


def is_candidate(token):
    """English word or phrase that might be a named quantity: no LaTeX, not
    purely numeric, not a single character, not a standard abbreviation,
    and not a short alphanumeric symbol such as `dx` or `pi`."""
    t = token.strip()
    if not t or len(t) == 1 or any(c in _LATEX_CHARS for c in t):
        return False
    if re.fullmatch(r'[\d\.\-\+\*/=<>(),%]+', t):
        return False
    if t.lower() in _SAFE_ABBREVIATIONS:
        return False
    return not (re.fullmatch(r'[a-zA-Z0-9]+', t) and len(t) <= 3)


def renamed_tokens(gid, gs, dlm):
    """(stage, token) for every token actually renamed in the group's GS and DLM maps."""
    pairs = []
    for stage, results in (('gs', gs), ('dlm', dlm)):
        entry = results.get(gid) or {}
        pairs.extend((stage, token) for token in entry.get('map', {}))
    return pairs
