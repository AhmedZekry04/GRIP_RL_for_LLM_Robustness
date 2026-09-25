"""Per-problem, per-stage seeded RNG (instead of GAP's single global
`random.seed(42)`), so any one problem can be regenerated in isolation."""

import hashlib
import random


def seeded_rng(*parts):
    """e.g. seeded_rng(problem_id, "GS") -> random.Random seeded from SHA-256 of the parts."""
    key = '|'.join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode('utf-8')).hexdigest()
    return random.Random(int(digest[:16], 16))
