"""
Stage 2 — GS (Garbled String) variant. Deterministic, no API calls.

Every tagged var/param is renamed to a random letter-first alphanumeric
string of length 4-16, in both the question and the answer. Adapted from
GAP's `mini_gap_math.py` (generate_garbled_name / apply_surface_rename), with
LLM-tagged tokens instead of regex-mined ones, a per-problem seeded RNG, and
a collision check against the problem text.

    python -m pipeline.gs
"""

import argparse

from .common import GS_PATH, GS_REJECTED_PATH, PROBLEMS_PATH, TAGS_PATH, write_json
from .merge import load_problems_with_tags
from .seeding import seeded_rng
from .substitution import apply_substitution

_LETTERS = 'abcdefghijklmnopqrstuvwxyz'
_ALPHABET = _LETTERS + '0123456789'
_MAX_RETRIES = 5


def _generate_garbled_name(rng):
    # Letter-first, so a name is never read as a number followed by letters.
    length = rng.randint(4, 16)
    return rng.choice(_LETTERS) + ''.join(rng.choice(_ALPHABET) for _ in range(length - 1))


def build_gs_map(problem_id, tokens, problem_text):
    """Returns {token: garbled_name}. Raises ValueError if a collision-free
    name cannot be found within _MAX_RETRIES."""
    rng = seeded_rng(problem_id, 'GS')
    used_names = set()
    gs_map = {}
    for token in tokens:
        for _ in range(_MAX_RETRIES):
            candidate = _generate_garbled_name(rng)
            if candidate not in used_names and candidate not in problem_text:
                break
        else:
            raise ValueError(f"GS collision for token {token!r} in {problem_id} after {_MAX_RETRIES} retries")
        used_names.add(candidate)
        gs_map[token] = candidate
    return gs_map


def generate_gs_variant(problem_id, question, answer, tokens):
    """Raises ValueError when there is nothing to rename (the variant would be
    an exact copy of the original) or on a name collision."""
    if not tokens:
        raise ValueError(f"GS: no tokens to substitute for {problem_id} (would produce a no-op variant)")
    gs_map = build_gs_map(problem_id, tokens, question)
    return {
        'question': apply_substitution(question, gs_map),
        'answer': apply_substitution(answer, gs_map),
        'map': gs_map,
    }


def run_gs(tagged_problems):
    """Returns ({id: variant or None}, {id: rejection reason})."""
    results, rejected = {}, {}
    for p in tagged_problems:
        try:
            results[p['id']] = generate_gs_variant(p['id'], p['question'], p['final_answer'], p['tokens'])
        except ValueError as e:
            results[p['id']] = None
            rejected[p['id']] = str(e)
    return results, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--problems', default=PROBLEMS_PATH)
    parser.add_argument('--tags', default=TAGS_PATH)
    parser.add_argument('--output', default=GS_PATH)
    parser.add_argument('--rejected-output', default=GS_REJECTED_PATH)
    args = parser.parse_args()

    results, rejected = run_gs(load_problems_with_tags(args.problems, args.tags))
    write_json(results, args.output)
    write_json(rejected, args.rejected_output)
    print(f"GS: {len(results) - len(rejected)}/{len(results)} accepted -> {args.output}")
    print(f"GS: {len(rejected)} rejected, with reasons -> {args.rejected_output}")


if __name__ == '__main__':
    main()
