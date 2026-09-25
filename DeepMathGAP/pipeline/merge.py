"""Joins Stage 0 problems with Stage 1 tags; used by the GS, DLM and KV stages."""

from .common import read_json, read_jsonl
from .substitution import token_occurs


def _matched_tokens(problem_id, question, tokens):
    """Drop tagged tokens that would never match in the question (tagging
    hallucinations, wrong casing, substrings of longer words), since they
    would otherwise silently become no-op substitutions."""
    kept = [t for t in tokens if token_occurs(question, t)]
    dropped = [t for t in tokens if t not in kept]
    if dropped:
        print(f"  [merge] {problem_id}: dropping unmatched tagged token(s) {ascii(dropped)}")
    return kept


def load_problems_with_tags(problems_path, tags_path):
    """Returns Stage 0 records extended with vars, params, sci_consts and
    tokens (= vars + params, the GS/DLM rename targets). Problems whose
    tagging failed are skipped."""
    tags = read_json(tags_path)
    merged = []
    skipped = 0
    for p in read_jsonl(problems_path):
        t = tags.get(p['id'])
        if t is None:
            skipped += 1
            continue
        kept_vars = _matched_tokens(p['id'], p['question'], t['vars'])
        kept_params = _matched_tokens(p['id'], p['question'], t['params'])
        merged.append({
            **p,
            'vars': kept_vars,
            'params': kept_params,
            'sci_consts': t['sci_consts'],
            'tokens': kept_vars + kept_params,
        })

    if skipped:
        print(f"  [merge] skipped {skipped} problem(s) with no tagging result")
    return merged
