"""
Stage 3 — DLM (Descriptive Long Misleading) variant. One GPT-4.1-mini call
per problem proposes, for every tagged var/param, a replacement name that is
a real mathematical concept from a different subfield, chosen to misdirect,
with a 0-3 self-rated misdirection score. The names are then substituted
into both the question and the answer (so a renamed variable that appears in
the answer, e.g. `r = 2`, stays consistent with the question).

No GAP equivalent: GAP's closest variant (DLC) samples from a static word
pool with no LLM call or misdirection scoring.

    python -m pipeline.dlm [--submit]
    python -m pipeline.dlm --poll-and-fetch BATCH_ID[,BATCH_ID...]
    python -m pipeline.dlm --apply
"""

import argparse
import json
import os

from .batch_api import make_chat_request, message_content, poll_and_fetch, submit_or_dry_run
from .common import (
    BATCH_DIR, DLM_MAPS_PATH, DLM_PATH, DLM_REJECTED_PATH, DLM_WARNINGS_PATH, PROBLEMS_PATH, TAGS_PATH,
    read_json, read_jsonl, write_json,
)
from .merge import load_problems_with_tags
from .substitution import apply_substitution

MODEL = 'gpt-4.1-mini'
MIN_ACCEPTABLE_SCORE = 2

SYSTEM_PROMPT = (
    "You replace mathematical identifiers in a problem with misleading "
    "descriptive names, to test whether a solver is anchored on superficial "
    "names rather than reasoning from the problem structure.\n\n"
    "Rules for each replacement name:\n"
    "1. It must denote a REAL mathematical concept (not a nonsense phrase).\n"
    "2. The concept must come from a different mathematical subfield than "
    "the problem's stated topic.\n"
    "3. It must actively misdirect intuition about what the identifier "
    "represents in this problem (not merely be unrelated/neutral).\n\n"
    "For every token given, return a replacement name and self-rate how "
    "strongly it misdirects on a 0-3 scale (0=no misdirection, 3=strong "
    "misdirection). Return strict JSON matching the schema."
)

RESPONSE_SCHEMA = {
    'type': 'json_schema',
    'json_schema': {
        'name': 'dlm_tags',
        'schema': {
            'type': 'object',
            'properties': {
                'replacements': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'token': {'type': 'string'},
                            'replacement': {'type': 'string'},
                            'misdirection_score': {'type': 'integer'},
                        },
                        'required': ['token', 'replacement', 'misdirection_score'],
                        'additionalProperties': False,
                    },
                },
            },
            'required': ['replacements'],
            'additionalProperties': False,
        },
        'strict': True,
    },
}

BATCH_INPUT_PATH = os.path.join(BATCH_DIR, 'dlm_input.jsonl')


def build_dlm_requests(problems_with_tokens):
    requests = []
    for p in problems_with_tokens:
        user_content = json.dumps({
            'question': p['question'],
            'topic': p.get('topic'),
            'tokens_to_replace': p['tokens'],
        })
        requests.append(make_chat_request(
            custom_id=p['id'],
            model=MODEL,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user_content},
            ],
            response_format=RESPONSE_SCHEMA,
        ))
    return requests


def parse_dlm_results(raw_results):
    """Returns {id: {token: {token, replacement, misdirection_score}}}, or {id: None} on failure."""
    parsed = {}
    for problem_id, body in raw_results.items():
        content = message_content(body)
        try:
            parsed[problem_id] = {r['token']: r for r in json.loads(content)['replacements']} if content else None
        except (KeyError, ValueError):
            parsed[problem_id] = None
    return parsed


def low_score_replacements(dlm_map):
    return [r for r in dlm_map.values() if r['misdirection_score'] < MIN_ACCEPTABLE_SCORE]


def generate_dlm_variant(question, final_answer, dlm_map):
    """Raises ValueError on an empty map (no-op variant), duplicate
    replacement names, or a replacement that already occurs in the question."""
    if not dlm_map:
        raise ValueError("DLM: empty dlm_map (would produce a no-op variant)")

    token_map = {token: r['replacement'] for token, r in dlm_map.items()}
    names = list(token_map.values())
    if len(names) != len(set(names)):
        raise ValueError("DLM collision: duplicate replacement names in DLM response")
    for name in names:
        if name in question:
            raise ValueError(f"DLM collision: replacement name {name!r} collides with existing problem text")

    return {
        'question': apply_substitution(question, token_map),
        'answer': apply_substitution(final_answer, token_map),
        'map': dlm_map,
    }


def apply_dlm_to_problems(problems, dlm_maps):
    """Returns ({id: variant or None}, {id: rejection reason})."""
    results, rejected = {}, {}
    for p in problems:
        dlm_map = dlm_maps.get(p['id'])
        if dlm_map is None:
            results[p['id']] = None
            rejected[p['id']] = 'no DLM tagging result (tagging failed, or this problem was never sent)'
            continue
        try:
            results[p['id']] = generate_dlm_variant(p['question'], p['final_answer'], dlm_map)
        except ValueError as e:
            results[p['id']] = None
            rejected[p['id']] = str(e)
    return results, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--problems', default=PROBLEMS_PATH)
    parser.add_argument('--tags', default=TAGS_PATH)
    parser.add_argument('--submit', action='store_true', help='send the batch to OpenAI (spends budget)')
    parser.add_argument('--poll-and-fetch', metavar='BATCH_IDS', help='comma-separated batch IDs to collect')
    parser.add_argument('--apply', action='store_true', help='substitute the fetched names into the problems')
    args = parser.parse_args()

    if args.poll_and_fetch:
        parsed = parse_dlm_results(poll_and_fetch(args.poll_and_fetch))
        write_json(parsed, DLM_MAPS_PATH)
        # Low-scoring replacements are kept but logged for review.
        warnings = {pid: low_score_replacements(m) for pid, m in parsed.items()
                    if m is not None and low_score_replacements(m)}
        write_json(warnings, DLM_WARNINGS_PATH)
        print(f"Wrote {len(parsed)} DLM maps -> {DLM_MAPS_PATH}")
        print(f"{len(warnings)} with a replacement scoring < {MIN_ACCEPTABLE_SCORE} -> {DLM_WARNINGS_PATH}")
        return

    if args.apply:
        results, rejected = apply_dlm_to_problems(read_jsonl(args.problems), read_json(DLM_MAPS_PATH))
        write_json(results, DLM_PATH)
        write_json(rejected, DLM_REJECTED_PATH)
        print(f"DLM: {len(results) - len(rejected)}/{len(results)} accepted -> {DLM_PATH}")
        print(f"DLM: {len(rejected)} rejected, with reasons -> {DLM_REJECTED_PATH}")
        return

    requests = build_dlm_requests(load_problems_with_tags(args.problems, args.tags))
    submit_or_dry_run(requests, BATCH_INPUT_PATH, 'deepmathgap-dlm', args.submit,
                      poll_hint='python -m pipeline.dlm --poll-and-fetch')


if __name__ == '__main__':
    main()
