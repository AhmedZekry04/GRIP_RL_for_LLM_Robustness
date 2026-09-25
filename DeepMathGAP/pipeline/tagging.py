"""
Stage 1 — token tagging. One GPT-4.1-mini call per problem (strict JSON
schema) labels the problem's identifiers as `vars`, `params` or `sci_consts`.
GS/DLM rename vars + params; KV never touches sci_consts. Replaces GAP's
regex-based variable extraction.

    python -m pipeline.tagging [--submit]
    python -m pipeline.tagging --poll-and-fetch BATCH_ID[,BATCH_ID...]
"""

import argparse
import json
import os

from .batch_api import make_chat_request, message_content, poll_and_fetch, submit_or_dry_run
from .common import BATCH_DIR, PROBLEMS_PATH, TAGS_PATH, read_jsonl, write_json

MODEL = 'gpt-4.1-mini'

SYSTEM_PROMPT = (
    "You tag mathematical identifiers in a problem statement. Given a math "
    "problem, its final answer, and its topic, identify:\n"
    "- vars: variable symbols representing unknowns to solve for or quantities "
    "introduced in the problem (e.g. x, n, theta)\n"
    "- params: named constants/parameters specific to this problem instance "
    "(e.g. a coefficient called k, a labeled point S)\n"
    "- sci_consts: universal mathematical/scientific constants (e.g. pi, e, i) "
    "— these must NOT appear in vars or params\n"
    "Return strict JSON matching the schema. Use the exact substrings as they "
    "appear in the question text."
)

RESPONSE_SCHEMA = {
    'type': 'json_schema',
    'json_schema': {
        'name': 'token_tags',
        'schema': {
            'type': 'object',
            'properties': {
                'vars': {'type': 'array', 'items': {'type': 'string'}},
                'params': {'type': 'array', 'items': {'type': 'string'}},
                'sci_consts': {'type': 'array', 'items': {'type': 'string'}},
            },
            'required': ['vars', 'params', 'sci_consts'],
            'additionalProperties': False,
        },
        'strict': True,
    },
}

BATCH_INPUT_PATH = os.path.join(BATCH_DIR, 'tagging_input.jsonl')


def build_tagging_requests(problems):
    requests = []
    for p in problems:
        user_content = json.dumps({
            'question': p['question'],
            'final_answer': p['final_answer'],
            'topic': p.get('topic'),
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


def _dedupe_overlaps(tags):
    """The model does not always keep sci_consts disjoint from vars/params.
    A token in both would be renamed by GS/DLM but protected by KV, so
    sci_consts wins."""
    sci_consts = set(tags.get('sci_consts', []))
    return {
        'vars': [v for v in tags.get('vars', []) if v not in sci_consts],
        'params': [p for p in tags.get('params', []) if p not in sci_consts],
        'sci_consts': tags.get('sci_consts', []),
    }


def parse_tagging_results(raw_results):
    """Returns {id: {vars, params, sci_consts}}, or {id: None} where tagging failed."""
    parsed = {}
    for problem_id, body in raw_results.items():
        content = message_content(body)
        try:
            parsed[problem_id] = _dedupe_overlaps(json.loads(content)) if content is not None else None
        except ValueError:
            parsed[problem_id] = None
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--problems', default=PROBLEMS_PATH)
    parser.add_argument('--submit', action='store_true', help='send the batch to OpenAI (spends budget)')
    parser.add_argument('--poll-and-fetch', metavar='BATCH_IDS', help='comma-separated batch IDs to collect')
    parser.add_argument('--output', default=TAGS_PATH)
    args = parser.parse_args()

    if args.poll_and_fetch:
        parsed = parse_tagging_results(poll_and_fetch(args.poll_and_fetch))
        write_json(parsed, args.output)
        print(f"Wrote {len(parsed)} tag records to {args.output}")
        return

    requests = build_tagging_requests(read_jsonl(args.problems))
    submit_or_dry_run(requests, BATCH_INPUT_PATH, 'deepmathgap-tagging', args.submit,
                      poll_hint='python -m pipeline.tagging --poll-and-fetch')


if __name__ == '__main__':
    main()
