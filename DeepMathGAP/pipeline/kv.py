"""
Stage 4 — KV (Kernel Variant): resample the numeric constants of a problem
and re-derive its answer. The expensive stage (o4-mini).

Per attempt (at most 3 per problem):
  1. build-synth / poll-synth: one o4-mini call finds the mutable constants,
     resamples them, rewrites the question and solves it (back-synthesis).
  2. build-blindsolve / poll-blindsolve: three independent o4-mini calls
     solve the rewritten question without seeing the synthesised answer.
  3. evaluate (local): accept only if the three blind answers agree with each
     other and with the synthesised answer (math_verify), and the new question
     differs from the original only in the resampled constants. Failures are
     written to pending_attempt{N+1}.json for the next attempt; after attempt
     3 they are discarded, which later drops the whole group at assembly.

Adapted from GAP's `kv_math_200.py` (slot discovery + back-synthesis, and
the \\boxed{} extraction; https://github.com/YurenHao0426/GAP, CC BY 4.0),
with the two calls merged into one and GAP's 3-judge accept/reject vote
replaced by blind re-solving plus the structural diff check.

    python -m pipeline.kv build-synth --attempt N [--pending FILE] [--submit]
    python -m pipeline.kv poll-synth --attempt N --batch-id IDS
    python -m pipeline.kv build-blindsolve --attempt N [--submit]
    python -m pipeline.kv poll-blindsolve --attempt N --batch-id IDS
    python -m pipeline.kv evaluate --attempt N
"""

import argparse
import json
import os

from .batch_api import make_chat_request, message_content, poll_and_fetch, submit_or_dry_run
from .common import BATCH_DIR, KV_ACCEPTED_PATH, KV_DIR, PROBLEMS_PATH, TAGS_PATH, read_json, write_json
from .merge import load_problems_with_tags
from .verify import blind_solve_consensus, structural_diff_ok

MODEL = 'o4-mini'
MAX_ATTEMPTS = 3
N_BLIND_SOLVERS = 3
CHECKPOINT_EVERY = 1000

SLOT_AND_SYNTH_PROMPT = (
    "You analyze a math problem to create a parameter-resampled variant.\n\n"
    "Given the problem, answer, and excluded scientific constants:\n"
    "1. Identify mutable numerical constants in the question (do not touch "
    "the excluded scientific constants).\n"
    "2. Infer the valid range for each constant from the problem's own "
    "constraints (so the new problem stays well-posed).\n"
    "3. Resample new values for each constant.\n"
    "4. Rewrite the question with the new values substituted in. This must "
    "be a byte-for-byte copy of the original question text, with ONLY the "
    "resampled digits/values changed in place — no added/removed clauses, "
    "no method changes, and CRITICALLY no reformatting: keep every LaTeX "
    "delimiter exactly as written (\\(...\\), \\[...\\], $...$, every "
    "backslash and brace), keep every word exactly as written (do not drop "
    "or rephrase words like 'unit' in 'unit circle'), and do not double "
    "backslashes or alter whitespace/newlines. If a number you change has "
    "an English ordinal suffix (e.g. '21st'), update the suffix to match "
    "the new number's grammar ('19th') but change nothing else nearby.\n"
    "5. Solve the new question yourself and give the new final answer.\n"
    "6. List preserved_steps: the high-level reasoning chain that must stay "
    "invariant between the original and the new problem.\n\n"
    "Return strict JSON matching the schema."
)

SLOT_SYNTH_SCHEMA = {
    'type': 'json_schema',
    'json_schema': {
        'name': 'kv_slot_synth',
        'schema': {
            'type': 'object',
            'properties': {
                'changed_constants': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'original_value': {'type': 'string'},
                            'new_value': {'type': 'string'},
                        },
                        'required': ['original_value', 'new_value'],
                        'additionalProperties': False,
                    },
                },
                'new_question': {'type': 'string'},
                'new_answer': {'type': 'string'},
                'preserved_steps': {'type': 'array', 'items': {'type': 'string'}},
            },
            'required': ['changed_constants', 'new_question', 'new_answer', 'preserved_steps'],
            'additionalProperties': False,
        },
        'strict': True,
    },
}

# No temperature: o-series reasoning models reject sampling parameters, so
# solver diversity comes from independent calls.
BLIND_SOLVE_PROMPT = "Solve the following math problem step by step. Put your final answer in \\boxed{}."


def _synth_path(attempt):
    return os.path.join(KV_DIR, f'synth_attempt{attempt}.json')


def _blindsolve_path(attempt):
    return os.path.join(KV_DIR, f'blindsolve_attempt{attempt}.json')


def _discard_log_path(attempt):
    return os.path.join(KV_DIR, f'discard_log_attempt{attempt}.json')


def _pending_path(attempt):
    return os.path.join(KV_DIR, f'pending_attempt{attempt}.json')


def extract_boxed_answer(text):
    """Contents of the last \\boxed{...} in `text` (brace-balanced), or None."""
    if not text:
        return None
    matches = []
    i = 0
    while i < len(text):
        idx = text.find('\\boxed{', i)
        if idx == -1:
            break
        depth = 1
        j = idx + 7
        while j < len(text) and depth > 0:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            j += 1
        if depth == 0:
            matches.append(text[idx + 7:j - 1].strip())
        i = j
    return matches[-1] if matches else None


def build_slot_synth_requests(problems, attempt):
    requests = []
    for p in problems:
        user_content = json.dumps({
            'question': p['question'],
            'answer': p['final_answer'],
            'excluded_sci_consts': p.get('sci_consts', []),
            'seed_note': f"deterministic seed key: ({p['id']}, KV, {attempt})",
        })
        requests.append(make_chat_request(
            custom_id=f"{p['id']}__kv_synth__attempt{attempt}",
            model=MODEL,
            messages=[
                {'role': 'system', 'content': SLOT_AND_SYNTH_PROMPT},
                {'role': 'user', 'content': user_content},
            ],
            response_format=SLOT_SYNTH_SCHEMA,
        ))
    return requests


def build_blind_solve_requests(problem_id, new_question, attempt):
    return [
        make_chat_request(
            custom_id=f"{problem_id}__kv_blindsolve__attempt{attempt}__solver{solver}",
            model=MODEL,
            messages=[
                {'role': 'system', 'content': BLIND_SOLVE_PROMPT},
                {'role': 'user', 'content': new_question},
            ],
        )
        for solver in range(N_BLIND_SOLVERS)
    ]


def parse_slot_synth_results(raw_results):
    """Returns {problem_id: synth dict or None}. A refusal (the model's way of
    saying there is nothing to resample) becomes an empty changed_constants
    list, which evaluate rejects, with the refusal text kept for review."""
    parsed = {}
    for custom_id, body in raw_results.items():
        problem_id = custom_id.split('__')[0]
        try:
            message = body['choices'][0]['message']
            if message['content'] is None:
                data = {'changed_constants': [], 'new_question': None, 'new_answer': None,
                        'preserved_steps': [], '_refusal': message.get('refusal')}
            else:
                data = json.loads(message['content'])
        except (KeyError, IndexError, json.JSONDecodeError):
            data = None
        parsed[problem_id] = data
    return parsed


def parse_blind_solve_results(raw_results):
    """Returns {problem_id: [boxed answer of each solver, None if missing]}."""
    parsed = {}
    for custom_id, body in raw_results.items():
        problem_id, _, _, solver_tag = custom_id.split('__')
        answers = parsed.setdefault(problem_id, [None] * N_BLIND_SOLVERS)
        answers[int(solver_tag.replace('solver', ''))] = extract_boxed_answer(message_content(body))
    return parsed


def has_real_change(changed_constants):
    """False for an empty or no-op change list, which would otherwise pass
    every check trivially as a duplicate of the original."""
    return bool(changed_constants) and any(c['original_value'] != c['new_value'] for c in changed_constants)


def evaluate_attempt(original_question, synth, blind_solve_answers):
    """Returns (accepted, rejection reason or None)."""
    if synth is None:
        return False, 'synthesis call failed or returned invalid JSON'
    if not has_real_change(synth.get('changed_constants')):
        return False, 'no actual constant change (empty or no-op changed_constants)'

    consensus_ok, reason = blind_solve_consensus(synth['new_answer'], blind_solve_answers)
    if not consensus_ok:
        return False, reason

    changed = [(c['original_value'], c['new_value']) for c in synth['changed_constants']]
    if not structural_diff_ok(original_question, synth['new_question'], changed):
        return False, 'structural diff check failed (more than constants changed)'
    return True, None


def evaluate(problems, attempt, verbose=False):
    """Checks every synth result of this attempt. Resumable: ids already in
    accepted.json or this attempt's discard log are skipped, and progress is
    flushed to disk every CHECKPOINT_EVERY problems."""
    questions = {p['id']: p['question'] for p in problems}
    synth_results = read_json(_synth_path(attempt))
    blindsolve_results = read_json(_blindsolve_path(attempt), default={})
    accepted = read_json(KV_ACCEPTED_PATH, default={})
    discarded = read_json(_discard_log_path(attempt), default={})
    final_prefix = f'discarded after {MAX_ATTEMPTS} attempts'

    def flush():
        write_json(accepted, KV_ACCEPTED_PATH)
        write_json(discarded, _discard_log_path(attempt))
        retry_ids = [pid for pid, reason in discarded.items() if not reason.startswith(final_prefix)]
        if retry_ids:
            write_json(retry_ids, _pending_path(attempt + 1))

    remaining = [(pid, s) for pid, s in synth_results.items() if pid not in accepted and pid not in discarded]
    print(f"Evaluating {len(remaining)} problems ({len(synth_results) - len(remaining)} already done)...")

    for i, (problem_id, synth) in enumerate(remaining, 1):
        if verbose:
            print(f"  [{i}/{len(remaining)}] {problem_id}", flush=True)
        blind_answers = blindsolve_results.get(problem_id, [None] * N_BLIND_SOLVERS)
        ok, reason = evaluate_attempt(questions.get(problem_id), synth, blind_answers)
        if ok:
            accepted[problem_id] = {'question': synth['new_question'], 'answer': synth['new_answer']}
        elif attempt < MAX_ATTEMPTS:
            discarded[problem_id] = reason
        else:
            discarded[problem_id] = f'{final_prefix}: {reason}'
        if i % CHECKPOINT_EVERY == 0 or i == len(remaining):
            flush()
            print(f"  progress: {i}/{len(remaining)} ({len(accepted)} accepted in total)")

    n_accepted = sum(1 for pid in synth_results if pid in accepted)
    n_retry = sum(1 for pid in synth_results if pid in discarded and not discarded[pid].startswith(final_prefix))
    print(f"Attempt {attempt}: {n_accepted} accepted, {n_retry} to retry "
          f"(pass --pending {_pending_path(attempt + 1)} to build-synth --attempt {attempt + 1})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['build-synth', 'poll-synth', 'build-blindsolve', 'poll-blindsolve', 'evaluate'])
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--problems', default=PROBLEMS_PATH)
    parser.add_argument('--tags', default=TAGS_PATH)
    parser.add_argument('--pending', help='JSON list of ids to restrict build-synth to (attempts 2 and 3)')
    parser.add_argument('--batch-id', help='comma-separated batch IDs (poll-synth / poll-blindsolve)')
    parser.add_argument('--submit', action='store_true', help='send the batch to OpenAI (spends budget)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    attempt = args.attempt

    if args.action == 'build-synth':
        problems = load_problems_with_tags(args.problems, args.tags)
        if args.pending:
            pending = set(read_json(args.pending))
            problems = [p for p in problems if p['id'] in pending]
        submit_or_dry_run(build_slot_synth_requests(problems, attempt),
                          os.path.join(BATCH_DIR, f'kv_synth_attempt{attempt}.jsonl'),
                          f'deepmathgap-kv-synth-attempt{attempt}', args.submit,
                          poll_hint=f'python -m pipeline.kv poll-synth --attempt {attempt} --batch-id')

    elif args.action == 'poll-synth':
        parsed = parse_slot_synth_results(poll_and_fetch(args.batch_id))
        write_json(parsed, _synth_path(attempt))
        print(f"Wrote {len(parsed)} synth results -> {_synth_path(attempt)}")

    elif args.action == 'build-blindsolve':
        requests = []
        for problem_id, synth in read_json(_synth_path(attempt)).items():
            if synth is not None and has_real_change(synth.get('changed_constants')):
                requests.extend(build_blind_solve_requests(problem_id, synth['new_question'], attempt))
        submit_or_dry_run(requests, os.path.join(BATCH_DIR, f'kv_blindsolve_attempt{attempt}.jsonl'),
                          f'deepmathgap-kv-blindsolve-attempt{attempt}', args.submit,
                          poll_hint=f'python -m pipeline.kv poll-blindsolve --attempt {attempt} --batch-id')

    elif args.action == 'poll-blindsolve':
        parsed = parse_blind_solve_results(poll_and_fetch(args.batch_id))
        write_json(parsed, _blindsolve_path(attempt))
        print(f"Wrote {len(parsed)} blind-solve results -> {_blindsolve_path(attempt)}")

    elif args.action == 'evaluate':
        evaluate(load_problems_with_tags(args.problems, args.tags), attempt, args.verbose)


if __name__ == '__main__':
    main()
