"""
Stage 0 — download DeepMath-103K, drop the R1 solution traces, filter by
difficulty and answer type, and assign stable `dmgap_XXXXXX` IDs.
Local and free.

    python -m pipeline.prepare [--skip-download]
"""

import argparse
import json
import os
import re

from .common import PROBLEMS_PATH, RAW_PATH, REJECTED_PROBLEMS_PATH, write_jsonl

KEEP_COLUMNS = ['question', 'final_answer', 'difficulty', 'topic']
# DeepSeek-R1 reasoning traces: never read past this point, to avoid a
# double-distillation path into anything trained on this dataset.
DROP_COLUMNS = ['r1_solution_1', 'r1_solution_2', 'r1_solution_3']

MIN_DIFFICULTY = 3.0        # inclusive; easier problems give little invariance signal
HELD_OUT_DIFFICULTY = 9.0   # inclusive; kept, but flagged `held_out` in metadata

# Only answers that cannot be perturbed at all are rejected. `other` is kept:
# the regex classifier below has too many false positives to drop it blindly.
REJECT_ANSWER_TYPES = {'boolean', 'multiple_choice'}

_BOOLEAN_VALUES = {'yes', 'no', 'true', 'false'}
_MCQ_PATTERN = re.compile(r'^\(?[A-E]\)?$')
_SET_INTERVAL_PATTERN = re.compile(r'^[\[\(\{].*[\]\)\}]$')
_EQUATION_PATTERN = re.compile(r'[a-zA-Z]\s*=')


def download(raw_path=RAW_PATH):
    """Download DeepMath-103K from Hugging Face and cache it locally as JSONL."""
    from datasets import load_dataset

    ds = load_dataset('zwhe99/DeepMath-103K')['train'].remove_columns(DROP_COLUMNS)
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    ds.to_json(raw_path)
    print(f"Saved {len(ds)} problems to {raw_path}")


def classify_answer_type(answer):
    """Heuristic answer-type label: one of numerical, expression,
    set_interval, equation, boolean, multiple_choice, other."""
    if answer is None:
        return 'other'
    a = answer.strip()
    a_lower = a.lower().strip('.')

    if a_lower in _BOOLEAN_VALUES:
        return 'boolean'
    if _MCQ_PATTERN.match(a):
        return 'multiple_choice'
    if _SET_INTERVAL_PATTERN.match(a):
        return 'set_interval'
    if _EQUATION_PATTERN.search(a) and not a.startswith('\\'):
        return 'equation'
    if re.fullmatch(r'-?[\d./π e^+\-*()\\a-zA-Z{}\[\]_,\s]+', a) and any(c.isdigit() for c in a):
        if any(op in a for op in ('^', '\\', 'sqrt', '_')) and re.search(r'[a-zA-Z]', a.replace('pi', '').replace('e', '')):
            return 'expression'
        return 'numerical'
    if re.search(r'[a-zA-Z]', a):
        return 'expression'
    return 'other'


def filter_problems(raw_path):
    """Returns (main_pool, held_out_pool, rejected). IDs are assigned in raw
    file order across both pools, so they never change between runs."""
    main_pool, held_out_pool, rejected = [], [], []
    next_id = 0

    with open(raw_path, encoding='utf-8') as f:
        for line in f:
            p = json.loads(line)
            record = {k: p.get(k) for k in KEEP_COLUMNS}
            difficulty = record['difficulty']
            answer_type = classify_answer_type(record['final_answer'])

            if answer_type in REJECT_ANSWER_TYPES:
                rejected.append({**record, 'answer_type': answer_type,
                                 'rejection_reason': f'answer_type={answer_type}'})
                continue
            if difficulty is None or difficulty < MIN_DIFFICULTY:
                rejected.append({**record, 'answer_type': answer_type,
                                 'rejection_reason': f'difficulty={difficulty}'})
                continue

            held_out = difficulty >= HELD_OUT_DIFFICULTY
            record.update(id=f'dmgap_{next_id:06d}', answer_type=answer_type, held_out=held_out)
            next_id += 1
            (held_out_pool if held_out else main_pool).append(record)

    return main_pool, held_out_pool, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--skip-download', action='store_true',
                        help=f'reuse an existing {os.path.relpath(RAW_PATH)} instead of re-downloading')
    args = parser.parse_args()

    if not args.skip_download or not os.path.exists(RAW_PATH):
        download()

    main_pool, held_out_pool, rejected = filter_problems(RAW_PATH)
    # Main pool first, then held-out: the order every later stage (and the
    # final dataset) follows.
    write_jsonl(main_pool + held_out_pool, PROBLEMS_PATH)
    write_jsonl(rejected, REJECTED_PROBLEMS_PATH)

    print(f"Kept {len(main_pool) + len(held_out_pool)} problems "
          f"({len(main_pool)} difficulty < {HELD_OUT_DIFFICULTY}, {len(held_out_pool)} held-out) -> {PROBLEMS_PATH}")
    print(f"Rejected {len(rejected)} -> {REJECTED_PROBLEMS_PATH}")


if __name__ == '__main__':
    main()
