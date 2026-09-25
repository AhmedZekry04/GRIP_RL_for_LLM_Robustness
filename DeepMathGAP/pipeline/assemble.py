"""
Stage 5 — assembly. Local and free.

Emits one group of four rows (k = 0 original, 1 GS, 2 DLM, 3 KV) per
problem for which all three perturbation stages succeeded. If any stage
failed, the whole group is dropped, so every group has the same shape for
per-group invariance statistics. Output: the pre-filtering v1 dataset
(38,942 groups), which pipeline.postprocess turns into the final v2.

    python -m pipeline.assemble
"""

import argparse
import glob
import os

from .common import (
    ASSEMBLED_METADATA_PATH, ASSEMBLED_PATH, AUDIT_DIR, DLM_PATH, DLM_REJECTED_PATH, GS_PATH, GS_REJECTED_PATH,
    KV_ACCEPTED_PATH, KV_DIR, PROBLEMS_PATH, read_json, read_jsonl, write_jsonl,
)

VARIANT_TYPES = ['original', 'surface_gs', 'surface_dlm', 'kernel']  # index = k


def assemble_group(problem, gs_result, dlm_result, kv_result):
    variants = [
        (problem['question'], problem['final_answer']),
        (gs_result['question'], gs_result['answer']),
        (dlm_result['question'], dlm_result['answer']),
        (kv_result['question'], kv_result['answer']),
    ]
    return [
        {
            'id': problem['id'],
            'k': k,
            'type': VARIANT_TYPES[k],
            'question': question,
            'answer': answer,
            'answer_type': problem['answer_type'],
            'difficulty': problem['difficulty'],
            'topic': problem['topic'],
        }
        for k, (question, answer) in enumerate(variants)
    ]


def assemble_all(problems, stage_results, stage_reasons):
    """stage_results / stage_reasons: {'gs'|'dlm'|'kv': {id: result or reason}}.
    Returns (rows, metadata, rejected_groups)."""
    rows, metadata, rejected = [], [], []
    for problem in problems:
        pid = problem['id']
        results = {stage: stage_results[stage].get(pid) for stage in ('gs', 'dlm', 'kv')}
        missing = [stage for stage, r in results.items() if r is None]
        if missing:
            rejected.append({
                'id': pid,
                'missing_stages': missing,
                'reasons': {stage: stage_reasons[stage].get(pid, 'missing') for stage in missing},
            })
            continue
        rows.extend(assemble_group(problem, results['gs'], results['dlm'], results['kv']))
        metadata.append({'id': pid, 'held_out': problem['held_out']})
    return rows, metadata, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--problems', default=PROBLEMS_PATH)
    args = parser.parse_args()

    # Per-attempt KV discard logs; the last attempt's reason wins.
    kv_discarded = {}
    for path in sorted(glob.glob(os.path.join(KV_DIR, 'discard_log_attempt*.json'))):
        kv_discarded.update(read_json(path))

    rows, metadata, rejected = assemble_all(
        read_jsonl(args.problems),
        stage_results={'gs': read_json(GS_PATH), 'dlm': read_json(DLM_PATH), 'kv': read_json(KV_ACCEPTED_PATH)},
        stage_reasons={'gs': read_json(GS_REJECTED_PATH, default={}),
                       'dlm': read_json(DLM_REJECTED_PATH, default={}),
                       'kv': kv_discarded},
    )
    rejected_path = os.path.join(AUDIT_DIR, 'rejected_groups.jsonl')
    write_jsonl(rows, ASSEMBLED_PATH)
    write_jsonl(metadata, ASSEMBLED_METADATA_PATH)
    write_jsonl(rejected, rejected_path)
    print(f"Assembled {len(metadata)} groups ({len(rows)} rows) -> {ASSEMBLED_PATH}")
    print(f"Rejected {len(rejected)} groups -> {rejected_path}")


if __name__ == '__main__':
    main()
