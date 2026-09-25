"""
Stage 6 — post-processing. Local and free. Turns the assembled v1 dataset
into the final deepmathgap_v2 by dropping whole groups in three passes:

  1. named-quantity filter v1 (word list)            38,942 -> 37,477 groups
  2. 9-gram contamination vs. benchmarks              37,477 -> 37,081
  3. named-quantity filter v2 (reviewed token list)   37,081 -> 36,057

Groups are always dropped whole, never row by row, so every remaining group
keeps all four variants. Each pass writes an audit file (group id -> reason)
to data/audit/.

    python -m pipeline.postprocess
"""

import argparse
import os

from . import named_quantity as nq
from .common import (
    ASSEMBLED_METADATA_PATH, ASSEMBLED_PATH, AUDIT_DIR, BENCHMARKS_DIR, DLM_PATH, FINAL_METADATA_PATH, FINAL_PATH,
    GS_PATH, read_json, read_jsonl, write_json, write_jsonl,
)
from .contamination import contaminated_benchmarks, load_benchmark_ngrams


def _group_rows(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row['id'], []).append(row)
    return groups


def _named_quantity_audit(group_ids, gs, dlm, match, reason_key):
    """{gid: [{token, <reason_key>, stage}]} for groups with a matching renamed token."""
    audit = {}
    for gid in group_ids:
        reasons, seen = [], set()
        for stage, token in nq.renamed_tokens(gid, gs, dlm):
            hit = match(token)
            if hit and token not in seen:
                reasons.append({'token': token, reason_key: hit, 'stage': stage})
                seen.add(token)
        if reasons:
            audit[gid] = reasons
    return audit


def candidate_tokens(group_ids, gs, dlm):
    """Every renamed token passing nq.is_candidate, most frequent first: the
    list that was reviewed by hand to build nq.CONFIRMED_DROPS."""
    hits = {}
    for gid in group_ids:
        for stage, token in nq.renamed_tokens(gid, gs, dlm):
            if nq.is_candidate(token):
                hits.setdefault(token, []).append((gid, stage))
    return {
        token: {
            'frequency': len(pairs),
            'groups': sorted({gid for gid, _ in pairs}),
            'stages': sorted({stage for _, stage in pairs}),
        }
        for token, pairs in sorted(hits.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    }


def _drop(groups, audit, label):
    kept = {gid: rows for gid, rows in groups.items() if gid not in audit}
    print(f"{label}: dropped {len(audit)} groups, {len(kept)} remain")
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', default=ASSEMBLED_PATH)
    parser.add_argument('--benchmarks-dir', default=BENCHMARKS_DIR)
    args = parser.parse_args()

    groups = _group_rows(read_jsonl(args.input))
    gs, dlm = read_json(GS_PATH), read_json(DLM_PATH)
    print(f"Assembled: {len(groups)} groups")

    audit_v1 = _named_quantity_audit(groups, gs, dlm, nq.v1_match, 'matched_word')
    groups = _drop(groups, audit_v1, 'Named-quantity v1')

    benchmark_ngrams = load_benchmark_ngrams(args.benchmarks_dir)
    contaminated = {}
    for gid, rows in groups.items():
        hit = contaminated_benchmarks(rows[0]['question'], benchmark_ngrams)  # rows[0] is k=0
        if hit:
            contaminated[gid] = hit
    groups = _drop(groups, contaminated, 'Contamination')

    candidates = candidate_tokens(groups, gs, dlm)
    audit_v2 = _named_quantity_audit(groups, gs, dlm, nq.v2_match, 'category')
    groups = _drop(groups, audit_v2, 'Named-quantity v2')

    write_jsonl([row for rows in groups.values() for row in rows], FINAL_PATH)
    write_jsonl([m for m in read_jsonl(ASSEMBLED_METADATA_PATH) if m['id'] in groups], FINAL_METADATA_PATH)
    write_json(audit_v1, os.path.join(AUDIT_DIR, 'dropped_named_quantity_v1.json'))
    write_json(contaminated, os.path.join(AUDIT_DIR, 'contaminated_groups.json'))
    write_json(candidates, os.path.join(AUDIT_DIR, 'named_quantity_candidates.json'))
    write_json(audit_v2, os.path.join(AUDIT_DIR, 'dropped_named_quantity_v2.json'))
    print(f"Final: {len(groups)} groups ({4 * len(groups)} rows) -> {FINAL_PATH}")


if __name__ == '__main__':
    main()
