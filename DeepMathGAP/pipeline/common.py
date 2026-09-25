"""Default file locations and JSON/JSONL helpers shared by every stage."""

import gzip
import io
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')

# Not committed: the raw source dataset and every per-stage working file.
RAW_DIR = os.path.join(DATA_DIR, 'raw')
WORK_DIR = os.path.join(DATA_DIR, 'intermediate')
BATCH_DIR = os.path.join(WORK_DIR, 'batches')
KV_DIR = os.path.join(WORK_DIR, 'kv')
BENCHMARKS_DIR = os.path.join(WORK_DIR, 'benchmarks')

# Committed: the final dataset and the audit trail of every dropped group.
FINAL_DIR = os.path.join(DATA_DIR, 'final')
AUDIT_DIR = os.path.join(DATA_DIR, 'audit')

RAW_PATH = os.path.join(RAW_DIR, 'deepmath103k.jsonl')
PROBLEMS_PATH = os.path.join(WORK_DIR, 'problems.jsonl')
REJECTED_PROBLEMS_PATH = os.path.join(WORK_DIR, 'rejected_problems.jsonl')
TAGS_PATH = os.path.join(WORK_DIR, 'tags.json')
GS_PATH = os.path.join(WORK_DIR, 'gs.json')
GS_REJECTED_PATH = os.path.join(WORK_DIR, 'gs_rejected.json')
DLM_MAPS_PATH = os.path.join(WORK_DIR, 'dlm.json')
DLM_WARNINGS_PATH = os.path.join(WORK_DIR, 'dlm_low_misdirection_warnings.json')
DLM_PATH = os.path.join(WORK_DIR, 'dlm_applied.json')
DLM_REJECTED_PATH = os.path.join(WORK_DIR, 'dlm_rejected.json')
KV_ACCEPTED_PATH = os.path.join(KV_DIR, 'accepted.json')
ASSEMBLED_PATH = os.path.join(WORK_DIR, 'deepmathgap_v1.jsonl')
ASSEMBLED_METADATA_PATH = os.path.join(WORK_DIR, 'metadata_v1.jsonl')

FINAL_PATH = os.path.join(FINAL_DIR, 'deepmathgap_v2.jsonl.gz')
FINAL_METADATA_PATH = os.path.join(FINAL_DIR, 'metadata.jsonl')


def _open_write(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith('.gz'):
        # mtime=0 keeps the compressed bytes identical across re-runs.
        return io.TextIOWrapper(gzip.GzipFile(path, 'wb', mtime=0), encoding='utf-8', newline='\n')
    return open(path, 'w', encoding='utf-8', newline='\n')


def _open_read(path):
    if path.endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8')
    return open(path, encoding='utf-8')


def read_jsonl(path):
    with _open_read(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records, path):
    with _open_write(path) as f:
        for r in records:
            f.write(json.dumps(r) + '\n')


def read_json(path, default=None):
    """Returns `default` when the file is missing and a default was given."""
    if default is not None and not os.path.exists(path):
        return default
    with _open_read(path) as f:
        return json.load(f)


def write_json(obj, path):
    with _open_write(path) as f:
        json.dump(obj, f, indent=2)
