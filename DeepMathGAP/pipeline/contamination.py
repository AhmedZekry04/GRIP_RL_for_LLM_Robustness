"""
9-gram contamination check against evaluation benchmarks that postdate
DeepMath-103K's own decontamination: MATH-Perturb, AIME 2025 and ASyMOB.
A group is flagged if its original (k=0) question shares any whitespace
9-gram with any benchmark problem.

    python -m pipeline.contamination    # download the benchmarks (run once)
"""

import json
import os
import urllib.request

from .common import BENCHMARKS_DIR, read_jsonl, write_jsonl

N_GRAM = 9

BENCHMARK_FILES = {
    'MATH-Perturb': 'math_perturb.jsonl',
    'AIME-2025': 'aime_2025.jsonl',
    'ASyMOB': 'asymob.jsonl',
}

_MATH_PERTURB_URL = 'https://raw.githubusercontent.com/Kaffaljidhmah2/MATH-Perturb/main/math_perturb/'
_ASYMOB_URL = ('https://huggingface.co/datasets/Shalyt/ASyMOB-Algebraic_Symbolic_Mathematical_Operations_Benchmark'
               '/resolve/main/Full_ASyMOB_Dataset.json')


def ngrams(text, n=N_GRAM):
    tokens = text.split()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def load_benchmark_ngrams(benchmarks_dir=BENCHMARKS_DIR):
    """{benchmark name: set of 9-grams}. Fails if a benchmark file is missing."""
    loaded = {}
    for name, filename in BENCHMARK_FILES.items():
        path = os.path.join(benchmarks_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found; run `python -m pipeline.contamination` first")
        loaded[name] = set().union(*(ngrams(row['text']) for row in read_jsonl(path)))
    return loaded


def contaminated_benchmarks(question, benchmark_ngrams):
    """Sorted names of the benchmarks sharing a 9-gram with `question`."""
    question_ngrams = ngrams(question)
    return sorted(name for name, bench in benchmark_ngrams.items() if question_ngrams & bench)


def _first_field(record, *fields):
    for field in fields:
        if record.get(field):
            return str(record[field])
    raise KeyError(f"none of {fields} in record with fields {list(record)}")


def _save(texts, filename):
    path = os.path.join(BENCHMARKS_DIR, filename)
    write_jsonl([{'text': t} for t in texts], path)
    print(f"  wrote {len(texts)} problems -> {path}")


def download_math_perturb():
    records = []
    for fname in ('math_perturb_simple.jsonl', 'math_perturb_hard.jsonl'):
        with urllib.request.urlopen(_MATH_PERTURB_URL + fname, timeout=30) as resp:
            records += [json.loads(line) for line in resp.read().decode('utf-8').splitlines() if line.strip()]
    _save([_first_field(r, 'problem', 'question', 'text', 'Problem', 'Challenge') for r in records],
          BENCHMARK_FILES['MATH-Perturb'])


def download_aime_2025():
    from datasets import load_dataset

    records = []
    for ds_id in ('MathArena/aime_2025', 'MathArena/aime_2025_II'):
        records += list(load_dataset(ds_id, split='train'))
    _save([_first_field(r, 'problem', 'question', 'Problem', 'text', 'statement') for r in records],
          BENCHMARK_FILES['AIME-2025'])


def download_asymob():
    # Downloaded directly: the HF loading script is rejected by current `datasets`.
    with urllib.request.urlopen(_ASYMOB_URL, timeout=120) as resp:
        raw = json.loads(resp.read().decode('utf-8'))
    records = raw if isinstance(raw, list) else next(v for v in raw.values() if isinstance(v, list))
    _save([_first_field(r, 'Challenge', 'question', 'problem', 'input', 'text') for r in records],
          BENCHMARK_FILES['ASyMOB'])


if __name__ == '__main__':
    download_math_perturb()
    download_aime_2025()
    download_asymob()
