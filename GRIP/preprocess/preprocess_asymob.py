import json
import multiprocessing as mp
import os
import re
import statistics
from pathlib import Path

from sympy import sympify
from transformers import AutoTokenizer

from gold_normalize import normalize_gold

SCRIPT_DIR = Path(__file__).resolve().parent   # preprocess/
REPO_ROOT = SCRIPT_DIR.parent                  # repo root
DATA_ROOT = REPO_ROOT / "data"

IN_PATH = DATA_ROOT / "ASyMOB_clean.jsonl"
OUT_PATH = DATA_ROOT / "asymob_prepped.jsonl"

TOKENIZER_NAME = "Qwen/Qwen2.5-Math-7B"
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
OVERLONG_THRESHOLD = 1024


def parse_intensity(variation):
    if not variation or variation == "Original":
        return None
    m = re.search(r"-(\d+)", variation)
    return int(m.group(1)) if m else None


def parse_family(variation):
    if not variation or variation == "Original":
        return "Original"
    fam = re.sub(r"-\d+", "", variation)
    return fam if fam else variation


def is_stability(variation):
    return bool(variation) and variation.endswith("-S")


def process_row(args):
    """CPU-bound per-row work (JSON parse + gold normalization + sympify
    validation) - runs in a worker process. Tokenization stays in the main
    process afterward since it's already batched/multi-threaded internally."""
    row_idx, line = args
    d = json.loads(line)

    challenge = d["Challenge"]
    answer_raw = d["Answer in Sympy"]
    variation = d["Variation"]
    seed_id = d["id"]

    empty_field = not str(challenge).strip() or not str(answer_raw).strip()
    problem = challenge.strip()
    gold_answers, flagged = normalize_gold(answer_raw)

    sympify_fails = []
    for g in gold_answers:
        try:
            sympify(g)
        except Exception as e:
            sympify_fails.append((seed_id, variation, g, str(e)))

    prepped_row = {
        "uid": f"{seed_id}__{variation}__{row_idx}",
        "group_id": seed_id,
        "arm": variation,
        "is_original": variation == "Original",
        "problem": problem,
        "gold_answers": gold_answers,
        "gold_format": "sympy",
        "n_samples": 8 if variation == "Original" else 1,
        "meta": {
            "family": parse_family(variation),
            "intensity": parse_intensity(variation),
            "is_stability": is_stability(variation),
        },
    }
    prompt = f"{problem}\n{INSTRUCTION}"
    return prepped_row, prompt, empty_field, flagged, sympify_fails


def main():
    print(f"reading from {IN_PATH}")
    print(f"writing to {OUT_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    with open(IN_PATH) as f:
        raw_lines = f.readlines()
    n_rows = len(raw_lines)

    try:
        n_workers = len(os.sched_getaffinity(0))   # respects PBS cpuset, not just host core count
    except AttributeError:
        n_workers = os.cpu_count() or 1
    print(f"using {n_workers} worker process(es) for gold validation")

    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(process_row, enumerate(raw_lines), chunksize=200)

    empty_field_count = 0
    flagged_gold_count = 0
    sympify_fail_count = 0
    sympify_fail_examples = []

    prompts = []
    prepped = []

    for prepped_row, prompt, empty_field, flagged, sympify_fails in results:
        prepped.append(prepped_row)
        prompts.append(prompt)
        if empty_field:
            empty_field_count += 1
        if flagged:
            flagged_gold_count += 1
        for ex in sympify_fails:
            sympify_fail_count += 1
            if len(sympify_fail_examples) < 5:
                sympify_fail_examples.append(ex)

    token_ids = tokenizer(prompts)["input_ids"]
    n_tokens = [len(t) for t in token_ids]
    for row, n in zip(prepped, n_tokens):
        row["n_prompt_tokens"] = n

    uids = [row["uid"] for row in prepped]
    n_original = sum(1 for row in prepped if row["is_original"])
    n_overlong = sum(1 for n in n_tokens if n > OVERLONG_THRESHOLD)

    print(f"rows: {n_rows}")
    print(f"unique uids: {len(set(uids))}")
    print(f"rows with empty Challenge or Answer in Sympy: {empty_field_count}")
    print(f"is_original rows: {n_original} (expect 100)")
    print(f"gold values flagged (normalization collapsed to nothing): {flagged_gold_count}")
    print(f"sympify failures: {sympify_fail_count}")
    for ex in sympify_fail_examples:
        print(f"  example: id={ex[0]} variation={ex[1]} gold={ex[2]!r} error={ex[3]}")
    print(
        f"n_prompt_tokens: max={max(n_tokens)} "
        f"p99={statistics.quantiles(n_tokens, n=100)[98]:.1f} "
        f"mean={statistics.mean(n_tokens):.1f}"
    )
    print(f"rows with n_prompt_tokens > {OVERLONG_THRESHOLD}: {n_overlong}")

    assert len(set(uids)) == n_rows, f"uid collision: {n_rows - len(set(uids))} duplicate uid(s)"
    assert empty_field_count == 0, f"{empty_field_count} rows have an empty Challenge or Answer in Sympy"

    with open(OUT_PATH, "w") as f:
        for row in prepped:
            f.write(json.dumps(row) + "\n")

    print(f"wrote {len(prepped)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
