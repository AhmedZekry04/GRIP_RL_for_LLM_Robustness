"""
GRIP: A0 generation on ASyMOB.

Input : data/asymob_prepped.jsonl  (canonical schema, see PREPROCESSING_SPEC.md)
Output: results/<RUN_ID>/eval/asymob/generations.jsonl

Reads only the canonical fields, so nothing here depends on ASyMOB's raw column
names or on parsing the variant taxonomy at runtime - preprocessing did that.
"""

import os
import re
import json
import time
import random
import subprocess
from collections import Counter
from datetime import datetime, timezone

from vllm import LLM, SamplingParams

# ── run constants ─────────────────────────────────────────────────────────────
USERNAME    = os.environ.get("USER", "amz25")
HOME        = f"/rds/general/user/{USERNAME}/home"
RUN_ID      = "a0_asymob_20260904_1.5B"
MODEL_NAME  = "Qwen/Qwen2.5-Math-1.5B"
BENCHMARK   = "asymob"
GLOBAL_SEED = 42

MAX_NEW_TOKENS = 3072      # tokens the model may GENERATE
MAX_MODEL_LEN  = 4096      # HARD ceiling: Qwen2.5-Math max_position_embeddings
CHUNK          = 512       # tasks per generate -> write cycle

DATA_FILE = f"{HOME}/GRIP/data/asymob_prepped.jsonl"
RUN_DIR   = f"{HOME}/GRIP/results/{RUN_ID}"
OUT_DIR   = f"{RUN_DIR}/eval/{BENCHMARK}"
OUT_FILE  = f"{OUT_DIR}/generations.jsonl"

INSTRUCTION = r"Please reason step by step, and put your final answer within \boxed{}."


def complete_prompt(problem):
    return f"{problem}\n{INSTRUCTION}"


# ── answer extraction (regex only: no sympy, no timeouts; verify.py parses) ──
_BOXED_RE = re.compile(r'\\boxed\s*\{')
_THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL)


def extract_boxed(text):
    """Brace-matched \\boxed{...}; takes the last one, after </think> if present."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    m = None
    for m in _BOXED_RE.finditer(text):
        pass
    if m is None:
        return None
    i = m.end()
    depth, start = 1, i
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    return text[start:i - 1].strip() if depth == 0 else None


def think_tokens(text, tokenizer):
    if "<think>" not in text:
        return 0
    m = _THINK_RE.search(text)
    return len(tokenizer.encode(m.group(1))) if m else 0


# ── tasks ─────────────────────────────────────────────────────────────────────
def build_tasks(rows, already_done):
    """One task per (row, sample_idx). n_samples comes from preprocessing."""
    tasks = []
    for row_idx, d in enumerate(rows):
        for sample_idx in range(d["n_samples"]):
            if (d["uid"], sample_idx) in already_done:
                continue
            seed = (GLOBAL_SEED + 1_000_003 * row_idx + 7919 * sample_idx) % (2**31 - 1)
            tasks.append({"row": d, "sample_idx": sample_idx, "seed": seed})
    return tasks


def _sum_logprob(comp):
    """vLLM 0.26 initialises cumulative_logprob to None unless SamplingParams
    .logprobs is set (v1/engine/logprobs.py:54). Fall back to summing per-token
    logprobs, and return None rather than a misleading 0.0 if neither exists."""
    if getattr(comp, "cumulative_logprob", None) is not None:
        return comp.cumulative_logprob
    lps = getattr(comp, "logprobs", None)
    if not lps:
        return None
    total = 0.0
    for tok_lp in lps:
        if not tok_lp:
            return None
        total += next(iter(tok_lp.values())).logprob
    return total


def build_row(task, comp, prompt_token_ids, tokenizer):
    d = task["row"]
    text   = comp.text
    pred   = extract_boxed(text)
    n_out  = len(comp.token_ids)
    sum_lp = _sum_logprob(comp)          # free; no logprobs=N request needed

    return {
        "run_id": RUN_ID, "ablation": "a0", "ckpt": MODEL_NAME, "benchmark": BENCHMARK,
        # ── identity / grouping (from preprocessing) ──
        "uid":          d["uid"],
        "group_id":     d["group_id"],
        "arm":          d["arm"],
        "is_original":  d["is_original"],
        "meta":         d.get("meta", {}),
        "sample_idx":   task["sample_idx"],
        "sample_seed":  task["seed"],
        # ── generation ──
        "completion":   text,
        "answer_raw":   pred,
        # ── gold travels with the row so verify.py never reads the dataset ──
        "gold_answers": d["gold_answers"],
        "gold_format":  d["gold_format"],
        # ── filled by verify.py ──
        "correct": None,
        "verify_status": "pending",
        # ── diagnostics ──
        "n_tokens_prompt":     len(prompt_token_ids),
        "n_tokens_completion": n_out,
        "n_tokens_think":      think_tokens(text, tokenizer),
        "truncated":     comp.finish_reason == "length",
        "format_ok":     pred is not None,
        "sum_logprob_completion": sum_lp,
        "n_tokens_logprob":       n_out,
        "mean_logprob": (sum_lp / n_out) if (sum_lp is not None and n_out > 0) else None,
    }


# ── main ──────────────────────────────────────────────────────────────────────
def run_eval(infile, outfile_path, llm, tokenizer):
    rows = []
    with open(infile) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Dataset rows: {len(rows)}", flush=True)

    uids = Counter(d["uid"] for d in rows)
    dupes = [u for u, n in uids.items() if n > 1]
    if dupes:
        raise ValueError(f"{len(dupes)} duplicate uids (e.g. {dupes[:5]}). "
                         f"Resume would silently skip rows. Fix preprocessing.")
    print(f"Originals: {sum(d['is_original'] for d in rows)}", flush=True)

    already_done = set()
    if os.path.exists(outfile_path):
        with open(outfile_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    already_done.add((r["uid"], r["sample_idx"]))
                except Exception:
                    pass
    print(f"Resume: {len(already_done)} rows already written", flush=True)

    tasks = build_tasks(rows, already_done)
    total_expected = sum(d["n_samples"] for d in rows)
    print(f"Total rows expected: {total_expected}", flush=True)
    print(f"Pending tasks: {len(tasks)} (skipping {total_expected - len(tasks)})", flush=True)
    if not tasks:
        print("Nothing to generate -- all rows complete.", flush=True)
        return

    # Shuffle: the file is ordered by family/intensity, so an interrupted run
    # would otherwise cover only the easy head of the benchmark.
    random.Random(GLOBAL_SEED).shuffle(tasks)

    written, t_start = 0, time.time()
    for start in range(0, len(tasks), CHUNK):
        chunk   = tasks[start:start + CHUNK]
        prompts = [complete_prompt(t["row"]["problem"]) for t in chunk]
        params  = [SamplingParams(temperature=0.6, top_p=0.95, n=1,
                                  max_tokens=MAX_NEW_TOKENS, seed=t["seed"], logprobs=0)
                   for t in chunk]

        t0 = time.time()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        t_gen = time.time() - t0

        out_rows, n_tok = [], 0
        for task, output in zip(chunk, outputs):
            comp = output.outputs[0]
            n_tok += len(comp.token_ids)
            out_rows.append(build_row(task, comp, output.prompt_token_ids, tokenizer))

        with open(outfile_path, "a") as f:
            for r in out_rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            os.fsync(f.fileno())

        # fail-fast: if the first chunk has no logprobs, abort before burning the run
        if start == 0 and all(r.get("sum_logprob_completion") is None for r in out_rows):
            raise RuntimeError(
                "sum_logprob_completion is None across the entire first chunk -- "
                "logprobs are not being captured. Aborting before burning the run.")

        written += len(out_rows)
        done    = start + len(chunk)
        rate    = done / (time.time() - t_start)
        eta_h   = (len(tasks) - done) / rate / 3600 if rate > 0 else float("inf")
        print(f"[{done}/{len(tasks)}] chunk {t_gen:.0f}s | {n_tok/len(chunk):.0f} tok/sample "
              f"| {rate*3600:.0f} rows/h | ETA {eta_h:.1f}h", flush=True)

    print(f"Wrote {written} new rows to {outfile_path}", flush=True)
    print(f"Total wall time: {(time.time()-t_start)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_commit = "unknown"

    manifest = {
        "run_id": RUN_ID, "ablation": "a0", "benchmark": BENCHMARK,
        "git_commit": git_commit,
        "vllm_version": "0.26.0",
        "timestamp_start": datetime.now(timezone.utc).isoformat(),
        "model": {"name": MODEL_NAME, "dtype": "bfloat16"},
        "data_file": DATA_FILE,
        "generation": {"temperature": 0.6, "top_p": 0.95,
                       "max_new_tokens": MAX_NEW_TOKENS, "max_model_len": MAX_MODEL_LEN,
                       "n_samples": "from preprocessing (8 originals / 1 variants)",
                       "per_sample_seed": "GLOBAL_SEED + 1000003*row_idx + 7919*sample_idx"},
        "scaffold": "boxed", "global_seed": GLOBAL_SEED,
        "answer_extraction": "regex_boxed", "verification": "offline (verify.py)",
        "bootstrap_unit": "group_id", "bootstrap_B": 10000,
    }
    with open(f"{RUN_DIR}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    llm = LLM(model=MODEL_NAME, gpu_memory_utilization=0.90,
              max_model_len=MAX_MODEL_LEN, tensor_parallel_size=1,
              enable_prefix_caching=True, seed=GLOBAL_SEED)
    run_eval(DATA_FILE, OUT_FILE, llm, llm.get_tokenizer())