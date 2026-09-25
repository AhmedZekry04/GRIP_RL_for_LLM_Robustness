"""
GRIP: offline verification. Common to ASyMOB and MATH-Perturb.

Reads  <run_dir>/generations.jsonl   (from either eval_a0_*.py)
Writes <run_dir>/verified_generations.jsonl  with `correct` and `verify_status` filled.

Benchmark-agnostic by construction: every row carries `gold_answers` (a list of
acceptable answers, already normalised by preprocessing) and `gold_format`
("sympy" or "latex", which parser to try first). Nothing here inspects the
benchmark name or reads the original dataset.

Usage:
  python -u verify.py --run_dir results/<RUN_ID>/eval/<benchmark> --workers 8
"""

import os
import json
import argparse
from collections import Counter
from concurrent.futures import TimeoutError as FuturesTimeout

from pebble import ProcessPool, ProcessExpired
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig
from sympy import sympify


# ── parsing ───────────────────────────────────────────────────────────────────
# gold and pred must reach verify() as the same kind of object. verify() takes
# lists and returns True if any gold matches any pred - which is exactly the
# semantics needed for multi-valued golds like "0 or 4" (already split into
# ["0", "4"] by preprocessing).

def _parse_expr(s):
    """sympy-syntax string (**, log, ...) -> sympy object, or None."""
    try:
        return sympify(s)
    except Exception:
        p = parse(f"${s}$", extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()])
        return p[0] if p else None


def _parse_latex(s):
    """LaTeX string (\\frac, \\sqrt, ...) -> sympy object, or None."""
    p = parse(f"${s}$", extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
    if p:
        return p[0]
    try:
        return sympify(s)
    except Exception:
        return None


def parse_golds(gold_answers, gold_format):
    """List of normalised gold strings -> list of sympy objects."""
    if not gold_answers:
        return None
    fn = _parse_expr if gold_format == "sympy" else _parse_latex
    out = [g for g in (fn(str(s)) for s in gold_answers) if g is not None]
    return out or None


def parse_pred(answer_raw):
    """
    answer_raw is the raw content of the model's last \\boxed{...}. Almost always
    LaTeX regardless of benchmark, because that is what the model writes.
    """
    if answer_raw is None or answer_raw == "None":
        return None
    p = parse(f"${answer_raw}$",
              extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
    if p:
        return p
    try:
        return [sympify(answer_raw)]
    except Exception:
        return None


def score_pair(payload, mv_timeout):
    """Worker -> (correct, status). Failure modes split so the breakdown is diagnostic."""
    gold_answers, gold_format, answer_raw = payload

    if answer_raw is None or answer_raw == "None":
        return None, "extract_fail"

    try:
        gold = parse_golds(gold_answers, gold_format)
    except Exception as e:
        return None, f"gold_parse_fail:{type(e).__name__}"
    if not gold:
        return None, "gold_parse_fail:empty"

    try:
        pred = parse_pred(answer_raw)
    except Exception as e:
        return None, f"pred_parse_fail:{type(e).__name__}"
    if not pred:
        return None, "extract_fail"

    try:
        # Soft in-worker timeout for Python-level slowness. The pebble hard
        # timeout below is the backstop for C-level sympy hangs that signals
        # cannot interrupt. math_verify prints "Timeout during comparison" to
        # stderr when this fires: that is a log line, not an error.
        ok = verify(gold, pred, strict=False, timeout_seconds=mv_timeout)
    except Exception as e:
        return None, f"verify_fail:{type(e).__name__}"

    return int(bool(ok)), "ok"


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=8,
                    help="math_verify internal timeout (soft, in-worker)")
    ap.add_argument("--hard_timeout", type=int, default=None,
                    help="pebble process-kill timeout. Default: timeout + 5")
    args = ap.parse_args()

    hard_timeout = args.hard_timeout or (args.timeout + 5)
    in_path  = os.path.join(args.run_dir, "generations.jsonl")
    out_path = os.path.join(args.run_dir, "verified_generations.jsonl")

    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["uid"], r["sample_idx"]))
                except Exception:
                    pass
    if done:
        print(f"Resume: {len(done)} rows already verified, skipping", flush=True)

    rows = []
    with open(in_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if (r["uid"], r["sample_idx"]) not in done:
                rows.append(r)

    total = len(rows)
    print(f"Verifying {total} rows | workers={args.workers} "
          f"soft={args.timeout}s hard={hard_timeout}s", flush=True)
    if total == 0:
        print("Nothing to verify.", flush=True)
        return

    n_ok = n_correct = 0
    status_counts = Counter()
    BATCH = 500

    with open(out_path, "a") as out_f, \
         ProcessPool(max_workers=args.workers, max_tasks=500) as pool:

        for start in range(0, total, BATCH):
            batch = rows[start:start + BATCH]
            futures = {}
            for r in batch:
                fut = pool.schedule(
                    score_pair,
                    args=[(r.get("gold_answers"), r.get("gold_format", "latex"),
                           r.get("answer_raw")), args.timeout],
                    timeout=hard_timeout,   # terminates + respawns the worker on expiry
                )
                futures[fut] = r

            for fut, r in futures.items():
                try:
                    correct, status = fut.result()
                except FuturesTimeout:
                    correct, status = None, "timeout"
                except ProcessExpired as e:
                    correct, status = None, f"worker_crash:{e.exitcode}"
                except Exception as e:
                    correct, status = None, f"worker_fail:{type(e).__name__}"

                r["correct"] = correct
                r["verify_status"] = status
                out_f.write(json.dumps(r) + "\n")

                status_counts[status] += 1
                if status == "ok":
                    n_ok += 1
                    n_correct += (correct == 1)

            out_f.flush()
            os.fsync(out_f.fileno())

            seen = min(start + BATCH, total)
            print(f"  {seen}/{total}  scoreable={n_ok}  "
                  f"scoreable_acc={(n_correct/n_ok if n_ok else 0):.3f}  "
                  f"overall_acc={n_correct/seen:.3f}  "
                  f"timeouts={status_counts['timeout']}", flush=True)

    print(f"\nDone -> {out_path}", flush=True)
    print(f"rows this run: {total}", flush=True)
    for status, n in status_counts.most_common():
        print(f"  {status:24s} {n}", flush=True)
    print(f"\nscoreable (status=ok): {n_ok}", flush=True)
    print(f"correct:               {n_correct}", flush=True)
    if n_ok:
        print(f"scoreable_acc:         {n_correct/n_ok:.4f}   (parseable answers only)", flush=True)
    # An unparseable answer is a wrong answer, not an excluded one.
    print(f"overall_acc:           {n_correct/total:.4f}   (non-ok counted wrong)", flush=True)


if __name__ == "__main__":
    main()