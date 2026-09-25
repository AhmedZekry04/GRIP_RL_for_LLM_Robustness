"""
DeepMathGAP -> training-ready jsonl for TRL GRPO.

THE PROBLEM
-----------
trl.rewards.accuracy_reward parses the gold with math_verify.parse(solution),
passing the raw string straight through. math_verify needs an ANCHOR to know a
string contains maths - \\boxed{...}, $...$ or \\[...\\]. DeepMath stores golds
bare, so:

    parse("\\dfrac{1}{2}")    -> []            gold unparseable -> reward=None,
                                               the example is silently SKIPPED
    parse("$\\dfrac{1}{2}$")  -> [1/2, ...]    a sympy Rational -> verify() works

The model's own output is fine either way because it contains \\boxed{}. Only the
gold side is broken, and only because it lacks a marker.

Measured on this dataset: bare golds parse for 74.7% of rows; wrapped, 99.996%.

THE FIX
-------
Wrap every gold in $...$ and store it in a NEW `solution` column. `answer` is
left exactly as it was, so nothing downstream that reads it changes.

Done here rather than in the training script because accuracy_reward parses the
dataset column directly - there is no hook to reach into it - and because doing
it once means A1, A2 and A3 all train on byte-identical data.

Usage:
  python -u preprocess_deepmathgap.py                 # write the prepped file
  python -u preprocess_deepmathgap.py --report_only   # diagnose, write nothing
  python -u preprocess_deepmathgap.py --limit 5000    # quick pass
"""

import argparse
import json
import os
import time
from collections import Counter, defaultdict

from math_verify import parse

HOME     = os.path.expanduser("~")
IN_PATH  = f"{HOME}/GRIP/data/deepmathgap.jsonl"
OUT_PATH = f"{HOME}/GRIP/data/deepmathgap_prepped.jsonl"


def wrap_gold(raw):
    """
    Add the $...$ anchor math_verify needs. Idempotent.

    Returns "" for golds that cannot be salvaged; the caller drops those rows.
    """
    s = str(raw).strip()              # some golds arrive as ints, e.g. 0
    if not s:
        return ""                     # empty answer -> unusable
    if s.startswith("\\(") and s.endswith("\\)"):
        s = s[2:-2].strip()           # \( ... \) is LaTeX's other inline delimiter;
                                      # nesting it inside $...$ would break parsing
    s = " ".join(s.split())           # collapse embedded newlines/tabs
    if s.startswith("$") and s.endswith("$"):
        return s                      # already anchored -- don't create $$...$$
    return f"${s}$"


def classify(wrapped):
    """
    -> "ok" | "string_only" | "failed"

    math_verify returns the raw string when it cannot build a maths object (e.g.
    a gold like 'Entropy'). Those parse, but can only ever match by exact text,
    so they are near-useless as reward signal - worth counting separately from
    a hard failure.
    """
    if not wrapped:
        return "failed"
    try:
        p = parse(wrapped, parsing_timeout=5)
    except Exception:
        return "failed"
    if not p:
        return "failed"
    return "ok" if any(not isinstance(x, str) for x in p) else "string_only"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", default=IN_PATH)
    ap.add_argument("--out_path", default=OUT_PATH)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--report_only", action="store_true")
    ap.add_argument("--drop_mode", choices=["seed", "row", "none"], default="seed",
                    help="seed (default): if ANY row of a seed has unusable gold, drop the "
                         "WHOLE seed -- keeps all 4 arms intact and makes A1/A2 lose "
                         "identical seeds. row: drop only the offending rows (biases the "
                         "arms). none: keep everything (unusable rows waste rollouts).")
    args = ap.parse_args()

    t0 = time.time()
    print(f"reading {args.in_path}", flush=True)

    rows = []
    with open(args.in_path) as f:
        for i, line in enumerate(f):
            if args.limit is not None and i >= args.limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"loaded {len(rows)} rows", flush=True)

    by_arm  = defaultdict(Counter)     # original vs perturbed
    by_type = defaultdict(Counter)     # answer_type
    examples = defaultdict(list)

    for n, r in enumerate(rows):
        wrapped = wrap_gold(r.get("answer"))
        kind = classify(wrapped)

        # NEW column. `answer` is untouched.
        r["solution"]       = wrapped
        r["gold_status"]    = kind

        arm = "original" if r.get("type") == "original" else "perturbed"
        by_arm[arm][kind] += 1
        by_type[r.get("answer_type", "?")][kind] += 1
        if kind != "ok" and len(examples[kind]) < 8:
            examples[kind].append(f"[{arm}/{r.get('answer_type')}] {str(r.get('answer'))[:50]!r}")

        if (n + 1) % 25000 == 0:
            print(f"  {n+1}/{len(rows)}  ({time.time()-t0:.0f}s)", flush=True)

    # ── report ──
    def table(title, d):
        print(f"\n{title}")
        print(f"{'':<16} {'ok':>8} {'string':>8} {'failed':>8} {'ok%':>8}")
        for k in sorted(d, key=str):
            c = d[k]
            tot = sum(c.values())
            print(f"{str(k):<16} {c['ok']:>8} {c['string_only']:>8} "
                  f"{c['failed']:>8} {c['ok']/max(tot,1):>7.2%}")

    tot = Counter()
    for c in by_arm.values():
        tot.update(c)
    n = len(rows)
    print(f"\n{'='*58}")
    print(f"ok (parses to a maths object): {tot['ok']:>7}  {tot['ok']/n:.3%}")
    print(f"string only:                   {tot['string_only']:>7}  {tot['string_only']/n:.3%}")
    print(f"failed:                        {tot['failed']:>7}  {tot['failed']/n:.3%}")
    print("="*58)

    table("by arm  (a large gap here would be a confound between A1 and A2)", by_arm)
    table("by answer_type", by_type)

    for kind in ("failed", "string_only"):
        if examples[kind]:
            print(f"\n{kind} examples:")
            for e in examples[kind]:
                print(f"  {e}")

    o = by_arm["original"]; p = by_arm["perturbed"]
    ro = o["ok"] / max(sum(o.values()), 1)
    rp = p["ok"] / max(sum(p.values()), 1)
    print(f"\nusable rate: originals {ro:.2%} vs perturbed {rp:.2%} (delta {ro-rp:+.2%})")
    if abs(ro - rp) > 0.02:
        print("  NOTE: the arms lose different fractions of their reward signal.")
        print("  A1 trains on originals only, A2 on both -- worth reporting.")

    if args.report_only:
        print("\n--report_only: nothing written")
        return

    # ── selection ────────────────────────────────────────────────────────────
    # Row-wise dropping looks cheaper but is BIASED. The same perturbation --
    # "the answer is a renamed variable": lands in both k=1 (surface_gs) and
    # k=2 (surface_dlm), but math_verify parses gibberish names as symbols while
    # bailing on real maths words:
    #
    #     k=1 'w8h7cx'      -> parses  -> KEPT
    #     k=2 'eigenvalue'  -> string  -> dropped
    #
    # So row-wise dropping strips surface_dlm and spares surface_gs, removing one
    # perturbation family from training on an accident of naming. Seed-wise
    # dropping costs ~4x the rows and avoids that entirely.
    bad_seeds = {r["id"] for r in rows if r["gold_status"] != "ok"}

    if args.drop_mode == "seed":
        keep = [r for r in rows if r["id"] not in bad_seeds]
        print(f"\ndrop_mode=seed: {len(bad_seeds)} seeds have >=1 unusable gold "
              f"({len(bad_seeds)/len(set(r['id'] for r in rows)):.2%} of seeds)")
    elif args.drop_mode == "row":
        keep = [r for r in rows if r["gold_status"] == "ok"]
        print("\ndrop_mode=row: WARNING -- this removes surface_dlm rows "
              "preferentially; the arms will not be balanced")
    else:
        keep = rows
        print("\ndrop_mode=none: unusable rows kept; accuracy_reward will return "
              "None for them and their rollouts are wasted")

    print(f"keeping {len(keep)}/{len(rows)} rows ({len(keep)/n:.2%})")

    # confirm every surviving seed still has all its arms
    per_seed = defaultdict(set)
    for r in keep:
        per_seed[r["id"]].add(r["k"])
    sizes = Counter(len(v) for v in per_seed.values())
    print(f"surviving seeds: {len(per_seed)}   arms per seed: {dict(sorted(sizes.items()))}")
    if args.drop_mode == "seed" and len(sizes) > 1:
        print("  NOTE: some seeds were already incomplete in the source data")

    # arm balance after the drop: this is the number that matters for A1 vs A2
    arm_after = Counter("original" if r.get("type") == "original" else "perturbed"
                        for r in keep)
    k_after = Counter(r.get("k") for r in keep)
    print(f"after drop -- arms: {dict(arm_after)}   by k: {dict(sorted(k_after.items(), key=str))}")

    tmp = args.out_path + ".tmp"
    with open(tmp, "w") as f:
        for r in keep:
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, args.out_path)        # atomic: never a half-written file
    print(f"wrote {len(keep)} rows to {args.out_path}")

    # write the dropped rows out so the exclusions are auditable, not invisible
    keep_ids = {(id(r)) for r in keep}
    dropped = [r for r in rows if id(r) not in keep_ids] if args.drop_mode != "none" else []
    if dropped:
        drop_path = args.out_path.replace(".jsonl", "_dropped.jsonl")
        with open(drop_path, "w") as f:
            for r in dropped:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(dropped)} dropped rows to {drop_path}")

    print("\nsample row:")
    print(json.dumps({k: keep[0][k] for k in
                      ("id", "k", "type", "answer", "solution", "gold_status")}, indent=2))
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()