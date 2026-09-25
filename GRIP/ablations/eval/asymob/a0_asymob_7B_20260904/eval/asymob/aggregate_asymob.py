"""
GRIP: aggregation for ASyMOB.

Reads  <run_dir>/verified_generations.jsonl
Writes <run_dir>/{items.jsonl, groups.jsonl, summary.json}

Structure: one original per group_id, many typed variants under it
(meta.family / meta.intensity / meta.is_stability). Metrics are means over a
variable-size variant set. All formulas come from metrics.py.

Usage: python aggregate_asymob.py --run_dir results/<RUN_ID>/eval/asymob
"""

from __future__ import annotations
import os, json, math, argparse
from collections import defaultdict, Counter

import numpy as np
import metrics as M

try:
    from math_verify import parse, verify
    _HAVE_MV = True
except Exception:
    _HAVE_MV = False


def transfer_group(family):
    """Cross-perturbation transfer grouping (spec 2.3.2)."""
    if family == "Original":       return "original"
    if family.startswith("Numeric"):     return "in_family"      # KV constant resampling
    if family.startswith("Symbolic"):    return "adjacent"       # abstracting upward
    if family.startswith("Equivalence"): return "out_of_family"  # the OOD claim
    return "other"


def surf_or_para(family):
    """Symbolic/Equivalence preserve the answer -> surface. Numeric resamples -> parametric."""
    if family.startswith("Numeric"):   return "para"
    if family.startswith("Symbolic") or family.startswith("Equivalence"): return "surf"
    return "other"


def _equiv_one(pair):
    """Worker: one symbolic equivalence check. Module-level so it pickles."""
    a, b = pair
    try:
        from math_verify import parse, verify
        return bool(verify(parse(f"${a}$"), parse(f"${b}$"),
                           strict=False, timeout_seconds=3))
    except Exception:
        return False


def equiv_answers(a, b):
    """Serial fallback: only used if the pooled path is unavailable."""
    if a is None or b is None:
        return False
    if str(a).strip() == str(b).strip():
        return True
    if not _HAVE_MV:
        return False
    return _equiv_one((a, b))


def batch_equiv(pairs, workers=8, timeout=6):
    """
    Resolve many (a, b) equivalence checks at once.

    ASyMOB needs ~35k of these (100 groups x ~350 variants), against golds like
    95**(-2*47)*((1 + 2*95**47)*log(...) - ...). sympy expands 95**47 into an exact
    93-digit integer while simplifying, so a single comparison can burn the full
    timeout. Run serially this would take tens of hours.

    Two mitigations: a string fast-path that skips sympy entirely when the modal
    answers are already identical, and a pebble pool that force-kills a worker whose
    comparison overruns (a C-level sympy hang ignores signal-based timeouts).
    Returns {(a, b) -> bool}.
    """
    out, todo = {}, []
    for a, b in pairs:
        if a is None or b is None:
            out[(a, b)] = False
        elif str(a).strip() == str(b).strip():
            out[(a, b)] = True          # fast path: no sympy needed
        else:
            todo.append((a, b))

    if not todo:
        return out
    if not _HAVE_MV:
        for k in todo:
            out[k] = False
        return out

    print(f"  answer agreement: {len(pairs)} pairs, {len(todo)} need sympy "
          f"({len(pairs)-len(todo)} resolved by string match)", flush=True)

    try:
        from pebble import ProcessPool, ProcessExpired
        from concurrent.futures import TimeoutError as FTimeout
        n_timeout = 0
        # max_tasks recycles workers so sympy's global caches cannot grow unbounded
        with ProcessPool(max_workers=workers, max_tasks=200) as pool:
            futs = {pool.schedule(_equiv_one, args=[pr], timeout=timeout): pr
                    for pr in todo}
            for fut, pr in futs.items():
                try:
                    out[pr] = fut.result()
                except (FTimeout, ProcessExpired):
                    out[pr] = False
                    n_timeout += 1
                except Exception:
                    out[pr] = False
        if n_timeout:
            print(f"  {n_timeout}/{len(todo)} comparisons timed out (counted as "
                  f"not-equivalent)", flush=True)
    except ImportError:
        print("  pebble not installed -- running serially WITHOUT hard timeouts; "
              "this may be very slow on ASyMOB", flush=True)
        for pr in todo:
            out[pr] = _equiv_one(pr)
    return out


def load_jsonl(p):
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _write_jsonl(p, rows):
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _mean_conf(samples):
    vals = [M.confidence(s["sum_logprob_completion"], s["n_tokens_logprob"])
            for s in samples
            if s.get("sum_logprob_completion") is not None and s.get("n_tokens_logprob")]
    return float(np.mean(vals)) if vals else float("nan")


# ── items: aggregate samples per uid ──────────────────────────────────────────
def build_items(generations, binarise="majority"):
    by_uid = defaultdict(list)
    for r in generations:
        by_uid[r["uid"]].append(r)

    items = []
    for uid, samples in by_uid.items():
        s0 = samples[0]
        # verifier failures are dropped from rate denominators, never counted wrong
        scored = [s for s in samples if s.get("correct") is not None]
        n = len(scored)
        n_attempted = len(samples)
        c = sum(int(s["correct"]) for s in scored) if n else 0

        answers = [s.get("answer_raw") for s in scored if s.get("answer_raw")]
        modal = Counter(answers).most_common(1)[0][0] if answers else None

        binarised = int((c / n) >= 0.5) if n else 0
        if binarise == "any":
            binarised = int(c > 0)

        meta = s0.get("meta", {})
        it = {
            "run_id": s0["run_id"], "ablation": s0["ablation"], "benchmark": s0["benchmark"],
            "uid": uid, "group_id": s0["group_id"], "arm": s0["arm"],
            "is_original": s0["is_original"],
            "family": meta.get("family"), "intensity": meta.get("intensity"),
            "is_stability": meta.get("is_stability", False),
            "n_correct": c,
            "n_total": n,                 # SCORABLE samples (verify_status == ok)
            "n_attempted": n_attempted,   # all samples generated for this item
            "avg_n": M.avg_at_n(c, n) if n else float("nan"),          # unscorable DROPPED
            "avg_n_strict": M.avg_at_n(c, n_attempted) if n_attempted else float("nan"),  # unscorable WRONG
            "binarised": binarised, "modal_answer": modal,
            "mean_n_tokens_completion": float(np.mean([s["n_tokens_completion"] for s in samples])),
            "truncation_rate": float(np.mean([int(s.get("truncated", False)) for s in samples])),
            "format_ok_rate": float(np.mean([int(s.get("format_ok", False)) for s in samples])),
            "extract_fail_rate": float(np.mean(
                [int(s.get("verify_status") == "extract_fail") for s in samples])),
            "mean_confidence": _mean_conf(samples),
        }
        for k in (1, 4, 8):
            if n >= k:
                it[f"pass_at_{k}"] = M.pass_at_k(c, n, k)
        # G-Pass@k (Liu et al. 2025): reliability, not just capability.
        #
        # Scored over attempted samples, not scorable ones. Guarding on scorable
        # n would restrict G-Pass to items where all 8 parsed - and those are
        # systematically the items the model handled cleanly, so the metric would
        # be computed on a biased subsample and read optimistically. Counting an
        # unextractable answer as wrong keeps the denominator at n_attempted for
        # every item (100% coverage) and is the right convention for a stability
        # measure: a model that cannot reliably emit a parseable answer has not
        # reliably solved the problem.
        if n_attempted >= 8:
            for tau in (0.5, 0.75, 1.0):
                it[f"g_pass_at_8_{tau}"] = M.g_pass_at_k(c, n_attempted, 8, tau=tau)
            it["mg_pass_at_8"] = M.mg_pass_at_k(c, n_attempted, 8)
        items.append(it)
    return items


# ── groups: one row per group_id ──────────────────────────────────────────────
def build_groups(items, skip_c_ans=False, workers=8, timeout=6):
    by_group = defaultdict(lambda: {"orig": None, "vars": []})
    for it in items:
        bucket = by_group[it["group_id"]]
        if it["is_original"]:
            bucket["orig"] = it
        else:
            bucket["vars"].append(it)

    groups, skipped = [], Counter()
    for gid, d in by_group.items():
        orig, variants = d["orig"], d["vars"]
        if orig is None:
            skipped["no_original"] += 1; continue
        if orig["n_total"] == 0:
            # Every original sample failed extraction. avg_n is NaN so the gap is
            # already dropped: but binarised falls through to 0, which would feed
            # RSR / recovery / R-hat a fabricated "model failed the original".
            # Absence of evidence is not evidence of failure: drop the group so
            # every metric reports the same denominator.
            skipped["original_unscorable"] += 1; continue
        if not variants:
            skipped["no_variants"] += 1; continue

        acc_orig = orig["avg_n"]
        acc_vars = np.asarray([v["avg_n"] for v in variants], dtype=float)

        g = {
            "group_id": gid, "n_variants": len(variants),
            # how many variants had >=1 scorable sample: if this is well below
            # n_variants the gap for this group rests on little evidence
            "n_variants_scored": int(np.sum(~np.isnan(acc_vars))),
            "acc_orig": acc_orig,
            "acc_mean_variants": float(np.nanmean(acc_vars)),
            "gap": M.robustness_gap(acc_orig, acc_vars),
            "within_var": M.within_seed_variance(acc_vars),
            "c_corr": M.correctness_consistency(orig["binarised"],
                                                [v["binarised"] for v in variants]),
            "e_i": orig["binarised"],
            "acc_by_arm": {v["arm"]: v["avg_n"] for v in variants},
            "gap_by_arm": {v["arm"]: acc_orig - v["avg_n"] for v in variants},
            "h_by_arm":   {v["arm"]: v["binarised"] for v in variants},
            "family_by_arm": {v["arm"]: (v["family"] or "") for v in variants},
        }
        # c_ans is deferred: collect the pairs now, resolve them all in one
        # pooled pass below, then fill the field in. Computing it inline would
        # mean ~35k serial sympy calls.
        if not skip_c_ans and orig.get("modal_answer"):
            g["_c_ans_pairs"] = [(orig["modal_answer"], v["modal_answer"])
                                 for v in variants if v.get("modal_answer")]
        groups.append(g)

    if skipped:
        print(f"skipped groups: {dict(skipped)}")

    if not skip_c_ans:
        all_pairs = [pr for g in groups for pr in g.get("_c_ans_pairs", [])]
        if all_pairs:
            resolved = batch_equiv(all_pairs, workers=workers, timeout=timeout)
            for g in groups:
                prs = g.pop("_c_ans_pairs", None)
                if prs:
                    g["c_ans"] = float(np.mean([int(resolved[pr]) for pr in prs]))
    else:
        for g in groups:
            g.pop("_c_ans_pairs", None)

    return groups


# ── summary ───────────────────────────────────────────────────────────────────
def build_summary(items, groups, binarise, B=10000):
    gap_ci    = M.cluster_bootstrap({g["group_id"]: g["gap"] for g in groups}, B=B)
    within_ci = M.cluster_bootstrap({g["group_id"]: g["within_var"] for g in groups}, B=B)
    ccorr_ci  = M.cluster_bootstrap({g["group_id"]: g["c_corr"] for g in groups}, B=B)
    frac_inv  = float(np.mean([int(g["c_corr"] == 1.0) for g in groups])) if groups else float("nan")

    # per-arm (variant_type)
    arms = sorted({a for g in groups for a in g["gap_by_arm"]})
    by_arm = {}
    for a in arms:
        e, h, gaps = [], [], []
        for g in groups:
            if a in g["h_by_arm"]:
                e.append(g["e_i"]); h.append(g["h_by_arm"][a]); gaps.append(g["gap_by_arm"][a])
        rr = M.r_hat(e, h)
        by_arm[a] = {"gap_mean": float(np.mean(gaps)) if gaps else float("nan"),
                     "rsr": M.relative_success_rate(e, h),
                     "recovery": M.recovery_rate(e, h),
                     "r_hat": rr["r_hat"], "r_hat_beta": rr["beta"], "n_groups": rr["N"]}

    # surface vs parametric: family read off the group's own map (O(1), no scan)
    def collect(pred):
        e, h = [], []
        for g in groups:
            for a, hi in g["h_by_arm"].items():
                if pred(g["family_by_arm"].get(a, "")):
                    e.append(g["e_i"]); h.append(hi)
        return e, h

    r_surf = M.r_hat(*collect(lambda f: surf_or_para(f) == "surf"))
    r_para = M.r_hat(*collect(lambda f: surf_or_para(f) == "para"))

    transfer = {}
    for name in ("in_family", "adjacent", "out_of_family"):
        e, h, gaps = [], [], []
        for g in groups:
            for a, hi in g["h_by_arm"].items():
                if transfer_group(g["family_by_arm"].get(a, "")) == name:
                    e.append(g["e_i"]); h.append(hi); gaps.append(g["gap_by_arm"][a])
        rr = M.r_hat(e, h)
        transfer[name] = {"gap_mean": float(np.mean(gaps)) if gaps else float("nan"),
                          "rsr": M.relative_success_rate(e, h),
                          "r_hat": rr["r_hat"], "n_groups": rr["N"]}

    # degradation curve: family x intensity
    deg = defaultdict(list)
    for it in items:
        if it.get("intensity") is not None and not it["is_original"] and not math.isnan(it["avg_n"]):
            deg[(it.get("family"), it["intensity"])].append(it["avg_n"])
    degradation = [{"family": f, "intensity": n, "acc": float(np.mean(v)), "n_items": len(v)}
                   for (f, n), v in sorted(deg.items(), key=lambda kv: (str(kv[0][0]), kv[0][1]))]

    return {
        "n_groups": len(groups), "binarisation_rule": binarise,
        "bootstrap_B": B, "bootstrap_unit": "group_id",
        "acc_orig_mean": float(np.nanmean([g["acc_orig"] for g in groups])),
        "acc_variants_mean": float(np.nanmean(
            [a for g in groups for a in g["acc_by_arm"].values()])),
        "gap_mean": gap_ci["mean"], "gap_ci": [gap_ci["ci_low"], gap_ci["ci_high"]],
        "gap_n": gap_ci["n"],
        "within_var_mean": within_ci["mean"],
        "within_var_ci": [within_ci["ci_low"], within_ci["ci_high"]],
        "within_var_n": within_ci["n"],
        "c_corr_mean": ccorr_ci["mean"], "c_corr_ci": [ccorr_ci["ci_low"], ccorr_ci["ci_high"]],
        "c_corr_n": ccorr_ci["n"],
        "frac_fully_invariant": frac_inv,
        # mG-Pass@8 is only defined for the n=8 arm (originals here)
        "mg_pass_at_8_orig_mean": float(np.nanmean(
            [it["mg_pass_at_8"] for it in items
             if it["is_original"] and "mg_pass_at_8" in it])) if any(
            it["is_original"] and "mg_pass_at_8" in it for it in items) else float("nan"),
        "by_arm": by_arm,
        "r_hat_surf": r_surf["r_hat"], "r_hat_para": r_para["r_hat"],
        "r_hat_global": M.r_hat_global(r_surf["r_hat"], r_para["r_hat"]),
        "transfer": transfer, "degradation_curve": degradation,
        "health": {
            "truncation_rate": float(np.mean([it["truncation_rate"] for it in items])),
            "format_ok_rate": float(np.mean([it["format_ok_rate"] for it in items])),
            "extract_fail_rate": float(np.mean([it["extract_fail_rate"] for it in items])),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--binarise", default="majority", choices=["majority", "any"])
    ap.add_argument("--bootstrap_B", type=int, default=10000)
    ap.add_argument("--skip_answer_agreement", action="store_true",
                    help="omit c_ans entirely (fastest; loses the answer-agreement metric)")
    ap.add_argument("--workers", type=int, default=8,
                    help="processes for the pooled c_ans comparisons")
    ap.add_argument("--equiv_timeout", type=int, default=6,
                    help="hard per-comparison kill timeout, seconds")
    args = ap.parse_args()

    gens = load_jsonl(os.path.join(args.run_dir, "verified_generations.jsonl"))
    print(f"loaded {len(gens)} generation rows")

    items = build_items(gens, binarise=args.binarise)
    # Write items before groups: build_groups is the expensive step, and a failure
    # there would otherwise discard the whole run.
    _write_jsonl(os.path.join(args.run_dir, "items.jsonl"), items)
    print(f"wrote items.jsonl ({len(items)} rows)", flush=True)

    groups  = build_groups(items, skip_c_ans=args.skip_answer_agreement,
                           workers=args.workers, timeout=args.equiv_timeout)
    _write_jsonl(os.path.join(args.run_dir, "groups.jsonl"), groups)
    print(f"wrote groups.jsonl ({len(groups)} rows)", flush=True)

    summary = build_summary(items, groups, args.binarise, args.bootstrap_B)
    with open(os.path.join(args.run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nitems: {len(items)}  groups: {len(groups)}")
    print("\n-- headline --")
    print(f"acc(orig)    = {summary['acc_orig_mean']:.3f}")
    print(f"acc(variant) = {summary['acc_variants_mean']:.3f}")
    print(f"gap          = {summary['gap_mean']:.3f} "
          f"[{summary['gap_ci'][0]:.3f}, {summary['gap_ci'][1]:.3f}] "
          f"(n={summary['gap_n']})")
    print(f"mG-Pass@8 (orig) = {summary['mg_pass_at_8_orig_mean']:.3f}")
    print(f"R_hat surf/para/global = {summary['r_hat_surf']:.3f} / "
          f"{summary['r_hat_para']:.3f} / {summary['r_hat_global']:.3f}")
    print(f"frac fully invariant = {summary['frac_fully_invariant']:.3f}")
    print(f"format_ok = {summary['health']['format_ok_rate']:.3f}  "
          f"truncation = {summary['health']['truncation_rate']:.3f}")


if __name__ == "__main__":
    main()