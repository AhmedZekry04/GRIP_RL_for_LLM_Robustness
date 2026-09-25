"""
GRIP: aggregation for MATH-Perturb.

Reads  <run_dir>/verified_generations.jsonl
Writes <run_dir>/{items.jsonl, groups.jsonl, summary.json}

Separate from aggregate_asymob.py because the group structures differ. ASyMOB
has one original and many typed variants per group, so its metrics average over
a variable-size variant set keyed by family/intensity. MATH-Perturb has exactly
three fixed arms per problem - original, simple, hard - with no taxonomy, so
every metric is a paired comparison. metrics.py is shared and unchanged.

Usage: python aggregate_mathperturb.py --run_dir results/<RUN_ID>/eval/MATH_Perturb
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

PERTURBED_ARMS = ("simple", "hard")


def equiv_answers(a, b):
    if a is None or b is None:
        return False
    if not _HAVE_MV:
        return str(a) == str(b)
    try:
        return bool(verify(parse(f"${a}$"), parse(f"${b}$"), strict=False, timeout_seconds=5))
    except Exception:
        return str(a) == str(b)


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


# ── items ─────────────────────────────────────────────────────────────────────
def build_items(generations, binarise="majority"):
    by_uid = defaultdict(list)
    for r in generations:
        by_uid[r["uid"]].append(r)

    items = []
    for uid, samples in by_uid.items():
        s0 = samples[0]
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
            "subject": meta.get("subject"), "level": meta.get("level"),
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


# ── groups ────────────────────────────────────────────────────────────────────
def build_groups(items):
    by_group = defaultdict(dict)
    for it in items:
        by_group[it["group_id"]][it["arm"]] = it

    groups, skipped = [], Counter()
    for gid, amap in by_group.items():
        orig = next((v for v in amap.values() if v["is_original"]), None)
        if orig is None:
            skipped["no_original"] += 1; continue
        if orig["n_total"] == 0:
            # Every original sample failed extraction. avg_n is NaN so the gap is
            # already dropped: but binarised falls through to 0, which would feed
            # RSR / recovery / R-hat a fabricated "model failed the original".
            # Absence of evidence is not evidence of failure: drop the group so
            # every metric reports the same denominator.
            skipped["original_unscorable"] += 1; continue
        pert = {a: amap[a] for a in PERTURBED_ARMS if a in amap}
        if not pert:
            skipped["no_perturbed"] += 1; continue

        acc_orig = orig["avg_n"]
        acc_pert = [v["avg_n"] for v in pert.values()]

        g = {
            "group_id": gid,
            "n_perturbed_scored": int(np.sum(~np.isnan(np.asarray(acc_pert, dtype=float)))),
            "subject": orig.get("subject"), "level": orig.get("level"),
            "arms_present": sorted(pert.keys()),
            "acc_original": acc_orig,
            "acc_mean_perturbed": float(np.nanmean(acc_pert)),
            "gap_mean": M.robustness_gap(acc_orig, acc_pert),
            "within_var": M.within_seed_variance(acc_pert),
            "c_corr": M.correctness_consistency(orig["binarised"],
                                                [v["binarised"] for v in pert.values()]),
            "e_i": orig["binarised"],
            "acc_by_arm": {a: v["avg_n"] for a, v in pert.items()},
            "gap_by_arm": {a: acc_orig - v["avg_n"] for a, v in pert.items()},
            "h_by_arm":   {a: v["binarised"] for a, v in pert.items()},
        }
        if orig.get("modal_answer"):
            g["c_ans"] = M.answer_agreement(
                orig["modal_answer"],
                [v.get("modal_answer") for v in pert.values() if v.get("modal_answer")],
                equiv_answers)
        groups.append(g)

    if skipped:
        print(f"skipped groups: {dict(skipped)}")
    return groups


# ── summary ───────────────────────────────────────────────────────────────────
def build_summary(items, groups, binarise, B=10000):
    acc_by_arm = {}
    for arm in ("original",) + PERTURBED_ARMS:
        vals = [it["avg_n"] for it in items if it["arm"] == arm and not math.isnan(it["avg_n"])]
        acc_by_arm[arm] = float(np.mean(vals)) if vals else float("nan")

    mg_by_arm = {}
    for arm in ("original",) + PERTURBED_ARMS:
        vals = [it["mg_pass_at_8"] for it in items
                if it["arm"] == arm and "mg_pass_at_8" in it]
        mg_by_arm[arm] = float(np.nanmean(vals)) if vals else float("nan")

    by_arm = {}
    for arm in PERTURBED_ARMS:
        e, h, gaps, gids = [], [], [], []
        for g in groups:
            if arm in g["h_by_arm"]:
                e.append(g["e_i"]); h.append(g["h_by_arm"][arm])
                gaps.append(g["gap_by_arm"][arm]); gids.append(g["group_id"])
        rr = M.r_hat(e, h)
        ci = M.cluster_bootstrap({p: v for p, v in zip(gids, gaps)}, B=B)
        by_arm[arm] = {"n_groups": len(gids),
                       "gap_mean": ci["mean"], "gap_ci": [ci["ci_low"], ci["ci_high"]],
                       "rsr": M.relative_success_rate(e, h),
                       "recovery": M.recovery_rate(e, h),
                       "r_hat": rr["r_hat"], "r_hat_beta": rr["beta"]}

    # 'simple' preserves the solution strategy (surface); 'hard' requires a new
    # one (parametric): the same split aggregate_asymob draws for R-hat.
    r_s = by_arm.get("simple", {}).get("r_hat", float("nan"))
    r_h = by_arm.get("hard", {}).get("r_hat", float("nan"))
    r_global = (M.r_hat_global(r_s, r_h)
                if not (math.isnan(r_s) or math.isnan(r_h)) else float("nan"))

    gap_ci    = M.cluster_bootstrap({g["group_id"]: g["gap_mean"] for g in groups}, B=B)
    within_ci = M.cluster_bootstrap({g["group_id"]: g["within_var"] for g in groups}, B=B)
    ccorr_ci  = M.cluster_bootstrap({g["group_id"]: g["c_corr"] for g in groups}, B=B)
    frac_inv  = float(np.mean([int(g["c_corr"] == 1.0) for g in groups])) if groups else float("nan")

    # is 'hard' actually harder than 'simple' on the same problems?
    both = [g for g in groups if "simple" in g["gap_by_arm"] and "hard" in g["gap_by_arm"]]
    hms = (M.paired_cluster_bootstrap(
        {g["group_id"]: g["gap_by_arm"]["hard"] - g["gap_by_arm"]["simple"] for g in both}, B=B)
        if both else None)

    def strat(key):
        out, buckets = {}, defaultdict(list)
        for g in groups:
            if g.get(key) is not None:
                buckets[g[key]].append(g)
        for name, gs in sorted(buckets.items(), key=lambda kv: str(kv[0])):
            row = {"n_groups": len(gs),
                   "acc_original": float(np.nanmean([g["acc_original"] for g in gs]))}
            for arm in PERTURBED_ARMS:
                accs = [g["acc_by_arm"][arm] for g in gs if arm in g["acc_by_arm"]]
                gaps = [g["gap_by_arm"][arm] for g in gs if arm in g["gap_by_arm"]]
                row[f"acc_{arm}"] = float(np.nanmean(accs)) if accs else float("nan")
                row[f"gap_{arm}"] = float(np.nanmean(gaps)) if gaps else float("nan")
            out[str(name)] = row
        return out

    summary = {
        "n_groups": len(groups), "binarisation_rule": binarise,
        "bootstrap_B": B, "bootstrap_unit": "group_id",
        "acc_by_arm": acc_by_arm,
        "gap_mean": gap_ci["mean"], "gap_ci": [gap_ci["ci_low"], gap_ci["ci_high"]],
        "gap_n": gap_ci["n"],
        "within_var_mean": within_ci["mean"],
        "within_var_ci": [within_ci["ci_low"], within_ci["ci_high"]],
        "within_var_n": within_ci["n"],
        "c_corr_mean": ccorr_ci["mean"], "c_corr_ci": [ccorr_ci["ci_low"], ccorr_ci["ci_high"]],
        "c_corr_n": ccorr_ci["n"],
        "frac_fully_invariant": frac_inv,
        "mg_pass_at_8_by_arm": mg_by_arm,
        "by_arm": by_arm,
        "r_hat_simple": r_s, "r_hat_hard": r_h, "r_hat_global": r_global,
        "by_subject": strat("subject"), "by_level": strat("level"),
        "health": {
            "truncation_rate": float(np.mean([it["truncation_rate"] for it in items])),
            "format_ok_rate": float(np.mean([it["format_ok_rate"] for it in items])),
            "extract_fail_rate": float(np.mean([it["extract_fail_rate"] for it in items])),
        },
    }
    if hms is not None:
        summary["hard_minus_simple_gap"] = {
            "mean": hms["mean"], "ci": [hms["ci_low"], hms["ci_high"]], "n_groups": hms["n"]}
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--binarise", default="majority", choices=["majority", "any"])
    ap.add_argument("--bootstrap_B", type=int, default=10000)
    args = ap.parse_args()

    gens = load_jsonl(os.path.join(args.run_dir, "verified_generations.jsonl"))
    print(f"loaded {len(gens)} generation rows")

    items   = build_items(gens, binarise=args.binarise)
    groups  = build_groups(items)
    summary = build_summary(items, groups, args.binarise, args.bootstrap_B)

    _write_jsonl(os.path.join(args.run_dir, "items.jsonl"), items)
    _write_jsonl(os.path.join(args.run_dir, "groups.jsonl"), groups)
    with open(os.path.join(args.run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nitems: {len(items)}  groups: {len(groups)}")
    print("\n-- headline --")
    for arm, acc in summary["acc_by_arm"].items():
        print(f"acc({arm:8s}) = {acc:.3f}")
    for arm in PERTURBED_ARMS:
        if arm in summary["by_arm"]:
            b = summary["by_arm"][arm]
            print(f"{arm:8s}: gap={b['gap_mean']:.3f} "
                  f"[{b['gap_ci'][0]:.3f}, {b['gap_ci'][1]:.3f}]  "
                  f"RSR={b['rsr']:.3f}  R_hat={b['r_hat']:.3f}")
    print(f"R_hat_global = {summary['r_hat_global']:.3f}")
    mg = summary["mg_pass_at_8_by_arm"]
    print("mG-Pass@8    = " + "  ".join(f"{a}={v:.3f}" for a, v in mg.items()))
    if "hard_minus_simple_gap" in summary:
        d = summary["hard_minus_simple_gap"]
        print(f"hard-simple gap = {d['mean']:.3f} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]")
    print(f"format_ok = {summary['health']['format_ok_rate']:.3f}  "
          f"truncation = {summary['health']['truncation_rate']:.3f}")


if __name__ == "__main__":
    main()