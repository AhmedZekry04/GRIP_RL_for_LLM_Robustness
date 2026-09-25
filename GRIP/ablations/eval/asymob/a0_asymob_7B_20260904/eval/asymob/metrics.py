"""
GRIP: metric definitions (pure functions).

Every formula lives here exactly once. aggregate_asymob.py, aggregate_mathperturb.py
and any analysis notebook import from this module so definitions cannot drift.
Functions are pure: they take arrays/lists and return numbers, no file IO, no
globals. This makes them unit-testable and safe to reuse across A0-A3.

Conventions
-----------
- "correct" is 0/1 for a scored sample, or None when verify_status != "ok".
  None samples are dropped before any rate is computed - never counted as wrong.
  (verify.py separately prints an overall_acc that does count them wrong; the two
  conventions are intentional; state which one a reported number uses.)
- A "group" is all rows sharing group_id (ASyMOB: seed id; MATH-Perturb: problem id).
- Binarisation for group-level metrics: majority-over-n (c/n >= 0.5).
- All aggregate CIs use a cluster bootstrap over groups, never over items.
"""

from __future__ import annotations
import math
from collections import Counter
from typing import Sequence

import numpy as np


# Accuracy family: operate on per-item (c, n) counts

def avg_at_n(c: int, n: int) -> float:
    """avg@n for one item: fraction of n samples that were correct."""
    if n == 0:
        return float("nan")
    return c / n


def pass_at_k(c: int, n: int, k: int) -> float:
    """
    Unbiased pass@k estimator (Chen et al. 2021).

        pass@k = 1 - C(n-c, k) / C(n, k)

    n = samples drawn, c = number correct, k = budget being estimated (k <= n).
    Computed in the numerically stable product form to avoid huge binomials:
        1 - prod_{i=0}^{k-1} (n-c-i)/(n-i)
    If n-c < k every k-subset contains a correct sample -> pass@k = 1.
    """
    if k > n:
        raise ValueError(f"pass@{k} requires n>={k}, got n={n}")
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod


def maj_at_k(answers: Sequence[str], golds_equiv, k: int | None = None) -> int:
    """
    Majority-vote correctness (cons@k / maj@k).

    answers    : normalised answer strings from k samples of one item
    golds_equiv: callable(answer_str) -> bool, True if that answer is correct
                 (uses symbolic equivalence, not string equality)
    Returns 1 if the modal answer is correct, else 0.
    Ties broken by Counter.most_common order (first-seen wins).
    """
    if not answers:
        return 0
    if k is not None:
        answers = answers[:k]
    modal, _ = Counter(answers).most_common(1)[0]
    return int(golds_equiv(modal))


# Stability family: G-Pass@k (Liu et al. 2025, arXiv 2412.13147)
#
# "Are Your LLMs Capable of Stable Reasoning?" - pass@k measures whether the
# model can solve an item; G-Pass@k measures whether it does so reliably. Both
# read off the same (c, n) counts, so this costs no extra generation.

def g_pass_at_k(c: int, n: int, k: int, tau: float = 1.0) -> float:
    """
    G-Pass@k_tau: probability that a random k-subset of the n samples contains
    at least ceil(tau*k) correct solutions.

        G-Pass@k_tau = sum_{j=ceil(tau*k)}^{k} C(c,j) * C(n-c, k-j) / C(n,k)

    tau = 1.0 is the strict form (all k correct), equal to C(c,k)/C(n,k).
    tau < 1.0 tolerates up to k - ceil(tau*k) failures within the k draws.
    Hypergeometric (sampling without replacement from the n we actually drew),
    which is the paper's approximation to the latent binomial.

    math.comb returns 0 when its second argument exceeds the first, so the
    impossible terms vanish without special-casing.
    """
    if k > n:
        raise ValueError(f"G-Pass@{k} requires n>={k}, got n={n}")
    if not (0.0 < tau <= 1.0):
        raise ValueError(f"tau must be in (0, 1], got {tau}")
    denom = math.comb(n, k)
    if denom == 0:
        return float("nan")
    lo = math.ceil(tau * k)
    num = sum(math.comb(c, j) * math.comb(n - c, k - j) for j in range(lo, k + 1))
    return num / denom


def mg_pass_at_k(c: int, n: int, k: int) -> float:
    """
    mG-Pass@k: G-Pass@k_tau integrated over tau in [0.5, 1.0].

        mG-Pass@k = 2 * int_0.5^1.0 G-Pass@k_tau d_tau
                  ~= (2/k) * sum_{i=floor(0.5k)+1}^{k} G-Pass@k_{i/k}

    A single number combining performance potential and reasoning stability.
    Ignores the regime where the model is right less than half the time, which
    is why it is a stability measure rather than an accuracy one.
    """
    if k > n:
        raise ValueError(f"mG-Pass@{k} requires n>={k}, got n={n}")
    lo = math.floor(0.5 * k) + 1
    if lo > k:
        return float("nan")
    return (2.0 / k) * sum(g_pass_at_k(c, n, k, tau=i / k) for i in range(lo, k + 1))


# Robustness family: operate on per-group aggregates

def robustness_gap(acc_orig: float, acc_variants: Sequence[float]) -> float:
    """
    Per-group robustness gap  D_i = acc(orig) - nanmean_v acc(variant_v).
    Positive = variants are harder than the original (the usual case).

    nanmean, not mean: an item with zero scorable samples has avg_n = NaN, and
    with n=1 variants (ASyMOB) a single unparseable answer would otherwise
    poison the whole group's gap. NaN only if the original is NaN or every
    variant is NaN.
    """
    acc_variants = np.asarray(acc_variants, dtype=float)
    if acc_variants.size == 0 or np.all(np.isnan(acc_variants)):
        return float("nan")
    return acc_orig - float(np.nanmean(acc_variants))


def within_seed_variance(acc_variants: Sequence[float]) -> float:
    """
    Var_v acc(variant_v): eval-side analogue of the training CVP penalty.
    nanvar for the same reason as robustness_gap.
    """
    acc_variants = np.asarray(acc_variants, dtype=float)
    if acc_variants.size == 0 or np.all(np.isnan(acc_variants)):
        return float("nan")
    return float(np.nanvar(acc_variants))


def correctness_consistency(binarised_orig: int,
                            binarised_variants: Sequence[int]) -> float:
    """
    C_corr = fraction of variants whose correctness matches the original's.
    Both sides are 0/1 (majority-over-n binarised).
    """
    if len(binarised_variants) == 0:
        return float("nan")
    return float(np.mean([int(b == binarised_orig) for b in binarised_variants]))


def answer_agreement(modal_orig: str,
                     modal_variants: Sequence[str],
                     equiv) -> float:
    """
    C_ans = fraction of variants whose modal answer is symbolically-equivalent
    to the original's modal answer. Catches consistent-but-wrong behaviour that
    correctness_consistency misses. `equiv(a, b) -> bool` uses math_verify.
    """
    if len(modal_variants) == 0:
        return float("nan")
    return float(np.mean([int(equiv(modal_orig, m)) for m in modal_variants]))


def relative_success_rate(e: Sequence[int], h: Sequence[int]) -> float:
    """
    RSR = Pr[variant correct | original correct]
        = sum_i e_i h_i / sum_i e_i
    e_i = 1 if original i solved, h_i = 1 if the variant of i solved.
    Conditions away capability - the cleanest single robustness number.
    Returns nan if no originals were solved (denominator 0).
    """
    e = np.asarray(e); h = np.asarray(h)
    denom = e.sum()
    if denom == 0:
        return float("nan")
    return float((e * h).sum() / denom)


def recovery_rate(e: Sequence[int], h: Sequence[int]) -> float:
    """
    Pr[variant correct | original incorrect]. Usually ~0. If high, the
    perturbation is easier than the original - a dataset red flag.
    """
    e = np.asarray(e); h = np.asarray(h)
    denom = (1 - e).sum()
    if denom == 0:
        return float("nan")
    return float(((1 - e) * h).sum() / denom)


# MATH-Perturb R-hat: saturating robustness score

def r_hat(e: Sequence[int], h: Sequence[int], k: float = 0.5) -> dict:
    """
    MATH-Perturb robustness score R_hat in (0, 1].

    e, h : paired binary vectors over N groups (same index = same group).
           e_j=1 iff original solved, h_j=1 iff perturbed solved.
    Returns dict with r_hat, beta, d_tilde, N so the adaptive scale is reported
    alongside the score (R_hat is not comparable across benchmarks without beta).

    Steps (Huang et al. 2025):
      p_e, p_h  : Jeffreys-smoothed accuracies  (+1/2 over N+1)
      sigma     : pooled std
      d_j       : normalised drop (e_j - h_j)/sigma
      d_hat_j   : softplus(k*d_j)/k - clamps improvements, passes drops
      beta      : ln2 / median_positive_drop - typical drop scores 0.5
      R_hat     : mean_j exp(-beta * d_hat_j)
    """
    e = np.asarray(e, dtype=float)
    h = np.asarray(h, dtype=float)
    N = len(e)
    if N == 0:
        return {"r_hat": float("nan"), "beta": float("nan"),
                "d_tilde": float("nan"), "N": 0}

    p_e = (e.sum() + 0.5) / (N + 1)
    p_h = (h.sum() + 0.5) / (N + 1)
    sigma = math.sqrt(0.5 * (p_e * (1 - p_e) + p_h * (1 - p_h)))
    if sigma == 0:
        sigma = 1e-8

    d = (e - h) / sigma
    # softplus(k*d)/k, computed stably to avoid overflow in exp for large k*d
    kd = k * d
    d_hat = np.where(kd > 30, d, np.log1p(np.exp(np.minimum(kd, 30))) / k)

    pos = d[d > 0]
    if len(pos) == 0:
        # no drops anywhere -> perfect robustness
        return {"r_hat": 1.0, "beta": float("nan"), "d_tilde": 0.0, "N": N}
    d_tilde = float(np.median(pos))
    beta = math.log(2) / d_tilde if d_tilde > 0 else float("inf")

    r = float(np.mean(np.exp(-beta * d_hat)))
    return {"r_hat": r, "beta": beta, "d_tilde": d_tilde, "N": N}


def r_hat_global(r_surf: float, r_para: float) -> float:
    """Geometric mean of surface and parametric R-hat sub-scores."""
    if r_surf is None or r_para is None:
        return float("nan")
    if math.isnan(r_surf) or math.isnan(r_para):
        return float("nan")
    return math.sqrt(r_surf * r_para)


# Confidence / likelihood / memorisation: from sum_logprob and n_tokens

def perplexity(sum_logprob: float, n_tokens: int) -> float:
    """PPL = exp(-sum_logprob / n_tokens)."""
    if n_tokens == 0:
        return float("nan")
    return math.exp(-sum_logprob / n_tokens)


def confidence(sum_logprob: float, n_tokens: int) -> float:
    """
    Sequence confidence = geometric mean token probability
                        = exp(sum_logprob / n_tokens)
                        = 1 / perplexity   (exact reciprocal).
    """
    if n_tokens == 0:
        return float("nan")
    return math.exp(sum_logprob / n_tokens)


def ppi(sum_lp_prompt: float, n_prompt: int,
        sum_lp_answer: float, n_answer: int) -> float:
    """
    Perplexity Paradox Index = log PPL_prompt - log PPL_answer.
    Simplifies to (-sum_lp_prompt/n_prompt) - (-sum_lp_answer/n_answer):
    no exp, no overflow. Positive = memorisation signature.
    """
    if n_prompt == 0 or n_answer == 0:
        return float("nan")
    return (-sum_lp_prompt / n_prompt) - (-sum_lp_answer / n_answer)


# Calibration

def ece(confidences: Sequence[float], correct: Sequence[int],
        n_bins: int = 10) -> float:
    """
    Expected Calibration Error over n_bins equal-width confidence bins.
    ECE = sum_b (|B_b|/N) * |acc(B_b) - conf(B_b)|.
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    N = len(confidences)
    if N == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (confidences >= lo) & (confidences < hi if b < n_bins - 1
                                      else confidences <= hi)
        if mask.sum() == 0:
            continue
        acc_b = correct[mask].mean()
        conf_b = confidences[mask].mean()
        total += (mask.sum() / N) * abs(acc_b - conf_b)
    return float(total)


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """
    AUROC of a score predicting a binary label, via the rank (Mann-Whitney)
    identity. No sklearn dependency. Returns nan if only one class present.
    Ranks are tie-averaged.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    start = cum - counts
    avg_rank = (start + cum + 1) / 2.0      # tie-averaged rank per distinct value
    ranks = avg_rank[inv]
    sum_ranks_pos = ranks[labels == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# Cluster bootstrap over groups

def cluster_bootstrap(group_values: dict[str, float],
                      B: int = 10000,
                      seed: int = 0) -> dict:
    """
    Bootstrap a mean by resampling groups with replacement.

    group_values : {group_id -> per-group statistic}  (e.g. D_i per group)
    Returns mean, 2.5/97.5 percentile CI, and n.

    Resampling at the group level (not item level) is essential: variants of one
    group are correlated, and item-level bootstrap gives CIs ~sqrt(|V_i|) too narrow.

    NaN values are dropped before resampling, so the returned "n" can be smaller
    than len(group_values) - e.g. a group with no scorable variants has a NaN
    c_corr. Always report the returned n alongside the CI rather than the number
    of groups in the run, or the two will silently disagree.
    """
    keys = list(group_values.keys())
    vals = np.array([group_values[k] for k in keys], dtype=float)
    n_input = len(vals)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": 0, "n_input": n_input,
                "n_dropped_nan": n_input}
    rng = np.random.default_rng(seed)
    n = len(vals)
    means = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        means[b] = vals[idx].mean()
    return {
        "mean":    float(vals.mean()),
        "ci_low":  float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "n":       n,
        "n_input": n_input,
        "n_dropped_nan": n_input - n,
    }


def paired_cluster_bootstrap(group_diffs: dict[str, float],
                             B: int = 10000, seed: int = 0) -> dict:
    """
    Bootstrap the mean of a paired per-group difference (e.g. D_A0 - D_A3 per group).
    A CI excluding 0 is the significance test for gap reduction between ablations.
    Identical mechanics to cluster_bootstrap; named separately for intent.
    """
    return cluster_bootstrap(group_diffs, B=B, seed=seed)