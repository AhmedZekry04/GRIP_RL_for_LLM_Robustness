# MATH-Perturb cross-ablation analysis — brief

Build a Jupyter notebook that loads four MATH-Perturb evaluation runs, compares
them, and produces the figures, tables and statistical tests for the results
chapter.

## Inputs

```python
BASE = "/home/amz/projects/GRIP/Ablations/Eval/MATH-Perturb"
RUNS = {
    "A0": f"{BASE}/a0_mathperturb_20260904_1.5B",   # untrained Qwen2.5-Math-1.5B
    "A1": f"{BASE}/a1_mathperturb_20260915",        # GRPO, originals only
    "A2": f"{BASE}/a2_mathperturb_20260915",        # GRPO, originals + variants
    "A3": f"{BASE}/a3_mathperturb_20260915",        # GRPO + GRIP, gamma = 2
}
```

Each run contains `eval/MATH_Perturb/` with `summary.json`, `groups.jsonl`,
`items.jsonl`, `verified_generations.jsonl`. Glob for `**/summary.json` rather
than hardcoding depth. **Fail loudly** with the attempted path if a run is
missing — never silently skip an ablation.

Figures → `{BASE}/figures/`, both PDF and PNG at 300 dpi. One colour per
ablation used consistently; A0 grey, A3 the strongest colour.

---

## What this benchmark measures

Every problem is drawn from **MATH Level 5** and paired with two perturbations:

- **simple** — surface change; the original solution method still applies
- **hard** — structural change; the original method no longer applies

All 277–278 problems are Level 5, so **there is no difficulty variation**. Do not
produce a by-level figure; `summary["by_level"]` will have a single bucket. Use
`by_subject` instead.

The benchmark's memorisation claim: a model relying on memorised *techniques*
rather than reasoning will solve the original, solve the simple perturbation
(same technique still works), and fail the hard one (technique no longer applies,
but it applies it anyway). That specific three-way pattern is the signature — not
a raw accuracy drop.

## What the ablations are, and which comparisons are valid

| | training data | objective | on the γ path? |
|---|---|---|---|
| A0 | none (base model) | — | no |
| A1 | originals only | GRPO | no — different data |
| A2 | originals + all variants | GRPO | **yes, γ = 1** |
| A3 | originals + all variants | GRPO + GRIP | **yes, γ = 2** |

- **A1 → A2** isolates *data augmentation*. Same objective, different data.
- **A2 → A3** isolates *the objective*. Same data, same hyperparameters, only the
  invariance penalty differs. **This is the thesis test.**

Only A2 and A3 differ solely in γ. A0 and A1 are not points on that path.

---

## The headline numbers (already computed — use to sanity-check the loader)

| | A0 | A1 | A2 | A3 |
|---|---|---|---|---|
| acc original | 0.573 | 0.658 | 0.671 | 0.548 |
| acc simple | 0.492 | 0.619 | 0.633 | 0.465 |
| acc hard | 0.269 | 0.389 | 0.378 | 0.259 |
| gap | 0.191 | 0.154 | 0.165 | 0.186 |
| within-seed var | 0.0515 | 0.0636 | 0.0686 | 0.0511 |
| RSR simple | 0.739 | 0.834 | 0.891 | 0.732 |
| RSR hard | 0.330 | 0.497 | 0.492 | 0.339 |
| R̂ global | 0.579 | 0.594 | 0.599 | 0.579 |
| mG-Pass@8 orig | 0.373 | 0.559 | 0.569 | 0.359 |
| extract-fail | 0.064 | 0.017 | 0.013 | 0.042 |

If your loader disagrees with these, the loader is wrong — stop and fix it.

**The pattern to explain.** A1 and A2 improved substantially over A0. A3 reverted
to approximately A0 on every metric — R̂ global identical to three decimals, RSR
hard 0.339 vs 0.330, accuracy slightly *below* A0 on all three arms.

**The trap.** A3's within-seed variance (0.0511) is *lower* than A2's (0.0686),
which superficially looks like the penalty succeeded. It did not: A3's variance
matches A0's because A3's *accuracies* match A0's. Variance fell by lowering
everything, not by equalising anything. This is *levelling down*, and the
analysis must make it visible rather than let the variance number stand alone.

---

## Figures

Each needs a caption cell above stating what it shows and what to conclude, plus
a printed n.

**Fig 1 — Accuracy by arm.** Grouped bars: original / simple / hard on x, one bar
per ablation, 95% CI from cluster bootstrap over `group_id` (B=10,000, seed 0).
Annotate values. *The headline capability picture.*

**Fig 2 — Degradation trajectory.** Slope plot, x = original → simple → hard,
one line per ablation. Second panel with each ablation normalised to 1.0 at its
own original. *The shape of degradation. Simple and hard falling together is
surface sensitivity; hard falling much further is technique dependence — the
distinction the benchmark exists to draw.*

**Fig 3 — Capability versus invariance scatter.** x = accuracy on originals,
y = invariance gap Δ, one point per ablation, error bars both directions.
*The trade-off in one panel. A3 down-and-left of A2 — worse accuracy, no better
gap — is levelling down made visible without argument. Likely the single most
informative figure in the chapter.*

**Fig 4 — Within-seed variance beside accuracy.** Two panels sharing an x-axis of
ablations: `within_var_mean` with CI, and mean accuracy across arms. *The
essential pairing. Variance falling while accuracy also falls is levelling down;
variance falling while accuracy holds is genuine invariance. Neither panel alone
distinguishes them.*

**Fig 5 — RSR and recovery.** Grouped bars: RSR for simple and hard per ablation,
with recovery rate as a lighter secondary bar. *RSR conditions away capability
and is the fair comparison across models of different strength. Recovery should
be near zero; flag it if not.*

**Fig 6 — Outcome taxonomy (the memorisation figure).** For each ablation,
classify every seed by the triple `(e_i, h_simple, h_hard)` from `groups.jsonl`:

| pattern | label |
|---|---|
| 1,1,1 | robust — solved throughout |
| 1,1,0 | **technique memorisation** — solved original and simple, failed hard |
| 1,0,0 | brittle — solved original only |
| 1,0,1 | inconsistent |
| 0,*,* | not solved originally (one bucket) |

Stacked horizontal bars, one per ablation, counts and percentages annotated,
strongest colour on `1,1,0`. *This is the benchmark's own memorisation signature
made explicit — the shortcut solution that survives the simple perturbation and
fails when the technique no longer applies. **Report the `1,1,0` count as a
fraction of seeds solved originally; that single number is what this benchmark
exists to produce.** Whether RL training reduces it is a direct test of whether
the model learned reasoning or technique.*

**Fig 7 — Per-seed gap scatter, simple vs hard.** x = `gap_by_arm.simple`,
y = `gap_by_arm.hard`, one point per seed, y=x diagonal, faceted by ablation.
Annotate with the fraction above the diagonal and `hard_minus_simple_gap` mean
and CI. *Tests whether hard is reliably harder on the same problems, not merely
on average. State whether the CI excludes zero.*

**Fig 8 — R̂ by arm and global.** Horizontal bars for `r_hat_simple`,
`r_hat_hard`, `r_hat_global` per ablation, annotated with `r_hat_beta` beside
each. *R̂ is not comparable across benchmarks without β. Note also that
MATH-Perturb's simple/hard axis is solution-strategy, not the surface/parametric
split R̂_global was defined over, so `r_hat_global` here is an analogue.*

**Fig 9 — G-Pass@8 reliability profile.** For each arm, three bars at
τ = 0.5, 0.75, 1.0 per ablation, annotated with mG-Pass@8. *pass@k asks whether
the model can solve a problem; G-Pass asks whether it does so dependably. The
gap between avg@n and mG-Pass@n is the reliability deficit.*

**Fig 10 — By subject.** Grouped horizontal bars or small heatmap:
`gap_simple` and `gap_hard` per subject per ablation, sorted by `acc_original`
descending. Include `n_groups` in each label and **grey out or annotate subjects
with fewer than 10 seeds** — several are thin and their numbers are noisy.

**Fig 11 — Generation health and the format decomposition.** Two panels.

*Left*: `extract_fail_rate`, `truncation_rate`, `mean_n_tokens_completion` per
ablation.

*Right*: for each ablation and arm, three bars —

```
acc_unconditional = n_correct / n_attempted     # headline
acc_given_parsed  = n_correct / n_parseable     # diagnostic
format_rate       = n_parseable / n_attempted
```

*On ASyMOB, A3's extract-failure rate was 12.9× A2's and ~96% of those failures
were truncations — the model ran out of tokens before emitting `\boxed{}`, a
consequence of its elevated entropy (0.145 vs 0.058 in training). Check whether
MATH-Perturb shows the same. If `acc_given_parsed` is similar for A2 and A3 while
`format_rate` differs sharply, the penalty damaged termination rather than
reasoning — a far more precise finding than "A3 got worse".*

**Do not filter seeds on which A3 truncated and re-score the others.** That is
conditioning on the outcome: the truncations are not random, they concentrate on
the discriminating items, and method-dependent exclusion manufactures a
favourable result. Truncation without an answer is a genuine task failure and
belongs in the headline number. The conditional accuracy is a *diagnostic
alongside* it, never a replacement.

**Fig 12 — Confidence by arm and correctness**, only if `mean_confidence` is
non-NaN. Check first and state in the notebook if unavailable.

---

## Metrics: what to use and what to leave out

**Use**: accuracy per arm, Δ with CI, within-seed variance, RSR, recovery rate,
R̂ (with β), G-Pass@k / mG-Pass@k, `frac_fully_invariant`, `hard_minus_simple_gap`,
the outcome taxonomy, generation health.

**Do not use `c_ans`.** The three arms have *different correct answers by
construction* — the hard perturbation changes the problem. A correctly reasoning
model should give different answers across arms, so low answer agreement is the
expected behaviour of a good model and carries no signal. Note the exclusion in
one line.

**`c_corr` needs care.** `c_corr = 1` means *same outcome*, which includes wrong
on everything. Report split by `e_i`, or rely on RSR which already conditions on
the original being solved.

**No by-level figure** — all problems are Level 5.

---

## Data validation (short)

Print once per run: `n_groups`, groups with `n_perturbed_scored == 0`, and NaN
counts in `gap_mean`, `within_var`, `c_corr`, `acc_original`. If all zero, say so
in one line and proceed. If not, use `np.nanmean`/`nanvar` throughout and report
effective n with every statistic.

---

## Statistical tests

Implement `paired_cluster_bootstrap(stat_a, stat_b, group_ids, B=10000, seed=0)`
resampling **seeds**, not items, returning mean difference with a 95% interval.

Seed-level resampling is required because the arms of one seed are rewrites of
the same problem; item-level resampling treats correlated observations as
independent and yields intervals roughly √3 too narrow.

Align `group_id` across ablations first; print how many seeds are shared and drop
the rest with a count.

| contrast | Δ gap | 95% CI | Δ acc(orig) | 95% CI | Δ RSR-hard | 95% CI |
|---|---|---|---|---|---|---|
| A1 − A0 | | | | | | |
| A2 − A0 | | | | | | |
| A2 − A1 | | | | | | |
| **A3 − A2** | | | | | | |
| A3 − A0 | | | | | | |

**A3 − A2 on the gap is the hypothesis test.** An interval excluding zero is the
evidence GRIP changed invariance. State plainly whether it does.

**A3 − A0 on accuracy is the levelling-down test.** An interval containing zero
means A3 is statistically indistinguishable from the untrained model — a strong
and directly reportable statement.

Also test **A3 − A0 on within-seed variance**: if that also contains zero, it
confirms the variance reduction was a return to baseline rather than an
improvement.

---

## Tables

**Table 1 — main results.** Rows A0/A1/A2/A3; columns acc(original), acc(simple),
acc(hard), Δ with CI, within-seed variance, RSR-simple, RSR-hard, R̂ global,
mG-Pass@8 on originals, frac fully invariant. Bold best per column.

**Table 2 — outcome taxonomy.** Rows A0/A1/A2/A3; columns the five patterns from
Fig 6 as counts and as percentages of seeds solved originally.

**Table 3 — paired contrasts**, as above.

**Table 4 — by subject.** Subjects as rows with n, ablations as column groups
showing `gap_simple` and `gap_hard`. Mark subjects with n < 10.

**Table 5 — generation health and format decomposition** per ablation: extract-
fail, truncation, format-ok, mean completion length, and the unconditional vs
conditional accuracy pair.

Emit every table as LaTeX (`booktabs`) **and** printed markdown.

---

## Written analysis

Continuous prose for a results chapter, not bullets. Every number read from the
files; invent nothing. Cover in order:

1. Baseline at A0 — how large is the gap before training, and what the
   `1,1,0` taxonomy bucket says about memorisation in the base model
2. Whether RL improved capability (A0→A1) and whether it improved robustness
3. Whether augmentation closed the gap further (A1→A2)
4. Whether the GRIP penalty closed it further (A2→A3), with the paired interval
5. The capability cost, and whether it bought anything — Figs 3 and 4 together
6. The within-variance trap explicitly: A3's variance is lower than A2's *and*
   matches A0's, and why that means levelling down rather than invariance
7. Whether the memorisation bucket shrank under any training condition
8. Reliability via G-Pass, subject-level variation, and generation health as a
   threat to the above — including whether A3's truncation reproduces the ASyMOB
   pattern

Close with limitations: 277 problems all at Level 5, single model scale, one seed
per ablation, 500 training steps, one γ value evaluated so far, and the structural
point that **a variance penalty on accuracy is capability-blind by construction**
— it cannot distinguish equalising upward from equalising downward, and
suppressing a strong arm is a cheaper gradient direction than mastering a weak one.

---

## Engineering notes

- One loader returning `{ablation: {"summary": dict, "groups": df, "items": df}}`.
  Never hardcode a single run.
- Verify against the headline table above before producing any figure.
- Bootstrap everywhere: resample `group_id`, B = 10,000, seed 0.
- Print an n-table for every figure.
- Save `results_summary.json` containing every number that appears in a table so
  the report can be regenerated without re-running the notebook.
- Keep plotting functions parameterised by `{label: run_key}` so a γ = 1.5 run
  can be added later with a one-line change.
