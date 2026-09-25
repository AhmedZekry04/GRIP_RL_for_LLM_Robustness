# ASyMOB cross-ablation analysis — brief

Build a Jupyter notebook that loads four ASyMOB evaluation runs, compares them,
and produces the figures, tables and statistical tests for the results chapter.

## Inputs

```python
BASE = "/home/amz/projects/GRIP/Ablations/Eval/Asymob"
RUNS = {
    "A0": f"{BASE}/a0_asymob_20260904_1.5B",   # untrained Qwen2.5-Math-1.5B
    "A1": f"{BASE}/a1_asymob_20260915",        # GRPO, originals only
    "A2": f"{BASE}/a2_asymob_20260915",        # GRPO, originals + variants
    "A3": f"{BASE}/a3_asymob_20260915",        # GRPO + GRIP, gamma = 2
}
```

Each run directory contains `eval/asymob/` with `summary.json`, `groups.jsonl`,
`items.jsonl`, `verified_generations.jsonl`. Glob for `**/summary.json` rather
than hardcoding depth. **Fail loudly** with the attempted path if a run is
missing — never silently skip an ablation.

Figures → `{BASE}/figures/`, both PDF and PNG at 300 dpi. One colour per
ablation used consistently; A0 grey, A3 the strongest colour.

---

## What the ablations are, and which comparisons are valid

| | training data | objective | on the γ path? |
|---|---|---|---|
| A0 | none (base model) | — | no |
| A1 | originals only | GRPO | no — different data |
| A2 | originals + all variants | GRPO | **yes, γ = 1** |
| A3 | originals + all variants | GRPO + GRIP | **yes, γ = 2** |

Two distinct comparisons, and they must not be conflated:

- **A1 → A2** isolates *data augmentation*. Same objective, different data.
- **A2 → A3** isolates *the objective*. Same data, same hyperparameters, only
  the invariance penalty differs. **This is the thesis test.**

A2 and A3 are the only two runs that differ solely in γ, which matters for
Figure 1 below.

---

## Known result to verify, not to assume

On MATH-Perturb, A1 and A2 improved substantially over A0 (RSR-hard 0.330 →
≈0.49; mG-Pass@8 on originals 0.373 → ≈0.57) while **A3 reverted to
approximately A0 on every metric**. The diagnosis was *levelling down*: the
penalty reduces within-seed variance partly by suppressing accuracy on the
variants the model already handles well, rather than lifting the weak ones. At
γ=2, roughly 14% of correct rollouts had their advantage flipped negative.

Check whether ASyMOB reproduces this. It may not — ASyMOB has a far richer
family taxonomy and the picture may differ by family. **Compute before
concluding.**

---

## Figure 1 — γ-path trade-off curves *(the centrepiece; build this first)*

Reproduce the structure of Rothenhäusler et al. Figure 2.2 (accuracy–robustness
trade-off of γ) using measured data.

**Four panels**, one each for: `Symbolic-One`, `Symbolic-All`, `Numeric-One`,
`Numeric-All`. Shared axes across panels.

- **x-axis**: perturbation strength / intensity level
- **y-axis**: accuracy (or 1 − accuracy if you prefer the MSE-like orientation of
  the original figure; pick one and be consistent, and say which in the caption)
- **one curve per γ**, with CI bands from the cluster bootstrap:
  - **A2 = γ 1** — plain GRPO, the unpenalised endpoint
  - **A3 = γ 2** — GRIP
  - leave the plotting function parameterised by a `{label: run}` dict so a
    γ = 1.5 run can be added later with a one-line change

**What the figure is testing.** Anchor regression predicts that a larger γ trades
accuracy at low perturbation strength for stability at high strength — the curves
should *cross*. A flatter, lower-starting A3 curve that ends above A2 at high
intensity is the signature. A3 below A2 everywhere is levelling down, not a
trade-off.

### Should A0 and A1 be on this figure?

**Not as members of the γ family.** The x-axis of the original figure is a
regularisation path: every curve on it is the same estimator at a different γ.
A0 is untrained and A1 saw different data, so neither is a point on that path.
Drawing them as equals would misrepresent the figure.

**Do plot them as context**, visually subordinate: thin grey dashed for A0,
thin grey dotted for A1, in the legend under a separate heading such as
*"reference (not on the γ path)"*. A0 shows where the model starts; A1 shows what
augmentation alone achieves. Both are useful for judging whether the γ curves are
in a sensible range, and neither should be styled like a γ curve.

### Determining the strength axis

Before building this, **print the taxonomy**:

```python
for name, d in data.items():
    g = d["groups"]
    print(name, "families:", sorted(g["meta.family"].dropna().unique()))
    print(name, "intensity:", sorted(g["meta.intensity"].dropna().unique()))
    print(name, "n per (family, intensity):")
    print(g.groupby(["meta.family", "meta.intensity"]).size())
```

Then pick the x-axis from what exists:

- If `meta.intensity` is populated with ordered levels, use it directly.
- If strength is encoded in the arm/family name (e.g. a suffix), parse it.
- If `One` vs `All` is the *only* strength distinction available, collapse to a
  two-point x-axis (One → All) and say so in the caption. A two-point curve still
  shows a crossing or its absence.
- If no strength ordering exists at all, state that plainly and replace this
  figure with a grouped bar chart of accuracy by family per γ. **Do not fabricate
  an axis.**

Annotate each panel with its n.

---

## Figures 2–10

Each needs a caption cell above it stating what it shows and what to conclude,
and a printed n.

**Fig 2 — Accuracy by arm.** Grouped bars, families on x, one bar per ablation,
95% CI from cluster bootstrap over `group_id` (B=10,000, seed 0). The headline
capability picture.

**Fig 3 — Invariance gap with CI.** Overall Δ per ablation, then Δ per family.
Mark where intervals overlap.

**Fig 4 — RSR by family.** Grouped bars, RSR per family per ablation, with
recovery rate as a lighter secondary bar. *RSR conditions away capability, so it
is the fair comparison between models of different strength. Recovery should be
near zero; flag it if not, since a non-trivial value means the perturbation
sometimes makes a problem easier.*

**Fig 5 — Capability versus invariance scatter.** x = accuracy on originals,
y = invariance gap, one point per ablation with error bars in both directions.
*The trade-off in one panel. A3 down-and-left of A2 — worse accuracy, no better
gap — is levelling down made visible without argument.*

**Fig 6 — Within-seed variance beside accuracy.** Two panels sharing an
x-axis of ablations: `within_var_mean` with CI, and mean accuracy across arms.
*Essential pairing. On MATH-Perturb A3's variance was lower than A2's, which
looks like success until you see A3's accuracies matched A0's — variance fell
because everything fell. Neither panel alone distinguishes equalising upward
from equalising downward.*

**Fig 7 — Transfer by distance from training.** Order the x-axis: Numeric
(matches the `kernel` training perturbation) → Symbolic (matches surface
renaming) → **Equivalence (out of family; nothing in training resembles it)**.
Plot Δ and RSR. *Anchor regression guarantees robustness only within the span of
the training anchors, so Equivalence is where theory predicts nothing. A3 beating
A2 there would be the interesting result.*

**Fig 8 — R̂ by family and global.** Horizontal bars annotated with
`r_hat_beta` beside each value. *R̂ is not comparable across benchmarks without β.*

**Fig 9 — G-Pass@8 reliability profile.** For originals and each family, three
bars at τ = 0.5, 0.75, 1.0 per ablation, annotated with mG-Pass@8. *Separates
"can solve" from "solves dependably".*

**Fig 10 — Generation health.** `extract_fail_rate`, `truncation_rate`,
`mean_n_tokens_completion` by ablation and family. *Defends every number above.
If A3's extract-fail is markedly higher, part of its measured drop is format
compliance rather than reasoning and must be disclosed.*

**Fig 11 — Confidence by arm and correctness**, only if `mean_confidence` is
non-NaN. Check first; state in the notebook if unavailable.

---

## Metrics: what to use and what to leave out

**Use**: accuracy per arm, invariance gap Δ, within-seed variance, RSR, recovery
rate, R̂ (with β), G-Pass@k / mG-Pass@k, `frac_fully_invariant`, `c_corr` split
by `e_i`, generation health.

**Do not use `c_ans`.** Answer agreement is only interpretable where the
perturbation preserves the reference answer, which across ASyMOB's families holds
for almost none of them — Symbolic changes the answer's form, Numeric changes its
value. Reporting it would require a per-family caveat that adds nothing the other
metrics do not already provide. Exclude it and note the exclusion in one line.

**`c_corr` needs care.** `c_corr = 1` means *same outcome*, which includes wrong
on everything. Report it split by whether the original was solved, or rely on RSR
which conditions on that already.

---

## Data validation (short — run it, don't dwell on it)

`gap_mean` returned NaN in an earlier ASyMOB aggregation. That has since been
addressed, so this is a regression check rather than an expected problem. Print
once per run:

```
run, n_groups, n with n_perturbed_scored == 0,
NaN counts in: gap_mean, within_var, c_corr, acc_original
```

If all zero, proceed and say so in one line. If not, use `np.nanmean`/`nanvar`
throughout, report effective n with every statistic, and state how many groups
were dropped.

---

## Statistical tests

Implement `paired_cluster_bootstrap(stat_a, stat_b, group_ids, B=10000, seed=0)`
resampling **seeds**, not items, returning mean difference with a 95% interval.

Seed-level resampling is required because variants of one seed are rewrites of
the same problem; item-level resampling treats correlated observations as
independent and yields intervals roughly √|V| too narrow.

Align `group_id` across ablations before any paired test; print how many seeds are
shared and drop the rest with a count.

| contrast | Δ gap | 95% CI | Δ acc(orig) | 95% CI | Δ RSR | 95% CI |
|---|---|---|---|---|---|---|
| A1 − A0 | | | | | | |
| A2 − A0 | | | | | | |
| A2 − A1 | | | | | | |
| **A3 − A2** | | | | | | |
| A3 − A0 | | | | | | |

**A3 − A2 on the gap is the hypothesis test.** An interval excluding zero is the
evidence GRIP changed invariance. State plainly whether it does.

**A3 − A0 on accuracy** is the levelling-down test: an interval containing zero
means A3 is statistically indistinguishable from the untrained model.

---

## Tables

**Table 1 — main results.** Rows A0/A1/A2/A3; columns acc(original), acc(mean
perturbed), Δ with CI, within-seed variance, RSR overall, R̂ global, mG-Pass@8 on
originals, frac fully invariant. Bold best per column.

**Table 2 — per-family breakdown.** Families as rows ordered by distance from
training (Fig 7 order), ablations as column groups, each showing Δ and RSR.

**Table 3 — paired contrasts**, as above.

**Table 4 — generation health** per ablation: extract-fail, truncation, format-ok,
mean completion length.

Emit every table as LaTeX (`booktabs`) **and** printed markdown.

---

## Written analysis

Continuous prose for a results chapter, not bullets. Every number read from the
files; invent nothing. Cover in order:

1. Baseline robustness at A0 — how large is the gap before any training
2. Whether RL improved capability (A0→A1) and whether it improved robustness
3. Whether augmentation closed the gap (A1→A2)
4. Whether the GRIP penalty closed it further (A2→A3), with the paired interval
5. The capability cost and whether it bought anything — Figures 5 and 6 together
6. Per-family behaviour, with attention to out-of-family Equivalence
7. What Figure 1 shows about the γ trade-off: do the curves cross, or is A3 below
   A2 everywhere
8. Reliability via G-Pass, and generation health as a threat to the above

If the levelling-down pattern reproduces, say so explicitly and support it from
Figures 5 and 6 rather than asserting it.

Close with limitations: single model scale, one seed per ablation, 500 training
steps, two γ values, and the structural point that **a variance penalty on
accuracy is capability-blind by construction** — it cannot distinguish equalising
upward from equalising downward, and suppressing a strong variant is a cheaper
gradient direction than mastering a weak one.

---

## Engineering notes

- One loader returning `{ablation: {"summary": dict, "groups": df, "items": df}}`.
  Never hardcode a single run.
- Figure 1's plotting function takes `{label: run_key}` so γ = 1.5 can be added
  by editing one dict.
- Bootstrap everywhere: resample `group_id`, B = 10,000, seed 0.
- Print an n-table for every figure.
- Save `results_summary.json` containing every number that appears in a table, so
  the report can be regenerated without re-running the notebook.
