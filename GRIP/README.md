# GRIP: reinforcement learning for LLM robustness

GRIP (Group-Relative Invariance Penalty) adds an invariance penalty to GRPO, with
the aim of making a maths-reasoning model answer *rewritten* versions of a problem
as reliably as it answers the original.

This repository contains the training scripts, the evaluation runs, and the
analysis notebooks that test whether that works. It does not work as intended, and
the analysis is set up to show why rather than to argue the point: the penalty
reduces the spread of outcomes within a problem mostly by lowering accuracy, not by
lifting the weaker rewrites.

## Result in short

Two benchmarks, five runs each. `gamma = 1` is plain GRPO (no penalty); `gamma`
above 1 turns the penalty on.

| | training data | objective | on the gamma path |
|---|---|---|---|
| A0 | none (untrained base model) | - | no |
| A1 | originals only | GRPO | no (different data) |
| A2 | originals + all variants | GRPO | yes, gamma = 1 |
| A3 (gamma 1.5) | originals + all variants | GRPO + GRIP | yes |
| A3 (gamma 2) | originals + all variants | GRPO + GRIP | yes |

MATH-Perturb, 277 problems shared by all runs:

| run | acc original | acc simple | acc hard | invariance gap | within-seed var | RSR hard |
|---|---|---|---|---|---|---|
| A0 | 0.573 | 0.493 | 0.270 | 0.191 | 0.052 | 0.330 |
| A1 | 0.660 | 0.618 | 0.390 | 0.156 | 0.064 | 0.497 |
| A2 (gamma 1) | 0.672 | 0.633 | 0.380 | 0.166 | 0.069 | 0.492 |
| A3 (gamma 1.5) | 0.604 | 0.545 | 0.311 | 0.176 | 0.055 | 0.344 |
| A3 (gamma 2) | 0.549 | 0.453 | 0.276 | 0.185 | 0.048 | 0.343 |

The main findings, all from paired seed-level bootstraps (B = 10,000):

- **The penalty does not close the invariance gap.** A3 - A2 on the gap is
  +0.019 [-0.018, 0.056] on MATH-Perturb and -0.021 [-0.053, 0.010] on ASyMOB.
  Both intervals contain zero.
- **It costs capability.** By gamma = 2, accuracy on the originals is back at the
  untrained baseline: A3 - A0 is -0.024 [-0.050, 0.002].
- **Lower variance here is not invariance.** A3's within-seed variance is below
  A2's and also matches A0's, because its accuracies match A0's. Variance falls
  because everything falls. The notebooks pair every variance number with an
  accuracy number so this cannot be read the other way.
- **Scaling the base model does not fix robustness either.** Qwen2.5-Math-7B beats
  1.5B on every arm (+0.08 to +0.11) but its invariance gap is no smaller
  (+0.027 [-0.007, 0.061]), and both keep only about half their accuracy under a
  structural rewrite. On RSR-hard, which conditions on the original being solved,
  the 7B model *is* significantly better (+0.095 [0.034, 0.157]), so the claim
  holds for the gap and the degradation shape but not for that measure.

## Layout

```
ablations/
  train/            training and evaluation launchers, one directory per ablation
    baseline_a0/      evaluation only (A0 is the untrained base model), 1.5B and 7B
    train_a1/         GRPO on originals
    train_a2/         GRPO on originals + variants
    train_a3/         GRPO + GRIP invariance penalty
  eval/
    asymob/         ASyMOB runs, analysis notebook, figures, results_summary.json
    math_perturb/   MATH-Perturb runs, analysis notebook, figures, results_summary.json
data/               source and prepared datasets
preprocess/         dataset preparation scripts
```

Each evaluation run directory is named `<ablation>_<benchmark>[_<variant>]_<date>`
and holds the exact scripts it was scored with, so a run can always be traced back
to the code that produced it:

```
a3_mathperturb_gamma2_20260915/
  manifest.json                 model, data file, sampling settings, seed, vLLM version
  eval/MATH_Perturb/
    generations.jsonl           raw samples          (not in git, see below)
    verified_generations.jsonl  samples + verdicts   (not in git, see below)
    items.jsonl                 per-item aggregates  (not in git, see below)
    groups.jsonl                per-problem aggregates
    summary.json                run-level metrics
    metrics.py, verify.py, aggregate_mathperturb.py
```

### What is not in the repository

Raw model outputs are too large for GitHub (up to ~180 MB per file, against a
100 MB limit), so three files per run are gitignored:

| file | size | rebuild with |
|---|---|---|
| `generations.jsonl` | 20-180 MB | `ablations/train/*/eval_*.py` |
| `verified_generations.jsonl` | 20-180 MB | `verify.py` in the run directory |
| `items.jsonl` | 0.5-22 MB | `aggregate_asymob.py` / `aggregate_mathperturb.py` |

`groups.jsonl`, `summary.json` and `manifest.json` are kept, so every number in
the figures and tables can be checked without re-running anything. The notebooks
themselves need `items.jsonl` for the family, generation-health and G-Pass
figures, so a fresh clone can read the stored outputs but cannot re-execute those
cells until the aggregates are rebuilt.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Training needs a GPU and was run on an HPC cluster; the analysis notebooks need
only numpy, pandas, matplotlib and scipy.

## Analysis

```bash
cd ablations/eval/math_perturb
jupyter lab mathperturb_cross_ablation_analysis.ipynb
```

Each notebook loads every run, validates it, and produces the figures, LaTeX
tables and paired statistical tests for one benchmark. They locate the repository
root themselves, so they can be run from anywhere in the tree.

Conventions used throughout:

- **Cluster bootstrap over problems**, never over items. The arms of one problem
  are rewrites of the same question, so item-level resampling would treat
  correlated observations as independent and give intervals that are far too
  narrow. B = 10,000, seed 0.
- **Figures show point estimates only.** Intervals are printed beneath each figure
  and in the tables; claims of a difference come from the paired tests, not from
  whether error bars overlap.
- Figures are written to `figures/` as PDF and PNG at 300 dpi, and every number
  that appears in a table is also written to `results_summary.json`.

`ASYMOB_ANALYSIS_BRIEF.md` and `MATHPERTURB_ANALYSIS_BRIEF.md` in the two
evaluation directories are the specifications the notebooks implement, including
which metrics to use and which to leave out.

## Data

| file | rows | what it is |
|---|---|---|
| `ASyMOB_clean.jsonl` | 35,368 | ASyMOB regrouped by seed question (see below) |
| `ASyMOB_clean_dedup.jsonl` | | deduplicated variant set |
| `asymob_prepped.jsonl` | | ASyMOB in the prompt/answer schema the eval scripts read |
| `deepmathgap_prepped.jsonl` | | DeepMathGAP training set in the trainer's schema |
| `mathperturb_final.jsonl` | 834 | MATH-Perturb, 278 problems x 3 arms |

### Why ASyMOB needs regrouping

ASyMOB is 100 seed problems, each expanded into many perturbed variants that
preserve the seed's mathematical content. The raw `Index` field is a row counter
and does not say which rows came from the same seed. `Source` (benchmark +
subcategory + UUID) is the real grouping key: every variant of a seed shares it.

The preparation adds two fields:

- **`id`** - stable id shared by every variant of a seed, e.g. `dmgap_000001`.
- **`k`** - one of 9 perturbation categories, consistent across seed families:

  | k | category | meaning |
  |---|---|---|
  | 0 | Original | the unperturbed seed question |
  | 1 | Symbolic | numeric constants replaced by free symbols |
  | 2 | Numeric-All | all symbols replaced by random N-digit numbers |
  | 3 | Numeric-One | one symbol replaced by a number, the rest removed |
  | 4 | Numeric-All-S | 50 resamples per seed, for measuring answer variance |
  | 5 | Equivalence-One-Easy | one symbol replaced by an easy "equals 1" identity |
  | 6 | Equivalence-One-Hard | one symbol replaced by a hard "equals 1" identity |
  | 7 | Equivalence-All-Easy | all symbols replaced by easy identities |
  | 8 | Equivalence-All-Hard | all symbols replaced by hard identities |

Several raw `Variation` values collapse into one `k` (for example `Symbolic-1`
through `Symbolic-5`), so several rows can share an `(id, k)` pair. The original
`Variation` string is kept so the finer distinction is still available; the
notebooks use it as the arm label.

`Answer in Sympy` is the reliable ground-truth field. `Answer in Latex` is
populated only for a subset of rows, mostly the originals.

The uncompressed DeepMathGAP source file is not kept here. DeepMathGAP is
maintained in its own repository; `preprocess/preprocess_deepmathgap.py` turns its
released dataset into `data/deepmathgap_prepped.jsonl`, which is the file these
training runs actually consumed and is included here.

## Known caveats

- The ASyMOB 7B baseline (`a0_asymob_7B_20260904`) was aggregated with an earlier
  version of `aggregate_asymob.py` that used `mean` where the current script uses
  `nanmean`. Its per-arm `gap_mean` fields are therefore NaN and it has no
  `gap_n`. No notebook or figure uses that run; re-run `aggregate_asymob.py`
  against its `verified_generations.jsonl` before comparing it to the others.
- Default input paths in `preprocess/preprocess_mathperturb.py` and
  `preprocess/preprocess_deepmathgap.py` still point at an older data layout
  (a nested `data/eval/...` tree, and a cluster home directory). Pass `--in_path`
  / `--out_path`, or update the constants, before re-running them.
- Every result rests on one training seed per ablation and a single model scale
  for the trained runs, so between-run training variability is unmeasured.
  Re-running an evaluation alone moved individual metrics by up to 0.025, which
  is the noise floor quoted in the notebooks.
