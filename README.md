# GRIP: RL for LLM Robustness

Code and data for the MSc thesis *Invariance-Regularised Rewards for LRMs*.

GRIP (Group-Relative Invariance Penalty) adds an invariance penalty to GRPO, with
the aim of making a maths-reasoning model answer *rewritten* versions of a problem
as reliably as it answers the original. It is trained on DeepMathGAP and evaluated
on MATH-Perturb and ASyMOB.

| Folder | Contents |
|---|---|
| [`DeepMathGAP/`](DeepMathGAP/) | The DeepMathGAP dataset (4 equivalent variants per problem) and the pipeline that built it |
| [`GRIP/`](GRIP/) | Training and evaluation scripts for ablations A0-A3, evaluation runs, analysis notebooks, figures and tables |

## Result in short

The penalty does not work as intended. It reduces the spread of outcomes within a
problem mostly by lowering accuracy, not by lifting the weaker rewrites:

- A3 - A2 on the invariance gap is +0.019 [-0.018, 0.056] on MATH-Perturb and
  -0.021 [-0.053, 0.010] on ASyMOB; both intervals contain zero.
- By gamma = 2, accuracy on the original problems falls back to the untrained
  base model (A3 - A0 = -0.024 [-0.050, 0.002] on MATH-Perturb).

See [`GRIP/README.md`](GRIP/README.md) for the full results, repository layout,
setup, and known caveats, and [`DeepMathGAP/README.md`](DeepMathGAP/README.md)
for how the training dataset was built.

## Attribution

DeepMathGAP is built from [DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K)
(MIT) using the perturbation framework and adapted code of
[GAP](https://github.com/YurenHao0426/GAP) (CC BY 4.0); see
[`DeepMathGAP/README.md`](DeepMathGAP/README.md) §6–7 for citations and attribution.
The evaluation benchmarks are ASyMOB and MATH-Perturb; `GRIP/data/` holds
preprocessed copies of both.

**License:** code under MIT ([`LICENSE`](LICENSE)); the DeepMathGAP dataset under
CC BY 4.0 ([`DeepMathGAP/data/LICENSE`](DeepMathGAP/data/LICENSE)).
