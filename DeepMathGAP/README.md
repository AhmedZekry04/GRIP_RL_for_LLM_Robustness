# DeepMathGAP

DeepMathGAP is a dataset for training and evaluating how robust LLMs are on
mathematical reasoning. It expands problems from
[DeepMath-103K](https://arxiv.org/abs/2504.11456) into **groups of four
mathematically equivalent variants**. It follows the perturbation taxonomy of
[GAP / PutnamGAP](https://arxiv.org/abs/2508.08833). A model that really
solves a problem should solve all four variants.

> **The dataset is [`data/final/deepmathgap_v2.jsonl.gz`](data/final/deepmathgap_v2.jsonl.gz):
> 36,057 groups, 144,228 rows.** Every other file in this folder is either the
> code that built it or an audit trail explaining what was dropped. "v2" is the
> final version; "v1" is the unfiltered assembly produced during construction
> (§3, Stage 5), and is not distributed.

| `k` | `type` | What changes |
|---|---|---|
| 0 | `original` | Nothing: the DeepMath-103K problem |
| 1 | `surface_gs` | **Garbled String**: every variable/parameter name is replaced by a random string (`x` → `lr4dr`) |
| 2 | `surface_dlm` | **Descriptive Long Misleading**: every variable/parameter name is replaced by a real term from an unrelated subfield, chosen to misdirect (`x` → `Hilbert space`) |
| 3 | `kernel` | **Kernel Variant**: numeric constants are resampled and the answer is re-derived and independently verified (`x+1` → `x+7`) |

---

## 1. Using the dataset

```python
import pandas as pd
df = pd.read_json("data/final/deepmathgap_v2.jsonl.gz", lines=True)
groups = df.groupby("id")          # 4 rows per id, k = 0..3
```

One JSON object per row. The four rows of a group share `id`:

```json
{"id": "dmgap_000000", "k": 0, "type": "original",    "question": "Evaluate the limit: \\[ \\lim_{x \\to \\infty} \\sqrt{x} \\left( \\sqrt[3]{x+1} - \\sqrt[3]{x-1} \\right) \\]", "answer": "0", "answer_type": "numerical", "difficulty": 4.5, "topic": "Mathematics -> Precalculus -> Limits"}
{"id": "dmgap_000000", "k": 1, "type": "surface_gs",  "question": "... \\lim_{lr4dr \\to \\infty} \\sqrt{lr4dr} ...", "answer": "0", ...}
{"id": "dmgap_000000", "k": 2, "type": "surface_dlm", "question": "... \\lim_{Hilbert space \\to \\infty} \\sqrt{Hilbert space} ...", "answer": "0", ...}
{"id": "dmgap_000000", "k": 3, "type": "kernel",      "question": "... \\sqrt[3]{x+7} - \\sqrt[3]{x-7} ...", "answer": "0", ...}
```

| Field | Meaning |
|---|---|
| `id` | Group id, `dmgap_XXXXXX`, stable across the whole pipeline |
| `k`, `type` | Variant index and name (table above) |
| `question`, `answer` | The variant's problem (LaTeX) and its ground-truth final answer |
| `answer_type` | `numerical`, `expression`, `set_interval`, `equation` or `other` (regex heuristic, §3 Stage 0) |
| `difficulty` | DeepMath-103K difficulty, 3.0–9.5 in 0.5 steps |
| `topic` | DeepMath-103K topic path, e.g. `Mathematics -> Calculus -> Integral Calculus` |

**Answers can be symbolic, and can contain renamed symbols.** GS and DLM
apply the same renaming to the answer as to the question. So when a gold
answer mentions a variable from the problem, the GS/DLM answer mentions its
new name. For example, `\dfrac{1}{n+1}` becomes `\dfrac{1}{ouua8043o2arrq+1}`.
Symbols that first appear in the answer (for example `m` in an auxiliary
equation `m^2 + 1 = 0`) are left unchanged. Compare answers with a symbolic
checker such as [`math_verify`](https://github.com/huggingface/math-verify),
not by string equality.

`data/final/metadata.jsonl` is a per-group sidecar. Its only field,
`held_out`, is `true` for the 45 groups with source difficulty ≥ 9.0, a pool
originally set aside for held-out evaluation.

### Composition (36,057 groups)

| Answer type | Groups | | Topic (2nd level) | Groups |
|---|---|---|---|---|
| numerical | 23,659 | | Calculus | 11,295 |
| expression | 11,592 | | Algebra | 9,515 |
| set_interval | 585 | | Precalculus | 4,822 |
| other | 140 | | Number Theory | 3,064 |
| equation | 81 | | Applied Mathematics | 2,538 |
| | | | Geometry | 2,237 |
| | | | Discrete Mathematics | 1,756 |
| | | | Other | 573 |
| | | | Differential Equations | 257 |

Difficulty: 3.0–3.5: 1,902 · 4.0–4.5: 4,162 · 5.0–5.5: 13,038 · 6.0–6.5: 9,727 ·
7.0–7.5: 4,644 · 8.0–8.5: 2,539 · 9.0–9.5: 45.

---

## 2. Pipeline overview

```
DeepMath-103K (103,022 problems)
  │ Stage 0  prepare        filter + classify + assign IDs            local
  ▼
81,019 problems
  │ Stage 1  tagging        vars / params / sci_consts                GPT-4.1-mini
  ├──────────────────┬──────────────────────┐
  ▼                  ▼                      ▼
Stage 2 gs        Stage 3 dlm            Stage 4 kv
80,137 ok         79,904 ok              39,508 ok (synthesis + 3 blind solves)
  local           GPT-4.1-mini           o4-mini
  └──────────────────┴──────────────────────┘
  │ Stage 5  assemble       keep a group only if GS, DLM and KV all succeeded
  ▼
38,942 groups  (v1, unfiltered)
  │ Stage 6  postprocess    named-quantity filter v1                   −1,465
  │                         9-gram contamination check                 −396
  │                         named-quantity filter v2                   −1,024
  ▼
36,057 groups  →  data/final/deepmathgap_v2.jsonl.gz
```

Groups are only ever dropped whole. Every group in the final dataset has
all four variants, so per-group statistics (such as the variance of reward
across `k = 0..3`) always compare four variants.

---

## 3. Construction, stage by stage

### Stage 0: filtering (`pipeline/prepare.py`)

- Downloads `zwhe99/DeepMath-103K` (103,022 problems) and drops the three
  `r1_solution_*` columns before anything is written to disk. These are
  DeepSeek-R1 reasoning traces; keeping them would open a
  double-distillation path.
- Drops problems with difficulty < 3.0, which give little invariance
  signal.
- Classifies each `final_answer` with regex heuristics into `numerical`,
  `expression`, `set_interval`, `equation`, `boolean`, `multiple_choice`
  or `other`. It rejects only `boolean` and `multiple_choice`, which cannot
  be perturbed. `other` is kept because the classifier has too many false
  positives to drop it blindly.
- Assigns sequential IDs in source order. Problems with difficulty ≥ 9.0
  (363) are flagged `held_out`. They were meant to be a separate held-out
  split, but only 45 survived the pipeline, so they stay in the main file
  and are marked in `metadata.jsonl`.

Result: 81,019 kept, 22,003 rejected.

### Stage 1: token tagging (`pipeline/tagging.py`, GPT-4.1-mini)

One call per problem, using strict JSON-schema output. It labels the
identifiers in the question:

- `vars`: unknowns or introduced quantities (`x`, `n`, `theta`)
- `params`: problem-specific named constants (a coefficient `k`, a point `S`)
- `sci_consts`: universal constants (`pi`, `e`, `i`)

GS and DLM rename `vars ∪ params`. KV must never resample `sci_consts`.
Tokens tagged as both are removed from vars/params. Tokens that would not
actually match in the question text are also dropped (`pipeline/merge.py`),
so they cannot become silent no-op substitutions. This replaces GAP's
regex-based variable extraction, which misses multi-word tokens and
cannot tell a free variable from a constant.

### Stage 2: GS, Garbled String (`pipeline/gs.py`, local, deterministic)

Each tagged token is renamed to a random letter-first alphanumeric string of
length 4–16. The random generator is seeded per problem with
`SHA-256(id | "GS")`, so any single problem can be regenerated on its own. A
name that already occurs in the problem is redrawn (up to 5 retries).

Substitution (`pipeline/substitution.py`) matches all tokens in one regex
pass over the original text, longest first. It is LaTeX-aware:

- `2x` becomes `2 \cdot lr4dr`, not `2lr4dr`.
- `x^2` becomes `{lr4dr}^2`, so the exponent still binds to the whole name.
- A token followed by a digit is not matched, so `x` in `x2` stays intact.

Problems with no taggable token are rejected, because their variant would
be an exact copy. 80,137 of 81,019 problems succeeded.

### Stage 3: DLM, Descriptive Long Misleading (`pipeline/dlm.py`, GPT-4.1-mini)

For every tagged token, one call proposes a replacement that (1) is a real
mathematical concept, (2) comes from a different subfield than the
problem's topic, and (3) actively misdirects. The model also rates each
replacement's misdirection from 0 to 3. The replacements are substituted
with the same substitution code as GS. A variant is rejected if two
replacements coincide or a replacement already appears in the question.
Low misdirection scores (< 2) are **logged, not rejected**
(`dlm_low_misdirection_warnings.json`). 53 groups in the final dataset
have such a replacement. 79,904 of 81,019 problems succeeded.

### Stage 4: KV, Kernel Variant (`pipeline/kv.py`, `pipeline/verify.py`, o4-mini)

Each attempt has three steps:

1. **Synthesis.** One call finds the mutable numeric constants (never
   `sci_consts`), infers valid ranges from the problem's own constraints,
   resamples them, rewrites the question with only those values changed,
   and solves the new question.
2. **Blind solve.** Three independent calls solve the new question without
   seeing the synthesised answer.
3. **Acceptance** (local). All of the following must hold:
   - the change list is non-empty and actually changes a value;
   - the three blind answers are pairwise equivalent under `math_verify`;
   - the blind answers are equivalent to the synthesised answer;
   - after masking every old and new constant (digit or word form, e.g.
     `6`/`six`, and ordinal suffixes such as `21st` → `19th`), the two
     questions are token-for-token identical, so no clause was added,
     removed or reworded.

Failed problems are retried in the next attempt. 31,283 were accepted at
attempt 1 and 8,225 at attempt 2, for 39,508 in total. A third synthesis
batch was generated, but its blind-solve verification was not run, so the
41,511 problems still failing after attempt 2 have no kernel variant.

Two implementation details matter if you re-run this stage:

- `math_verify`'s built-in timeout uses `signal.alarm` on POSIX and a
  subprocess on Windows. The subprocess can fail silently, making every
  comparison return `False`. The pipeline disables those timeouts and uses
  its own daemon-thread timeout, so any comparison that times out or errors
  counts as "not equivalent".
- Answers are wrapped in `\boxed{}` before parsing, because otherwise
  `math_verify` fails to parse `\dfrac` and bare symbolic answers.
  Exponent towers such as `37^{4^{15}}` are rejected up front, because
  evaluating them would try to build billion-digit integers.

The design follows GAP's `kv_math_200.py`, which uses slot discovery and
back-synthesis, but with two changes: the two calls are merged into one, and
GAP's three-judge accept/reject vote is replaced by independent blind
solving plus the structural check.

### Stage 5: assembly (`pipeline/assemble.py`, local)

For each problem, this writes the four rows `k = 0..3`, but only if GS, DLM
and KV all succeeded. The 42,077 problems with a missing variant are listed
with their reasons in `data/audit/rejected_groups.jsonl`. KV is the
bottleneck: 40,888 of those are missing only the kernel variant. The result
is the unfiltered v1 set: 38,942 groups.

### Stage 6: post-processing (`pipeline/postprocess.py`, local)

**Named-quantity filters.** Renaming a free variable (`x`, `n`) keeps the
problem the same. Renaming a token whose meaning depends on an outside
convention does not: renaming `radius` loses the link to `area = πr²`.
The same happens when a token carries a value: renaming `(0, 0)` or `x = 0`
throws the value away. DeepMath-103K has many word problems in which such
tokens appear and get tagged. PutnamGAP largely avoids this because Putnam
problems are formal. A group is dropped if a token in its **actual** GS or
DLM rename map matches a filter (`pipeline/named_quantity.py`):

- **v1 (−1,465 groups):** a word of the token is on a list of named
  quantities (`radius`, `area`, `mean`, `distance`, `probability`, …).
- **v2 (−1,024 groups):** every renamed token that looked like an English
  word or phrase was extracted: 6,757 candidates, listed in
  `named_quantity_candidates.json`. Each one was reviewed by hand. The
  confirmed tokens form `CONFIRMED_DROPS`: named objects (`matrix`,
  `determinant`, `origin`), shapes (`circle`, `sphere`), statistics terms
  (`expected value`), number-theory terms (`remainder`, `units digit`),
  named sets (`positive integers`), time units, and number words. In
  addition, four regex patterns catch value-bearing tokens. Groups
  affected: coordinate tuples 449, named quantities 326, `var = value`
  equations 203, number + unit 65, clock times 4. One group can match more
  than one category.

**Contamination (−396 groups).** DeepMath-103K was already decontaminated
by its authors against 14 benchmarks (MATH, AIME, AMC, OlympiadBench,
GSM8K, …). This check covers three newer benchmarks: MATH-Perturb (558
problems), AIME 2025 (45) and ASyMOB (35,368). A group is dropped if its
original question shares any whitespace-token 9-gram with a benchmark
problem. Hits: MATH-Perturb 336, AIME 2025 83, ASyMOB 0 (some groups hit
both). Exact n-gram overlap will not catch paraphrased duplicates.

### Cost

Everything ran through the OpenAI Batch API, which is 50% cheaper than
synchronous calls. The total spend was **$2,570.78**. Most of it went to
the o4-mini synthesis and blind-solve calls of Stage 4. Tagging and DLM
were budgeted at about $10 and $20; per-stage spend was not metered.

---

## 4. What was adapted from GAP and what is new

| Component | Origin |
|---|---|
| GS/DLM/KV variant taxonomy | GAP / PutnamGAP |
| GS name generation and substitution boundary rule | Adapted from GAP `mini_gap_math.py`: seeded per problem, letter-first names of length 4–16, collision check, single-pass LaTeX-aware substitution |
| KV synthesis | Adapted from GAP `kv_math_200.py`: one merged call; blind-solve and `math_verify` verification instead of a judge vote |
| LLM token tagging, DLM generation with misdirection scoring, structural diff check, answer-type classification, assembly, named-quantity filters, contamination check | New |

---

## 5. Reproducing the dataset

```bash
cd DeepMathGAP
pip install -r requirements.txt
cp .env.example .env                     # set OPENAI_API_KEY (Stages 1, 3, 4 only)

python -m pipeline.prepare                                   # Stage 0
python -m pipeline.tagging --submit                          # Stage 1 (batch)
python -m pipeline.tagging --poll-and-fetch <batch_ids>
python -m pipeline.gs                                        # Stage 2
python -m pipeline.dlm --submit                              # Stage 3 (batch)
python -m pipeline.dlm --poll-and-fetch <batch_ids>
python -m pipeline.dlm --apply

# Stage 4, per attempt N = 1, 2, 3 (attempts 2 and 3 add
# --pending data/intermediate/kv/pending_attemptN.json to build-synth)
python -m pipeline.kv build-synth      --attempt N --submit
python -m pipeline.kv poll-synth       --attempt N --batch-id <batch_ids>
python -m pipeline.kv build-blindsolve --attempt N --submit
python -m pipeline.kv poll-blindsolve  --attempt N --batch-id <batch_ids>
python -m pipeline.kv evaluate         --attempt N

python -m pipeline.assemble                                  # Stage 5
python -m pipeline.contamination                             # download benchmarks (once)
python -m pipeline.postprocess                               # Stage 6 -> data/final/
```

Every command that spends money needs an explicit `--submit`. Without it,
the command only writes the request file to `data/intermediate/batches/`.
Each batch has a 24-hour completion window, so the `--poll-*` steps are run
separately; they print the command to use. Every stage reads and writes
default paths under `data/intermediate/` (`pipeline/common.py`), which is
not committed.

GS, assembly and post-processing are deterministic. Fed the original
tagging, DLM and KV outputs, this code regenerates the final dataset and
every audit file exactly. The LLM stages are not bit-reproducible: a
re-run gives a dataset of the same shape, but not the same file.

### Repository layout

```
DeepMathGAP/
├── README.md                this document
├── requirements.txt
├── .env.example
├── pipeline/
│   ├── common.py            default paths, JSON helpers
│   ├── prepare.py           Stage 0
│   ├── tagging.py           Stage 1
│   ├── merge.py             joins problems with tags (used by 2–4)
│   ├── gs.py                Stage 2
│   ├── substitution.py      LaTeX-aware renaming (GS, DLM)
│   ├── seeding.py           per-problem seeded RNG
│   ├── dlm.py               Stage 3
│   ├── kv.py                Stage 4
│   ├── verify.py            KV acceptance checks
│   ├── batch_api.py         OpenAI Batch API helpers
│   ├── assemble.py          Stage 5
│   ├── named_quantity.py    Stage 6 named-quantity filters
│   ├── contamination.py     Stage 6 contamination check + benchmark download
│   └── postprocess.py       Stage 6 driver
└── data/
    ├── final/
    │   ├── deepmathgap_v2.jsonl.gz      ← THE DATASET
    │   └── metadata.jsonl               held_out flag per group
    └── audit/
        ├── rejected_groups.jsonl            Stage 5: groups missing a variant, with reasons
        ├── dropped_named_quantity_v1.json   Stage 6: group → matched token/word
        ├── contaminated_groups.json         Stage 6: group → benchmark(s)
        ├── named_quantity_candidates.json   Stage 6: the 6,757 reviewed candidate tokens
        └── dropped_named_quantity_v2.json   Stage 6: group → matched token/category
```

---

## 6. Citation

DeepMathGAP is derived work. If you use it, cite both sources:

```bibtex
@article{he2025deepmath,
  title   = {DeepMath-103K: A Large-Scale, Challenging, Decontaminated, and Verifiable
             Mathematical Dataset for Advancing Reasoning},
  author  = {He, Zhiwei and Liang, Tian and Xu, Jiahao and Liu, Qiuzhi and Chen, Xingyu and
             Wang, Yue and Song, Linfeng and Yu, Dian and Liang, Zhenwen and Wang, Wenxuan and
             Zhang, Zhuosheng and Wang, Rui and Tu, Zhaopeng and Mi, Haitao and Yu, Dong},
  journal = {arXiv preprint arXiv:2504.11456},
  year    = {2025}
}

@article{hao2025gap,
  title   = {An Investigation of Robustness of {LLM}s in Mathematical Reasoning: Benchmarking
             with Mathematically-Equivalent Transformation of Advanced Mathematical Problems},
  author  = {Hao, Yuren and Wan, Xiang and Zhai, ChengXiang},
  journal = {arXiv preprint arXiv:2508.08833},
  year    = {2025}
}
```

Source dataset: [`zwhe99/DeepMath-103K`](https://huggingface.co/datasets/zwhe99/DeepMath-103K) ·
GAP code: [`YurenHao0426/GAP`](https://github.com/YurenHao0426/GAP) ·
Answer checking: [`huggingface/math-verify`](https://github.com/huggingface/math-verify).

## 7. License and third-party attribution

The code is released under the MIT License ([`LICENSE`](../LICENSE)). The
dataset and audit files are released under CC BY 4.0
([`data/LICENSE`](data/LICENSE)).

- **DeepMath-103K** (He et al., 2025) is released under the MIT License.
  Every `question` and `answer` in `k = 0` rows, and the text that the
  `k = 1..3` variants are derived from, comes from DeepMath-103K. Copyright
  in that material belongs to the DeepMath-103K authors, and its MIT
  license notice applies to it.
- **GAP framework** (Hao, Wan & Zhai, 2025) code is released under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Three parts of
  this pipeline are adapted from it. The changes are listed in §4:
  - the garbled-name generator and substitution boundary rule, from
    `mini_gap_math.py` (`pipeline/gs.py`, `pipeline/substitution.py`);
  - the slot-discovery and back-synthesis design, from `kv_math_200.py`
    (`pipeline/kv.py`);
  - the `\boxed{}` answer extraction, from `kv_math_200.py`
    (`pipeline/kv.py`).
- DeepMathGAP contains **no Putnam problems**. The four MAA problem-book
  citations that the GAP repository requires apply to the PutnamGAP
  dataset, not to this one.
