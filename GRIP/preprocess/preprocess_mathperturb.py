import json
import statistics
from pathlib import Path

from math_verify import parse
from transformers import AutoTokenizer

from gold_normalize import normalize_gold

SCRIPT_DIR = Path(__file__).resolve().parent   # preprocess/
REPO_ROOT = SCRIPT_DIR.parent                  # repo root
DATA_ROOT = REPO_ROOT / "data"

IN_PATH = DATA_ROOT / "eval" / "MATH-Perturb" / "math_perturb_final.jsonl"
OUT_PATH = DATA_ROOT / "mathperturb_prepped.jsonl"

TOKENIZER_NAME = "Qwen/Qwen2.5-Math-7B"
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
OVERLONG_THRESHOLD = 1024
ARM_KEYS = ("original", "simple", "hard")


def detect_arm(row):
    present = [k for k in ARM_KEYS if k in row]
    if len(present) != 1:
        raise ValueError(
            f"problem_id={row.get('problem_id')} has {len(present)} of {ARM_KEYS} present: {present}"
        )
    return present[0]


def main():
    print(f"reading from {IN_PATH}")
    print(f"writing to {OUT_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    with open(IN_PATH) as f:
        raw_lines = f.readlines()
    n_rows = len(raw_lines)

    flagged_gold_count = 0
    parse_fail_count = 0
    parse_fail_examples = []
    arms_by_group = {}

    prompts = []
    prepped = []

    for line in raw_lines:
        d = json.loads(line)
        arm = detect_arm(d)
        problem_id = d["problem_id"]
        group_id = str(problem_id)
        problem = d[arm]
        answer_raw = d[f"{arm}_answer"]

        gold_answers, flagged = normalize_gold(answer_raw)
        if flagged:
            flagged_gold_count += 1

        for g in gold_answers:
            try:
                parse(f"${g}$")
            except Exception as e:
                parse_fail_count += 1
                if len(parse_fail_examples) < 5:
                    parse_fail_examples.append((group_id, arm, g, str(e)))

        arms_by_group.setdefault(group_id, set()).add(arm)

        prompts.append(f"{problem}\n{INSTRUCTION}")

        prepped.append({
            "uid": f"{problem_id}__{arm}",
            "group_id": group_id,
            "arm": arm,
            "is_original": arm == "original",
            "problem": problem,
            "gold_answers": gold_answers,
            "gold_format": "latex",
            "n_samples": 8,
            "meta": {
                "level": d["level"],
                "subject": d["type"],
                "original_split": d["original_split"],
            },
        })

    token_ids = tokenizer(prompts)["input_ids"]
    n_tokens = [len(t) for t in token_ids]
    for row, n in zip(prepped, n_tokens):
        row["n_prompt_tokens"] = n

    uids = [row["uid"] for row in prepped]
    arm_counts = {arm: sum(1 for row in prepped if row["arm"] == arm) for arm in ARM_KEYS}
    missing_original = sorted(g for g, arms in arms_by_group.items() if "original" not in arms)
    missing_perturbed = sorted(
        g for g, arms in arms_by_group.items() if not (arms & {"simple", "hard"})
    )
    n_overlong = sum(1 for n in n_tokens if n > OVERLONG_THRESHOLD)

    print(f"rows: {n_rows}")
    print(f"unique uids: {len(set(uids))}")
    print(f"arm counts: {arm_counts}")
    print(f"groups missing 'original': {len(missing_original)} {missing_original[:10]}")
    print(f"groups missing any perturbed arm: {len(missing_perturbed)} {missing_perturbed[:10]}")
    print(f"gold values flagged (normalization collapsed to nothing): {flagged_gold_count}")
    print(f"math_verify parse failures: {parse_fail_count}")
    for ex in parse_fail_examples:
        print(f"  example: problem_id={ex[0]} arm={ex[1]} gold={ex[2]!r} error={ex[3]}")
    print(
        f"n_prompt_tokens: max={max(n_tokens)} "
        f"p99={statistics.quantiles(n_tokens, n=100)[98]:.1f} "
        f"mean={statistics.mean(n_tokens):.1f}"
    )
    print(f"rows with n_prompt_tokens > {OVERLONG_THRESHOLD}: {n_overlong}")

    assert len(set(uids)) == n_rows, f"uid collision: {n_rows - len(set(uids))} duplicate uid(s)"

    with open(OUT_PATH, "w") as f:
        for row in prepped:
            f.write(json.dumps(row) + "\n")

    print(f"wrote {len(prepped)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
