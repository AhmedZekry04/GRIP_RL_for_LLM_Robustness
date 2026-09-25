"""
LR probe: A2 configuration, 60 steps, learning rate from $GRIP_LR.

Identical to train_a2.py except:
  - learning_rate comes from the environment, and RUN_ID encodes it
  - max_steps=60 (10 warmup + 50 measurable)
  - eval disabled (62 min/pass, tells us nothing over 60 steps)
  - no checkpoints, no final save
Everything that could confound the comparison - data, seeds, batch, generation,
scale_rewards, loss_type - is untouched.
"""

import os, random, sys, secrets, string
sys.set_int_max_str_digits(100000)

from datasets import load_dataset
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward as _accuracy_reward
from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError as FTimeout

USERNAME   = os.environ.get("USER", "amz25")
HOME       = f"/rds/general/user/{USERNAME}/home"
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
DATA_FILE  = f"{HOME}/GRIP/data/deepmathgap_prepped.jsonl"

LR         = 5e-5# fail loudly if unset
RUN_ID     = f"lrprobe_{LR:.0e}".replace("-0", "-")
OUTPUT_DIR = f"{HOME}/GRIP/train_a2/results/checkpoints/{RUN_ID}"

MATH_INSTRUCTION = r"Please reason step by step, and put your final answer within \boxed{}."
N_EVAL_SEEDS = 100          # kept so the train split matches A2 exactly
MAX_STEPS    = 60


# ── reward: per-completion process isolation so a sympy hang cannot stall a rank
def _score_one(payload):
    content, sol = payload
    from trl.rewards import accuracy_reward as _ar
    return _ar([[{"role": "assistant", "content": content}]], [sol])[0]


def accuracy_reward(completions, solution, **kwargs):
    pairs = list(zip(completions, solution))
    out = [None] * len(pairs)
    n_timeout = 0
    with ProcessPool(max_workers=8, max_tasks=200) as pool:
        futs = {pool.schedule(_score_one, args=[p], timeout=20): i
                for i, p in enumerate(pairs)}
        for fut, i in futs.items():
            try:
                out[i] = fut.result()
            except (FTimeout, ProcessExpired):
                n_timeout += 1
            except Exception:
                pass
    if n_timeout:
        print(f"[reward] {n_timeout}/{len(pairs)} timed out -> skipped", flush=True)
    return out


class GRIPMetricCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if logs is None:
            return
        r, s = logs.get("reward"), logs.get("reward_std")
        if r is not None and s is not None and abs(r) > 1e-8:
            logs["grip/reward_cv"] = float(s) / float(r)


def build_prompt(example):
    example["prompt"] = f"{example['question']}\n{MATH_INSTRUCTION}"
    return example


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── wandb: one id per lr, written up front so both ranks read the same file
    RUN_ID_FILE = f"{OUTPUT_DIR}/wandb_run_id.txt"
    if os.path.exists(RUN_ID_FILE):
        wid = open(RUN_ID_FILE).read().strip()
        os.environ["WANDB_RESUME"] = "must"
    else:
        wid = "".join(secrets.choice(string.ascii_lowercase + string.digits)
                      for _ in range(8))
        with open(RUN_ID_FILE, "w") as f:
            f.write(wid)
        os.environ["WANDB_RESUME"] = "allow"
    os.environ["WANDB_RUN_ID"] = wid
    print(f"lr={LR:.0e}  run_id={RUN_ID}  wandb={wid}", flush=True)

    # ── data: identical split to A2 ──
    ds = load_dataset("json", data_files=DATA_FILE, split="train")
    ds = ds.shuffle(seed=42)
    ds = ds.map(build_prompt)

    all_ids = sorted(set(ds["id"]))
    rng = random.Random(42)
    rng.shuffle(all_ids)
    eval_ids  = set(all_ids[:N_EVAL_SEEDS])
    train_ids = set(all_ids[N_EVAL_SEEDS:])
    train_ds = ds.filter(lambda x: x["id"] in train_ids)   # A2: no type filter
    train_ds = train_ds.shuffle(seed=42)
    print(f"train rows: {len(train_ds)}", flush=True)

    cfg = GRPOConfig(
        output_dir=OUTPUT_DIR,
        run_name=RUN_ID,
        model_init_kwargs={"dtype": "bfloat16"},
        seed=42, data_seed=42,

        # ── THE only VARIABLE ──
        learning_rate=LR,

        # ── everything below is A2, unchanged ──
        loss_type="dapo",
        beta=0.0,
        epsilon=0.2,
        epsilon_high=0.28,
        num_iterations=1,
        scale_rewards="group",
        mask_truncated_completions=True,

        temperature=0.6,
        top_p=0.95,
        max_completion_length=3072,
        num_generations=8,

        per_device_train_batch_size=2,
        gradient_accumulation_steps=64,     # 2 x 64 x 2 GPUs = 256 completions

        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        weight_decay=0.1,
        max_grad_norm=1.0,

        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.12,
        vllm_tensor_parallel_size=1,
        vllm_max_model_length=4096,

        # ── probe-specific ──
        max_steps=MAX_STEPS,
        eval_strategy="no",
        save_steps=10**6,
        save_total_limit=1,
        logging_steps=1,
        log_completions=False,
        report_to="wandb",

        remove_unused_columns=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = GRPOTrainer(
        model=MODEL_NAME,
        args=cfg,
        reward_funcs=accuracy_reward,
        train_dataset=train_ds,
        callbacks=[GRIPMetricCallback()],
    )
    print("param dtype:", next(trainer.model.parameters()).dtype, flush=True)
    trainer.train()          # no resume, no save -- this is a measurement only


if __name__ == "__main__":
    main()