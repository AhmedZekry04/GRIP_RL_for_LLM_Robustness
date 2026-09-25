import os
import wandb
from datasets import load_dataset
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward as _accuracy_reward
import random
import sys
from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError as FTimeout

sys.set_int_max_str_digits(100000)   
#########################################################################
#PATHS
#########################################################################
USERNAME   = os.environ.get("USER", "amz25")
HOME       = f"/rds/general/user/{USERNAME}/home"
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
RUN_ID     = "a1_grpo_originals_v3"
DATA_FILE  = f"{HOME}/GRIP/data/deepmathgap_prepped.jsonl"
OUTPUT_DIR = f"{HOME}/GRIP/train_a1/results/checkpoints/{RUN_ID}"

MATH_INSTRUCTION = r"Please reason step by step, and put your final answer within \boxed{}."



#########################################################################
#FUNCTIONS
#########################################################################
class GRIPMetricCallback(TrainerCallback):
    """
    Adds GRIP-relevant scalar metrics to the same log stream TRL uses, so they
    render as plots alongside reward/kl/entropy in wandb/tensorboard.
 
    TRL already logs the core RLVR health metrics automatically:
      reward, reward_std, kl, entropy, clip_ratio (clip fraction),
      completions/mean_length, grad_norm, learning_rate, loss.
    This callback derives a few extra ones from the reward tensor TRL exposes in
    the log dict on each logging step. All processes must log the same keys in
    distributed training, so we always emit every key (0.0 if unavailable).
    """
    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if logs is None:
            return
        # TRL logs "reward" and "reward_std" already; these callbacks add a few
        # derived stability signals so a single dashboard tells the whole story.
        reward     = logs.get("reward", None)
        reward_std = logs.get("reward_std", None)
        entropy    = logs.get("entropy", None)
 
        derived = {}
        # reward coefficient of variation: a compact "is the signal collapsing?"
        if reward is not None and reward_std is not None and abs(reward) > 1e-8:
            derived["grip/reward_cv"] = float(reward_std) / float(reward)
        # entropy is the canonical collapse alarm; mirror it under a grip/ key so
        # it sits next to the other grip metrics on the dashboard
        if entropy is not None:
            derived["grip/entropy_mirror"] = float(entropy)
 
        if derived:
            # state.log_history is append-only; use the trainer's logger via logs
            for k, v in derived.items():
                logs[k] = v
 

def build_prompt(example):
    example["prompt"] = f"{example['question']}\n{MATH_INSTRUCTION}"
    return example


# Example of a multi-task reward function (math-specific).

def _score_one(payload):
    """One (completion_str, solution_str) pair in an isolated process."""
    content, sol = payload
    from trl.rewards import accuracy_reward as _ar
    return _ar([[{"role": "assistant", "content": content}]], [sol])[0]


def accuracy_reward(completions, solution, **kwargs):
    """
    Wraps TRL's accuracy_reward for base-model raw-string completions,
    with per-completion process isolation so a sympy hang cannot stall a rank.
    """
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
        print(f"[reward] {n_timeout}/{len(pairs)} timed out -> skipped",
              flush=True)
    return out

#########################################################################
#TRAINER
#########################################################################

def main():
    os.makedirs(f"{OUTPUT_DIR}/logs", exist_ok=True)
    # load the prepared dataset
    N_EVAL_SEEDS = 100
    ds = load_dataset("json", data_files=DATA_FILE, split="train")
    ds = ds.shuffle(seed=42)
    #ds = ds.rename_column("answer", "solution")
    ds = ds.map(build_prompt)
    #ds = DatasetDict.from_json(path_or_paths="/home/amz/projects/GRIP/data/processed/deepmathgap/deepmathgap.jsonl", features=)
    # filter to k=0 only (the seed/original problems)
    #ds = ds.filter(lambda x: x["type"] == "original")




    # ── seed-level split: sample SEEDS, keep each seed's group whole ──────────
    all_ids = sorted(set(ds["id"]))          # every unique seed id, sorted for determinism
    rng = random.Random(42)                  # fixed seed -> same split in A1, A2, A3
    rng.shuffle(all_ids)                     # randomise seed order

    
    eval_ids  = set(all_ids[:N_EVAL_SEEDS])        # first 5% after shuffling -> eval
    train_ids = set(all_ids[N_EVAL_SEEDS:])        # remaining 95% -> train

    eval_ds  = ds.filter(lambda x: x["id"] in eval_ids)    # all rows of eval seeds
    train_ds = ds.filter(lambda x: x["id"] in train_ids)   # all rows of train seeds

    # ── A1 only: train on originals. For A2/A3, delete this line. ────────────
    train_ds = train_ds.filter(lambda x: x["type"] == "original")

    train_ds = train_ds.shuffle(seed=42)     # order matters: decides what co-occurs per step

    print(f"seeds: {len(all_ids)} (train {len(train_ids)} / eval {len(eval_ids)})")
    print(f"rows:  train={len(train_ds)}  eval={len(eval_ds)}")


    # ── wandb resume ─────────────────────────────────────────────────────────
    RUN_ID_FILE = f"{OUTPUT_DIR}/wandb_run_id.txt"
    if os.path.exists(RUN_ID_FILE):
        wid = open(RUN_ID_FILE).read().strip()
        os.environ["WANDB_RUN_ID"] = wid        # "htrjr3z7" on first resume
        os.environ["WANDB_RESUME"] = "must"
        print(f"resuming wandb run {wid}", flush=True)
    
    else:
        import secrets, string
        wid = "".join(secrets.choice(string.ascii_lowercase + string.digits)
                      for _ in range(8))
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(RUN_ID_FILE, "w") as f:
            f.write(wid)
        os.environ["WANDB_RUN_ID"] = wid
        os.environ["WANDB_RESUME"] = "allow"
        print(f"new wandb run {wid}", flush=True)


    grpo_configuration = GRPOConfig(

        output_dir=OUTPUT_DIR,
        run_name=RUN_ID,
        model_init_kwargs={"dtype": "bfloat16"},
        seed = 42,
        data_seed = 42,

 
        # Loss formulation (controlled: identical in A2; A3 adds to it)
        loss_type="dapo",
        # DAPO token-level normalisation. Plain "grpo" has response-length bias
        # (2503.20783): it under-penalises long wrong answers. The variants here have
        # different lengths, so that bias would be an uncontrolled confound.
        # DAPO (2503.14476) normalises by total active tokens, removing it.
 
        beta=0.0,
        # No KL penalty, so the reference model is not loaded (saves ~14GB VRAM
        # and speeds training). Standard for Zero-RL on base models
        # (Open-Reasoner-Zero 2503.24290, DAPO). There is no instruction-tuned
        # behaviour to stay close to, so a reference anchor is unnecessary.
 
        epsilon=0.2,
        # Lower PPO clip bound. Standard GRPO/PPO value.
 
        epsilon_high=0.28,
        # Upper clip bound ("clip-higher" from DAPO 2503.14476). Decoupling the
        # upper bound to 0.28 counteracts entropy collapse - the canonical RLVR
        # failure where the policy goes deterministic and rollouts stop differing.
        # This is the single most important stability lever for base-model RL.
 
        num_iterations=1,
        # μ=1: one optimiser update per generation batch (on-policy, TRL default).
 
        scale_rewards="group",
        # Std-normalise advantages within each group (default GRPO). If difficulty
        # bias appears (2503.20783), switch to False or "batch"; either way keep
        # the setting identical across all three ablations.
 
        mask_truncated_completions=True,
        # DAPO (2503.14476) best practice: exclude completions that hit the length
        # cap from the loss. A truncated trace isn't a "wrong answer" - counting it
        # as one injects noise. Important here because base-model traces can be long.
        # Evaluation  (held out in distribution evaluation)
        eval_strategy = "steps",
        eval_steps = 50,
        per_device_eval_batch_size = 4,

        # GENERATION  (controlled: identical in A2/A3 AND matched to A0 eval)
        temperature=0.6,               # same operating point as A0 eval
        top_p=0.95,                    # same as A0 eval
        max_completion_length=3072,    # room for long CoT (matches DeepMath recipe)
        num_generations=8,             # G=8 rollouts per prompt (group size);
        #vllm_enable_sleep_mode=True,
 
        # BATCH / OPTIMISATION  (controlled: identical across ablations)
        per_device_train_batch_size=2,
        gradient_accumulation_steps=64,
        # Effective batch = per_device(8) × grad_accum(32) × num_processes(1 GPUs)
        # = 256. Must be divisible by num_generations(8): 256/8 = 32 prompts/step. OK.
        learning_rate=2e-5,
        # Low LR standard for RL fine-tuning (DeepMath recipe). Policy updates
        # compound over steps, so LR is ~100x lower than supervised fine-tuning.
 
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,               # short warmup (DeepMath recipe)
        weight_decay=0.1,              # DeepMath recipe
        max_grad_norm=1.0,             # gradient clipping — stability
 
        # vLLM GENERATION  (colocate: shares the training GPUs)
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.12,
        # Lower than eval's 0.9: in colocate the optimiser states + gradients + KV
        # cache share the card. 0.4 leaves headroom for the backward pass. Tune
        # down if memory is tight, but keep it identical across ablations.
        vllm_tensor_parallel_size=1,
        # Each GPU runs its own vLLM replica (data-parallel generation). For a 7B
        # on 48GB L40S, TP=1 is correct: TP adds comms overhead with no benefit here.
        vllm_max_model_length=4096,    # max_prompt(1024)+max_completion(3072)=4096
 
        # TRAINING LENGTH & CHECKPOINTS  (controlled)
        max_steps=500,                 # DeepMath recipe uses 500 GRPO steps
        save_steps=50,                 # checkpoint every 50 steps → 10 checkpoints
        save_total_limit=4,            # keep last 4 to bound disk
 
        # LOGGING  (controlled: same metrics across ablations for comparison)
        logging_steps=1,               # log every step, to keep the full curve
        log_completions=False,          # sample (prompt, completion) pairs
        num_completions_to_print=4,    # cap console spam
        report_to="wandb",              # switch to "wandb" for live curves if set up
        #logging_dir=f"{OUTPUT_DIR}/logs",
 
        remove_unused_columns=False,
 
        # PRECISION / MEMORY  (controlled)
        bf16=True,
        gradient_checkpointing=True,   # trade compute for memory — needed for 7B on L40S
        gradient_checkpointing_kwargs = {"use_reentrant": False}
    )
    

    trainer = GRPOTrainer(
        model = MODEL_NAME,
        args = grpo_configuration,
        reward_funcs = accuracy_reward,
        train_dataset = train_ds,
        # runs a small generation pass on held-out prompts every eval_steps steps and logs reward statistics on that set.
        # for out-of-sample reward monitoring, to see see whether the reward improvements on training data are generalising to unseen problems, or whether the model is overfitting to the training distribution.
        eval_dataset = eval_ds,
        callbacks = [GRIPMetricCallback()]
    )
    
    print("param dtype:", next(trainer.model.parameters()).dtype, flush=True)

    last_ckpt = None
    if os.path.isdir(OUTPUT_DIR):
        cks = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if cks:
            last_ckpt = os.path.join(OUTPUT_DIR, max(cks, key=lambda d: int(d.split("-")[1])))
            print(f"resuming from {last_ckpt}", flush=True)

    trainer.train(resume_from_checkpoint=last_ckpt)

    trainer.save_model(f"{OUTPUT_DIR}/final")


#########################################################################
#RUN
#########################################################################

if __name__ == "__main__":
    main()
