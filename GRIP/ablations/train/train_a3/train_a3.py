"""
A3: GRPO + GRIP invariance penalty.

Identical to A2 (originals + variants, GRPO/DAPO) with one change: an exact
advantage adjustment implementing

    L_GRIP = L_GRPO + (gamma - 1) * L_inv
    L_inv  = (1/K) sum_k (a_hat_k - a_bar)^2          per seed, averaged over seeds

which, via the score-function identity, reduces exactly to

    A_grip[k,g] = A_grpo[k,g] * [ 1 - 2*lambda*(a_hat_k - a_bar)*std_k ]

i.e. a MULTIPLICATIVE rescaling of the advantage, with lambda = gamma - 1.

This uses the BASELINE-CORRECTED score-function estimator
    grad a_hat_k = E[(r - a_hat_k) grad log pi]        (unbiased for any baseline)
rather than the raw form E[r grad log pi]. Both are unbiased, but the raw form
distributes gradient mass as G*a_hat_k - proportional to how often the variant is
already solved - so suppression of easy variants outweighs lift of hard ones
(1.71:1 at the observed accuracies) and Var_k never falls. The baselined form has
mass 2*G*a(1-a), symmetric under a <-> 1-a.

Because A = (r - a_hat_k)/std_k, substituting gives the multiplicative form above,
whose factor is strictly positive for lambda < ~8: it can never flip the sign of an
advantage, so correct completions are never unlearned and entropy does not inflate.

The K and G factors cancel because L_inv is averaged over the seeds in the batch,
giving its gradient the same 1/N prefactor as the GRPO term.

Every TRL internal this relies on was read from trl==1.10.0 source:
  - _calculate_rewards returns the GATHERED rewards_per_func (line 1727)
  - gather order == sampler order; rows 8p..8p+7 are prompt p (line 2755)
  - local rows are gathered[rank*n : (rank+1)*n]              (line 2803)
  - per_token_loss = -min(c1*A, c2*A), so +A raises log pi     (line 3136)
  - unscorable rows -> NaN reward -> advantage forced to 0     (line 2800)
  - shuffle_dataset is a GRPOConfig field                     (config line 504)
"""

import os, random, sys, secrets, string
sys.set_int_max_str_digits(100000)

import torch
from datasets import load_dataset
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward as _accuracy_reward
from accelerate.utils import gather
from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError as FTimeout

# PATHS AND CONSTANTS
USERNAME   = os.environ.get("USER", "amz25")
HOME       = f"/rds/general/user/{USERNAME}/home"
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
DATA_FILE  = f"{HOME}/GRIP/data/deepmathgap_prepped.jsonl"

GRIP_LAMBDA = float(os.environ.get("GRIP_LAMBDA", "2.5"))  # = gamma - 1, gamma = 3.5
# Calibrated to the MULTIPLICATIVE form below: adv_ratio ~= 0.111 * lambda at the
# observed within-seed spread, so lambda=2.5 lands at ~0.28, inside the 0.2-0.5 band.
# gamma=2 (the anchor-regression default) would give ~0.11 here and be too weak.
RUN_ID      = os.environ.get("GRIP_RUN_ID", f"a3_grpo_grip_g{1+GRIP_LAMBDA:g}")
OUTPUT_DIR  = f"{HOME}/GRIP/train_a3/results/checkpoints/{RUN_ID}"

K_VARIANTS   = 4          # original, surface_gs, surface_dlm, kernel
N_EVAL_SEEDS = 100        # MUST match A1/A2
LR           = 2e-5       # from the LR probe; MUST match A1/A2 reruns
MAX_STEPS    = int(os.environ.get("GRIP_MAX_STEPS", "500"))

MATH_INSTRUCTION = r"Please reason step by step, and put your final answer within \boxed{}."


# REWARD  (unchanged from A1/A2: process-isolated so a sympy hang cannot stall a rank)
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


# GRIP TRAINER
class GRIPTrainer(GRPOTrainer):
    """
    GRPOTrainer with the GRIP advantage adjustment applied on the GATHERED
    generation batch, so it is correct even when a prompt's G rollouts are
    split across GPUs (which TRL's sampler does by design).
    """

    def __init__(self, *args, grip_lambda: float, grip_k: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.grip_lambda = float(grip_lambda)
        self.grip_k = int(grip_k)
        self._grip_rpf = None          # gathered rewards_per_func, cached per generation batch
        self._grip_checked = False

    # ── 1. cache the gathered rewards ────────────────────────────────────────
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        rpf = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        self._grip_rpf = rpf.detach()          # already gathered: (N_total, n_funcs)
        return rpf

    # ── 2. compute the adjustment on the full batch, slice to local rows ─────
    def _generate_and_score_completions(self, inputs):
        out = super()._generate_and_score_completions(inputs)
        mode = "train" if self.model.training else "eval"
        if mode != "train" or self._grip_rpf is None or self.grip_lambda == 0.0:
            return out

        device = self.accelerator.device
        G = self.num_generations
        K = self.grip_k
        n_local = len(inputs)

        # ── gathered keys, aligned row-for-row with the gathered rewards ──
        keys_local = torch.tensor(
            [[int(x["seed_int"]), int(x["k"])] for x in inputs],
            dtype=torch.long, device=device)
        keys = gather(keys_local)                                   # (N_total, 2)
        seed, kk = keys[:, 0], keys[:, 1]

        # ── scalar reward per row, NaN where unscorable (mirrors TRL) ──
        rpf = self._grip_rpf
        unscorable = torch.isnan(rpf).all(dim=1)
        r = (rpf * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        r[unscorable] = float("nan")
        N = r.shape[0]
        if N != keys.shape[0]:
            raise RuntimeError(f"grip: {N} rewards vs {keys.shape[0]} keys -- alignment broken")

        # ── a_hat per (seed, k): nan-aware mean over its G rows ──
        pair = seed * K + kk
        upair, pinv = torch.unique(pair, return_inverse=True)
        valid = ~torch.isnan(r)
        r0 = torch.nan_to_num(r, nan=0.0)
        sum_r = torch.zeros(len(upair), device=device).scatter_add_(0, pinv, r0)
        cnt_r = torch.zeros(len(upair), device=device).scatter_add_(0, pinv, valid.float())
        a_hat = sum_r / cnt_r.clamp_min(1.0)                        # (n_pairs,)
        has_data = (cnt_r > 0).float()      # a variant whose G rollouts are ALL unscorable
                                            # must be excluded from a_bar, not scored as 0

        # ── a_bar per seed: mean of a_hat over its scorable variants ──
        seed_of_pair = upair // K
        useed, sinv = torch.unique(seed_of_pair, return_inverse=True)
        sum_a = torch.zeros(len(useed), device=device).scatter_add_(0, sinv, a_hat * has_data)
        n_k   = torch.zeros(len(useed), device=device).scatter_add_(0, sinv, has_data)
        a_bar = sum_a / n_k.clamp_min(1.0)                          # (n_seeds,)

        # ── exact gradient coefficient, per row ──
        # L_inv is averaged over the |B| seeds in the batch, so its gradient carries
        #   2/(|B|*K*G) = 2/N : the same 1/N prefactor as the GRPO term, hence K and G
        # cancel. delta is added to the advantage and so inherits TRL's token-level
        # normalisation identically to A, which makes the two terms commensurable.
        #
        # Baseline-corrected estimator (see class docstring):
        #     grad a_hat_k = E[(r - a_hat_k) grad log pi]
        # and since A = (r - a_hat_k)/std_k this becomes
        #     delta = -2*lambda*(a_hat_k - a_bar)*std_k * A
        # a multiplicative rescaling: easy variants scaled down, hard ones scaled up,
        # with both correct and incorrect rollouts moving together and no sign flips.
        dev_pair = a_hat - a_bar[sinv]                              # a_hat_k - a_bar
        dev_row  = dev_pair[pinv]
        std_pair = torch.sqrt((a_hat * (1.0 - a_hat)).clamp_min(1e-8))
        std_row  = std_pair[pinv]

        # advantages must be gathered so delta lines up with the global keys
        adv = out["advantages"]
        adv_all = gather(adv.detach())                              # (N_total,)
        if adv_all.shape[0] != N:
            raise RuntimeError(f"grip: {adv_all.shape[0]} gathered advantages vs {N} rewards")

        delta = -2.0 * self.grip_lambda * dev_row * std_row * adv_all
        delta = torch.where(torch.isnan(r), torch.zeros_like(delta), delta)  # unscorable -> 0

        # ── slice to this process's rows, exactly as TRL slices advantages ──
        rank = self.accelerator.process_index
        sl = slice(rank * n_local, (rank + 1) * n_local)
        delta_local = delta[sl]
        if delta_local.shape != adv.shape:
            raise RuntimeError(f"grip: delta {tuple(delta_local.shape)} vs adv {tuple(adv.shape)}")
        out["advantages"] = adv + delta_local.to(adv.dtype)

        # ── one-time structural check: every seed has K variants, each with G rows ──
        if not self._grip_checked:
            bad_k = (n_k != K).sum().item()
            bad_g = int((torch.bincount(pinv, minlength=len(upair)) != G).sum().item())
            print(f"[grip] batch check: {len(useed)} seeds, {len(upair)} (seed,k) pairs, "
                  f"seeds with !=K variants: {bad_k}, pairs with !=G rows: {bad_g}", flush=True)
            if bad_k or bad_g:
                print("[grip] WARNING: block structure not intact -- check shuffle_dataset=False "
                      "and that generation batch is a multiple of K prompts", flush=True)
            self._grip_checked = True

        # ── diagnostics ──
        inv_per_seed = torch.zeros(len(useed), device=device).scatter_add_(0, sinv, (dev_pair ** 2) * has_data) / n_k.clamp_min(1)
        active = (inv_per_seed > 0).float().mean().item()
        ratio = (delta.abs().mean() / adv_all.abs().mean().clamp_min(1e-8)).item()
        mult_min = (1.0 - 2.0*self.grip_lambda*dev_pair*std_pair).min().item()
        m = self._metrics[mode]
        m["grip/inv_loss"].append(inv_per_seed.mean().item())
        m["grip/adv_ratio"].append(ratio)
        m["grip/frac_seeds_active"].append(active)
        m["grip/n_seeds_in_batch"].append(float(len(useed)))
        m["grip/lambda"].append(self.grip_lambda)
        m["grip/mult_min"].append(mult_min)   # <0 would mean a sign flip; should stay >0

        self._grip_rpf = None
        return out


class GRIPMetricCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if logs is None:
            return
        r, s = logs.get("reward"), logs.get("reward_std")
        if r is not None and s is not None and abs(r) > 1e-8:
            logs["grip/reward_cv"] = float(s) / float(r)


# DATA
def build_prompt(example):
    example["prompt"] = f"{example['question']}\n{MATH_INSTRUCTION}"
    example["seed_int"] = int(str(example["id"]).rsplit("_", 1)[-1])
    example["k"] = int(example["k"])
    return example


def block_order(ds, K, seed=42):
    """
    Arrange rows so every seed's K variants are contiguous (k ascending), with
    seeds shuffled. Combined with shuffle_dataset=False, the sampler then draws
    generation batches of 32 prompts = 8 complete seeds.
    """
    ids, ks = ds["id"], ds["k"]
    by_seed = {}
    for i, (s, k) in enumerate(zip(ids, ks)):
        by_seed.setdefault(s, {})[int(k)] = i
    complete = sorted(s for s, d in by_seed.items() if len(d) == K)
    dropped = len(by_seed) - len(complete)
    random.Random(seed).shuffle(complete)
    order = [by_seed[s][k] for s in complete for k in range(K)]
    print(f"block_order: {len(complete)} seeds x {K} = {len(order)} rows "
          f"(dropped {dropped} incomplete seeds)", flush=True)
    return ds.select(order)


# MAIN
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── wandb resume (write the id file BEFORE submitting so ranks don't race) ──
    RUN_ID_FILE = f"{OUTPUT_DIR}/wandb_run_id.txt"
    if os.path.exists(RUN_ID_FILE):
        wid = open(RUN_ID_FILE).read().strip()
        os.environ["WANDB_RESUME"] = "must"
    else:
        wid = "".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(8))
        with open(RUN_ID_FILE, "w") as f:
            f.write(wid)
        os.environ["WANDB_RESUME"] = "allow"
    os.environ["WANDB_RUN_ID"] = wid
    print(f"run={RUN_ID}  lambda={GRIP_LAMBDA}  lr={LR}  wandb={wid}", flush=True)

    # ── data: identical seed split to A1/A2 ──
    ds = load_dataset("json", data_files=DATA_FILE, split="train")
    ds = ds.map(build_prompt)

    all_ids = sorted(set(ds["id"]))
    rng = random.Random(42)
    rng.shuffle(all_ids)
    eval_ids  = set(all_ids[:N_EVAL_SEEDS])
    train_ids = set(all_ids[N_EVAL_SEEDS:])
    eval_ds  = ds.filter(lambda x: x["id"] in eval_ids)
    train_ds = ds.filter(lambda x: x["id"] in train_ids)      # A2/A3: all variants

    train_ds = block_order(train_ds, K_VARIANTS, seed=42)      # replaces .shuffle(seed=42)

    print(f"seeds: {len(all_ids)} (train {len(train_ids)} / eval {len(eval_ids)})", flush=True)
    print(f"rows:  train={len(train_ds)}  eval={len(eval_ds)}", flush=True)

    cfg = GRPOConfig(
        output_dir=OUTPUT_DIR,
        run_name=RUN_ID,
        model_init_kwargs={"dtype": "bfloat16"},
        seed=42, data_seed=42,

        # ── GRIP requires the block order to reach the batch intact ──
        shuffle_dataset=False,

        # ── loss (identical to A1/A2) ──
        loss_type="dapo",
        beta=0.0,
        epsilon=0.2,
        epsilon_high=0.28,
        num_iterations=1,
        scale_rewards="group",
        mask_truncated_completions=True,

        # ── generation (identical to A1/A2 and to A0 eval) ──
        temperature=0.6,
        top_p=0.95,
        max_completion_length=3072,
        num_generations=8,

        # ── batch: 2 x 64 x 2 GPUs = 256 completions = 32 prompts = 8 seeds ──
        per_device_train_batch_size=2,
        gradient_accumulation_steps=64,
        per_device_eval_batch_size=2,

        # ── optimisation (LR from the probe; identical to A1/A2 reruns) ──
        learning_rate=LR,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        weight_decay=0.1,
        max_grad_norm=1.0,

        # ── vLLM ──
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.12,
        vllm_tensor_parallel_size=1,
        vllm_max_model_length=4096,

        # ── length / checkpoints / eval ──
        max_steps=MAX_STEPS,
        save_steps=50,
        save_total_limit=4,
        eval_strategy="steps",
        eval_steps=50,

        # ── logging ──
        logging_steps=1,
        log_completions=False,
        report_to="wandb",

        remove_unused_columns=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # sanity: generation batch must be a multiple of K prompts
    gen_prompts = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps \
                  * int(os.environ.get("WORLD_SIZE", "1")) // cfg.num_generations
    if gen_prompts % K_VARIANTS != 0:
        raise ValueError(f"generation batch = {gen_prompts} prompts, not a multiple of K={K_VARIANTS}")

    trainer = GRIPTrainer(
        model=MODEL_NAME,
        args=cfg,
        reward_funcs=accuracy_reward,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=[GRIPMetricCallback()],
        grip_lambda=GRIP_LAMBDA,
        grip_k=K_VARIANTS,
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
    trainer.processing_class.save_pretrained(f"{OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()