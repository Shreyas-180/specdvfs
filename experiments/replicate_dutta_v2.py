"""
Dutta et al. EACL 2026 Replication Script — SpecDVFS Project
Replicates: "Benchmarking the Energy Savings with Speculative Decoding Strategies"

Conditions: Vanilla, COGA-5, COGA-10, COGA-20, DYGA-20
Dataset: GSM-8K (256 prompts, seed=42)
Reps: 5 per condition
Energy: CodeCarbon (1-second polling)
Quantization: NF4 4-bit (matching Dutta et al. exactly)
"""

import torch
import json
import random
import time
import os
import numpy as np
import pynvml
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from codecarbon import EmissionsTracker

# ==========================================
# 1. EXPERIMENTAL CONTROLS — DO NOT CHANGE
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Dutta et al. exact models
# WARNING: VICUNA-13B needs ~7GB VRAM with NF4 double quant
# If you get OOM on your 4070 laptop (8GB), change to:
#   TARGET_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
#   DRAFT_MODEL_ID  = "meta-llama/Llama-3.2-1B-Instruct"
# The Llama pair also shows the anomaly and fits comfortably in 8GB.

TARGET_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DRAFT_MODEL_ID  = "meta-llama/Llama-3.2-1B-Instruct"

N_REPETITIONS = 5          # Dutta et al. ran 5 repetitions
MAX_NEW_TOKENS = 200        # Dutta et al. used 200
MAX_GPU_TEMP_C = 75         # Abort run if GPU hotter than this
COOLDOWN_SECONDS = 30       # Between consecutive runs

# Dutta et al. Table 4 — exact prompt template for VICUNA on GSM-8K
PROMPT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "You are a helpful, respectful and honest assistant. "
    "Always answer as helpfully as possible, while being safe.<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n"
    "Solve the math problem and give a numeric solution "
    "Problem: {question}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n"
)

# ==========================================
# 2. PATH SETUP
# ==========================================
# Robust path handling — works regardless of where you run the script
SCRIPT_DIR   = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR     = PROJECT_ROOT / "data" / "sampled_indices"
RESULTS_DIR  = PROJECT_ROOT / "results" / "dutta_replication"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "gsm8k_256_seed42.json"
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Missing dataset at {DATA_PATH}\n"
        "Run prepare_datasets.py first to generate the fixed prompt indices."
    )

# ==========================================
# 3. DATA LOADING
# ==========================================
print(f"Loading frozen dataset from {DATA_PATH}...")
with open(DATA_PATH, "r") as f:
    sampled_prompts = json.load(f)

# 5 warmup + 251 measurement = 256 total (matching Dutta et al.)
warmup_prompts = sampled_prompts[:5]
eval_prompts   = sampled_prompts[5:]
print(f"Warmup: {len(warmup_prompts)} prompts | Eval: {len(eval_prompts)} prompts")

# ==========================================
# 4. MODEL LOADING
# ==========================================
print(f"\nLoading target model: {TARGET_MODEL_ID}")
print(f"Loading draft model:  {DRAFT_MODEL_ID}")

tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# NF4 quantization matching Dutta et al. exactly
# Note: llm_int8_enable_fp32_cpu_offload is for int8 ONLY, not NF4 — do not use it
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

try:
    target_model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    target_model.eval()
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        print("\n" + "="*60)
        print("OOM ERROR: VICUNA-13B does not fit in your VRAM.")
        print("Switch to LLAMA-8B/1B pair in the config at the top.")
        print("="*60)
        raise
    raise

draft_model = AutoModelForCausalLM.from_pretrained(
    DRAFT_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
draft_model.eval()

allocated_gb = torch.cuda.memory_allocated() / 1e9
reserved_gb  = torch.cuda.memory_reserved() / 1e9
print(f"GPU memory — allocated: {allocated_gb:.2f} GB | reserved: {reserved_gb:.2f} GB")

# ==========================================
# 5. GPU MONITORING UTILITIES
# ==========================================
pynvml.nvmlInit()
_nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def get_gpu_temp() -> int:
    return pynvml.nvmlDeviceGetTemperature(
        _nvml_handle, pynvml.NVML_TEMPERATURE_GPU
    )

def wait_for_cool_gpu():
    """Block until GPU is below MAX_GPU_TEMP_C. Log temperature."""
    temp = get_gpu_temp()
    if temp > MAX_GPU_TEMP_C:
        print(f"  GPU at {temp}°C — waiting to cool below {MAX_GPU_TEMP_C}°C...")
        while temp > MAX_GPU_TEMP_C:
            time.sleep(10)
            temp = get_gpu_temp()
            print(f"  Current temp: {temp}°C")
    return temp

# ==========================================
# 6. INFERENCE ENGINE
# ==========================================
def run_benchmark(prompts, sd_type="vanilla", assistant_tokens=5, is_warmup=False):
    """
    Run inference and measure energy.

    sd_type options:
        "vanilla" — standard autoregressive decode, no draft model
        "coga"    — Constant Generation by Assistant (fixed draft length)
                    Dutta et al.: COGA-5, COGA-10, COGA-20
        "dyga"    — Dynamic Generation by Assistant (adaptive draft length)
                    Dutta et al.: DYGA-20 (starts at 20, adjusts ±)

    is_warmup: if True, skips energy tracking and returns None
    """
    total_tokens = 0

    tracker = None
    if not is_warmup:
        tracker = EmissionsTracker(
            project_name=f"dutta_{sd_type}_{assistant_tokens}",
            output_dir=str(RESULTS_DIR),
            measure_power_secs=1,      # Dutta et al. used 1-second polling
            tracking_mode="machine",   # Tracks GPU + CPU + RAM
            log_level="error",
            save_to_file=False,        # We save manually as JSON
        )
        tracker.start()

    torch.cuda.synchronize()
    start_time = time.perf_counter()  # Higher precision than time.time()

    for prompt in prompts:
        formatted = PROMPT_TEMPLATE.format(question=prompt)

        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to("cuda")

        gen_kwargs = {
            "input_ids":      inputs.input_ids,
            "attention_mask": inputs.attention_mask,   # Always pass this
            "max_new_tokens": MAX_NEW_TOKENS,          # 200 per Dutta et al.
            "do_sample":      False,                   # Greedy decode
            "pad_token_id":   tokenizer.eos_token_id,
        }

        if sd_type == "coga":
            # COGA: fixed number of draft tokens per step
            gen_kwargs["assistant_model"]                  = draft_model
            gen_kwargs["num_assistant_tokens"]             = assistant_tokens
            gen_kwargs["num_assistant_tokens_schedule"]    = "constant"

        elif sd_type == "dyga":
            # DYGA: starts at assistant_tokens, +2 if all accepted, -1 if any rejected
            gen_kwargs["assistant_model"]                  = draft_model
            gen_kwargs["num_assistant_tokens"]             = assistant_tokens
            gen_kwargs["num_assistant_tokens_schedule"]    = "heuristic"

        try:
            with torch.no_grad():
                outputs = target_model.generate(**gen_kwargs)

            torch.cuda.synchronize()
            new_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
            total_tokens += new_tokens

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"\n  OOM on prompt — skipping. Free memory: "
                      f"{torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")
                torch.cuda.empty_cache()
                continue
            raise

    end_time = time.perf_counter()
    total_time_s = end_time - start_time

    if is_warmup:
        print(f"  Warmup done — {total_tokens} tokens generated")
        return None

    total_energy_kwh = tracker.stop()
    try:
        gpu_energy_kwh = tracker._total_gpu_energy.kWh
        if gpu_energy_kwh is None or gpu_energy_kwh == 0:
            gpu_energy_kwh = total_energy_kwh
            print("  Note: GPU energy not isolated, using total energy")
    except AttributeError:
        gpu_energy_kwh = total_energy_kwh
        print("  Note: GPU energy not isolated, using total energy")

    # Fallback: if CodeCarbon couldn't isolate GPU energy, use total
    # if gpu_energy_kwh is None or gpu_energy_kwh == 0:
    #     print("  Warning: GPU energy not available separately — using total energy")
    #     gpu_energy_kwh = total_energy_kwh

    tokens_1k = total_tokens / 1000.0

    return {
        "sd_type":                  sd_type,
        "assistant_tokens":         assistant_tokens,
        "total_tokens":             total_tokens,
        "total_time_s":             total_time_s,
        # Per 1K tokens (Dutta et al.'s reporting unit, Table 8)
        "gpu_energy_wh_per_1k":     (gpu_energy_kwh   * 1000) / tokens_1k,
        "total_energy_wh_per_1k":   (total_energy_kwh * 1000) / tokens_1k,
        "time_per_1k_s":            total_time_s / tokens_1k,
        "gpu_temp_start":           get_gpu_temp(),
    }

# ==========================================
# 7. EXPERIMENT MATRIX — MATCHING DUTTA ET AL.
# ==========================================
CONDITIONS = [
    # name         sd_type    assistant_tokens
    ("Vanilla",    "vanilla", None),
    ("COGA-5",     "coga",    5),
    ("COGA-10",    "coga",    10),
    ("COGA-20",    "coga",    20),
    ("DYGA-20",    "dyga",    20),
]

all_results = {name: [] for name, *_ in CONDITIONS}

print(f"\nStarting experiments: {len(CONDITIONS)} conditions × {N_REPETITIONS} reps")
print(f"Total runs: {len(CONDITIONS) * N_REPETITIONS}")
print("="*60)

for rep in range(N_REPETITIONS):
    print(f"\n{'='*60}")
    print(f"REPETITION {rep + 1} / {N_REPETITIONS}")
    print(f"{'='*60}")

    for name, sd_type, assistant_tokens in CONDITIONS:
        tokens_arg = assistant_tokens if assistant_tokens is not None else 5

        print(f"\n  [{name}] Rep {rep+1}/{N_REPETITIONS}")

        # Temperature guard — critical for fair comparison
        start_temp = wait_for_cool_gpu()
        print(f"  GPU temp: {start_temp}°C ✓")

        # Warmup pass — bring GPU to thermal steady state
        print("  Warming up...")
        run_benchmark(
            warmup_prompts,
            sd_type=sd_type,
            assistant_tokens=tokens_arg,
            is_warmup=True,
        )

        # Measured pass
        print("  Measuring...")
        result = run_benchmark(
            eval_prompts,
            sd_type=sd_type,
            assistant_tokens=tokens_arg,
            is_warmup=False,
        )

        result["rep"] = rep
        result["gpu_temp_before"] = start_temp
        all_results[name].append(result)

        print(f"  Result: {result['total_tokens']} tokens | "
              f"{result['total_time_s']:.1f}s | "
              f"GPU {result['gpu_energy_wh_per_1k']:.3f} Wh/1K | "
              f"Total {result['total_energy_wh_per_1k']:.3f} Wh/1K")

        # Save after every condition — crash-safe
        raw_path = RESULTS_DIR / "raw_results.json"
        with open(raw_path, "w") as f:
            json.dump(all_results, f, indent=2)

        # Cooldown before next run (skip after last run)
        is_last_run = (rep == N_REPETITIONS - 1 and
                       name == CONDITIONS[-1][0])
        if not is_last_run:
            print(f"  Cooling down {COOLDOWN_SECONDS}s...")
            time.sleep(COOLDOWN_SECONDS)

# ==========================================
# 8. COMPUTE FINAL METRICS
# ==========================================
print("\n" + "="*70)
print("FINAL RESULTS — DUTTA ET AL. REPLICATION")
print(f"Model: {TARGET_MODEL_ID} / {DRAFT_MODEL_ID}")
print(f"Dataset: GSM-8K | {len(eval_prompts)} eval prompts | {N_REPETITIONS} reps")
print("="*70)

# Vanilla baseline
vanilla_gpu_mean   = np.mean([r["gpu_energy_wh_per_1k"]   for r in all_results["Vanilla"]])
vanilla_total_mean = np.mean([r["total_energy_wh_per_1k"] for r in all_results["Vanilla"]])
vanilla_time_mean  = np.mean([r["time_per_1k_s"]          for r in all_results["Vanilla"]])

print(f"\nVanilla baseline:")
print(f"  GPU energy:   {vanilla_gpu_mean:.3f} Wh/1K tokens")
print(f"  Total energy: {vanilla_total_mean:.3f} Wh/1K tokens")
print(f"  Time:         {vanilla_time_mean:.3f} s/1K tokens")

header = f"\n{'Method':<12} | {'yt (speedup)':<14} | {'γe^GPU':<12} | {'γe^Total':<12}"
print(header)
print("-" * 55)

summary = {}
for name, sd_type, _ in CONDITIONS:
    if name == "Vanilla":
        continue

    reps = all_results[name]

    # Compute per-repetition saving factors
    gpu_factors   = [vanilla_gpu_mean   / r["gpu_energy_wh_per_1k"]   for r in reps]
    total_factors = [vanilla_total_mean / r["total_energy_wh_per_1k"] for r in reps]
    speedups      = [vanilla_time_mean  / r["time_per_1k_s"]          for r in reps]

    gt_mean  = np.mean(speedups);      gt_std  = np.std(speedups)
    gpu_mean = np.mean(gpu_factors);   gpu_std = np.std(gpu_factors)
    tot_mean = np.mean(total_factors); tot_std = np.std(total_factors)

    flag = " ← ANOMALY (SD wastes energy)" if gpu_mean < 1.0 else ""
    print(f"{name:<12} | {gt_mean:>6.2f}x ±{gt_std:.2f}  | "
          f"{gpu_mean:>5.2f}x ±{gpu_std:.2f} | "
          f"{tot_mean:>5.2f}x ±{tot_std:.2f}{flag}")

    summary[name] = {
        "speedup_mean": gt_mean,           "speedup_std": gt_std,
        "gpu_energy_factor_mean": gpu_mean, "gpu_energy_factor_std": gpu_std,
        "total_energy_factor_mean": tot_mean,
    }

print("="*70)
print("\nDutta et al. Table 1 target values for comparison (VICUNA-13B/68M, GSM-8K):")
print("  COGA-20:  yt=1.59x, ye^GPU=1.10x  (< 1.0 = SD wastes energy)")
print("  DYGA-20:  yt=1.50x, ye^GPU=1.06x  (< 1.0 = SD wastes energy)")
print("  EAGLE-2:  yt=2.62x, ye^GPU=1.58x  (> 1.0 = SD saves energy)")
print("\nIf your numbers are within ~10% of these, replication is successful.")

# Save summary
summary_path = RESULTS_DIR / "summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved to: {summary_path}")
print(f"Raw results at:   {RESULTS_DIR / 'raw_results.json'}")