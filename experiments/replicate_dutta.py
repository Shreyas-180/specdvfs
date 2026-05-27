import torch
import json
import random
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from codecarbon import EmissionsTracker

# ==========================================
# 1. SETUP & STRICT CONTROLS
# ==========================================
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

TARGET_MODEL_ID = "lmsys/vicuna-13b-v1.3"
DRAFT_MODEL_ID = "double7/vicuna-68m"

VICUNA_PROMPT_TEMPLATE = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user’s questions. "
    "USER: Solve the math problem and give a numeric solution Problem: {question} ASSISTANT:"
)

# ==========================================
# 2. LOAD FROZEN DATASET
# ==========================================
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sampled_indices", "gsm8k_256_seed42.json")
print(f"Loading frozen dataset from {data_path}...")
with open(data_path, "r") as f:
    sampled_prompts = json.load(f)

# 5 for warmup, 251 for actual measurement
warmup_prompts = sampled_prompts[:5]
eval_prompts = sampled_prompts[5:]

# ==========================================
# 3. MODEL LOADING (NF4 Quantization)
# ==========================================
print("Loading Models onto GPU...")
tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

target_model = AutoModelForCausalLM.from_pretrained(
    TARGET_MODEL_ID, quantization_config=quant_config, device_map="cuda"
)

draft_model = AutoModelForCausalLM.from_pretrained(
    DRAFT_MODEL_ID, torch_dtype=torch.float16, device_map="cuda"
)

# ==========================================
# 4. INFERENCE FUNCTION
# ==========================================
def run_inference(prompts, use_sd=False, track_energy=False):
    total_new_tokens = 0
    
    # Save CodeCarbon results into the results/ folder
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    tracker = EmissionsTracker(
        measure_power_secs=1.0, 
        project_name="SpecDVFS", 
        log_level="error",
        output_dir=results_dir
    )
    
    if track_energy:
        torch.cuda.synchronize()
        tracker.start()
        start_time = time.time()

    for prompt in prompts:
        formatted_prompt = VICUNA_PROMPT_TEMPLATE.format(question=prompt)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
        input_length = inputs.input_ids.shape[1]
        
        gen_kwargs = {
            "input_ids": inputs.input_ids,
            "max_new_tokens": 128,
            "do_sample": False,
        }
        
        if use_sd:
            gen_kwargs["assistant_model"] = draft_model

        outputs = target_model.generate(**gen_kwargs)
        torch.cuda.synchronize() 
        
        total_new_tokens += (outputs.shape[1] - input_length)

    if track_energy:
        torch.cuda.synchronize()
        emissions = tracker.stop()
        end_time = time.time()
        
        energy_wh = tracker.final_emissions_data.energy_consumed * 1000
        total_time = end_time - start_time
        return energy_wh, total_time, total_new_tokens
    return None, None, None

# ==========================================
# 5. EXPERIMENT EXECUTION
# ==========================================
print("\n--- Phase A: Vanilla Decoding ---")
print("Running 5 warmup prompts...")
run_inference(warmup_prompts, use_sd=False, track_energy=False)

print("Running Vanilla Evaluation...")
vanilla_energy, vanilla_time, vanilla_tokens = run_inference(eval_prompts, use_sd=False, track_energy=True)
vanilla_energy_per_1k = vanilla_energy / (vanilla_tokens / 1000)

print("\n--- Phase B: Speculative Decoding ---")
print("Running 5 warmup prompts...")
run_inference(warmup_prompts, use_sd=True, track_energy=False)

print("Running SD Evaluation...")
sd_energy, sd_time, sd_tokens = run_inference(eval_prompts, use_sd=True, track_energy=True)
sd_energy_per_1k = sd_energy / (sd_tokens / 1000)

# ==========================================
# 6. RESULTS
# ==========================================
gamma_e = vanilla_energy / sd_energy
gamma_t = vanilla_time / sd_time

print("\n" + "="*50)
print("RESULTS: DUTTA ET AL. REPLICATION")
print("="*50)
print(f"Vanilla Time:       {vanilla_time:.2f} s")
print(f"SD Time:            {sd_time:.2f} s")
print(f"Speedup (γ_t):      {gamma_t:.2f}x")
print("-" * 50)
print(f"Vanilla Energy:     {vanilla_energy_per_1k:.4f} Wh per 1K tokens")
print(f"SD Energy:          {sd_energy_per_1k:.4f} Wh per 1K tokens")
print(f"Energy Saving (γ_e): {gamma_e:.4f}x")
print("="*50)