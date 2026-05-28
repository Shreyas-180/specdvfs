"""
Run this ONCE before any experiments.
Creates the fixed prompt index files that all conditions share.
"""
import json
import random
from pathlib import Path
from datasets import load_dataset

SEED = 42
random.seed(SEED)

OUTPUT_DIR = Path(__file__).parent / "sampled_indices"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading GSM-8K...")
gsm8k = load_dataset("gsm8k", "main", split="test")

# Sample 256 prompts — fixed forever
all_questions = [item["question"] for item in gsm8k]
sampled = random.sample(all_questions, 256)

out_path = OUTPUT_DIR / "gsm8k_256_seed42.json"
with open(out_path, "w") as f:
    json.dump(sampled, f, indent=2)

print(f"Saved {len(sampled)} prompts to {out_path}")
print("First prompt preview:")
print(f"  {sampled[0][:80]}...")
