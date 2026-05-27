import json
import random
import os
from datasets import load_dataset

def main():
    print("Loading GSM-8K...")
    # Downloads automatically if not cached
    dataset = load_dataset("gsm8k", "main", split="test")
    
    # STRICT CONTROL: Seed 42
    random.seed(42)
    
    all_indices = list(range(len(dataset)))
    sampled_indices = random.sample(all_indices, 256)
    
    sampled_prompts = [dataset[i]["question"] for i in sampled_indices]
    
    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), "sampled_indices")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "gsm8k_256_seed42.json")
    
    with open(output_path, "w") as f:
        json.dump(sampled_prompts, f, indent=4)
        
    print(f"Success! Saved 256 frozen prompts to {output_path}")

if __name__ == "__main__":
    main()