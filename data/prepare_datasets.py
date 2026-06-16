import json
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR / "sampled_indices"

SEED    = 42
N_SMALL = 256   # sample size for GSM-8K and CNN-DM

# CNN-DM 3.0.0 test split is consistently 11490 on HuggingFace Hub.
# If the assertion below fails, the Hub dataset changed — update this constant
# and re-verify that sampling with seed=42 still produces the same indices.
EXPECTED_CNNDM_TEST_SIZE = 11490

# GSM-8K 'main' test split size; used for the reproducibility check in verify_datasets.py.
EXPECTED_GSM8K_TEST_SIZE = 1319

# Prompt templates keyed by model family.
# Applied at inference time — not baked into the JSON so the same dataset file
# works across all three model families without re-preparation.
# {text} is substituted with the raw sample text.
PROMPT_TEMPLATES = {
    # lmsys/vicuna-13b-v1.3  /  double7/vicuna-68m
    "vicuna": (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
        "\nUSER: {text}\nASSISTANT:"
    ),
    # meta-llama/Llama-3.1-8B-Instruct  /  Llama-3.2-1B-Instruct
    "llama3": (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
    # Qwen/Qwen3-8B  /  Qwen/Qwen3-0.6B
    "qwen3": "<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n",
}

# Prefix prepended to CNN-DM articles before substituting into templates.
CNN_DM_PREFIX = "Summarize the following news article in 2-3 sentences:\n\n"


def _sample_indices(total_size: int, n: int, seed: int) -> list:
    """Return a sorted list of n unique indices sampled from range(total_size).

    Sorting keeps the JSON human-readable and makes git diffs cleaner.
    The result is fully determined by (total_size, n, seed) — no external state
    can affect it, so any machine running this function with the same arguments
    produces an identical list.
    """
    rng = random.Random(seed)
    return sorted(rng.sample(range(total_size), n))


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  saved  {path.name}  ({payload['n_samples']} samples)")


def prepare_gsm8k() -> Path:
    from datasets import load_dataset
    print("gsm8k ...")
    # Newer huggingface_hub versions require 'namespace/name' format.
    # If 'gsm8k' (short name) fails on a fresh machine, try 'openai-community/gsm8k'.
    ds      = load_dataset("gsm8k", "main", split="test")
    total   = len(ds)
    indices = _sample_indices(total, N_SMALL, SEED)
    samples = [{"id": i, "dataset_index": idx, "text": ds[idx]["question"]}
               for i, idx in enumerate(indices)]
    payload = {
        "dataset": "gsm8k", "config": "main", "split": "test",
        "seed": SEED, "n_samples": len(samples),
        "total_dataset_size": total, "indices": indices, "samples": samples,
    }
    path = DATA_DIR / "gsm8k_256.json"
    _save(path, payload)
    return path


def prepare_humaneval() -> Path:
    from datasets import load_dataset
    print("human_eval ...")
    # 'openai_humaneval' (no namespace) was deprecated in newer huggingface_hub versions.
    # The canonical identifier is 'openai/openai_humaneval'.
    ds    = load_dataset("openai/openai_humaneval", split="test")
    total = len(ds)
    # All 164 samples are taken in natural order — no random subsampling.
    samples = [
        {"id": i, "dataset_index": i,
         "task_id": ds[i]["task_id"],
         "text": ds[i]["prompt"],          # function signature + docstring
         "entry_point": ds[i]["entry_point"]}
        for i in range(total)
    ]
    payload = {
        "dataset": "openai/openai_humaneval", "split": "test",
        "seed": SEED, "n_samples": len(samples),
        "total_dataset_size": total,
        "indices": list(range(total)),
        "samples": samples,
    }
    path = DATA_DIR / "humaneval_164.json"
    _save(path, payload)
    return path


def prepare_cnndm() -> Path:
    from datasets import load_dataset
    print("cnn_dailymail ...")
    # 'cnn_dailymail' (unprefixed) was deprecated in newer huggingface_hub versions.
    ds    = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")
    total = len(ds)
    assert total == EXPECTED_CNNDM_TEST_SIZE, (
        f"cnn_dailymail 3.0.0 test split: expected {EXPECTED_CNNDM_TEST_SIZE} rows, "
        f"got {total}. Update EXPECTED_CNNDM_TEST_SIZE if the Hub dataset changed."
    )
    indices = _sample_indices(total, N_SMALL, SEED)
    samples = [
        {"id": i, "dataset_index": idx,
         "text": ds[idx]["article"],
         # Reference kept for optional output-quality metrics; not used for input formatting.
         "reference_summary": ds[idx]["highlights"]}
        for i, idx in enumerate(indices)
    ]
    payload = {
        "dataset": "cnn_dailymail", "config": "3.0.0", "split": "test",
        "seed": SEED, "n_samples": len(samples),
        "total_dataset_size": total, "indices": indices, "samples": samples,
    }
    path = DATA_DIR / "cnndm_256.json"
    _save(path, payload)
    return path


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    prepare_gsm8k()
    prepare_humaneval()
    prepare_cnndm()
    print("done.")

# ── USAGE ──────────────────────────────────────────────────────────────────────
# Run once, before any experiment:
#   conda activate specdvfs
#   python data/prepare_datasets.py
#
# After this, run:
#   python data/verify_datasets.py
# and record the printed MD5 checksums in your lab notebook.
# Never re-run this script mid-project — it would regenerate the same files,
# but any accidental change to sampling logic would break reproducibility.