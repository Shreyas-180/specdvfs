import json
import random
import hashlib
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "sampled_indices"
SEED     = 42


def _md5(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}", file=sys.stderr)
    sys.exit(1)


def _check(condition: bool, msg: str) -> None:
    if not condition:
        _fail(msg)


def verify(
    filename: str,
    expected_n: int,
    subsampled: bool,
    has_task_id: bool = False,
    has_reference_summary: bool = False,
) -> str:
    """Run all integrity checks on one dataset file. Returns the MD5 checksum."""
    path = DATA_DIR / filename
    print(f"\n{filename}")

    _check(path.exists(), f"file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    samples = data["samples"]
    indices = data["indices"]

    _check(data["n_samples"] == expected_n,
           f"n_samples={data['n_samples']}, expected {expected_n}")
    _check(data["seed"] == SEED,
           f"seed={data['seed']}, expected {SEED}")
    _check(len(samples) == expected_n,
           f"samples list has {len(samples)} items, expected {expected_n}")
    _check(len(indices) == expected_n,
           f"indices list has {len(indices)} items, expected {expected_n}")

    # No empty texts — a silent HuggingFace API failure can produce empty strings
    # that would corrupt energy measurements without raising an obvious error.
    empty = [s["id"] for s in samples if not str(s.get("text", "")).strip()]
    _check(len(empty) == 0, f"{len(empty)} samples have empty text: ids={empty[:5]}")

    # Sequential IDs so samples can be referenced by position without ambiguity.
    for i, s in enumerate(samples):
        _check(s["id"] == i, f"sample at position {i} has id={s['id']}")

    _check(indices == sorted(indices), "indices not sorted")
    _check(len(set(indices)) == expected_n, "duplicate indices found")

    for i, s in enumerate(samples):
        _check(s["dataset_index"] == indices[i],
               f"sample {i}: dataset_index={s['dataset_index']} != indices[{i}]={indices[i]}")

    # Reproducibility: re-derive indices from scratch. If this fails, the file
    # was hand-edited or created with different sampling logic — experiments
    # using it would not be reproducible.
    if subsampled:
        total = data["total_dataset_size"]
        rng   = random.Random(SEED)
        expected_indices = sorted(rng.sample(range(total), expected_n))
        _check(indices == expected_indices,
               "indices are not reproducible from the stated seed and total_dataset_size")

    if has_task_id:
        missing = [s["id"] for s in samples if not s.get("task_id", "").strip()]
        _check(len(missing) == 0, f"{len(missing)} samples missing task_id")

    if has_reference_summary:
        missing = [s["id"] for s in samples if not s.get("reference_summary", "").strip()]
        _check(len(missing) == 0, f"{len(missing)} CNN-DM samples missing reference_summary")

    lengths = [len(s["text"]) for s in samples]
    checksum = _md5(path)

    print(f"  count        : {expected_n}")
    print(f"  reproducible : {'yes (subsampled)' if subsampled else 'yes (all samples taken)'}")
    print(f"  text len     : min={min(lengths)}  max={max(lengths)}  avg={sum(lengths)/len(lengths):.0f} chars")
    print(f"  MD5          : {checksum}")
    return checksum


def smoke_test_templates() -> None:
    """Verify that every prompt template substitutes correctly on a real sample."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "prepare_datasets", Path(__file__).parent / "prepare_datasets.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    files_and_prefixes = {
        "gsm8k_256.json":     "",
        "humaneval_164.json": "",
        "cnndm_256.json":     mod.CNN_DM_PREFIX,
    }

    print("\ntemplate smoke test")
    for filename, prefix in files_and_prefixes.items():
        with open(DATA_DIR / filename, encoding="utf-8") as f:
            data = json.load(f)
        raw = prefix + data["samples"][0]["text"]
        for key, tmpl in mod.PROMPT_TEMPLATES.items():
            rendered = tmpl.format(text=raw)
            _check("{text}" not in rendered,
                   f"{filename} / {key}: placeholder not substituted")
            _check(len(rendered) > 50,
                   f"{filename} / {key}: rendered output suspiciously short ({len(rendered)} chars)")
        print(f"  {filename}  ok")


if __name__ == "__main__":
    checksums = {}
    checksums["gsm8k_256.json"]     = verify("gsm8k_256.json",     256, subsampled=True)
    checksums["humaneval_164.json"] = verify("humaneval_164.json", 164, subsampled=False, has_task_id=True)
    checksums["cnndm_256.json"]     = verify("cnndm_256.json",     256, subsampled=True,  has_reference_summary=True)
    smoke_test_templates()

    print("\n" + "=" * 52)
    print("ALL CHECKS PASSED")
    print("=" * 52)
    for fname, cs in checksums.items():
        print(f"  {cs}  {fname}")

# ── USAGE ──────────────────────────────────────────────────────────────────────

# Also run before every batch of experiments to confirm nothing changed.
# If any check fails, do not run experiments — investigate the cause first.
# Record the printed MD5 checksums in your lab notebook on first run.
# A checksum mismatch on a later run means the JSON file was modified.