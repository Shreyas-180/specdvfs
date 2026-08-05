"""Per-GPU hardware profiles so the whole codebase runs on either the RTX 3090 or the
RTX 4090 (the cloud instance may turn out to be either). This is the SINGLE SOURCE OF
TRUTH for the two hardware-specific things the study depends on:

  * the clock anchors  f_high / f_low   (used by experiments/run_experiment.py and the
    profiling drivers to lock the verify / draft phases), and
  * the roofline ceilings  peak bf16 TFLOP/s  and  peak DRAM bandwidth  (used by the
    roofline analysis to place the ridge point).

Both cards have 24 GB, so model-memory fit is identical — nothing else changes.

"""
from __future__ import annotations

GPU_PROFILES = {
    "rtx_3090": {
        "aliases": ["3090"],
        "f_high": 1935, "f_low": 735,
        "peak_tflops_bf16": 71.0, "peak_bw_gbs": 936.0,
    },
    "rtx_4090": {
        "aliases": ["4090"],
        "f_high": 2520, "f_low": 735,
        "peak_tflops_bf16": 165.2, "peak_bw_gbs": 1008.0,
    },
}

# Used (with a printed warning) when the GPU can't be identified, so analysis still runs.
DEFAULT_GPU = "rtx_3090"

_MOBILE_MARKERS = ("laptop", "mobile", "max-q")


def profile_for_name(name):
    """(key, profile) for a GPU name string via alias substring match; (None, None) if no match
    or if the name is a mobile part. Pure string logic — needs no GPU/NVML, so it is safe to
    call on the laptop (e.g. from the ncu CSV's Device column)."""
    if not name:
        return None, None
    low = name.lower()
    if any(m in low for m in _MOBILE_MARKERS):
        return None, None
    for key, prof in GPU_PROFILES.items():
        if any(a in low for a in prof["aliases"]):
            return key, prof
    return None, None


def detect_profile_via_nvml():
    """(key, profile, name) from NVML on a machine WITH a GPU. Returns (None, None, name) if the
    GPU is unrecognized, and (None, None, None) if there is no GPU / NVML. Never raises."""
    name = None
    try:
        import pynvml as N
        N.nvmlInit()
        h = N.nvmlDeviceGetHandleByIndex(0)
        raw = N.nvmlDeviceGetName(h)
        name = raw.decode() if isinstance(raw, bytes) else raw
        N.nvmlShutdown()
    except Exception:
        return None, None, name
    key, prof = profile_for_name(name)
    return key, prof, name


def peaks_for(key_or_name, default=DEFAULT_GPU):
    """(peak_tflops_bf16, peak_bw_gbs) for a profile key ('rtx_3090'), a GPU name, or a short
    alias ('3090'/'4090'); falls back to `default` if nothing matches."""
    if key_or_name in GPU_PROFILES:
        p = GPU_PROFILES[key_or_name]
    else:
        _, p = profile_for_name(key_or_name)
        if p is None:
            p = GPU_PROFILES.get(f"rtx_{key_or_name}", GPU_PROFILES[default])
    return p["peak_tflops_bf16"], p["peak_bw_gbs"]