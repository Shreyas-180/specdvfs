"""Server GPU readiness check for SpecDVFS.

Runs every hardware/software check the project depends on, in increasing
order of how likely each is to fail, and prints a final GO / NO-GO verdict.

The decisive checks are not "does the call succeed" but "did the GPU
actually do what was asked" — clock locking and power telemetry can both
report success while silently doing nothing on cloud VMs and partitioned
HPC GPUs.

Exit code 0 = all critical checks passed (safe to run experiments).
Exit code 1 = at least one critical check failed (do not commit to this machine).
"""

import sys
import time
import subprocess


# ── result tracking ───────────────────────────────────────────────────────────

CRITICAL_FAILS = []
WARNINGS = []


def critical(name: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}]  {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        CRITICAL_FAILS.append(name)


def warn(name: str, ok: bool, detail: str = "") -> None:
    mark = "ok" if ok else "WARN"
    print(f"  [{mark}]  {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        WARNINGS.append(name)


def section(title: str) -> None:
    print(f"\n{'='*64}\n{title}\n{'='*64}")


# ── 1. basic CUDA / PyTorch ──────────────────────────────────────────────────

def check_torch():
    section("1. PyTorch + CUDA")
    try:
        import torch
    except ImportError:
        critical("torch importable", False, "PyTorch not installed")
        return None

    cuda_ok = torch.cuda.is_available()
    critical("torch.cuda.is_available()", cuda_ok)
    if not cuda_ok:
        return None

    name = torch.cuda.get_device_properties(0).name
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"         device: {name}")
    print(f"         VRAM:   {vram_gb:.1f} GB")

    # Need >= 24GB for VICUNA-13B NF4 + draft + KV cache + vLLM overhead.
    critical("VRAM >= 24GB", vram_gb >= 23.5, f"{vram_gb:.1f} GB")

    # Laptop GPUs and MIG slices flagged here.
    lowered = name.lower()
    is_laptop = any(t in lowered for t in ("laptop", "mobile", "max-q"))
    warn("not a laptop GPU", not is_laptop, name if is_laptop else "")
    if "mig" in lowered:
        critical("not a MIG partition", False,
                 "MIG slices cannot control clocks — request a full GPU")

    return torch


# ── 2. NVML available and not running under MIG ──────────────────────────────

def check_nvml_basic():
    section("2. NVML availability")
    try:
        import pynvml
    except ImportError:
        critical("pynvml importable", False, "install nvidia-ml-py")
        return None

    try:
        pynvml.nvmlInit()
    except Exception as e:
        critical("nvmlInit()", False, str(e)[:60])
        return None
    critical("nvmlInit()", True)

    h = pynvml.nvmlDeviceGetHandleByIndex(0)

    # MIG mode check — clock control is unavailable on MIG-partitioned GPUs.
    try:
        mig_mode, _ = pynvml.nvmlDeviceGetMigMode(h)
        mig_on = (mig_mode == pynvml.NVML_DEVICE_MIG_ENABLE)
        critical("MIG mode disabled", not mig_on,
                 "MIG is ON — clock locking will not work" if mig_on else "")
    except pynvml.NVMLError:
        warn("MIG mode query", True, "not supported on this GPU (fine)")

    return pynvml, h


# ── 3. persistence mode ───────────────────────────────────────────────────────

def check_persistence(pynvml, h):
    section("3. Persistence mode")
    try:
        mode = pynvml.nvmlDeviceGetPersistenceMode(h)
        on = (mode == pynvml.NVML_FEATURE_ENABLED)
        warn("persistence mode enabled", on,
             "run: sudo nvidia-smi -pm 1" if not on else "")
    except pynvml.NVMLError as e:
        warn("persistence mode query", False, str(e)[:50])


# ── 4. supported clock levels ─────────────────────────────────────────────────

def check_clock_levels(pynvml, h):
    section("4. Supported clock levels")
    try:
        mem_clocks = pynvml.nvmlDeviceGetSupportedMemoryClocks(h)
        gfx_clocks = sorted(pynvml.nvmlDeviceGetSupportedGraphicsClocks(h, mem_clocks[0]))
    except pynvml.NVMLError as e:
        critical("query supported clocks", False, str(e)[:50])
        return None

    f_high = gfx_clocks[-1]
    f_low_target = int(f_high * 0.35)
    # nearest actual level to the 35% target
    f_low = min(gfx_clocks, key=lambda c: abs(c - f_low_target))

    critical("multiple clock levels exist", len(gfx_clocks) >= 3,
             f"{len(gfx_clocks)} levels")
    print(f"         f_high (max):     {f_high} MHz")
    print(f"         f_low  (~35%):    {f_low} MHz")
    print(f"         range:            {gfx_clocks[0]}–{gfx_clocks[-1]} MHz")
    print(f"         >>> put these into controller/core.py: f_high={f_high}, f_low={f_low}")
    return f_high, f_low, gfx_clocks


# ── 5. THE DECISIVE CHECK: clock lock actually changes the clock ─────────────

def check_clock_lock_effective(pynvml, h, f_high, f_low):
    section("5. Clock locking is EFFECTIVE (read-back verified)")

    def read_clock():
        return pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)

    # Lock to f_low, read back, confirm it actually moved near f_low.
    try:
        pynvml.nvmlDeviceSetGpuLockedClocks(h, f_low, f_low)
    except pynvml.NVMLError as e:
        critical("nvmlDeviceSetGpuLockedClocks(low)", False, str(e)[:50])
        critical("clock lock is EFFECTIVE", False, "could not set lock")
        return

    time.sleep(0.5)
    sm_low = read_clock()
    # Allow a tolerance — driver may snap to nearest supported level.
    low_ok = abs(sm_low - f_low) <= max(60, f_low * 0.10)
    critical("clock lock takes effect (low)", low_ok,
             f"requested {f_low}, read {sm_low} MHz")

    # Lock to f_high, read back.
    try:
        pynvml.nvmlDeviceSetGpuLockedClocks(h, f_high, f_high)
        time.sleep(0.5)
        sm_high = read_clock()
        high_ok = abs(sm_high - f_high) <= max(60, f_high * 0.10)
        critical("clock lock takes effect (high)", high_ok,
                 f"requested {f_high}, read {sm_high} MHz")
        # The two must actually differ — otherwise the lock is a no-op.
        critical("low and high clocks differ", abs(sm_high - sm_low) > 100,
                 f"low={sm_low}, high={sm_high} MHz")
    except pynvml.NVMLError as e:
        critical("nvmlDeviceSetGpuLockedClocks(high)", False, str(e)[:50])

    # reset
    try:
        pynvml.nvmlDeviceResetGpuLockedClocks(h)
    except pynvml.NVMLError:
        WARNINGS.append("could not reset clocks (run nvmlDeviceResetGpuLockedClocks manually)")


# ── 6. clock lock SURVIVES load (no thermal/power override) ──────────────────

def check_lock_survives_load(pynvml, h, torch, f_low):
    section("6. Clock lock survives sustained GPU load")
    if torch is None:
        warn("load test", False, "torch unavailable, skipped")
        return

    try:
        pynvml.nvmlDeviceSetGpuLockedClocks(h, f_low, f_low)
    except pynvml.NVMLError as e:
        critical("lock for load test", False, str(e)[:50])
        return

    # Generate real sustained load: large matmuls in a loop for ~20s.
    dev = torch.device("cuda:0")
    a = torch.randn(8192, 8192, device=dev)
    b = torch.randn(8192, 8192, device=dev)

    clocks_under_load = []
    start = time.time()
    while time.time() - start < 20:
        for _ in range(20):
            a = a @ b
            a = a * 0.0001 + 0.1   # keep values bounded
        torch.cuda.synchronize()
        clocks_under_load.append(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))

    max_drift = max(clocks_under_load) - min(clocks_under_load)
    mean_clock = sum(clocks_under_load) / len(clocks_under_load)
    stayed = abs(mean_clock - f_low) <= max(80, f_low * 0.15) and max_drift <= 100

    critical("clock stays locked under load", stayed,
             f"target {f_low}, mean {mean_clock:.0f}, drift {max_drift} MHz")
    if not stayed:
        print("         >>> clock drifted under load — thermal/power management")
        print("             may be overriding the lock. Energy results would be unreliable.")

    try:
        pynvml.nvmlDeviceResetGpuLockedClocks(h)
    except pynvml.NVMLError:
        pass
    del a, b
    torch.cuda.empty_cache()


# ── 7. power telemetry is real (CodeCarbon depends on it) ────────────────────

def check_power_telemetry(pynvml, h, torch):
    section("7. Power telemetry (CodeCarbon depends on this)")
    try:
        p_idle = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0   # mW → W
    except pynvml.NVMLError as e:
        critical("nvmlDeviceGetPowerUsage", False, str(e)[:50])
        critical("power telemetry usable", False,
                 "CodeCarbon would silently estimate from TDP — energy numbers invalid")
        return

    critical("power reading is nonzero", p_idle > 1.0, f"idle {p_idle:.1f} W")

    # Power must rise under load — confirms telemetry is live, not a constant.
    if torch is not None:
        dev = torch.device("cuda:0")
        a = torch.randn(8192, 8192, device=dev)
        b = torch.randn(8192, 8192, device=dev)
        for _ in range(50):
            a = (a @ b) * 0.0001 + 0.1
        torch.cuda.synchronize()
        p_load = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        critical("power rises under load", p_load > p_idle + 5,
                 f"idle {p_idle:.1f} W → load {p_load:.1f} W")
        del a, b
        torch.cuda.empty_cache()

    # Confirm CodeCarbon itself can see the GPU.
    try:
        from codecarbon import EmissionsTracker
        warn("codecarbon importable", True)
    except ImportError:
        warn("codecarbon importable", False, "pip install codecarbon")


# ── 8. exclusivity ───────────────────────────────────────────────────────────

def check_exclusivity(pynvml, h):
    section("8. GPU exclusivity (no co-tenants)")
    try:
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
        # This process itself may appear; warn if there is more than one.
        n_other = len(procs)
        warn("no other compute processes", n_other <= 1,
             f"{n_other} compute processes on GPU — others' jobs corrupt energy data"
             if n_other > 1 else "")
    except pynvml.NVMLError as e:
        warn("running process query", False, str(e)[:50])


# ── 9. temperature ───────────────────────────────────────────────────────────

def check_temperature(pynvml, h):
    section("9. GPU temperature")
    try:
        t = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        warn("temperature < 75°C", t < 75, f"{t}°C")
    except pynvml.NVMLError as e:
        warn("temperature query", False, str(e)[:50])


# ── 10. vLLM importable at the pinned version ────────────────────────────────

def check_vllm():
    section("10. vLLM")
    try:
        import vllm
        ver = getattr(vllm, "__version__", "unknown")
        warn("vllm importable", True, f"version {ver}")
        warn("vllm version is 0.6.6", ver == "0.6.6",
             f"got {ver} — patch blanks were written against 0.6.6" if ver != "0.6.6" else "")
    except ImportError:
        warn("vllm importable", False, "pip install vllm==0.6.6")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("SpecDVFS server readiness check")
    print("Running checks from least to most likely to fail …")

    torch = check_torch()
    nvml = check_nvml_basic()
    if nvml is None:
        _verdict()
        return
    pynvml, h = nvml

    check_persistence(pynvml, h)
    levels = check_clock_levels(pynvml, h)
    if levels is not None:
        f_high, f_low, _ = levels
        check_clock_lock_effective(pynvml, h, f_high, f_low)
        check_lock_survives_load(pynvml, h, torch, f_low)
        check_power_telemetry(pynvml, h, torch)
    check_exclusivity(pynvml, h)
    check_temperature(pynvml, h)
    check_vllm()

    _verdict()


def _verdict():
    section("VERDICT")
    if CRITICAL_FAILS:
        print(f"  NO-GO. {len(CRITICAL_FAILS)} critical check(s) failed:")
        for f in CRITICAL_FAILS:
            print(f"    - {f}")
        print("\n  Do not run experiments on this machine until these are resolved.")
    else:
        print("  GO. All critical checks passed.")
    if WARNINGS:
        print(f"\n  {len(WARNINGS)} warning(s) (non-blocking but review):")
        for w in WARNINGS:
            print(f"    - {w}")
    sys.exit(1 if CRITICAL_FAILS else 0)


if __name__ == "__main__":
    main()

# ── USAGE ──────────────────────────────────────────────────────────────────────
# Run as the FIRST thing on any candidate server, before installing models:
#   sudo nvidia-smi -pm 1          # if you have sudo (skip on HPC if not allowed)
#   python verify_server.py
#
# Exit code 0 = safe to commit to this machine for the full experiment sweep.
# Exit code 1 = at least one critical check failed; do not use this machine.
#
# The decisive checks are #5 (clock lock read-back), #6 (lock survives load),
# and #7 (power telemetry is real). A machine can pass a naive "does NVML
# error" test while failing all three of these silently.

