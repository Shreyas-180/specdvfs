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

import shutil          # check 12: locate nvidia-cuda-mps-control
import sys
import time

# --- The project's ACTUALLY CONFIGURED clocks (must track run_experiment.py's F_HIGH/F_LOW).
# Deliberately NOT the card's reported max: different physical units of the "same" GPU model
# are binned differently, and the very top of the boost range is frequently NOT sustain-
# lockable — the firmware throttles it down even under an explicit clock-lock request (e.g.
# "requested 2115, read 1725"), which is a normal phenomenon rather than a broken card. This
# is why F_HIGH=1935 was chosen as a separately-validated, sustainable value rather than
# trusting nvmlDeviceGetSupportedGraphicsClocks()[-1]. Gating GO/NO-GO on the reported max
# would test a frequency the project never requests, producing a false NO-GO on any card
# whose absolute ceiling happens to be unstable while saying nothing about whether the value
# actually used is safe on this physical unit.
PROJECT_F_HIGH = 1935   # keep in sync with run_experiment.py's F_HIGH
PROJECT_F_LOW = 735     # keep in sync with run_experiment.py's F_LOW


def _snap(target, levels):
    """Nearest supported clock level to `target` — same rule run_experiment.py's
    resolve_clocks() uses, so this tests the EXACT value a real run would request."""
    return min(levels, key=lambda c: abs(c - target))


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

def check_ceiling_lock_informational(pynvml, h, card_max):
    """Does the card's own reported max hold under a lock request? WARN-level only.

    The project never requests this frequency (see PROJECT_F_HIGH's comment) — this exists
    purely to explain, when it fails, why a lower F_HIGH was the right call, and to flag a
    card whose real usable range is narrower than nvmlDeviceGetSupportedGraphicsClocks()
    advertises. It must never gate GO/NO-GO: a flaky boost ceiling says nothing about the
    value actually used.
    """
    try:
        pynvml.nvmlDeviceSetGpuLockedClocks(h, card_max, card_max)
        time.sleep(0.5)
        sm = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
        ok = abs(sm - card_max) <= max(60, card_max * 0.10)
        warn("card's reported ceiling actually sustain-locks", ok,
             f"requested {card_max}, read {sm} MHz"
             + ("" if ok else " — expected on many cards; firmware throttles the literal max "
                              "even under an explicit lock. Not a problem for this project."))
        pynvml.nvmlDeviceResetGpuLockedClocks(h)
    except pynvml.NVMLError as e:
        warn("card's reported ceiling actually sustain-locks", False, str(e)[:50])


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
        from codecarbon import EmissionsTracker  # noqa: F401  (availability check only)
        warn("codecarbon importable", True)
    except ImportError:
        warn("codecarbon importable", False, "pip install codecarbon")


# ── 8. exclusivity ───────────────────────────────────────────────────────────

def check_exclusivity(pynvml, h):
    section("8. GPU exclusivity (no co-tenants)")
    try:
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
        names = []
        for p in procs:
            try:
                names.append(_proc_name(p.pid))
            except Exception:
                names.append(f"pid{p.pid}")
        # If MPS is active, its control/server processes are EXPECTED co-tenants — that is
        # the whole point of running under MPS (Approach 2's SM restriction requires it) — not
        # a violation of exclusivity. Excluding them by name avoids a false positive that would
        # otherwise fire on every '--sweep sm' run and mask a REAL second tenant if one existed.
        mps_names = {"nvidia-cuda-mps-control", "nvidia-cuda-mps-server"}
        other = [n for n in names if n not in mps_names]
        n_other = len(other)
        detail = f"{n_other} non-MPS compute process(es) on GPU — others' jobs corrupt energy data"
        if len(names) != n_other:
            detail += f" ({len(names) - n_other} MPS process(es) excluded, expected under --sweep sm)"
        warn("no other compute processes", n_other <= 1, detail if n_other > 1 else "")
    except pynvml.NVMLError as e:
        warn("running process query", False, str(e)[:50])


def _proc_name(pid):
    try:
        with open(f"/proc/{pid}/comm") as f:
            return f.read().strip()
    except Exception:
        return f"pid{pid}"


# ── 9. temperature ───────────────────────────────────────────────────────────

def check_temperature(pynvml, h):
    section("9. GPU temperature")
    try:
        t = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        warn("temperature < 75°C", t < 75, f"{t}°C")
    except pynvml.NVMLError as e:
        warn("temperature query", False, str(e)[:50])


# ── 10. vLLM importable at the pinned version ────────────────────────────────

def check_sweep_modules():
    """11. Modules the ridge-crossing sweeps import (--sweep batch / --sweep sm).

    These are WARN-level, not critical: the main pilot runs fine without them. They only
    matter if you intend to run the batch or SM sweeps on this machine.
    """
    section("11. Ridge-crossing sweep modules")
    sm = None
    try:
        import sm_partition as sm
        warn("sm_partition importable", True, "repo-root module found")
    except ImportError:
        warn("sm_partition importable", False,
             "MISSING — must sit at the REPO ROOT (beside experiments/, not inside it). "
             "'--sweep sm' and prof_roofline.py both import it.")
    try:
        import gpu_profiles  # noqa: F401
        warn("gpu_profiles importable", True, "repo-root module found")
    except ImportError:
        warn("gpu_profiles importable", False,
             "MISSING at repo root — analyze_roofline.py imports it for peak TFLOPS/bandwidth.")
    return sm


def check_sm_restriction(pynvml, h, sm_mod):
    """12. Approach 2 prerequisites: SM count + MPS.

    SM restriction uses CUDA_MPS_ACTIVE_THREAD_PERCENTAGE, which the driver reads ONCE when a
    CUDA context is created. Without the MPS control daemon running, that variable is SILENTLY
    IGNORED — every 'SM level' would then run on the full GPU and produce a fake sweep that
    looks perfectly plausible. This check exists to make that failure loud and early.
    """
    section("12. SM restriction (Approach 2, '--sweep sm')")
    if sm_mod is None:
        warn("SM restriction usable", False, "sm_partition not importable (see check 11)")
        return
    # Report the card's real SM count; the sweep ladder is calibrated for 82 (RTX 3090/GA102).
    n_sm = None
    try:
        n_sm = pynvml.nvmlDeviceGetNumGpuCores(h) // 128   # cores/SM = 128 on Ampere consumer
    except Exception:
        try:
            import torch as _t
            p = _t.cuda.get_device_properties(0)
            n_sm = p.multi_processor_count
        except Exception:
            pass
    if n_sm:
        expected = sm_mod.SM_TOTAL_RTX3090
        warn(f"SM count = {n_sm}", True,
             "" if n_sm == expected else
             f"sweep ladder PILOT_SM_SWEEP_COUNTS is calibrated for {expected} SMs (RTX 3090). "
             f"On a {n_sm}-SM card, recompute the premise window before trusting the ladder.")
        lo, hi = sm_mod.premise_window(41.12, 58.53, sm_total=n_sm)
        print(f"      premise window on THIS card (I_draft=41.1 < I* <= I_verify=58.5): "
              f"SMs {int(lo)+1}..{int(hi)}  ({(int(lo)+1)/n_sm*100:.0f}%-{int(hi)/n_sm*100:.0f}% of GPU)")
    else:
        warn("SM count readable", False, "could not determine SM count from NVML or torch")

    have_ctl = shutil.which("nvidia-cuda-mps-control") is not None
    warn("nvidia-cuda-mps-control present", have_ctl,
         "" if have_ctl else "MPS tooling not installed — '--sweep sm' cannot restrict SMs.")
    if have_ctl:
        running = sm_mod.mps_daemon_running()
        warn("MPS daemon running", running,
             "" if running else
             "NOT running. Start it BEFORE '--sweep sm':  nvidia-cuda-mps-control -d   "
             "(without it the thread-percentage cap is silently ignored and every SM level "
             "would secretly run on the full GPU).")


def check_batch_sweep_fit(pynvml, h):
    """13. Approach 1 prerequisites: does the batch ladder fit this card's KV pool?

    Recomputes the batch-sweep feasibility from THIS GPU's memory rather than trusting the
    numbers baked into run_experiment.py (which were measured on a 24GB 3090).
    """
    section("13. Batch sweep feasibility (Approach 1, '--sweep batch')")
    # Config mirrored from run_experiment.py; kept in sync deliberately so this check is
    # meaningful on a machine where the repo has not been edited.
    batches, mml, gen, margin = [4, 8, 11, 16, 22], 1024, 256, 48
    prompt_budget = mml - gen - margin
    warn(f"prompt budget at max_model_len={mml}", prompt_budget > 0,
         f"{mml} - {gen} (generation) - {margin} (margin) = {prompt_budget} tokens "
         f"{'(covers gsm8k/humaneval; cnndm excluded by design)' if prompt_budget > 0 else 'INFEASIBLE'}")
    try:
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        total_gib = info.total / (1024 ** 3)
    except Exception:
        warn("VRAM readable", False, "could not query memory")
        return
    # KV pool ~= 0.90*total - target weights(14.99) - draft(2.32 loaded after) - overhead(2.63)
    # Matches the pilot's observed 3.71 GiB pool -> 1517 blocks x 16 = 24,272 token slots.
    kv_gib = 0.90 * total_gib - 14.99 - 2.41 - 0.22
    slots = int(kv_gib * (1024 ** 3) / (16 * 160 * 1024)) * 16 if kv_gib > 0 else 0
    worst = max(batches) * mml
    ok = slots > 0 and worst <= slots
    warn(f"KV pool fits batch ladder {batches}", ok,
         f"~{kv_gib:.2f} GiB pool ~= {slots:,} token slots; worst case "
         f"{max(batches)}x{mml} = {worst:,} -> {'fits' if ok else 'OVER BUDGET, lower max batch or max_model_len'}")


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
        _, _, gfx_clocks = levels
        # THE check that actually gates GO/NO-GO: lock to what the project will really
        # request, not the card's raw ceiling. Snapped the same way resolve_clocks() snaps
        # at runtime, so a PASS here is a direct guarantee about the real experiment.
        proj_high = _snap(PROJECT_F_HIGH, gfx_clocks)
        proj_low = _snap(PROJECT_F_LOW, gfx_clocks)
        if proj_high != PROJECT_F_HIGH or proj_low != PROJECT_F_LOW:
            warn("PROJECT_F_HIGH/F_LOW are exact card levels",
                 proj_high == PROJECT_F_HIGH and proj_low == PROJECT_F_LOW,
                 f"snapped to {proj_high}/{proj_low} on this card — "
                 f"energy numbers would not be directly comparable to a run at the exact "
                 f"1935/735 used elsewhere; consider --allow-clock-mismatch semantics.")
        check_clock_lock_effective(pynvml, h, proj_high, proj_low)
        check_lock_survives_load(pynvml, h, torch, proj_low)
        check_power_telemetry(pynvml, h, torch)
        # Informational only: does the card's own reported ceiling hold under a lock request?
        # NOT part of the GO/NO-GO verdict — the project never requests this value — but
        # useful to know (e.g. it explains WHY 1935 was chosen below the reported max, and
        # flags a card whose usable range may be narrower than its spec sheet suggests).
        card_max = gfx_clocks[-1]
        if card_max != proj_high:
            section("5b. Card's reported ceiling (informational — NOT gating)")
            check_ceiling_lock_informational(pynvml, h, card_max)
    check_exclusivity(pynvml, h)
    check_temperature(pynvml, h)
    check_vllm()
    # Ridge-crossing sweep prerequisites (WARN-level: the main pilot does not need them).
    sm_mod = check_sweep_modules()
    check_sm_restriction(pynvml, h, sm_mod)
    check_batch_sweep_fit(pynvml, h)

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

