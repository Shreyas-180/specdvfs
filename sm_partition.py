"""SM (core) restriction for SpecDVFS — Approach 2: lower the roofline ridge point.

WHY THIS EXISTS
    The pilot's roofline came back NO-GO: ridge I* = 75.9 FLOPs/byte, but the MEASURED
    intensities were draft I=41.1 and verify I=58.5 — BOTH left of the ridge, i.e. both
    memory-bound. The premise of per-phase DVFS (downclock a memory-bound draft, hold a
    compute-bound verify at f_high) therefore had no clean hardware basis on a full 3090.

    The ridge point is a property of the MACHINE, not the kernel:
        I* = peak_FLOPS / peak_bandwidth
    Restricting the number of SMs available to the process cuts peak_FLOPS roughly linearly
    while leaving DRAM bandwidth ~unchanged (the memory controllers and L2 are attached to
    the crossbar, not to individual SMs). So I* falls, and the ridge slides DOWN through the
    two fixed, measured phase intensities.

    Arithmetic intensity of the phases does NOT change under SM restriction: the same kernels
    issue the same FLOPs and move the same bytes. Only the machine balance moves. That is what
    makes this a clean instrument for testing the premise.

THE WINDOW (computed from the pilot's own measured intensities, 3090: 71.0 TFLOPS / 936 GB/s)
        I*(N) = (71.0 * N/82) / 0.936
    Premise needs  I_draft < I* <= I_verify  ->  41.1 < I*(N) <= 58.5  ->  N in [45 .. 63].
        N=82 (100%)  I*=75.9   draft mem-bound, verify mem-bound   <- the pilot's NO-GO
        N=64  (78%)  I*=59.2   both still memory-bound (marginal)
        N=56  (68%)  I*=51.8   draft memory-bound, verify COMPUTE-bound  <- PREMISE HOLDS
        N=48  (59%)  I*=44.4   draft memory-bound, verify COMPUTE-bound  <- PREMISE HOLDS
        N=40  (49%)  I*=37.0   BOTH compute-bound -> premise breaks the OTHER way
    So the window is narrow (~54%-77% of the GPU) and bounded on BOTH sides. Sweeping past it
    in both directions is the point: it brackets the regime rather than assuming it.

MECHANISM: MPS active-thread percentage
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE caps the fraction of SMs a client may use. It requires
    the MPS control daemon to be running, and it is read ONCE when the client process creates
    its CUDA context. Consequences that drive the design here:
      * The env var must be set BEFORE torch/vLLM initialise CUDA -> set it at process start.
      * It cannot be changed mid-process -> an SM SWEEP must run ONE SUBPROCESS PER SM LEVEL.
        run_experiment.py does exactly that (see run_sm_sweep()).
    Requires compute capability >= 7.0 for the thread-percentage control; the 3090 is 8.6.

ALTERNATIVE CONSIDERED (not used): CUDA Green Contexts (cuGreenCtxCreate, CUDA 12.4+) give
    exact SM-count partitions rather than a percentage, but the green context must be made
    current before vLLM builds its own context inside the worker process, which means patching
    vLLM's worker init. MPS achieves the same experimental effect with no in-process CUDA API
    surgery, so it is preferred here. If you later need exact SM counts, that is the upgrade
    path; the percentages below are rounded to the nearest whole SM to stay auditable.

HONEST CAVEAT
    Percentage-based capping is not a guarantee of exactly N SMs, and with fewer SMs issuing
    memory requests the ACHIEVED DRAM bandwidth can also drop somewhat (fewer outstanding
    requests to saturate the controllers). If achieved bandwidth falls, the true ridge sits a
    little LOWER than the linear model above predicts. That is why every SM level should be
    re-profiled with ncu (analyze_roofline.py --sm-count N) rather than trusting this model:
    the model picks the sweep points, the measurement decides the verdict.
"""

from __future__ import annotations

import os
import shutil
import subprocess

# RTX 3090 (GA102). Override via --sm-total if you move to another card.
SM_TOTAL_RTX3090 = 82

ENV_MPS_PCT = "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
ENV_SPECDVFS_SM = "SPECDVFS_SM_COUNT"   # our own breadcrumb, recorded into result JSONs


def sm_to_percent(n_sm: int, sm_total: int = SM_TOTAL_RTX3090) -> int:
    """SM count -> integer MPS percentage (>=1). Rounds to nearest whole percent."""
    if n_sm <= 0 or n_sm > sm_total:
        raise ValueError(f"sm_count {n_sm} out of range 1..{sm_total}")
    return max(1, min(100, int(round(n_sm / sm_total * 100))))


def percent_to_sm(pct: float, sm_total: int = SM_TOTAL_RTX3090) -> int:
    """Inverse of sm_to_percent — the EFFECTIVE SM count a percentage actually buys."""
    return max(1, int(round(pct / 100.0 * sm_total)))


def ridge_point(n_sm: int, peak_tflops_full: float = 71.0, peak_bw_gbs: float = 936.0,
                sm_total: int = SM_TOTAL_RTX3090) -> float:
    """Predicted roofline ridge (FLOPs/byte) when only n_sm SMs are available.

    Compute scales with the SM fraction; bandwidth is assumed unchanged (see the caveat in
    the module docstring — verify by measurement, this only picks sweep points).
    """
    return (peak_tflops_full * n_sm / sm_total) / (peak_bw_gbs / 1000.0)


def premise_window(i_draft: float, i_verify: float, peak_tflops_full: float = 71.0,
                   peak_bw_gbs: float = 936.0, sm_total: int = SM_TOTAL_RTX3090):
    """SM counts N for which  i_draft < I*(N) <= i_verify  (i.e. the premise holds).

    Returns (lo_exclusive, hi_inclusive) as floats; usable N are ceil(lo)..floor(hi).
    """
    k = (peak_bw_gbs / 1000.0) * sm_total / peak_tflops_full
    return i_draft * k, i_verify * k


def mps_daemon_running() -> bool:
    """True if an MPS control daemon appears to be up."""
    if shutil.which("nvidia-cuda-mps-control") is None:
        return False
    try:
        p = subprocess.run(["nvidia-cuda-mps-control", "-s"], capture_output=True,
                           text=True, timeout=5)
        return p.returncode == 0 and bool(p.stdout.strip())
    except Exception:
        return False


def start_mps_daemon() -> bool:
    """Try to start the MPS control daemon. Returns True if it is running afterwards.

    Needs root (you are root on the Vast.ai VM). Safe to call when already running.
    """
    if mps_daemon_running():
        return True
    if shutil.which("nvidia-cuda-mps-control") is None:
        return False
    try:
        subprocess.run(["nvidia-cuda-mps-control", "-d"], capture_output=True,
                       text=True, timeout=15)
    except Exception:
        return False
    return mps_daemon_running()


def stop_mps_daemon() -> None:
    """Shut the MPS daemon down (best effort) — leaves the GPU in its normal state."""
    if shutil.which("nvidia-cuda-mps-control") is None:
        return
    try:
        subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", capture_output=True,
                       text=True, timeout=15)
    except Exception:
        pass


def env_for_sm_limit(n_sm, sm_total: int = SM_TOTAL_RTX3090, base_env=None) -> dict:
    """Env dict for a CHILD process restricted to n_sm SMs.

    n_sm=None (or == sm_total) means 'unrestricted': the MPS percentage var is REMOVED so the
    child is a true full-GPU baseline rather than a 100%-capped one (they should be equivalent,
    but removing it keeps the baseline free of the MPS code path entirely).
    """
    env = dict(os.environ if base_env is None else base_env)
    if n_sm is None or int(n_sm) >= sm_total:
        env.pop(ENV_MPS_PCT, None)
        env[ENV_SPECDVFS_SM] = str(sm_total)
    else:
        env[ENV_MPS_PCT] = str(sm_to_percent(int(n_sm), sm_total))
        env[ENV_SPECDVFS_SM] = str(int(n_sm))
    return env


def active_sm_count(sm_total: int = SM_TOTAL_RTX3090):
    """SM count this process is actually running under, from the env it was launched with.

    Returns None when unrestricted/unknown. Read this INSIDE the run to stamp result JSONs, so
    the recorded value reflects what the process really got, not what a caller intended.
    """
    if ENV_SPECDVFS_SM in os.environ:
        try:
            return int(os.environ[ENV_SPECDVFS_SM])
        except ValueError:
            pass
    if ENV_MPS_PCT in os.environ:
        try:
            return percent_to_sm(float(os.environ[ENV_MPS_PCT]), sm_total)
        except ValueError:
            pass
    return None


def preflight(n_sm, sm_total: int = SM_TOTAL_RTX3090, mock: bool = False) -> bool:
    """Check SM restriction can actually take effect. Returns True if it will.

    Prints a LOUD warning when it will not: without the MPS daemon the percentage var is
    silently ignored by the driver, so an unguarded sweep would produce five identical
    full-GPU runs that merely LOOK like an SM sweep. Failing loudly here is the whole point.
    """
    if mock or n_sm is None or int(n_sm) >= sm_total:
        return True
    if not mps_daemon_running():
        print("    !! MPS daemon NOT running -> CUDA_MPS_ACTIVE_THREAD_PERCENTAGE will be")
        print("       IGNORED and every SM level would silently run on the FULL GPU.")
        print("       Start it first:  nvidia-cuda-mps-control -d      (as root)")
        return False
    return True