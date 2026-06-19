"""Phase 1 profiling — per-phase (draft vs verify) GPU time via CUDA events.

Reuses the SAME hook points as the DVFS patch (the confirmed proposer/scorer method
names, imported from patch_spec_decode so they can never drift apart), but instead of
setting clocks it brackets each call with torch.cuda.Event timers and records
per-iteration draft_ms / verify_ms.

WHY: this quantifies the OPPORTUNITY for per-phase DVFS. The draft/verify time ratio
(and how it grows with gamma) tells you how much wall time is spent in the memory-bound
draft phase — i.e. how much there is to save by downclocking it. It also gives the phase
boundaries in time, to align with the GpuMonitor's util/clock trace.

WHEN: Phase 1, after the patch works and the pilot passes, before the full sweep.
One-time characterization, not part of the measured runs.

Fork-safety / in-process: like the DVFS patch, install_timing() patches
SpecDecodeWorker.init_device at the class level so the timer attaches inside the worker
process. For TP=1 single-GPU offline inference the worker is in-process, so the timer
object is reachable from the main process for dumping.
"""

from __future__ import annotations

import functools
import importlib
import json
import logging
from pathlib import Path

import torch  # GPU-only tool

from vllm_hooks.patch_spec_decode import (
    WORKER_MODULE, WORKER_CLASS_NAME, PROPOSER_ATTR, SCORER_ATTR,
    DRAFT_METHOD, VERIFY_METHOD,
)

log = logging.getLogger(__name__)
_TIMER_FLAG = "_specdvfs_timed"


class PhaseTimer:
    """Brackets draft/verify calls with CUDA events; computes elapsed ms after sync.

    CUDA work is asynchronous, so we record start/end events around each call and only
    call elapsed_time() after torch.cuda.synchronize() in summary()/save().
    """

    def __init__(self):
        self._draft = []   # list of (start_event, end_event)
        self._verify = []

    def _bracket(self, store):
        def deco(orig):
            @functools.wraps(orig)
            def wrapper(*args, **kwargs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                out = orig(*args, **kwargs)
                end.record()
                store.append((start, end))
                return out
            return wrapper
        return deco

    def attach(self, worker):
        proposer = getattr(worker, PROPOSER_ATTR)
        scorer = getattr(worker, SCORER_ATTR)
        setattr(proposer, DRAFT_METHOD,
                self._bracket(self._draft)(getattr(proposer, DRAFT_METHOD)))
        setattr(scorer, VERIFY_METHOD,
                self._bracket(self._verify)(getattr(scorer, VERIFY_METHOD)))
        worker._phase_timer = self
        log.info("specdvfs.timer: attached to %s", type(worker).__name__)

    # ── readout ──────────────────────────────────────────────────────────────

    def _ms(self, pairs):
        torch.cuda.synchronize()
        return [s.elapsed_time(e) for s, e in pairs]   # ms

    def summary(self) -> dict:
        d = self._ms(self._draft)
        v = self._ms(self._verify)
        out = {
            "n_draft_calls": len(d), "n_verify_calls": len(v),
            "draft_ms_total": sum(d), "verify_ms_total": sum(v),
            "draft_ms_mean": (sum(d) / len(d)) if d else None,
            "verify_ms_mean": (sum(v) / len(v)) if v else None,
        }
        if d and v and sum(d) > 0:
            out["verify_to_draft_time_ratio"] = sum(v) / sum(d)
            out["draft_time_fraction"] = sum(d) / (sum(d) + sum(v))
        return out

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        s = self.summary()
        s["draft_ms_per_call"] = self._ms(self._draft)
        s["verify_ms_per_call"] = self._ms(self._verify)
        Path(path).write_text(json.dumps(s, indent=2))
        print(f"  phase timings -> {path}")
        print(f"  draft mean={s['draft_ms_mean']}  verify mean={s['verify_ms_mean']}  "
              f"verify/draft={s.get('verify_to_draft_time_ratio')}")


def install_timing(timer_factory=PhaseTimer):
    """Class-level patch of SpecDecodeWorker.init_device to attach a PhaseTimer in-worker.

    Call in the MAIN process before building the LLM (mutually exclusive with the DVFS
    install for a clean timing run — run profiling separately from the DVFS sweep so the
    clock is fixed and timings aren't confounded).
    """
    mod = importlib.import_module(WORKER_MODULE)
    cls = getattr(mod, WORKER_CLASS_NAME)
    if getattr(cls.init_device, _TIMER_FLAG, False):
        return cls
    orig = cls.init_device

    @functools.wraps(orig)
    def wrapper(self, *a, **k):
        orig(self, *a, **k)
        try:
            timer_factory().attach(self)
        except Exception as e:
            log.error("specdvfs.timer: attach failed: %s", e, exc_info=True)

    wrapper._specdvfs_timed = True
    cls.init_device = wrapper
    log.info("specdvfs.timer: patched %s.init_device", WORKER_CLASS_NAME)
    return cls


# ── USAGE ─────────────────────────────────────────────────────────────────────
# Run a SHORT SD job at a FIXED clock (lock once with nvidia-smi -lgc, or just default),
# with timing installed, then dump:
#
#   from profiling.phase_timer import install_timing
#   install_timing()                       # in the MAIN process, before building the LLM
#   from vllm import LLM, SamplingParams
#   llm = LLM(model=..., speculative_model=..., num_speculative_tokens=5, ...)
#   llm.generate(prompts, SamplingParams(temperature=0, max_tokens=128))
#   # reach the timer (in-process, TP=1):
#   w = llm.llm_engine.model_executor.driver_worker
#   w._phase_timer.save("profiling/out/phase_times_gamma5.json")
#
# Repeat for gamma in {3,5,7} to see the draft-time fraction grow with gamma — the
# empirical basis for "savings scale with gamma".
