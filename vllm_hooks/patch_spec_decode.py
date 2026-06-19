"""Monkey-patch that drives per-phase GPU DVFS during vLLM speculative decoding.

Confirmed against the installed vLLM 0.6.6 spec_decode source. The "8 blanks"
are the module-level constants below; if `grep` on your server shows different
names for your build (see the VERIFY block at the bottom), edit the constants
here and nothing else.

What gets wrapped, and why each is the right hook (all in vllm/spec_decode/):

  proposer_worker.get_spec_proposals   -> DRAFT phase   -> set f_low before it
  scorer.score_proposals               -> VERIFY phase  -> set verify clock before it
  SpecDecodeWorker._verify_tokens       -> rejection sampling: where acceptance is
                                          actually decided -> read α HERE (not from
                                          score_proposals, which only returns logits)
  SpecDecodeWorker._run_no_spec         -> non-speculative fallback decode -> f_low

In addition, a best-effort forward hook is registered on the draft model's
lm_head (reached through the proposer). It computes the mean Shannon entropy of
the proposed draft tokens and stores it on controller.last_entropy. This is the
*leading* signal used by the adaptive_entropy mode (entropy is available between
draft and verify within the same iteration) and the (entropy, α) source for the
calibration pre-pass (collect mode). The hook is fully guarded: if torch is
unavailable (e.g. the CPU mock test) or the lm_head can't be located, it is
skipped and last_entropy stays None.

Phases are detected by wrapping the calls (not by polling the GPU): the clock is
set immediately before the kernel launches, so it launches at the right
frequency. Rollback is NOT wrapped — in vLLM the bonus/corrective token is
produced inside the scorer's single forward pass, so draft and verify are the
only distinct GPU phase boundaries.

Two entry points:

  apply_patch(worker, controller)
      Wrap the calls on an already-constructed SpecDecodeWorker INSTANCE. Pure
      attribute juggling; imports neither vLLM nor NVML. This is what the mock
      test exercises. Raises AttributeError if a name constant is wrong for the
      build — the intended loud failure.

  install(controller_factory)
      Patch SpecDecodeWorker.init_device at the CLASS level. After the original
      init_device runs inside each forked worker process (building .scorer,
      .proposer_worker, .spec_decode_sampler), it constructs a controller via
      controller_factory() — so nvmlInit runs IN THAT PROCESS, keeping the NVML
      handle fork-safe — and calls apply_patch on that worker. Call install() in
      the MAIN process BEFORE the engine spawns workers. No vLLM source is edited.
"""

from __future__ import annotations

import functools
import importlib
import logging
from typing import Callable

from vllm_hooks import metrics_reader

log = logging.getLogger(__name__)

# ── the 8 blanks, confirmed from vLLM 0.6.6 (vllm/spec_decode/*) ──────────────
WORKER_MODULE        = "vllm.spec_decode.spec_decode_worker"  # module holding [C]
WORKER_CLASS_NAME    = "SpecDecodeWorker"     # [C]
PROPOSER_ATTR        = "proposer_worker"      # [D]  worker.proposer_worker  (NOT .proposer)
SCORER_ATTR          = "scorer"               # [E]  worker.scorer           (NOT .scorer_worker)
DRAFT_METHOD         = "get_spec_proposals"   # [A]  proposer_worker.get_spec_proposals(...)
VERIFY_METHOD        = "score_proposals"      # [B]  scorer.score_proposals(...)
VERIFY_TOKENS_METHOD = "_verify_tokens"       # rejection sampling (acceptance decided here)
NO_SPEC_METHOD       = "_run_no_spec"         # non-speculative / fallback decode path
SAMPLER_ATTR         = "spec_decode_sampler"  # holds the [H] counters (read by metrics_reader)

_WRAPPED_FLAG = "_specdvfs_wrapped"
_ENTROPY_HANDLE_ATTR = "_specdvfs_entropy_handle"


# ── entropy hook helpers (adaptive_entropy + collect) ─────────────────────────
#
# The clock-decision logic lives entirely in controller/core.py; these helpers
# only feed it controller.last_entropy. They are intentionally best-effort and
# never raise: a missing lm_head or absent torch leaves last_entropy = None, so
# adaptive_entropy degrades to the lagging-α path and collect records nothing —
# inference is never taken down by entropy bookkeeping.

def _find_draft_lm_head(proposer):
    """Best-effort locate the draft model's lm_head module under the proposer.

    vLLM nests the draft model slightly differently across builds, so try the
    common layouts in order and return the first lm_head found (or None):

        proposer[.worker|._worker].model_runner.model.lm_head
        proposer[.worker|._worker].model.lm_head

    Returning None (rather than raising) lets apply_patch keep working with the
    entropy signal simply disabled — verify the real path on the VM (see the
    VERIFY block) if adaptive_entropy logs the "lm_head not found" warning.
    """
    bases = [proposer,
             getattr(proposer, "worker", None),
             getattr(proposer, "_worker", None)]
    for base in bases:
        if base is None:
            continue
        # proposer[...].model_runner.model.lm_head, then proposer[...].model.lm_head
        for holder in (getattr(base, "model_runner", None), base):
            if holder is None:
                continue
            model = getattr(holder, "model", None)
            if model is not None and hasattr(model, "lm_head"):
                return model.lm_head
    return None


def _install_entropy_hook(worker, proposer, controller):
    """Register a forward hook on the draft lm_head -> controller.last_entropy.

    Best-effort and idempotent (a handle is stashed on the worker). Mirrors the
    entropy definition in calibration/entropy_calibration.py::EntropyCollector —
    per-distribution Shannon entropy over the vocab dim, H = -Σ p·log(p + 1e-10)
    — averaged across the proposed draft-token positions in each forward call, so
    the fitted a·exp(b·H) coefficients describe the same quantity the live
    controller consumes. torch is imported lazily inside this function so module
    import and the CPU mock test stay torch-free.
    """
    if getattr(worker, _ENTROPY_HANDLE_ATTR, None) is not None:
        return  # already installed
    try:
        import torch
        import torch.nn.functional as F
    except Exception as e:                       # no torch (e.g. CPU mock test)
        log.warning("specdvfs: entropy hook skipped (torch unavailable: %s)", e)
        return
    lm_head = _find_draft_lm_head(proposer)
    if lm_head is None:
        log.warning("specdvfs: entropy hook skipped (draft lm_head not found "
                    "under %s); adaptive_entropy falls back to lagging α",
                    PROPOSER_ATTR)
        return

    def _entropy_hook(module, inp, output):
        # output: logits, shape (batch, seq, vocab) or (batch, vocab); some
        # builds wrap it in a tuple. Average per-token entropy over all positions.
        try:
            logits = output[0] if isinstance(output, (tuple, list)) else output
            if logits.dim() == 3:                                  # (batch, seq, vocab)
                logits = logits.reshape(-1, logits.shape[-1])
            probs = F.softmax(logits, dim=-1)
            H = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)    # per-token entropy
            controller.last_entropy = float(H.mean().detach().cpu().item())
        except Exception as e:                   # never let bookkeeping crash inference
            log.debug("specdvfs: entropy hook compute failed: %s", e)

    handle = lm_head.register_forward_hook(_entropy_hook)
    setattr(worker, _ENTROPY_HANDLE_ATTR, handle)
    log.info("specdvfs: entropy hook installed on draft lm_head")


def apply_patch(worker, controller):
    """Wrap draft / verify / acceptance / fallback on a single worker instance.

    Idempotent. Raises AttributeError if any [A]-[E] name is wrong for this
    build (re-check the constants against the VERIFY block below). Also installs
    a best-effort entropy forward hook on the draft lm_head (feeds
    controller.last_entropy for adaptive_entropy and collect); the hook never
    raises and is silently skipped when torch or the lm_head is unavailable.
    """
    if getattr(worker, _WRAPPED_FLAG, False):
        return worker  # never double-wrap

    proposer = getattr(worker, PROPOSER_ATTR)   # [D]
    scorer   = getattr(worker, SCORER_ATTR)     # [E]
    sampler  = getattr(worker, SAMPLER_ATTR)    # for [H]

    # --- DRAFT: f_low before the draft proposer runs (memory-bound) ---
    orig_draft = getattr(proposer, DRAFT_METHOD)          # [A]

    @functools.wraps(orig_draft)
    def draft_wrapper(*args, **kwargs):
        controller.on_draft_start()
        return orig_draft(*args, **kwargs)

    setattr(proposer, DRAFT_METHOD, draft_wrapper)

    # --- VERIFY: set the verify clock from the lagging (previous-iteration) α
    #     before the target forward pass. score_proposals returns only logits;
    #     it does NOT decide acceptance, so α is read later in _verify_tokens.
    #     on_verify_start is mode-aware (controller/core.py): two_level / fixed_low
    #     / coarse / adaptive_entropy each decide what (if anything) to do with the
    #     passed α, so this wrapper stays identical across every condition. ---
    orig_verify = getattr(scorer, VERIFY_METHOD)          # [B]

    @functools.wraps(orig_verify)
    def verify_wrapper(*args, **kwargs):
        controller.on_verify_start(alpha=controller.tracker.estimate)
        return orig_verify(*args, **kwargs)

    setattr(scorer, VERIFY_METHOD, verify_wrapper)

    # --- ACCEPTANCE [H]: per-iteration α = Δaccepted / Δdraft across _verify_tokens.
    #     The α definition lives in ONE place — metrics_reader — and is imported here so the
    #     per-iteration value (this patch) and the per-run value (the orchestrator) are the
    #     same quantity (== vLLM's draft_acceptance_rate). Feeds the EMA tracker for the NEXT
    #     iteration's verify clock. Runs in BOTH enabled and disabled modes so baseline and
    #     treatment carry identical α-reading overhead (record_acceptance_rate is not gated by
    #     `enabled`; only frequency-setting is). metrics_reader never raises and returns None
    #     on a fallback / non-spec step, so bookkeeping can never crash inference. ---
    orig_verify_tokens = getattr(worker, VERIFY_TOKENS_METHOD)

    @functools.wraps(orig_verify_tokens)
    def verify_tokens_wrapper(*args, **kwargs):
        before = metrics_reader.snapshot_counts(sampler)
        out = orig_verify_tokens(*args, **kwargs)
        alpha = metrics_reader.iteration_acceptance_rate(
            before, metrics_reader.snapshot_counts(sampler)
        )
        if alpha is not None:
            controller.record_acceptance_rate(alpha)
            # Calibration pre-pass: pair this iteration's measured α with the
            # draft entropy from the lm_head hook. Trigger is mode == "collect"
            # (set by run_experiment.collect_entropy_pairs, which runs DVFS-off).
            # Guarded on last_entropy so a None (hook never fired) can't poison
            # the curve fit, which asserts H >= 0.
            if (getattr(controller, "mode", None) == "collect"
                    and getattr(controller, "last_entropy", None) is not None):
                controller.entropy_pairs.append({
                    "entropy": float(controller.last_entropy),
                    "alpha":   alpha,
                })
        return out

    setattr(worker, VERIFY_TOKENS_METHOD, verify_tokens_wrapper)

    # --- FALLBACK: non-speculative single-token decode (target model) -> f_low ---
    orig_no_spec = getattr(worker, NO_SPEC_METHOD)

    @functools.wraps(orig_no_spec)
    def no_spec_wrapper(*args, **kwargs):
        controller.on_fallback_decode()
        return orig_no_spec(*args, **kwargs)

    setattr(worker, NO_SPEC_METHOD, no_spec_wrapper)

    # --- ENTROPY: forward hook on the draft lm_head -> controller.last_entropy.
    #     Best-effort; on the CPU mock (no torch) or if the lm_head can't be found
    #     it is silently skipped and last_entropy stays None, so neither apply_patch
    #     nor inference can be taken down by it. ---
    try:
        _install_entropy_hook(worker, proposer, controller)
    except Exception as e:
        log.error("specdvfs: entropy-hook install failed (continuing without "
                  "entropy signal): %s", e, exc_info=True)

    setattr(worker, _WRAPPED_FLAG, True)
    worker._dvfs_controller = controller
    log.info(
        "specdvfs: wrapped %s (draft=%s.%s, verify=%s.%s, accept=%s, fallback=%s)",
        type(worker).__name__, PROPOSER_ATTR, DRAFT_METHOD,
        SCORER_ATTR, VERIFY_METHOD, VERIFY_TOKENS_METHOD, NO_SPEC_METHOD,
    )
    return worker


def install(controller_factory: Callable[[], object]):
    """Class-level patch of SpecDecodeWorker.init_device. Call in the MAIN process.

    The wrapper runs inside each worker PROCESS, after the original init_device
    has built the scorer/proposer/sampler, so the controller (and its nvmlInit)
    is created in the right process. apply_patch is always applied — including
    for disabled (baseline) controllers — so every condition takes an identical
    code path with only frequency-setting toggled.

    controller_factory must build a fresh controller per process, e.g.:
        install(lambda: DVFSController(f_high=1935, f_low=735, enabled=True))
    """
    mod = importlib.import_module(WORKER_MODULE)
    cls = getattr(mod, WORKER_CLASS_NAME)                 # [C]
    if getattr(cls.init_device, _WRAPPED_FLAG, False):
        return cls
    orig_init_device = cls.init_device

    @functools.wraps(orig_init_device)
    def init_device_wrapper(self, *args, **kwargs):
        orig_init_device(self, *args, **kwargs)  # builds scorer/proposer/sampler in-process
        try:
            controller = controller_factory()
            apply_patch(self, controller)
        except Exception as e:
            # A wiring failure must not take the worker (and the run) down.
            log.error("specdvfs: install/apply_patch failed in worker: %s", e,
                      exc_info=True)

    init_device_wrapper._specdvfs_wrapped = True
    cls.init_device = init_device_wrapper
    log.info("specdvfs: patched %s.init_device", WORKER_CLASS_NAME)
    return cls


# ── USAGE ───────────────────────────────────────────────────────────────────────
# Mock test (any machine, no GPU, no vLLM):
#   pytest tests/test_patch_mock.py -v
#
# Real run wiring (RTX 3090 VM), in the MAIN process before building the engine:
#   from vllm_hooks.patch_spec_decode import install
#   from vllm_hooks.dvfs_controller import DVFSController
#   install(lambda: DVFSController(f_high=1935, f_low=735, enabled=True))
#   # ...build vLLM LLM/engine with speculative decoding and run...
#
# VERIFY the 8 blanks against YOUR installed source (these were confirmed on
# v0.6.6; run on the VM and confirm the names match — if any differ, edit the
# constants above). The same VM check should confirm the draft lm_head path used
# by _find_draft_lm_head (model_runner.model.lm_head) and that the lm_head
# forward output is the logits tensor the entropy hook expects:
#   VLLM_ROOT=$(python -c "import vllm; print(vllm.__path__[0])")
#   grep -nE "def (get_spec_proposals|score_proposals)" "$VLLM_ROOT"/spec_decode/interfaces.py
#   grep -nE "class SpecDecodeWorker|self\.(proposer_worker|scorer|spec_decode_sampler)" \
#        "$VLLM_ROOT"/spec_decode/spec_decode_worker.py
#   grep -nE "def (_verify_tokens|_run_no_spec)" "$VLLM_ROOT"/spec_decode/spec_decode_worker.py
#   grep -nE "num_(accepted|draft)_tokens|draft_acceptance_rate" \
#        "$VLLM_ROOT"/spec_decode/metrics.py
