"""vllm_hooks/metrics_reader.py — the single source of truth for the speculative-decoding
acceptance rate α and token counts (blank [H] / SEAM 3).

Confirmed against vLLM 0.6.6 source (vllm/spec_decode/* and
vllm/model_executor/layers/spec_decode_base_sampler.py). α is defined EXACTLY as vLLM
defines it, read off the speculative sampler (RejectionSampler / TypicalAcceptanceSampler),
which lives on the SpecDecodeWorker as ``worker.spec_decode_sampler``:

    α = num_accepted_tokens / num_draft_tokens

vLLM increments these per verify as ``num_accepted_tokens += accepted.sum()`` and
``num_draft_tokens += batch_size * k`` (spec_decode_base_sampler.py:127-129), and computes
its own ``draft_acceptance_rate = accepted_tokens / draft_tokens`` from exactly them
(metrics.py:172). So the per-ITERATION α (a counter *delta*) and the per-RUN α (the
cumulative ratio) are literally the same quantity at two granularities — which is the whole
point of centralizing here: one definition, two callers.

WHY NOT parse the scorer's / verifier's output tensor (the build-spec skeleton's first
guess). Two source-confirmed reasons, both of which would make the number disagree with
vLLM's own metric:
  * ``score_proposals`` returns only the target model's logits/probs; it does not decide
    acceptance, so α is simply not in it.
  * The ``accepted_token_ids`` tensor returned by ``_verify_tokens`` ([batch, k+1], -1
    padded, last column = the single bonus token) reflects the EMITTED prefix (accepted up
    to the first rejection). vLLM flags that this can differ from the raw acceptance count
    (spec_decode_base_sampler.py:118-119, "accepted may have True values inconsistent with
    causal acceptance"); its metric uses ``accepted.sum()`` — the counter, not the tensor.
    The tensor also can't tell a non-speculative row apart from an all-rejected one, so it
    mis-counts the denominator in a mixed batch. A tensor parser IS provided below, but only
    as a clearly-labelled DIAGNOSTIC (emitted-prefix rate) for the smoke-run cross-check —
    never as the headline α.
  (Aside: vLLM 0.6.6 has no per-sequence γ anyway — ``_verify_tokens`` only supports a
   single batch proposal length, and metrics.py:200 asserts ``draft_tokens % k == 0`` — so
   the counter denominator is exact, and the spec's variable-γ worry does not arise here.)

FIXED CONTRACT (do not weaken):
  * every returned α is clamped to [0, 1] or is None;
  * None means "no usable measurement this call" — NEVER 0.0 (all-rejected is a real 0.0);
  * nothing here raises (parse failures return None / {}).
core.py's AcceptanceRateTracker.update() raises ValueError outside [0, 1] and
entropy_calibration.calibrate asserts α ∈ [0, 1], so a stray 1.0001 / NaN would crash a
run — hence the clamp-or-None discipline.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

log = logging.getLogger(__name__)

# Confirmed against vLLM 0.6.6 source.
_REJECT = -1              # sentinel for rejected / padding slots
_NUM_BONUS_TOKENS = 1     # bonus-token columns in the verifier output ([batch, k+1])

ACCEPTED_COUNTER = "num_accepted_tokens"   # CUDA tensor on the sampler
DRAFT_COUNTER = "num_draft_tokens"         # python int on the sampler
EMITTED_COUNTER = "num_emitted_tokens"     # CUDA tensor on the sampler (optional here)
SAMPLER_ATTR = "spec_decode_sampler"       # attribute on SpecDecodeWorker


def _clamp01(x) -> Optional[float]:
    """Coerce to a float in [0, 1]; None for NaN / non-numeric (never raises)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if x != x:  # NaN
        return None
    return max(0.0, min(1.0, x))


def _read_counter(x) -> int:
    """Read a counter that may be a CUDA tensor (.item()) or a plain int."""
    item = getattr(x, "item", None)
    return int(item()) if callable(item) else int(x)


# ─────────────────────────── per-iteration (in the worker) ───────────────────────────
# Source-confirmed: α for one iteration is a delta of the sampler's cumulative counters
# across a single _verify_tokens call. The patch snapshots the counters before/after that
# call and passes both snapshots here, so the α definition lives in ONE place.

def snapshot_counts(sampler) -> Optional[Dict[str, int]]:
    """Cumulative ``{'accepted', 'draft', 'emitted'}`` read off the sampler, or None on
    failure. ``'emitted'`` is omitted if the sampler doesn't expose that counter."""
    try:
        snap = {
            "accepted": _read_counter(getattr(sampler, ACCEPTED_COUNTER)),
            "draft": _read_counter(getattr(sampler, DRAFT_COUNTER)),
        }
        if hasattr(sampler, EMITTED_COUNTER):
            snap["emitted"] = _read_counter(getattr(sampler, EMITTED_COUNTER))
        return snap
    except Exception as e:
        log.debug("specdvfs.metrics: snapshot_counts failed: %s", e)
        return None


def iteration_acceptance_rate(
    before: Optional[Dict[str, int]],
    after: Optional[Dict[str, int]],
) -> Optional[float]:
    """α for ONE SD iteration, in [0, 1], or None.

    α = Δaccepted / Δdraft, i.e. vLLM's draft_acceptance_rate at per-iteration granularity.
    Returns None when no draft tokens were proposed this iteration (a fallback /
    non-speculative step): callers MUST skip controller.record_acceptance_rate() /
    collector.end_iteration() when this is None (AcceptanceRateTracker.update would
    otherwise be fed a meaningless value).

    NB this deliberately takes the two counter snapshots (from ``snapshot_counts``), not a
    ``scorer_output`` — see the module docstring: the source proves α is not recoverable
    from the scorer/verifier output in a way that matches vLLM's own metric.
    """
    try:
        if not before or not after:
            return None
        d_draft = after["draft"] - before["draft"]
        if d_draft <= 0:                       # fallback / non-spec iteration → None
            return None
        d_acc = after["accepted"] - before["accepted"]
        return _clamp01(d_acc / d_draft)
    except Exception as e:
        log.debug("specdvfs.metrics: iteration_acceptance_rate failed: %s", e)
        return None


def iteration_token_counts(
    before: Optional[Dict[str, int]],
    after: Optional[Dict[str, int]],
) -> dict:
    """``{'proposed', 'accepted', 'emitted'}`` for this iteration, or {} if unavailable."""
    try:
        if not before or not after:
            return {}
        out = {
            "proposed": after["draft"] - before["draft"],
            "accepted": after["accepted"] - before["accepted"],
        }
        if "emitted" in before and "emitted" in after:
            out["emitted"] = after["emitted"] - before["emitted"]
        return out
    except Exception as e:
        log.debug("specdvfs.metrics: iteration_token_counts failed: %s", e)
        return {}


# ───────────── per-iteration DIAGNOSTIC (emitted-prefix; NOT vLLM's α) ─────────────

def iteration_acceptance_rate_from_tensor(accepted_token_ids, proposal_lens=None) -> Optional[float]:
    """DIAGNOSTIC ONLY — do not use for the headline α.

    α estimated from the verifier's ``accepted_token_ids`` tensor ([n, k+1], -1 padded,
    last column the bonus token): per spec row, accepted_draft = (#non-(-1)) - 1. This is
    the EMITTED-prefix rate; it can read LOWER than the counter-based draft_acceptance_rate
    when the sampler accepts a token after an earlier rejection (vLLM counts that in
    ``accepted.sum()`` but does not emit it). Use only to cross-check the counter path in
    the VM smoke run — a large gap flags a parsing/version problem.

    ``proposal_lens`` (per-row proposed counts) is needed for a correct denominator in a
    MIXED batch; without it every row is assumed speculative with proposed = (ncols - 1).
    Rows with proposed == 0 (non-speculative) are skipped.
    """
    try:
        if hasattr(accepted_token_ids, "tolist"):
            rows = accepted_token_ids.tolist()
        else:
            rows = list(accepted_token_ids)
        accepted = proposed = 0
        for i, row in enumerate(rows):
            ncols = len(row)
            non_pad = sum(1 for t in row if t != _REJECT)
            acc_draft = max(0, non_pad - _NUM_BONUS_TOKENS)
            p = int(proposal_lens[i]) if proposal_lens is not None else max(0, ncols - _NUM_BONUS_TOKENS)
            if p <= 0:
                continue
            accepted += min(acc_draft, p)      # never credit more than were proposed
            proposed += p
        if proposed <= 0:
            return None
        return _clamp01(accepted / proposed)
    except Exception as e:
        log.debug("specdvfs.metrics: tensor α (diagnostic) failed: %s", e)
        return None


# ─────────────────────────── per-run (in the main process) ───────────────────────────
# The robust per-run source is the sampler's CUMULATIVE counters, read once at end-of-run.
# This is the same accepted/draft ratio vLLM publishes as draft_acceptance_rate, and unlike
# the AsyncMetricsCollector it is not rate-limited (the collector may return None on any
# given call). Reaching the sampler assumes the worker is in-process (TP=1, single-GPU
# offline LLM()) — the project's confirmed case; for forked workers this returns None and
# the orchestrator falls back to averaging the controller log.

_WORKER_PATHS = (
    ("llm_engine", "model_executor", "driver_worker"),
    ("model_executor", "driver_worker"),
    ("driver_worker",),
    (),  # handle is already the worker (or the sampler)
)


def _resolve_sampler(handle):
    """Find the spec_decode_sampler from whatever the orchestrator can pass (the LLM, the
    engine, or the worker). Defensive across small layout differences; returns None if it
    can't be reached from this process."""
    candidates = []
    for path in _WORKER_PATHS:
        obj = handle
        ok = True
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                ok = False
                break
        if ok and obj is not None:
            candidates.append(obj)
            wrapped = getattr(obj, "worker", None)   # some executors wrap the real worker
            if wrapped is not None:
                candidates.append(wrapped)
    for obj in candidates:
        if hasattr(obj, ACCEPTED_COUNTER) and hasattr(obj, DRAFT_COUNTER):
            return obj                                # handle/obj already IS the sampler
        sampler = getattr(obj, SAMPLER_ATTR, None)
        if sampler is not None and hasattr(sampler, ACCEPTED_COUNTER):
            return sampler                            # obj is the worker holding it
    return None


def run_mean_acceptance(handle) -> Optional[float]:
    """Cumulative draft acceptance rate over the whole run, in [0, 1], or None if
    unavailable (sampler unreachable, e.g. forked workers; or no SD steps ran yet)."""
    try:
        sampler = _resolve_sampler(handle)
        if sampler is None:
            return None
        draft = _read_counter(getattr(sampler, DRAFT_COUNTER))
        if draft <= 0:
            return None
        accepted = _read_counter(getattr(sampler, ACCEPTED_COUNTER))
        return _clamp01(accepted / draft)
    except Exception as e:
        log.debug("specdvfs.metrics: run_mean_acceptance failed: %s", e)
        return None


def run_token_counts(handle) -> dict:
    """Cumulative ``{'num_draft', 'num_accepted', 'num_emitted'}`` or {}."""
    try:
        sampler = _resolve_sampler(handle)
        if sampler is None:
            return {}
        out = {
            "num_draft": _read_counter(getattr(sampler, DRAFT_COUNTER)),
            "num_accepted": _read_counter(getattr(sampler, ACCEPTED_COUNTER)),
        }
        if hasattr(sampler, EMITTED_COUNTER):
            out["num_emitted"] = _read_counter(getattr(sampler, EMITTED_COUNTER))
        return out
    except Exception as e:
        log.debug("specdvfs.metrics: run_token_counts failed: %s", e)
        return {}


# ── HOW IT PLUGS IN (the wiring is the next step; shown here for reference) ──────────────
# Patch (per iteration, in the worker) — one definition, imported:
#   from vllm_hooks import metrics_reader
#   before = metrics_reader.snapshot_counts(sampler)
#   out = orig_verify_tokens(*a, **k)
#   alpha = metrics_reader.iteration_acceptance_rate(before, metrics_reader.snapshot_counts(sampler))
#   if alpha is not None:
#       controller.record_acceptance_rate(alpha)          # and, in collect mode,
#       # collector.end_iteration(alpha) under the same guard
#   (patch_spec_decode.py already does exactly this.)
#
# Orchestrator (per run, main process) — fill SEAM 3 in run_experiment.read_alpha_and_tokens,
# passing the llm handle through:
#   alpha_mean = metrics_reader.run_mean_acceptance(llm)
#   if alpha_mean is None and controller is not None and controller.transition_log:
#       alphas = [r.alpha for r in controller.transition_log if r.alpha is not None]
#       alpha_mean = sum(alphas) / len(alphas) if alphas else None   # Option-B fallback
#   counts = metrics_reader.run_token_counts(llm)   # optional: enrich the result JSON
#
# VERIFY the counter names against YOUR installed source (confirmed on v0.6.6):
#   VLLM_ROOT=$(python -c "import vllm; print(vllm.__path__[0])")
#   grep -nE "num_(accepted|draft|emitted)_tokens" \
#        "$VLLM_ROOT"/model_executor/layers/spec_decode_base_sampler.py
#   grep -nE "draft_acceptance_rate|accepted_tokens|draft_tokens" "$VLLM_ROOT"/spec_decode/metrics.py
