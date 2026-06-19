"""Mock test for vllm_hooks/patch_spec_decode.apply_patch.

No GPU and no vLLM needed: a stand-in worker exposes exactly the attributes and
methods the patch wraps (the confirmed 8 blanks), and we assert the controller's
transition log and acceptance tracking behave correctly. This proves the wiring
(right names -> no AttributeError, right phase at the right clock, α read at the
right place). The real multiprocessing/NVML clock-toggle is proven separately by
a one-process smoke run on the VM.

    pytest tests/test_patch_mock.py -v
"""

from __future__ import annotations

import pytest

from controller.core import SimulatedDVFSController, Phase
from vllm_hooks.patch_spec_decode import apply_patch

F_HIGH, F_LOW = 1935, 735          # RTX 3090 VM
F_FLOOR = int(round(0.6 * F_HIGH))  # 1161
VERIFY_AT_DEFAULT_ALPHA = 1548      # verify_freq(0.5): midway between f_floor and f_high


class _FakeTensor:
    """Stand-in for the sampler's CUDA counter tensor: supports .item()."""
    def __init__(self, v=0):
        self.v = int(v)

    def item(self):
        return self.v


class _MockSampler:
    """Mimics worker.spec_decode_sampler's cumulative counters."""
    def __init__(self):
        self.num_accepted_tokens = _FakeTensor(0)  # CUDA tensor in reality
        self.num_draft_tokens = 0                  # plain int in reality

    def accept(self, accepted, drafted):
        self.num_accepted_tokens.v += accepted
        self.num_draft_tokens += drafted


class _MockProposer:
    def get_spec_proposals(self, *args, **kwargs):
        return "proposals"


class _MockScorer:
    def score_proposals(self, *args, **kwargs):
        return "scores"


class _MockWorker:
    """Exposes exactly what apply_patch reaches for (the confirmed blanks)."""
    def __init__(self, accepted=3, drafted=4):
        self.proposer_worker = _MockProposer()   # [D]
        self.scorer = _MockScorer()              # [E]
        self.spec_decode_sampler = _MockSampler()
        self._accepted, self._drafted = accepted, drafted

    def _verify_tokens(self, *args, **kwargs):
        # Emulate the rejection sampler bumping cumulative counters this step.
        self.spec_decode_sampler.accept(self._accepted, self._drafted)
        return ("accepted_token_ids", "logprobs")

    def _run_no_spec(self, *args, **kwargs):
        return "no_spec_output"


def _phases(controller):
    return [(r.phase, r.freq_mhz) for r in controller.transition_log]


def test_wraps_without_attributeerror_and_logs_draft_then_verify():
    c = SimulatedDVFSController(F_HIGH, F_LOW, enabled=True)
    w = _MockWorker(accepted=3, drafted=4)   # this iteration's α = 0.75
    apply_patch(w, c)

    w.proposer_worker.get_spec_proposals("req", set())
    w.scorer.score_proposals("req", "proposals")
    w._verify_tokens("sgml", "scores", "proposals", 4)

    assert _phases(c) == [(Phase.DRAFT, F_LOW),
                          (Phase.VERIFY, VERIFY_AT_DEFAULT_ALPHA)]
    # α read at _verify_tokens, from the counter delta 3/4 = 0.75 (first obs).
    assert c.tracker.n_observations == 1
    assert c.tracker.estimate == pytest.approx(0.75)


def test_lagging_alpha_feeds_next_verify_clock():
    c = SimulatedDVFSController(F_HIGH, F_LOW, enabled=True)
    w = _MockWorker(accepted=3, drafted=4)   # α = 0.75 -> next verify uses f_high
    apply_patch(w, c)

    # iteration 1
    w.proposer_worker.get_spec_proposals("req", set())
    w.scorer.score_proposals("req", "proposals")
    w._verify_tokens("sgml", "scores", "proposals", 4)
    # iteration 2: tracker.estimate is now 0.75 (>= alpha_high) -> f_high
    w.proposer_worker.get_spec_proposals("req", set())
    w.scorer.score_proposals("req", "proposals")

    assert _phases(c) == [
        (Phase.DRAFT, F_LOW),
        (Phase.VERIFY, VERIFY_AT_DEFAULT_ALPHA),  # iter 1 used DEFAULT_ALPHA=0.5
        (Phase.DRAFT, F_LOW),
        (Phase.VERIFY, F_HIGH),                   # iter 2 used observed α=0.75
    ]


def test_low_acceptance_uses_floor_clock():
    c = SimulatedDVFSController(F_HIGH, F_LOW, enabled=True)
    w = _MockWorker(accepted=0, drafted=4)   # α = 0.0 (<= alpha_low) -> f_floor
    apply_patch(w, c)

    w._verify_tokens("sgml", "scores", "proposals", 4)  # record α = 0.0
    w.proposer_worker.get_spec_proposals("req", set())
    w.scorer.score_proposals("req", "proposals")

    assert (Phase.VERIFY, F_FLOOR) in _phases(c)
    assert c.tracker.estimate == pytest.approx(0.0)


def test_fallback_decode_uses_f_low_logged_as_draft():
    c = SimulatedDVFSController(F_HIGH, F_LOW, enabled=True)
    w = _MockWorker()
    apply_patch(w, c)

    assert w._run_no_spec("req") == "no_spec_output"  # original return preserved
    # on_fallback_decode logs as DRAFT at f_low (a non-speculative step).
    assert _phases(c) == [(Phase.DRAFT, F_LOW)]


def test_wrong_attribute_name_raises_attributeerror():
    """A wrong blank must fail loudly at patch time, not silently mis-clock."""
    class _BadWorker:
        proposer_worker = _MockProposer()
        spec_decode_sampler = _MockSampler()
        # no `scorer` attribute -> getattr(worker, "scorer") raises

        def _verify_tokens(self, *a, **k):
            return ("x", "y")

        def _run_no_spec(self, *a, **k):
            return "z"

    c = SimulatedDVFSController(F_HIGH, F_LOW, enabled=True)
    with pytest.raises(AttributeError):
        apply_patch(_BadWorker(), c)


def test_disabled_controller_is_overhead_symmetric():
    """enabled=False: identical code path, frequency-setting off.

    No transitions logged (frequency-setting is gated), but the α read still
    happens (record_acceptance_rate is not gated) so baseline and treatment
    carry the same per-iteration bookkeeping overhead.
    """
    c = SimulatedDVFSController(F_HIGH, F_LOW, enabled=False)
    w = _MockWorker(accepted=3, drafted=4)
    apply_patch(w, c)

    w.proposer_worker.get_spec_proposals("req", set())
    w.scorer.score_proposals("req", "proposals")
    w._verify_tokens("sgml", "scores", "proposals", 4)
    w._run_no_spec("req")

    assert _phases(c) == []                     # no clock changes when disabled
    assert c.tracker.n_observations == 1        # but α still read (symmetric cost)
    assert c.tracker.estimate == pytest.approx(0.75)


def test_apply_patch_is_idempotent():
    c = SimulatedDVFSController(F_HIGH, F_LOW, enabled=True)
    w = _MockWorker()
    apply_patch(w, c)
    apply_patch(w, c)  # second call is a no-op (no double-wrap)

    w.proposer_worker.get_spec_proposals("req", set())
    # exactly one DRAFT, not two
    assert _phases(c) == [(Phase.DRAFT, F_LOW)]


def test_original_return_values_preserved():
    c = SimulatedDVFSController(F_HIGH, F_LOW, enabled=True)
    w = _MockWorker()
    apply_patch(w, c)
    assert w.proposer_worker.get_spec_proposals("req", set()) == "proposals"
    assert w.scorer.score_proposals("req", "proposals") == "scores"
    assert w._verify_tokens("a", "b", "c", 4) == ("accepted_token_ids", "logprobs")
