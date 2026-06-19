"""CPU-only test of the patch's part-2 behaviour: the collect-mode (entropy, α)
append and the acceptance-rate recording, exercised through a fake SpecDecodeWorker.

No torch and no vLLM: apply_patch is pure attribute juggling, and the entropy
forward hook is skipped gracefully when torch is unavailable (so last_entropy is
driven directly here to simulate the hook having fired). The α path runs through
the real metrics_reader, so this also pins the collect trigger to the same α
definition used in production.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from controller.core import SimulatedDVFSController, Phase
from vllm_hooks.patch_spec_decode import apply_patch, _ENTROPY_HANDLE_ATTR

F_HIGH, F_LOW = 1410, 510


# ── minimal stand-in for vLLM's SpecDecodeWorker ──────────────────────────────
class FakeSampler:
    """Holds the cumulative counters metrics_reader reads (plain ints here)."""
    def __init__(self):
        self.num_accepted_tokens = 0
        self.num_draft_tokens = 0


class FakeProposer:
    def get_spec_proposals(self, *a, **k):
        return "proposals"


class FakeScorer:
    def score_proposals(self, *a, **k):
        return "scores"


class FakeWorker:
    """_verify_tokens bumps the counters by (accepted_delta, draft_delta).

    Defaults 8/20 -> per-iteration α = 0.4. draft_delta=0 simulates a fallback /
    non-speculative step, for which metrics_reader yields α = None.
    """
    def __init__(self, accepted_delta=8, draft_delta=20):
        self.proposer_worker = FakeProposer()
        self.scorer = FakeScorer()
        self.spec_decode_sampler = FakeSampler()
        self._accepted_delta = accepted_delta
        self._draft_delta = draft_delta

    def _verify_tokens(self, *a, **k):
        self.spec_decode_sampler.num_accepted_tokens += self._accepted_delta
        self.spec_decode_sampler.num_draft_tokens += self._draft_delta
        return "verified"

    def _run_no_spec(self, *a, **k):
        return "no_spec"


def _patched(controller, **worker_kw):
    w = FakeWorker(**worker_kw)
    apply_patch(w, controller)
    return w


# ── apply_patch wiring on a torch-free fake worker ────────────────────────────
class TestApplyPatchWiring:
    def test_apply_patch_does_not_crash_and_skips_entropy_hook(self):
        c = SimulatedDVFSController(F_HIGH, F_LOW)
        w = _patched(c)
        assert w._specdvfs_wrapped is True
        # No torch -> entropy hook is skipped, so no handle is stored.
        assert getattr(w, _ENTROPY_HANDLE_ATTR, None) is None

    def test_draft_wrapper_invokes_controller(self):
        c = SimulatedDVFSController(F_HIGH, F_LOW)   # enabled, adaptive_alpha
        w = _patched(c)
        w.proposer_worker.get_spec_proposals()
        assert c.transition_log[-1].phase == Phase.DRAFT
        assert c.transition_log[-1].freq_mhz == F_LOW


# ── collect mode: (entropy, α) pairs ──────────────────────────────────────────
class TestCollectMode:
    def test_pair_appended_when_entropy_present(self):
        c = SimulatedDVFSController(F_HIGH, F_LOW, mode="collect", enabled=False)
        c.last_entropy = 1.5                         # simulate the hook having fired
        w = _patched(c)
        w._verify_tokens()
        assert len(c.entropy_pairs) == 1
        pair = c.entropy_pairs[0]
        assert pair["entropy"] == 1.5
        assert pair["alpha"] == pytest.approx(0.4)   # 8/20

    def test_no_pair_when_entropy_none(self):
        c = SimulatedDVFSController(F_HIGH, F_LOW, mode="collect", enabled=False)
        assert c.last_entropy is None                # hook never fired
        w = _patched(c)
        w._verify_tokens()
        assert c.entropy_pairs == []
        # α was still valid, so the tracker is updated regardless of entropy.
        assert c.tracker.n_observations == 1

    def test_no_pair_and_no_record_on_fallback_step(self):
        # Δdraft = 0 -> metrics_reader returns None -> neither record nor append.
        c = SimulatedDVFSController(F_HIGH, F_LOW, mode="collect", enabled=False)
        c.last_entropy = 1.5
        w = _patched(c, draft_delta=0)
        w._verify_tokens()
        assert c.entropy_pairs == []
        assert c.tracker.n_observations == 0

    def test_multiple_iterations_accumulate_pairs(self):
        c = SimulatedDVFSController(F_HIGH, F_LOW, mode="collect", enabled=False)
        w = _patched(c)
        for i in range(3):
            c.last_entropy = float(i)                # fresh hook value each iter
            w._verify_tokens()
        assert len(c.entropy_pairs) == 3
        assert [p["entropy"] for p in c.entropy_pairs] == [0.0, 1.0, 2.0]


# ── non-collect modes never append, but still record α ────────────────────────
class TestNonCollectMode:
    def test_adaptive_alpha_records_alpha_but_no_pairs(self):
        c = SimulatedDVFSController(F_HIGH, F_LOW)    # default adaptive_alpha
        c.last_entropy = 1.5                          # present, but must be ignored
        w = _patched(c)
        w._verify_tokens()
        assert c.tracker.n_observations == 1
        assert c.entropy_pairs == []

    def test_other_modes_do_not_append(self):
        for mode in ("two_level", "fixed_low", "coarse", "adaptive_entropy"):
            c = SimulatedDVFSController(F_HIGH, F_LOW, mode=mode)
            c.last_entropy = 1.5
            w = _patched(c)
            w._verify_tokens()
            assert c.entropy_pairs == [], f"{mode} should not collect pairs"
