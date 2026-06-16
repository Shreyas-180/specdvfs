"""Unit tests for the SpecDVFS controller decision logic.

Covers AcceptanceRateTracker, FrequencyMapper, and SimulatedDVFSController.
Zero GPU dependency — runs on any machine.
"""

import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from controller.core import (
    AcceptanceRateTracker, FrequencyMapper,
    Phase, SimulatedDVFSController, TransitionRecord,
)

# Representative A100 clock range used as test constants.
F_HIGH, F_LOW = 1410, 510


# ── AcceptanceRateTracker ─────────────────────────────────────────────────────

class TestAcceptanceRateTracker:

    def test_default_before_observations(self):
        assert AcceptanceRateTracker().estimate == AcceptanceRateTracker.DEFAULT_ALPHA

    def test_cold_start_equals_first_observation(self):
        t = AcceptanceRateTracker()
        assert t.update(0.8) == pytest.approx(0.8)

    def test_ema_moves_toward_new_value(self):
        t = AcceptanceRateTracker(ema_decay=0.5)
        t.update(1.0)
        prev = t.estimate
        t.update(0.0)
        assert t.estimate < prev

    def test_ema_converges_to_constant_signal(self):
        t = AcceptanceRateTracker(ema_decay=0.7)
        for _ in range(200):
            t.update(0.9)
        assert t.estimate == pytest.approx(0.9, abs=0.01)

    def test_sustained_drop_lowers_estimate(self):
        t = AcceptanceRateTracker(ema_decay=0.7)
        for _ in range(50):
            t.update(0.85)
        high = t.estimate
        for _ in range(50):
            t.update(0.1)
        assert t.estimate < high - 0.2

    def test_estimate_stays_in_unit_interval(self):
        t = AcceptanceRateTracker()
        for v in [0.0, 0.3, 0.7, 1.0, 0.5, 0.0, 1.0]:
            t.update(v)
            assert 0.0 <= t.estimate <= 1.0

    def test_update_return_equals_estimate(self):
        t = AcceptanceRateTracker()
        for v in [0.2, 0.5, 0.8]:
            assert t.update(v) == t.estimate

    def test_n_observations_increments(self):
        t = AcceptanceRateTracker()
        for i in range(5):
            t.update(0.5)
            assert t.n_observations == i + 1

    def test_reset_clears_all_state(self):
        t = AcceptanceRateTracker()
        for _ in range(20):
            t.update(0.9)
        t.reset()
        assert t.estimate == AcceptanceRateTracker.DEFAULT_ALPHA
        assert t.n_observations == 0

    def test_invalid_alpha_raises(self):
        t = AcceptanceRateTracker()
        with pytest.raises(ValueError): t.update(-0.1)
        with pytest.raises(ValueError): t.update(1.01)

    def test_boundary_values_accepted(self):
        t = AcceptanceRateTracker()
        t.update(0.0)
        t.update(1.0)


# ── FrequencyMapper ───────────────────────────────────────────────────────────

class TestFrequencyMapper:

    @pytest.fixture
    def m(self): return FrequencyMapper(F_HIGH, F_LOW)

    def test_high_alpha_returns_f_high(self, m):
        for a in (0.7, 0.85, 1.0):
            assert m.verify_freq(a) == F_HIGH

    def test_low_alpha_returns_floor(self, m):
        floor = int(round(0.6 * F_HIGH))
        for a in (0.0, 0.1, 0.3):
            assert m.verify_freq(a) == floor

    def test_midpoint_is_strictly_between_floor_and_max(self, m):
        assert m.f_floor < m.verify_freq(0.5) < F_HIGH

    def test_interpolation_is_monotone(self, m):
        alphas = [i / 10 for i in range(11)]
        freqs  = [m.verify_freq(a) for a in alphas]
        for i in range(len(freqs) - 1):
            assert freqs[i] <= freqs[i + 1], \
                f"non-monotone at α={alphas[i]:.1f}: {freqs[i]} > {freqs[i+1]}"

    def test_draft_always_returns_f_low(self, m):
        assert m.draft_freq() == F_LOW

    def test_f_floor_is_60_pct_of_f_high(self, m):
        assert m.f_floor == int(round(0.6 * F_HIGH))

    def test_inverted_f_high_f_low_raises(self):
        with pytest.raises(ValueError):
            FrequencyMapper(f_high=500, f_low=1000)

    def test_entropy_gelato_baseline(self, m):
        # At H=0 (certain): α = exp(0) = 1.0
        assert m.entropy_to_alpha(0.0, 1.0, -0.35) == pytest.approx(1.0)
        # At H=2: α = exp(-0.70) ≈ 0.497
        assert m.entropy_to_alpha(2.0, 1.0, -0.35) == pytest.approx(math.exp(-0.70), abs=0.001)

    def test_entropy_to_alpha_clamps(self, m):
        result = m.entropy_to_alpha(999.0, 1.0, -0.35)
        assert 0.0 <= result <= 1.0


# ── SimulatedDVFSController ───────────────────────────────────────────────────

class TestSimulatedDVFSController:

    @pytest.fixture
    def ctrl(self): return SimulatedDVFSController(F_HIGH, F_LOW)

    def test_draft_start_logs_f_low(self, ctrl):
        ctrl.on_draft_start()
        r = ctrl.transition_log[-1]
        assert r.phase == Phase.DRAFT and r.freq_mhz == F_LOW

    def test_verify_start_no_alpha_uses_default(self, ctrl):
        ctrl.on_verify_start()
        freq = ctrl.transition_log[-1].freq_mhz
        assert ctrl.mapper.f_floor <= freq <= F_HIGH

    def test_verify_high_alpha_gives_f_high(self, ctrl):
        ctrl.on_verify_start(alpha=0.95)
        assert ctrl.transition_log[-1].freq_mhz == F_HIGH

    def test_verify_low_alpha_gives_floor(self, ctrl):
        ctrl.on_verify_start(alpha=0.05)
        assert ctrl.transition_log[-1].freq_mhz == ctrl.mapper.f_floor

    def test_rollback_uses_f_low(self, ctrl):
        ctrl.on_rollback()
        r = ctrl.transition_log[-1]
        assert r.phase == Phase.ROLLBACK and r.freq_mhz == F_LOW

    def test_fallback_decode_uses_f_low(self, ctrl):
        ctrl.on_fallback_decode()
        assert ctrl.transition_log[-1].freq_mhz == F_LOW

    def test_full_iteration_alternates_low_high(self, ctrl):
        for _ in range(5):
            ctrl.on_draft_start()
            ctrl.on_verify_start(alpha=0.9)
            ctrl.record_acceptance_rate(0.9)
        log = ctrl.transition_log
        assert len(log) == 10
        for i, r in enumerate(log):
            if i % 2 == 0:
                assert r.phase == Phase.DRAFT  and r.freq_mhz == F_LOW
            else:
                assert r.phase == Phase.VERIFY and r.freq_mhz == F_HIGH

    def test_verify_adapts_to_alpha_drop(self, ctrl):
        ctrl.on_verify_start(alpha=0.9)
        freq_high = ctrl.transition_log[-1].freq_mhz
        ctrl.on_verify_start(alpha=0.1)
        freq_low  = ctrl.transition_log[-1].freq_mhz
        assert freq_high > freq_low

    def test_entropy_low_entropy_gives_high_freq(self, ctrl):
        # entropy=0.1 → α ≈ 0.97 → f_high
        ctrl.on_verify_start_entropy(0.1, 1.0, -0.35)
        assert ctrl.transition_log[-1].freq_mhz == F_HIGH

    def test_entropy_high_entropy_gives_lower_freq(self, ctrl):
        ctrl.on_verify_start_entropy(0.1, 1.0, -0.35)
        f1 = ctrl.transition_log[-1].freq_mhz
        ctrl.on_verify_start_entropy(4.0, 1.0, -0.35)
        f2 = ctrl.transition_log[-1].freq_mhz
        assert f1 >= f2

    def test_disabled_logs_nothing(self):
        ctrl = SimulatedDVFSController(F_HIGH, F_LOW, enabled=False)
        ctrl.on_draft_start()
        ctrl.on_verify_start(alpha=0.9)
        ctrl.on_rollback()
        assert len(ctrl.transition_log) == 0

    def test_phase_frequencies_groups_correctly(self, ctrl):
        ctrl.on_draft_start()
        ctrl.on_verify_start(alpha=1.0)
        ctrl.on_draft_start()
        ctrl.on_verify_start(alpha=1.0)
        pf = ctrl.phase_frequencies
        assert len(pf[Phase.DRAFT])  == 2
        assert len(pf[Phase.VERIFY]) == 2
        assert all(f == F_LOW  for f in pf[Phase.DRAFT])
        assert all(f == F_HIGH for f in pf[Phase.VERIFY])

    def test_reset_log_clears_transitions(self, ctrl):
        ctrl.on_draft_start()
        ctrl.on_verify_start(alpha=0.8)
        ctrl.reset_log()
        assert len(ctrl.transition_log) == 0


# ── Integration ───────────────────────────────────────────────────────────────

class TestIntegration:

    def test_declining_alpha_lowers_verify_frequency_over_time(self):
        ctrl   = SimulatedDVFSController(F_HIGH, F_LOW)
        alphas = [0.9] * 10 + [0.2] * 10
        verify_freqs = []
        for alpha in alphas:
            ctrl.on_draft_start()
            ctrl.on_verify_start(alpha=ctrl.tracker.estimate)
            verify_freqs.append(ctrl.transition_log[-1].freq_mhz)
            ctrl.record_acceptance_rate(alpha)
        avg_first  = sum(verify_freqs[:10])  / 10
        avg_second = sum(verify_freqs[10:]) / 10
        assert avg_first > avg_second, \
            f"expected frequency to drop as α dropped; first={avg_first:.0f} second={avg_second:.0f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

# ── USAGE ──────────────────────────────────────────────────────────────────────
# Run from the project root:
#   pytest tests/test_controller.py -v
#
# All 33 tests must pass before any server-GPU integration work begins.
# These tests are the contract for the decision logic; the real DVFSController
# in vllm_hooks/dvfs_controller.py must not change any behavior tested here.