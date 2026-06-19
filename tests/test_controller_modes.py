"""Unit tests for the mode-aware verify/draft policy added to core.py.

Covers the four previously-pending conditions plus the calibration helpers:
  two_level         — verify always f_high, draft f_low, passed α ignored
  fixed_low         — verify f_low and draft f_low (global low)
  adaptive_entropy  — verify clock from last_entropy via a·exp(b·H); α ignored;
                      None last_entropy falls back to the lagging-α path
  coarse            — one clock per ~window held across BOTH phases (no per-phase
                      f_low drop); a new window recomputes the clock
Plus: default mode is adaptive_alpha (regression), the `collecting` property, and
reset_log() clearing the coarse window state.

Zero GPU dependency — runs on any machine.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from controller.core import Phase, FrequencyMapper, SimulatedDVFSController

# Same representative clock range as test_controller.py.
F_HIGH, F_LOW = 1410, 510
F_FLOOR = int(round(0.6 * F_HIGH))   # 846


def _ctrl(mode, **kw):
    return SimulatedDVFSController(F_HIGH, F_LOW, mode=mode, **kw)


# ── two_level: static phase-aware (f_low draft, f_high verify), α ignored ──────
class TestTwoLevel:
    def test_verify_always_f_high_even_with_low_alpha(self):
        c = _ctrl("two_level")
        c.record_acceptance_rate(0.05)          # would map well below f_high
        c.on_verify_start(alpha=0.05)
        assert c.transition_log[-1].freq_mhz == F_HIGH

    def test_verify_alpha_is_not_consumed(self):
        c = _ctrl("two_level")
        c.on_verify_start(alpha=0.05)
        # two_level ignores α entirely, so nothing α-derived is recorded.
        assert c.transition_log[-1].alpha is None

    def test_draft_is_f_low(self):
        c = _ctrl("two_level")
        c.on_draft_start()
        r = c.transition_log[-1]
        assert r.phase == Phase.DRAFT and r.freq_mhz == F_LOW

    def test_full_iteration_two_levels_only(self):
        c = _ctrl("two_level")
        for _ in range(5):
            c.on_draft_start()
            c.on_verify_start(alpha=c.tracker.estimate)
            c.record_acceptance_rate(0.2)
        pf = c.phase_frequencies
        assert set(pf[Phase.DRAFT]) == {F_LOW}
        assert set(pf[Phase.VERIFY]) == {F_HIGH}


# ── fixed_low: naive global low (f_low on verify too) ──────────────────────────
class TestFixedLow:
    def test_verify_is_f_low(self):
        c = _ctrl("fixed_low")
        c.record_acceptance_rate(0.95)          # high α must NOT raise the clock
        c.on_verify_start(alpha=0.95)
        assert c.transition_log[-1].freq_mhz == F_LOW

    def test_draft_is_f_low(self):
        c = _ctrl("fixed_low")
        c.on_draft_start()
        assert c.transition_log[-1].freq_mhz == F_LOW

    def test_everything_is_global_low(self):
        c = _ctrl("fixed_low")
        for _ in range(4):
            c.on_draft_start()
            c.on_verify_start(alpha=0.9)
            c.record_acceptance_rate(0.9)
        freqs = [r.freq_mhz for r in c.transition_log]
        assert freqs and all(f == F_LOW for f in freqs)


# ── adaptive_entropy: leading signal from last_entropy, passed α ignored ───────
class TestAdaptiveEntropy:
    def test_low_entropy_gives_high_frequency(self):
        c = _ctrl("adaptive_entropy")
        c.last_entropy = 0.0                    # α = a*exp(0) = 1.0 -> f_high
        c.on_verify_start(alpha=0.05)           # passed α would give f_floor
        assert c.transition_log[-1].freq_mhz == F_HIGH

    def test_high_entropy_gives_low_frequency(self):
        c = _ctrl("adaptive_entropy")
        c.last_entropy = 5.0                    # α = exp(-1.75) ≈ 0.17 -> f_floor
        c.on_verify_start(alpha=0.95)           # passed α would give f_high
        assert c.transition_log[-1].freq_mhz == F_FLOOR

    def test_passed_alpha_is_ignored_when_entropy_present(self):
        c = _ctrl("adaptive_entropy")
        c.last_entropy = 2.0
        expected_alpha = c.mapper.entropy_to_alpha(2.0, c.entropy_a, c.entropy_b)
        expected_freq = c.mapper.verify_freq(expected_alpha)
        c.on_verify_start(alpha=0.05)           # arbitrary, must be ignored
        rec = c.transition_log[-1]
        assert rec.freq_mhz == expected_freq
        assert rec.alpha == pytest.approx(expected_alpha)

    def test_uses_instance_entropy_coefficients(self):
        # Override the coefficients; the mapped frequency must follow them.
        c = _ctrl("adaptive_entropy", entropy_a=0.5, entropy_b=-0.35)
        c.last_entropy = 0.0                    # α = 0.5*exp(0) = 0.5 (mid-range)
        c.on_verify_start(alpha=0.99)
        expected = c.mapper.verify_freq(0.5)
        assert F_FLOOR < expected < F_HIGH      # genuinely interpolated
        assert c.transition_log[-1].freq_mhz == expected

    def test_none_entropy_falls_back_to_passed_alpha(self):
        c = _ctrl("adaptive_entropy")
        assert c.last_entropy is None           # hook never fired
        c.on_verify_start(alpha=0.9)            # high α -> f_high via fallback
        assert c.transition_log[-1].freq_mhz == F_HIGH
        c.on_verify_start(alpha=0.05)           # low α -> f_floor via fallback
        assert c.transition_log[-1].freq_mhz == F_FLOOR

    def test_none_entropy_fallback_uses_tracker_when_no_alpha(self):
        c = _ctrl("adaptive_entropy")
        c.record_acceptance_rate(0.9)           # tracker estimate -> 0.9
        c.on_verify_start()                     # no α passed, no entropy
        assert c.transition_log[-1].freq_mhz == F_HIGH

    def test_draft_still_f_low(self):
        c = _ctrl("adaptive_entropy")
        c.last_entropy = 0.0
        c.on_draft_start()
        assert c.transition_log[-1].freq_mhz == F_LOW


# ── coarse: one clock per window across BOTH phases (granularity ablation) ─────
class TestCoarse:
    def test_draft_does_not_drop_to_f_low(self):
        # Default tracker estimate 0.5 -> verify_freq(0.5) is strictly above f_low;
        # the whole point of coarse is that draft does NOT get its own f_low.
        c = _ctrl("coarse", coarse_window_ms=1e9, now_fn=lambda: 0.0)
        c.on_draft_start()
        r = c.transition_log[-1]
        assert r.phase == Phase.DRAFT
        assert r.freq_mhz == c.mapper.verify_freq(c.tracker.estimate)
        assert r.freq_mhz != F_LOW

    def test_both_phases_share_one_clock_within_window(self):
        clock = [0.0]
        c = _ctrl("coarse", coarse_window_ms=1e9, now_fn=lambda: clock[0])
        c.on_draft_start()
        c.on_verify_start(alpha=c.tracker.estimate)
        # Change the estimate mid-window; the frozen clock must NOT move.
        c.record_acceptance_rate(0.95)
        c.on_draft_start()
        c.on_verify_start(alpha=c.tracker.estimate)
        freqs = {r.freq_mhz for r in c.transition_log}
        assert len(freqs) == 1                  # one clock held across all 4 calls

    def test_new_window_recomputes_clock(self):
        clock = [0.0]
        c = _ctrl("coarse", coarse_window_ms=100.0, now_fn=lambda: clock[0])
        c.on_draft_start()                      # opens window at estimate 0.5
        first = c.transition_log[-1].freq_mhz
        c.record_acceptance_rate(0.95)          # estimate -> 0.95
        clock[0] = 0.5                          # 500 ms later: window expired
        c.on_draft_start()                      # must recompute with new estimate
        second = c.transition_log[-1].freq_mhz
        assert second == c.mapper.verify_freq(0.95)
        assert second != first

    def test_reset_log_clears_window_state(self):
        c = _ctrl("coarse", coarse_window_ms=1e9, now_fn=lambda: 0.0)
        c.on_draft_start()
        assert c._coarse_freq is not None
        c.reset_log()
        assert c._coarse_freq is None and c._coarse_window_t0 is None
        assert c.transition_log == []


# ── regressions: default behaviour and helpers ────────────────────────────────
class TestDefaultsAndHelpers:
    def test_default_mode_is_adaptive_alpha(self):
        c = SimulatedDVFSController(F_HIGH, F_LOW)
        assert c.mode == "adaptive_alpha"

    def test_default_mode_consumes_alpha(self):
        c = SimulatedDVFSController(F_HIGH, F_LOW)
        c.on_verify_start(alpha=0.9)
        assert c.transition_log[-1].freq_mhz == F_HIGH
        c.on_verify_start(alpha=0.05)
        assert c.transition_log[-1].freq_mhz == F_FLOOR

    def test_collecting_property(self):
        assert SimulatedDVFSController(F_HIGH, F_LOW).collecting is False
        assert _ctrl("collect").collecting is True

    def test_collect_mode_clock_degrades_to_adaptive_alpha(self):
        # As a clock policy, collect behaves like adaptive_alpha (it normally runs
        # enabled=False, but if ever enabled it must still be well-defined).
        c = _ctrl("collect")
        c.on_verify_start(alpha=0.9)
        assert c.transition_log[-1].freq_mhz == F_HIGH

    def test_disabled_controller_is_noop_in_every_mode(self):
        for mode in ("two_level", "fixed_low", "coarse", "adaptive_entropy"):
            c = SimulatedDVFSController(F_HIGH, F_LOW, mode=mode, enabled=False)
            c.last_entropy = 0.0
            c.on_draft_start()
            c.on_verify_start(alpha=0.9)
            assert c.transition_log == []
