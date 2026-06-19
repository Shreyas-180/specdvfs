"""Controller decision logic for SpecDVFS.

Three classes implement the full decision pipeline:

  AcceptanceRateTracker  — maintains a smoothed estimate of draft-token
                           acceptance rate (α) using an EMA.

  FrequencyMapper        — maps α (or entropy-derived α) to a GPU clock
                           frequency for the verify step, and returns f_low
                           unconditionally for draft and rollback steps.

  SimulatedDVFSController — coordinates the two classes above, exposes
                           the three injection-point methods used by the
                           vLLM monkey-patch, and logs every frequency
                           decision without calling NVML.

On the server, DVFSController (vllm_hooks/dvfs_controller.py) subclasses
SimulatedDVFSController and overrides set_frequency_mhz() with an actual
pynvml.nvmlDeviceSetGpuLockedClocks() call.  All decision logic stays here;
the subclass only adds the hardware call.  This means every logical branch
in this file is exercised by the unit tests (tests/test_controller.py)
without requiring a GPU.

MODE-AWARENESS (the per-condition decision policy lives entirely here so the
CPU unit tests keep full coverage; the patch only juggles attributes):

  adaptive_alpha  (default) — verify clock = verify_freq(lagging α).  Identical
                              to the original behaviour, so the existing tests
                              are unchanged.
  two_level                 — static phase-aware: f_low draft, f_high verify
                              (α ignored).
  fixed_low                 — naive global low: f_low on verify too (draft is
                              already f_low).
  coarse                    — granularity ablation: ONE clock per ~100 ms window
                              held across BOTH phases (no per-phase drop).
  adaptive_entropy          — leading signal: verify clock = verify_freq(α̂) with
                              α̂ = entropy_to_alpha(last_entropy, entropy_a,
                              entropy_b); the passed α is ignored.
  collect                   — calibration pre-pass.  Runs DVFS-off (enabled=False),
                              so the on_* hooks no-op and the GPU stays at the
                              default clock; the patch records (last_entropy, α)
                              pairs into entropy_pairs.  As a clock policy it
                              degrades to adaptive_alpha if ever run enabled.
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

log = logging.getLogger(__name__)


class Phase(Enum):
    IDLE     = "idle"
    DRAFT    = "draft"
    VERIFY   = "verify"
    ROLLBACK = "rollback"


class AcceptanceRateTracker:
    """Smoothed estimate of the draft-token acceptance rate (α).

    Uses an exponential moving average so recent iterations count more
    than older ones, while avoiding over-reaction to single outliers.

    α is the fraction of draft tokens accepted by the verifier per SD
    iteration.  It drives the frequency mapping for the verify step:
      high α  →  verify is doing substantial compute  →  use high clock
      low  α  →  verify is lightweight               →  use lower clock
    """

    DEFAULT_ALPHA = 0.5   # returned before any observations are available

    def __init__(self, ema_decay: float = 0.7):
        """
        Args:
            ema_decay: weight given to the previous estimate on each update.
                new_ema = ema_decay * old_ema + (1 - ema_decay) * new_obs
                Higher values = smoother signal, slower to react to changes.
        """
        if not 0.0 < ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1), got {ema_decay}")
        self._decay = ema_decay
        self._ema:   Optional[float] = None
        self._n_obs: int = 0

    def update(self, alpha: float) -> float:
        """Record one observed acceptance rate and return the updated estimate."""
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self._n_obs += 1
        self._ema = alpha if self._ema is None else (
            self._decay * self._ema + (1.0 - self._decay) * alpha
        )
        log.debug("tracker  obs=%.3f  ema=%.3f  n=%d", alpha, self._ema, self._n_obs)
        return self._ema

    @property
    def estimate(self) -> float:
        """Current smoothed α.  Returns DEFAULT_ALPHA before any observations."""
        return self._ema if self._ema is not None else self.DEFAULT_ALPHA

    @property
    def n_observations(self) -> int:
        return self._n_obs

    def reset(self) -> None:
        self._ema   = None
        self._n_obs = 0


class FrequencyMapper:
    """Maps acceptance rate α to a GPU clock for the verify step.

    Mapping:
      α ≥ alpha_high  →  f_high          (fully compute-bound verify)
      α ≤ alpha_low   →  f_floor          (lightweight verify, 60% of f_high)
      in between      →  linear interpolation between f_floor and f_high

    Draft and rollback are always memory-bound regardless of α, so they
    always return f_low.

    Callers are responsible for snapping the returned value to the nearest
    clock level actually supported by the GPU.  On the server, the real
    DVFSController reads the supported levels via NVML and snaps there.
    """

    def __init__(
        self,
        f_high:     int,
        f_low:      int,
        alpha_high: float = 0.7,
        alpha_low:  float = 0.3,
    ):
        if f_high <= f_low:
            raise ValueError(f"f_high ({f_high}) must exceed f_low ({f_low})")
        if not 0.0 < alpha_low < alpha_high < 1.0:
            raise ValueError("Need 0 < alpha_low < alpha_high < 1")
        self.f_high     = f_high
        self.f_low      = f_low
        self.f_floor    = int(round(0.6 * f_high))
        self.alpha_high = alpha_high
        self.alpha_low  = alpha_low

    def verify_freq(self, alpha: float) -> int:
        """Clock frequency for the verify step, given the previous iteration's α."""
        if alpha >= self.alpha_high:
            return self.f_high
        if alpha <= self.alpha_low:
            return self.f_floor
        t = (alpha - self.alpha_low) / (self.alpha_high - self.alpha_low)
        return int(round(self.f_floor + t * (self.f_high - self.f_floor)))

    def draft_freq(self) -> int:
        """Draft and fallback/rollback are memory-bound — always use f_low."""
        return self.f_low

    def entropy_to_alpha(self, entropy: float, a: float, b: float) -> float:
        """Predict α from draft model entropy H using α = a * exp(b * H).

        This is a leading-indicator approach: entropy is computed from the
        draft model's output distribution BEFORE the verify step begins, so
        the verify frequency can be set within the same iteration rather
        than the next one (which is what the lagging α-based path does).

        Coefficients from GELATO (Qwen2.5 pair):  a=1.0, b=-0.35.
        Must be recalibrated for each model pair using scipy.curve_fit.
        """
        return max(0.0, min(1.0, a * math.exp(b * entropy)))


@dataclass
class TransitionRecord:
    iteration: int
    phase:     Phase
    freq_mhz:  int
    alpha:     Optional[float]


class SimulatedDVFSController:
    """CPU-only DVFS controller that logs decisions without calling NVML.

    Used for all development and unit testing on any machine.  On the server,
    DVFSController (vllm_hooks/dvfs_controller.py) inherits this class and
    overrides set_frequency_mhz() with a real NVML call.

    Injection points — called from the vLLM monkey-patch:

      on_draft_start()            — before proposer.get_spec_proposals()
      on_verify_start()           — before scorer.score_proposals()
      on_verify_start_entropy()   — entropy-based variant of the above
      on_fallback_decode()        — before non-speculative target-model call
      record_acceptance_rate()    — after scorer returns, with observed α

    The verify-step policy is selected by ``self.mode`` (see module docstring).
    ``on_verify_start`` is mode-aware, so the patch can keep calling
    ``on_verify_start(alpha=tracker.estimate)`` unchanged for every condition —
    each mode decides what (if anything) to do with that α.

    Note on rollback: in vLLM's standard SD implementation (COGA, DYGA,
    EAGLE), the "bonus token" generated when draft tokens are rejected is
    computed inside the scorer's target-model forward pass — there is no
    separate GPU kernel for rollback.  on_rollback() is provided for
    completeness and for SD backends that do issue a distinct rollback pass,
    but it is not called in the standard vLLM monkey-patch.
    """

    def __init__(
        self,
        f_high: int,
        f_low: int,
        enabled: bool = True,
        mode: str = "adaptive_alpha",
        entropy_a: float = 1.0,
        entropy_b: float = -0.35,
        coarse_window_ms: float = 100.0,
        now_fn=None,
    ):
        self.mapper   = FrequencyMapper(f_high, f_low)
        self.tracker  = AcceptanceRateTracker()
        self.enabled  = enabled
        # Per-condition verify policy.  Default reproduces the original behaviour
        # exactly, so the existing unit tests are unaffected.
        self.mode     = mode

        # Entropy (leading-signal) state.  ``last_entropy`` is written by the
        # patch's forward hook on the draft lm_head; ``entropy_a``/``entropy_b``
        # are the per-pair α = a·exp(b·H) coefficients (GELATO baseline by
        # default, overwritten with a fitted result when one exists).
        self.last_entropy:  Optional[float] = None
        self.entropy_a:     float = entropy_a
        self.entropy_b:     float = entropy_b
        # Filled by the patch in ``collect`` mode: [{'entropy': H, 'alpha': α}, ...].
        self.entropy_pairs: list = []

        # ``coarse`` ablation: one clock per ~coarse_window_ms window, held across
        # both phases.  ``now_fn`` is injectable so the window logic is testable
        # without real time.
        self._coarse_window_ms = float(coarse_window_ms)
        self._now_fn           = now_fn if now_fn is not None else time.monotonic
        self._coarse_freq:      Optional[int]   = None
        self._coarse_window_t0: Optional[float] = None

        self._iter:   int = 0
        self._log:    list = []

    # ── injection-point methods (called from vllm_hooks/patch_spec_decode.py) ─

    def on_draft_start(self) -> None:
        """Set the draft-phase GPU clock before the proposer runs.

        Per-phase modes pin f_low (draft is memory-bound, so a higher clock buys
        no speedup).  The ``coarse`` ablation instead holds the current window
        clock across BOTH phases — it deliberately does NOT drop to f_low per
        phase, which is exactly the granularity being ablated.
        """
        if not self.enabled:
            return
        if self.mode == "coarse":
            freq = self._coarse_window_freq()
        else:
            freq = self.mapper.draft_freq()
        self.set_frequency_mhz(freq)
        self._record(Phase.DRAFT, freq, None)

    def on_verify_start(self, alpha: Optional[float] = None) -> None:
        """Set the verify-phase GPU clock before the scorer runs (mode-aware).

        The mode selects the policy:
          adaptive_alpha  — verify_freq(lagging α)            [default behaviour]
          two_level       — always f_high  (α ignored)
          fixed_low       — always f_low   (draft already f_low → global low)
          coarse          — the current ~100 ms window clock (one freq / window)
          adaptive_entropy— verify_freq(α̂), α̂ from last_entropy (passed α ignored)
          collect / other — degrade to the lagging-α path

        Args:
            alpha: α from the previous iteration; None → use the tracker.  Only
                   the adaptive_alpha path consumes it.
        """
        if not self.enabled:
            return
        mode = self.mode
        if mode == "two_level":
            a, freq = None, self.mapper.f_high
        elif mode == "fixed_low":
            a, freq = None, self.mapper.draft_freq()          # f_low on verify too
        elif mode == "coarse":
            a, freq = self.tracker.estimate, self._coarse_window_freq()
        elif mode == "adaptive_entropy":
            if self.last_entropy is None:                     # hook hasn't fired yet
                a = alpha if alpha is not None else self.tracker.estimate
            else:
                a = self.mapper.entropy_to_alpha(
                    self.last_entropy, self.entropy_a, self.entropy_b)
            freq = self.mapper.verify_freq(a)
        else:                                                 # adaptive_alpha / collect / unknown
            a = alpha if alpha is not None else self.tracker.estimate
            freq = self.mapper.verify_freq(a)
        self.set_frequency_mhz(freq)
        self._record(Phase.VERIFY, freq, a)

    def on_verify_start_entropy(
        self, entropy: float, a_coeff: float, b_coeff: float
    ) -> None:
        """Entropy-based variant: predict α from the current draft's output entropy.

        Leading indicator — entropy is available between draft and verify
        within the same iteration, so the frequency is set proactively
        rather than using the previous iteration's result.

        This explicit-argument form is kept for direct use and for the unit
        tests; the live ``adaptive_entropy`` mode reaches the same logic through
        ``on_verify_start`` using ``self.last_entropy`` and the stored
        ``self.entropy_a`` / ``self.entropy_b`` coefficients.

        Args:
            entropy:  Shannon entropy of the draft model's final logit distribution.
            a_coeff:  calibrated 'a' coefficient in α = a * exp(b * H).
            b_coeff:  calibrated 'b' coefficient.
        """
        if not self.enabled:
            return
        predicted_alpha = self.mapper.entropy_to_alpha(entropy, a_coeff, b_coeff)
        freq = self.mapper.verify_freq(predicted_alpha)
        self.set_frequency_mhz(freq)
        self._record(Phase.VERIFY, freq, predicted_alpha)

    def on_rollback(self) -> None:
        """Set GPU to f_low for a rollback step (memory-bound like draft)."""
        if not self.enabled:
            return
        freq = self.mapper.draft_freq()
        self.set_frequency_mhz(freq)
        self._record(Phase.ROLLBACK, freq, None)

    def on_fallback_decode(self) -> None:
        """Set GPU to f_low when vLLM skips speculation for an iteration.

        This happens when continuous batching inserts a request that forces
        a non-speculative step.  Single-token decode is memory-bound.
        """
        if not self.enabled:
            return
        freq = self.mapper.draft_freq()
        self.set_frequency_mhz(freq)
        self._record(Phase.DRAFT, freq, None)

    def record_acceptance_rate(self, alpha: float) -> None:
        """Update the EMA tracker after an SD iteration completes."""
        self.tracker.update(alpha)
        self._iter += 1

    # ── coarse-granularity helper ─────────────────────────────────────────────

    def _coarse_window_freq(self) -> int:
        """One clock per ~coarse_window_ms window, applied to BOTH phases.

        The clock is chosen with the SAME α→freq rule as adaptive_alpha
        (``verify_freq`` on the current EMA estimate) but is frozen for the
        duration of the window.  The only difference from per-phase DVFS is
        therefore the granularity — which is the point of the ablation.
        """
        now = self._now_fn()
        if (self._coarse_freq is None
                or self._coarse_window_t0 is None
                or (now - self._coarse_window_t0) * 1000.0 >= self._coarse_window_ms):
            self._coarse_window_t0 = now
            self._coarse_freq = self.mapper.verify_freq(self.tracker.estimate)
        return self._coarse_freq

    # ── hardware interface — override in DVFSController ───────────────────────

    def set_frequency_mhz(self, freq_mhz: int) -> None:
        """In simulation mode: no-op.  In DVFSController: calls NVML."""
        log.debug("[sim] set_frequency_mhz(%d)", freq_mhz)

    # ── introspection ─────────────────────────────────────────────────────────

    @property
    def collecting(self) -> bool:
        """True while running the calibration pre-pass (mode == 'collect')."""
        return self.mode == "collect"

    @property
    def transition_log(self) -> list:
        return list(self._log)

    @property
    def phase_frequencies(self) -> dict:
        result = {p: [] for p in Phase}
        for r in self._log:
            result[r.phase].append(r.freq_mhz)
        return result

    def reset_log(self) -> None:
        self._log.clear()
        self._iter = 0
        # Start each run with a fresh coarse window so the first draft opens one.
        self._coarse_freq = None
        self._coarse_window_t0 = None

    def _record(self, phase: Phase, freq: int, alpha: Optional[float]) -> None:
        self._log.append(TransitionRecord(self._iter, phase, freq, alpha))

# ── USAGE ──────────────────────────────────────────────────────────────────────
# Development (any machine, no GPU required):
#   pytest tests/test_controller.py tests/test_controller_modes.py -v
#
# Server integration:
#   DVFSController in vllm_hooks/dvfs_controller.py inherits this class
#   and overrides set_frequency_mhz() with pynvml.nvmlDeviceSetGpuLockedClocks().
#   Import DVFSController there; import SimulatedDVFSController here for tests.
