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
"""

from __future__ import annotations

import math
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

    Note on rollback: in vLLM's standard SD implementation (COGA, DYGA,
    EAGLE), the "bonus token" generated when draft tokens are rejected is
    computed inside the scorer's target-model forward pass — there is no
    separate GPU kernel for rollback.  on_rollback() is provided for
    completeness and for SD backends that do issue a distinct rollback pass,
    but it is not called in the standard vLLM monkey-patch.
    """

    def __init__(self, f_high: int, f_low: int, enabled: bool = True):
        self.mapper   = FrequencyMapper(f_high, f_low)
        self.tracker  = AcceptanceRateTracker()
        self.enabled  = enabled
        self._iter:   int = 0
        self._log:    list = []

    # ── injection-point methods (called from vllm_hooks/patch_spec_decode.py) ─

    def on_draft_start(self) -> None:
        """Set GPU to f_low before the draft proposer runs."""
        if not self.enabled:
            return
        freq = self.mapper.draft_freq()
        self.set_frequency_mhz(freq)
        self._record(Phase.DRAFT, freq, None)

    def on_verify_start(self, alpha: Optional[float] = None) -> None:
        """Set GPU frequency before the scorer runs.

        Uses α from the previous iteration (lagging signal).  If no prior
        iteration exists, falls back to the tracker's DEFAULT_ALPHA.

        Args:
            alpha: α from the previous iteration, or None to use the tracker.
        """
        if not self.enabled:
            return
        a    = alpha if alpha is not None else self.tracker.estimate
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

    # ── hardware interface — override in DVFSController ───────────────────────

    def set_frequency_mhz(self, freq_mhz: int) -> None:
        """In simulation mode: no-op.  In DVFSController: calls NVML."""
        log.debug("[sim] set_frequency_mhz(%d)", freq_mhz)

    # ── introspection ─────────────────────────────────────────────────────────

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

    def _record(self, phase: Phase, freq: int, alpha: Optional[float]) -> None:
        self._log.append(TransitionRecord(self._iter, phase, freq, alpha))

# ── USAGE ──────────────────────────────────────────────────────────────────────
# Development (any machine, no GPU required):
#   pytest tests/test_controller.py -v
#
# Server integration:
#   DVFSController in vllm_hooks/dvfs_controller.py inherits this class
#   and overrides set_frequency_mhz() with pynvml.nvmlDeviceSetGpuLockedClocks().
#   Import DVFSController there; import SimulatedDVFSController here for tests.