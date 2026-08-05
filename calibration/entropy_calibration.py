"""Entropy–acceptance rate calibration for SpecDVFS.

Fits the relationship α = a · exp(b · H) between the draft model's
output entropy (H) and the verifier's acceptance rate (α).

GELATO (Tang et al., 2026) found a=1.0, b=-0.35 for Qwen2.5.
Coefficients vary by model pair and must be fitted empirically.

Two components in this file:

  FITTING (no GPU, runs anywhere)
    calibrate()               — fits a, b from observed (H, α) pairs
    calibrate_from_file()     — loads pairs from JSON, then calls calibrate()
    plot_calibration()        — saves a diagnostic plot
    self_test()               — round-trip validation with synthetic data

  COLLECTION (requires torch, runs on server during Phase 3)
    EntropyCollector          — forward hook on draft model lm_head that
                                records per-iteration (H, α) pairs to JSON
"""

from __future__ import annotations

import json
import numpy as np
from scipy.optimize import curve_fit
from dataclasses import dataclass, asdict
from typing import Tuple


# ── model ─────────────────────────────────────────────────────────────────────

def _exp_decay(H: np.ndarray, a: float, b: float) -> np.ndarray:
    """α = a · exp(b · H).  b is negative so α falls as entropy rises."""
    return a * np.exp(b * H)


# ── calibration result ────────────────────────────────────────────────────────

@dataclass
class CalibrationResult:
    a:         float    # scale coefficient (near 1.0 for well-matched model pairs)
    b:         float    # decay rate (negative; GELATO baseline = -0.35)
    r_squared: float    # coefficient of determination
    a_stderr:  float    # standard error of a
    b_stderr:  float    # standard error of b
    n_points:  int      # number of (H, α) pairs used in the fit

    def to_json(self, path: str) -> None:
        payload = {"model": "alpha = a * exp(b * H)", **asdict(self)}
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def from_json(path: str) -> "CalibrationResult":
        with open(path) as f:
            d = json.load(f)
        d.pop("model", None)
        return CalibrationResult(**d)


# ── fitting ───────────────────────────────────────────────────────────────────

def calibrate(
    entropies: np.ndarray,
    alphas:    np.ndarray,
    p0:        Tuple[float, float] = (1.0, -0.35),
) -> CalibrationResult:
    """Fit α = a · exp(b · H) to observed data.

    Args:
        entropies: 1-D array of draft model entropies (H ≥ 0).
        alphas:    1-D array of observed acceptance rates (0 ≤ α ≤ 1).
        p0:        initial guess for (a, b).

    Returns:
        CalibrationResult with fitted coefficients and diagnostics.
    """
    H = np.asarray(entropies, dtype=np.float64)
    A = np.asarray(alphas,    dtype=np.float64)

    assert H.shape == A.shape,     "entropies and alphas must have the same length"
    assert len(H) >= 3,            "need ≥ 3 data points for 2-parameter fit"
    assert np.all(H >= 0),         "entropies must be non-negative"
    assert np.all((A >= 0) & (A <= 1)), "alphas must be in [0, 1]"

    popt, pcov = curve_fit(
        _exp_decay, H, A,
        p0=p0,
        bounds=([0.0, -10.0], [5.0, 0.0]),
        maxfev=10_000,
    )
    a_fit, b_fit = popt
    a_se,  b_se  = np.sqrt(np.diag(pcov))

    predicted = _exp_decay(H, a_fit, b_fit)
    ss_res = np.sum((A - predicted) ** 2)
    ss_tot = np.sum((A - np.mean(A)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return CalibrationResult(
        a=float(a_fit), b=float(b_fit), r_squared=float(r2),
        a_stderr=float(a_se), b_stderr=float(b_se), n_points=len(H),
    )


def calibrate_from_file(path: str) -> CalibrationResult:
    """Load (entropy, alpha) pairs from JSON and fit.

    Expected format:
      {"pairs": [{"entropy": 0.5, "alpha": 0.85}, ...]}
    """
    with open(path) as f:
        data = json.load(f)
    pairs = data["pairs"]
    H = np.array([p["entropy"] for p in pairs])
    A = np.array([p["alpha"]   for p in pairs])
    return calibrate(H, A)


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_calibration(
    entropies: np.ndarray,
    alphas:    np.ndarray,
    result:    CalibrationResult,
    save_path: str,
) -> None:
    """Diagnostic plot: observed points + fitted curve + GELATO baseline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.scatter(entropies, alphas, alpha=0.25, s=8,
               color="steelblue", label="observed")

    H_line = np.linspace(0, float(np.max(entropies)) * 1.1, 300)

    ax.plot(H_line, _exp_decay(H_line, result.a, result.b),
            color="coral", linewidth=2,
            label=(f"fit: α = {result.a:.3f}·exp({result.b:.3f}·H)"
                   f"  R²={result.r_squared:.4f}"))

    ax.plot(H_line, _exp_decay(H_line, 1.0, -0.35),
            color="gray", linewidth=1, linestyle="--",
            label="GELATO baseline (1.0, −0.35)")

    ax.set_xlabel("Draft entropy H (nats)")
    ax.set_ylabel("Acceptance rate α")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ── synthetic data (for self-test and development) ────────────────────────────

def _generate_synthetic(
    a: float = 1.0, b: float = -0.35,
    n: int = 500, noise_std: float = 0.05, seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    H = rng.uniform(0.0, 5.0, size=n)
    A = np.clip(a * np.exp(b * H) + rng.normal(0, noise_std, n), 0.0, 1.0)
    return H, A


def self_test() -> None:
    """Generate synthetic data, fit, verify coefficients are recovered."""
    TRUE_A, TRUE_B = 1.0, -0.35
    H, A   = _generate_synthetic(TRUE_A, TRUE_B)
    result = calibrate(H, A)

    ok = (abs(result.a - TRUE_A) < 0.1
          and abs(result.b - TRUE_B) < 0.1
          and result.r_squared > 0.85)

    status = "PASS" if ok else "FAIL"
    print(f"self_test  {status}")
    print(f"  true   a={TRUE_A:.3f}  b={TRUE_B:.3f}")
    print(f"  fit    a={result.a:.3f}  b={result.b:.3f}  R²={result.r_squared:.4f}")
    assert ok, "self-test failed"


# ═══════════════════════════════════════════════════════════════════════════════
# SERVER-SIDE DATA COLLECTION (requires torch — runs during Phase 3)
# ═══════════════════════════════════════════════════════════════════════════════

class EntropyCollector:
    """Forward hook that records per-iteration (entropy, alpha) pairs.

    Attach to the draft model's lm_head layer.  The hook fires once per
    draft token and records the Shannon entropy of the output distribution.
    At the end of each SD iteration, the monkey-patch calls end_iteration()
    with the observed acceptance rate, which flushes the buffered token
    entropies into one (avg_entropy, alpha) pair.

    Typical usage inside the patched vLLM worker:

        collector = EntropyCollector(draft_model)
        # ... run inference ...
        # inside patched score_proposals:
        #   alpha = compute_acceptance_rate(result)
        #   collector.end_iteration(alpha)
        # after inference completes:
        collector.save("calibration/raw_pairs.json")
        result = calibrate_from_file("calibration/raw_pairs.json")
        result.to_json("calibration/fitted.json")
    """

    def __init__(self, draft_model):
        import torch
        import torch.nn.functional as F
        self._F = F
        self._torch = torch
        self._token_entropies: list[float] = []
        self._pairs: list[dict] = []
        self._handle = draft_model.lm_head.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        # output shape: (batch, seq_len, vocab) or (batch, vocab)
        logits = output[:, -1, :] if output.dim() == 3 else output
        probs  = self._F.softmax(logits, dim=-1)
        H      = -self._torch.sum(probs * self._torch.log(probs + 1e-10), dim=-1)
        self._token_entropies.extend(H.detach().cpu().tolist())

    def end_iteration(self, alpha: float) -> None:
        """Mark the boundary between SD iterations.

        Called by the monkey-patch after the scorer returns with the
        observed acceptance rate for this iteration.  Averages all
        draft-token entropies collected since the previous call.
        """
        if self._token_entropies:
            avg_H = sum(self._token_entropies) / len(self._token_entropies)
            self._pairs.append({
                "entropy": avg_H,
                "alpha":   alpha,
                "n_draft_tokens": len(self._token_entropies),
            })
            self._token_entropies.clear()

    def save(self, path: str) -> None:
        """Remove the forward hook and write all pairs to JSON."""
        self._handle.remove()
        with open(path, "w") as f:
            json.dump({"pairs": self._pairs}, f, indent=2)
        print(f"  saved {len(self._pairs)} (entropy, alpha) pairs to {path}")

    @property
    def pairs(self) -> list[dict]:
        return list(self._pairs)


if __name__ == "__main__":
    self_test()

# ── USAGE ──────────────────────────────────────────────────────────────────────
# Self-test (any machine):
#   python calibration/entropy_calibration.py
#
# Full calibration workflow (server, Phase 3):
#   1. During inference with the entropy forward hook, collect pairs:
#        collector = EntropyCollector(draft_model)
#        # ... run SD inference, calling collector.end_iteration(alpha) each iter ...
#        collector.save("calibration/raw_pairs.json")
#   2. Fit the model:
#        result = calibrate_from_file("calibration/raw_pairs.json")
#        result.to_json("calibration/fitted.json")
#        print(f"a={result.a:.3f}  b={result.b:.3f}  R²={result.r_squared:.4f}")
#   3. Use the fitted coefficients in the entropy-based controller:
#        ctrl.on_verify_start_entropy(entropy, a_coeff=result.a, b_coeff=result.b)
