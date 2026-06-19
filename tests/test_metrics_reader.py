"""Unit tests for vllm_hooks/metrics_reader.py — no GPU, no vLLM.

Covers the fixed contract (clamp to [0,1] / None / never-raise), the per-iteration counter
delta (== vLLM's draft_acceptance_rate), the per-run cumulative read with defensive handle
resolution, and the labelled emitted-prefix diagnostic. See metrics_reader's build spec §9.

    pytest tests/test_metrics_reader.py -v
"""

from __future__ import annotations

import pytest

from vllm_hooks import metrics_reader as mr


# ── fakes ─────────────────────────────────────────────────────────────────────

class _FakeTensor:
    """Mimics a CUDA counter tensor: supports .item()."""
    def __init__(self, v=0):
        self.v = int(v)

    def item(self):
        return self.v


class _Sampler:
    """Cumulative counters like vLLM's SpecDecodeBaseSampler."""
    def __init__(self, accepted=0, draft=0, emitted=0, with_emitted=True):
        self.num_accepted_tokens = _FakeTensor(accepted)  # CUDA tensor in reality
        self.num_draft_tokens = int(draft)                # plain int in reality
        if with_emitted:
            self.num_emitted_tokens = _FakeTensor(emitted)

    def accept(self, accepted, draft, emitted=0):
        self.num_accepted_tokens.v += accepted
        self.num_draft_tokens += draft
        if hasattr(self, "num_emitted_tokens"):
            self.num_emitted_tokens.v += emitted


class _Worker:
    def __init__(self, sampler):
        self.spec_decode_sampler = sampler


class _WorkerWrapper:
    """A driver_worker that wraps the real worker as `.worker` (some executors)."""
    def __init__(self, worker):
        self.worker = worker


class _Executor:
    def __init__(self, driver_worker):
        self.driver_worker = driver_worker


class _Engine:
    def __init__(self, model_executor):
        self.model_executor = model_executor


class _LLM:
    def __init__(self, llm_engine):
        self.llm_engine = llm_engine


def _llm_with(sampler, wrap=False):
    worker = _Worker(sampler)
    driver = _WorkerWrapper(worker) if wrap else worker
    return _LLM(_Engine(_Executor(driver)))


# ── _clamp01 ──────────────────────────────────────────────────────────────────

def test_clamp01_in_range_and_bounds():
    assert mr._clamp01(0.5) == 0.5
    assert mr._clamp01(0.0) == 0.0
    assert mr._clamp01(1.0) == 1.0
    assert mr._clamp01(1.0001) == 1.0      # out-of-range high → clamped, not crash
    assert mr._clamp01(-0.2) == 0.0        # out-of-range low → clamped


def test_clamp01_nan_and_nonnumeric_return_none():
    assert mr._clamp01(float("nan")) is None
    assert mr._clamp01(None) is None
    assert mr._clamp01("oops") is None


# ── per-iteration counter delta ────────────────────────────────────────────────

def test_iteration_alpha_basic_delta():
    before = {"accepted": 10, "draft": 20}
    after = {"accepted": 13, "draft": 24}   # +3 accepted / +4 draft
    assert mr.iteration_acceptance_rate(before, after) == pytest.approx(0.75)


def test_iteration_alpha_all_rejected_is_zero_not_none():
    before = {"accepted": 5, "draft": 8}
    after = {"accepted": 5, "draft": 12}    # +0 / +4 → a real 0.0
    assert mr.iteration_acceptance_rate(before, after) == 0.0


def test_iteration_alpha_no_draft_is_none():
    before = {"accepted": 5, "draft": 8}
    after = {"accepted": 5, "draft": 8}     # fallback / non-spec step → None
    assert mr.iteration_acceptance_rate(before, after) is None


def test_iteration_alpha_missing_snapshot_is_none():
    assert mr.iteration_acceptance_rate(None, {"accepted": 1, "draft": 1}) is None
    assert mr.iteration_acceptance_rate({"accepted": 0, "draft": 0}, None) is None


def test_iteration_alpha_is_clamped():
    before = {"accepted": 0, "draft": 0}
    after = {"accepted": 9, "draft": 4}     # impossible >1 ratio → clamp to 1.0
    assert mr.iteration_acceptance_rate(before, after) == 1.0


# ── snapshot + token counts ────────────────────────────────────────────────────

def test_snapshot_counts_reads_tensor_and_int():
    s = _Sampler(accepted=3, draft=4, emitted=7)
    assert mr.snapshot_counts(s) == {"accepted": 3, "draft": 4, "emitted": 7}


def test_snapshot_counts_omits_missing_emitted():
    s = _Sampler(accepted=3, draft=4, with_emitted=False)
    assert mr.snapshot_counts(s) == {"accepted": 3, "draft": 4}


def test_snapshot_counts_failure_returns_none():
    class _Bad:
        num_draft_tokens = 0

        @property
        def num_accepted_tokens(self):
            raise RuntimeError("boom")

    assert mr.snapshot_counts(_Bad()) is None


def test_iteration_token_counts_delta():
    before = {"accepted": 1, "draft": 4, "emitted": 2}
    after = {"accepted": 4, "draft": 8, "emitted": 6}
    assert mr.iteration_token_counts(before, after) == {
        "proposed": 4, "accepted": 3, "emitted": 4,
    }


def test_iteration_token_counts_empty_on_missing():
    assert mr.iteration_token_counts(None, {"accepted": 0, "draft": 0}) == {}


# ── the "same quantity at two granularities" property ───────────────────────────

def test_per_iteration_delta_equals_per_run_for_one_iteration():
    s = _Sampler()
    before = mr.snapshot_counts(s)
    s.accept(accepted=3, draft=4, emitted=4)        # one verify step
    after = mr.snapshot_counts(s)
    it = mr.iteration_acceptance_rate(before, after)
    run = mr.run_mean_acceptance(_llm_with(s))
    assert it == run == pytest.approx(0.75)


# ── tensor diagnostic (emitted-prefix; NOT the headline α) ──────────────────────

BONUS = 999  # any non-(-1) token id

def test_tensor_diag_partial_accept():
    rows = [[1, 2, 3, BONUS, -1, -1]]               # 3 accepted draft, proposed 5
    assert mr.iteration_acceptance_rate_from_tensor(rows) == pytest.approx(3 / 5)


def test_tensor_diag_all_accepted():
    rows = [[1, 2, 3, 4, 5, BONUS]]                 # 5 draft + bonus, proposed 5
    assert mr.iteration_acceptance_rate_from_tensor(rows) == pytest.approx(1.0)


def test_tensor_diag_all_rejected():
    rows = [[BONUS, -1, -1, -1, -1, -1]]            # 0 accepted draft, proposed 5
    assert mr.iteration_acceptance_rate_from_tensor(rows) == 0.0


def test_tensor_diag_empty_is_none():
    assert mr.iteration_acceptance_rate_from_tensor([]) is None


def test_tensor_diag_mixed_batch_needs_proposal_lens():
    # row0: spec, proposed 5, accepted 2 ; row1: non-spec, proposed 0
    rows = [[1, 2, BONUS, -1, -1, -1], [42, -1, -1, -1, -1, -1]]
    # naive: the non-spec row is wrongly treated as proposed 5 → 2 / 10
    assert mr.iteration_acceptance_rate_from_tensor(rows) == pytest.approx(2 / 10)
    # with proposal_lens the proposed-0 row is skipped → 2 / 5
    assert mr.iteration_acceptance_rate_from_tensor(rows, proposal_lens=[5, 0]) == pytest.approx(2 / 5)


# ── per-run handle resolution ───────────────────────────────────────────────────

def test_run_mean_via_llm_chain():
    assert mr.run_mean_acceptance(_llm_with(_Sampler(accepted=30, draft=50))) == pytest.approx(0.6)


def test_run_mean_via_worker_wrapper():
    assert mr.run_mean_acceptance(_llm_with(_Sampler(accepted=30, draft=50), wrap=True)) == pytest.approx(0.6)


def test_run_mean_handle_is_worker():
    assert mr.run_mean_acceptance(_Worker(_Sampler(accepted=1, draft=4))) == pytest.approx(0.25)


def test_run_mean_handle_is_sampler():
    assert mr.run_mean_acceptance(_Sampler(accepted=1, draft=4)) == pytest.approx(0.25)


def test_run_mean_zero_draft_is_none():
    assert mr.run_mean_acceptance(_llm_with(_Sampler(accepted=0, draft=0))) is None


def test_run_mean_unreachable_is_none():
    class _Nothing:
        pass
    assert mr.run_mean_acceptance(_Nothing()) is None


def test_run_token_counts_cumulative():
    s = _Sampler(accepted=30, draft=50, emitted=40)
    assert mr.run_token_counts(_llm_with(s)) == {
        "num_draft": 50, "num_accepted": 30, "num_emitted": 40,
    }


def test_run_token_counts_unreachable_is_empty():
    class _Nothing:
        pass
    assert mr.run_token_counts(_Nothing()) == {}
