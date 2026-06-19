"""Server-side DVFS controller: real NVML clock control for SpecDVFS.

`DVFSController` subclasses `SimulatedDVFSController` (controller/core.py) and
overrides ONLY `set_frequency_mhz()` with a real
`pynvml.nvmlDeviceSetGpuLockedClocks()` call (plus clock snapping and a reset
helper). Every decision branch already lives in core.py, so the 34 CPU-only
unit tests fully cover the logic; this file adds the hardware effect and nothing
else.

Multiprocessing (the easy-to-get-wrong part): NVML handles are NOT fork-safe and
vLLM spawns its GPU workers as separate OS processes. Therefore this controller
must be *constructed inside the worker process* — constructing it is what calls
`nvmlInit()`. `vllm_hooks/patch_spec_decode.install()` arranges exactly that by
taking a factory (`lambda: DVFSController(...)`) and calling it from within the
patched `init_device`, which runs post-fork in the worker. Do not construct this
in the parent process.

`pynvml` is provided by the `nvidia-ml-py` package (import name stays `pynvml`);
the legacy `pynvml` package is deprecated.

Device index note: NVML enumerates *physical* GPUs. If vLLM sets
CUDA_VISIBLE_DEVICES per worker, the CUDA ordinal may not equal the NVML index.
On the single-GPU RTX 3090 VM, index 0 is correct. For multi-GPU, resolve the
handle by UUID / PCI bus id instead.
"""

from __future__ import annotations

import logging

import pynvml

from controller.core import SimulatedDVFSController

log = logging.getLogger(__name__)


class DVFSController(SimulatedDVFSController):
    """`SimulatedDVFSController` + a real NVML locked-clock call.

    On the RTX 3090 VM instantiate with ``DVFSController(f_high=1935, f_low=735)``.
    """

    def __init__(
        self,
        f_high: int,
        f_low: int,
        device_index: int = 0,
        enabled: bool = True,
    ):
        super().__init__(f_high=f_high, f_low=f_low, enabled=enabled)
        self.device_index = device_index
        self._handle = None
        self._supported: list[int] = []   # sorted supported graphics clocks (MHz)
        self._nvml_ready = False
        # Only touch NVML for real (treatment) runs. A disabled baseline keeps
        # the GPU at its default governor; set_frequency_mhz() no-ops anyway.
        if self.enabled:
            self._init_nvml()

    # ── NVML setup ────────────────────────────────────────────────────────────

    def _init_nvml(self) -> None:
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            # Enumerate every supported graphics clock across all memory clocks,
            # then dedupe + sort. snap() picks the nearest of these (127 levels
            # on this GPU) — the GPU rejects clocks it does not actually support.
            clocks: set[int] = set()
            for mem in pynvml.nvmlDeviceGetSupportedMemoryClocks(self._handle):
                for g in pynvml.nvmlDeviceGetSupportedGraphicsClocks(
                    self._handle, mem
                ):
                    clocks.add(int(g))
            self._supported = sorted(clocks)
            self._nvml_ready = True
            log.info(
                "NVML ready: device %d, %d supported graphics clocks (%d-%d MHz)",
                self.device_index,
                len(self._supported),
                self._supported[0] if self._supported else -1,
                self._supported[-1] if self._supported else -1,
            )
        except pynvml.NVMLError as e:
            # Never abort inference because clock control is unavailable: log and
            # fall through to default clocks (set_frequency_mhz() will no-op).
            self._nvml_ready = False
            log.error(
                "NVML init failed (%s); DVFS disabled, inference continues at "
                "default clocks", e,
            )

    def _snap(self, freq_mhz: int) -> int:
        """Snap a requested frequency to the nearest GPU-supported clock."""
        if not self._supported:
            return freq_mhz
        return min(self._supported, key=lambda c: abs(c - freq_mhz))

    # ── the one overridden method ───────────────────────────────────────────────

    def set_frequency_mhz(self, freq_mhz: int) -> None:
        """Lock the GPU graphics clock to (the nearest supported) ``freq_mhz``."""
        if not self.enabled or not self._nvml_ready:
            return
        target = self._snap(freq_mhz)
        try:
            # min == max pins the clock. Lock slightly below boost upstream
            # (f_high=1935, the read-back value) so thermal headroom does not let
            # the GPU override the lock.
            pynvml.nvmlDeviceSetGpuLockedClocks(self._handle, target, target)
            log.debug("locked GPU clock to %d MHz (requested %d)", target, freq_mhz)
        except pynvml.NVMLError as e:
            log.error(
                "nvmlDeviceSetGpuLockedClocks(%d) failed: %s (continuing)",
                target, e,
            )

    # ── teardown ────────────────────────────────────────────────────────────────

    def reset_clocks(self) -> None:
        """Release the lock so later runs do not inherit a pinned clock.

        Call after every experiment (and on shutdown). The handle lives on the
        worker as ``worker._dvfs_controller`` once apply_patch has run.
        """
        if not self._nvml_ready:
            return
        try:
            pynvml.nvmlDeviceResetGpuLockedClocks(self._handle)
            log.info("reset GPU locked clocks to default")
        except pynvml.NVMLError as e:
            log.error("nvmlDeviceResetGpuLockedClocks failed: %s", e)

    def shutdown(self) -> None:
        """Reset clocks then shut NVML down for this process."""
        self.reset_clocks()
        if self._nvml_ready:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError as e:
                log.error("nvmlShutdown failed: %s", e)
            self._nvml_ready = False


# ── USAGE ───────────────────────────────────────────────────────────────────────
# This file is GPU-only and is NOT importable without `nvidia-ml-py` installed.
# All decision logic is in controller/core.py and is tested there without a GPU.
#
# Server wiring (single long session, RTX 3090 VM):
#   from vllm_hooks.patch_spec_decode import install
#   from vllm_hooks.dvfs_controller import DVFSController
#   # BEFORE constructing the vLLM engine, in the MAIN process:
#   install(lambda: DVFSController(f_high=1935, f_low=735, enabled=True))
#   # ... then build the LLM/engine and run as usual ...
#   # After experiments, release the lock from inside the worker, e.g. via the
#   # controller stashed on the worker as worker._dvfs_controller.reset_clocks(),
#   # or run `nvidia-smi -rgc` from the shell.
#
# Baseline (no DVFS) run: pass enabled=False to the factory. apply_patch still
# wraps the same calls (identical overhead) but set_frequency_mhz() no-ops, so
# the GPU stays at default clocks — the correct vanilla baseline.
