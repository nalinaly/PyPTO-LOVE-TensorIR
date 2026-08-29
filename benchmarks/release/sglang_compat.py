"""Backend-neutral compatibility fixes for the pinned SGLang release."""

from __future__ import annotations

import functools

from .workload import ReleaseContractError


GEMMA_RMSNORM_WEIGHT_LOADER_TARGET = (
    "sglang.srt.layers.layernorm.GemmaRMSNorm._weight_loader"
)
_MARKER = "_pypto_release_gemma_offload_compatible"


def _colocate_gemma_weight(layer: object, param: object) -> bool:
    """Move Gemma's derived weight to the CPU when SGLang offloads its parameter."""

    derived = getattr(layer, "gemma_weight", None)
    param_device = getattr(param, "device", None)
    derived_device = getattr(derived, "device", None)
    if param_device is None or derived_device is None:
        raise ReleaseContractError(
            "pinned Gemma RMSNorm weight-loader tensor contract changed"
        )
    if param_device == derived_device:
        return False
    if (
        getattr(param_device, "type", None) != "cpu"
        or getattr(derived_device, "type", None) != "cuda"
    ):
        raise ReleaseContractError(
            "Gemma RMSNorm offload produced an unsupported device split: "
            f"parameter={param_device}, derived={derived_device}"
        )
    layer.gemma_weight = derived.to(device=param_device)
    if layer.gemma_weight.device != param_device:
        raise ReleaseContractError(
            "Gemma RMSNorm derived weight did not follow its parameter"
        )
    return True


def _install_on_class(gemma_rmsnorm: type) -> dict[str, object]:
    current = gemma_rmsnorm._weight_loader
    if getattr(current, _MARKER, False):
        return compatibility_record(installed=True, disposition="already-installed")

    @functools.wraps(current)
    def compatible_weight_loader(layer, param, loaded_weight):
        _colocate_gemma_weight(layer, param)
        result = current(layer, param, loaded_weight)
        if layer.gemma_weight.device != param.device:
            raise ReleaseContractError(
                "Gemma RMSNorm derived weight did not remain colocated"
            )
        return result

    setattr(compatible_weight_loader, _MARKER, True)
    gemma_rmsnorm._weight_loader = compatible_weight_loader
    return compatibility_record(installed=True, disposition="installed")


def install_gemma_rmsnorm_offload_compatibility() -> dict[str, object]:
    """Install the load-time-only fix equally for baseline and PyPTO lanes."""

    from sglang.srt.layers.layernorm import GemmaRMSNorm

    return _install_on_class(GemmaRMSNorm)


def compatibility_record(
    *, installed: bool, disposition: str
) -> dict[str, object]:
    return {
        "name": "sglang-gemma-rmsnorm-offload-device-colocation",
        "target": GEMMA_RMSNORM_WEIGHT_LOADER_TARGET,
        "applies_equally_to_lanes": ["pypto", "sglang-matched", "sglang-optimized"],
        "scope": "model-weight-load-only",
        "installed": installed,
        "disposition": disposition,
        "performance_claim_scope": "excluded-from-steady-state-kernel-comparison",
    }
