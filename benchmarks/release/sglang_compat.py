"""Backend-neutral compatibility fixes for the pinned SGLang release."""

from __future__ import annotations

import functools
import inspect

from .workload import ReleaseContractError


GEMMA_RMSNORM_WEIGHT_LOADER_TARGET = (
    "sglang.srt.layers.layernorm.GemmaRMSNorm._weight_loader"
)
OFFLOADER_FUNCTIONAL_CALL_TARGET = "sglang.srt.utils.offloader.functional_call"
_GEMMA_MARKER = "_pypto_release_gemma_offload_compatible"
_OFFLOADER_MARKER = "_pypto_untied_parameter_compatible"
_VIEW_MARKER = "_pypto_release_offload_view_alias_compatible"


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
    if getattr(current, _GEMMA_MARKER, False):
        return _component_record(
            name="gemma-rmsnorm-derived-weight-colocation",
            target=GEMMA_RMSNORM_WEIGHT_LOADER_TARGET,
            scope="model-weight-load-only",
            disposition="already-installed",
        )

    @functools.wraps(current)
    def compatible_weight_loader(layer, param, loaded_weight):
        _colocate_gemma_weight(layer, param)
        result = current(layer, param, loaded_weight)
        if layer.gemma_weight.device != param.device:
            raise ReleaseContractError(
                "Gemma RMSNorm derived weight did not remain colocated"
            )
        return result

    setattr(compatible_weight_loader, _GEMMA_MARKER, True)
    gemma_rmsnorm._weight_loader = compatible_weight_loader
    return _component_record(
        name="gemma-rmsnorm-derived-weight-colocation",
        target=GEMMA_RMSNORM_WEIGHT_LOADER_TARGET,
        scope="model-weight-load-only",
        disposition="installed",
    )


def install_gemma_rmsnorm_offload_compatibility() -> dict[str, object]:
    """Install the load-time-only fix equally for baseline and PyPTO lanes."""

    from sglang.srt.layers.layernorm import GemmaRMSNorm

    return _install_on_class(GemmaRMSNorm)


def _repair_qwen_gdn_view_alias(
    module: object, replacements: dict[object, object]
) -> object | None:
    """Temporarily bind RadixLinearAttention's conv view to the replacement weight."""

    linear_attn = getattr(module, "linear_attn", None)
    attn = getattr(linear_attn, "attn", None)
    if attn is None or not hasattr(attn, "conv_weights"):
        return None
    replacement = replacements.get("linear_attn.conv1d.weight")
    if replacement is None:
        return None
    current_view = getattr(attn, "conv_weights")
    shape = getattr(replacement, "shape", None)
    view_shape = getattr(current_view, "shape", None)
    if shape is None or view_shape is None or len(shape) != 3 or len(view_shape) != 2:
        raise ReleaseContractError(
            "Qwen GDN offload view contract changed: expected conv1d [O,1,K]"
        )
    if int(shape[0]) != int(view_shape[0]) or int(shape[2]) != int(view_shape[1]):
        raise ReleaseContractError(
            "Qwen GDN offload replacement shape differs from conv view"
        )
    if not hasattr(replacement, "view"):
        raise ReleaseContractError("Qwen GDN offload replacement cannot form a view")
    setattr(attn, "conv_weights", replacement.view(int(shape[0]), int(shape[2])))
    return current_view


def _install_offloader_functional_call(offloader_module: object) -> dict[str, object]:
    current = getattr(offloader_module, "functional_call", None)
    if not callable(current):
        raise ReleaseContractError(
            "pinned SGLang offloader functional_call contract changed"
        )
    if getattr(current, _VIEW_MARKER, False):
        return _component_record(
            name="offloader-explicit-parameter-aliases-and-views",
            target=OFFLOADER_FUNCTIONAL_CALL_TARGET,
            scope="cpu-offloaded-module-forward",
            disposition="already-installed",
        )
    try:
        parameters = inspect.signature(current).parameters
    except (TypeError, ValueError) as error:
        raise ReleaseContractError(
            "pinned SGLang offloader functional_call is not inspectable"
        ) from error
    if "tie_weights" not in parameters:
        raise ReleaseContractError(
            "pinned SGLang offloader functional_call lacks tie_weights"
        )

    @functools.wraps(current)
    def functional_call_with_explicit_aliases(*args, **kwargs):
        if "tie_weights" in kwargs and kwargs["tie_weights"] is not False:
            raise ReleaseContractError(
                "Qwen offload requires functional_call tie_weights=False"
            )
        kwargs["tie_weights"] = False
        restore_view = None
        if len(args) >= 2 and isinstance(args[1], dict):
            module = args[0]
            replacements = args[1]
            try:
                named_parameters = dict(
                    module.named_parameters(remove_duplicate=False)
                )
            except (AttributeError, TypeError) as error:
                raise ReleaseContractError(
                    "pinned SGLang offloader module parameter contract changed"
                ) from error
            for name, replacement in replacements.items():
                original = named_parameters.get(name)
                if original is None or not hasattr(replacement, "shape"):
                    continue
                replacement._pypto_offload_source_signature = (
                    "offloaded",
                    int(original.data_ptr()),
                    int(original._version),
                    tuple(original.shape),
                )
            restore_view = _repair_qwen_gdn_view_alias(module, replacements)
        try:
            return current(*args, **kwargs)
        finally:
            if restore_view is not None:
                setattr(module.linear_attn.attn, "conv_weights", restore_view)

    setattr(functional_call_with_explicit_aliases, _OFFLOADER_MARKER, True)
    setattr(functional_call_with_explicit_aliases, _VIEW_MARKER, True)
    offloader_module.functional_call = functional_call_with_explicit_aliases
    return _component_record(
        name="offloader-explicit-parameter-aliases-and-views",
        target=OFFLOADER_FUNCTIONAL_CALL_TARGET,
        scope="cpu-offloaded-module-forward",
        disposition="installed",
    )


def install_sglang_release_compatibility() -> dict[str, object]:
    """Install every backend-neutral compatibility fix for the pinned runtime."""

    from sglang.srt.utils import offloader

    components = [
        install_gemma_rmsnorm_offload_compatibility(),
        _install_offloader_functional_call(offloader),
    ]
    return {
        "name": "pinned-sglang-shared-release-compatibility",
        "applies_equally_to_lanes": ["pypto", "sglang-matched", "sglang-optimized"],
        "components": components,
        "all_installed": all(component["installed"] for component in components),
        "performance_claim_scope": (
            "shared correctness compatibility; no component is credited to PyPTO"
        ),
    }


def _component_record(
    *, name: str, target: str, scope: str, disposition: str
) -> dict[str, object]:
    return {
        "name": name,
        "target": target,
        "scope": scope,
        "installed": True,
        "disposition": disposition,
    }
