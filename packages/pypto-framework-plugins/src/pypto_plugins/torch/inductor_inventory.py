"""Pinned TorchInductor extension-surface inventory; imports no Torch modules."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class SourceSpec:
    name: str
    relative_path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class MethodSpec:
    name: str
    arguments: tuple[str, ...]
    signature: str | None = None
    decorators: tuple[str, ...] | None = None
    is_async: bool | None = None


@dataclass(frozen=True, slots=True)
class FieldSpec:
    name: str
    annotation: str
    default: str | None


@dataclass(frozen=True, slots=True)
class SymbolSpec:
    source: str
    symbol: str
    kind: str
    arguments: tuple[str, ...] = ()
    methods: tuple[MethodSpec, ...] = ()
    fields: tuple[FieldSpec, ...] = ()
    signature: str | None = None
    decorators: tuple[str, ...] | None = None
    is_async: bool | None = None


@dataclass(frozen=True, slots=True)
class InductorInventoryAudit:
    sources: tuple[tuple[str, str], ...]
    symbols: tuple[tuple[str, str, str], ...]
    capabilities: tuple[tuple[str, object], ...]
    scope_digest: str


INDUCTOR_ADAPTER_SCOPE_V1 = MappingProxyType(
    {
        "registration_ready": False,
        "pytorch_source_diff_required": False,
        "device_registry_dispatch": True,
        "atomic_registry_install_implemented": False,
        "cuda_backend_patch": "pypto",
        "tilekernel_present": False,
        "cutile_present": False,
        "usable_triton_install_required": True,
        "exact_triton_source_audit_complete": False,
        "triton_compute_allowed": False,
        "python_wrapper_in_scope": True,
        "python_wrapper_implemented": False,
        "python_subgraph_wrapper_implemented": False,
        "cseproxy_pypto_dtype_shape_implemented": False,
        "strict_lowering_choice_filter_implemented": False,
        "cpp_wrapper_supported": False,
        "fx_wrapper_supported": False,
        "foreach_supported": False,
        "extern_compute_supported": False,
        "multi_template_supported": False,
        "subprocess_compile_supported": False,
        "async_compile_owner": "pypto-plugin",
        "template_strategy": "generic-TemplateBuffer-ChoiceCaller",
        "implementation_order": (
            "single-dso",
            "target-info",
            "compile-request-current-stream-artifact",
            "operator-executables",
            "exact-torch-triton-source-audit",
            "pypto-scheduling-wrapper-template",
            "atomic-device-registry-dispatch",
            "strict-compile",
        ),
    }
)


INDUCTOR_PYTHON_MANIFEST_COUNT = 346
INDUCTOR_PYTHON_MANIFEST_SHA256 = (
    "3bef71727acfceb4c1dbc1f433ac26e21c65ff715435feefc80cc32d4cb88cd6"
)


INDUCTOR_SOURCE_SPECS = (
    SourceSpec("common", "torch/_inductor/codegen/common.py",
               "bb56bdbbaf16a5c9cc9eaceac6237e53187d9994cdaf860ffd1f0be8f9b94496"),
    SourceSpec("config", "torch/_inductor/config.py",
               "4cdbe0025fd4890e2dfaa6395e57a08971c33889265cbcd26b33a1cf9ee1fa7f"),
    SourceSpec("scheduler", "torch/_inductor/scheduler.py",
               "829becc15ce05534a1ee4fefd2319cd5d5a5bf4483cc51827b83556a39bc5aa3"),
    SourceSpec("simd", "torch/_inductor/codegen/simd.py",
               "643c93b29c9abf5e6a8a5a626dd07122f083199116d494b9860e1e9536f9bb1b"),
    SourceSpec("cuda_combined", "torch/_inductor/codegen/cuda_combined_scheduling.py",
               "1438a502671eb173cda7aadfe0d3aa4efdd71db88924688bbf6db7937040746f"),
    SourceSpec("triton", "torch/_inductor/codegen/triton.py",
               "76db8896b4ddac153d7cc7ea7a97b3974da01c110a7f56c8be234281545617e8"),
    SourceSpec("wrapper", "torch/_inductor/codegen/wrapper.py",
               "f257762a74660e19f0b75e3a4bbc41cfd15afdb91226dec8fd49f75691c357b6"),
    SourceSpec("graph", "torch/_inductor/graph.py",
               "c440d8ae5e9c048b887e80b9d0df226843db9f1cefaa9ef76eada6fd2946b8ad"),
    SourceSpec("async_compile", "torch/_inductor/async_compile.py",
               "401933b518592b8958bb137ddd963c8dd623f4c63b9de215c933424ccc296c7d"),
    SourceSpec("ir", "torch/_inductor/ir.py",
               "82b628840a9de70f7d09e9c4ac0bdaa6a6ff085bf1f705b9e5d9349da0355d1d"),
    SourceSpec("select_algorithm", "torch/_inductor/select_algorithm.py",
               "2e6d3297a9ee51c2d33a7becd3e656a5641ce0effae4c53ae58a87946e5b66db"),
    SourceSpec("kernel_template_choice", "torch/_inductor/kernel_template_choice.py",
               "e113350a7fdacc58645d57e86744949e94f887181123cdac01739ad6eacbbb3d"),
    SourceSpec("choices", "torch/_inductor/choices.py",
               "e501f7bebb99812cb9e100097a011d237754338b2b2577891b077f0ad65f0556"),
    SourceSpec("virtualized", "torch/_inductor/virtualized.py",
               "0e8676238801ec9f24352b05c9be775df5de2d384ef64e86b3171c1c7b7bfd8b"),
    SourceSpec("utils", "torch/_inductor/utils.py",
               "c1aedd001340ec0410c251a3bb7176d96085b04fbd25f58d740bd642c73d4537"),
    SourceSpec("config_module", "torch/utils/_config_module.py",
               "19005fcb96c23ce7028bfb7e327da1346e63c1b22ac90665d5ae1a2d5f00f76c"),
    SourceSpec("compile_fx", "torch/_inductor/compile_fx.py",
               "a9b68fb1e93a95d8ea01f12c53fd65232007679ff14a51d52814e5d85c1c5b67"),
    SourceSpec("mm_common", "torch/_inductor/kernel/mm_common.py",
               "d0856b3c7be6b83f882f91d8508473b5d73ec580aac87e961a078702f91dcb18"),
    SourceSpec("post_grad", "torch/_inductor/fx_passes/post_grad.py",
               "a1779017efc75841fc03a9561f902ec2a25e8a6de709356f1a9700a077ec803a"),
    SourceSpec(
        "codecache",
        "torch/_inductor/codecache.py",
        "1de34a274c58a2b826a6c46829e94f1df5b24a3d83841a6c4ff8a691a50d8c59",
    ),
    SourceSpec(
        "cuda_device_overrides",
        "torch/_inductor/codegen/cuda/device_op_overrides.py",
        "734ec4f3a94844d8771bf68aa18e002a9ff543b4106d499c4ec78c73af3fb632",
    ),
    SourceSpec(
        "stream_utils",
        "torch/_inductor/stream_utils.py",
        "e0e54879098f103de5029538881adfd313cae5b6ce22d2b3b81e4bedd9250ec8",
    ),
    SourceSpec(
        "lowering",
        "torch/_inductor/lowering.py",
        "d969efbd0db1bde6a7f6c0b58a8549de1654c97df56b772b41d08d11f8a563fe",
    ),
    SourceSpec(
        "mm",
        "torch/_inductor/kernel/mm.py",
        "d92c98a688aca11f88852ecd125be1e6b00056e58d3ce87d9692d7da36871783",
    ),
    SourceSpec(
        "bmm",
        "torch/_inductor/kernel/bmm.py",
        "87fd9d816bdaf1252422f7690a1e4f7a7165331ebc065eecad10734dcb242f7d",
    ),
    SourceSpec(
        "mm_plus_mm",
        "torch/_inductor/kernel/mm_plus_mm.py",
        "938f0980dba8f9de59329fe41244864bf18c361ab7ea5049a4af17f43ae77b83",
    ),
    SourceSpec(
        "autotune_process",
        "torch/_inductor/autotune_process.py",
        "5c764d3d5bf814e058c2fdd4b5699e7599252a5c6b45f8d7c9b86aa816c40cef",
    ),
    SourceSpec(
        "cutedsl_scheduling",
        "torch/_inductor/codegen/cutedsl/cutedsl_scheduling.py",
        "13fdb9fc2aaaa70af7818bad6d2987c5997d051abadb069b4299e2966b6aef5b",
    ),
    SourceSpec(
        "cutedsl_kernel",
        "torch/_inductor/codegen/cutedsl/cutedsl_kernel.py",
        "adffef2c0524a9efa2badf2c826997235e243e4fa9e6164cf3ef8ea1b510dac8",
    ),
    SourceSpec(
        "cutedsl_template",
        "torch/_inductor/codegen/cutedsl/cutedsl_template.py",
        "44309bf98dd4e51d186939e76f6ea8f3b245ad77dc0a1d295844a43017763d42",
    ),
    SourceSpec(
        "torch_triton_utils",
        "torch/utils/_triton.py",
        "ca6b84b5751d68afef137a77fbb53f0c947f363e158518d63851032ea298bac4",
    ),
)


INDUCTOR_SYMBOL_SPECS = (
    SymbolSpec(
        "common",
        "DeviceCodegen",
        "class",
        fields=(
            FieldSpec("scheduling", "SchedulingConstructor", None),
            FieldSpec("wrapper_codegen", "WrapperConstructor", None),
            FieldSpec(
                "cpp_wrapper_codegen", "WrapperConstructor | None", "None"
            ),
            FieldSpec("fx_wrapper_codegen", "WrapperConstructor | None", "None"),
        ),
    ),
    SymbolSpec(
        "common",
        "register_backend_for_device",
        "function",
        (
            "device",
            "device_scheduling",
            "device_wrapper_codegen",
            "device_cpp_wrapper_codegen",
            "device_fx_wrapper_codegen",
            "device_custom_pass",
            "device_custom_config",
        ),
        signature=(
            "device: str, device_scheduling: SchedulingConstructor, "
            "device_wrapper_codegen: WrapperConstructor, "
            "device_cpp_wrapper_codegen: WrapperConstructor | None=None, "
            "device_fx_wrapper_codegen: WrapperConstructor | None=None, "
            "device_custom_pass: CustomGraphModulePass | None=None, "
            "device_custom_config: ConfigModule | None=None"
        ),
    ),
    SymbolSpec("common", "init_backend_registration", "function", ()),
    SymbolSpec("common", "get_scheduling_for_device", "function", ("device",)),
    SymbolSpec("common", "KernelTemplate", "class"),
    SymbolSpec("scheduler", "BaseSchedulerNode", "class"),
    SymbolSpec("scheduler", "SchedulerNode", "class"),
    SymbolSpec("scheduler", "FusedSchedulerNode", "class"),
    SymbolSpec("scheduler", "ExternKernelSchedulerNode", "class"),
    SymbolSpec("scheduler", "BaseScheduling", "class", methods=(
        MethodSpec("can_fuse_vertical", ("self", "node1", "node2")),
        MethodSpec("can_fuse_horizontal", ("self", "node1", "node2")),
        MethodSpec("group_fn", ("self", "sizes")),
        MethodSpec("codegen_template", ("self", "template_node",
                   "epilogue_nodes", "prologue_nodes")),
        MethodSpec("generate_kernel_code_from_nodes",
                   ("self", "nodes", "benchmark_kernel", "hint_override")),
        MethodSpec("codegen_node", ("self", "node")),
        MethodSpec("codegen_sync", ("self",)),
        MethodSpec("flush", ("self",)),
        MethodSpec("benchmark_fused_nodes", ("self", "nodes")),
        MethodSpec("benchmark_codegened_module", ("self", "module")),
    )),
    SymbolSpec("scheduler", "Scheduler", "class", methods=(
        MethodSpec("create_backend", ("self", "device")),
        MethodSpec("get_backend", ("self", "device")),
        MethodSpec("flush", ("self",)),
        MethodSpec("codegen", ("self",)),
    )),
    SymbolSpec("simd", "SIMDKernel", "class"),
    SymbolSpec("simd", "SIMDScheduling", "class", methods=(
        MethodSpec("group_fn", ("self", "sizes")),
        MethodSpec("codegen_node", ("self", "node")),
        MethodSpec("codegen_template", ("self", "template_node", "epilogue_nodes",
                   "prologue_nodes", "only_gen_src_code", "hint_override")),
        MethodSpec("generate_kernel_code_from_nodes",
                   ("self", "nodes", "benchmark_kernel", "hint_override")),
        MethodSpec("codegen_sync", ("self",)),
        MethodSpec("flush", ("self",)),
    )),
    SymbolSpec("cuda_combined", "CUDACombinedScheduling", "class", methods=(
        MethodSpec("choose_node_backend", ("self", "node")),
        MethodSpec("codegen_node", ("self", "node")),
        MethodSpec("codegen_template", ("self", "template_node",
                   "epilogue_nodes", "prologue_nodes")),
        MethodSpec("generate_kernel_code_from_nodes",
                   ("self", "nodes", "benchmark_kernel", "hint_override")),
    )),
    SymbolSpec("triton", "TritonScheduling", "class", methods=(
        MethodSpec("define_kernel", ("self", "src_code", "node_schedule", "kernel")),
        MethodSpec("benchmark_fused_nodes", ("self", "nodes", "n_spills_threshold")),
    )),
    SymbolSpec("wrapper", "PythonWrapperCodegen", "class", methods=(
        MethodSpec(
            "create",
            (
                "is_subgraph",
                "subgraph_name",
                "parent_wrapper",
                "partition_signatures",
            ),
            signature=(
                "is_subgraph: bool, subgraph_name: str | None, "
                "parent_wrapper: PythonWrapperCodegen | None, "
                "partition_signatures: ir.GraphPartitionSignature | None=None"
            ),
            decorators=("staticmethod",),
            is_async=False,
        ),
        MethodSpec("write_get_raw_stream", ("self", "device_idx", "graph_name")),
        MethodSpec("write_async_compile_wait", ("self",)),
        MethodSpec("define_kernel", ("self", "kernel_name",
                   "kernel_body", "metadata", "gpu", "cpp_definition")),
        MethodSpec(
            "generate_kernel_call",
            (
                "self",
                "kernel_name",
                "call_args",
                "device",
                "triton",
                "arg_types",
                "raw_keys",
                "raw_args",
                "triton_meta",
                "inductor_meta",
                "original_fxnode_name",
            ),
            signature=(
                "self, kernel_name: str, call_args, *, device=None, triton=True, "
                "arg_types=None, raw_keys=None, raw_args=None, triton_meta=None, "
                "inductor_meta=None, original_fxnode_name=None"
            ),
            decorators=(),
            is_async=False,
        ),
        MethodSpec("generate", ("self", "is_inference")),
    )),
    SymbolSpec("graph", "GraphLowering", "class", methods=(
        MethodSpec("init_wrapper_code", ("self", "is_subgraph",
                   "subgraph_name", "parent_wrapper_code", "partition_signatures")),
        MethodSpec("codegen", ("self",)),
    )),
    SymbolSpec("async_compile", "AsyncCompile", "class", methods=(
        MethodSpec("submit", ("cls", "task")),
        MethodSpec(
            "triton",
            ("self", "kernel_name", "source_code", "device_str"),
            signature=(
                "self, kernel_name: str, source_code: str, device_str: str='cuda'"
            ),
            decorators=(),
            is_async=False,
        ),
        MethodSpec(
            "cuda",
            ("self", "source_code", "dst_file_ext", "aot_compile"),
            signature="self, source_code, dst_file_ext, aot_compile=False",
            decorators=(),
            is_async=False,
        ),
        MethodSpec("wait", ("self", "scope")),
    )),
    SymbolSpec("ir", "TemplateBuffer", "class"),
    SymbolSpec("ir", "ChoiceCaller", "class"),
    SymbolSpec("select_algorithm", "TritonTemplate", "class"),
    SymbolSpec("select_algorithm", "TritonTemplateCaller", "class"),
    SymbolSpec("select_algorithm", "ExternKernelChoice", "class"),
    SymbolSpec("select_algorithm", "AlgorithmSelectorCache", "class"),
    SymbolSpec("kernel_template_choice", "KernelTemplateChoice", "class"),
    SymbolSpec("choices", "InductorChoices", "class"),
    SymbolSpec("config_module", "ConfigModule", "class", methods=(
        MethodSpec("__setattr__", ("self", "name", "value")),
        MethodSpec(
            "patch",
            ("self", "arg1", "arg2"),
            signature=(
                "self, arg1: str | dict[str, Any] | None=None, arg2: Any=None, "
                "**kwargs: dict[str, Any]"
            ),
            decorators=(),
            is_async=False,
        ),
        MethodSpec("get_hash", ("self",)),
    )),
    SymbolSpec(
        "codecache",
        "FxGraphHashDetails",
        "class",
        methods=(
            MethodSpec(
                "__init__",
                ("self", "gm", "example_inputs", "fx_kwargs", "inputs_to_check"),
            ),
        ),
    ),
    SymbolSpec(
        "cuda_device_overrides",
        "CUDADeviceOpOverrides",
        "class",
        methods=(
            MethodSpec("import_get_raw_stream_as", ("self", "name")),
            MethodSpec("current_stream", ("self",)),
        ),
    ),
    SymbolSpec(
        "stream_utils",
        "get_raw_stream_name",
        "function",
        ("device_idx",),
        signature="device_idx: int",
        decorators=("functools.lru_cache",),
        is_async=False,
    ),
    SymbolSpec(
        "lowering",
        "fallback_handler",
        "function",
        ("kernel", "add_to_fallback_set"),
        signature="kernel, add_to_fallback_set=True",
    ),
    SymbolSpec(
        "lowering",
        "make_fallback",
        "function",
        ("op", "layout_constraint", "warn", "override_decomp", "get_decomp_fn"),
        signature=(
            "op, layout_constraint=None, warn=True, override_decomp=False, "
            "get_decomp_fn=None"
        ),
    ),
    SymbolSpec(
        "mm",
        "tuned_mm",
        "function",
        ("mat1", "mat2", "out_dtype", "layout"),
        signature="mat1, mat2, out_dtype=None, *, layout=None",
    ),
    SymbolSpec(
        "mm",
        "tuned_addmm",
        "function",
        ("inp", "mat1", "mat2", "alpha", "beta", "layout"),
        signature="inp, mat1, mat2, *, alpha=1, beta=1, layout=None",
    ),
    SymbolSpec(
        "bmm",
        "tuned_bmm",
        "function",
        ("mat1", "mat2", "out_dtype", "layout"),
        signature="mat1, mat2, out_dtype=None, *, layout=None",
    ),
    SymbolSpec(
        "bmm",
        "tuned_baddbmm",
        "function",
        ("inp", "mat1", "mat2", "alpha", "beta", "layout"),
        signature="inp, mat1, mat2, *, alpha=1, beta=1, layout=None",
    ),
    SymbolSpec(
        "mm_plus_mm",
        "tuned_mm_plus_mm",
        "function",
        ("mat1", "mat2", "mat3", "mat4", "layout"),
        signature="mat1, mat2, mat3, mat4, *, layout=None",
    ),
    SymbolSpec("autotune_process", "TuningProcessPool", "class"),
    SymbolSpec(
        "autotune_process",
        "run_autotune_in_subprocess",
        "function",
        ("benchmark_request",),
        signature="benchmark_request: BenchmarkRequest",
    ),
    SymbolSpec("cutedsl_scheduling", "CuteDSLScheduling", "class"),
    SymbolSpec("cutedsl_kernel", "CuteDSLTemplateKernel", "class"),
    SymbolSpec("cutedsl_template", "CuteDSLTemplate", "class"),
    SymbolSpec("cutedsl_template", "CuteDSLTemplateCaller", "class"),
    SymbolSpec(
        "torch_triton_utils",
        "has_triton",
        "function",
        (),
        signature="",
        decorators=("functools.cache",),
        is_async=False,
    ),
)


def _arguments(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    return tuple(
        value.arg
        for value in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        )
    )


def _scope_payload(value: Mapping[str, object]) -> dict[str, object]:
    payload = dict(value)
    expected = dict(INDUCTOR_ADAPTER_SCOPE_V1)
    if payload != expected:
        raise RuntimeError(
            f"Inductor adapter scope mismatch: expected={expected}, got={payload}")
    return payload


def validate_inductor_adapter_scope(value: Mapping[str, object]) -> str:
    payload = _scope_payload(value)
    encoded = json.dumps(payload, allow_nan=False, ensure_ascii=True,
                         separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _audit_inductor_python_manifest(root: Path) -> tuple[int, str]:
    source_root = root / "torch/_inductor"
    paths = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix in {".py", ".pyi"}
    )
    records = [
        [
            path.relative_to(source_root).as_posix(),
            hashlib.sha256(path.read_bytes()).hexdigest(),
        ]
        for path in paths
    ]
    encoded = json.dumps(
        records,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    observed = (len(records), digest)
    expected = (INDUCTOR_PYTHON_MANIFEST_COUNT, INDUCTOR_PYTHON_MANIFEST_SHA256)
    if observed != expected:
        raise RuntimeError(
            f"pinned Inductor Python manifest mismatch: expected={expected}, "
            f"got={observed}"
        )
    return observed


def _find_cuda_backend(tree: ast.Module) -> tuple[tuple[str, ...], str]:
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "cuda_backend"
        ):
            values = tuple(
                value.value
                for value in ast.walk(node.annotation)
                if isinstance(value, ast.Constant) and isinstance(value.value, str)
            )
            default = node.value.value if isinstance(node.value, ast.Constant) else None
            return values, default
    raise RuntimeError("pinned config.cuda_backend declaration is absent")


def _cuda_backend_dict_keys(tree: ast.Module) -> tuple[str, ...]:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "cuda_backends"
                for target in node.targets
            )
            and isinstance(node.value, ast.Dict)
        ):
            if not all(
                isinstance(key, ast.Constant) and isinstance(key.value, str)
                for key in node.value.keys
            ):
                raise RuntimeError(
                    "pinned CUDA backend closure contains a dynamic or unpacked key"
                )
            return tuple(key.value for key in node.value.keys)
    raise RuntimeError("pinned CUDA backend closure is absent")


def _method_map(
    node: ast.ClassDef,
) -> dict[str, tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...]]:
    collected: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    for method in node.body:
        if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        collected.setdefault(method.name, []).append(method)
    return {name: tuple(methods) for name, methods in collected.items()}


def _decorators(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, ...]:
    return tuple(ast.unparse(decorator) for decorator in node.decorator_list)


def _validate_callable(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    owner: str,
    arguments: tuple[str, ...],
    signature: str | None,
    decorators: tuple[str, ...] | None,
    is_async: bool | None,
) -> None:
    if _arguments(node) != arguments:
        raise RuntimeError(
            f"Inductor callable arguments mismatch for {owner}: "
            f"expected {arguments}, got {_arguments(node)}"
        )
    observed_signature = ast.unparse(node.args)
    if signature is not None and observed_signature != signature:
        raise RuntimeError(
            f"Inductor callable signature mismatch for {owner}: "
            f"expected {signature!r}, got {observed_signature!r}"
        )
    observed_decorators = _decorators(node)
    if decorators is not None and observed_decorators != decorators:
        raise RuntimeError(
            f"Inductor callable decorators mismatch for {owner}: "
            f"expected {decorators}, got {observed_decorators}"
        )
    observed_async = isinstance(node, ast.AsyncFunctionDef)
    if is_async is not None and observed_async is not is_async:
        raise RuntimeError(
            f"Inductor callable async mismatch for {owner}: "
            f"expected {is_async}, got {observed_async}"
        )


def _class_fields(node: ast.ClassDef) -> tuple[FieldSpec, ...]:
    return tuple(
        FieldSpec(
            field.target.id,
            ast.unparse(field.annotation),
            ast.unparse(field.value) if field.value is not None else None,
        )
        for field in node.body
        if isinstance(field, ast.AnnAssign)
        and isinstance(field.target, ast.Name)
        and field.simple == 1
    )


def _class_method(
    tree: ast.Module, class_name: str, method_name: str
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(classes) != 1:
        raise RuntimeError(f"required class {class_name} is absent/ambiguous")
    methods = _method_map(classes[0])
    candidates = methods.get(method_name, ())
    if len(candidates) != 1:
        raise RuntimeError(
            f"required method {class_name}.{method_name} is absent/ambiguous"
        )
    return candidates[0]


def _codecache_registry_sequence(tree: ast.Module) -> tuple[str, ...]:
    method = _class_method(tree, "FxGraphHashDetails", "__init__")
    entries: list[tuple[int, str]] = []
    for index, statement in enumerate(method.body):
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "init_backend_registration"
            for node in ast.walk(statement)
        ):
            entries.append((index, "init_backend_registration"))
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else (statement.target,)
            )
            value = statement.value
            for target in targets:
                if not (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    continue
                if target.attr not in {
                    "custom_backend_passes",
                    "custom_backend_codegen_configs",
                }:
                    continue
                if not any(
                    isinstance(node, ast.Name) and node.id == target.attr
                    for node in ast.walk(value)
                ):
                    raise RuntimeError(
                        f"FxGraphHashDetails.{target.attr} no longer consumes the "
                        "matching backend registry"
                    )
                entries.append((index, target.attr))
    sequence = tuple(name for _, name in sorted(entries))
    expected = (
        "init_backend_registration",
        "custom_backend_passes",
        "custom_backend_codegen_configs",
    )
    if sequence != expected:
        raise RuntimeError(
            f"codecache backend registry sequence mismatch: expected={expected}, "
            f"got={sequence}"
        )
    return sequence


def _named_call_sites(
    trees: Mapping[str, ast.Module], name: str
) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(
        (source, node.lineno)
        for source, tree in trees.items()
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ))


def _class_string_tuple_membership(
    tree: ast.Module, class_name: str, method_name: str, subject: str
) -> tuple[str, ...]:
    method = _class_method(tree, class_name, method_name)
    candidates = []
    for node in ast.walk(method):
        if not (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Name)
            and node.left.id == subject
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.In)
            and len(node.comparators) == 1
            and isinstance(node.comparators[0], ast.Tuple)
        ):
            continue
        values = tuple(
            element.value
            for element in node.comparators[0].elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        )
        if len(values) == len(node.comparators[0].elts):
            candidates.append(values)
    if not candidates:
        raise RuntimeError(
            f"{class_name}.{method_name} has no static {subject} membership tuple"
        )
    return max(candidates, key=len)


def _scheduler_extern_bypasses_backend(tree: ast.Module) -> bool:
    method = _class_method(tree, "Scheduler", "_codegen")
    for node in ast.walk(method):
        if not isinstance(node, ast.If):
            continue
        tests_extern = any(
            isinstance(candidate, ast.Call)
            and isinstance(candidate.func, ast.Attribute)
            and isinstance(candidate.func.value, ast.Name)
            and candidate.func.value.id == "node"
            and candidate.func.attr == "is_extern"
            for candidate in ast.walk(node.test)
        )
        calls_extern_codegen = any(
            isinstance(candidate, ast.Call)
            and isinstance(candidate.func, ast.Attribute)
            and isinstance(candidate.func.value, ast.Name)
            and candidate.func.value.id == "self"
            and candidate.func.attr == "codegen_extern_call"
            for statement in node.body
            for candidate in ast.walk(statement)
        )
        if tests_extern and calls_extern_codegen:
            return True
    return False


def _scheduler_foreach_backend_types(tree: ast.Module) -> tuple[str, ...]:
    method = _class_method(tree, "Scheduler", "_codegen")
    for node in ast.walk(method):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "isinstance"
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "backend_"
            and isinstance(node.args[1], ast.Tuple)
        ):
            continue
        names = tuple(
            element.id
            for element in node.args[1].elts
            if isinstance(element, ast.Name)
        )
        if len(names) == len(node.args[1].elts):
            return names
    raise RuntimeError("Scheduler._codegen foreach backend whitelist is absent")


def _scheduler_multi_template_extern_fallback(tree: ast.Module) -> bool:
    method = _class_method(tree, "Scheduler", "finalize_multi_template_buffers")
    return any(
        isinstance(node, ast.Attribute) and node.attr == "ExternKernelCaller"
        for node in ast.walk(method)
    )


def _module_call_count(tree: ast.Module, name: str) -> int:
    return sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
        for node in ast.walk(tree)
    )


def _derive_inductor_capabilities(
    root: Path,
    trees: Mapping[str, ast.Module],
    texts: Mapping[str, str],
) -> tuple[tuple[str, object], ...]:
    inductor_sources = tuple(
        path
        for path in (root / "torch/_inductor").rglob("*")
        if path.is_file() and path.suffix in {".py", ".pyi"}
    )
    inductor_texts = tuple(
        (path, path.read_text(errors="strict")) for path in inductor_sources
    )
    cuda_values, cuda_default = _find_cuda_backend(trees["config"])
    cuda_dict_keys = _cuda_backend_dict_keys(trees["common"])
    scheduler_create = _class_method(trees["scheduler"], "Scheduler", "create_backend")
    scheduler_requires_triton = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "has_triton"
        for node in ast.walk(scheduler_create)
    )
    get_current_backend_occurrences = sum(
        text.count("get_current_backend") for text in texts.values()
    )
    get_current_backend_call_sites = _named_call_sites(
        trees, "get_current_backend"
    )
    config_cuda_backend_occurrences = sum(
        text.count("config.cuda_backend") for text in texts.values()
    )
    return (
        (
            "tilekernel_present",
            any(
                "class TileKernel" in text or "class TileKernelScheduling" in text
                for _, text in inductor_texts
            ),
        ),
        (
            "cutile_present",
            any(
                "cutile" in path.as_posix().lower()
                or "cutile" in text.lower()
                or "cu_tile" in text.lower()
                for path, text in inductor_texts
            ),
        ),
        (
            "cutedsl_present",
            any("cutedsl" in path.as_posix().lower() for path, _ in inductor_texts),
        ),
        ("cuda_backend_values", cuda_values),
        ("cuda_backend_default", cuda_default),
        ("cuda_registry_closed_keys", cuda_dict_keys),
        ("scheduler_requires_has_triton", scheduler_requires_triton),
        (
            "async_compile_has_pypto",
            any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "pypto"
                for node in ast.walk(trees["async_compile"])
            ),
        ),
        ("get_current_backend_occurrences", get_current_backend_occurrences),
        ("get_current_backend_call_sites", get_current_backend_call_sites),
        ("config_cuda_backend_occurrences", config_cuda_backend_occurrences),
        (
            "codecache_backend_registry_sequence",
            _codecache_registry_sequence(trees["codecache"]),
        ),
        (
            "cseproxy_dtype_shape_backends",
            _class_string_tuple_membership(
                trees["common"], "CSEProxy", "_default", "backend"
            ),
        ),
        (
            "scheduler_extern_bypasses_backend",
            _scheduler_extern_bypasses_backend(trees["scheduler"]),
        ),
        (
            "scheduler_foreach_backend_types",
            _scheduler_foreach_backend_types(trees["scheduler"]),
        ),
        (
            "scheduler_multi_template_extern_fallback",
            _scheduler_multi_template_extern_fallback(trees["scheduler"]),
        ),
        (
            "explicit_extern_choice_counts",
            tuple(
                (source, _module_call_count(trees[source], "ExternKernelChoice"))
                for source in ("mm", "bmm", "mm_plus_mm")
            ),
        ),
        (
            "autotune_subprocess_present",
            any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "run_autotune_in_subprocess"
                for node in trees["autotune_process"].body
            ),
        ),
    )


def audit_inductor_inventory(torch_root: str | Path) -> InductorInventoryAudit:
    root = Path(torch_root).resolve()
    trees: dict[str, ast.Module] = {}
    texts: dict[str, str] = {}
    source_results: list[tuple[str, str]] = []
    for spec in INDUCTOR_SOURCE_SPECS:
        path = root / spec.relative_path
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != spec.sha256:
            raise RuntimeError(
                f"TorchInductor source fingerprint mismatch for {spec.name}: "
                f"expected {spec.sha256}, got {digest}"
            )
        trees[spec.name] = ast.parse(data, filename=str(path))
        texts[spec.name] = data.decode()
        source_results.append((spec.name, digest))

    manifest_count, manifest_digest = _audit_inductor_python_manifest(root)

    symbol_results: list[tuple[str, str, str]] = []
    for spec in INDUCTOR_SYMBOL_SPECS:
        candidates = [
            node
            for node in trees[spec.source].body
            if (
                spec.kind == "class"
                and isinstance(node, ast.ClassDef)
                and node.name == spec.symbol
            )
            or (
                spec.kind == "function"
                and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == spec.symbol
            )
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"required Inductor symbol {spec.source}:{spec.symbol} is absent/ambiguous"
            )
        node = candidates[0]
        if spec.kind == "function":
            assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            _validate_callable(
                node,
                owner=spec.symbol,
                arguments=spec.arguments,
                signature=spec.signature,
                decorators=spec.decorators,
                is_async=spec.is_async,
            )
        if isinstance(node, ast.ClassDef):
            methods = _method_map(node)
            for method in spec.methods:
                method_candidates = methods.get(method.name, ())
                if len(method_candidates) != 1:
                    raise RuntimeError(
                        f"required Inductor method {spec.symbol}.{method.name} "
                        "is absent/ambiguous"
                    )
                method_node = method_candidates[0]
                _validate_callable(
                    method_node,
                    owner=f"{spec.symbol}.{method.name}",
                    arguments=method.arguments,
                    signature=method.signature,
                    decorators=method.decorators,
                    is_async=method.is_async,
                )
            if spec.fields and _class_fields(node) != spec.fields:
                raise RuntimeError(
                    f"Inductor class fields mismatch for {spec.symbol}: "
                    f"expected {spec.fields}, got {_class_fields(node)}"
                )
        symbol_results.append((spec.source, spec.symbol, spec.kind))

    capabilities = (
        *_derive_inductor_capabilities(root, trees, texts),
        ("inductor_python_manifest_count", manifest_count),
        ("inductor_python_manifest_sha256", manifest_digest),
    )
    expected = {
        "tilekernel_present": False,
        "cutile_present": False,
        "cuda_backend_values": ("triton", "halide", "pallas"),
        "cuda_backend_default": "triton",
        "cuda_registry_closed_keys": ("triton", "halide", "pallas"),
        "scheduler_requires_has_triton": True,
        "async_compile_has_pypto": False,
        "get_current_backend_occurrences": 12,
        "get_current_backend_call_sites": (
            ("common", 730),
            ("common", 739),
            ("common", 779),
            ("common", 2111),
            ("common", 2675),
            ("mm_common", 155),
            ("scheduler", 200),
            ("utils", 4133),
        ),
        "config_cuda_backend_occurrences": 2,
        "codecache_backend_registry_sequence": (
            "init_backend_registration",
            "custom_backend_passes",
            "custom_backend_codegen_configs",
        ),
        "cutedsl_present": True,
        "cseproxy_dtype_shape_backends": ("triton", "cpp", "mps"),
        "scheduler_extern_bypasses_backend": True,
        "scheduler_foreach_backend_types": (
            "SIMDScheduling",
            "CUDACombinedScheduling",
            "XPUCombinedScheduling",
        ),
        "scheduler_multi_template_extern_fallback": True,
        "explicit_extern_choice_counts": (
            ("mm", 8),
            ("bmm", 3),
            ("mm_plus_mm", 1),
        ),
        "autotune_subprocess_present": True,
        "inductor_python_manifest_count": INDUCTOR_PYTHON_MANIFEST_COUNT,
        "inductor_python_manifest_sha256": INDUCTOR_PYTHON_MANIFEST_SHA256,
    }
    if dict(capabilities) != expected:
        raise RuntimeError(
            f"pinned Inductor capability mismatch: expected={expected}, got={dict(capabilities)}"
        )
    return InductorInventoryAudit(
        tuple(source_results),
        tuple(symbol_results),
        capabilities,
        validate_inductor_adapter_scope(INDUCTOR_ADAPTER_SCOPE_V1),
    )


__all__ = (
    "FieldSpec",
    "INDUCTOR_ADAPTER_SCOPE_V1",
    "INDUCTOR_SOURCE_SPECS",
    "INDUCTOR_SYMBOL_SPECS",
    "InductorInventoryAudit",
    "MethodSpec",
    "SourceSpec",
    "SymbolSpec",
    "audit_inductor_inventory",
    "validate_inductor_adapter_scope",
)
