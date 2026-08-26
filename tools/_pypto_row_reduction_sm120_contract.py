#!/usr/bin/env python3
"""Pure contract for the bounded RowReductionV3 real-SM120 gate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


SMOKE_SCHEMA_VERSION = 1
SMOKE_NAME = "pypto-row-reduction-sm120"
GPU_SMOKE_POLICY_VERSION = 2
GPU_SMOKE_AUTHORIZATION = (
    "user-authorized-protected-zero-nvidia-gpu-smoke-host-floor-22gib-v2"
)
PROTECTED_GPU_SMOKE_MEMORY_FLOOR_KIB = 22 * 1024 * 1024
EXCLUSIVE_GPU_SMOKE_MEMORY_FLOOR_KIB = 32 * 1024 * 1024
OWNED_RUN_ABORT_MEMORY_FLOOR_KIB = 16 * 1024 * 1024
GPU_FREE_MEMORY_FLOOR_MIB = 4 * 1024
GPU_SMOKE_TIMEOUT_SECONDS = 1_800
GPU_SMOKE_MINIMUM_FREE_DISK_GIB = 64

PYPTO_HEAD = "62eb88251df5bdad95277a9d619d20da9bf121eb"
PYPTO_TREE = "04d3bca3e0b35b796f7745ded27a26dd61e25c67"
TENSOR_IR_HEAD = "1dcb38c20e53d07c97d3781cae538e33901bae30"
CUDA_TILE_HEAD = "af2417041cc939b87ef56d92cfdcf61737c5457e"
LLVM_HEAD = "57109befac92811d2253109242ca6fa69c961fb2"
PYPTO_DSO_RELATIVE_PATH = Path(
    "builds/pypto-row-reduction-v3-on-62eb882-final/product/"
    "pypto_core.cpython-314-x86_64-linux-gnu.so"
)
PYPTO_DSO_SIZE = 784_224_056
PYPTO_DSO_SHA256 = "e1213cf31972664a66012f95f1ebf003623dfebb54accdff3ab47cd6ca3e4220"

CUDA_RUNTIME_RELATIVE_PATH = Path(
    "envs/pypto-nvidia/lib/python3.14/site-packages/nvidia/cu13/lib/libcudart.so.13"
)
CUDA_RUNTIME_SIZE = 704_288
CUDA_RUNTIME_SHA256 = "96c42e418cec19054186b9429c321603cc190bf26a18104e19408117a2a817b0"
EXPECTED_DRIVER_RELEASE = "610.74"
EXPECTED_CUDA_TOOLKIT_VERSION = "13.3.73"
EXPECTED_TORCH_VERSION = "2.13.0+cu130"
EXPECTED_TORCH_GIT = "cf30153c4c131c8164ee7798e5022d810682e2cb"
EXPECTED_TORCH_CUDA = "13.0"
EXPECTED_DEVICE_NAME = "NVIDIA GeForce RTX 5090 Laptop GPU"
EXPECTED_COMPUTE_CAPABILITY = (12, 0)
EXPECTED_SM_COUNT = 82
EXPECTED_SUPPORTED_COMPUTE_DTYPES = ("FP32", "BF16")
MINIMUM_CUDA_DRIVER_API_VERSION = 13_000
MINIMUM_CUDA_RUNTIME_API_VERSION = 13_000
ENVIRONMENT_LOCK_SHA256 = (
    "29800d50f635e7188e55a6d6f43bfb4b8ac9ab16c4a21687db2960f18941932a"
)
PYTHON_REAL_RELATIVE_PATH = Path("envs/pypto-nvidia/bin/python3.14")
PYTHON_SIZE = 35_989_864
PYTHON_SHA256 = "aa85b78409de29d21c7db9a6ea0479fd73a4e245a733ea325f5ecf21772d030f"
ANCHOR_REQUEST_RELATIVE_PATH = Path(
    "runs/pypto-20260825T080254Z-910620-c669d9/"
    "pypto-nvidia-executable-sm120/compile-request.msgpack"
)
ANCHOR_REQUEST_SIZE = 1_583
ANCHOR_REQUEST_SHA256 = (
    "13c319b832c51188678b51a32b155253a6f896bfd1395044832611df0843adda"
)

RUNNER_RELATIVE_PATH = Path("benchmarks/operators/pypto_row_reduction_sm120.py")
RUNNER_SIZE = 42_637
RUNNER_SHA256 = "0b99dbc6b0b1f32ae0379362e2aa50ed4a0a913d9ea4875f1f76cbffee7efa55"
ANCHOR_GENERATOR_RELATIVE_PATH = Path("tools/generate_pypto_row_reduction_anchors.py")
ANCHOR_GENERATOR_SIZE = 21_172
ANCHOR_GENERATOR_SHA256 = (
    "2bbfa3dce913412ef6226e1518eb0004469b70c9f2b62ce2181add33979d9861"
)
COMPILE_ANCHORS_RELATIVE_PATH = Path(
    "state/contracts/pypto_row_reduction_compile_anchors_v1.json"
)
COMPILE_ANCHORS_SIZE = 18_276
COMPILE_ANCHORS_SHA256 = (
    "f54b008b216ff8f98e9cdcb42a7fe1b332474fe9354fb45b643d9c4811fa4689"
)
PREFLIGHT_ADAPTER_RELATIVE_PATH = Path("tools/preflight_gpu_smoke_v2.py")
CONTROLLER_RELATIVE_PATH = Path("tools/run_pypto_row_reduction_sm120_isolated.py")
FINALIZER_RELATIVE_PATH = Path("tools/finalize_pypto_row_reduction_sm120.py")
CONTROL_VALIDATOR_RELATIVE_PATH = Path(
    "tools/_pypto_row_reduction_sm120_control_manifest.py"
)
CP48_REPORT_RELATIVE_PATH = Path(
    "reports/data/pypto-row-reduction-v3-compiler-cubin-records.json"
)
CP48_REPORT_SIZE = 5_311
CP48_REPORT_SHA256 = "d06765beaf4fd3ebec3c023b473a904bc704f6ae3a3491b157913ff49e338abb"
REPLAY_DIRECTORY_NAME = "pypto-row-reduction-sm120"
PROVISIONAL_NAME = "provisional.json"
FINAL_REPORT_DIRECTORY = Path("reports/data")

REPETITIONS = 2
INPUT_GUARD_ELEMENTS = 4_096
OUTPUT_GUARD_ELEMENTS = 16
INPUT_GUARD_PREFIX = -101.0
INPUT_GUARD_SUFFIX = 103.0
OUTPUT_GUARD_PREFIX = -107.0
OUTPUT_GUARD_SUFFIX = 109.0
SENTINEL_WORDS = {
    "float32": (0xC2CA0000, 0x42CE0000, 0xC2D60000, 0x42DA0000),
    "bfloat16": (0xC2CA, 0x42CE, 0xC2D6, 0x42DA),
}
if any(len(set(words)) != 4 for words in SENTINEL_WORDS.values()):
    raise RuntimeError("row-reduction guard sentinel words are not role-distinct")
REFERENCE_STREAM_POLICY = "distinct-nondefault-eager-torch-fp32-reduction"
CANDIDATE_STREAM_POLICY = "selected-nondefault-current-stream"
REFERENCE_COMPUTE_BOUNDARY = "outside-pypto-candidate-coverage"
FP32_SUM_MAX_ULP = 16
FP32_SUM_RTOL = 2.0e-6
BF16_SUM_MAX_ULP = 1
BF16_SUM_RTOL = 1.0 / 128.0
REDUCTION_ATOL = 0.0


@dataclass(frozen=True, slots=True)
class CaseSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]
    op_name: str
    row_tile: int
    special_mode: str = "none"
    cp48_case: str | None = None

    @property
    def rows(self) -> int:
        value = 1
        for extent in self.shape[:-1]:
            value *= extent
        return value

    @property
    def contraction(self) -> int:
        return self.shape[-1]

    @property
    def result_shape(self) -> tuple[int, ...]:
        return (*self.shape[:-1], 1)

    @property
    def grid(self) -> tuple[int, int, int]:
        return ((self.rows + self.row_tile - 1) // self.row_tile, 1, 1)

    @property
    def tensor_ir_mode(self) -> str:
        return "add" if self.op_name == "tensor.row_sum" else "max"

    @property
    def comparison(self) -> str:
        if self.op_name == "tensor.row_max":
            return "exact-max-classification-sign"
        if self.dtype == "bfloat16":
            return "bf16-fp32-sum-ulp1-relative"
        return "fp32-sum-ulp16-relative"

    @property
    def max_ulp(self) -> int:
        return BF16_SUM_MAX_ULP if self.dtype == "bfloat16" else FP32_SUM_MAX_ULP

    @property
    def rtol(self) -> float:
        return BF16_SUM_RTOL if self.dtype == "bfloat16" else FP32_SUM_RTOL


CASE_SPECS = (
    CaseSpec(
        "rank1_fp32_sum_n1", "float32", (1,), "tensor.row_sum", 1, "negative-zero"
    ),
    CaseSpec(
        "rank1_fp32_sum_n7",
        "float32",
        (7,),
        "tensor.row_sum",
        1,
        cp48_case="rank1_fp32_sum",
    ),
    CaseSpec(
        "rank2_bf16_sum_n256_tail",
        "bfloat16",
        (17, 256),
        "tensor.row_sum",
        16,
        "bf16-accumulation",
    ),
    CaseSpec(
        "rank2_bf16_max_n17",
        "bfloat16",
        (2, 17),
        "tensor.row_max",
        2,
        "max-infinities",
        "rank2_bf16_max",
    ),
    CaseSpec("rank2_fp32_max_n128_tail", "float32", (5, 128), "tensor.row_max", 4),
    CaseSpec("rank2_fp32_max_n96_tail", "float32", (5, 96), "tensor.row_max", 4),
    CaseSpec("rank2_bf16_sum_n129", "bfloat16", (8, 129), "tensor.row_sum", 8),
    CaseSpec(
        "rank3_fp32_sum_n17_tail",
        "float32",
        (2, 3, 17),
        "tensor.row_sum",
        4,
        cp48_case="rank3_fp32_sum",
    ),
    CaseSpec(
        "rank3_bf16_max_n17",
        "bfloat16",
        (2, 16, 17),
        "tensor.row_max",
        16,
        cp48_case="rank3_bf16_max",
    ),
    CaseSpec("rank3_fp32_max_n257_tail", "float32", (2, 3, 257), "tensor.row_max", 4),
)
CASE_ORDER = tuple(case.name for case in CASE_SPECS)


def contraction_tile(case: CaseSpec) -> int:
    lowbit = case.contraction & -case.contraction
    return min(128, lowbit)


def contraction_chunks(case: CaseSpec) -> int:
    tile = contraction_tile(case)
    return (case.contraction + tile - 1) // tile


def required_input_guard_elements(case: CaseSpec) -> int:
    tail_rows = case.rows % case.row_tile
    speculative_rows = 0 if tail_rows == 0 else case.row_tile - tail_rows
    return speculative_rows * case.contraction + contraction_tile(case) - 1


def required_output_guard_elements(case: CaseSpec) -> int:
    return case.row_tile - 1


MAXIMUM_REQUIRED_INPUT_GUARD_ELEMENTS = max(
    required_input_guard_elements(case) for case in CASE_SPECS
)
MAXIMUM_REQUIRED_OUTPUT_GUARD_ELEMENTS = max(
    required_output_guard_elements(case) for case in CASE_SPECS
)
if MAXIMUM_REQUIRED_INPUT_GUARD_ELEMENTS != 3_967:
    raise RuntimeError("row-reduction input speculative-span bound differs")
if MAXIMUM_REQUIRED_OUTPUT_GUARD_ELEMENTS != 15:
    raise RuntimeError("row-reduction output speculative-span bound differs")
if INPUT_GUARD_ELEMENTS < MAXIMUM_REQUIRED_INPUT_GUARD_ELEMENTS:
    raise RuntimeError("row-reduction input guard is too small")
if OUTPUT_GUARD_ELEMENTS < MAXIMUM_REQUIRED_OUTPUT_GUARD_ELEMENTS:
    raise RuntimeError("row-reduction output guard is too small")


def dtype_value(pypto: Any, dtype: str) -> Any:
    if dtype == "float32":
        return pypto.DataType.FP32
    if dtype == "bfloat16":
        return pypto.DataType.BF16
    raise RuntimeError(f"unsupported row-reduction dtype: {dtype}")


def make_program(pypto: Any, ir: Any, case: CaseSpec) -> Any:
    span = ir.Span("pypto_row_reduction_sm120.py", 1, 1)
    input_type = ir.TensorType(list(case.shape), dtype_value(pypto, case.dtype))
    result_type = ir.TensorType(list(case.result_shape), dtype_value(pypto, case.dtype))
    input_value = ir.Var("input", input_type, span)
    result = ir.Var("result", result_type, span)
    call = ir.Call(ir.get_op(case.op_name), [input_value], result_type, span)
    body = ir.SeqStmts(
        [ir.AssignStmt(result, call, span), ir.ReturnStmt([result], span)], span
    )
    function = ir.Function("row_main", [input_value], [result_type], body, span)
    return ir.Program([function], "pypto_row_reduction_sm120", span)


def schedule(compiler: Any, row_tile: int) -> Any:
    parameter = compiler.ScheduleParameter
    unsigned = compiler.ScheduleValueKind.UnsignedInteger
    return compiler.CanonicalSchedule(
        [
            parameter(
                "codegen_strategy",
                compiler.ScheduleValueKind.Text,
                "layout-propagation",
            )
        ],
        [parameter("dim_000", unsigned, str(row_tile))],
        [],
        [],
        [parameter("count", unsigned, "1")],
        [parameter("count", unsigned, "4")],
        [],
        [
            parameter("bytecode_major", unsigned, "13"),
            parameter("bytecode_minor", unsigned, "3"),
            parameter("bytecode_tag", unsigned, "0"),
            parameter("max_candidates", unsigned, "0"),
            parameter("uniform_signature", compiler.ScheduleValueKind.Boolean, "false"),
        ],
    )


def dense_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    running = 1
    output = [1] * len(shape)
    for index in range(len(shape) - 1, -1, -1):
        output[index] = running
        running *= shape[index]
    return tuple(output)


def tensor_type(shape: tuple[int, ...], dtype: str) -> str:
    suffix = "f32" if dtype == "float32" else "bf16"
    return "tensor<" + "x".join(str(value) for value in shape) + f"x{suffix}>"


def tensor_with_stride(shape: tuple[int, ...], dtype: str) -> str:
    value = tensor_type(shape, dtype)
    if len(shape) == 1:
        return value
    strides = ",".join(str(item) for item in dense_strides(shape))
    return f'{value} {{nv_tensor_ir.stride = "({strides})"}}'


def canonical_tensor_ir_source(case: CaseSpec) -> bytes:
    input_type = tensor_type(case.shape, case.dtype)
    input_with_stride = tensor_with_stride(case.shape, case.dtype)
    result_type = tensor_type(case.result_shape, case.dtype)
    result_with_stride = tensor_with_stride(case.result_shape, case.dtype)
    flatten = len(case.shape) > 2
    reduction_input_shape = (case.rows, case.contraction) if flatten else case.shape
    reduction_result_shape = (case.rows, 1) if flatten else case.result_shape
    axis = 1 if flatten else len(case.shape) - 1
    widen = case.dtype == "bfloat16"
    compute_input = tensor_type(case.shape, "float32")
    compute_result = tensor_type(case.result_shape, "float32")
    compute_reduction_input = tensor_type(reduction_input_shape, "float32")
    compute_reduction_result = tensor_type(reduction_result_shape, "float32")
    reduction_input = tensor_type(reduction_input_shape, case.dtype)
    reduction_result = tensor_type(reduction_result_shape, case.dtype)
    lines = [
        "module {",
        "  nv_tensor_ir.graph @pypto_row_reduction_v3(",
        f"    %arg0: {input_with_stride}",
        (
            f"  ) -> {result_type} {{"
            if len(case.result_shape) == 1
            else f"  ) -> ({result_with_stride}) {{"
        ),
    ]
    if widen:
        lines.append(f"    %wide0 = convert %arg0 : {input_type} -> {compute_input}")
    if flatten:
        source = "%wide0" if widen else "%arg0"
        source_type = compute_input if widen else input_type
        target_type = compute_reduction_input if widen else reduction_input
        lines.append(f"    %flat0 = reshape {source} : {source_type} -> {target_type}")
    reduction_source = "%flat0" if flatten else ("%wide0" if widen else "%arg0")
    reduction_source_type = compute_reduction_input if widen else reduction_input
    reduction_target_type = compute_reduction_result if widen else reduction_result
    reduction_name = "%result0" if not flatten and not widen else "%reduce0"
    lines.append(
        f"    {reduction_name} = reduce({reduction_source}) "
        f"<dimensions = [{axis}], reduction_mode = <{case.tensor_ir_mode}>> : "
        f"{reduction_source_type} -> {reduction_target_type}"
    )
    if flatten:
        reshape_name = "%shape0" if widen else "%result0"
        reshape_target = compute_result if widen else result_type
        lines.append(
            f"    {reshape_name} = reshape %reduce0 : {reduction_target_type} -> "
            f"{reshape_target}"
        )
    if widen:
        conversion_source = "%shape0" if flatten else "%reduce0"
        lines.append(
            f"    %result0 = convert {conversion_source} : {compute_result} -> {result_type}"
        )
    lines.extend([f"    results %result0 : {result_type}", "  }", "}"])
    return ("\n".join(lines) + "\n").encode("ascii")


def fixed_child_command(workspace: Path) -> list[str]:
    root = workspace.resolve()
    return [
        str(root / "envs/pypto-nvidia/bin/python"),
        "-I",
        "-B",
        "-S",
        str(root / RUNNER_RELATIVE_PATH),
    ]


def replay_directory(workspace: Path, run_id: str) -> Path:
    return workspace.resolve() / "runs" / run_id / REPLAY_DIRECTORY_NAME


def provisional_path(workspace: Path, run_id: str) -> Path:
    return replay_directory(workspace, run_id) / PROVISIONAL_NAME


def final_report_path(workspace: Path, run_id: str) -> Path:
    return workspace.resolve() / FINAL_REPORT_DIRECTORY / f"{SMOKE_NAME}-{run_id}.json"
