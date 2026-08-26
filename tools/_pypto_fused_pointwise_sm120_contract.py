#!/usr/bin/env python3
"""Pure constants for the v1 PyPTO fused-pointwise SM120 correctness gate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SMOKE_SCHEMA_VERSION = 1
SMOKE_NAME = "pypto-fused-pointwise-sm120"
GPU_SMOKE_POLICY_VERSION = 1
GPU_SMOKE_AUTHORIZATION = (
    "user-authorized-protected-cpu-lane-with-zero-nvidia-runtime-or-compute"
)
GPU_SMOKE_TIMEOUT_SECONDS = 1_800
GPU_SMOKE_MINIMUM_FREE_DISK_GIB = 64

PYPTO_HEAD = "b83fcd3ddc497d585bcc45883eede179aff7d4d2"
PYPTO_TREE = "49eda98f3ed8d72bfd14d5a5900cdc0e71ca699d"
TENSOR_IR_HEAD = "1dcb38c20e53d07c97d3781cae538e33901bae30"
CUDA_TILE_HEAD = "af2417041cc939b87ef56d92cfdcf61737c5457e"
LLVM_HEAD = "57109befac92811d2253109242ca6fa69c961fb2"
PYPTO_DSO_RELATIVE_PATH = Path(
    "builds/pypto-fused-pointwise-v2-on-b83fcd3-final/product/"
    "pypto_core.cpython-314-x86_64-linux-gnu.so"
)
PYPTO_DSO_SIZE = 784_043_568
PYPTO_DSO_SHA256 = "0e8f33c263e06777aec06263bf32ca59ac554868529f3fa085212cf27e2facbe"
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
CUDA_RUNTIME_DISTRIBUTION = "nvidia-cuda-runtime"
CUDA_RUNTIME_VERSION = "13.0.96"

RUNNER_RELATIVE_PATH = Path("benchmarks/operators/pypto_fused_pointwise_sm120.py")
ANCHOR_GENERATOR_RELATIVE_PATH = Path("tools/generate_pypto_fused_pointwise_anchors.py")
ANCHOR_GENERATOR_SIZE = 11_387
ANCHOR_GENERATOR_SHA256 = (
    "89f06a416622e1d78595c0a086db4dce66bebbf70f3867b2601885767e85c54e"
)
ANCHOR_REQUEST_RELATIVE_PATH = Path(
    "runs/pypto-20260825T080254Z-910620-c669d9/"
    "pypto-nvidia-executable-sm120/compile-request.msgpack"
)
ANCHOR_REQUEST_SIZE = 1_583
ANCHOR_REQUEST_SHA256 = (
    "13c319b832c51188678b51a32b155253a6f896bfd1395044832611df0843adda"
)
COMPILE_ANCHORS_RELATIVE_PATH = Path(
    "state/contracts/pypto_fused_pointwise_compile_anchors_v1.json"
)
COMPILE_ANCHORS_SIZE = 21_490
COMPILE_ANCHORS_SHA256 = (
    "584f6755bbd248de5bb6ddd3ff610da8082667bc892a6cff6583ea42d4c44c97"
)
# Filled after the implementation blob and all compile anchors are frozen.
RUNNER_SIZE = 66_999
RUNNER_SHA256 = "b7960cc894834b3ba05476943e774cfc8602891faa5b9137b3d97a6aac40ab15"
REPLAY_DIRECTORY_NAME = "pypto-fused-pointwise-sm120"
PROVISIONAL_NAME = "provisional.json"
FINAL_REPORT_DIRECTORY = Path("reports/data")

REPETITIONS = 2
GUARD_ELEMENTS = 16
INPUT_GUARD_PREFIX_BASE = -256.0
INPUT_GUARD_SUFFIX_BASE = 256.0
OUTPUT_GUARD_PREFIX = -511.0
OUTPUT_GUARD_SUFFIX = 511.0
REFERENCE_STREAM_POLICY = "distinct-nondefault-eager-torch-one-op-per-call"
CANDIDATE_STREAM_POLICY = "selected-nondefault-current-stream"
REFERENCE_COMPUTE_BOUNDARY = "outside-pypto-candidate-coverage"
NO_SUBNORMAL_INPUTS = True
HIGH_PRECISION_ALLOWED = False
FP32_TRANSCENDENTAL_MAX_ULP = 4
FP32_TRANSCENDENTAL_RTOL = 2.0e-6
BF16_TRANSCENDENTAL_MAX_ULP = 1
BF16_TRANSCENDENTAL_RTOL = 1.0 / 128.0
TRANSCENDENTAL_ATOL = 0.0


@dataclass(frozen=True, slots=True)
class CaseSpec:
    """Process-independent identity for one fixed numerical fixture."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    tile_sizes: tuple[int, ...]
    expected_grid: tuple[int, int, int]
    family: str
    input_count: int
    assignment_count: int
    operator_sequence: tuple[str, ...]
    scalar_literals: tuple[float, ...]
    comparison: str
    max_ulp: int
    rtol: float
    atol: float
    special_prefix_count: int
    expected_hir_bytes: int
    expected_hir_sha256: str
    expected_source_ir_bytes: int
    expected_source_ir_digest: str
    expected_static_specialization_digest: str
    expected_symbolic_specialization_digest: str
    expected_argument_abi_digest: str
    expected_result_abi_digest: str
    expected_mutation_abi_digest: str
    expected_callable_abi_digest: str
    expected_build_spec_identity_digest: str
    expected_artifact_identity_digest: str
    expected_device_code_bytes: int
    expected_device_code_sha256: str
    repetitions: int = REPETITIONS

    @property
    def expected_kernel_arguments(self) -> int:
        return self.input_count + 1


_ANCHORS: dict[str, tuple[int | str, ...]] = {
    "arith_fp32_tail": (
        7_390,
        "f7cd96727a6601f6f32de25c09c625db216b4679797d91448f798db0faad81a6",
        1_097,
        "e5d2275c70f95a32047743880cc8bc54a84e41e984936855c5a583eeb69721f0",
        "541e614501314372a9e84bb19be297c1b9beda9446c563d717cfd0a27bb2aba2",
        "9372e99f73d788b3d6431e5e5532d5b3926a46fc14e679c980ca5f0a3517a905",
        "12a28e966be5e9c4595f0b48138b095e973dc6b14479f5d4b2fe63b22f301438",
        "ed8836313ff91be1cbaee6584b244cdd7904217cf70ef21ac8401d295953523c",
        "55941266a7b5f38c0f7758330a3804562a00647270d4dd650b5f14d4480c75fd",
        "c1dc16b29af4ea044aa2334b609ad8dd2c11bc89ca5d86f91cc58b61e2fefe60",
        15_128,
        "54e7aac7f5d0ed39e6bf7537201f64629375ed89e7743fbf02effdf9a0536d58",
    ),
    "arith_bf16_rank3_tail": (
        7_858,
        "c15563c288685576d131a6b492367e1ca22a69e33e5315a2d1a76a46c76b6689",
        1_162,
        "d5819a4048127083acaa34edc781432a625c951b339e5fa21722961b7a52b740",
        "6789cbb549f7404120b2a8e97b2b385db6f3bb6a2ee10fe96e6db2c72ee65582",
        "9372e99f73d788b3d6431e5e5532d5b3926a46fc14e679c980ca5f0a3517a905",
        "ede9371ece486907772fe98721994ee09e326db281ab5f0c76090a116255e9e4",
        "a34a664de763e6cb694ebb2eb1fcfd5dda5d6f40d1bc3b73de768837d7ce0dc8",
        "55941266a7b5f38c0f7758330a3804562a00647270d4dd650b5f14d4480c75fd",
        "402e45f0066edf4807cf3733d97db8eb12e8a61f9451f2c347d52ae7fbbc240d",
        15_136,
        "ed89b671ab98ecdf38c69b51c42e3a5a23b20772edce16e3ddb73663854e237a",
    ),
    "exp_fp32_tail1": (
        1_766,
        "69afe75eb2a82986b5329a5040c4a7e6039c77551b8eb13f44b3da2b4c4aa630",
        191,
        "3a9328a0aad4427374ccb56272ec89116093ab0fc385981dfcaf6f38dce280e3",
        "8e94d10076beaad340e5045ecebb2f015ecd03dbc488a5b90f5a21e580e5d69b",
        "9372e99f73d788b3d6431e5e5532d5b3926a46fc14e679c980ca5f0a3517a905",
        "09e2888bac856ef869a84dafbab03a2f81d3b8027bba6dcc42233a156166a3bc",
        "d81c8395d1805105a449f047083515ff3fe11445f7d5ae93a67e3205a23777fe",
        "1c10569fb6e5c1532f4019a0a66a50858338397a51b028f9b7336ce856803c47",
        "3934c0909ac19b3280bf326cf1c414912b1038553d56483678010a5a8c778f8f",
        13_888,
        "d6b5d8968b2f9282c9898de60460652fa6ca84e5faa1671f1ae5fa7507544804",
    ),
    "exp_bf16_exact_tile": (
        1_961,
        "2b646fe17d05aa3c57c095aaa489a8bca773233651ee42ad97369c7090ceef22",
        265,
        "0f21270b0b1a9a30b4df1e7a2de3e38e6bedf484fbde7127139bf53f0c59d9f8",
        "fae4a803ed8a5ddf145363a45ba520ddf634b5002d756d77ae9808b2d0935006",
        "9372e99f73d788b3d6431e5e5532d5b3926a46fc14e679c980ca5f0a3517a905",
        "b81159b08ec07df41a2dd17c2b22f17241f265c1564fa24d59bf06f737b0a870",
        "82d52c8ea547920ea649b16eda1b9b995df05abec9f83ea23d7ec9609bcd8329",
        "1c10569fb6e5c1532f4019a0a66a50858338397a51b028f9b7336ce856803c47",
        "e2903ff8af453fe58a0ada387b68a22648f50ae924753d3da8705257bb3faab2",
        13_680,
        "5c4efdc785a2a9b648cb9292922474b89f934b3c8f33cb14cafe510024a42608",
    ),
    "recip_fp32_tail": (
        1_927,
        "1bca35843dd9f7639129d007732169a37667f5c5761a205c3250cdbc5438cc91",
        268,
        "96bd090ead61b49a0f6a9abf9a8bf6fd3d6d65a26798aca74e8cb2bbe11e92b1",
        "9d1d1b829f19447e03e2c7e061215e29adc1056fdd7cb6f72f08199b68ffbc34",
        "9372e99f73d788b3d6431e5e5532d5b3926a46fc14e679c980ca5f0a3517a905",
        "5d5b85c8980f80454b658ff2d6a56906a6bf1e51e0630f4910bf4b48313fad3b",
        "ed8836313ff91be1cbaee6584b244cdd7904217cf70ef21ac8401d295953523c",
        "1c10569fb6e5c1532f4019a0a66a50858338397a51b028f9b7336ce856803c47",
        "6366509845649ad50855675d79cabf767108349553185b869545ea9662aee227",
        16_560,
        "2c339a23ae31f2c3c697a0f2b35b5f75dba42976efcb0bbe111733bc8016f95b",
    ),
    "recip_bf16_tail1": (
        1_786,
        "b3c652d781ff3d7dace289e7b4c0288011ca2be9a0e8e6f4b1deb014ce5ab7fd",
        202,
        "278c016c1712b02abf94e2e697df5a0b0f25c40b0c14006e4106604f455d5bb8",
        "8e94d10076beaad340e5045ecebb2f015ecd03dbc488a5b90f5a21e580e5d69b",
        "9372e99f73d788b3d6431e5e5532d5b3926a46fc14e679c980ca5f0a3517a905",
        "b417dca1da56bbd5c774e8ac3b809c167424590066cc7641983472204c65f127",
        "c91513be1d3301cad02dd83f849200d1f1fb44f2ec43b557079f8fbb5640bf29",
        "1c10569fb6e5c1532f4019a0a66a50858338397a51b028f9b7336ce856803c47",
        "3934c0909ac19b3280bf326cf1c414912b1038553d56483678010a5a8c778f8f",
        16_688,
        "8a5313b536f0543bf6dba6a18737a9ffe90958565330445cfd0a055eb61aad4b",
    ),
    "rsqrt_fp32_rank3_tail": (
        2_131,
        "2a90ef8ec803c094912514ee80145fedffddf63204bc6ef2ce5d7761838947dc",
        277,
        "d50cec267d5ce639c7c5e6a2e906f7ffc3367a3f42019965abf709af2e2ae591",
        "dd66068dffc39d3dd4a79c877b751f4f9e9cb92e298660143492439d6461bcc5",
        "9372e99f73d788b3d6431e5e5532d5b3926a46fc14e679c980ca5f0a3517a905",
        "2cab0028ec574e2a31808c61e90ea69e8201304abbf521a7d5b2830f017fe3f9",
        "8623014bdd694da338601d8f1d46130564a4030ae5fd429a473acfc333e8a156",
        "1c10569fb6e5c1532f4019a0a66a50858338397a51b028f9b7336ce856803c47",
        "a1f918a38df76e355b672591911667b1216281c43e8a4a256ff064470ee77386",
        13_880,
        "c81fb8b186046f0927d3f281f06e27f5524a1a06b5deb729650beb228b069378",
    ),
    "rsqrt_bf16_exact_tile": (
        1_981,
        "4d40c1340335c6ac5358b0d8f4c8154edf82f4f044e1a2fb74348926f0601b77",
        267,
        "94cc16f92ca2d04f24778094e6fc5146fe2b6422f532a49c2ebb872420fd972b",
        "fae4a803ed8a5ddf145363a45ba520ddf634b5002d756d77ae9808b2d0935006",
        "9372e99f73d788b3d6431e5e5532d5b3926a46fc14e679c980ca5f0a3517a905",
        "b81159b08ec07df41a2dd17c2b22f17241f265c1564fa24d59bf06f737b0a870",
        "82d52c8ea547920ea649b16eda1b9b995df05abec9f83ea23d7ec9609bcd8329",
        "1c10569fb6e5c1532f4019a0a66a50858338397a51b028f9b7336ce856803c47",
        "e2903ff8af453fe58a0ada387b68a22648f50ae924753d3da8705257bb3faab2",
        13_680,
        "8b5106458a6025aa89b01c316949caef7d6019ad09346b8d856a59f347d0c13c",
    ),
    "max16x64_fp32_tail1": (
        43_558,
        "f5b0bc5cb6f2d4d01b5c9167e4b60ec3d8362b83a4cf8a7929148ab9abcf6ae7",
        3_530,
        "dbd228b8eed9689754326ae8be540941fbf8b113178cbd86bf64af1c98d3c06d",
        "5c609af8db0eff21c92a0a5af8d30a32e238228df72511aba9880e8dbe7c5360",
        "9372e99f73d788b3d6431e5e5532d5b3926a46fc14e679c980ca5f0a3517a905",
        "4a7c19ef388ae38291a27efa019af54a8ddf9d1058679ab3ae4f79a058f29d6c",
        "d81c8395d1805105a449f047083515ff3fe11445f7d5ae93a67e3205a23777fe",
        "4d27232983f5fe538dda27faa303a0d9f1c8cb0f2e3fde9ca6022e1381599f2a",
        "176a4472446721cbe8d0407e8584c2d600cb843964d2d2f6b3a0557d3f6fd1ce",
        21_216,
        "83ab7ee75be6a8b2c4cfcf8bd4c61a50d54261982f73e318adc78e0637f9b53f",
    ),
}

_TRANSACTION_IDENTITIES = {
    "arith_fp32_tail": (
        "70a9fc00fc5d3203049acd53edb39730521d3fbb9d4c3c338afc271d79bf2cf9",
        "04c16e199be485ad314466f9fc7bb76ff223398bf41bd36bfcbd43f930ab6f25",
    ),
    "arith_bf16_rank3_tail": (
        "9e506071b674aeb4a2d67f11f4bb2810c118726112d21b73664fae0c55f1e728",
        "76c650bf5caf49786c2e753f0985c72c456a19a7e1c33758e23535696bc4c488",
    ),
    "exp_fp32_tail1": (
        "653fd49725c8421362541a2a1febd6dcb817cb28f44ddcfa12465002a8639ac4",
        "f2453f3d7c9e0830ef741a516c1fd60c95b4c2dd560fbd7eca082204da71e529",
    ),
    "exp_bf16_exact_tile": (
        "5af7f9290eeb6fc2ed65368169cc057139ef1fa578e70cc6d10c55d3a810b1fb",
        "6f692785da6ae4b87b07d2b338836ad97917bfe01d810726175fee978ac6677f",
    ),
    "recip_fp32_tail": (
        "ef6fc26ff3ff03a6fbbe2c200bf85fb6d2e2aafad8f8d2b49fd5a0dd49fa0287",
        "3da13cdbaad387ca0dd78ad5a3bdbf6d15e0c38f3340476146b88d484558b5d2",
    ),
    "recip_bf16_tail1": (
        "55e16140e457c3692b3641f5fc74233bc65fd85d4c984c72f4a42694afb11edc",
        "dc7c47ff0f82d84f0e539af44a840ba7fc2e6529c6949e6ce170f986453b8515",
    ),
    "rsqrt_fp32_rank3_tail": (
        "0547be0d66ed9927cbe5d388856a022deadeac45e8da90ed44fdcf13cae85ba9",
        "9618a95f8cd35b72a8cb96c7915b6be086d598771144b324eef81afae9bb31bd",
    ),
    "rsqrt_bf16_exact_tile": (
        "bf12c3373014637a3d04d8f46eeaee64aa867cc5b499f724ca505520eb22f4ea",
        "49860aa4959eaf6b27ec582bb8fef47828e429800fb079ea7c94bbc58bef4924",
    ),
    "max16x64_fp32_tail1": (
        "3aa8fec91ee401fc1f1baa3a61f2b4db649d7194f7f82dcde35229d3bd6b3877",
        "42067515f7bb30002f6165cbd1bb6ebdbfbee5716f333d533b24ecb1738a7a56",
    ),
}


def _case(
    *,
    name: str,
    dtype: str,
    shape: tuple[int, ...],
    strides: tuple[int, ...],
    tile: int,
    grid_x: int,
    family: str,
    input_count: int,
    assignment_count: int,
    operator_sequence: tuple[str, ...],
    scalar_literals: tuple[float, ...] = (),
    comparison: str,
    max_ulp: int = 0,
    rtol: float = 0.0,
    special_prefix_count: int = 0,
) -> CaseSpec:
    """Declare one case with its independently frozen compiler identities."""

    anchors = _ANCHORS[name]
    build_spec_identity, artifact_identity = _TRANSACTION_IDENTITIES[name]
    return CaseSpec(
        name=name,
        dtype=dtype,
        shape=shape,
        strides=strides,
        tile_sizes=(tile,),
        expected_grid=(grid_x, 1, 1),
        family=family,
        input_count=input_count,
        assignment_count=assignment_count,
        operator_sequence=operator_sequence,
        scalar_literals=scalar_literals,
        comparison=comparison,
        max_ulp=max_ulp,
        rtol=rtol,
        atol=TRANSCENDENTAL_ATOL,
        special_prefix_count=special_prefix_count,
        expected_hir_bytes=int(anchors[0]),
        expected_hir_sha256=str(anchors[1]),
        expected_source_ir_bytes=int(anchors[2]),
        expected_source_ir_digest=str(anchors[3]),
        expected_static_specialization_digest=str(anchors[4]),
        expected_symbolic_specialization_digest=str(anchors[5]),
        expected_argument_abi_digest=str(anchors[6]),
        expected_result_abi_digest=str(anchors[7]),
        expected_mutation_abi_digest=str(anchors[8]),
        expected_callable_abi_digest=str(anchors[9]),
        expected_build_spec_identity_digest=build_spec_identity,
        expected_artifact_identity_digest=artifact_identity,
        expected_device_code_bytes=int(anchors[10]),
        expected_device_code_sha256=str(anchors[11]),
    )


_ARITHMETIC_SEQUENCE = (
    "tensor.mul",
    "tensor.add",
    "tensor.muls",
    "tensor.adds",
    "tensor.subs",
    "tensor.mul",
    "tensor.sub",
    "tensor.neg",
)

CASE_SPECS = (
    _case(
        name="arith_fp32_tail",
        dtype="float32",
        shape=(3, 5),
        strides=(5, 1),
        tile=8,
        grid_x=2,
        family="arithmetic",
        input_count=4,
        assignment_count=8,
        operator_sequence=_ARITHMETIC_SEQUENCE,
        scalar_literals=(8_388_608.0, 0.3, 0.3),
        comparison="exact",
    ),
    _case(
        name="arith_bf16_rank3_tail",
        dtype="bfloat16",
        shape=(2, 3, 5),
        strides=(15, 5, 1),
        tile=16,
        grid_x=2,
        family="arithmetic",
        input_count=4,
        assignment_count=8,
        operator_sequence=_ARITHMETIC_SEQUENCE,
        scalar_literals=(128.0, 0.3, 0.3),
        comparison="exact",
    ),
    _case(
        name="exp_fp32_tail1",
        dtype="float32",
        shape=(17,),
        strides=(1,),
        tile=16,
        grid_x=2,
        family="exp",
        input_count=1,
        assignment_count=1,
        operator_sequence=("tensor.exp",),
        comparison="ulp-and-relative",
        max_ulp=FP32_TRANSCENDENTAL_MAX_ULP,
        rtol=FP32_TRANSCENDENTAL_RTOL,
        special_prefix_count=5,
    ),
    _case(
        name="exp_bf16_exact_tile",
        dtype="bfloat16",
        shape=(8, 8),
        strides=(8, 1),
        tile=16,
        grid_x=4,
        family="exp",
        input_count=1,
        assignment_count=1,
        operator_sequence=("tensor.exp",),
        comparison="ulp-and-relative",
        max_ulp=BF16_TRANSCENDENTAL_MAX_ULP,
        rtol=BF16_TRANSCENDENTAL_RTOL,
        special_prefix_count=5,
    ),
    _case(
        name="recip_fp32_tail",
        dtype="float32",
        shape=(3, 5),
        strides=(5, 1),
        tile=8,
        grid_x=2,
        family="recip",
        input_count=1,
        assignment_count=1,
        operator_sequence=("tensor.recip",),
        comparison="exact-with-special-classification",
        special_prefix_count=5,
    ),
    _case(
        name="recip_bf16_tail1",
        dtype="bfloat16",
        shape=(17,),
        strides=(1,),
        tile=16,
        grid_x=2,
        family="recip",
        input_count=1,
        assignment_count=1,
        operator_sequence=("tensor.recip",),
        comparison="exact-with-special-classification",
        special_prefix_count=5,
    ),
    _case(
        name="rsqrt_fp32_rank3_tail",
        dtype="float32",
        shape=(2, 3, 5),
        strides=(15, 5, 1),
        tile=16,
        grid_x=2,
        family="rsqrt",
        input_count=1,
        assignment_count=1,
        operator_sequence=("tensor.rsqrt",),
        comparison="ulp-and-relative",
        max_ulp=FP32_TRANSCENDENTAL_MAX_ULP,
        rtol=FP32_TRANSCENDENTAL_RTOL,
        special_prefix_count=7,
    ),
    _case(
        name="rsqrt_bf16_exact_tile",
        dtype="bfloat16",
        shape=(8, 8),
        strides=(8, 1),
        tile=16,
        grid_x=4,
        family="rsqrt",
        input_count=1,
        assignment_count=1,
        operator_sequence=("tensor.rsqrt",),
        comparison="ulp-and-relative",
        max_ulp=BF16_TRANSCENDENTAL_MAX_ULP,
        rtol=BF16_TRANSCENDENTAL_RTOL,
        special_prefix_count=7,
    ),
    _case(
        name="max16x64_fp32_tail1",
        dtype="float32",
        shape=(17,),
        strides=(1,),
        tile=16,
        grid_x=2,
        family="maximum-boundary",
        input_count=16,
        assignment_count=64,
        operator_sequence=("tensor.add",) * 15 + ("tensor.neg",) * 49,
        comparison="exact",
    ),
)
CASE_ORDER = tuple(case.name for case in CASE_SPECS)
if CASE_ORDER != tuple(_ANCHORS) or CASE_ORDER != tuple(_TRANSACTION_IDENTITIES):
    raise RuntimeError("fused-pointwise case-keyed anchor order differs")


def fixed_child_command(workspace: Path) -> list[str]:
    """Return the only direct child accepted by this GPU-smoke lane."""

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
