"""Load and validate the two frozen Qwen3.5 release geometries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Qwen35Shape:
    model: str
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    attention_heads: int
    kv_heads: int
    attention_head_dim: int
    rotary_dim: int
    linear_q_heads: int
    linear_value_heads: int
    linear_key_dim: int
    linear_value_dim: int
    conv_channels: int
    config_sha256: str

    def record(self) -> dict[str, object]:
        return asdict(self)


# These values are the TP=1 text_config fields used by the frozen release
# models.  Loading a different config must fail instead of silently turning a
# "real model shape" regression into a synthetic-shape regression.
_LOCKED = {
    "Qwen3.5-0.8B": {
        "hidden_size": 1024,
        "intermediate_size": 3584,
        "vocab_size": 248320,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "max_position_embeddings": 262144,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    },
    "Qwen3.5-9B": {
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "vocab_size": 248320,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "max_position_embeddings": 262144,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    },
}

RELEASE_ROWS = (1, 19)


def parse_release_rows(value: str) -> tuple[int, ...]:
    try:
        rows = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise ValueError("--rows must be the frozen value 1,19") from error
    if rows != RELEASE_ROWS:
        raise ValueError("--rows must be the frozen value 1,19")
    return rows


def load_release_shapes(model_root: Path) -> tuple[Qwen35Shape, ...]:
    root = model_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("--model-root must resolve to a directory")
    shapes = []
    for model, expected in _LOCKED.items():
        config_path = (root / model / "config.json").resolve(strict=True)
        if root not in config_path.parents or not config_path.is_file():
            raise ValueError(f"model config escaped --model-root: {model}")
        raw = config_path.read_bytes()
        payload = json.loads(raw)
        text = payload.get("text_config", payload)
        if type(text) is not dict:
            raise ValueError(f"{model} text_config is not an object")
        observed = {field: text.get(field) for field in expected}
        if observed != expected or any(
            type(observed[field]) is not type(value)
            for field, value in expected.items()
        ):
            raise ValueError(
                f"{model} release geometry changed: "
                f"expected={expected!r}, observed={observed!r}"
            )
        rope = text.get("rope_parameters")
        if (
            type(rope) is not dict
            or type(rope.get("partial_rotary_factor")) is not float
            or rope.get("partial_rotary_factor") != 0.25
        ):
            raise ValueError(f"{model} partial rotary factor must remain 0.25")
        head_dim = int(expected["head_dim"])
        linear_q_heads = int(expected["linear_num_key_heads"])
        linear_value_heads = int(expected["linear_num_value_heads"])
        linear_key_dim = int(expected["linear_key_head_dim"])
        linear_value_dim = int(expected["linear_value_head_dim"])
        shapes.append(
            Qwen35Shape(
                model=model,
                hidden_size=int(expected["hidden_size"]),
                intermediate_size=int(expected["intermediate_size"]),
                vocab_size=int(expected["vocab_size"]),
                attention_heads=int(expected["num_attention_heads"]),
                kv_heads=int(expected["num_key_value_heads"]),
                attention_head_dim=head_dim,
                rotary_dim=int(head_dim * 0.25),
                linear_q_heads=linear_q_heads,
                linear_value_heads=linear_value_heads,
                linear_key_dim=linear_key_dim,
                linear_value_dim=linear_value_dim,
                conv_channels=(
                    2 * linear_q_heads * linear_key_dim
                    + linear_value_heads * linear_value_dim
                ),
                config_sha256=hashlib.sha256(raw).hexdigest(),
            )
        )
    return tuple(shapes)
