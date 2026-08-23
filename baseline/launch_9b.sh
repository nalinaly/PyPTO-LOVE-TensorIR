#!/usr/bin/env bash
set -euo pipefail

PYPTO_ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CONTROL_PYTHON="$PYPTO_ROOT/envs/pypto-nvidia/bin/python"
BASELINE_PYTHON="$PYPTO_ROOT/envs/sglang-baseline-py312/bin/python"

exec "$CONTROL_PYTHON" "$PYPTO_ROOT/tools/run_isolated.py" \
  --mode gpu-benchmark \
  --environment sglang-baseline-py312 \
  --framework-profile baseline \
  --framework-launch \
  -- "$BASELINE_PYTHON" -m sglang.launch_server \
  --model-path "$PYPTO_ROOT/models/Qwen3.5-9B" \
  --served-model-name Qwen3.5-9B-text-r0 \
  --model-impl sglang \
  --load-format safetensors \
  --json-model-override-args '{"language_model_only":true}' \
  --dtype bfloat16 \
  --kv-cache-dtype bfloat16 \
  --tp-size 1 \
  --context-length 2048 \
  --max-total-tokens 2048 \
  --max-prefill-tokens 2048 \
  --max-running-requests 1 \
  --chunked-prefill-size -1 \
  --mem-fraction-static 0.80 \
  --reasoning-parser qwen3 \
  --attention-backend flashinfer \
  --sampling-backend flashinfer \
  --linear-attn-backend triton \
  --linear-attn-decode-backend triton \
  --linear-attn-prefill-backend triton \
  --mamba-backend triton \
  --mamba-ssm-dtype float32 \
  --disable-overlap-schedule \
  --mamba-radix-cache-strategy no_buffer \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled \
  --host 127.0.0.1 \
  --port 43190 \
  --engine-info-bootstrap-port 44190 \
  --nccl-port 45190 \
  --watchdog-timeout 1200
