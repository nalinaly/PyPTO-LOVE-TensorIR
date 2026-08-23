# CHECKPOINT

**Checkpoint:** `CP-0001`

**Status:** R0 started; no compiler or model milestone accepted.

## Current truth

- The control repository was initialized in
  `/home/zhaosiying/pypto-love-tensor-ir`.
- Authorized PyPTO, standalone kernel/plugin projects, and clean official
  PyTorch/SGLang checkouts are materialized at the exact locked identities.
- The local target is an NVIDIA GeForce RTX 5090 Laptop GPU, SM120, 24,463 MiB.
- CUDA Toolkit 13.3 is installed under `/usr/local/cuda-13.3`; the installed
  PyTorch is 2.13.0+cu130.
- Prior reconnaissance did not complete a TensorIR SM120 runtime launch. Static
  target support is not runtime acceptance.
- Protected zcode work was active at R0 start. No heavy PyPTO task may begin
  until the live safety preflight is green.

## Resume action

Run `python tools/preflight.py --mode light`, inspect the protected-process
report, then continue TensorIR/CUDA Tile source integration and the remaining
unchecked R0 items. Heavy work remains gated while protected gem5/SGLang jobs
are present.
