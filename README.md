# PyPTO Love TensorIR

Control-plane workspace for the SM120 NVIDIA backend, standalone PyPTO kernel
library, TorchInductor/SGLang compatibility plugins, and Qwen3.5 acceptance on
the local NVIDIA GeForce RTX 5090 Laptop GPU.

The three implementation repositories under `projects/` have independent Git
histories. Official PyTorch and SGLang checkouts under `upstream/` must remain
clean. Generated artifacts, environments, caches, runs, and model weights stay
inside this workspace but are not committed by the control repository.

Resume from zero context in this order:

1. `CHECKPOINT.md`
2. `GOAL.md`
3. `PLAN.md`
4. `TODO.md`
5. `HANDOVER.md`

Never run project commands directly in `/home/zhaosiying/amdgpu-sim` or
`/home/zhaosiying/zcode-lane`. Those trees and their processes are external and
must not be changed or signalled.

