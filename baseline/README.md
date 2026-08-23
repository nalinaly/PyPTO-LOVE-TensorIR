# Unmodified SGLang baseline launchers

These commands target the clean checkout recorded in `VERSIONS.lock` and the
independent model copies under `models/`. They intentionally use the
`baseline` isolation profile:

- Python comes from `envs/sglang-baseline-py312`;
- `PYTHONPATH` contains only the pinned official SGLang source;
- `SGLANG_PLUGINS` is a non-matching whitelist, so no general plugin is loaded;
- all caches, temporary files, sockets and run metadata remain below this
  workspace;
- the environment's complete Torch tree must match
  `state/environments/sglang-baseline-py312.lock.json` before a server process
  is created;
- `--framework-launch` binds the child executable to that prefix's Python and
  rechecks Torch CUDA/HIP identity, SM120, loaded DSOs and the clean SGLang
  source before spawning the server.

`launch_0p8b.sh` and `launch_9b.sh` are frozen initial bring-up commands with
CUDA Graph and chunked prefill disabled. Later benchmark commands enable those
features in controlled workload stages.

The scripts have not yet been executed. Their presence is not baseline,
correctness or performance evidence; the CPython 3.12 environment and exact
Triton wheel must be built first, and every launch remains subject to the live
zcode/gem5 heavy-work gate.
