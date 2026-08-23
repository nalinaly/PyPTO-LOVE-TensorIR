# Runtime coverage collector map

This is the implementation boundary for the future
`pypto-framework-plugins` CUDA activity collector. It is grounded in the
currently pinned PyTorch and SGLang sources. It is not runtime evidence and no
collector has been executed yet.

## Pinned extension points

- PyTorch `cf30153c4c131c8164ee7798e5022d810682e2cb` includes the experimental
  `torch.profiler._cupti_monitor` backend. Through
  `torch.profiler._ExperimentalConfig(custom_profiler_config=...)`, it can own
  CUDA kernel, memcpy, memset, runtime, driver and external-correlation
  activities while Kineto records CPU scopes.
- `record_function` emits CUPTI external correlation and is the eager launch
  annotation mechanism. It must wrap the final runtime artifact launch, not an
  FX node that Dynamo/Inductor may erase.
- `torch.cuda._graph_annotations` provides `mark_kernels`,
  `resolve_pending_annotations` and `remap_to_exec_graph` for a later
  CUDA-Graph implementation. The API may silently no-op when its CUDA binding
  prerequisites are missing; strict mode must probe and reject that state.
- SGLang `71de97b264b04dcd514cf904003028aefe9775c8` exposes the official
  `sglang.srt.plugins`/`HookRegistry` extension boundary. `ModelRunner.forward`
  already has a broad `record_function` step scope, but the plugin must add a
  stable versioned PyPTO model-forward scope rather than parse its display
  text.

Relevant pinned files:

```text
envs/pypto-nvidia/lib/python3.14/site-packages/torch/profiler/_cupti_monitor.py
envs/pypto-nvidia/lib/python3.14/site-packages/torch/profiler/profiler.py
envs/pypto-nvidia/lib/python3.14/site-packages/torch/autograd/profiler.py
envs/pypto-nvidia/lib/python3.14/site-packages/torch/cuda/_graph_annotations.py
upstream/sglang/python/sglang/srt/plugins/__init__.py
upstream/sglang/python/sglang/srt/plugins/hook_registry.py
upstream/sglang/python/sglang/srt/model_executor/model_runner.py
upstream/sglang/python/sglang/srt/managers/scheduler_components/profiler_manager.py
```

## Eager-first implementation

The first collector commit must deliberately disable Torch compile and every
SGLang CUDA-Graph backend. It belongs entirely in the plugin project.

```text
SGLang AROUND hook
  -> pypto.model_forward.v1 scope
  -> one process-local CUPTI-monitor owner
  -> final PyPTO artifact launch wrapper
       -> canonical artifact external-correlation tag
       -> CUDA kernel launch on current stream
  -> forced activity flush/drain
  -> normalize every observed activity
  -> TraceManifest + KernelEvent stream
  -> CoverageAuditor
```

Normalization must retain nanosecond duration and every kernel, memcpy and
memset in the audited interval. An untagged compute activity is fallback, not
an ignorable event. Unknown activity types, missing correlations, duplicate
identities, dropped records, worker failure or a zero-kernel window fail
closed. Each process/rank produces and verifies its own trace before any
cross-process report is assembled.

Artifact correlation comes from immutable compiler-cache and operator
provenance snapshots. A kernel name or caller-supplied provider string is never
sufficient.

## Closed-world gate

The stock monitor exposes `dropped_records`, but its public drain can time out
without returning an explicit `drain_complete` result. Its kernel record also
does not independently expose cubin bytes or a module hash. Therefore the first
working collector must emit `closed_world=false` and is development evidence
only. The strict auditor is expected to reject it.

`closed_world=true` requires an additional reviewed protocol proving all of:

- one CUPTI activity owner in the process;
- completed drain/flush with no loss;
- quiescent interval boundaries and complete process/rank accounting;
- exact artifact load-to-launch binding;
- no unresolved or unknown compute activity;
- for graphs, complete capture-node annotation and replay-node remapping.

Ordinary Kineto, NVTX, `torch.library`, or `TorchDispatchMode` can add useful
diagnostics but cannot independently satisfy this gate.

## CUDA Graph follow-on

Graph support is a separate milestone. Capture must annotate every artifact
launch, resolve annotations, bind them to the instantiated executable graph,
and map replay activities through `(graph id, graph node id)`. Recapture,
bucket rebuild or graph replacement creates a new immutable manifest. Full,
breakable, piecewise, prefill and decode paths must each be covered; enabling
an unimplemented graph backend is a strict error.
