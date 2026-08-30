# Article Demo Import

This tree preserves every runnable code sample shown in the referenced PyPTO-
Lib article, plus the transitive files required by its Golden Harness,
Qwen3-14B entry point, and DeepSeek V4-Flash MTP entry points. The article-time
snapshot contains 11 teaching examples and 66 CLI entry points (57 runnable,
9 explicitly excluded draft or Ascend-CANN-only entries). See
`SOURCE_NOTICE.md` and `SOURCE_MANIFEST.json` for the exact source commit and
hashes; do not edit the imported files.

## Typical execution through this checkout

The upstream commands target Ascend software and are retained verbatim in
`UPSTREAM_README.md`. On an NVIDIA checkout, use the project compatibility
launcher from the repository root so platform setup is explicit:

```bash
python tools/run_article_demo.py --demo examples/beginner/hello_world.py

# Audit every imported CLI entry point without requiring an accelerator.
python tools/run_article_demo_matrix.py --mode help \
  --output state/evidence/article-demo-matrix-help.json
```

The launcher writes a machine-readable result and terminal transcript under
`state/evidence/article-demos/`. A successful result must include
`status: pass`, the imported source SHA-256, and a golden comparison summary.
Until a real Ubuntu/RTX5090 run is recorded, this directory intentionally does
not claim that the upstream Ascend command has passed on NVIDIA.
