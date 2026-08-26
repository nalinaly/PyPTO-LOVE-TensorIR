# CP48 RowReductionV3 compiler and Cubin review

## Decision

Two independent reviews return GO with P0/P1/P2 equal to zero for PyPTO
`62eb88251df5bdad95277a9d619d20da9bf121eb`, tree
`04d3bca3e0b35b796f7745ded27a26dd61e25c67`.

## Source and diagnostic closure

The first real OFF build exposed two negative fixtures that assigned into
immutable `ArtifactArgumentDescriptor` values. Commit `62eb882` reconstructs
the complete descriptor vectors and preserves the exact shape-first and
stride-second rejection intent. Both reviewers verified the one-file `+6/-2`
delta, clean primary checkout and six clean gitlinks.

The resource watchdog paused only the owned build while an unrelated external
CUDA compiler used over 40 GiB RSS, then resumed it automatically after memory
recovered. No external process was signalled. Later command diagnostics are
also retained: disabled pytest plugin autoload made `-n 0` invalid, one copied
Base64 payload had bad padding, and two exploratory row tiles violated the
documented `power-of-two && tile <= rows` law. Corrected commands and legal
tiles pass without changing product bytes.

## Fresh products and tests

- OFF DSO: 438,701,720 bytes, SHA-256
  `95fc6579572faa5026cd3931da4b73773e2f879237e2c76088af8ca288c93451`.
- ON DSO: 784,224,056 bytes, SHA-256
  `e1213cf31972664a66012f95f1ebf003623dfebb54accdff3ab47cd6ca3e4220`.
- Both DSOs contain only the five expected system `DT_NEEDED` entries and no
  `RPATH` or `RUNPATH`.
- OFF CTest passes 11/11; LastTest SHA
  `106d5654cac225fe64d8e3d5864f25761be26f0efcff0f834cd6c45388191f49`.
- ON CTest passes 13/13; LastTest SHA
  `3ed4564516e04a17c6b0a88ca3e562909ac51ddfa61c92b81fab06dc0b2aae0b`.
- Exact-product Python passes 1/1 for OFF and ON; JUnit SHAs are
  `21db37ab...bb049` and `f98960d9...6d91`.

## Compiler and Cubin records

Run `pypto-20260826T090132Z-1506220-c226a2` produces four nonempty SM120
Cubins through `structured-tensorir`, with two pointers, zero workspace and no
fallback. The machine-readable report is 5,311 bytes, SHA-256
`d06765beaf4fd3ebec3c023b473a904bc704f6ae3a3491b157913ff49e338abb`.

Independent CUDA-hidden run `pypto-20260826T090758Z-1508157-9870e7` recompiles
all four cases from the exact ON DSO. Every canonical source digest, BuildSpec
byte/hash, Artifact byte/hash, Cubin byte/hash, input/result shape and stride,
grid, entry, semantic route, zero-workspace value and fallback flag matches the
report. Torch CUDA remains uninitialized and no runtime launch occurs.

## Claim boundary

This accepts the clean OFF/ON products, host-side tests and exact compiler/
Cubin records for the four rank-1/2/3 FP32/BF16 sum/max fixtures. It does not
accept GPU loading or launch, numerical correctness, general reduction
coverage, repeat determinism across builds, profiling or performance.
