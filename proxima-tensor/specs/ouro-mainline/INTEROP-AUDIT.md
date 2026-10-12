# Interop spec audit

VERDICT: ADMIT
gates: G1 pass, G2 pass, G3 pass, G4 pass, G5 pass, G6 pass, G7 pass
counts: R 6 / discharged 6 | AC 4 / orphaned 0 | slices 1 / unvalidated 0 | paths 15 / dangling 0

## Counts and checked contracts

- Requirements I1-I6 are each named by at least one of IAC1-IAC4; no AC is orphaned.
- IAC1 expects 8 CPU cases; IAC2 expects 3 Metal cases; IAC3 expects 1 library artifact and 0 compiler errors; IAC4 expects 0 compiler/clippy errors.
- The single interop slice gives IAC1 and IAC2-IAC4 as its validation set with count-bearing expected values.
- `INTEROP-SPEC.md:81-89` requires wrong-depth, wrong HF permutation, and aliased-cache controls to fail the same assertions as the positive full-vector fixture.
- `INTEROP-SPEC.md:124-128` bounds the component: no checkpoint IO, tensor execution, backend kernels, alternative sampler, or duplicate cache/plan owner.
- Context paths: 15 checked, 0 dangling. The prompt-cache key now resolves to `proxima-model-interop/src/generate/prompt_cache_key.rs`.
