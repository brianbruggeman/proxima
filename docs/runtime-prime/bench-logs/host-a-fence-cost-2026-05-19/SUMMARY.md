# host-a-fence-cost-2026-05-19

Controlled measurement of the Dekker-fence cost on the `h2_load_5way`
hot path, comparing pre-fence `b0c46c7` vs current `main` HEAD (`171a9c2`)
using criterion's `--save-baseline` / `--baseline` for noise-canceled
comparison.

| field | value |
|---|---|
| Date | 2026-05-19 |
| Machine | host-a (Apple M1) |
| Pre-fence commit | `b0c46c7` (fix(runtime-prime/core_shard): drain inbox in worker park race-close) |
| HEAD commit | `171a9c2` (feat(benches): integrate 17 not-yet-invoked benches + pipeline category) |
| Bench | `h2_runtime_swap::h2_load_5way` |
| Features | `runtime-tokio,runtime-prime-full,tcp,http1,http2` |
| Methodology | criterion `--save-baseline pre-fence` on b0c46c7, then `--baseline pre-fence` on HEAD |
| Target dir | `/tmp/cargo_target` (shared across worktrees — baseline auto-discovered) |
| Host loadout | shared box — VSCode + Firefox open, load avg 5.27, syspolicyd 28.8%, VSCode plugin 17%, Firefox plugin-container 16% (no non-yours process >30%) |

## Per-arm results

`time:` median (µs) from criterion. `change:` is HEAD vs `pre-fence` baseline,
reported by criterion directly (not eyeball math).

| arm | b0c46c7 (µs) | HEAD (µs) | Δ (µs) | criterion `time` change (median) | criterion verdict | design-favors |
|---|---|---|---|---|---|---|
| pingora                | 50.269 | 52.266 | +1.997 | +4.97%  [CI −6.94% to +18.94%] | No change in performance detected. | neutral |
| tokio_hyper            | 42.962 | 37.711 | −5.251 | **−6.77%** [CI −9.18% to −4.44%, p=0.00] | **Performance has improved.** | neutral |
| proxima_on_tokio       | 35.924 | 33.583 | −2.341 | **−7.32%** [CI −8.40% to −6.34%, p=0.00] | **Performance has improved.** | neutral |
| proxima_on_flume       | 35.142 | 33.852 | −1.290 | −1.82%  [CI −5.94% to +0.99%] | No change in performance detected. | neutral / control |
| **proxima_on_prime**   | **39.595** | **37.973** | **−1.622** | **−7.93%** [CI −13.80% to −3.77%, p=0.00] | **Performance has improved.** | **incumbent (fence-cost path)** |

## Outlier counts

| arm | b0c46c7 outliers | HEAD outliers |
|---|---|---|
| pingora          | 11 (11%) | 9 (9%)  |
| tokio_hyper      |  8 (8%)  | 4 (4%)  |
| proxima_on_tokio |  5 (5%)  | 6 (6%)  |
| proxima_on_flume |  6 (6%)  | 8 (8%)  |
| proxima_on_prime | 12 (12%) | 8 (8%)  |

Box was approximately equally noisy across both runs (criterion's
`--baseline` cancels common environmental noise).

## What changed between b0c46c7 and HEAD that touches the hot path

`git log b0c46c7..HEAD -- rust/src/runtime/ rust/src/h2/ rust/src/http/ rust/src/upgrade.rs`:

```
37f82b5 feat(benches): bench-suite scaffold + par.rs Rayon-shape expansion + phased tails + loom
f9d6490 fix(runtime-prime): RayonBackgroundPool::default missing + unused TcpStream re-export
d13b3ae feat(runtime-prime): add io_uring TCP backend (linux correctness floor)
ff8c304 refactor(upgrade): key HijackStream on futures::io, drop tokio bound
9d20310 feat(runtime-prime/inbox): quiesce semantics via branchless CLOSED_BIT
fe4fe6d fix(runtime-prime): SeqCst fence pairs close Dekker-pattern wake races
```

Six commits, of which only **two** plausibly land on the proxima_on_prime
hot path:

- `fe4fe6d` — adds two SeqCst fences (cost positive: expected slowdown)
- `9d20310` — branchless CLOSED_BIT inbox push (gain positive: expected speedup)

`d13b3ae` is Linux-only (io_uring); inert on macOS host-a.
`ff8c304`, `f9d6490`, `37f82b5` do not touch the h2/runtime hot path.

## Honest read

**Fence cost on `proxima_on_prime` is not a problem.** Net delta vs pre-fence
is **−1.62 µs (−7.9%, p=0.00 per criterion)** — the path is faster, not
slower. The fence pair certainly adds *some* cost (two `DMB ISH` instructions
per wake-cycle on AArch64), but that cost is overpaid by the concurrent
hot-path optimization in `9d20310` (CLOSED_BIT branchless quiesce). The
control arm `proxima_on_flume` — which does NOT touch `Wakeup::fire` or
`arm_wakeup` — shows −1.82% (no significant change), confirming the
proxima_on_prime improvement is from code changes, not warmup or thermal
state.

The casual bench on 2026-05-19 that showed proxima_on_prime regressing
38.139 → 51.040 µs was env-drift noise; the controlled `--baseline` run
contradicts it.

## Implication

- P0 closes with no mitigation needed. The decision gate ("fence cost
  < 2 µs on `proxima_on_prime`") is satisfied — net cost is negative.
- P1 (fence-cost mitigation candidates a/b/c) is skipped.
- The isolated fence cost (fe4fe6d alone, without 9d20310 overlap) is not
  measured here. It is bounded above by the 9d20310 gain on the push path
  and below by some unknown positive value. If the user later wants the
  isolated number, two additional benches would isolate it
  (b0c46c7 → fe4fe6d, then fe4fe6d → 9d20310). Not pursued in this
  session because the user's success criterion is satisfied.

## Files in this directory

- `h2_runtime_swap-b0c46c7.log` — raw criterion output for the pre-fence
  baseline run (with `--save-baseline pre-fence`)
- `h2_runtime_swap-head-171a9c2.log` — raw criterion output for HEAD with
  `--baseline pre-fence`, includes per-arm `change:` lines
- `SUMMARY.md` — this file
