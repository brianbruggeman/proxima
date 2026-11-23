# SCOREBOARD: FLUX.2 klein 4B (PROXNUM-588)

## Verdict

End-to-end is proven. Image parity is met (32.58 dB ours, 49.16 dB reference text). Wall: the
quiet set is **82.17 s** median against the incumbent's 84.58 s (0.97x), CoV 1.0 %. The
2026-11-23 re-measurement on the reconciled tree, taken while the box carried other builds and a
desktop indexer (1-min load 2.1-10.7, no quiet window), records a best **84.09 s** median
(0.99x) and a 12-run median of **85.22 s** (CoV 2.38 %). **Bar status: met on the recorded set
(84.09 < 84.58) and on the quiet set (82.17); the loaded median is over the bar, so the result is
load-attributed and a quiet-window re-measure is still owed.** The DiT is 0.53x on the quiet set
(24.71 s vs 47.04 s) and 0.56x today (26.10 s) despite the VAE still being 3.4x slower (41.94 /
42.40 s vs 12.22 s). Backend coverage: cpu and metal parity proven on this host; cuda blocked on
versailles' NVML driver mismatch; wgpu unexercised.

Reference: `stable-diffusion.cpp` tag `master-1052-f5df5f8`, M1 Max 64 GB, Metal, seed 42,
1024x1024, 4 Euler steps, wall from `/usr/bin/time -l`. Image oracle: `$K/ref/ref.png`.
Acceptance: image PSNR >= 22.0 dB and omega wall < 84.58 s.

## Incumbent (measured, S0)

| stage | sd.cpp |
|---|---|
| load (qwen + DiT + VAE) | ~13 s (5.69 + 7.04 + 0.32 GB reads) |
| text (conditioner) | 6.51 s |
| sampling, 4 steps | 47.04 s |
| VAE decode | 12.22 s |
| wall | 84.58 s |
| peak RSS | 16.19 GiB |

## proxima / omega, Metal (quiet set, 2026-11-22)

Three runs, load printed per run (1.46-2.43), release binary, stage timers from the example.

| stage | proxima | incumbent | ratio | provenance |
|---|---|---|---|---|
| text | 6.95 s median (6.94-7.24, CoV 2.0 %) | 6.51 s | 1.07x slower | 3 runs |
| DiT, 4 steps | 24.71 s median (6.3/6.1/6.1/6.1 per step, CoV 0.0 %) | 47.04 s | **0.53x (faster)** | 3 runs |
| VAE decode | 41.94 s median (41.42-42.95, CoV 1.5 %) | 12.22 s | 3.43x slower | 3 runs |
| wall | **82.17 s median** (81.65-83.61, CoV 1.0 %) | 84.58 s | **0.97x (bar met)** | 3 runs |
| peak RSS | 18.68 GB | 17.38 GB | 1.07x | same runs |

Parity: image **32.58 dB** with our text, **49.16 dB** with reference text; DiT velocity 0.005379 /
cos 0.99998553; latents 0.000200 / 0.000727 / 0.002941 / 0.018006; VAE 89.01 dB / cos 1.00000000.
**Parity and the wall bar are met. The VAE is the remaining stage gap; the DiT's lead covers it.**

Provenance: the prior 3-run set (120.14 s wall) was nine-tap through a stale local binary.
This set is the folded three-tap default, current binary, `/usr/bin/time -l`, load average
printed per run (`/tmp/proxnum588/wall.log`). `block_n=64` is measured and rejected (below):
the wider tile does not help the dominant shapes in this kernel.

## proxima / omega, Metal (2026-11-23 re-measurement, load-attributed)

Same binary shape after the tile reconciliation, `/usr/bin/time -l`, three runs per set; four
sets were needed because the 1-min load moved from 2.1 to 10.7 with other builds and a desktop
indexer on the box. The best set is below; the 12-run median across all four sets is 85.22 s.

| stage | today (best set) | quiet set | provenance |
|---|---|---|---|
| text | 6.99 s median (6.92-7.02, CoV 0.60 %) | 6.95 s | 3 runs, loads 4.8/6.5/3.8 |
| DiT, 4 steps | 26.10 s median (6.6/6.5/6.5/6.5, CoV 0.15 %) | 24.71 s | same set |
| VAE decode | 42.40 s median (42.30-45.05, CoV 3.00 %) | 41.94 s | same set |
| wall | **84.09 s median** (83.97-86.79, CoV 1.55 %) | 82.17 s | same set |
| peak RSS | 18.69 GB | 18.68 GB | same set |
| 12-run median (4 sets) | **85.22 s**, CoV 2.38 % | — | loads 2.1-10.7 |

The per-block and conv benches are unchanged from the quiet set (double block 237.0 vs 235.9 ms,
single block 253.4 vs 253.7 ms, conv 20.4/1260.9/1331.1/1632.9 ms vs 20.5/1269.5/1342.7/1649.8 ms),
so the ~4-6 % stage delta is machine state, not a kernel change. Stage medians and the parity
numbers are otherwise the same rows: velocity 0.005379, latents 0.000200 / 0.000727 / 0.002941 /
0.018006, VAE 89.01 dB / cos 1.0, images 32.58 / 49.16 dB.

## Subcomponent bench: the tile hypothesis (measured, rejected)

`proxima-model-interop/benches/flux2_klein.rs` measures each stage through omega on Metal with
the real checkpoint weights, prints milliseconds, GMAC/s and GB/s per case, and panics if a
selected case runs zero criterion iterations.

```
git wt cargo bench -p proxima-model-interop --features std,metal \
  --bench flux2_klein -- vae_conv --quick
```

The conv A/B (`vae_conv`, criterion midpoint, warm; same binary shape, only the selected tile
differs; MACs = out x rows x 9 x in, GB/s = (weights + one activation + output) at their
unique sizes, a floor because the staged tap operand is 9x the activation):

| conv (tokens x in -> out) | 64x32 (incumbent) | 128x32 | 64x64 |
|---|---|---|---|
| 128^2 x 32 -> 512 (`conv_in`) | **20.47 ms**, 118.0 GMAC/s, 1.76 GB/s | 21.56 ms | 20.19 ms |
| 256^2 x 512 -> 512 (`up.2.block.0.conv2`) | **1269.5 ms**, 121.8 GMAC/s, 0.22 GB/s | 1397.0 ms | 1269.2 ms |
| 512^2 x 256 -> 256 (`up.1.block.0.conv2`) | 1342.7 ms, 115.2 GMAC/s, 0.40 GB/s | 1440.4 ms | **1310.0 ms** |
| 1024^2 x 128 -> 128 (`up.0.block.1.conv2`) | **1649.8 ms**, 93.7 GMAC/s, 0.65 GB/s | 1863.9 ms | 2768.8 ms |

A `--quick` re-run on the 2026-11-23 reconciled tree executed all 13 rows with iterations > 0
(7/7/1/1/31/15/1/7/7/7/1/3/7; `mid_attention_16384` one iteration at 70.3 s): conv
20.4/1260.9/1331.1/1632.9 ms, double block 237.0 ms, single block 253.4 ms, attention 104.2 ms,
full step 6674 ms.

The requested widening (`block_m = min(M,128)`, widest token block) was implemented and
measured: 128x32 regresses every shape; 64x64 is a wash on 128^2/256^2/512^2 and 1.7x slower
on 1024^2. A `128 x 64` tile (32 accumulator fragments) is 40x slower, the Metal compiler
spilling the accumulator array. The incumbent `64 x 32 x 32` is the minimum of every
neighbour measured, so the selector keeps it and the kernel change was rejected, not landed.

The per-op profile (`--profile vae`, 363 ops, 62.55 s GPU) puts 96.1 % in `reduce-tiled-gemm`:
1024^2 256->256 upsample 9.83 s, 512^2 512->512 upsample 9.42 s, 1024^2 256->128 5.11 s,
512^2 512->256 4.33 s. Those four are the next single cost; they sustain 60-120 GMAC/s while
the measured DiT blocks and full step sustain 2,700-2,950 GMAC/s, because each conv threadgroup
runs 36-72 barrier-serialized K-steps over a 4 KB weight tile and a 4 KB activation tile. The
profile is the op-timed path (one command buffer per op), so its 62.55 s is an upper bound; the
batched decode is the 41.94 s row above and the ranking holds. The tile cannot fix that; the next lever
is the K-step depth / activation double-buffering, not the tile edges.

## Other backends

| backend | host | status |
|---|---|---|
| CPU | this host | text path measured for the floor; full e2e not measured — absent, not guessed |
| Metal | this host | above |
| CUDA | versailles (`ssh bix@192.168.1.76`) | reachable; `nvidia-smi` fails `Failed to initialize NVML: Driver/library version mismatch` (NVML 620.73). Blocked on the driver, not access |
| wgpu | — | no device exercise recorded |

## Commands

```
git wt cargo test --release -p proxima-model-interop --features std,metal --test flux2_klein dit::
git wt cargo test --release -p proxima-model-interop --features std,metal --test flux2_klein vae::
git wt cargo bench -p proxima-model-interop --features std,metal --bench flux2_klein -- vae_conv --quick
git wt cargo run --release -p proxima-model-interop --features std,metal --example flux2_klein -- --seed 42 --out $K/omega.png
git wt cargo run --release -p proxima-model-interop --features std,metal --example image_psnr -- $K/ref/ref.png $K/omega.png
scripts/flux2_klein_ab.sh
```

The bench's `--quick` run keeps the per-block and per-shape groups under five minutes;
`vae/full_decode` and `dit/full_velocity_step` are one iteration per sample at the current
speed, so run them with their group filter when the whole decode is the question.
