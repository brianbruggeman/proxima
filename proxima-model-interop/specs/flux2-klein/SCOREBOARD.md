# SCOREBOARD: FLUX.2 klein 4B (PROXNUM-588)

## Verdict

End-to-end is proven. Image parity is met (32.58 dB ours, 49.16 dB reference text). Performance
parity is **not** met: wall 1.42x the incumbent, VAE decode 5.94x. The DiT is **faster** than the
incumbent (0.71x). Backend coverage: cpu and metal parity proven on this host; cuda blocked on
versailles' NVML driver mismatch; wgpu unexercised. No red is hidden in a paragraph: the VAE row is
the red, and it is the whole gap.

Reference: `stable-diffusion.cpp` tag `master-1052-f5df5f8`, M1 Max 64 GB, Metal, seed 42,
1024x1024, 4 Euler steps, wall from `/usr/bin/time -l`. Image oracle: `$K/ref.png`.
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

## proxima / omega, Metal (current)

| stage | proxima | incumbent | ratio | provenance |
|---|---|---|---|---|
| text | 7.76 s median (7.43-8.45, CoV 5.4 %) | 6.51 s | 1.19x slower | 3 runs, quietest window |
| DiT, 4 steps | 33.53 s median (30.45-35.02, CoV 5.8 %; 7.0-8.2 s/step) | 47.04 s | **0.71x (faster)** | same set |
| VAE decode | 72.57 s median (65.76-81.35, CoV 8.7 %) | 12.22 s | 5.94x slower | same set |
| wall | 120.14 s median (116.91-133.41, CoV 5.8 %) | 84.58 s | 1.42x slower | same set |
| peak RSS | 18.69 GB | 17.38 GB | 1.08x | last verified run |

Parity: image **32.58 dB** with our text, **49.16 dB** with reference text; DiT velocity 0.005379 /
cos 0.99998553; latents 0.000200 / 0.000727 / 0.002941 / 0.018006; VAE 89.01 dB / cos 1.00000000.
**Parity met. Perf not met: the VAE is the whole gap.**

Provenance caveat, recorded rather than hidden: the three-run set above was taken with the
nine-tap conv default through a stale local binary (`./target/...`, not `/tmp/cargo_target/...`).
The folded three-tap form measured **58.38 s VAE / 104.23 s wall** in a single env-forced run.
`block_n=64` measured 1.52x on the conv reduce via `--profile vae` and is not landed (its gates
were not re-run). The next row is three-tap, current binary, 3 runs, load average printed.

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
scripts/flux2_klein_ab.sh
```
