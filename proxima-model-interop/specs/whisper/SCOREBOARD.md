# Whisper scoreboard

Measured on the M1 Max (macOS) for CPU/Metal and the incumbent; the CUDA row
is the Versailles host (RTX 2070). Every proxima row also checks token parity
against `ref-f32/tokens.i32` (tiny) or `ref-turbo/tokens.i32` (large-v3-turbo).

| date | model | device | engine | wall | vs whisper.cpp | ids | text | note |
|---|---|---|---|---|---:|---:|---|
| 2026-11-22 | tiny | CPU | proxima | 20.92 s | 41.3× slower | 24/24 | exact | `evaluate_named` interpreter |
| 2026-11-22 | tiny | Metal | proxima | 3.50 s | 6.9× slower | 24/24 | exact | per-step plan; decode 2.70 s |
| 2026-11-22 | tiny | Metal | whisper.cpp v1.9.5 | 0.506 s | 1.00 | 24/24 | exact | incumbent, `-ng` off |
| 2026-11-22 | large-v3-turbo | CPU | proxima | 150.92 s | 114.5× slower | 27/27 | exact | 32-layer encoder |
| 2026-11-22 | large-v3-turbo | Metal | proxima | 62.14 s | 47.2× slower | 27/27 | exact | encode 39.95 s, decode 21.12 s |
| 2026-11-22 | large-v3-turbo | Metal | whisper.cpp v1.9.5 | 1.318 s | 1.00 | 27/27 | exact | incumbent |

## Stage rows (proxima)

| model | device | load | mel | encode | decode | decode/token |
|---|---|---:|---:|---:|---:|---:|
| tiny | CPU | 0.051 s | 0.075 s | 15.743 s | 4.865 s | 202.7 ms |
| tiny | Metal | 0.041 s | 0.160 s | 0.413 s | 2.697 s | 112.4 ms |
| turbo | Metal | 0.871 s | 0.195 s | 39.948 s | 21.122 s | 782.3 ms |

## Device matrix

| device | proof | status |
|---|---|---|
| CPU | 24/24 and 27/27 ids, exact text | **pass**, speed gap recorded |
| Metal | 24/24 and 27/27 ids, exact text; stage timers above | **pass** for execution and parity; speed gap recorded |
| CUDA (Versailles) | host reachable (`ssh versailles`); RTX 2070 present; `nvidia-smi` fails with `NVML driver/library version mismatch` (loaded module 615.78.08, userspace 620.73.04); `nvcc` absent | **blocked on host maintenance**, not on the code: rebuild/load the 620 module and install the CUDA toolkit |

### CUDA unblock commands (Versailles, owner-authorized maintenance)

```sh
sudo dkms autoinstall
sudo mkinitcpio -P
# reboot into the 620.73.04 module, then:
nvidia-smi
sudo pacman -S cuda
nvcc --version
```

After that, build on Versailles:

```sh
cargo build --release -p proxima-model-interop \
  --features std,toml,proxima-autograd,cuda --example whisper
```

## The gap, named

- **Decode** is planned 24/27 times, once per token count; that plan cost is
  the 2.70 s tiny / 21.12 s turbo row. A cached-KV step or one max-length plan
  is the next change.
- **Encode** on turbo is 39.95 s against the incumbent's 0.444 s; the generic
  graph is not yet fused into the same kernels whisper.cpp uses.
- **Mel** on Metal is slower than CPU (0.160 s vs 0.075 s); the cheap win is
  to run the STFT on the host and keep encode/decode on the device.
- The bar is the owner's 5% (SPEC AC14); none of these rows meet it yet.
