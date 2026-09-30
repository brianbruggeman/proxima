# YaRN worked example (slice 5, /algorithm-development)

Primary source: `transformers/src/transformers/modeling_rope_utils.py`,
`_compute_yarn_parameters`, fetched from main 2026-09-29. Paper:
https://arxiv.org/abs/2309.00071.

## reference algorithm (transcribed, not paraphrased)

```
pos_freqs          = base ** (arange(0, dim, 2) / dim)
inv_extrapolation  = 1 / pos_freqs
inv_interpolation  = 1 / (factor * pos_freqs)
find_dim(r)        = dim * ln(orig / (r * 2*pi)) / (2 * ln(base))
low, high          = floor(find_dim(beta_fast)), ceil(find_dim(beta_slow))    # truncate=True default
low, high          = max(low, 0), min(high, dim - 1)
if low == high: high += 0.001
ramp(i)            = clamp((i - low) / (high - low), 0, 1)          # i in 0..dim/2
extrap_factor(i)   = 1 - ramp(i)
inv_freq(i)        = inv_interpolation(i) * (1 - extrap_factor(i)) + inv_extrapolation(i) * extrap_factor(i)
attention_factor   = 0.1 * ln(factor) + 1                           # when no attention_factor / mscale keys
cos, sin           = cos(pos * inv_freq) * attention_factor, sin(pos * inv_freq) * attention_factor
```

Defaults: `beta_fast = 32` and `beta_slow = 1`.

The attention factor multiplies both cos and sin, and both q and k are rotated. So the
attention logit is scaled by `attention_factor^2`. That is why no kernel change is needed.

## inputs

The inputs are the Qwen3-8B card's recommended YaRN settings:
- `base = 1e6`
- `dim = 128`, which gives 64 pairs
- `factor = 4`
- `orig = 32768`

## walk (computed with `bc -l`, scale 12)

| step | value |
|---|---|
| find_dim(32) | 23.595947608338 -> low = 23 |
| find_dim(1) | 39.650880710418 -> high = 40 |
| attention_factor | 1.138629436111 |

| pair | ramp | inv_freq |
|---|---|---|
| 0 | 0 (pure extrapolation) | 1.000000000000 |
| 20 | 0 | 0.013335214321 |
| 30 | 7/17 | 0.001064360980 |
| 40 | 1 (pure interpolation, original/4) | 0.000044456985 |
| 63 | 1 | 0.000000310234 |

These are the scaled cos/sin values the test asserts:

| position | pair | theta | cos x af | sin x af |
|---|---|---|---|---|
| 0 | all | 0 | 1.138629436109 | 0 |
| 32767 | 0 | 32767 | 1.118433966337 | 0.213500481795 |
| 32767 | 20 | 436.954967656207 | -1.096280897338 | -0.307644578868 |
| 32767 | 30 | 34.875916231660 | -1.081400139014 | -0.356441765391 |
| 32767 | 40 | 1.456722027495 | 0.129606833300 | 1.131229004905 |
| 32767 | 63 | 0.010165437478 | 1.138570605843 | 0.011574466996 |
| 131071 | 0 | 131071 | -0.931380090655 | -0.654987114049 |
| 131071 | 20 | 1747.859876267791 | 0.481312027679 | 1.031899086533 |
| 131071 | 30 | 139.506858009580 | 0.329971771655 | 1.089768609699 |
| 131071 | 40 | 5.827021480935 | 1.022203394571 | -0.501574733118 |
| 131071 | 63 | 0.040662680614 | 1.137688230341 | 0.046286967077 |

## tolerance, derived rather than chosen

`build_position_inputs` (`generate/residency_caches.rs:1126-1132`) computes theta in f32.
The reference uses torch f32 too: `pos_freqs` is float, and the position times inv_freq
matmul is float. At `pos = 131071` the f32 ulp is 2^-7 = 0.0078 rad. So pair 0's cos and
sin can differ from the f64 walk above by up to about `0.0078 x af = 0.0089`. That error
already exists today and is not introduced by YaRN.

The test therefore asserts:
- inv_freq and attention_factor to 1e-6 relative, against `bc -l` values at scale 22
  (the 12-digit table above is too short for 1e-6 relative at pair 63);
- cos/sin to `|theta| x 2^-22 x af + 1e-6` absolute.

The second bound counts two f32 roundings on theta, each at most 2^-23 relative: one
in `inv_freq` (`powf`) and one in the product `pos x inv_freq`. The result then goes
through a unit-slope sin/cos.

Amended 2026-09-29. The first draft used 2^-23, which counts only the product's
rounding. At position 131071, pair 30, that left 1.9e-5 of budget against a worst case
near 1.8e-5, so the test could fail on correct code.

## pairs the walk exercises

- the extrapolation branch: pairs 0 and 20
- the ramp interior: pair 30
- the interpolation branch: pairs 40 and 63
- the attention-factor scaling at position 0

Not exercised here: the `low == high` singularity guard. The test adds that case with
`beta_fast = beta_slow`, where both ends land on the same integer.
