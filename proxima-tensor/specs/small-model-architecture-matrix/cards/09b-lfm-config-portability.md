# Card 09b: keep LFM FFN width derivation portable

## Contract

`lfm_feed_forward_width` must derive the checkpoint FFN width in the crate's
no-std-compatible build. After rejecting non-finite and negative multipliers,
compute the scaled width as `f64`, reject values at or above the representable
`u64` boundary, and use Rust's float-to-integer cast truncation before the
existing `block_multiple_of` alignment. Do not add a dependency or a numeric
primitive.

## Worked value

The pinned config has `intermediate_size=12288`, multiplier `1.0`, and
`block_multiple_of=256`, producing width `8192`. With multiplier `1.03125`,
the scaled width is exactly `8448`, already aligned to 256; the derived width
must be `8448`.

## Acceptance

Run the library check and the pinned LFM schedule test:

```sh
cargo check -p proxima-model-interop --lib
cargo test -p proxima-model-interop --lib architecture_matrix_lfm_schedule -- --nocapture
```

Require both commands to exit 0, the test marker to include
`layers=16 schedule_matches=16`, and the fractional multiplier assertion to
produce `8448`. Retain host, compiler, complete test output, command statuses,
and `git diff --check` in
`evidence/card-09-lfm-config-native.txt`.
