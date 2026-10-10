# Card 13b2b: carry short-convolution state through a mixed cached graph

## Contract

Extend the existing two-range cached layer schedule so a `LayerKind::ShortConv`
layer consumes its named history and roll-index inputs and returns the updated
history as a typed cache root. Keep attention cache roots and their public tuple
API intact. Reuse `append_short_conv_cached_mixer_step` and existing graph
operations; add no computational `Op` or model-specific LFM path. The runtime
cache binder must use the same per-layer leaf names and declared shapes as the
graph, account for the state in cache memory, and decline row-only prefix/ring
serialization for this recurrent state.

## Worked schedule

Use one `Attention` layer followed by one `ShortConv` layer, with embedding
width 2 and convolution history length 3. The built graph has one attention
cache root and one short-convolution state root of shape `[2, 3]`. Its input
leaves include `kv_cache.0.k_even`,
`shortconv_cache.1.history`, and
`shortconv_cache.1.roll_indices`; the latter two match the runtime layer cache.

## Acceptance

Run the focused graph test:

```sh
nice -n 20 env CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test -p proxima-tensor --lib architecture_matrix_mixed_cached_conv_roots -- --nocapture
```

Require one passing named test and the exact marker
`attention_layers=1 shortconv_layers=1 kv_roots=1 conv_roots=1`.

Compile every std-gated interop cache consumer:

```sh
nice -n 20 env CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 cargo check -p proxima-model-interop --features std --lib
```

Require exit status 0. Then require `git diff --check` to emit no diagnostics.
Record the host, compiler, commands, full output, and exit statuses in
`evidence/card-13b2b-mixed-cached-shortconv-roots.txt`. This card establishes
graph construction and std cache-path compilation; it does not establish a
full checkpoint invocation or numeric LFM parity.
