# Card 13b2a: slice a fused short-convolution projection

## Contract

The scheduled short-convolution layer consumes its checkpoint projection as
one row-major `[3 * embedding, embedding]` matrix named
`blk.{layer}.shortconv.in_proj.weight`. Build B, C, and X row views at offsets
`0`, `embedding`, and `2 * embedding` using the existing affine
`Elementwise::Identity` graph operation, then pass those views into the shared
short-convolution mixer composition. Do not add a tensor op or split/copy the
checkpoint tensor in the binder.

## Worked vector

For `embedding=2`, use fused rows
`[[1,2],[3,4],[5,6],[7,8],[9,10],[11,12]]`. The three views must be:

- B: `[[1,2],[3,4]]`
- C: `[[5,6],[7,8]]`
- X: `[[9,10],[11,12]]`

Each output has shape `[2,2]`, and the twelve values remain in row-major order.

## Acceptance

Run:

```sh
taskpolicy -b nice -n 20 env CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test -p proxima-tensor --lib architecture_matrix_shortconv_fused_projection_row_views -- --nocapture
```

Require one passing named test and the exact marker
`branches=3 shapes=3 values=12 offsets=0,2,4`. The test evaluates the
existing graph operations over synthetic F32 weights; it does not invoke a
checkpoint runtime or claim a full model result. Record the command, host,
compiler, complete output, exit status, and `git diff --check` in
`evidence/card-13b2a-shortconv-fused-projection.txt`.
