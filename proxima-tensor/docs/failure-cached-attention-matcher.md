# failure: cached-attention matcher lacks provenance

Status: superseded failure record; no performance or quality verdict.

The attempted positive matcher could not be admitted from the current
`BoundOp` representation. `BoundOp` exposes node IDs, layouts, lookups, and
scalar bodies, but not the semantic roles needed to distinguish query from
cached/new key and value sources. The exact representation is
`proxima-tensor/src/bind.rs:201-267`.

Shape coincidence is not sufficient: selecting a cached-attention cluster from
extents or reduction bodies alone could fuse an unrelated graph. The
`weights_cached` negative control is also not available through the public
cached builder, which returns logits and cache roots at
`proxima-tensor/src/spec.rs:6030`.

The failed approach was a matcher over only `&[BoundOp]`. It is abandoned
because it would be a heuristic and because it cannot prove the semantic
roles. The next TODO is to attach the source provenance to one existing
`BoundOp` work item after binding, preserving node-role provenance, then pass
that bound item through the existing CPU and Metal pipes. A second physical
schedule is rejected because it would duplicate `BoundOpBuilder -> ReadyBatch
-> Interpreter` ordering and retirement. The matcher must still reject
effective outputs, external consumers, gathers, unsupported layouts/dtypes,
and any non-exact topology.

This record was superseded by the structural post-bind matcher in
`proxima-tensor/src/bind.rs`. It now passes positive synthetic and two-layer
cached-length-5 fixture tests, and the real Metal root test records CPU/Metal
`max_diff=0.0000047683716`. The original failure remains useful because it
identifies why the BoundOp-only matcher was abandoned; it is not a description
of the current implementation.
