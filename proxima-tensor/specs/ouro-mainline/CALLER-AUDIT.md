# Independent caller-spec admission

ADMIT

Opened `CALLER-SPEC.md`, `SHARED-GRAPH-SPEC.md`, and the actual production
`generate_from_ids` body at `generate/decode.rs:2277–2332`. Native nextest help
confirms --example/--features support; current manifests define all named
features, including omega/metal-tiled-gemm.

The initial audit refused one interface mismatch; the revised caller spec
resolves it by limiting diagnostics to completed-depth vectors:

The earlier `CALLER-SPEC.md` said "Full residual roots provide layer traces." The admitted
shared graph contract exposes completed-depth raw/normalized/head readouts but
does not promise per-logical-layer residual roots. Current TwoRange's
BuildForwardProgram residual vector is empty (`descriptor.rs` TwoRange arm),
and three frozen ordinary baselines explicitly preserve that empty vector.
The revised caller spec promises only declared completed-depth roots, explicitly
declines synthetic layer traces, and does not widen the shared component.
Ordinary baseline residual vectors remain empty. No private layer graph is
authorized to fill a diagnostic gap. This was the sole refusal item.

Other gates inspected:

- C1–C5 map to CAC1–CAC5; native checks bind 4 caller cases, 3 planner cases,
  1 example artifact, 2 cases and 12 requests per runtime.
- Cold/setup, diagnostic and cached timing paths are explicitly distinguished.
- Runtime config refusal before unsupported placement work is explicit; no
  hidden fallback is authorized.
- Same frozen panel IDs, every generated ID and latency, early EOS counts,
  failures and raw sample arrays are retained.
- Learned early exit is not inserted into the fixed-depth recurrent control.
- Source ownership separates tensor, interop and root caller changes; no second
  executor, tokenizer, checkpoint-transformer implementation or serving loop.

No caller builds or timing runs were executed during this audit.
