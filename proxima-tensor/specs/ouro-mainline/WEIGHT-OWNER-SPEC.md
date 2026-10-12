# shared checkpoint weights across forward descriptors

owner: Sol
status: independently admitted; counted execution artifacts recorded in TASKS.md and results/weight-owner/COMMANDS.md

## observed failure and reuse boundary

The full `--performance` command retains four `LoadedModel` arms in one lab
`Model`. Each HF descriptor load currently binds another `BoundWeights`, whose
owned norm vectors have different pointers under the same semantic resident
names. The existing Metal resident contract rejects rebinding a name to a
different pointer. A separate numeric-policy mismatch currently prevents the
real four-arm runner from reaching that boundary in the first attempt. After
numeric/math policy alignment, `results/cached-checkpoint/ouro-mainline-methods-live-r2.txt`
records the completed France baseline warmup followed by
`ResidentNameRebound { name: "token_embd.weight", cached_len: 402653184,
offered_len: 402653184 }`. The first attempted policy failure remains beside it.
Thus equal byte counts do not establish payload pointer identity. Retain both
failure directories and logs; independent fixture evidence remains in interop.

Do not change tensor names, clear all resident caches, or drop each arm before
the next. Keep the existing binding and graph builder, with one shared immutable
weight owner and independent forward graphs, KV state, prompt caches and expert
state per descriptor.

## exact source seam

1. Change the private `LoadedModel.weights` field in
   `proxima-model-interop/src/generate/load_model.rs` to
   `Arc<ModelWeights<'file>>`. `ModelWeights` is a private owner in
   `generate/model_weights.rs`, containing the completed `BoundWeights` and the
   existing checkpoint mapping view. It implements `Deref<Target=BoundWeights>`
   so evaluation reads the existing fields without new dispatch. Its `Drop`
   performs the existing resident-name and mapping cleanup. This owner is needed
   because cleanup must run exactly once when the Arc releases its payload,
   including concurrent final graph drops; a `strong_count == 1` check followed
   by cleanup is forbidden. `BoundProgram.weights` and binding functions retain
   their current mutable construction type. Every existing production constructor
   in `generate/pregather.rs` wraps its completed weight binding exactly once;
   update the full fixture initializer in `generate/tests_all.rs` too.
2. Store `safetensors_descriptor: Option<ModelDescriptor>` on `LoadedModel`.
   HF construction stores the descriptor clone; GGUF/registry constructors and
   fixtures store `None`. This is the original physical binding description,
   including Q/K pairing, norm choices, cache source and physical layer schedule.
3. Extract the existing HF post-binding construction body into
   `fn from_safetensors_weights(weights: Arc<ModelWeights<'file>>,
   architecture: ModelArchitecture, vocab: Vocab, descriptor: &ModelDescriptor,
   file_bytes: &'file [u8]) -> Result<Self, InteropError>` in `pregather.rs`.
   Reuse its `build_forward_with_readouts`, logical layer expansion, existing
   `single_range` admission, expert slab and `.validated()` calls. No second
   constructor body or transformer graph body.
4. Add the public cold method
   `LoadedModel::with_forward_descriptor(&self, descriptor: &ModelDescriptor)
   -> Result<Self, InteropError>`. Require a stored HF descriptor. Compare the
   supplied descriptor to that stored descriptor after replacing precisely
   `recurrence` and `loopcd` in a temporary clone. Any other difference rejects
   before graph construction with `InvalidRecurrentConfig`; non-HF origin also
   rejects explicitly. Reconstruct the physical `ModelArchitecture` by restoring
   its block count from the stored descriptor and its physical KV-head list.
   Clone the existing Arc and vocabulary, then call the shared constructor.
   The method changes only recurrence/readout composition, never weight layout.
5. Remove the per-graph `LoadedModel::drop` cleanup and transfer it to
   `ModelWeights::drop`. The Arc owns that destructor, with no count polling.
   Move the existing resident-name enumeration into the owner and delegate the
   existing LoadedModel helper to it; keep one enumeration body. Move or delegate
   the existing mapping identity field after checking all its consumers. Dropping
   a graph variant leaves the shared payload owner alive. Final owner destruction
   invokes existing cleanup once. Backend thread-local resource scope remains the
   current scope; this does not introduce cross-thread global resource caches.
6. In `examples/ouro_loopcd_lab/model.rs`, the first Arm binds its HF weights
   through the existing loader. Subsequent missing Arms call
   `with_forward_descriptor` on the first LoadedModel. Retain all arms in the
   existing vector and retain exact recurrent-config matching.

All imports stay at module scope. Arc ownership changes are cold load/graph
construction work; no Arc clone is added to a decode step. BoundWeights remains
immutable after load. No Cargo dependency is needed and allocation-free tiers
do not compile LoadedModel.

## literal fixture and controls

Reuse the existing tiny HF Ouro fixture, encoded as BF16 so learned norms and
embedding actually take the owned decode path. Its embedding rows are
`[3,4]` and `[0,5]`, norm gammas `[1,1]`, key/value/head identity matrices,
zero query/output/FFN matrices, epsilon one. One physical block, two iterations,
prefix `[0]`. Disabled output is `[3,4]*sqrt(2/52)`. Fixed logit strength .5
is `[3,4]*(1.5*sqrt(2/52)-.5*sqrt(2/27))`. Fixed hidden output is
`[-.26182377,-.34909838]`, with raw contrast
`[3,4]*(1.5*sqrt(2/27)-.5)`.

Three new synchronous named cases, in a narrow test module beside the HF
constructor (reuse fixture helper through test-only visibility):

- `shared_forward_weights_pointer_identity`: base plus fixed-logit and
  fixed-hidden variants have `Arc::ptr_eq` and equal every owned and packed
  payload pointer/length/codec. Capture full tiny payload values. Execute CPU
  roots against the literals above, and verify independent program/root and
  recurrence identities. A separately bound BF16 owner has different owned
  pointers while both owners remain live, providing the pointer negative control.
  Also drop two CPU-only shared graph owners concurrently using scoped native
  threads, with no active Metal residency or checkpoint mapping registrations;
  after both joins a Weak owner cannot upgrade. This control checks Arc payload
  destruction, not cross-thread cleanup of thread-local backend registrations.
  No sleeps, custom lock or
  strong-count-based cleanup. Source inspection retains the one Arc-owned
  destructor that performs resource cleanup.
- `shared_forward_weights_descriptor_rejection`: reject a changed Q/K pairing,
  changed norm choice, changed physical dimension and non-HF origin; retain
  typed errors. Admitted recurrence/readout changes retain shared payloads.
- `shared_forward_weights_survivor_drop`: Metal/macOS actual forward through
  base and a shared variant, drop base, then execute the survivor again against
  the full literal vector. Record actual host payload pointers and the existing
  resident-cache names, lengths and total bytes before/after drop. These
  accessors do not expose device object identities; no new Omega diagnostic API
  is added. Execute and drop all Metal owners on the owning thread; the final
  owner cleanup uses the existing resident release mechanism. Reject the
  separate-owner same-name control while both allocations remain live. No
  hardware simulation is counted as this case.

## native validation counts

Use the existing script-free nextest config and dedicated target. Exactly three
new cases (two CPU and one real Metal case), plus existing eight interop CPU and
three Metal recurrent cases:

```text
cargo nextest --config-file proxima-tensor/specs/ouro-mainline/results/shared-graph/validation/native-unit-nextest.toml run -p proxima-model-interop --features 'std metal metal-output-placement' --lib -E 'test(shared_forward_weights_) | test(recurrent_mainline_) | test(recurrent_metal_)' --no-capture --target-dir /private/tmp/proxima-ouro-mainline-target
```

Expected 14 executed cases, with no selected case omitted; report the separate
excluded-by-filter count. Retain actual GPU vectors and wrong
owner/descriptor controls. Run existing caller four and planner three selections,
alloc library floor (one artifact, zero errors), shared tensor selection 20,
and interop production-library strict clippy (zero diagnostics). Also run
all-target clippy with the command-line exceptions for `expect_used` and
`unwrap_used` permitted by the owner's test instructions; all other warnings
remain denied. The strict library run retains those production lints. Preserve
the initial all-target failures and the exact commands; do not add source
`allow` attributes or suppress the remaining diagnostics.

Run the existing mainline `--performance --backend metal --samples 5` entry with
all four arms live: exactly 48 forward records (eight warmups and forty measured)
across the two frozen prefixes. No substitution with eight isolated processes.
Retain failure artifacts and repeat after repairs.

Run C2's eight actual method/prefix cells and eight independent full-vector
comparisons: reuse the four archived France references; generate four native
ball references at strength .5, depth four, F32 mode. Do not rebuild the oracle
using shared tensor arithmetic.

Finally run a fresh cached GGUF checkpoint because the private owner type changes
all load arms: configured gate one, batch/microbatch 256, context 320, five
measured samples per case, profile/capture condition identical to CAC7 on-r3.
Expected 12 request records, two diagnostics, two summaries, twelve matched
profile brackets, zero drops, complete token/EOS/latency payloads. Label this as
a new source checkpoint; do not substitute it into the frozen CAC7 pair.
