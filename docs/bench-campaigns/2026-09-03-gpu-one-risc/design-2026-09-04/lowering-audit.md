# omega lowering audit - Op -> BoundOpKind -> kernel body

Read-only. Paths relative to /Users/brianbruggeman/repos/slot-0/proxima
STRUCTURE = Op/ScalarOp/IndexMap/Layout/extents/codec enum.
NAME = a string, a substring of generated source, a bool flag, an operand count, a build-feature flag, a model/tensor name, an env var.

## A. Decision table - emitter (msl.rs)
Format: N. decision -- file:line -- chosen by -- selects -- defect if NAME

1. which renderer -- omega/src/msl.rs:875-886 -- STRUCTURE (BoundOpKind + Keep) -- cached-attention/elementwise/reduce/scan/iota/constant -- n/a
2. cooperative vs serial fold -- omega/src/msl.rs:1039-1053 -- STRUCTURE (Keep, ScalarOp, gather_count, reduce extents) + build const COOPERATIVE_REDUCE_MIN_LEN -- push_cooperative_reduce_body vs push_serial_reduce_body -- n/a
3. row-blocked packed matvec eligibility -- omega/src/msl.rs:1389-1477 -- STRUCTURE (codec enum, Layout::stride, extents, axes_fold_contiguously) -- push_packed_row_blocked_body -- n/a
3b. operand-count gate inside 3 -- omega/src/msl.rs:1404 -- NAME (quantized.len() != 2) -- rejects any matmul with a third operand -- an arity literal, not a shape predicate; a bias-fused matmul is demoted to the generic path
4. tiled simdgroup_matrix GEMM eligibility -- omega/src/msl.rs:1575-1692 -- NAME (cfg(not(feature="metal-tiled-gemm")) -> FeatureDisabled, :1582-1586) then STRUCTURE -- push_tiled_gemm_body -- the classifier is short-circuited by a compile flag; TiledGemmRejection::FeatureDisabled describes the build, not the op
4b. Q4_K-only narrowing in 4 -- omega/src/msl.rs:1602-1604 -- STRUCTURE (codec enum) -- rejects Q5_K/Q6_K -- stated reason at :1596-1601 is "never measured"; policy encoded as a structural gate
5. plain_product fast body -- omega/src/msl.rs:3274-3277 -- MIXED: NAME (element_type == "float" string compare) + cfg!(feature="metal-q5k-pair-dot") + STRUCTURE (codec, is_plain_product_reduce) -- q4k_pair_dot/q5k_pair_dot vs scale-deferred vs per-element -- the dtype decision is re-derived from the rendered type token produced at msl.rs:1940-1958, not from DType
6. single-fetch lane remap -- omega/src/msl.rs:3301-3304 -- NAME (2 feature flags) -- push_q4k_single_fetch_body -- body identity depends on the build invocation; :3289-3300 records both-on produced a sum ~7x too large
7. verbatim ggml port body -- omega/src/msl.rs:3313-3315 -- NAME (2 feature flags) -- push_q4k_ggml_port_body -- same
8. Q4_K header decode variant -- omega/src/msl.rs:2944 / 2949 -- NAME (metal-q4k-mask-fma) -- q4k_header_for vs q4k_header_for_bf -- two whole function definitions selected by cfg
9. Q4_K product-reduce body -- omega/src/msl.rs:2988 / 3009 -- NAME (metal-q4k-mask-fma) -- shift-then-mask vs mask-without-shift -- same
10. packed-row combine/write -- omega/src/msl.rs:3059 / 3116 -- NAME (metal-q4k-split-k) -- plain write vs threadgroup split-K combine -- same
11. split-K loop stride in body -- omega/src/msl.rs:3345-3358 -- NAME (cfg! metal-q4k-split-k) -- interleaved vs plain super-block loop -- same
12. split-K factor -- omega/src/msl.rs:1783-1793 / 1801-1803 -- NAME (feature) + build consts -- simdgroups per row-group -- same
13. Q4_K super-block tiled cooperative arm -- omega/src/msl.rs:4344-4390 -- STRUCTURE (codec, reduce-dim count, stride, extent multiple) -- pins width to SIMD_WIDTH -- n/a
14. cooperative threadgroup width -- omega/src/msl.rs:4390-4406 / 4413-4419 -- NAME (metal-wide-cooperative-reduce) + STRUCTURE (extents) -- {width} baked into source at :4540, :4542, :4706, :4710, :4738 -- a concrete-extent function baked into source text that kernel_cache_key does not carry (H1)
15. nsg widening factor -- omega/src/msl.rs:4446-4448 / 4457-4459 -- NAME (3-way cfg) -- threadgroup width x2 -- same
16. grid thread count -- omega/src/msl.rs:1818-1888 -- STRUCTURE (kind, extents, output_axes) via 2/3/4 -- dispatch size -- n/a
17. threadgroup width -- omega/src/msl.rs:4265-4332 -- STRUCTURE (kind, query_groups, gates) -- GridSpec::threadgroup_width -- branch at :4322 unreachable (H4)
18. tiled-GEMM body vs stub -- omega/src/msl.rs:3961 / 4234 -- NAME (feature) -- real body vs unreachable! stub -- n/a
19. MSL scalar type -- omega/src/msl.rs:1940-1958 -- STRUCTURE (DType) producing NAME output ("half"/"float") -- declared element type -- 8 DType variants collapse to 2 strings; Int32/UInt8/Bool all become "float"; decision 5 keys on the collapsed string
20. packed operand read expression -- omega/src/msl.rs:2390-2424 -- STRUCTURE (PackedCodec) -- q4k_element/q5k_element/... -- n/a
21. buffer binding type -- omega/src/msl.rs:2189-2191 -- STRUCTURE (Option<PackedCodec>) -- device const half* vs uchar* -- n/a
22. pipeline cache identity -- omega/src/msl.rs:930-981 -- NAME (a String) -- which compiled pipeline is reused -- rows 1-21 re-encoded as chars '4','5','6','8','0','h','b','f' then 'G'/'B'/'S'; its own doc at :898-921 lists three axes, row 14 is a fourth (H1)
23. scatter rejection -- omega/src/msl.rs:1176-1180 -- STRUCTURE (out_scatter.is_some()) -- EmitError::ScatterNotSupported -- n/a

## B. Decision table - driver (metal.rs, backend.rs)

24. placements vs no-placements entry point -- omega/src/lib.rs:55-70, omega/src/metal.rs:977-978 -- NAME (feature metal-output-placement plus which of 8 exported fns the caller typed) -- whole execute path -- 8 public execute* entry points, a cfg/name cross product rather than a parameter
25. plan arena vs per-call alloc -- omega/src/metal.rs:488-508, 341-427 -- NAME (metal-plan-stable-buffers) -- BufferArena + PlanUniforms vs fresh buffers -- also changes the public backend::Plan::Metal payload to Box<metal::Plan> (backend.rs:211-214)
26. encoder dispatch type -- omega/src/metal.rs:1045-1062 -- NAME (metal-concurrent-dispatch) -- Serial vs Concurrent encoder -- n/a
27. barrier emission -- omega/src/metal.rs:1094-1119 -- STRUCTURE (HazardTracker, RAW/WAW/WAR over buffer pointer identity) -- memoryBarrierWithScope -- n/a
28. output buffer allocation -- omega/src/metal.rs:2402 / 2424 -- NAME (metal-buffer-pool) -- thread-local pool vs newBufferWithLength -- n/a
29. placement vs allocate per op -- omega/src/metal.rs:3789-3795 -- NAME (Option<(&MetalBuffer, usize)>) -- where the op writes -- an unexplained Option at the hot seam; no type distinguishes a placed plan from an unplaced one
30. uniforms source -- omega/src/metal.rs:3796-3806 -- NAME (Option<&MetalBuffer> plus cfg) -- plan-owned uniform vs content-keyed upload cache -- same shape as 29
31. gather fault buffer -- omega/src/metal.rs:3807-3810 -- STRUCTURE (gather_count(bound) > 0) -- fault buffer binding -- n/a
32. pipeline cache hit -- omega/src/metal.rs:2374-2400 -- NAME (string key) -- returns a cached pipeline without re-emitting or comparing source -- the string from row 22 is the sole identity of a compiled kernel
33. op profile bucket -- omega/src/metal.rs:1631-1691 -- NAME (substring of generated MSL: "simdgroup_multiply_accumulate", "q4k_pair_dot(blk", "acc1_0", "simd_sum(") -- the label every default-on/off decision was measured against -- re-runs emit() and greps text; :1666-1672 records it mislabelled 216 of 225 ops
34. packed kernel variant label -- omega/src/metal.rs:1694-1719 -- NAME (substring of generated MSL) -- q4k-paired/q4k-run8/q5k-paired/... -- same
35. backend selection default -- omega/src/backend.rs:128-136 -- NAME (env var OMEGA_BACKEND) -- Cpu/Metal/Wgpu/... -- process-global OnceLock parsed from a string; unknown names silently fall back (:132-134)
36. backend name parse -- omega/src/backend.rs:147-163 -- NAME (string) -- Backend variant -- n/a
37. backend execution arm -- omega/src/backend.rs:265-375 -- STRUCTURE (Backend enum) plus NAME (per-arm cfg) -- cpu/metal/wgpu plan -- four of seven variants are name reservations with no arm
38. block -> codec -- omega/src/metal.rs:450-456 -- STRUCTURE (QuantizedBlock enum) -- PackedCodec -- n/a
39. residency -- omega/src/backend.rs:400 -- NAME (BTreeSet<&str> of tensor names); the parameter is _resident_names and unused -- nothing -- a public API taking tensor names that discards them
40. weight binding -- omega/src/backend.rs:258-263, metal.rs:1619 -- NAME (&[(&str, QuantizedBlock)] -> resolve_named_blocks) -- which bytes bind to which Op::Input -- argued for at op.rs:180-184, but still a string keyspace at the executor boundary
41. wgsl cooperative vs serial -- omega/src/wgsl.rs:186-206, 244-258 -- STRUCTURE plus runtime WgslCaps.subgroup_size: Option<u32> -- render_reduce_cooperative vs render_reduce -- the only decision in the crate consulting real device capability; MSL and CUDA hardcode 32
42. cuda cooperative -- omega/src/cuda.rs:356-366 -- STRUCTURE -- __shfl_down_sync path -- n/a
43. cuda/wgsl unsupported kinds -- omega/src/cuda.rs:161-177, omega/src/wgsl.rs:209-214 -- STRUCTURE (BoundOpKind) producing NAME output (kind: &'static str) -- EmitError::{CudaUnsupportedOpKind, UnsupportedOpKind} -- the variant is destroyed into "iota"/"constant"/"cached_attention"; two error variants differ only by which backend raised them

Counts: 43 rows. STRUCTURE-only: 16 (1, 2, 3, 4b, 13, 16, 17, 20, 21, 23, 27, 31, 38, 41, 42, and 37's enum half). NAME or NAME-mixed: 27. Of the NAME rows, 14 are build-feature flags, 4 are strings or substrings of generated source, 2 are env/tensor names, the rest are Option/arity/bools.

## C. Module table

omega/src/lib.rs        86 lines  - crate root; tier gate, re-exports across 6 cfg combinations
omega/src/msl.rs      6576 lines  - MSL emitter plus every Metal eligibility classifier and dispatch-geometry function
omega/src/wgsl.rs     1948 lines  - WGSL emitter, independent re-derivation of the same lowering
omega/src/cuda.rs     1857 lines  - CUDA C emitter, independent re-derivation; no driver exists
omega/src/metal.rs    4796 lines  - macOS Metal driver: plan, buffers, uniforms, encode, dispatch, readback, profiling
omega/src/wgpu_driver.rs 888 lines - wgpu driver over wgsl
omega/src/backend.rs   630 lines  - backend-name fan-out; Backend, Plan, BackendError
omega/src/error.rs      84 lines  - EmitError, shared by all three emitters
omega/src/sized.rs     101 lines  - build-time constants

No lower, ir, or kernel module exists; ls omega/src returns exactly these nine files.

Duplicated responsibility, by identical function names across the three emitters:

- 28 names in all three of msl.rs, wgsl.rs, cuda.rs: bindings, body_fingerprint, body_token, cooperative_identity_token, entry_name, fold_init_tokens, grid_threads, init_token, is_cooperative_reduce_op, is_leaf, keep_token, kernel_signature, op_token, operand_codecs, preamble, push_body_steps, push_gather_fault_check, push_gather_fetch, push_gather_uniform_fields, reduce_is_cooperative, reduction_dims, render_elementwise, render_reduce, render_scan, scalar_op_expr, type_token, validate, validate_body.
- 2 more in msl+wgsl (render_iota, render_constant); 5 more in msl+cuda (gather_count, gather_slots, operand_read, push_cooperative_reduce_body, push_serial_reduce_body).
- cuda.rs has 35 top-level functions; 28 (80 percent) are name-duplicates of msl.rs.
- One item is genuinely shared: msl::reduction_dims is pub(crate) for metal.rs (msl.rs:1195-1199) with a doc naming the drift hazard. wgsl.rs:441 and cuda.rs:265 each carry their own copy anyway.

## D. CISC list - nodes carrying semantics rather than structure

1. BoundOpKind::CachedAttention -- proxima-tensor/src/bind.rs:240-251. Nine semantic fields (query_rows, cached_key_rows, new_key_rows, kv_heads, query_groups, head_dim, scale, cached_lower_inclusive, new_upper_inclusive). head_dim and query_groups restate quantities extents already carries; scale is a folded Constant. It is the only variant with a semantic name, and:
   - operand count is a discriminator: 8 vs 9 changes which field is read (bind.rs:236-239, "new_upper_inclusive here is unused filler" at 9). A runtime length stands in for a two-variant enum, and the invalidated field stays in the struct carrying garbage.
   - no element_body (bind.rs:326 returns EMPTY_BODY).
   - no split axis (bind.rs:409 returns None) - cannot be parallel-chunked.
   - no lowering on two of three backends: wgsl.rs:209-214 and cuda.rs:173-178 return UnsupportedOpKind; wgpu_driver.rs:533 excludes it.
2. The pattern matcher that mints it -- proxima-tensor/src/bind.rs:2038-2274 (cached_attention_candidates, 236 lines) plus cached_attention_single_range_candidates (:2276+). A hard-coded template Multiply(Add(Reduce(..)), Reciprocal(Add(Reduce(..)))) with exact-causal-mask sub-matchers (is_exact_causal_mask :1990, exact_merged_causal_mask_cached_len :2016). A program spelling the same math with a different node order, a fused body, or a different mask composition silently misses fusion and takes the longer path (omega/Cargo.toml:63 records emit_calls 938 -> 616). Nothing reports the miss.
3. The fusion is a build feature -- cached-attention-streaming, 20 cfg sites in bind.rs. The IR's effective variant set depends on a build flag while the variant itself is always compiled, so every backend and the CPU interpreter carry an arm for a node one build can never produce.
4. Codec-specific eligibility inside the emitter -- PackedRowBlockRejection::NotKQuantCodec (msl.rs:1332-1345, 1424-1429) and TiledGemmRejection::NotQ4K (msl.rs:1503-1506, 1602-1604). A codec is a first-class routing key at the kernel-body level rather than a decode expression plugged into one body.
5. A renderer keying on generated source text -- metal.rs:1631-1691, 1694-1719. Same class as a tensor-name key: the classification that produced the kernel is thrown away and reconstructed by grepping the output.

## E. Headline defects

H1 - the pipeline cache key does not cover every axis the emitted source varies on. kernel_cache_key (msl.rs:930-981) carries entry_name (rank, output-rank, operand count, body, reduce op, keep, init, gather shape), the type token, per-operand codec chars, one of 'G'/'B'/'S', and the output_axes sequence. It does not carry the reduce extents. cooperative_reduce_width (msl.rs:4390-4406) computes reduction_total.div_ceil(4).next_multiple_of(32).clamp(32, WIDE_COOPERATIVE_REDUCE_MAX_WIDTH) from concrete extents, and that value is written into the source at msl.rs:4540 (long output_index = (long)gid / {width};), :4542, :4706, :4710, :4738. pipeline_for (metal.rs:2374-2400) returns the cached pipeline on a key hit without re-emitting or comparing source. metal-wide-cooperative-reduce is in the default metal feature set (omega/Cargo.toml:75-97). Two cooperative reduces sharing rank, output_axes, codecs, reduce op, init and keep but differing in reduce extent collide on one compiled kernel with the wrong lane stride. The key's own doc (msl.rs:898-921) states the invariant and enumerates three axes; this is a fourth. No reproduction executed.

H2 - the profiler that every default-on decision was measured against classifies by grepping generated source. classify_kind (metal.rs:1631-1691) and classify_packed_kernel_variant (metal.rs:1694-1719) call emit() a second time and match substrings. The comment at metal.rs:1666-1672 records this already misreported: reduce-packed-row-blocked read 9 instead of 225 ops, reduce-cooperative overcounted by 216, found only by a bake-off. omega/Cargo.toml:33-98 cites per-op profile numbers from this instrument as the reason metal-q5k-pair-dot, metal-concurrent-dispatch and cached-attention-streaming are default-on.

H3 - 14 kernel bodies and dispatch shapes are selected by build feature, not by the op: metal-tiled-gemm, metal-q4k-mask-fma, metal-q4k-single-fetch, metal-wide-cooperative-reduce, metal-buffer-pool, metal-plan-stable-buffers, metal-concurrent-dispatch, metal-q4k-split-k, metal-packed-row-nsg2, metal-q5k-pair-dot, metal-output-placement, metal-q4k-ggml-port, cached-attention-streaming, kv-capacity-bucket (omega/Cargo.toml:99-320). They do not compose: msl.rs:3289-3300 records a sum ~7x too large with two on; msl.rs:4310-4319 records ggml-port plus split-K "broke Q4_K parity outright, relative=1". Three pairwise exclusions are hand-coded (msl.rs:3301-3304, 3313-3315, 4446-4459). The set of runnable configurations is not enumerable from the type system.

H4 - a build-time tunable's only consumer is unreachable code. msl.rs:4322 is the sole code reference to crate::sized::PACKED_ROW_BLOCK_SIMDGROUPS. It sits after the if-let BoundOpKind::Reduce { keep: Keep::Reduce, .. } block at :4279-4321; packed_row_block returns Some only for Keep::Reduce ops (msl.rs:1393 -> reduce_is_cooperative :1041-1043), and every such op already returned inside that block. The author's own comment at :4319-4322 states the adjacent arm is unreachable dead code for exactly this reason. Hanging off that line: an omega-runtime.toml section, a build.rs emitter (build.rs:191) and cross-axis validator (build.rs:77), a 30-line sized.rs doc, and a measured negative sweep (ROW 234, N in {1,2,4,8}).

H5 - per-op heap allocation on the steady-state dispatch path. encode_op calls kernel_cache_key (metal.rs:3772) and kernel_dispatch_shape (:3773) for every op on every step. Between them each op allocates: operand_codecs' Vec<Option<PackedCodec>> twice (msl.rs:860-869, called at :934 and :1001), entry_name's String (:935), a String per output axis (:977), bindings' Vec<Binding> (:1003), reduction_dims' Vec<u16> (:1199) inside grid_threads, plus cache_key.to_string() on a miss (metal.rs:2397). omega/Cargo.toml:63 records 616 ops per decode step.

H6 - no shared lowering layer; three emitters re-derive 28 identical decisions (section C). reduce_is_cooperative exists three times with three signatures (msl.rs:1039 no caps, wgsl.rs:244 takes WgslCaps, cuda.rs:356 no caps); reduction_dims, entry_name, body_fingerprint, type_token, validate likewise. error.rs:76-84 then carries two near-identical variants that differ only in which copy raised them.

H7 - a doc contradicted by the code it documents. error.rs:72-74 says the wgsl v1 op set is elementwise, Keep::Reduce and Keep::Scan only and that Iota/Constant have no renderer yet. wgsl.rs:207-208 renders both, defined at wgsl.rs:1459 and :1480.

H8 - information destroyed at three boundaries: type_token (msl.rs:1940-1958) collapses 8 DType variants into two strings and msl.rs:3276 reconstructs a dtype decision from the collapsed value; EmitError::{UnsupportedOpKind, CudaUnsupportedOpKind} (error.rs:76-84) collapse a BoundOpKind into a &'static str; kernel_cache_key collapses the whole lowering decision into a String and classify_kind reconstructs part of it by grepping the source that decision produced.

## F. What a fully structural lowering would key on

Every decision in section A that is currently a build-feature flag, a rendered type-token string, an operand-count literal, or a substring of generated source is derivable from four things the BoundOp already carries: the operand Layouts (strides, including the zero-stride broadcasts classify_tiled_gemm at msl.rs:1628-1637 already reads to partition token from feature axes); the per-operand PackedCodec as block geometry rather than a family whitelist; the concrete extents with output_axes, which give reduction arity, reduction total, token extent and feature extent; and the reduce arity/associativity already exposed by ScalarOp::arity and ScalarOp::is_associative (op.rs:85-117). Keyed on those alone the kernel-body choice is a total function of the op, which means the cache key is that tuple rather than a rendered string, the profiler reads the tuple rather than grepping MSL, CachedAttention is a Reduce whose online-softmax accumulation is expressed through Keep/ReduceInit rather than a ninth variant with nine semantic fields, and a codec is a decode expression plugged into one body rather than a routing key that forks it. The three emitters then differ only in leaf token tables (simd_sum vs subgroupAdd vs __shfl_down_sync; half vs f16 vs __half) - which is exactly the 28-function overlap measured in section C.
