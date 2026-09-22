use super::*;

pub(super) fn grid_threads(
    resolved: &BoundOp,
    quantized: &[Option<Codec>],
    numeric_policy: NumericPolicy,
    expert_source_mode: bool,
) -> Result<u64, EmitError> {
    let threads = match &resolved.kind {
        BoundOpKind::CachedAttention {
            head_dim,
            cached_key_rows,
            new_key_rows,
            query_groups,
            ..
        } => {
            // The single-range fused path (nine operands) always dispatches
            // the compiled MAXIMUM chunk count (`cap`) -- redesign §5 option
            // 2, `render_cached_attention`'s own doc -- so idle simdgroups
            // above the live `context_chunks_for(...)` result are always
            // present in the grid and contribute the merge's identity
            // partial rather than being sized out of the dispatch. Redesign
            // §4c: `splits` multiplies the THREADGROUP count (this product
            // divided by `threadgroup_width`, `tiled_gemm_threadgroup_
            // width`'s own `CachedAttention` arm) by the compiled maximum
            // `ATTENTION_SPLIT_MAX` ONLY when [`cached_attention_merge_needed`]
            // holds for THIS op's own context length -- unlike `chunks`
            // (whose idle simdgroups stay inside one threadgroup and merge
            // there, so widening them costs nothing under `bit_exact`), an
            // idle SPLIT is a whole extra threadgroup that would otherwise
            // race the real one to write `out` directly (the single-dispatch
            // `final_store` branch has no scratch hop to arbitrate that
            // race). Gating this exactly like `render_cached_attention`'s own
            // `final_store` branch keeps grid width and kernel body in
            // lock-step: both flip together, at the same context length.
            let context_length = *cached_key_rows + *new_key_rows;
            // Only the single-range fused path (nine operands, `cached_key_rows
            // == 0`) widens to the compiled cap -- `two_range_cached_bound`
            // (nine operands, `cached_key_rows != 0`) already dispatches
            // against a bucket-padded, compile-time-fixed `context_length`, so
            // its live `chunks`/`splits` never grow between calls the way the
            // single-range path's do (`render_cached_attention`'s own doc).
            let single_range_dynamic = (resolved.operands().len() == 9
                || resolved.operands().len() == 12)
                && *cached_key_rows == 0;
            let (chunks, splits) = if single_range_dynamic {
                (
                    effective_context_chunk_cap(*query_groups, *head_dim),
                    if cached_attention_merge_needed(&resolved.kind, numeric_policy) {
                        crate::sized::ATTENTION_SPLIT_MAX
                    } else {
                        1
                    },
                )
            } else {
                (
                    context_chunks_for(context_length, *query_groups, *head_dim, numeric_policy),
                    1,
                )
            };
            resolved
                .extents
                .iter()
                .product::<u64>()
                .checked_div(*head_dim)
                .unwrap_or(0)
                * chunks
                * splits
                * SIMD_WIDTH
        }
        BoundOpKind::Elementwise { .. } => resolved.extents.iter().product(),
        BoundOpKind::Reduce {
            keep: Keep::Reduce,
            reduce_op,
            init,
            output_axes,
            ..
        } => {
            let output_total: u64 = output_axes
                .iter()
                .map(|dim| resolved.extents[*dim as usize])
                .product();
            if let Some(block) =
                tiled_gemm_block(resolved, quantized, *reduce_op, *init, output_axes)
            {
                // TILED_GEMM_NSG simdgroups per BLOCK_M x BLOCK_N output
                // tile, tiled over BOTH the feature axis and the token axis
                // — the amortization the row-blocked path does not do (it
                // tiles the feature axis alone; see `push_tiled_gemm_body`'s
                // doc).
                tiled_gemm_threadgroups(
                    resolved.node,
                    block
                        .feature_axes
                        .iter()
                        .map(|&axis| resolved.extents[axis as usize])
                        .product(),
                    block
                        .token_axes
                        .iter()
                        .map(|&axis| resolved.extents[axis as usize])
                        .product(),
                )?
            } else if let Some(block) = packed_row_block(resolved, quantized) {
                // one simdgroup per `codec_rows_per_simdgroup(block.codec)`
                // feature rows, times the split-K factor (1 = no-op unless
                // `metal-q4k-split-k` is active AND this shape is below the
                // target simdgroup count), tiled again by
                // `ceil(token_total / PACKED_ROW_ACTIVATION_GROUP)` once
                // `block.token_axes` folds more than one activation row per
                // streamed weight row -- `token_total == 1` collapses that
                // factor to `1`, the byte-identical single-row shape.
                let feature_total: u64 = block
                    .feature_axes
                    .iter()
                    .map(|&axis| resolved.extents[axis as usize])
                    .product();
                let token_total = packed_row_block_token_total(&block, &resolved.extents);
                let (base, split) = packed_row_dispatch(feature_total, token_total, block.codec);
                base * SIMD_WIDTH * split
            } else if reduce_is_cooperative_dispatch(
                resolved,
                quantized,
                numeric_policy,
                *reduce_op,
                *init,
                output_axes,
                expert_source_mode,
            ) {
                // one cooperative-reduce threadgroup per output element,
                // `cooperative_reduce_width` lanes wide (SIMD_WIDTH with
                // `metal-wide-cooperative-reduce` off, matching
                // `reduce_is_cooperative`'s prior doc byte-for-byte) — see
                // that function's own doc for the scaling policy.
                let reduce_dims = reduction_dims(resolved, output_axes);
                output_total * cooperative_reduce_width(resolved, quantized, &reduce_dims)
            } else {
                output_total
            }
        }
        BoundOpKind::Reduce {
            keep: Keep::Scan, ..
        } => {
            let rank = resolved.extents.len();
            resolved.extents[..rank.saturating_sub(1)].iter().product()
        }
        // `round_zero_reduce_bound`'s own doc: the per-round thread count is
        // whatever the round-0 `Reduce` this fold replaced would dispatch --
        // `GridSpec::depth` (not this) carries the round axis. Delegating
        // (rather than re-deriving `packed_row_block`/`tiled_gemm_block`
        // classification here) is what keeps this in lock-step with
        // `render_reduce(round_zero, ..)`'s own dispatch-style decision.
        #[cfg(feature = "metal-moe-mul-mat-id")]
        BoundOpKind::RoundBatchedReduce { .. } => {
            let round_zero = round_zero_reduce_bound(resolved);
            return grid_threads(&round_zero, quantized, numeric_policy, expert_source_mode);
        }
        #[cfg(not(feature = "metal-moe-mul-mat-id"))]
        BoundOpKind::RoundBatchedReduce { .. } => {
            return Err(EmitError::EpilogueNotSupported {
                node: resolved.node,
                reason: "round-merged reduce (BoundOpKind::RoundBatchedReduce) has no \
                         Metal grid-sizing renderer without metal-moe-mul-mat-id",
            });
        }
        BoundOpKind::Iota | BoundOpKind::Constant { .. } => resolved.extents.iter().product(),
        // one thread per `(v_head, value_row)` pair -- `render_gated_delta_net`'s
        // own doc; `tiled_gemm_threadgroup_width`'s sibling arm widens the
        // threadgroup to exactly `head_v_dim` so `num_v_heads` threadgroups
        // land, one per head.
        BoundOpKind::GatedDeltaNet {
            num_v_heads,
            head_v_dim,
            ..
        } => num_v_heads * head_v_dim,
        // One threadgroup, `expert_count` threads -- `dispatch`'s own doc:
        // `grid.threadgroup_width` left `None` at this kind's call site
        // defaults the threadgroup to the WHOLE grid, exactly one
        // threadgroup, which is what `render_moe_topk`'s own threadgroup
        // reduction (`live`/`reduce_val`/`reduce_idx` are `threadgroup`
        // arrays, coherent only within one threadgroup) requires.
        BoundOpKind::MoeTopK { expert_count, .. } => *expert_count,
        // One threadgroup per attention row, `width` lanes cooperating --
        // `render_cached_softmax_weights`'s own doc; `tiled_gemm_
        // threadgroup_width`'s sibling arm sets `width` as the threadgroup
        // width so this total splits evenly into `attention_rows`
        // threadgroups.
        BoundOpKind::CachedSoftmaxWeights {
            cached_key_rows,
            attention_rows,
            ..
        } => *attention_rows * wide_cooperative_reduce_width(*cached_key_rows),
    };
    Ok(threads)
}

/// The MSL scalar type a `BoundOp`'s own dtype declares its buffers,
/// scratch array, and accumulator as. `Float16` is the one narrower type
/// this backend emits (`half`, MSL's IEEE-754 binary16) — every other
/// dtype that already reached the "float" bucket before `DType` widened
/// keeps emitting `float`, matching this module's stance before `BoundOp`
/// carried a dtype at all. `omega::execute`'s upstream gate is what keeps
/// anything other than `Float32`/`Float16` from ever reaching [`emit`], so
/// those are the only two cases that matter in practice, but the match
/// stays total over every [`DType`] variant rather than assuming that gate
/// ran — a width this backend has never emitted (the 64/128-bit integers,
/// `Float64`) is rejected here by name instead of silently folded into the
/// 4-byte `float` bucket it does not fit.
pub(super) fn type_token(node: NodeId, dtype: DType) -> Result<&'static str, EmitError> {
    match dtype {
        DType::Float16 => Ok("half"),
        DType::Float32
        | DType::BFloat16
        | DType::Bool
        | DType::Int8
        | DType::UInt8
        | DType::Int32
        | DType::UInt32 => Ok("float"),
        DType::Int16
        | DType::UInt16
        | DType::Int64
        | DType::UInt64
        | DType::Int128
        | DType::UInt128
        | DType::Float64 => Err(EmitError::UnsupportedDType { node, dtype }),
    }
}

/// A structural fingerprint, not a hash of anything runtime: rank, operand
/// count, every `ScalarOp`/`ReduceInit`/`Keep` involved, and — since a gather
/// changes the generated source (extra buffer params, extra uniforms, extra
/// fetch code) — which operands gather. That last part is a suffix appended
/// only when at least one operand gathers, so a gather-free `BoundOp`'s name is
/// unchanged from before this existed.
pub(super) fn entry_name(resolved: &BoundOp) -> String {
    let rank = resolved.extents.len();
    let operand_count = resolved.operands().len();
    let base = match &resolved.kind {
        BoundOpKind::CachedAttention {
            query_rows,
            cached_key_rows,
            new_key_rows,
            kv_heads,
            query_groups,
            head_dim,
            scale,
            cached_lower_inclusive,
            new_upper_inclusive,
            ..
        } => {
            // `operand_count == 9` names a runtime ninth operand, but that
            // operand carries two DIFFERENT scalars discriminated by
            // `cached_key_rows` (`BoundOpKind::CachedAttention`'s own doc):
            // `cached_key_rows == 0` is `single_range_dynamic` -- the ninth
            // operand IS the real `new_upper_inclusive`, so the static field
            // here is unused filler and the key names the STRUCTURE ("dyn")
            // rather than that filler value. `cached_key_rows != 0` is
            // `two_range_cached_bound` -- the ninth operand is the CACHED
            // range's own live row count, but `new_upper_inclusive` is still
            // the real, query-independent compiled bound (this path's causal
            // band never depends on it), so that token is real, not "dyn".
            let single_range_dynamic = operand_count == 9 && *cached_key_rows == 0;
            let two_range_cached_bound = operand_count == 9 && *cached_key_rows != 0;
            let upper_token = if single_range_dynamic {
                "dyn".to_string()
            } else {
                signed_name_part(*new_upper_inclusive)
            };
            if single_range_dynamic {
                // `_b{width}` names the build-time block-staging width
                // (`block_width_for`) -- a build-time constant, so a build
                // whose `OMEGA_ATTENTION_BLOCK_WIDTH` override changed emits
                // a distinct name rather than reusing a cached kernel
                // compiled for a different in-block staging width. `_qh{0|1}`
                // names `cached_attention_per_query_head_grid`'s own decision
                // -- unlike the row counts this token intentionally reduces
                // the bucket extent to a boolean, never the extent itself, so
                // every bucket ON ONE SIDE of the knee still shares a single
                // compiled kernel, but the two regimes render genuinely
                // different MSL text (a different `tgid` decode -- see
                // `render_cached_attention`'s own doc) and so cannot be
                // allowed to collide on the same cache key. On this SAME
                // path, `cached_key_rows`/`new_key_rows` are also runtime
                // `Uniforms` fields now (redesign §5 option 2's
                // `render_cached_attention`), so the row-count tokens drop
                // from the name entirely -- two different `kv-capacity-
                // bucket` extents share one compiled kernel, keyed only by
                // the shape-bounded compiled MAXIMUM chunk count
                // (`_x{cap}`, `effective_context_chunk_cap`), never by the
                // live capacity.
                let per_query_head_grid =
                    cached_attention_per_query_head_grid(true, *cached_key_rows + *new_key_rows);
                format!(
                    "omega_cached_attention_q{query_rows}_h{kv_heads}_g{query_groups}_d{head_dim}_s{:08x}_l{}_u{upper_token}_x{}_b{}_qh{}",
                    scale.to_bits(),
                    signed_name_part(*cached_lower_inclusive),
                    effective_context_chunk_cap(*query_groups, *head_dim),
                    crate::sized::ATTENTION_BLOCK_WIDTH,
                    u8::from(per_query_head_grid),
                )
            } else if two_range_cached_bound {
                // `_cb` marks the ninth-operand, runtime-`cached_key_rows`
                // body (`long cached_key_rows = (long)in8[0];` --
                // `render_cached_attention`'s own doc) as structurally
                // distinct from the eight-operand, fully-`constexpr` body
                // below: same row counts, same upper token, same grid, but a
                // different buffer signature (the extra `in8` param) and a
                // different generated statement for `cached_key_rows`, so
                // the two must never share a compiled pipeline.
                format!(
                    "omega_cached_attention_q{query_rows}_c{cached_key_rows}_n{new_key_rows}_h{kv_heads}_g{query_groups}_d{head_dim}_s{:08x}_l{}_u{upper_token}_cb",
                    scale.to_bits(),
                    signed_name_part(*cached_lower_inclusive),
                )
            } else {
                format!(
                    "omega_cached_attention_q{query_rows}_c{cached_key_rows}_n{new_key_rows}_h{kv_heads}_g{query_groups}_d{head_dim}_s{:08x}_l{}_u{upper_token}",
                    scale.to_bits(),
                    signed_name_part(*cached_lower_inclusive),
                )
            }
        }
        BoundOpKind::Elementwise { .. } => {
            let body = body_token(resolved.element_body());
            format!("omega_elementwise_r{rank}_n{operand_count}_{body}")
        }
        BoundOpKind::Reduce {
            reduce_op,
            init,
            keep,
            output_axes,
            epilogue_body,
            epilogue_operands,
            ..
        } => {
            let body = body_token(resolved.element_body());
            let kind = keep_token(*keep);
            let reduce_body = op_token(*reduce_op);
            let init = init_token(*init);
            // `rank` alone does not fix the output/reduce split -- two folds
            // over the same total rank can keep a different number of axes
            // (e.g. one output axis folding three vs one folding one), which
            // sizes `output_extents`/`reduction_extents` differently in
            // `render_reduce`'s own uniform struct. Without `output_rank`
            // here, two such ops would share this name despite emitting
            // different source -- see `distinct_output_rank_at_same_total_rank_yields_distinct_entry_names`.
            let output_rank = output_axes.len();
            // A fused epilogue changes both the `Uniforms` layout (the extra
            // `epilogue_operand_base`/`_strides` fields) and the body text
            // (`push_reduce_epilogue_write`'s emitted tail) -- the untouched
            // identity default contributes nothing here, so a program with
            // no fused epilogue anywhere names exactly what it always did.
            let epilogue = if reduce_epilogue_is_identity(epilogue_body, epilogue_operands) {
                String::new()
            } else {
                format!(
                    "_epi{}_{}",
                    epilogue_operands.len(),
                    body_token(epilogue_body)
                )
            };
            format!(
                "omega_{kind}_r{rank}_o{output_rank}_n{operand_count}_{body}_{reduce_body}_{init}{epilogue}"
            )
        }
        // computed even though `emit_inner` always declines this kind
        // before rendering a body -- `entry_name` runs unconditionally
        // ahead of that decline (see this function's own call site), so
        // this still needs a real (if never-compiled) name rather than
        // panicking on an unmatched arm.
        BoundOpKind::RoundBatchedReduce {
            reduce_op,
            init,
            keep,
            output_axes,
            round_count,
            ..
        } => {
            let body = body_token(resolved.element_body());
            let kind = keep_token(*keep);
            let reduce_body = op_token(*reduce_op);
            let init = init_token(*init);
            let output_rank = output_axes.len();
            format!(
                "omega_{kind}_r{rank}_o{output_rank}_n{operand_count}_{body}_{reduce_body}_{init}_k{round_count}"
            )
        }
        // no operand count, no body: an `Iota`'s whole structure is its
        // rank (always 1 in practice, since `Op::Iota` resolves one
        // `Extent` — see `op.rs`'s doc — but this reads `extents.len()`
        // rather than assuming that, matching every other arm here).
        BoundOpKind::Iota => format!("omega_iota_r{rank}"),
        // the literal is baked into the source (see `render_constant`), so
        // it has to be part of the entry name too - otherwise two constants
        // of the same rank would share one cached kernel and the second
        // would run the first one's value. Raw bits, not the decimal, so
        // the name is exact and identifier-safe.
        BoundOpKind::Constant { value } => {
            format!("omega_constant_r{rank}_v{:08x}", value.to_bits())
        }
        BoundOpKind::GatedDeltaNet {
            kv_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            ..
        } => format!(
            "omega_gated_delta_net_h{kv_heads}_v{num_v_heads}_k{head_k_dim}_d{head_v_dim}"
        ),
        BoundOpKind::MoeTopK {
            expert_count,
            top_k,
            ..
        } => format!("omega_moe_topk_e{expert_count}_k{top_k}"),
        // `render_cached_softmax_weights` renders one kernel per
        // `(cached_key_rows, attention_rows, head_dim)` triple -- no other
        // field this kind carries changes the emitted text (operand
        // strides/bases are baked `constexpr`, but they never change WHICH
        // statements get emitted, only the literals inside them), so this
        // is a real, reachable cache key, shaped like every other arm here.
        BoundOpKind::CachedSoftmaxWeights {
            cached_key_rows,
            attention_rows,
            head_dim,
            ..
        } => format!(
            "omega_cached_softmax_weights_c{cached_key_rows}_a{attention_rows}_d{head_dim}"
        ),
    };
    let gather_bits: String = resolved
        .operands()
        .iter()
        .map(|(_, _, gather)| if gather.is_some() { '1' } else { '0' })
        .collect();
    if gather_bits.contains('1') {
        format!("{base}_g{gather_bits}")
    } else {
        base
    }
}

pub(super) fn scalar_op_expr(op: ScalarOp, args: &[&str]) -> String {
    match op {
        ScalarOp::Identity => (*args.first().unwrap_or(&"0.0f")).into(),
        ScalarOp::Add => format!("({} + {})", args[0], args[1]),
        ScalarOp::Subtract => format!("({} - {})", args[0], args[1]),
        ScalarOp::Multiply => format!("({} * {})", args[0], args[1]),
        ScalarOp::Divide => format!("({} / {})", args[0], args[1]),
        ScalarOp::Maximum => format!("max({}, {})", args[0], args[1]),
        ScalarOp::Minimum => format!("min({}, {})", args[0], args[1]),
        ScalarOp::Negate => format!("(-{})", args[0]),
        ScalarOp::Reciprocal => format!("(1.0f / {})", args[0]),
        ScalarOp::Exponential => format!("exp({})", args[0]),
        ScalarOp::Logarithm => format!("log({})", args[0]),
        ScalarOp::SquareRoot => format!("sqrt({})", args[0]),
        // metal's tanh = (exp(2x)-1)/(exp(2x)+1) overflows to NaN past |x| ~ 45;
        // clamp first since tanh saturates to +-1.0 at f32 precision well inside +-20
        ScalarOp::Tanh => format!("tanh(clamp({}, -20.0f, 20.0f))", args[0]),
        ScalarOp::Erf => format!("proxima_erf({})", args[0]),
        ScalarOp::Greater => format!("(({} > {}) ? 1.0f : 0.0f)", args[0], args[1]),
        ScalarOp::Equal => format!("((fabs({} - {}) == 0.0f) ? 1.0f : 0.0f)", args[0], args[1]),
        ScalarOp::Select => format!("(({} != 0.0f) ? {} : {})", args[0], args[1], args[2]),
    }
}

/// `(init expression, seeded-from-the-start)`. `FirstElement` mirrors
/// `cpu::initial_value`/`cpu::run_reduce`'s `seeded` flag: the accumulator
/// starts unseeded and is instead set from the first reduction step's value —
/// the init expression here is never actually read in that case.
pub(super) fn fold_init_tokens(init: ReduceInit) -> (&'static str, &'static str) {
    match init {
        ReduceInit::Zero => ("0.0f", "true"),
        ReduceInit::One => ("1.0f", "true"),
        ReduceInit::NegativeInfinity => ("-INFINITY", "true"),
        ReduceInit::PositiveInfinity => ("INFINITY", "true"),
        ReduceInit::FirstElement => ("0.0f", "false"),
    }
}

/// Emits one `float step{n} = ...;` declaration per [`ComposedBody`] step,
/// each reading `scratch[i]` for an `Operand` arg or an earlier `step{k}`
/// for a `Step` arg — the MSL counterpart of `cpu::apply_body`'s scratch
/// walk. Returns the C expression for the body's own result (its last
/// step), which a caller splices directly into whatever it does with the
/// value (`out[gid] = ...` for elementwise, `float value = ...` for a
/// reduce/scan step).
pub(super) fn push_body_steps(
    source: &mut String,
    body: &ComposedBody,
    indent: &str,
    element_type: &str,
) -> String {
    crate::epilogue::declare_steps(
        source,
        body,
        "scratch",
        "step",
        scalar_op_expr,
        |source, index, expr| {
            source.push_str(&format!("{indent}{element_type} step{index} = {expr};\n"));
        },
    )
}

// only the row-blocked packed-matmul path's caller ever passes `true` --
// every other kernel keeps the same signature it always has, so this stays
// off by construction wherever split-K does not engage (see
// `push_cooperative_reduce_body`'s own call site for the gate).
pub(super) fn kernel_signature(
    source: &mut String,
    quantized: &[Option<Codec>],
    epilogue_operand_count: usize,
    gather_count: usize,
    entry: &str,
    element_type: &str,
    include_threadgroup_width: bool,
) {
    let operand_count = quantized.len();
    source.push_str(&format!("kernel void {entry}(\n"));
    for (index, &codec) in quantized.iter().enumerate() {
        // a packed operand's buffer is BYTES, not elements — the shader
        // turns an element offset into a super-block plus a position inside
        // it at the read (`operand_read`), so the binding has to be typed
        // for what is actually in the buffer. `Float16` is the one codec
        // whose buffer is neither raw bytes nor the kernel's own
        // `element_type`: its bytes ARE valid `half` elements already, so it
        // binds as `half` directly rather than `uchar` -- see
        // `FLOAT16_BLOCK_BYTES`'s own doc.
        let binding_type = match codec {
            None => element_type,
            Some(Codec::Float16) => "half",
            Some(_) => "uchar",
        };
        source.push_str(&format!(
            "    device const {binding_type}* in{index} [[buffer({index})]],\n"
        ));
    }
    // A [`BoundOpKind::Reduce::epilogue_operands`] entry is always a plain,
    // un-gathered, un-packed `element_type` buffer -- the same restriction
    // `cpu::apply_reduce_epilogue` already enforces (`operand_read` there has
    // no codec branch) -- so each gets one flat device buffer, positioned
    // right after the fold's own operands and before anything gather adds.
    for index in 0..epilogue_operand_count {
        source.push_str(&format!(
            "    device const {element_type}* epi{index} [[buffer({})]],\n",
            operand_count + index
        ));
    }
    let base = operand_count + epilogue_operand_count;
    for slot in 0..gather_count {
        // a gather's fetched index is always carried as an exact-integer
        // `float`, independent of the op's own element type — see this
        // crate's doc for `gather_idx` and `cpu::reject_non_float32`'s own
        // note on indices being the one deliberate non-dtype exception.
        source.push_str(&format!(
            "    device const float* gather_idx{slot} [[buffer({})]],\n",
            base + slot
        ));
    }
    source.push_str(&format!(
        "    device {element_type}* out [[buffer({})]],\n",
        base + gather_count
    ));
    source.push_str(&format!(
        "    constant Uniforms& u [[buffer({})]],\n",
        base + gather_count + 1
    ));
    if gather_count > 0 {
        source.push_str(&format!(
            "    device atomic_uint* fault [[buffer({})]],\n",
            base + gather_count + 2
        ));
    }
    source.push_str("    uint gid [[thread_position_in_grid]]");
    if include_threadgroup_width {
        // the actual per-dispatch threadgroup width -- `crate::metal::dispatch`
        // sets this from `GridSpec::threadgroup_width`, which
        // `tiled_gemm_threadgroup_width` computes FRESH per concrete dispatch
        // (unlike `Kernel::source`, cached and shared across every dispatch
        // with the same structural `kernel_cache_key`). Reading it back here
        // is what lets one compiled kernel body serve both a starved shape
        // (split > 1) and a saturated one (split == 1) without two kernel
        // bodies existing per structural shape.
        source.push_str(",\n    uint tptg [[threads_per_threadgroup]]");
    }
    source.push_str(")\n{\n");
}

/// Declares the `Uniforms` fields a gather needs — `index_base`/`index_strides`
/// (per-gather addressing into its `indices` buffer, over the *same* rank as
/// every other operand), `element_stride` (the operand's own stride along
/// its gathered dim), and `extent` (the gathered dim's size, for the clamp
/// [`push_gather_fetch`] emits). Declared only when `gather_count > 0`, so a
/// gather-free kernel's `Uniforms` struct is byte-for-byte what it was
/// before gather existed.
pub(super) fn push_gather_uniform_fields(source: &mut String, gather_count: usize, rank_len: usize) {
    if gather_count == 0 {
        return;
    }
    source.push_str(&format!("    long gather_index_base[{gather_count}];\n"));
    source.push_str(&format!(
        "    long gather_index_strides[{gather_count}][{rank_len}];\n"
    ));
    source.push_str(&format!(
        "    long gather_element_stride[{gather_count}];\n"
    ));
    source.push_str(&format!("    long gather_extent[{gather_count}];\n"));
}

/// Emits the out-of-range check for one just-fetched, not-yet-clamped
/// `fetched{operand_index}`: when it falls outside
/// `[0, u.gather_extent[gather_slot])`, records it (plus one, so a slot
/// left at zero unambiguously means "no fault") into that gathered
/// operand's slot of the `fault` buffer via `atomic_fetch_max`. A negative
/// fetched index is reported as `0` (mapped through `max(fetched, 0)`
/// before the `+1`) rather than reinterpreting a negative `long` as a huge
/// `uint` — this crate's sad-path tests only exercise the far-more-common
/// too-large case, so that is the one case whose reported value round-trips
/// exactly. `atomic_fetch_max` (not a plain write) is what makes this safe
/// under concurrent threads without a CAS loop: whichever value "wins" the
/// max is still a genuine fault, and the driver only needs to know that one
/// occurred and at what value to build a `TensorError`.
pub(super) fn push_gather_fault_check(
    source: &mut String,
    operand_index: usize,
    gather_slot: usize,
    indent: &str,
) {
    source.push_str(&format!(
        "{indent}if (fetched{operand_index} < 0 || fetched{operand_index} >= u.gather_extent[{gather_slot}]) {{\n"
    ));
    source.push_str(&format!(
        "{indent}    atomic_fetch_max_explicit(&fault[{gather_slot}], (uint)max(fetched{operand_index}, (long)0) + 1u, memory_order_relaxed);\n"
    ));
    source.push_str(&format!("{indent}}}\n"));
}

/// Emits the fetch for one gathered operand: reads its index from
/// `gather_idx{slot}` at the same coordinate `coord_var` addresses every
/// other buffer with, checks it against `[0, extent)` — recording a fault
/// (see [`push_gather_fault_check`]) since a GPU kernel cannot return a
/// `Result` the way `cpu::evaluate` does — then clamps it into `[0, extent)`
/// regardless, so the read this value drives always lands in bounds even
/// when a fault was just recorded, and adds the resulting offset into
/// `offset_var`.
pub(super) fn push_gather_fetch(
    source: &mut String,
    operand_index: usize,
    gather_slot: usize,
    rank: usize,
    coord_var: &str,
    offset_var: &str,
) {
    source.push_str(&format!(
        "    long gather_off{operand_index} = u.gather_index_base[{gather_slot}];\n"
    ));
    for dim in 0..rank {
        source.push_str(&format!(
            "    gather_off{operand_index} += {coord_var}[{dim}] * u.gather_index_strides[{gather_slot}][{dim}];\n"
        ));
    }
    source.push_str(&format!(
        "    long fetched{operand_index} = (long)gather_idx{gather_slot}[gather_off{operand_index}];\n"
    ));
    push_gather_fault_check(source, operand_index, gather_slot, "    ");
    source.push_str(&format!(
        "    fetched{operand_index} = max((long)0, min(fetched{operand_index}, u.gather_extent[{gather_slot}] - 1));\n"
    ));
    source.push_str(&format!(
        "    {offset_var} += fetched{operand_index} * u.gather_element_stride[{gather_slot}];\n"
    ));
}

pub(super) fn push_cooperative_gather_fetch(
    source: &mut String,
    operand_index: usize,
    gather_slot: usize,
    rank: usize,
    coord_var: &str,
    offset_var: &str,
) {
    source.push_str(&format!(
        "    long gather_off{operand_index} = u.gather_index_base[{gather_slot}];\n"
    ));
    for dim in 0..rank {
        source.push_str(&format!(
            "    gather_off{operand_index} += {coord_var}[{dim}] * u.gather_index_strides[{gather_slot}][{dim}];\n"
        ));
    }
    source.push_str(&format!(
        "    long fetched{operand_index} = (long)gather_idx{gather_slot}[gather_off{operand_index}];\n"
    ));
    source.push_str("    if (lane == 0u) {\n");
    push_gather_fault_check(source, operand_index, gather_slot, "    ");
    source.push_str(&format!(
        "    }}\n    fetched{operand_index} = max((long)0, min(fetched{operand_index}, u.gather_extent[{gather_slot}] - 1));\n"
    ));
    source.push_str(&format!(
        "    fetched{operand_index} = (long)simd_broadcast_first((uint)fetched{operand_index});\n    {offset_var} += fetched{operand_index} * u.gather_element_stride[{gather_slot}];\n"
    ));
}

/// `metal_stdlib` has no `erf` in any namespace — verified against the real
/// toolchain (`xcrun -sdk macosx metal -c`, `no member named 'erf'`, tried
/// bare, `metal::`, and `metal::precise::`), not assumed from the ONNX
/// survey that first named it. This is the same Abramowitz & Stegun 7.1.26
/// approximation [`crate cpu::erf_f32`](../../proxima_tensor/src/cpu.rs) uses
/// on the CPU path, so a kernel and the CPU interpreter it is checked
/// against agree on more than "close enough" — they run the identical
/// formula.
pub(super) const PROXIMA_ERF_FN: &str = "\
inline float proxima_erf(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float magnitude = fabs(x);
    float t = 1.0f / fma(0.3275911f, magnitude, 1.0f);
    float poly = t * fma(fma(fma(fma(1.061405429f, t, -1.453152027f), t, 1.421413741f), t, -0.284496736f), t, 0.254829592f);
    return sign * fma(poly, -exp(-magnitude * magnitude), 1.0f);
}
";

pub(super) fn preamble(source: &mut String) {
    source.push_str("#include <metal_stdlib>\n");
    source.push_str("using namespace metal;\n\n");
    source.push_str(PROXIMA_ERF_FN);
    source.push('\n');
    source.push_str(Q2K_UNPACK_MSL);
    source.push('\n');
    // emitted unconditionally, the same way `PROXIMA_ERF_FN` is: a
    // `static inline` the kernel never calls costs nothing in the compiled
    // AIR, and making it conditional would mean threading "does this kernel
    // read a packed operand" into the preamble for no gain.
    source.push_str(Q3K_UNPACK_MSL);
    source.push('\n');
    // structural, not feature-gated: `Q3K_PAIR_DOT_MSL` is always a real
    // Rust symbol (unlike `Q5K_PAIR_DOT_MSL`'s `#[cfg]`), so this splice is
    // unconditional the same way `Q4K_UNPACK_MSL`'s own splice below is --
    // `push_packed_row_blocked_body`'s `plain_product` check decides
    // whether the KERNEL calls it, not whether it compiles into the AIR.
    source.push_str(Q3K_PAIR_DOT_MSL);
    source.push('\n');
    source.push_str(Q4K_UNPACK_MSL);
    source.push('\n');
    // feature-gated, unlike the constants around it: `Q4K_MASK_FMA_MSL`
    // does not exist as a Rust symbol at all without `metal-q4k-mask-fma`
    // (see its own doc), so this splice is the one place in `preamble`
    // that must itself be `#[cfg]`'d rather than relying on "unused static
    // inline costs nothing" the way the codec preambles around it do.
    #[cfg(feature = "metal-q4k-mask-fma")]
    {
        source.push_str(Q4K_MASK_FMA_MSL);
        source.push('\n');
    }
    source.push_str(Q5K_UNPACK_MSL);
    source.push('\n');
    // unconditional, same posture as `Q4K_UNPACK_MSL` above: an unused
    // `static inline` the kernel never calls costs nothing in the compiled
    // AIR, and the selector is now the codec's own layout
    // ([`Codec::supports_pair_dot`]), not a cargo feature.
    source.push_str(Q5K_PAIR_DOT_MSL);
    source.push('\n');
    source.push_str(Q6K_UNPACK_MSL);
    source.push('\n');
    source.push_str(Q6K_PAIR_DOT_MSL);
    source.push('\n');
    // The mixed selector calls the Q2_K, Q4_K, and Q6_K element decoders,
    // so it follows all three declarations in the generated translation unit.
    source.push_str(MIXED_EXPERT_READ_MSL);
    source.push('\n');
    source.push_str(UNIFORM_EXPERT_READ_MSL);
    source.push('\n');
    source.push_str(Q8_0_UNPACK_MSL);
    source.push('\n');
    source.push_str(Q8_0_SUPER_ELEMENT_MSL);
    source.push('\n');
    // unconditional, same posture as `Q5K_PAIR_DOT_MSL` above: the selector
    // is `is_plain_product_reduce`, decided at EMIT time by
    // `push_packed_row_blocked_body`'s `Codec::Q8_0` arm, not a cargo
    // feature -- an unused `static inline` costs nothing in the compiled AIR.
    source.push_str(Q8_0_PAIR_DOT_MSL);
    source.push('\n');
    source.push_str(Q4_0_UNPACK_MSL);
    source.push('\n');
    source.push_str(Q4_0_SUPER_ELEMENT_MSL);
    source.push('\n');
    source.push_str(Q4_0_PAIR_DOT_MSL);
    source.push('\n');
    source.push_str(Q5_1_UNPACK_MSL);
    source.push('\n');
    source.push_str(Q5_0_UNPACK_MSL);
    source.push('\n');
    source.push_str(BF16_UNPACK_MSL);
    source.push('\n');
}

/// How operand `index` is READ, given the element-offset expression the
/// caller already computed. A float operand is a direct index. A `Q4_K`
/// operand's buffer is PACKED BYTES, so that same element offset splits into
/// a super-block and a position inside it: element `n` lives in super-block
/// `n / 256` at position `n % 256`, and that super-block starts at byte
/// `(n / 256) * 144`. The uniforms stay in elements either way — only the
/// read shape changes, which is the entire point of unpacking at the read
/// instead of materializing a dequantized tensor first.
pub(super) fn operand_read(index: usize, offset: &str, codec: Option<Codec>) -> String {
    match codec {
        None => format!("in{index}[{offset}]"),
        Some(Codec::Q2K) => format!(
            "q2k_element(in{index} + ({offset} / {Q2K_BLOCK_ELEMENTS}) * {Q2K_BLOCK_BYTES}, (uint)({offset} % {Q2K_BLOCK_ELEMENTS}))"
        ),
        Some(Codec::Q3K) => format!(
            "q3k_element(in{index} + ({offset} / {Q4K_BLOCK_ELEMENTS}) * {Q3K_BLOCK_BYTES}, (uint)({offset} % {Q4K_BLOCK_ELEMENTS}))"
        ),
        Some(Codec::Q4K) => format!(
            "q4k_element(in{index} + ({offset} / {Q4K_BLOCK_ELEMENTS}) * {Q4K_BLOCK_BYTES}, (uint)({offset} % {Q4K_BLOCK_ELEMENTS}))"
        ),
        Some(Codec::Q5K) => format!(
            "q5k_element(in{index} + ({offset} / {Q4K_BLOCK_ELEMENTS}) * {Q5K_BLOCK_BYTES}, (uint)({offset} % {Q4K_BLOCK_ELEMENTS}))"
        ),
        Some(Codec::Q6K) => format!(
            "q6k_element(in{index} + ({offset} / {Q4K_BLOCK_ELEMENTS}) * {Q6K_BLOCK_BYTES}, (uint)({offset} % {Q4K_BLOCK_ELEMENTS}))"
        ),
        // `Q8_0`'s block is 32 elements, not 256 -- its own
        // [`Q8_0_BLOCK_ELEMENTS`], never [`Q4K_BLOCK_ELEMENTS`].
        Some(Codec::Q8_0) => format!(
            "q8_0_element(in{index} + ({offset} / {Q8_0_BLOCK_ELEMENTS}) * {Q8_0_BLOCK_BYTES}, (uint)({offset} % {Q8_0_BLOCK_ELEMENTS}))"
        ),
        // `Q4_0`'s block is 32 elements, not 256 -- its own
        // [`Q4_0_BLOCK_ELEMENTS`], never [`Q4K_BLOCK_ELEMENTS`].
        Some(Codec::Q4_0) => format!(
            "q4_0_element(in{index} + ({offset} / {Q4_0_BLOCK_ELEMENTS}) * {Q4_0_BLOCK_BYTES}, (uint)({offset} % {Q4_0_BLOCK_ELEMENTS}))"
        ),
        // `Q5_1`'s block is 32 elements, not 256 -- its own
        // [`Q5_1_BLOCK_ELEMENTS`], never [`Q4K_BLOCK_ELEMENTS`].
        Some(Codec::Q5_1) => format!(
            "q5_1_element(in{index} + ({offset} / {Q5_1_BLOCK_ELEMENTS}) * {Q5_1_BLOCK_BYTES}, (uint)({offset} % {Q5_1_BLOCK_ELEMENTS}))"
        ),
        // `Q5_0`'s block is 32 elements, not 256 -- its own
        // [`Q5_0_BLOCK_ELEMENTS`], never [`Q4K_BLOCK_ELEMENTS`].
        Some(Codec::Q5_0) => format!(
            "q5_0_element(in{index} + ({offset} / {Q5_0_BLOCK_ELEMENTS}) * {Q5_0_BLOCK_BYTES}, (uint)({offset} % {Q5_0_BLOCK_ELEMENTS}))"
        ),
        // `Float16`'s buffer already binds as `device const half*`
        // (`kernel_signature`'s own match), so reading it is a plain index
        // exactly like a `None` operand -- MSL implicitly promotes the
        // resulting `half` to `float` wherever the body assigns it into a
        // `float` scratch slot, no cast needed.
        Some(Codec::Float16) => format!("in{index}[{offset}]"),
        // `BFloat16`'s block is 1 element, 2 bytes -- its own
        // [`BFLOAT16_BLOCK_ELEMENTS`]/[`BFLOAT16_BLOCK_BYTES`], never
        // [`Q4K_BLOCK_ELEMENTS`].
        Some(Codec::BFloat16) => format!(
            "bf16_element(in{index} + ({offset} / {BFLOAT16_BLOCK_ELEMENTS}) * {BFLOAT16_BLOCK_BYTES}, (uint)({offset} % {BFLOAT16_BLOCK_ELEMENTS}))"
        ),
        // No Metal unpack kernel exists for any of these 18 -- `PackedOperands`
        // is only ever populated via `codec_from_quantized_block`, which
        // maps just the 11 codecs above, so this arm is unreachable by
        // construction; kept exhaustive so a future codec forces a decision
        // here rather than slipping through.
        Some(
            Codec::Q4_1
            | Codec::Q8_1
            | Codec::Q8K
            | Codec::Iq1S
            | Codec::Iq1M
            | Codec::Iq2Xxs
            | Codec::Iq2Xs
            | Codec::Iq2S
            | Codec::Iq3Xxs
            | Codec::Iq3S
            | Codec::Iq4Nl
            | Codec::Iq4Xs
            | Codec::Tq10
            | Codec::Tq20
            | Codec::Mxfp4
            | Codec::Nvfp4
            | Codec::Q1_0
            | Codec::Q2_0,
        ) => format!("in{index}[{offset}]"),
    }
}

/// [`BoundOpKind::Iota`]'s kernel: no operand buffers, no gather, no body —
/// the output value at each position is the thread's own grid coordinate,
/// which every kernel already computes as `gid`, so there is nothing to
/// derive beyond casting it to the node's element type. Reuses
/// [`kernel_signature`] with `operand_count = 0`, `gather_count = 0` so the
/// buffer-index arithmetic (`out` at 0, `Uniforms` at 1) stays the one place
/// that owns it rather than being re-derived here.
pub(super) fn render_iota(resolved: &BoundOp, entry: &str) -> Result<String, EmitError> {
    let element_type = type_token(resolved.node, resolved.dtype)?;

    let mut source = String::new();
    preamble(&mut source);

    source.push_str("struct Uniforms {\n");
    source.push_str("    long total_elements;\n");
    source.push_str("};\n\n");

    kernel_signature(&mut source, &[], 0, 0, entry, element_type, false);
    source.push_str("    if ((long)gid >= u.total_elements) { return; }\n");
    source.push_str(&format!("    out[gid] = ({element_type})gid;\n"));
    source.push_str("}\n");
    Ok(source)
}

/// [`BoundOpKind::Constant`]'s kernel, the same shape as [`render_iota`]'s
/// with the position swapped for the literal. The literal is baked into the
/// source rather than passed as a uniform so the `Uniforms` struct stays
/// byte-identical to `render_iota`'s and both share
/// [`crate::metal`]'s `pack_leaf_uniforms`; `kernel_entry` folds the value's
/// bits into the entry name to keep the kernel cache correct.
pub(super) fn render_constant(resolved: &BoundOp, entry: &str, value: f32) -> Result<String, EmitError> {
    let element_type = type_token(resolved.node, resolved.dtype)?;

    let mut source = String::new();
    preamble(&mut source);

    source.push_str("struct Uniforms {\n");
    source.push_str("    long total_elements;\n");
    source.push_str("};\n\n");

    kernel_signature(&mut source, &[], 0, 0, entry, element_type, false);
    source.push_str("    if ((long)gid >= u.total_elements) { return; }\n");
    source.push_str(&format!(
        "    out[gid] = ({element_type}){};\n",
        msl_literal(value)
    ));
    source.push_str("}\n");
    Ok(source)
}

/// [`BoundOpKind::GatedDeltaNet`]'s fused kernel -- one thread per
/// `(v_head, value_row)` pair, the state row for that pair resident in
/// registers across the whole `n_tokens` loop, mirroring llama.cpp's own
/// fused Metal kernel (`gated_delta_net.metal:8-141`) and ported line for
/// line from `proxima_tensor::gdn::run_gdn_prefill_scan` -- the SAME scalar
/// loop `crate::cpu::run_gated_delta_net` (via that function) already runs
/// on CPU, so this parity test's oracle and this kernel compute the exact
/// same arithmetic in the exact same order (design doc `fused-gdn-kernel.md`
/// §2/§4). Dispatch is `num_v_heads` threadgroups of `head_v_dim` threads
/// each (`grid_threads`/`tiled_gemm_threadgroup_width`'s own
/// `GatedDeltaNet` arms); `thread_position_in_grid = vh * head_v_dim + row`
/// under `dispatchThreads_threadsPerThreadgroup`'s linear grouping (the
/// same derivation `render_cached_attention`'s own doc uses), so `vh`/`row`
/// recover cleanly from `gid` alone with no extra kernel parameter.
///
/// `kv_heads`/`num_v_heads`/`head_k_dim`/`head_v_dim` are baked `constexpr`
/// -- they size the register array and the dispatch grid, and
/// [`crate::identity::kernel_identity`]'s own `GatedDeltaNet` arm keys the
/// pipeline cache on exactly these four. `n_tokens`,
/// `query_key_head_stride`, `query_key_dim_stride`, and `inv_sqrt_key_dim`
/// stay genuine runtime uniforms instead: two structurally-identical binds
/// may legitimately carry different strides (`BoundOpKind::GatedDeltaNet`'s
/// own doc on the pre-/post-`repeat_kv_heads` addressing split), so baking
/// them would fragment the pipeline cache for values the kernel body can
/// read once, cheaply, from a buffer. `inv_sqrt_key_dim` travels as raw
/// bits (`as_type<float>`, the same reinterpret [`crate::cpu`]'s dequant
/// path already relies on) since every other uniform field here is `long`.
///
/// `state_in` (buffer 5) is one of `bindings`'s ordinary read operands, and
/// stays `const` -- this kernel never mutates it. `state_out` (buffer 8,
/// past `bindings`'s own `Output`/`Uniforms` slots) is `BoundOpKind::
/// GatedDeltaNet::state_out`'s own SECOND output (ROW 547,
/// `docs/discipline.md`): the kernel reads `state_in` once per thread at
/// entry and writes the updated row to `state_out` at exit, a DIFFERENT
/// device buffer, never back into `state_in` -- `crate::metal::encode_op`'s
/// own `GatedDeltaNet` arm resolves and binds that buffer manually (`bind_
/// buffers`'s single `output` parameter cannot carry two distinct node
/// identities) and registers it with the hazard tracker as written.
pub(super) fn render_gated_delta_net(resolved: &BoundOp, entry: &str) -> Result<String, EmitError> {
    let BoundOpKind::GatedDeltaNet {
        kv_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        ..
    } = &resolved.kind
    else {
        return Err(EmitError::RenderKindMismatch {
            node: resolved.node,
            expected: "gated_delta_net",
            found: resolved.kind.name(),
        });
    };
    let head_k_dim_max = crate::sized::GATED_DELTA_NET_HEAD_K_DIM_MAX;
    if *head_k_dim > head_k_dim_max {
        return Err(EmitError::GatedDeltaNetHeadKDimExceedsCap {
            node: resolved.node,
            head_k_dim: *head_k_dim,
            cap: head_k_dim_max,
        });
    }

    let mut source = String::new();
    preamble(&mut source);
    source.push_str(
        "struct Uniforms { long n_tokens; long query_key_head_stride; long query_key_dim_stride; long inv_sqrt_key_dim_bits; };\n\n",
    );
    source.push_str(&format!(
        "kernel void {entry}(device const float* query [[buffer(0)]], device const float* key [[buffer(1)]], device const float* value [[buffer(2)]], device const float* gate [[buffer(3)]], device const float* beta [[buffer(4)]], device const float* state_in [[buffer(5)]], device float* out [[buffer(6)]], constant Uniforms& u [[buffer(7)]], device float* state_out [[buffer(8)]], uint gid [[thread_position_in_grid]]) {{\n"
    ));
    source.push_str(&format!(
        "    constexpr long kv_heads = {kv_heads}; constexpr long num_v_heads = {num_v_heads}; \
         constexpr long head_k_dim = {head_k_dim}; constexpr long head_v_dim = {head_v_dim}; \
         constexpr long group = num_v_heads / kv_heads; constexpr long max_head_k_dim = {head_k_dim_max};\n"
    ));
    source.push_str(
        "    const long vh = (long)gid / head_v_dim;\n\
         const long row = (long)gid % head_v_dim;\n\
         if (vh >= num_v_heads) { return; }\n\
         const long kh = vh / group;\n\
         const long n_tokens = u.n_tokens;\n\
         const long qk_head_stride = u.query_key_head_stride;\n\
         const long qk_dim_stride = u.query_key_dim_stride;\n\
         const float inv_sqrt_key_dim = as_type<float>((uint)u.inv_sqrt_key_dim_bits);\n\
         float state_row[max_head_k_dim];\n\
         for (long i = 0; i < head_k_dim; i++) {\n\
         \tstate_row[i] = state_in[(i * head_v_dim + row) * num_v_heads + vh];\n\
         }\n\
         for (long t = 0; t < n_tokens; t++) {\n\
         \tconst float decay = exp(gate[t * num_v_heads + vh]);\n\
         \tconst long key_row_base = t * kv_heads * head_k_dim + kh * qk_head_stride;\n\
         \tfloat predicted = 0.0;\n\
         \tfor (long i = 0; i < head_k_dim; i++) {\n\
         \t\tstate_row[i] *= decay;\n\
         \t\tpredicted += state_row[i] * key[key_row_base + i * qk_dim_stride];\n\
         \t}\n\
         \tconst long vout = (t * head_v_dim + row) * num_v_heads + vh;\n\
         \tconst float delta = (value[vout] - predicted) * beta[t * num_v_heads + vh];\n\
         \tfloat readout = 0.0;\n\
         \tfor (long i = 0; i < head_k_dim; i++) {\n\
         \t\tstate_row[i] += key[key_row_base + i * qk_dim_stride] * delta;\n\
         \t\treadout += state_row[i] * (query[key_row_base + i * qk_dim_stride] * inv_sqrt_key_dim);\n\
         \t}\n\
         \tout[vout] = readout;\n\
         }\n\
         for (long i = 0; i < head_k_dim; i++) {\n\
         \tstate_out[(i * head_v_dim + row) * num_v_heads + vh] = state_row[i];\n\
         }\n\
         }\n",
    );
    Ok(source)
}

/// [`BoundOpKind::MoeTopK`]'s fused kernel -- one threadgroup of
/// `expert_count` threads (ROW 569, `docs/discipline.md`), `top_k` rounds,
/// ported from the identical scalar loop `proxima_tensor::cpu::run_moe_topk`
/// already runs on CPU. This is the SIMD-group rewrite of this function's
/// first draft (a full threadgroup-memory tree reduction, measured at a
/// 5.4ms median bare-dispatch cost for 256 experts -- `omega/tests/
/// moe_topk_bare_dispatch_timing.rs`'s own oracle): 256 lanes are 8
/// simdgroups of 32, and `simd_max` reduces within one simdgroup in
/// lock-step, no barrier, so only the CROSS-simdgroup exchange (one value
/// per simdgroup) needs `threadgroup_barrier` -- 2 barriers per round
/// (under the ROW 569 target of <= 3), not the tree reduction's
/// `2 * log2(expert_count)`.
///
/// Per round: `simd_max(masked)` gives each simdgroup's own max VALUE in
/// lock-step; lane 0 of each simdgroup writes it to `sg_max[sg_id]`; ONE
/// barrier makes every simdgroup's entry visible; every thread then folds
/// the (at most 8) `sg_max` entries into `max_value` redundantly -- cheap,
/// and avoids a second reduction pass for something this small. The same
/// two-level shape finds the winner index: `simd_max` over
/// `masked == max_value ? tid : -1` (a `max` over a candidate/`-1` pair
/// can only ever prefer the larger valid index, so ties resolve to the
/// HIGHEST index within a simdgroup for free); lane 0 writes the
/// simdgroup's own candidate to `sg_idx[sg_id]`; a SECOND barrier, then
/// every thread folds `sg_idx` the same way -- the exact bit-exact match
/// for `proxima_tensor::spec::append_moe_ffn`'s own `mask * expert_index ->
/// reduce(Maximum)` construction (ROW 569's own tie fixture: ties resolve
/// to the HIGHER index) `run_moe_topk`'s own doc and this kernel's parity
/// test both prove. Exclusion (`masked = (masked == max_value) ? -inf :
/// masked`) is then a per-lane compare with no shared state and no
/// barrier at all -- each thread only ever mutates its OWN register, and
/// the next round's `simd_max` reads it fresh from that same lane.
///
/// `max0`/`weight_total` are plain per-thread registers, not threadgroup
/// memory: only thread 0 ever reads or writes them, so nothing needs
/// synchronizing across threads for those two values either.
///
/// `expert_count`/`top_k` are baked `constexpr` -- [`crate::identity::kernel_identity`]'s
/// own `MoeTopK` arm keys the pipeline cache on exactly these two, the same
/// shape `render_gated_delta_net`'s own `kv_heads`/`num_v_heads`/
/// `head_k_dim`/`head_v_dim` bake. `scores` (buffer 0) is this op's one true
/// operand; `route0` (buffer 1) is [`bindings`]'s own `Binding::Output`
/// slot; buffer 2 is the ordinary (unread) `Uniforms` slot every kind gets
/// from [`bindings`]; buffers 3.. are this op's own `2 * top_k` extra
/// outputs (`routes[1..]`, every `weights` entry, `weight_total`, in
/// exactly `crate::metal::moe_topk_extra_node_order`'s own order) --
/// `crate::metal::encode_op`'s own `MoeTopK` arm binds them manually past
/// `bindings.len()`, the same "second output binds at the next free slot"
/// shape [`render_gated_delta_net`]'s own doc names for `state_out`.
pub(super) fn render_moe_topk(resolved: &BoundOp, entry: &str) -> Result<String, EmitError> {
    let BoundOpKind::MoeTopK {
        expert_count,
        top_k,
        ..
    } = &resolved.kind
    else {
        return Err(EmitError::RenderKindMismatch {
            node: resolved.node,
            expected: "moe_topk",
            found: resolved.kind.name(),
        });
    };
    let expert_count = *expert_count;
    let top_k = *top_k;
    // `crate::metal::encode_op`'s own `MoeTopK` arm binds exactly
    // `2 * top_k` extra buffers -- `routes[1..top_k]` (`top_k - 1` entries),
    // every `weights` entry (`top_k` entries), then `weight_total` (1) --
    // the same `extra_nodes` order `proxima_tensor::cpu::run_moe_topk`'s own
    // `moe_topk_extra_node_order` uses on CPU.
    let extra_count = 2 * top_k;
    let num_simdgroups = expert_count.div_ceil(32);

    let mut source = String::new();
    preamble(&mut source);
    source.push_str("struct Uniforms { long unused; };\n\n");
    source.push_str(&format!(
        "kernel void {entry}(device const float* scores [[buffer(0)]], device float* route0 [[buffer(1)]], constant Uniforms& u [[buffer(2)]],\n"
    ));
    for index in 0..extra_count {
        let buffer_index = 3 + index;
        source.push_str(&format!(
            "    device float* extra{index} [[buffer({buffer_index})]],\n"
        ));
    }
    source.push_str(
        "    uint tid [[thread_position_in_threadgroup]],\n\
         \tuint sg_id [[simdgroup_index_in_threadgroup]],\n\
         \tuint sg_lane [[thread_index_in_simdgroup]]) {\n",
    );
    let extra_params: Vec<String> = (0..extra_count).map(|index| format!("extra{index}")).collect();
    source.push_str(&format!(
        "    device float* extras[{extra_count}] = {{ {} }};\n",
        extra_params.join(", ")
    ));
    source.push_str(&format!(
        "    constexpr uint expert_count = {expert_count}u; constexpr uint top_k = {top_k}u; \
         constexpr uint num_simdgroups = {num_simdgroups}u;\n\
         \t(void)u;\n\
         \tthreadgroup float sg_max[num_simdgroups];\n\
         \tthreadgroup int sg_idx[num_simdgroups];\n\
         \tfloat masked = (tid < expert_count) ? scores[tid] : -INFINITY;\n\
         \tfloat max0 = 0.0;\n\
         \tfloat weight_total = 0.0;\n\
         \tfor (uint round = 0; round < top_k; round++) {{\n\
         \t\tconst float sg_max_value = simd_max(masked);\n\
         \t\tif (sg_lane == 0) {{ sg_max[sg_id] = sg_max_value; }}\n\
         \t\tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t\tfloat max_value = sg_max[0];\n\
         \t\tfor (uint i = 1; i < num_simdgroups; i++) {{ max_value = max(max_value, sg_max[i]); }}\n\
         \t\tconst int candidate = (masked == max_value) ? int(tid) : -1;\n\
         \t\tconst int sg_candidate = simd_max(candidate);\n\
         \t\tif (sg_lane == 0) {{ sg_idx[sg_id] = sg_candidate; }}\n\
         \t\tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t\tint winner_index = sg_idx[0];\n\
         \t\tfor (uint i = 1; i < num_simdgroups; i++) {{ winner_index = max(winner_index, sg_idx[i]); }}\n\
         \t\tif (round == 0) {{ max0 = max_value; }}\n\
         \t\tif (tid == 0) {{\n\
         \t\t\tconst float weight = exp(max_value - max0);\n\
         \t\t\tweight_total += weight;\n\
         \t\t\tif (round == 0) {{\n\
         \t\t\t\troute0[0] = float(winner_index);\n\
         \t\t\t}} else {{\n\
         \t\t\t\textras[round - 1][0] = float(winner_index);\n\
         \t\t\t}}\n\
         \t\t\textras[(top_k - 1) + round][0] = weight;\n\
         \t\t\tif (round == top_k - 1) {{ extras[{extra_count} - 1][0] = weight_total; }}\n\
         \t\t}}\n\
         \t\tif (masked == max_value) {{ masked = -INFINITY; }}\n\
         \t}}\n\
         }}\n"
    ));
    Ok(source)
}

/// One `f32` as MSL source text. `Debug`'s shortest round-trip decimal is
/// what MSL's own float grammar accepts, except for the values it has no
/// decimal spelling for.
pub(super) fn msl_literal(value: f32) -> String {
    if value.is_nan() {
        return "NAN".to_string();
    }
    if value.is_infinite() {
        return if value.is_sign_negative() {
            "-INFINITY".to_string()
        } else {
            "INFINITY".to_string()
        };
    }
    format!("{value:?}")
}

/// Number of simdgroups `render_cached_attention` splits one
/// `(query_row, kv_head, group)` triple's key range across -- llama.cpp's
/// `kernel_flash_attn_ext_vec` (ggml-metal.metal:4016-4017) partitions the
/// context across `nsg` simdgroups with a stride of `C*nsg`; this ports the
/// same stride partition to a `C=1` per-key loop. This is CHUNK SIZING, a
/// compile-time partitioning decision, and it stays bound to the compiled
/// range extent `cached_key_rows + new_key_rows` (the same static fields the
/// kernel body bakes as `constexpr`) -- a build-time-configurable but
/// bind-time-deterministic function of shape alone, never a runtime read.
/// This is a SEPARATE decision from the per-simdgroup stride LOOP's own
/// upper bound (`render_cached_attention`'s `last_key`), which as of the
/// merged-KV fix reads the live runtime prefix when one is carried (the
/// ninth operand) rather than always walking the full compiled extent.
/// Neither value is "the context length": one sizes the partition, the
/// other bounds each partition's own walk. See `omega-runtime.toml`'s
/// `[attention_context_chunks]` for the two knobs.
///
/// A chunk count above 1 folds partial online-softmax state across
/// simdgroups (`render_cached_attention`'s cross-simdgroup merge) -- a
/// reassociation of the reduce, [`NumericRewrite::ContextChunkMerge`].
/// `policy` gates it: without `reassociation` granted, [`admit`] rejects the
/// rewrite and this falls back to `1` (the single-pass
/// kernel `render_cached_attention` already renders for that case) rather
/// than failing the whole plan -- the same "reject/fallback at bind time"
/// shape `fuse_cached_attention: false` gives wgpu/cuda
/// (`proxima_tensor::bind::bind_with_fusion`'s own doc).
/// `pub`, not `pub(crate)` -- the attention fuse/unfuse parity harness
/// (`proxima-model-interop`'s `decode.rs`) calls this with the SAME
/// `cached_key_rows`/`query_groups`/`head_dim`/`numeric_policy` the real
/// step's own `render_cached_attention` call used, to report the actual
/// `context_chunks` a production run selected instead of inferring it from
/// the `cached_len <= ATTENTION_CONTEXT_KEYS_PER_CHUNK` relationship.
pub fn context_chunks_for(
    context_length: u64,
    query_groups: u64,
    head_dim: u64,
    policy: NumericPolicy,
) -> u64 {
    if admit(policy, NumericRewrite::ContextChunkMerge).is_err() {
        return 1;
    }
    context_length
        .div_ceil(crate::sized::ATTENTION_CONTEXT_KEYS_PER_CHUNK)
        .clamp(1, effective_context_chunk_cap(query_groups, head_dim))
}

/// The PLAN-TIME chunk cap `render_cached_attention`'s single-range dynamic
/// path may actually declare -- `omega-runtime.toml`'s `[attention_context_
/// chunks].cap` (`ATTENTION_CONTEXT_CHUNK_CAP`) is a shape-INDEPENDENT
/// build-time ceiling, but the `shared_m`/`shared_l`/`shared_o` threadgroup
/// arrays it sizes (`render_cached_attention`'s own doc) scale with
/// `query_groups * cap * (head_dim + 2)` floats -- at `query_groups=8`,
/// `head_dim=256` the compiled cap (4) alone already declares 33024 bytes
/// against Metal's 32768-byte `threadgroup` ceiling
/// (`CACHED_ATTENTION_THREADGROUP_MEMORY_BYTES`), which fails pipeline
/// compilation (`CompileFailed`) rather than degrading. This clamps the
/// compiled cap down, per shape, to whatever the budget actually admits --
/// every one of [`context_chunks_for`]'s call sites and every raw
/// `ATTENTION_CONTEXT_CHUNK_CAP` read in the single-range dynamic path
/// (dispatch grid width, entry-name cache key, scratch declaration) must
/// call THIS function instead, or the dispatch/scratch/identity can disagree
/// on how many simdgroups the compiled kernel actually has room for.
pub(crate) fn effective_context_chunk_cap(query_groups: u64, head_dim: u64) -> u64 {
    let bytes_per_chunk = 4 * query_groups.max(1) * (head_dim + 2);
    let budget_cap = crate::sized::CACHED_ATTENTION_THREADGROUP_MEMORY_BYTES / bytes_per_chunk;
    crate::sized::ATTENTION_CONTEXT_CHUNK_CAP.min(budget_cap.max(1))
}

/// [`render_cached_attention`]'s single-range dynamic path's in-block
/// Q·K/softmax/V staging width -- llama.cpp's `kernel_flash_attn_ext_vec`
/// keys-per-block constant `C` (ggml-metal.metal:4016-4017, `ic0 += C*nsg`),
/// ported onto our per-simdgroup key walk. Grouping `width` sequential
/// online-softmax updates into one block-level `simd_max`/`simd_sum` combine
/// reorders the fold as a tree instead of a left fold --
/// [`NumericRewrite::TreeReduce`], the same permission bit
/// [`context_chunks_for`]'s `NumericRewrite::ContextChunkMerge` already
/// needs. Under `bit_exact` (`admit` rejects the rewrite), this returns `1`,
/// which renders EXACTLY today's per-key loop, byte for byte -- the
/// sequential path IS the bit-exact lowering, by construction, the same way
/// `context_chunks_for` keeps chunk-count `1` under `bit_exact` today.
pub(crate) fn block_width_for(policy: NumericPolicy) -> u64 {
    if admit(policy, NumericRewrite::TreeReduce).is_err() {
        1
    } else {
        crate::sized::ATTENTION_BLOCK_WIDTH
    }
}

/// Number of THREADGROUPS `render_cached_attention`'s single-range dynamic
/// path splits one `(query_row, kv_head)` pair's key range across --
/// [`context_chunks_for`] one level down (simdgroups sharing ONE
/// threadgroup); this is the same partition one hardware level up
/// (threadgroups sharing one dispatch), llama.cpp's `nwg`
/// (`ggml-metal-ops.cpp:3457-3466`) ported as a compiled-capacity-derived
/// count rather than llama's fixed 32, because this backend's threadgroup is
/// heavier (cooperative `query_groups`-way K/V sharing across its `cap`
/// simdgroups) than llama's per-Q-head one.
///
/// `context_length` is the SAME compiled-capacity input
/// [`context_chunks_for`] already takes (`cached_key_rows + new_key_rows`),
/// not a per-call live value -- so, exactly like `chunks`/`cap`, one
/// compiled kernel serves every `kv-capacity-bucket` crossing: the dispatch
/// always issues the compiled MAXIMUM ([`crate::sized::ATTENTION_SPLIT_MAX`])
/// worth of threadgroups, and the live `splits` value this function returns
/// travels as a runtime `Uniforms` field so an idle split (`split >= splits`)
/// contributes the merge's own identity partial rather than being sized out
/// of the dispatch -- the same "idle unit, identity partial" shape `chunks`/
/// `cap` already uses.
///
/// Splitting the key range across dispatch boundaries reassociates the
/// online-softmax combine one hardware level above [`ContextChunkMerge`] --
/// [`NumericRewrite::ContextSplitMerge`]. Under `bit_exact` (`admit` rejects
/// the rewrite), this returns `1`, rendering exactly today's single-dispatch
/// `render_cached_attention` body, byte for byte -- the single-dispatch path
/// IS the bit-exact lowering, the same way `context_chunks_for`/
/// `block_width_for` fall back to `1` under `bit_exact`.
///
/// ROW 383: the divisor has a KNEE rather than being one constant. More
/// splits helps at both ends -- the tiny decode window (normally a single,
/// occupancy-starved threadgroup) and huge contexts (enough keys per split
/// to amortize the merge dispatch's fixed cost) -- but HURTS in the middle,
/// where occupancy is already adequate at a handful of splits and a finer
/// divisor only adds fixed per-split overhead with nothing to amortize it
/// against (measured: 512 keys at 8/16/32 splits all land 170-177us, no
/// better than each other, vs 89.5us at 4 splits). So `context_length`
/// below `ATTENTION_SPLIT_KEYS_PER_SPLIT_AT_SCALE` uses the small divisor
/// (`ATTENTION_SPLIT_KEYS_PER_SPLIT`, ROW 381's decode-window fix); at or
/// above it, the ORIGINAL pre-ROW-381 divisor
/// (`ATTENTION_SPLIT_KEYS_PER_SPLIT_AT_SCALE`) applies, unchanged from what
/// already won at both 512 keys (splits=4) and 4096 keys (splits=32, the
/// same value the small divisor also reaches once clamped to `max`).
pub(crate) fn splits_for(context_length: u64, policy: NumericPolicy) -> u64 {
    if admit(policy, NumericRewrite::ContextSplitMerge).is_err() {
        return 1;
    }
    let keys_per_split = if context_length < crate::sized::ATTENTION_SPLIT_KEYS_PER_SPLIT_AT_SCALE {
        crate::sized::ATTENTION_SPLIT_KEYS_PER_SPLIT
    } else {
        crate::sized::ATTENTION_SPLIT_KEYS_PER_SPLIT_AT_SCALE
    };
    context_length
        .div_ceil(keys_per_split)
        .clamp(1, crate::sized::ATTENTION_SPLIT_MAX)
}

/// Whether `render_cached_attention`'s single-range dynamic path renders as
/// TWO Metal dispatches (this function's `true`) or the single, unchanged
/// dispatch (`false`) -- delegates to [`splits_for`]'s own `context_length`
/// AND policy gate, factored out so [`render_cached_attention`]'s own
/// final-store branch, [`split_bindings_with_scratch`]'s caller, and
/// [`emit_cached_attention_merge`] all key off one boolean rather than three
/// independent `splits_for` calls that could drift. Taking `context_length`
/// (not policy alone) matters: a policy that admits `ContextSplitMerge` but
/// whose live capacity clamps `splits_for` to `1` (ROW 376's own 40-key
/// scoreboard window, well under `omega-runtime.toml`'s `keys_per_split`)
/// must still render the byte-identical single dispatch -- admitting the
/// rewrite is necessary but not sufficient for a SECOND dispatch to be
/// worth its own fixed overhead (design risk 2). ROW 385: a merge additionally
/// requires `context_length >= ATTENTION_SPLIT_KEYS_PER_SPLIT_AT_SCALE` --
/// below that knee [`cached_attention_per_query_head_grid`] takes over the
/// short-context case with one threadgroup per query head instead, and
/// `splits_for`'s own small-divisor branch would otherwise still report
/// `> 1` there, which [`crate::metal::pack_cached_attention_uniforms`] would
/// then slice the live key range by with no second dispatch to merge the
/// slices back -- an out-of-bounds-shaped undercount, not merely a missed
/// optimization.
///
/// Takes `kind` (not a bare `context_length: u64`) because `ContextSplitMerge`
/// is implemented ONLY inside [`render_cached_attention`]'s `single_range_
/// dynamic` branch (`cached_key_rows == 0`, the nine-operand fused kind):
/// that branch alone writes the `(max, sum, weighted)` scratch layout
/// [`render_cached_attention_merge`] reads back. The `two_range_cached_bound`
/// branch (nine operands, `cached_key_rows != 0`, `render_cached_attention`'s
/// own two-range `context_chunks <= 1` / `> 1` arms) ALWAYS writes the
/// complete, correctly-normalized attention output straight to `out[]` in one
/// pass, regardless of `context_length` -- it has no split/merge protocol at
/// all. A bare `context_length: u64` signature let three of this function's
/// four call sites (`emit`, `kernel_dispatch_shape`,
/// `emit_cached_attention_merge`) answer `true` for the two-range kind once
/// `cached_key_rows + new_key_rows >= ATTENTION_SPLIT_KEYS_PER_SPLIT_AT_SCALE`
/// (measured: `cached_len=200`/`700`, `NumericPolicy::llama_relaxed()`,
/// `qwen3_gqa_qk_norm_forward_fixture` -- max_diff 12.6 / 10.4 against the CPU
/// oracle) -- `emit`/`kernel_dispatch_shape` then swapped the op's bindings to
/// `split_bindings_with_scratch` (treating `out` as a raw per-split scratch
/// table it was never written as) and `emit_cached_attention_merge` dispatched
/// a companion merge kernel that reinterpreted the already-correct, already-
/// normalized `out[]` bytes as `(max, sum, weighted[head_dim])` triples and
/// overwrote them with garbage. `single_range_dynamic` here reproduces the
/// SAME discriminator [`render_cached_attention`] (`:3414`) and `grid_threads`
/// (`:2477`) already compute correctly, so all four call sites now agree by
/// construction instead of by convention.
pub(crate) fn cached_attention_merge_needed(kind: &BoundOpKind, policy: NumericPolicy) -> bool {
    let BoundOpKind::CachedAttention {
        operands,
        cached_key_rows,
        new_key_rows,
        ..
    } = kind
    else {
        return false;
    };
    let single_range_dynamic =
        (operands.len() == 9 || operands.len() == 12) && *cached_key_rows == 0;
    if !single_range_dynamic {
        return false;
    }
    let context_length = cached_key_rows + new_key_rows;
    splits_for(context_length, policy) > 1
        && context_length >= crate::sized::ATTENTION_SPLIT_KEYS_PER_SPLIT_AT_SCALE
}

/// Whether `render_cached_attention`'s single-range dynamic path dispatches
/// one threadgroup per `(query_row, kv_head, group)` triple -- narrowing
/// [`tiled_gemm_threadgroup_width`]'s `CachedAttention` width from
/// `query_groups * chunks * SIMD_WIDTH` to `chunks * SIMD_WIDTH` and turning
/// `query_groups` into an extra THREADGROUP-count factor instead -- rather
/// than today's one threadgroup per `(query_row, kv_head)` pair shared by
/// every query head in the group. Below
/// [`crate::sized::ATTENTION_SPLIT_KEYS_PER_SPLIT_AT_SCALE`] keys the
/// occupancy problem is too few threadgroups, not too little per-threadgroup
/// parallelism (ROW 383's own knee), so spending `query_groups` as more,
/// narrower threadgroups instead of more warps inside one threadgroup gives
/// the GPU more independent units of work to schedule at a window where it
/// is otherwise starved. `dynamic_cached_len` gates this to the single-range
/// fused (nine-operand) path alone -- the compiled fixed-shape path already
/// sizes its own dispatch from a KNOWN `context_length`, so it never carries
/// this trade. Not policy-gated: unlike [`cached_attention_merge_needed`],
/// this reshapes an EXISTING dispatch rather than reassociating the online-
/// softmax fold, so `bit_exact` renders it exactly like every other policy.
pub(crate) fn cached_attention_per_query_head_grid(
    dynamic_cached_len: bool,
    context_length: u64,
) -> bool {
    dynamic_cached_len && context_length < crate::sized::ATTENTION_SPLIT_KEYS_PER_SPLIT_AT_SCALE
}

