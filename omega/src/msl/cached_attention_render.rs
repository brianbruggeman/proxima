use super::*;

/// `BoundOpKind::CachedAttention`'s Metal kernel: online (running max/sum,
/// register-resident weighted-value accumulator) softmax attention over a
/// cached range plus a new range, one 32-lane simdgroup per
/// `(query_row, kv_head, group[, chunk])` tuple. Each simdgroup reads ONLY
/// its own assigned keys straight from device memory into per-lane
/// registers (`head_dim / 2` even/odd K pairs, `head_dim` V elements,
/// lane-strided), reduces the QK dot with `simd_sum`, and folds the running
/// max/sum/weighted-value purely in registers — the same per-simdgroup
/// register-resident shape as llama.cpp's `kernel_flash_attn_ext_vec`
/// (ggml-metal.metal:4014-4110), which loads K/V per simdgroup from device
/// memory and reduces with `simd_shuffle_down`/`simd_max` rather than
/// staging through `threadgroup` memory. There is no `threadgroup_barrier`
/// inside the per-key loop: nothing is shared across simdgroups until the
/// keys are exhausted, so there is nothing to synchronize on every
/// iteration. The only `threadgroup_barrier` calls left are around the
/// cross-simdgroup merge below, when `context_chunks > 1` splits one
/// `(query_row, kv_head, group)` triple's key range across simdgroups that
/// must combine their partial online-softmax state afterward.
/// Shared body of `render_cached_attention`'s three per-key sequential score
/// loops (`chunks<=1`, `context_chunks>1`, and the single-range dynamic
/// path's `block_width<=1` arm) -- byte-identical across all three, and
/// identical to the pre-pass-plane text, whenever `pass_present` is `false`
/// (every full-rotary caller today). `true` inserts one extra lane-strided
/// accumulation into the SAME `partial_score` reduction the rotary planes
/// already feed (`physical.rs:477-488`'s additive pass-dot term, the CPU
/// reference this must match) and switches the V read off the `kbase * 2`
/// pair-doubling shortcut -- only valid when `rotary_dim == head_dim`, since
/// it assumes `pair_dim * 2 == head_dim` -- to a value address computed
/// straight from `head_dim`, mirroring `physical.rs`'s own separate
/// `value_start`/`key_start` addressing.
pub(super) fn cached_attention_scalar_score_body(pass_present: bool) -> String {
    let pair_expr = if pass_present { "pair_dim" } else { "head_dim / 2" };
    let value_addr = if pass_present { "value_base" } else { "kbase * 2" };
    let value_base_decl = if pass_present {
        "long value_base = (cached ? key : new_index) * (kv_heads * head_dim) + kv_head * head_dim;\n        "
    } else {
        ""
    };
    let pass_accum = if pass_present {
        "\n        long pass_kbase = (cached ? key : new_index) * (kv_heads * pass_dim) + kv_head * pass_dim;\n        for (long dimension = (long)lane; dimension < pass_dim; dimension += 32L) {\n            partial_score += pass_query[pass_qbase + dimension] * (cached ? pass_cached_key[pass_kbase + dimension] : pass_new_key[pass_kbase + dimension]);\n        }".to_string()
    } else {
        String::new()
    };
    format!(
        "        bool cached = key < cached_key_rows; long new_index = key - cached_key_rows;\n        long relative = (cached ? key - cached_key_rows : new_index) - query_row;\n        if (cached && relative < cached_lower) {{ continue; }}\n        if (!cached && relative > new_upper) {{ continue; }}\n        long kbase = (cached ? key : new_index) * (kv_heads * ({pair_expr})) + kv_head * ({pair_expr});\n        {value_base_decl}float partial_score = 0.0f;\n        for (long pair = (long)lane; pair < {pair_expr}; pair += 32L) {{\n            partial_score += in0[qbase + pair] * (cached ? in2[kbase + pair] : in4[kbase + pair]);\n            partial_score += in1[qbase + pair] * (cached ? in3[kbase + pair] : in5[kbase + pair]);\n        }}{pass_accum}\n        float score = simd_broadcast_first(simd_sum(partial_score)) * scale;\n        float next_max = max(maximum, score);\n        float weight = exp(score - next_max); float rescale = (maximum == -INFINITY) ? 0.0f : exp(maximum - next_max);\n        sum = sum * rescale + weight;\n        for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {{\n            long local_dimension = dimension / 32L;\n            weighted[local_dimension] = weighted[local_dimension] * rescale + weight * (cached ? in6[{value_addr} + dimension] : in7[{value_addr} + dimension]);\n        }}\n        maximum = next_max;\n"
    )
}

pub(super) fn render_cached_attention(
    resolved: &BoundOp,
    entry: &str,
    numeric_policy: NumericPolicy,
) -> Result<String, EmitError> {
    let BoundOpKind::CachedAttention {
        query_rows,
        cached_key_rows,
        new_key_rows,
        kv_heads,
        query_groups,
        head_dim,
        rotary_dim,
        scale,
        cached_lower_inclusive,
        new_upper_inclusive,
        two_pass,
        ..
    } = &resolved.kind
    else {
        return Err(EmitError::RenderKindMismatch {
            node: resolved.node,
            expected: "cached_attention",
            found: resolved.kind.name(),
        });
    };
    // the gemma4 recognizer's own `staged_decode_only`/
    // `staged_scale_not_unity` decline stages
    // (`proxima-tensor/src/bind/dead_code_cached_attention.rs`) already
    // guarantee `two_pass` ops are decode-only at unity scale before this
    // op can exist -- select the staged emitter BEFORE any of the
    // online-softmax text below is built.
    #[cfg(feature = "metal-fuse-attn-decode")]
    if *two_pass {
        return crate::msl::render_cached_attention_two_pass(resolved, entry, false);
    }
    #[cfg(not(feature = "metal-fuse-attn-decode"))]
    let _ = two_pass;
    // `rotary_dim < head_dim` (`BoundOpKind::CachedAttention`'s own doc) is
    // qwen35's partial-rotary shape: the trailing `pass_query`/
    // `pass_cached_key`/`pass_new_key` operands carry one un-rotated plane
    // per side, scored as an extra additive dot term (`physical.rs:477-488`,
    // the same CPU reference this kernel must match). `rotary_dim > head_dim`
    // is never legal (`physical.rs`'s own `score_cached_attention_rotary`
    // guard), so that shape alone still rejects.
    if rotary_dim > head_dim {
        return Err(EmitError::CachedAttentionPartialRotaryNotSupported { node: resolved.node });
    }
    let pass_present = rotary_dim < head_dim;
    let pass_dim = head_dim - rotary_dim;
    let element_type = type_token(resolved.node, resolved.dtype)?;
    let cached_lower = if *cached_lower_inclusive == i64::MIN {
        "-9223372036854775807L".to_string()
    } else {
        format!("{cached_lower_inclusive}L")
    };
    // The ninth operand slot carries two DIFFERENT runtime scalars,
    // discriminated by `cached_key_rows` (`BoundOpKind::CachedAttention`'s
    // own doc): `cached_key_rows == 0` is the single-range fusion's true
    // `cached_len`, read here as `new_upper`; `cached_key_rows != 0` is the
    // two-range fusion's own live cached-row count, read further down as a
    // runtime `cached_key_rows` (`row_count_decl`) instead -- that shape
    // needs no dynamic `new_upper` at all, since its "new" range is never
    // bucketed. `entry_name`'s own "dyn"/"cb" markers are what let one
    // compiled kernel serve every live value on each path.
    let base_operand_len = if pass_present { 11 } else { 8 };
    let has_ninth_operand = resolved.operands().len() == base_operand_len + 1;
    let single_range_dynamic = has_ninth_operand && *cached_key_rows == 0;
    let two_range_cached_bound = has_ninth_operand && *cached_key_rows != 0;
    let (cached_len_param, new_upper_decl) = if single_range_dynamic {
        (
            format!(", device const {element_type}* in8 [[buffer(8)]]"),
            "long new_upper = (long)in8[0];".to_string(),
        )
    } else if two_range_cached_bound {
        (
            format!(", device const {element_type}* in8 [[buffer(8)]]"),
            format!("constexpr long new_upper = {new_upper_inclusive}L;"),
        )
    } else {
        (
            String::new(),
            format!("constexpr long new_upper = {new_upper_inclusive}L;"),
        )
    };
    // Trailing pass-plane buffers land immediately after the optional ninth
    // `cached_len` slot (`BoundOpKind::CachedAttention`'s own operand-layout
    // doc), so their buffer indices shift by one when that slot is present --
    // `bindings()` (`msl.rs`'s own doc) already derives the same order
    // generically from `resolved.all_read_sources()`, this just names the
    // matching Metal buffer indices in the kernel text.
    let pass_base_index = 8 + usize::from(has_ninth_operand);
    let pass_param = if pass_present {
        format!(
            ", device const {element_type}* pass_query [[buffer({})]], device const {element_type}* pass_cached_key [[buffer({})]], device const {element_type}* pass_new_key [[buffer({})]]",
            pass_base_index,
            pass_base_index + 1,
            pass_base_index + 2,
        )
    } else {
        String::new()
    };
    let out_index_offset = if pass_present { 3 } else { 0 };
    // The split kernel-plus-merge shape (redesign §4c) reads WHICH
    // threadgroup a simdgroup belongs to directly from Metal's own
    // per-dispatch coordinate rather than re-deriving it from `gid` --
    // `gid / threadgroup_width` would give the same value today, but that
    // equality holds only because `threadgroup_width` never itself depends
    // on `splits` (see `grid_threads`'s own doc); a hardware-native readback
    // does not carry that assumption. Only the single-range dynamic path
    // dispatches more than one threadgroup per `(query_row, kv_head)` pair,
    // so only that path declares the parameter.
    let tgid_param = if single_range_dynamic {
        ", uint tgid [[threadgroup_position_in_grid]]"
    } else {
        ""
    };
    let (out_buffer_index, uniforms_buffer_index) = {
        let out_index = pass_base_index + out_index_offset;
        (out_index, out_index + 1)
    };
    // `cached_key_rows`/`new_key_rows` are runtime `Uniforms` fields on the
    // single-range fused path (`dynamic_cached_len`, the ninth-operand
    // form). Redesign §5 option 2: `context_chunks` joins them as a fourth
    // runtime field on that SAME path, so `entry_name` (below) can drop the
    // `_c{}`/`_n{}` row-count tokens entirely -- the compiled kernel text no
    // longer depends on the compiled `kv-capacity-bucket` extent at all, and
    // one pipeline serves every bucket, matching llama's own `ne11`-as-
    // runtime-field property (`ggml-metal.m:4790-4813`).
    // `splits` (redesign §4c, [`NumericRewrite::ContextSplitMerge`]) rides
    // the SAME "runtime `Uniforms` field, never a `constexpr`" precedent
    // `context_chunks` just set two lines below -- packed by
    // `crate::metal::pack_cached_attention_uniforms` from the SAME
    // compiled-capacity input `context_chunks_for` already takes, via
    // `splits_for`. Declared here so the packed byte layout the driver
    // writes (`pack_cached_attention_uniforms`) and the struct this text
    // declares never drift out of sync. Not yet read anywhere in this
    // kernel body -- `render_cached_attention_merge`, the split/scratch
    // write, and the encoder's second dispatch are the follow-up that
    // consumes it (see this crate's `attention-kernel-design.md` §4c, risk
    // 1: the scratch-buffer plumbing this needs is unverified against
    // `metal.rs`'s current allocation call sites and is out of scope here).
    let uniforms_struct = if single_range_dynamic {
        "struct Uniforms { long total_elements; long cached_key_rows; long new_key_rows; long context_chunks; long splits; };\n\n"
    } else {
        "struct Uniforms { long total_elements; };\n\n"
    };
    // `two_range_cached_bound` substitutes a runtime-read `cached_key_rows`
    // for the compiled `constexpr` everywhere downstream that name already
    // appears (`bool cached = key < cached_key_rows`, the `new_index`
    // offset subtraction, `last_key_decl`'s sum below) -- one declaration
    // site changes what every existing use of the variable name means,
    // rather than a new gate at each of those uses (`BoundOpKind::
    // CachedAttention`'s own doc on why this is the fewer-renderer-lines
    // shape). `new_key_rows` is never bucketed on this path, so it stays a
    // compiled constant.
    let row_count_decl = if single_range_dynamic {
        "long cached_key_rows = u.cached_key_rows; long new_key_rows = u.new_key_rows;"
    } else if two_range_cached_bound {
        "long cached_key_rows = (long)in8[0]; constexpr long new_key_rows = {new_key_rows};"
    } else {
        "constexpr long cached_key_rows = {cached_key_rows}; constexpr long new_key_rows = {new_key_rows};"
    };
    let row_count_decl = row_count_decl
        .replace("{cached_key_rows}", &cached_key_rows.to_string())
        .replace("{new_key_rows}", &new_key_rows.to_string());
    let mut source = String::new();
    preamble(&mut source);
    source.push_str(uniforms_struct);
    source.push_str(&format!(
        "kernel void {entry}(device const {element_type}* in0 [[buffer(0)]], device const {element_type}* in1 [[buffer(1)]], device const {element_type}* in2 [[buffer(2)]], device const {element_type}* in3 [[buffer(3)]], device const {element_type}* in4 [[buffer(4)]], device const {element_type}* in5 [[buffer(5)]], device const {element_type}* in6 [[buffer(6)]], device const {element_type}* in7 [[buffer(7)]]{cached_len_param}{pass_param}, device {element_type}* out [[buffer({out_buffer_index})]], constant Uniforms& u [[buffer({uniforms_buffer_index})]], uint gid [[thread_position_in_grid]]{tgid_param}) {{\n"
    ));
    source.push_str("    if ((long)gid >= u.total_elements * 32L) { return; }\n");
    // `pair_dim`/`pass_dim` are only ever declared when a pass plane is
    // bound -- the full-rotary path never emits this text, keeping its
    // kernel source byte-identical to before this plane existed.
    let pass_dim_decl = if pass_present {
        format!(" constexpr long pair_dim = {rotary_dim} / 2; constexpr long pass_dim = {pass_dim};")
    } else {
        String::new()
    };
    source.push_str(&format!(
        "    {row_count_decl} constexpr long kv_heads = {kv_heads}; constexpr long query_groups = {query_groups}; constexpr long head_dim = {head_dim}; constexpr float scale = {}; constexpr long cached_lower = {cached_lower}; {new_upper_decl}{pass_dim_decl}\n",
        msl_literal(*scale),
    ));
    // One threadgroup per (query_row, kv_head) pair -- `tiled_gemm_
    // threadgroup_width`'s `CachedAttention` arm dispatches exactly
    // `query_groups * SIMD_WIDTH` threads per threadgroup, and `dispatchThreads_
    // threadsPerThreadgroup`'s linear grouping (`thread_position_in_grid =
    // threadgroup_position_in_grid * width + thread_position_in_threadgroup`)
    // lands every one of `query_groups` simdgroups (one per query head sharing
    // this kv_head) in the SAME threadgroup, because `vector_index`'s own
    // encoding already cycles `group` fastest, then `kv_head`, then
    // `query_row` -- see this function's doc. `tid`/`group_width` below are
    // therefore derivable from the existing `group`/`lane` split without a
    // new `[[thread_position_in_threadgroup]]` kernel parameter.
    // Two distinct decisions, both fed by the bind-time `cached_key_rows` /
    // `new_key_rows` fields but not the same computation: CHUNK SIZING
    // (`context_chunks`, immediately below -- how many simdgroups share this
    // key range) versus the per-simdgroup stride LOOP's own upper bound
    // (`last_key`, below that). A merged-KV bind that declares its empty
    // cached half as `cached_key_rows: 0` (rather than duplicating the
    // capacity into it) changes what this line feeds `context_chunks_for`,
    // which can change the chunk count -- a repartitioning, not merely a
    // loop-bound change.
    // The rotary Q/K planes are `rotary_dim / 2` pairs wide, never
    // `head_dim / 2` -- identical to `head_dim / 2` whenever `rotary_dim ==
    // head_dim` (every full-rotary caller today), so substituting this
    // token in place of the literal keeps full-rotary kernel text
    // byte-identical while partial rotary addresses only its rotated width.
    let pair_expr = if pass_present { "pair_dim" } else { "head_dim / 2" };
    // Companion to `qbase`, in `pass_dim` units instead of `pair_expr`
    // units -- `physical.rs:444`'s own `pass_query_start` addressing, ported
    // verbatim. Empty for full rotary, so that path's `qbase` line renders
    // byte-identical to before this plane existed.
    let pass_qbase_decl = if pass_present {
        " long pass_qbase = query_row * (kv_heads * query_groups * pass_dim) + query_head * pass_dim;"
    } else {
        ""
    };
    let context_chunks = context_chunks_for(
        *cached_key_rows + *new_key_rows,
        *query_groups,
        *head_dim,
        numeric_policy,
    );
    // Redesign §5 option 2: on the single-range fused path, the compiled
    // MAXIMUM simdgroup count (`cap`) sizes both the dispatch grid and the
    // threadgroup-memory merge arrays -- never the bind's own compiled
    // `context_chunks_for(...)` result, which varies with the compiled
    // `kv-capacity-bucket` extent and is exactly the quantity `entry_name`
    // must stop depending on. A live `chunks <= cap` still travels as a
    // runtime `Uniforms` field (packed by `pack_cached_attention_uniforms`),
    // and simdgroups `chunk >= chunks` contribute the merge's own identity
    // (`maximum = -INFINITY`, `sum = 0.0`) rather than looping -- the same
    // "idle simdgroup, identity partial" shape llama's own dispatch-time
    // `nsg` uses against a compiled maximum (`ggml-metal.m:4887-4913`).
    let cap = effective_context_chunk_cap(*query_groups, *head_dim);
    // The block-staged body's `float4` K/Q loads (below) reinterpret each
    // real/imaginary plane offset as `device const {element_type}4*` --
    // legal only when every `qbase`/`kbase` offset this kernel computes is a
    // multiple of 4 floats. Both offsets are integer multiples of
    // `head_dim / 2`, so `head_dim / 2` itself being a multiple of 4
    // (equivalently, `head_dim` a multiple of 8) is sufficient regardless of
    // `kv_heads`/`query_row`/`kv_head` -- see [`EmitError::
    // AttentionBlockMisaligned`]'s own doc.
    // Partial rotary falls back to the sequential per-key walk regardless of
    // policy: the block-staged float4 loads below assume the rotary plane
    // spans the whole head (`qbase`/`kbase` offsets a multiple of
    // `head_dim / 2`), and vectorizing the extra pass-plane dot term is
    // unimplemented -- correctness first, the TreeReduce speed rewrite for
    // this shape is a follow-up, not a blocker for landing the plane itself.
    let block_width = if pass_present {
        1
    } else {
        block_width_for(numeric_policy)
    };
    if single_range_dynamic && block_width > 1 && !head_dim.is_multiple_of(8) {
        return Err(EmitError::AttentionBlockMisaligned {
            node: resolved.node,
            head_dim: *head_dim,
        });
    }
    // The stride loop's live upper bound: every key past this point would
    // hit the `relative > new_upper` / `relative < cached_lower` `continue`
    // on every remaining iteration within THIS simdgroup's own assigned
    // subset (both bands are monotone in `key`), so stopping here changes
    // no arithmetic and no read this simdgroup would have skipped anyway --
    // it removes iteration/address/control overhead, not any accumulate or
    // K/V load, and the dead-band checks already preceded every load and
    // multiply-add in the loop this replaces. This does NOT halve dispatched
    // work: chunks stride-interleave keys across simdgroups, so a "fake"
    // key was never confined to one simdgroup's idle range, and this bound
    // is evaluated per simdgroup, independently of the chunk-sizing decision
    // above. `new_key_rows` is still the compiled buffer extent (the read
    // bound never moves); only the iteration count does.
    let last_key_decl = if single_range_dynamic {
        "long last_key = cached_key_rows + min(new_key_rows - 1L, query_row + new_upper);\n"
    } else {
        "long last_key = cached_key_rows + new_key_rows - 1L;\n"
    };
    // `grid_threads` widens the dispatch by the compiled `ATTENTION_SPLIT_MAX`
    // (never the live `splits_for` result) exactly when a merge is needed --
    // `tgid`'s own decode below must divide out that SAME compiled multiplier,
    // not the live `u.splits` value, or a `tgid` beyond `rows * kv_heads *
    // live_splits` (which the widened grid always dispatches) decodes a
    // `query_row` past the real row count and reads/writes out of bounds.
    // Computed once here so the decode and `final_store`'s scratch-vs-direct
    // branch below can never disagree on which case this compiled kernel is.
    let merge_needed = cached_attention_merge_needed(&resolved.kind, numeric_policy);
    // ROW 385: below the split-at-scale knee, `query_groups` moves out of
    // this threadgroup's own width (`tiled_gemm_threadgroup_width`'s own
    // `CachedAttention` arm) and into a threadgroup-COUNT factor instead, so
    // `group` decodes off `tgid` here rather than off `vector_index`.
    // `entry_name` names this decision (`_qh{0|1}`) precisely because it
    // changes the text below -- the compiled `kv-capacity-bucket` extent
    // still never appears in the text ITSELF, only this boolean does.
    let per_query_head_grid = cached_attention_per_query_head_grid(
        single_range_dynamic,
        *cached_key_rows + *new_key_rows,
    );
    if single_range_dynamic {
        let grid_splits = if merge_needed {
            crate::sized::ATTENTION_SPLIT_MAX
        } else {
            1
        };
        let tgid_decode = if per_query_head_grid {
            "long kv_head_and_group = (long)tgid % (kv_heads * query_groups);\n    long query_row_and_split = (long)tgid / (kv_heads * query_groups);\n    long kv_head = kv_head_and_group / query_groups;\n    long group = kv_head_and_group % query_groups;\n    long chunk = vector_index % cap;\n"
        } else {
            "long chunk = vector_index % cap;\n    long group = (vector_index / cap) % query_groups;\n    long kv_head = (long)tgid % kv_heads;\n    long query_row_and_split = (long)tgid / kv_heads;\n"
        };
        source.push_str(&format!(
            "    long vector_index = (long)gid / 32L; uint lane = gid % 32u;\n    if (vector_index >= u.total_elements) {{ return; }}\n    constexpr long cap = {cap};\n    long chunks = u.context_chunks;\n    long splits = u.splits;\n    {tgid_decode}    constexpr long grid_splits = {grid_splits};\n    long split = query_row_and_split % grid_splits;\n    long query_row = query_row_and_split / grid_splits;\n    if (split >= splits) {{ return; }}\n    long query_index = query_row * (kv_heads * query_groups) + kv_head * query_groups + group;\n    long query_head = kv_head * query_groups + group;\n    long qbase = query_row * (kv_heads * query_groups * ({pair_expr})) + query_head * ({pair_expr});{pass_qbase_decl}\n    long local_group_index = group * cap + chunk;\n    float maximum = -INFINITY; float sum = 0.0f; float weighted[(head_dim + 31) / 32];\n    for (long dimension = 0; dimension < (head_dim + 31) / 32; dimension++) {{ weighted[dimension] = 0.0f; }}\n"
        ));
        source.push_str(&format!("    {last_key_decl}"));
        // Declared once here (not per `block_width` arm) because the
        // cross-chunk merge below -- `shared_m`/`shared_l`/`shared_o` reads
        // and the `final_store` that follows it -- runs identically whether
        // `block_width <= 1` (bit_exact) or `> 1` (the block-staged body);
        // only the `block_width > 1` arm additionally reuses `shared_o` as
        // its own V-accumulate scratch (this function's own doc on that
        // arm), never a second, separate allocation.
        source.push_str("    threadgroup float shared_m[query_groups * cap]; threadgroup float shared_l[query_groups * cap]; threadgroup float shared_o[query_groups * cap * head_dim];\n");
        // Redesign §4c: this threadgroup's slice of the LIVE key range
        // (`last_key + 1`, ROW 366's live upper) -- `ceil_div(live, splits)`
        // sized so every split but the last is exactly `slice_len` keys and
        // the last absorbs the remainder; `lo >= hi` (a `split` past the end
        // of a short live range) leaves the per-key loop below empty, which
        // is already this kernel's identity partial (`maximum = -INFINITY`,
        // `sum = 0.0`, `weighted` zeroed above) -- no separate branch needed.
        source.push_str(
            "    long live = last_key + 1L;\n    long slice_len = (live + splits - 1L) / splits;\n    long lo = split * slice_len;\n    long hi = min(lo + slice_len, live);\n",
        );
        // Idle simdgroups (`chunk >= chunks`, live count below the compiled
        // maximum) skip the walk entirely and keep the identity partial
        // (`maximum = -INFINITY`, `sum = 0.0`, `weighted` zeroed above) --
        // the merge below already treats a `-INFINITY` partial as
        // zero-weight (`rescale = 0.0f`), so an idle simdgroup contributes
        // nothing to the merged result, bit-for-bit.
        // `block_width == 1` renders EXACTLY the strictly-sequential per-key
        // walk (byte for byte) -- see [`block_width_for`]'s own doc for why
        // this is the `bit_exact` lowering, not a fallback bolted on beside
        // it. `block_width > 1` renders §4b's block-staged walk: llama's
        // `kernel_flash_attn_ext_vec` shape (ggml-metal.metal:4014-4143)
        // ported onto this kernel's cached/new dual-range band masking --
        // `ss[]` stages `block_width` raw scores via a `float4` Q·K dot plus
        // an 8-lane `simd_shuffle_down` tree reduce (`ty = lane/8` selects
        // one of 4 concurrent keys, `tx = lane%8` selects a float4 chunk of
        // the real/imaginary planes), one `simd_max`/`simd_sum` combine per
        // 32-lane sub-block (not per key), then a threadgroup-broadcast read
        // of each key's softmax weight back out of the SAME `ss[]` slots for
        // the V accumulate -- llama's own `ss[]` reuse
        // (ggml-metal.metal:4114, `ss[tiisg] = vs;`). `ss[]` is sized and
        // indexed per `local_group_index` exactly like `shared_m`/
        // `shared_l`/`shared_o` below it, so concurrent simdgroups in the
        // same threadgroup never alias each other's staged scores.
        if block_width <= 1 {
            source.push_str(&format!(
                "    if (chunk < chunks) {{\n    for (long key = lo + chunk; key < hi; key += chunks) {{\n{}    }}\n    }}\n",
                cached_attention_scalar_score_body(pass_present),
            ));
        } else {
            // The float4/ty-group V accumulate (llama's `kernel_flash_attn_
            // ext_vec` register form, ggml-metal.metal:4125-4143) needs each
            // `tx` lane's `v_registers` float4 loads to land on a 16-byte
            // boundary and never read past `head_dim` floats -- true only
            // when `head_dim` is itself a multiple of 32 (8 `tx` lanes times
            // 4 floats/float4). Smaller/odd `head_dim` (the `head_dim == 8`
            // fixtures below) keep today's scalar per-key V loop, byte for
            // byte, rather than mis-sizing the vector loads or rejecting a
            // shape the Q·K side already renders correctly.
            let v_accumulate = if head_dim.is_multiple_of(32) {
                let v_registers = head_dim / 8 / 4;
                format!(
                    "            {{\n                constexpr long v_registers = {v_registers};\n                float4 v_acc[v_registers];\n                for (long register_index = 0L; register_index < v_registers; register_index++) {{ v_acc[register_index] = float4(0.0f); }}\n                for (long cc4 = 0L; cc4 < 8L; cc4++) {{\n                    long local_index = block_start + sub + 4L * cc4 + (long)ty;\n                    bool valid = local_index < min(block_start + sub + 32L, num_local_keys);\n                    long key = slice_start + local_index * chunks;\n                    bool cached = key < cached_key_rows; long new_index = key - cached_key_rows;\n                    long kbase = (cached ? key : new_index) * (kv_heads * (head_dim / 2)) + kv_head * (head_dim / 2);\n                    float key_weight = valid ? ss[local_group_index * block_width + (local_index - block_start)] : 0.0f;\n                    device const {element_type}4* v4 = (device const {element_type}4*)((cached ? in6 : in7) + kbase * 2);\n                    for (long register_index = 0L; register_index < v_registers; register_index++) {{\n                        v_acc[register_index] += valid ? float4(v4[(long)tx + 8L * register_index]) * key_weight : float4(0.0f);\n                    }}\n                }}\n                for (long register_index = 0L; register_index < v_registers; register_index++) {{\n                    v_acc[register_index] += simd_shuffle_xor(v_acc[register_index], 8);\n                    v_acc[register_index] += simd_shuffle_xor(v_acc[register_index], 16);\n                }}\n                if (ty == 0) {{\n                    threadgroup float4* shared_o4 = (threadgroup float4*)(shared_o + local_group_index * head_dim);\n                    for (long register_index = 0L; register_index < v_registers; register_index++) {{ shared_o4[(long)tx + 8L * register_index] = v_acc[register_index]; }}\n                }}\n                simdgroup_barrier(mem_flags::mem_threadgroup);\n                for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {{\n                    long local_dimension = dimension / 32L;\n                    weighted[local_dimension] += shared_o[local_group_index * head_dim + dimension];\n                }}\n                simdgroup_barrier(mem_flags::mem_threadgroup);\n            }}\n"
                )
            } else {
                "            for (long local_index = block_start + sub; local_index < min(block_start + sub + 32L, num_local_keys); local_index++) {\n                long key = slice_start + local_index * chunks;\n                bool cached = key < cached_key_rows; long new_index = key - cached_key_rows;\n                long kbase = (cached ? key : new_index) * (kv_heads * (head_dim / 2)) + kv_head * (head_dim / 2);\n                float key_weight = ss[local_group_index * block_width + (local_index - block_start)];\n                for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {\n                    long local_dimension = dimension / 32L;\n                    weighted[local_dimension] += key_weight * (cached ? in6[kbase * 2 + dimension] : in7[kbase * 2 + dimension]);\n                }\n            }\n".to_string()
            };
            source.push_str(&format!(
                "    constexpr long block_width = {block_width};\n    short ty = (short)(lane / 8u); short tx = (short)(lane % 8u);\n    threadgroup float ss[query_groups * cap * block_width];\n    if (chunk < chunks) {{\n    long slice_start = lo + chunk;\n    long num_local_keys = (slice_start < hi) ? ((hi - 1L - slice_start) / chunks) + 1L : 0L;\n    for (long block_start = 0L; block_start < num_local_keys; block_start += block_width) {{\n        for (long cc = 0L; cc < block_width / 4L; cc++) {{\n            long local_index = block_start + 4L * cc + (long)ty;\n            bool valid = local_index < num_local_keys;\n            long key = slice_start + local_index * chunks;\n            bool cached = key < cached_key_rows; long new_index = key - cached_key_rows;\n            if (valid) {{\n                long relative = (cached ? key - cached_key_rows : new_index) - query_row;\n                if (cached && relative < cached_lower) {{ valid = false; }}\n                if (!cached && relative > new_upper) {{ valid = false; }}\n            }}\n            long kbase = (cached ? key : new_index) * (kv_heads * (head_dim / 2)) + kv_head * (head_dim / 2);\n            float partial_score = 0.0f;\n            if (valid) {{\n                device const {element_type}4* qr4 = (device const {element_type}4*)(in0 + qbase);\n                device const {element_type}4* qi4 = (device const {element_type}4*)(in1 + qbase);\n                device const {element_type}4* kr4 = (device const {element_type}4*)((cached ? in2 : in4) + kbase);\n                device const {element_type}4* ki4 = (device const {element_type}4*)((cached ? in3 : in5) + kbase);\n                for (short index = tx; index < (short)((head_dim / 2) / 4L); index += 8) {{\n                    partial_score += dot(kr4[index], qr4[index]);\n                    partial_score += dot(ki4[index], qi4[index]);\n                }}\n            }}\n            partial_score += simd_shuffle_down(partial_score, 4);\n            partial_score += simd_shuffle_down(partial_score, 2);\n            partial_score += simd_shuffle_down(partial_score, 1);\n            if (tx == 0) {{ ss[local_group_index * block_width + 4L * cc + (long)ty] = valid ? partial_score * scale : -INFINITY; }}\n        }}\n        simdgroup_barrier(mem_flags::mem_threadgroup);\n        for (long sub = 0L; sub < block_width; sub += 32L) {{\n            long lane_index = sub + (long)lane;\n            float raw_score = ss[local_group_index * block_width + lane_index];\n            float next_max = simd_max(max(maximum, raw_score));\n            float rescale = (maximum == -INFINITY) ? 0.0f : exp(maximum - next_max);\n            float weight = exp(raw_score - next_max);\n            sum = sum * rescale + simd_sum(weight);\n            ss[local_group_index * block_width + lane_index] = weight;\n            for (long dimension = 0L; dimension < (head_dim + 31) / 32; dimension++) {{ weighted[dimension] *= rescale; }}\n            maximum = next_max;\n            simdgroup_barrier(mem_flags::mem_threadgroup);\n{v_accumulate}            simdgroup_barrier(mem_flags::mem_threadgroup);\n        }}\n    }}\n    }}\n"
            ));
        }
        // Redesign §4c ([`NumericRewrite::ContextSplitMerge`]): once the
        // active policy admits it, this simdgroup-level combine's own
        // result (`merged_max`/`merged_sum`/`weighted`, exactly what the
        // `bit_exact` arm below normalizes straight into `out`) is instead
        // the SPLIT kernel's own partial for split 0 -- written raw
        // (un-normalized) into the scratch buffer this position's
        // `Binding::Scratch` slot backs, at `splits` (the LIVE `u.splits`
        // value, not the compiled `ATTENTION_SPLIT_MAX`) stride so a later
        // slice's real `split = threadgroup_index % splits` can address its
        // own slot with the identical layout. This must be the same live
        // value `crate::metal::cached_attention_scratch_len` sizes the
        // buffer with (`splits_for(cached_key_rows + new_key_rows, ..)` at
        // `query_rows > 1`, the compiled max only at `query_rows == 1`) --
        // striding by the compiled max here while the buffer holds only
        // `splits` slots per row was an out-of-bounds write on every
        // `query_index > 0` (production, 2026-09-07: turn two of a chat,
        // ~600 cached keys plus ~100 new rows, produced 1024 `!!!!` tokens).
        // `render_cached_attention_merge` reads exactly this layout back and
        // performs the online-softmax combine over every live split -- the
        // final `weighted[..] / sum` normalize this arm always did is
        // deferred to that kernel, generalizing this exact combine one
        // hardware level up (`ContextSplitMerge`'s own doc). `split` is the
        // real per-threadgroup value this function's own index-unpack
        // derived from `tgid` above -- at `u.splits == 1` it is always `0`,
        // so this reduces to exactly the prior forced-`0L` behaviour byte
        // for byte.
        let final_store = if merge_needed {
            "        device float* attn_scratch = (device float*)out;\n        long scratch_index = (query_index * splits + split) * (2L + head_dim);\n        if (lane == 0u) { attn_scratch[scratch_index] = merged_max; attn_scratch[scratch_index + 1] = merged_sum; }\n        for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) { long local_dimension = dimension / 32L; attn_scratch[scratch_index + 2L + dimension] = weighted[local_dimension]; }\n".to_string()
        } else {
            format!(
                "        for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {{ long local_dimension = dimension / 32L; out[query_index * head_dim + dimension] = ({element_type})(sum == 0.0f ? 0.0f : weighted[local_dimension] / sum); }}\n"
            )
        };
        source.push_str(&format!(
            "    if (lane == 0u) {{ shared_m[local_group_index] = maximum; shared_l[local_group_index] = sum; }}\n    for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {{ shared_o[local_group_index * head_dim + dimension] = weighted[dimension / 32L]; }}\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n    if (chunk == 0L) {{\n        float merged_max = -INFINITY;\n        for (long c = 0; c < cap; c++) {{ merged_max = max(merged_max, shared_m[group * cap + c]); }}\n        float merged_sum = 0.0f;\n        for (long c = 0; c < cap; c++) {{\n            float partial_max = shared_m[group * cap + c];\n            float rescale = (partial_max == -INFINITY) ? 0.0f : exp(partial_max - merged_max);\n            merged_sum += shared_l[group * cap + c] * rescale;\n        }}\n        for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {{\n            long local_dimension = dimension / 32L;\n            float acc = 0.0f;\n            for (long c = 0; c < cap; c++) {{\n                float partial_max = shared_m[group * cap + c];\n                float rescale = (partial_max == -INFINITY) ? 0.0f : exp(partial_max - merged_max);\n                acc += shared_o[(group * cap + c) * head_dim + dimension] * rescale;\n            }}\n            weighted[local_dimension] = acc;\n        }}\n        sum = merged_sum;\n{final_store}    }}\n}}\n"
        ));
    } else if context_chunks <= 1 {
        source.push_str(&format!("    long vector_index = (long)gid / 32L; uint lane = gid % 32u;\n    if (vector_index >= u.total_elements) {{ return; }}\n    long query_index = vector_index;\n    long query_row = query_index / (kv_heads * query_groups);\n    long remainder = query_index % (kv_heads * query_groups);\n    long kv_head = remainder / query_groups;\n    long group = remainder % query_groups;\n    long query_head = kv_head * query_groups + group;\n    long qbase = query_row * (kv_heads * query_groups * ({pair_expr})) + query_head * ({pair_expr});{pass_qbase_decl}\n    float maximum = -INFINITY; float sum = 0.0f; float weighted[(head_dim + 31) / 32];\n    for (long dimension = 0; dimension < (head_dim + 31) / 32; dimension++) {{ weighted[dimension] = 0.0f; }}\n"));
        source.push_str(&format!("    {last_key_decl}"));
        // Each lane reads its own K/V elements straight from device memory
        // into registers, llama.cpp's `kernel_flash_attn_ext_vec` shape
        // (ggml-metal.metal:4037-4058 for K, the V accumulate mirrors it) --
        // no `threadgroup` staging, so no barrier is needed inside the loop:
        // nothing is shared across lanes or simdgroups until `simd_sum`
        // reduces the per-lane partial dot product within this simdgroup.
        source.push_str(&format!(
            "    for (long key = 0; key <= last_key; key++) {{\n{}    }}\n",
            cached_attention_scalar_score_body(pass_present),
        ));
        source.push_str(&format!("    for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {{ long local_dimension = dimension / 32L; out[query_index * head_dim + dimension] = ({element_type})(sum == 0.0f ? 0.0f : weighted[local_dimension] / sum); }}\n}}\n"));
    } else {
        // `context_chunks` simdgroups per (query_row, kv_head, group) triple
        // now share one threadgroup -- `local_group_index` (`group *
        // context_chunks + chunk`) is what `tiled_gemm_threadgroup_width`'s
        // `CachedAttention` arm sizes the dispatch width against, and its
        // encoding cycles `chunk` fastest so the SAME `dispatchThreads_
        // threadsPerThreadgroup` linear-grouping invariant this function's
        // doc already relies on for `group` still holds. Each simdgroup
        // walks ONLY the keys assigned to its `chunk` (`key = chunk; key <
        // total; key += context_chunks`), llama.cpp's own context-stride
        // partition (`ic0 += C*nsg`, ggml-metal.metal:4016), reading K/V
        // straight from device memory into registers exactly like the
        // `chunks<=1` body above -- so there is no per-key barrier here
        // either, and the set of keys this chunk visits is identical to the
        // old `key % context_chunks == chunk` gate, visited in the same
        // increasing order, so the accumulated floats are bit-identical.
        source.push_str(&format!(
            "    long vector_index = (long)gid / 32L; uint lane = gid % 32u;\n    if (vector_index >= u.total_elements) {{ return; }}\n    constexpr long context_chunks = {context_chunks};\n    long query_index = vector_index / context_chunks;\n    long chunk = vector_index % context_chunks;\n    long query_row = query_index / (kv_heads * query_groups);\n    long remainder = query_index % (kv_heads * query_groups);\n    long kv_head = remainder / query_groups;\n    long group = remainder % query_groups;\n    long query_head = kv_head * query_groups + group;\n    long qbase = query_row * (kv_heads * query_groups * ({pair_expr})) + query_head * ({pair_expr});{pass_qbase_decl}\n    long local_group_index = group * context_chunks + chunk;\n    float maximum = -INFINITY; float sum = 0.0f; float weighted[(head_dim + 31) / 32];\n    for (long dimension = 0; dimension < (head_dim + 31) / 32; dimension++) {{ weighted[dimension] = 0.0f; }}\n"
        ));
        source.push_str(&format!("    {last_key_decl}"));
        source.push_str(&format!(
            "    for (long key = chunk; key <= last_key; key += context_chunks) {{\n{}    }}\n",
            cached_attention_scalar_score_body(pass_present),
        ));
        // Single cross-simdgroup merge: each chunk's own (max, sum,
        // weighted) is the same online-softmax state the `chunks<=1` body
        // already computes over its own key subset; combining them is the
        // one piece of NEW arithmetic (`render_cached_attention`'s own
        // doc) -- rescale each chunk's partial by `exp(m_i - m_max)`, sum
        // `l`, sum `o`, exactly llama.cpp's `kernel_flash_attn_ext_vec`
        // cross-simdgroup reduction (ggml-metal.metal). Only `chunk == 0`
        // writes the merged result and the final output -- the other
        // chunks' registers are dead past this point.
        source.push_str(&format!(
            "    threadgroup float shared_m[query_groups * context_chunks]; threadgroup float shared_l[query_groups * context_chunks]; threadgroup float shared_o[query_groups * context_chunks * head_dim];\n    if (lane == 0u) {{ shared_m[local_group_index] = maximum; shared_l[local_group_index] = sum; }}\n    for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {{ shared_o[local_group_index * head_dim + dimension] = weighted[dimension / 32L]; }}\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n    if (chunk == 0L) {{\n        float merged_max = -INFINITY;\n        for (long c = 0; c < context_chunks; c++) {{ merged_max = max(merged_max, shared_m[group * context_chunks + c]); }}\n        float merged_sum = 0.0f;\n        for (long c = 0; c < context_chunks; c++) {{\n            float partial_max = shared_m[group * context_chunks + c];\n            float rescale = (partial_max == -INFINITY) ? 0.0f : exp(partial_max - merged_max);\n            merged_sum += shared_l[group * context_chunks + c] * rescale;\n        }}\n        for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {{\n            long local_dimension = dimension / 32L;\n            float acc = 0.0f;\n            for (long c = 0; c < context_chunks; c++) {{\n                float partial_max = shared_m[group * context_chunks + c];\n                float rescale = (partial_max == -INFINITY) ? 0.0f : exp(partial_max - merged_max);\n                acc += shared_o[(group * context_chunks + c) * head_dim + dimension] * rescale;\n            }}\n            weighted[local_dimension] = acc;\n        }}\n        sum = merged_sum;\n        for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {{ long local_dimension = dimension / 32L; out[query_index * head_dim + dimension] = ({element_type})(sum == 0.0f ? 0.0f : weighted[local_dimension] / sum); }}\n    }}\n}}\n"
        ));
    }
    let _ = query_rows;
    Ok(source)
}

/// `BoundOpKind::CachedAttention`'s MERGE kernel -- [`render_cached_attention`]'s
/// own doc, redesign §4c: reads back the `(max, sum, weighted[head_dim])`
/// partial each of `u.splits` split-kernel threadgroups wrote into the
/// scratch buffer (one simdgroup per output row) and combines them with the
/// SAME online-softmax rescale llama.cpp's own
/// `kernel_flash_attn_ext_vec_reduce` uses (`ggml-metal-ops.cpp:2063-2097`
/// on `origin/master`, this crate's `attention-kernel-design.md` §4c cites
/// it verbatim): lane `i` (`i < splits`) loads that split's own `(M_i,
/// S_i)`; `simd_max`/`simd_sum` combine the up-to-32 lanes (one lane per
/// split, `ATTENTION_SPLIT_MAX <= 32` so every live split fits in one
/// simdgroup's shuffle network, the same reason llama's own reduce needs no
/// tree); each lane then walks its own lane-strided subset of `head_dim`
/// dimensions, re-deriving every split's rescale weight straight from
/// device memory (cheap here -- this kernel is `O(splits * head_dim)`, not
/// the hot per-key loop) rather than shuffling `M`/`S` across lanes. At
/// `u.splits == 1` this reduces to exactly the prior forced-`split = 0`
/// copy-and-normalize, since the single live lane's rescale weight is
/// always `1.0` (or `0.0` under the all-`-INFINITY`/empty-context corner).
#[cfg(any(test, all(feature = "metal", target_os = "macos")))]
pub(super) fn render_cached_attention_merge(resolved: &BoundOp, entry: &str) -> Result<String, EmitError> {
    let BoundOpKind::CachedAttention { head_dim, .. } = &resolved.kind else {
        return Err(EmitError::RenderKindMismatch {
            node: resolved.node,
            expected: "cached_attention_merge",
            found: resolved.kind.name(),
        });
    };
    let element_type = type_token(resolved.node, resolved.dtype)?;
    let mut source = String::new();
    preamble(&mut source);
    source.push_str("struct Uniforms { long total_elements; long splits; };\n\n");
    source.push_str(&format!(
        "kernel void {entry}(device const float* in0 [[buffer(0)]], device {element_type}* out [[buffer(1)]], constant Uniforms& u [[buffer(2)]], uint gid [[thread_position_in_grid]]) {{\n"
    ));
    source.push_str(
        "    long vector_index = (long)gid / 32L; uint lane = gid % 32u;\n    if (vector_index >= u.total_elements) { return; }\n",
    );
    // `splits` is the SAME live `u.splits` value the split kernel's own
    // `final_store` addresses its scratch write with
    // (`render_cached_attention`'s doc) -- and the same value
    // `crate::metal::cached_attention_scratch_len` sized this buffer's
    // per-row extent against. Reading it back with the compiled
    // `ATTENTION_SPLIT_MAX` instead (this function's own bug until fixed
    // alongside the split kernel) would address a stride the allocator
    // never reserved.
    source.push_str(&format!(
        "    constexpr long head_dim = {head_dim};\n    long query_index = vector_index;\n    long splits = u.splits;\n    long iwg = (long)lane;\n    long own_scratch = (query_index * splits + iwg) * (2L + head_dim);\n    float own_max = (iwg < splits) ? in0[own_scratch] : -INFINITY;\n    float own_sum = (iwg < splits) ? in0[own_scratch + 1] : 0.0f;\n    float global_max = simd_max(own_max);\n    float own_weight = (own_max == -INFINITY) ? 0.0f : exp(own_max - global_max);\n    float total_sum = simd_sum(own_weight * own_sum);\n    float inv_sum = (total_sum == 0.0f) ? 0.0f : 1.0f / total_sum;\n    for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {{\n        float accumulated = 0.0f;\n        for (long split = 0; split < splits; split++) {{\n            long scratch_index = (query_index * splits + split) * (2L + head_dim);\n            float split_max = in0[scratch_index];\n            float split_weight = (split_max == -INFINITY) ? 0.0f : exp(split_max - global_max);\n            accumulated += split_weight * in0[scratch_index + 2L + dimension];\n        }}\n        out[query_index * head_dim + dimension] = ({element_type})(accumulated * inv_sum);\n    }}\n}}\n"
    ));
    Ok(source)
}

/// Whether `resolved` needs a companion merge dispatch under `numeric_policy`
/// and, if so, that dispatch's [`Kernel`] -- `None` for every op kind other
/// than `CachedAttention` and for `CachedAttention` itself when
/// [`cached_attention_merge_needed`] withholds `ContextSplitMerge` (the
/// `bit_exact` lowering: single dispatch, byte-identical to before this
/// redesign). This is [`emit`]'s own companion rather than a change to
/// `emit`'s signature: every OTHER caller of `emit` (`cuda.rs`, `wgsl.rs`,
/// the module doc's own example) keeps binding one `BoundOp` to one
/// `Kernel`, and `CachedAttention`'s own two-dispatch shape is additive —
/// `crate::metal`'s plan-resolution path calls this alongside `emit` for
/// every position, exactly as it already calls `kernel_dispatch_shape`
/// alongside `emit`.
#[cfg(any(test, all(feature = "metal", target_os = "macos")))]
pub(crate) fn emit_cached_attention_merge(
    resolved: &BoundOp,
    numeric_policy: NumericPolicy,
) -> Result<Option<Kernel>, EmitError> {
    if !cached_attention_merge_needed(&resolved.kind, numeric_policy) {
        return Ok(None);
    }
    validate(resolved)?;
    let entry = alloc::format!("{}_merge", entry_name(resolved));
    let source = render_cached_attention_merge(resolved, &entry)?;
    let total_elements = resolved
        .extents
        .iter()
        .product::<u64>()
        .checked_div(match &resolved.kind {
            BoundOpKind::CachedAttention { head_dim, .. } => *head_dim,
            _ => 1,
        })
        .unwrap_or(0);
    Ok(Some(Kernel {
        source,
        entry,
        bindings: merge_bindings(resolved),
        grid: GridSpec {
            threads: total_elements * SIMD_WIDTH,
            threadgroup_width: None,
            depth: 1,
        },
    }))
}

