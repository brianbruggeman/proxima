use super::*;

/// `simdgroup_matrix`-tiled Q4_K x F32 GEMM (`docs/discipline.md` ROW 109,
/// superseding ROW 107's single-simdgroup design) -- ports
/// `ggml-metal.metal:6500-6600`'s `kernel_mul_mm` GEOMETRY, not just its
/// `simdgroup_float8x8` primitives: [`TILED_GEMM_NSG`] (4) `simdgroup`s
/// cooperate in ONE threadgroup, each owning a
/// `crate::sized::TILED_GEMM_BLOCK_M`/2 x `crate::sized::TILED_GEMM_BLOCK_N`/2
/// sub-tile of the threadgroup's full `BLOCK_M x BLOCK_N` output block (a
/// 2x2 simdgroup grid -- `sgitg & 1` the row half, `sgitg >> 1` the column
/// half, exactly ggml's own split), and the reduction steps by
/// `crate::sized::TILED_GEMM_BLOCK_K` (ggml's `BLOCK_SIZE_K`) rather than by
/// `TILE_DIM` alone: ROW 107's own root cause was pairing ONE simdgroup with
/// an 8-wide K-step, paying two `threadgroup_barrier`s per 8 elements of K
/// (up to 512 barrier round-trips at k=4096) for 64 output elements each --
/// this design pays the same two barriers per `BLOCK_K`(32)-wide step (128
/// round-trips at k=4096, 4x fewer) and each pair now amortizes across
/// `TILED_GEMM_NSG` simdgroups x `BLOCK_K`/`TILE_DIM` K-substeps computing
/// `BLOCK_M x BLOCK_N`(2048) output elements, not 64 -- the "work per
/// barrier" ROW 107's own recommendation named as the actual fix.
///
/// This crate's operand model reads through generic per-axis strides
/// (never assumes row-major-contiguous device memory the way ggml's raw
/// `nb01` byte strides do), so both operand tiles are staged the same way
/// [`push_packed_row_blocked_body`] already reads a strided operand, just
/// written into a fixed `threadgroup` array instead of a private register.
///
/// ROW 113 correction: the weight-tile staging loop itself now decodes with
/// [`push_packed_row_blocked_body`]'s OWN amortized pattern (`q4k_header_for`
/// once per 32-element sub-block, `q4k_run8` batching the nibble extract 8
/// at a time), matching ggml's `dequantize_q4_K` (`ggml-metal.metal:336-352`,
/// which computes `dl`/`ml` once and loops 16 elements). Before this row it
/// called the generic [`operand_read`] (`q4k_element`), which rederives the
/// full header from `device` memory on every element -- correct (cross-token
/// tile reuse via `threadgroup` staging was always real, confirmed by
/// reading the emitted MSL) but roughly 8-40x more device reads and
/// arithmetic per weight element than necessary, which a per-op profiling
/// harness measured as 58.35x slower than decode (ROW 112) even though the
/// tile itself was never re-streamed per token.
///
/// Threadgroup memory is three FIXED-SIZE local arrays declared directly in
/// the kernel body (`weight_tile`: `BLOCK_M * BLOCK_K` `half`; `act_tile`:
/// `BLOCK_N * BLOCK_K` `float`; `out_tile`: `BLOCK_M * BLOCK_N` `float`,
/// reused across `k0` steps but allocated once) -- every dimension is a
/// compile-time constant (`crate::sized::TILED_GEMM_BLOCK_M`/`_N`/`_K`), so
/// this needs no `[[threadgroup(n)]]` kernel parameter and no
/// `setThreadgroupMemoryLength` call on the driver side, unlike ggml's
/// dynamically-sized `shmem` (`ggml-metal.m:3101`): every existing call
/// site in `crate::metal` keeps dispatching through the same
/// `dispatchThreads:threadsPerThreadgroup:` path unchanged, now with
/// [`TILED_GEMM_NSG`] `* SIMD_WIDTH` (128) threads per threadgroup instead
/// of one simdgroup ([`crate::msl::tiled_gemm_threadgroup_width`]).
///
/// Boundary tiles (feature or token extent not a whole multiple of
/// `BLOCK_M`/`BLOCK_N`) are handled by zero-padding out-of-range reads
/// during staging (a true-zero contribution changes nothing) and skipping
/// out-of-range writes entirely during the final scatter -- the same
/// n_rows/n_cols masking `ggml-metal.metal`'s own kernel applies, at
/// `BLOCK_M`/`BLOCK_N` granularity instead of `TILE_DIM`'s. The reduction
/// dimension needs no such mask: [`PackedRowBlock`] already guarantees it
/// is a whole number of [`Q4K_BLOCK_ELEMENTS`] (256) super-blocks, and
/// `build.rs`'s `require_divides_q4k_block` guarantees `BLOCK_K` divides
/// 256 evenly.
///
/// `weight_tile`/`act_tile` are both stored simple row-major (`weight_tile`:
/// feature-row-major, `act_tile`: token-row-major, K fastest in both --
/// UNLIKE ggml's own custom bit-shuffled `sa`/`sb` packing, which exists
/// only so its `simdgroup_load` calls can omit `elements_per_row` and read
/// each fragment pre-packed). `a_frag` reads a `feature x k` fragment
/// straight off `weight_tile`, but `b_frag` reads `act_tile` in its
/// NATURAL `token x k` orientation -- the wrong shape for
/// `simdgroup_multiply_accumulate(acc, a_frag, b_frag, acc)`, which needs
/// its second operand `k x token` for the inner (`k`) dimensions to align.
/// `simdgroup_load`'s `transpose_matrix` flag supplies that without
/// restructuring the staging loop: `b_frag` is loaded with
/// `transpose_matrix = true`, turning the physical `token x k` read into
/// the logical `k x token` fragment the multiply needs. (A first pass
/// without this flag measured `relative=0.497` against the CPU oracle --
/// dimensionally valid MSL, semantically wrong matrix product -- caught by
/// `metal_matmul_on_packed_q4k_weights_matches_the_dequantized_f32_cpu_path_at_tile_scale`.)
#[cfg(feature = "metal-tiled-gemm")]
pub(super) fn push_tiled_gemm_body(
    source: &mut String,
    node: NodeId,
    output_axes: &[u16],
    rank: usize,
    block: &TiledGemmBlock,
    element_type: &str,
) -> Result<(), EmitError> {
    let TiledGemmBlock {
        weight,
        other,
        reduce_dim,
        ref token_axes,
        ref feature_axes,
    } = *block;
    // innermost (fastest, last-listed) of each group -- the single stride
    // the per-element reads below use; see `TiledGemmBlock`'s own doc.
    let Some(&token_axis) = token_axes.last() else {
        return Err(EmitError::EmptyAxisGroup {
            node,
            group: "token",
        });
    };
    let Some(&feature_axis) = feature_axes.last() else {
        return Err(EmitError::EmptyAxisGroup {
            node,
            group: "feature",
        });
    };
    let rank_len = rank.max(1);

    let block_m = crate::sized::TILED_GEMM_BLOCK_M;
    let block_n = crate::sized::TILED_GEMM_BLOCK_N;
    let block_k = crate::sized::TILED_GEMM_BLOCK_K;
    let block_threads = (TILED_GEMM_NSG as u64) * SIMD_WIDTH;
    // 2 row-halves x TILE_DIM(8)-wide simdgroup-matrix fragments per half --
    // ggml's own `THREAD_MAT_M`/`THREAD_MAT_N` (`ggml-metal.metal:6490-6491`).
    let thread_mat_m = block_m / (TILE_DIM as u64 * 2);
    let thread_mat_n = block_n / (TILE_DIM as u64 * 2);
    let mc_count = thread_mat_m * thread_mat_n;
    let sub_k_steps = block_k / TILE_DIM as u64;
    let weight_tile_elems = block_m * block_k;
    let act_tile_elems = block_n * block_k;
    let out_tile_elems = block_m * block_n;

    // `attn_q`/`attn_k`/`attn_v` fold TWO weight-owned axes (`heads`,
    // `head_dim`) into one flattened feature dimension -- `axes_fold_
    // contiguously` already proved the group is one contiguous block, so
    // the runtime extent is the PRODUCT of every axis's own
    // `u.output_extents` entry, read fresh per dispatch the same way a
    // single-axis group already was (the kernel source is reused across
    // concrete shapes; see `TiledGemmBlock`'s own doc). Every real matmul
    // this path has measured keeps `token_axes` a single axis, but the
    // product generalizes to that case for free (one factor, no-op).
    let group_extent_expr = |group: &[u16]| -> Result<String, EmitError> {
        let mut terms = Vec::with_capacity(group.len());
        for &dim in group {
            let Some(index) = output_axes.iter().position(|&candidate| candidate == dim) else {
                return Err(EmitError::AxisNotInOutputAxes { node, axis: dim });
            };
            terms.push(format!("u.output_extents[{index}]"));
        }
        Ok(terms.join(" * "))
    };

    source.push_str(&format!(
        "    long feature_extent = {};\n",
        group_extent_expr(feature_axes)?
    ));
    source.push_str(&format!(
        "    long token_extent = {};\n",
        group_extent_expr(token_axes)?
    ));
    source.push_str(&format!(
        "    long num_col_tiles = (token_extent + {}) / {block_n};\n",
        block_n - 1
    ));
    source.push_str(&format!("    long tiitg = (long)gid % {block_threads};\n"));
    source.push_str(&format!("    long sgitg = tiitg / {SIMD_WIDTH};\n"));
    source.push_str(&format!(
        "    long tile_index = (long)gid / {block_threads};\n"
    ));
    source.push_str("    long row_tile = tile_index / num_col_tiles;\n");
    source.push_str("    long col_tile = tile_index % num_col_tiles;\n");
    source.push_str("    long row_half = sgitg & 1;\n");
    source.push_str("    long col_half = sgitg >> 1;\n");
    source.push_str(&format!(
        "    threadgroup half weight_tile[{weight_tile_elems}];\n"
    ));
    source.push_str(&format!(
        "    threadgroup float act_tile[{act_tile_elems}];\n"
    ));
    source.push_str(&format!("    simdgroup_float8x8 acc[{mc_count}];\n"));
    source.push_str(&format!(
        "    for (int i = 0; i < {mc_count}; ++i) {{ acc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f); }}\n"
    ));
    source.push_str(&format!(
        "    for (long k0 = 0; k0 < u.reduction_total; k0 += {block_k}) {{\n"
    ));
    // ROW 113: weight staging amortizes the Q4_K sub-block header the same
    // way `push_packed_row_blocked_body` and ggml's own `dequantize_q4_K`
    // (ggml-metal.metal:336-352) both do -- one `q4k_header_for` per
    // 32-element sub-block, `q4k_run8` batching the nibble extract 8 at a
    // time -- instead of `operand_read`'s generic `q4k_element`, which
    // rederives the header (two `device` header reads plus the 6-bit
    // scale/min unpack) from scratch on every one of the tile's individual
    // elements. Staged by ROW rather than by flat index: `block_threads`
    // (128) exceeds `block_m` (64) with the default sizing, so the first
    // `block_m` threads each own exactly one row of the tile for this phase
    // and the rest do no extra weight work (`act_tile`'s own load below
    // still uses every thread).
    source.push_str(&format!(
        "        for (long w_row = tiitg; w_row < {block_m}; w_row += {block_threads}) {{\n"
    ));
    source.push_str(&format!(
        "            long w_feat = row_tile * {block_m} + w_row;\n"
    ));
    source.push_str("            if (w_feat < feature_extent) {\n");
    source.push_str(&format!(
        "                long row_base = u.operand_base[{weight}] + w_feat * u.operand_strides[{weight}][{feature_axis}] + k0 * u.operand_strides[{weight}][{reduce_dim}];\n"
    ));
    // `block_k` divides 256 (`Q4K_BLOCK_ELEMENTS`, `build.rs`'s
    // `require_divides_q4k_block`) and is a multiple of 8 (`build.rs`'s own
    // `require_multiple_of_eight`, added alongside this row), so it is
    // always either <= the Q4_K sub-block width (32) or a whole multiple of
    // it -- `chunk_width` picks the smaller, `num_chunks` covers `block_k`
    // exactly with no ragged remainder either way.
    let q4k_subblock_width: u64 = (Q4K_BLOCK_ELEMENTS / 8) as u64;
    let chunk_width = q4k_subblock_width.min(block_k);
    let num_chunks = block_k.div_ceil(chunk_width);
    for chunk_index in 0..num_chunks {
        let chunk_offset = chunk_index * chunk_width;
        source.push_str("                {\n");
        source.push_str(&format!(
            "                    long slot_off = row_base + {chunk_offset};\n"
        ));
        source.push_str(&format!(
            "                    device const uchar *blk = in{weight} + (slot_off / {Q4K_BLOCK_ELEMENTS}) * {Q4K_BLOCK_BYTES};\n"
        ));
        source.push_str(&format!(
            "                    uint slot = (uint)(slot_off % {Q4K_BLOCK_ELEMENTS});\n"
        ));
        source.push_str("                    q4k_header hdr = q4k_header_for(blk, slot);\n");
        let runs = chunk_width / 8;
        for run_index in 0..runs {
            let run_offset = run_index * 8;
            source.push_str("                    {\n");
            source.push_str("                        float levels[8];\n");
            source.push_str(&format!(
                "                        q4k_run8(blk, slot + {run_offset}u, levels);\n"
            ));
            source.push_str(&format!(
                "                        for (int j = 0; j < 8; ++j) {{ weight_tile[w_row * {block_k} + {chunk_offset} + {run_offset} + j] = (half)(hdr.scale * levels[j] - hdr.minimum); }}\n"
            ));
            source.push_str("                    }\n");
        }
        source.push_str("                }\n");
    }
    source.push_str("            } else {\n");
    source.push_str(&format!(
        "                for (long fill_k = 0; fill_k < {block_k}; ++fill_k) {{ weight_tile[w_row * {block_k} + fill_k] = 0.0h; }}\n"
    ));
    source.push_str("            }\n");
    source.push_str("        }\n");
    source.push_str(&format!(
        "        for (long idx = tiitg; idx < {act_tile_elems}; idx += {block_threads}) {{\n"
    ));
    source.push_str(&format!("            long a_col = idx / {block_k};\n"));
    source.push_str(&format!("            long a_k = idx % {block_k};\n"));
    source.push_str(&format!(
        "            long a_tok = col_tile * {block_n} + a_col;\n"
    ));
    source.push_str("            long a_k_global = k0 + a_k;\n");
    source.push_str("            float a_value = 0.0f;\n");
    source.push_str("            if (a_tok < token_extent) {\n");
    source.push_str(&format!(
        "                long aoff = u.operand_base[{other}] + a_tok * u.operand_strides[{other}][{token_axis}] + a_k_global * u.operand_strides[{other}][{reduce_dim}];\n"
    ));
    source.push_str(&format!(
        "                a_value = {};\n",
        operand_read(other, "aoff", None)
    ));
    source.push_str("            }\n");
    source.push_str(&format!(
        "            act_tile[a_col * {block_k} + a_k] = a_value;\n"
    ));
    source.push_str("        }\n");
    source.push_str("        threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    source.push_str(&format!(
        "        for (int sub_k = 0; sub_k < {sub_k_steps}; ++sub_k) {{\n"
    ));
    source.push_str(&format!(
        "            simdgroup_half8x8 a_frag[{thread_mat_m}];\n"
    ));
    source.push_str(&format!(
        "            for (int i = 0; i < {thread_mat_m}; ++i) {{\n"
    ));
    source.push_str(&format!(
        "                simdgroup_load(a_frag[i], weight_tile + (row_half * {thread_mat_m} + i) * 8 * {block_k} + sub_k * 8, {block_k});\n"
    ));
    source.push_str("            }\n");
    source.push_str("            simdgroup_barrier(mem_flags::mem_none);\n");
    source.push_str(&format!(
        "            simdgroup_float8x8 b_frag[{thread_mat_n}];\n"
    ));
    source.push_str(&format!(
        "            for (int j = 0; j < {thread_mat_n}; ++j) {{\n"
    ));
    source.push_str(&format!(
        "                simdgroup_load(b_frag[j], act_tile + (col_half * {thread_mat_n} + j) * 8 * {block_k} + sub_k * 8, {block_k}, ulong2(0), true);\n"
    ));
    source.push_str("            }\n");
    source.push_str(&format!(
        "            for (int i = 0; i < {thread_mat_m}; ++i) {{\n"
    ));
    source.push_str(&format!(
        "                for (int j = 0; j < {thread_mat_n}; ++j) {{\n"
    ));
    source.push_str(&format!(
        "                    simdgroup_multiply_accumulate(acc[i * {thread_mat_n} + j], a_frag[i], b_frag[j], acc[i * {thread_mat_n} + j]);\n"
    ));
    source.push_str("                }\n");
    source.push_str("            }\n");
    source.push_str("        }\n");
    source.push_str("        threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    source.push_str("    }\n");
    source.push_str(&format!(
        "    threadgroup float out_tile[{out_tile_elems}];\n"
    ));
    source.push_str(&format!(
        "    for (int i = 0; i < {thread_mat_m}; ++i) {{\n"
    ));
    source.push_str(&format!(
        "        for (int j = 0; j < {thread_mat_n}; ++j) {{\n"
    ));
    source.push_str(&format!(
        "            simdgroup_store(acc[i * {thread_mat_n} + j], out_tile + (row_half * {thread_mat_m} + i) * 8 * {block_n} + (col_half * {thread_mat_n} + j) * 8, {block_n});\n"
    ));
    source.push_str("        }\n");
    source.push_str("    }\n");
    source.push_str("    threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    source.push_str(&format!(
        "    for (long idx = tiitg; idx < {out_tile_elems}; idx += {block_threads}) {{\n"
    ));
    source.push_str(&format!("        long o_row = idx / {block_n};\n"));
    source.push_str(&format!("        long o_col = idx % {block_n};\n"));
    source.push_str(&format!(
        "        long o_feat = row_tile * {block_m} + o_row;\n"
    ));
    source.push_str(&format!(
        "        long o_tok = col_tile * {block_n} + o_col;\n"
    ));
    source.push_str("        if (o_feat < feature_extent && o_tok < token_extent) {\n");
    source.push_str(&format!("            long coord[{rank_len}];\n"));
    source.push_str(&format!(
        "            for (int d = 0; d < {rank}; ++d) {{ coord[d] = 0; }}\n"
    ));
    source.push_str(&format!("            coord[{feature_axis}] = o_feat;\n"));
    source.push_str(&format!("            coord[{token_axis}] = o_tok;\n"));
    source.push_str("            long out_offset = u.out_base;\n");
    for dim in 0..rank {
        source.push_str(&format!(
            "            out_offset += coord[{dim}] * u.out_strides[{dim}];\n"
        ));
    }
    source.push_str(&format!(
        "            out[out_offset] = ({element_type})out_tile[idx];\n"
    ));
    source.push_str("        }\n");
    source.push_str("    }\n");
    Ok(())
}

/// Never actually invoked: [`classify_tiled_gemm`]'s own `#[cfg(not(feature
/// = "metal-tiled-gemm"))]` arm always returns `None`, so no caller ever
/// holds a `&TiledGemmBlock` to pass here without the feature -- this stub
/// exists only so [`push_cooperative_reduce_body`]'s `if let Some(block) =
/// tiled_gemm_block(...)` arm still type-checks in that build.
#[cfg(not(feature = "metal-tiled-gemm"))]
pub(super) fn push_tiled_gemm_body(
    source: &mut String,
    node: NodeId,
    output_axes: &[u16],
    rank: usize,
    block: &TiledGemmBlock,
    element_type: &str,
) -> Result<(), EmitError> {
    let _ = (source, output_axes, rank, block, element_type);
    Err(EmitError::TiledGemmFeatureDisabled { node })
}

/// The threadgroup width [`emit`]/[`kernel_dispatch_shape`] must dispatch
/// with -- [`TILED_GEMM_NSG`]` * SIMD_WIDTH` (128) when `resolved` takes
/// [`push_tiled_gemm_body`]'s multi-simdgroup path (its coordinate math
/// depends on exactly this many threads per threadgroup, the same
/// correctness requirement `crate::metal::dispatch`'s own doc states for
/// `SIMD_WIDTH`), `SIMD_WIDTH * split` for the row-blocked packed path when a
/// `Reduce`'s own shape carries `reduce_op`/`init`/`output_axes` (see
/// [`packed_row_dispatch`] -- `split` is `1`, i.e. plain `SIMD_WIDTH`, unless
/// `metal-q4k-split-k` is active and this shape is below the target
/// simdgroup count), [`crate::sized::PACKED_ROW_BLOCK_SIMDGROUPS`] *
/// `SIMD_WIDTH` for a packed row-block reached outside that match (widens the
/// packed row-block arm's threadgroup beyond one simdgroup — see that
/// constant's doc for why this is dispatch-only and never touches the kernel
/// body), [`cooperative_reduce_width`] for every other cooperative-reduce
/// kernel, `None` otherwise. Single source of truth both dispatch-shape
/// functions read, so they cannot drift the way two independent copies of
/// this `if`/`else` could. Ordered after the tiled-GEMM check and before the
/// generic cooperative-reduce fallback, matching [`grid_threads`]' own
/// priority (the two paths are mutually exclusive by construction —
/// [`kernel_cache_key`]'s doc).
pub(super) fn tiled_gemm_threadgroup_width(
    resolved: &BoundOp,
    quantized: &[Option<Codec>],
    numeric_policy: NumericPolicy,
) -> Option<u64> {
    // `round_zero_reduce_bound`'s own doc: a round-batched fold's per-round
    // dispatch geometry is whatever the round-0 `Reduce` this collapse
    // replaced would use -- delegating keeps this in lock-step with
    // `grid_threads`'s own identical delegation and with
    // `render_reduce(round_zero, ..)`'s actual rendered body.
    #[cfg(feature = "metal-moe-mul-mat-id")]
    if let BoundOpKind::RoundBatchedReduce { .. } = &resolved.kind {
        let round_zero = round_zero_reduce_bound(resolved);
        return tiled_gemm_threadgroup_width(&round_zero, quantized, numeric_policy);
    }
    // `head_v_dim` threads per threadgroup -- correctness-load-bearing, not
    // an occupancy hint: `grid_threads`' own `GatedDeltaNet` arm dispatches
    // `num_v_heads * head_v_dim` threads total, and this width is what makes
    // `dispatchThreads_threadsPerThreadgroup`'s linear grouping land every
    // one of `head_v_dim` value rows for a head in the SAME threadgroup as
    // that head's own `vh` (`render_gated_delta_net`'s own doc).
    if let BoundOpKind::GatedDeltaNet { head_v_dim, .. } = &resolved.kind {
        return Some(*head_v_dim);
    }
    // `query_groups * SIMD_WIDTH` threads per threadgroup -- one simdgroup
    // per query head sharing this kv_head, cooperatively loading that
    // kv_head's K/V row once per key into `threadgroup` memory instead of
    // each of the `query_groups` simdgroups re-reading it from device memory
    // (`render_cached_attention`'s own doc). Correctness-load-bearing, not an
    // occupancy hint: the body's `tid`/`group_width` split assumes exactly
    // this many threads land in the same threadgroup. Below
    // `cached_attention_per_query_head_grid`'s own knee, `query_groups` moves
    // out of this width entirely -- one query head per threadgroup, decoded
    // from `tgid` instead of shared threadgroup memory -- so the width there
    // is `chunks * SIMD_WIDTH` alone.
    // Part D: must agree with `grid_threads`'s own two_pass arm exactly --
    // one threadgroup per `(query_row, kv_head)`, `query_groups` simdgroups
    // wide, no chunk/split widening (see that arm's own doc).
    if let BoundOpKind::CachedAttention {
        query_groups,
        two_pass: true,
        head_dim,
        cached_key_rows,
        ..
    } = &resolved.kind
    {
        return Some(two_pass_physical_threadgroup_width(*query_groups, *head_dim, *cached_key_rows));
    }
    if let BoundOpKind::CachedAttention {
        query_groups,
        head_dim,
        cached_key_rows,
        new_key_rows,
        ..
    } = &resolved.kind
    {
        // Must agree with `grid_threads`'s own `CachedAttention` arm: the
        // single-range fused (dynamic) path always dispatches the shape-
        // bounded compiled MAXIMUM chunk count (`effective_context_chunk_
        // cap`), so the threadgroup width has to widen to match -- a
        // mismatch here puts fewer threads in the threadgroup than
        // `local_group_index`'s own `cap`-sized addressing assumes, which is
        // an out-of-bounds `threadgroup` memory write, not merely a wrong
        // answer.
        // Same `cached_key_rows == 0` discriminator as `grid_threads`'s own
        // `CachedAttention` arm -- `two_range_cached_bound` never widens to
        // the compiled cap, since its `context_length` is already the
        // bucket-padded compile-time value.
        let dynamic_cached_len = (resolved.operands().len() == 9
            || resolved.operands().len() == 12)
            && *cached_key_rows == 0;
        let context_length = *cached_key_rows + *new_key_rows;
        let chunks = if dynamic_cached_len {
            effective_context_chunk_cap(*query_groups, *head_dim)
        } else {
            context_chunks_for(context_length, *query_groups, *head_dim, numeric_policy)
        };
        // Below the split-at-scale knee, `cached_attention_per_query_head_grid`
        // moves `query_groups` out of this width and into a threadgroup-count
        // factor instead (`grid_threads`' own total stays unchanged -- see
        // that function's doc) -- one query head per threadgroup rather than
        // `query_groups` of them sharing one.
        let width = if cached_attention_per_query_head_grid(dynamic_cached_len, context_length) {
            chunks * SIMD_WIDTH
        } else {
            *query_groups * chunks * SIMD_WIDTH
        };
        return Some(width);
    }
    if let BoundOpKind::Reduce {
        keep: Keep::Reduce,
        reduce_op,
        init,
        output_axes,
        ..
    } = &resolved.kind
    {
        if tiled_gemm_block(resolved, quantized, *reduce_op, *init, output_axes).is_some() {
            return Some((TILED_GEMM_NSG as u64) * SIMD_WIDTH);
        }
        if let Some(block) = packed_row_block(resolved, quantized) {
            let feature_total: u64 = block
                .feature_axes
                .iter()
                .map(|&axis| resolved.extents[axis as usize])
                .product();
            let token_total = packed_row_block_token_total(&block, &resolved.extents);
            let (_base, split) = packed_row_dispatch(feature_total, token_total, block.codec);
            // `metal-packed-row-nsg2`'s own `nsg=2` geometry (ggml's
            // `N_SG_Q4_K`, `ggml-metal-impl.h:33`) has to be applied HERE,
            // not in the `#[cfg(feature = "metal-packed-row-nsg2")]` arm
            // below -- this `if let` block's own `packed_row_block` check
            // returns unconditionally whenever it matches, so that arm below
            // is unreachable dead code for `Keep::Reduce` ops (every
            // `packed_row_block` match IS a `Keep::Reduce` op by
            // construction -- see `PackedRowBlock`'s own classification).
            // `metal-q4k-ggml-port` needs the identical nsg=2 width (its own
            // kernel body is ggml's, dispatched at ggml's own `N_SG_Q4_K`) --
            // [`packed_row_nsg_factor`] is the one place both features widen
            // from, so they cannot drift into two competing nsg constants.
            // `!metal-q4k-split-k` too: split-K's own combine (`push_packed_
            // row_combine_and_write`'s split-K arm) already picks a
            // cooperating `split` simdgroup count for a REAL reason -- a
            // starved shape's simdgroups share one output group via
            // `sgitg`/`threadgroup` memory/a barrier -- and doubling the
            // dispatched width again on top of that here, unconditionally,
            // would desync the combine's own `split` from the width the
            // driver actually dispatches (confirmed: `--all-features`,
            // ggml-port + split-K together, broke Q4_K parity outright,
            // relative=1). Every row-blocked body variant addresses its
            // output group purely from `gid / SIMD_WIDTH`
            // (`metal-packed-row-nsg2`'s own doc, still true here), so nsg=2
            // is correctness-neutral whenever split-K is off, regardless of
            // which body actually runs.
            return Some(SIMD_WIDTH * split * packed_row_nsg_factor());
        }
    }
    if packed_row_block(resolved, quantized).is_some() {
        return Some(crate::sized::PACKED_ROW_BLOCK_SIMDGROUPS * SIMD_WIDTH);
    }
    if !reduce_is_cooperative(resolved) {
        return None;
    }
    let BoundOpKind::Reduce {
        keep: Keep::Reduce,
        output_axes,
        ..
    } = &resolved.kind
    else {
        return None;
    };
    let reduce_dims = reduction_dims(resolved, output_axes);
    Some(cooperative_reduce_width(resolved, quantized, &reduce_dims))
}

/// Whether `resolved` takes [`push_cooperative_reduce_body`]'s "SUPER-BLOCK
/// TILED PACKED READ" arm -- exactly one Q4_K-packed operand, contiguous
/// along the single reduction dim, whose extent is a whole number of
/// super-blocks. That arm's lane math (`Q4K_BLOCK_ELEMENTS / SIMD_WIDTH`
/// contiguous elements per lane, `slot = lane * run`) is fixed to
/// `SIMD_WIDTH` lanes by construction -- widening the dispatch would push
/// `slot` past the super-block it is meant to stay inside. Extracted so
/// [`cooperative_reduce_width`] and the body can never disagree on which
/// shape a given op takes (mirrors [`tiled_gemm_threadgroup_width`]'s own
/// "single source of truth" doc).
pub(super) fn q4k_super_block_tiled(
    resolved: &BoundOp,
    quantized: &[Option<Codec>],
    reduce_dims: &[u16],
) -> bool {
    // Mixed expert sources have per-entry codecs and compact payload bases;
    // this Q4-only specialization cannot represent that ABI.
    if std::env::var_os("PROXIMA_ENABLE_UNSAFE_METAL_EXPERT_SOURCES").is_some() {
        return false;
    }
    if reduce_dims.len() != 1 {
        return false;
    }
    let reduce_dim = reduce_dims[0];
    let packed: Vec<usize> = quantized
        .iter()
        .enumerate()
        .filter_map(|(index, codec)| matches!(codec, Some(Codec::Q4K)).then_some(index))
        .collect();
    packed.len() == 1
        && resolved.operands()[packed[0]].1.stride(reduce_dim) == 1
        && (resolved.extents[reduce_dim as usize] as usize).is_multiple_of(Q4K_BLOCK_ELEMENTS)
}

/// Cooperative-reduce threadgroup width -- `SIMD_WIDTH` (32) with this
/// feature off, matching the byte-identical prior behaviour every existing
/// gate baselines against. With `metal-wide-cooperative-reduce` on, scales
/// with the reduction extent instead of pinning every cooperative reduce to
/// one simdgroup regardless of size (the measured defect:
/// `docs/discipline.md`'s row for this initiative -- a 4096-element
/// RMS-norm sum launches 32 threads and each lane loops 128 times
/// serially). `reduction_total / 4` rounded up to the next multiple of
/// `SIMD_WIDTH`, clamped to `[SIMD_WIDTH,
/// WIDE_COOPERATIVE_REDUCE_MAX_WIDTH]` -- four elements of serial work per
/// lane keeps a short reduction (64, 128) from over-launching (more
/// threadgroup-barrier / partial-fold overhead than the serial work it
/// removes) while a long one (4096+) saturates the cap. Never applied to
/// [`q4k_super_block_tiled`]'s arm: that lane math is fixed to `SIMD_WIDTH`
/// by construction, not a policy choice this scaling could touch.
///
/// `WIDE_COOPERATIVE_REDUCE_MAX_WIDTH` is a build-time-configured cap
/// (`omega-runtime.toml`'s `[wide_cooperative_reduce]` section,
/// `crate::sized`), NOT a query of the device's real
/// `maxTotalThreadsPerThreadgroup` -- emission has no device handle
/// (`crate::sized::SIMD_WIDTH`'s own doc states the same constraint for the
/// hardware-fixed 32). `crate::metal::dispatch` already clamps
/// `grid.threadgroup_width` to the pipeline's real cap before dispatching,
/// so an emit-time cap above the true hardware limit is a wasted grid, not
/// a correctness hazard -- 256 (8 simdgroups) is conservative against every
/// Apple GPU family this crate targets.
/// The pure numeric core of [`cooperative_reduce_width`] -- taking
/// `reduction_total` directly rather than reading it off a `BoundOp`'s
/// `extents`, so a caller with no real reduce `BoundOp` to point at (the
/// two-pass attention kernel's own K-dot and row-reduce folds, which bake
/// every shape constant at emit time instead of packing a `Uniforms::
/// reduction_total` field -- `cached_attention_two_pass.rs`'s own
/// `two_pass_threadgroup_width`) computes the SAME width production's real
/// per-node reduce kernels do, so the two can never drift onto different
/// topologies for the same `reduction_total`. `cooperative_reduce_width`
/// itself calls straight through to this after resolving its own
/// `reduction_total` from `reduce_dims`.
#[cfg(feature = "metal-wide-cooperative-reduce")]
pub(super) fn wide_cooperative_reduce_width(reduction_total: u64) -> u64 {
    let quarter = reduction_total.div_ceil(4).max(1);
    quarter
        .next_multiple_of(SIMD_WIDTH)
        .clamp(SIMD_WIDTH, crate::sized::WIDE_COOPERATIVE_REDUCE_MAX_WIDTH)
}

/// Feature off: always `SIMD_WIDTH`, matching [`cooperative_reduce_width`]'s
/// own feature-off arm -- see that function's doc.
#[cfg(not(feature = "metal-wide-cooperative-reduce"))]
pub(super) fn wide_cooperative_reduce_width(_reduction_total: u64) -> u64 {
    SIMD_WIDTH
}

#[cfg(feature = "metal-wide-cooperative-reduce")]
pub(super) fn cooperative_reduce_width(
    resolved: &BoundOp,
    quantized: &[Option<Codec>],
    reduce_dims: &[u16],
) -> u64 {
    if q4k_super_block_tiled(resolved, quantized, reduce_dims) {
        return SIMD_WIDTH;
    }
    let reduction_total: u64 = reduce_dims
        .iter()
        .map(|&dim| resolved.extents[dim as usize])
        .product();
    wide_cooperative_reduce_width(reduction_total)
}

/// Feature off: always `SIMD_WIDTH`, the byte-identical prior dispatch
/// shape -- see [`cooperative_reduce_width`]'s doc (the `metal-wide-
/// cooperative-reduce` arm) for the scaling this default-off build never
/// takes.
#[cfg(not(feature = "metal-wide-cooperative-reduce"))]
pub(super) fn cooperative_reduce_width(
    _resolved: &BoundOp,
    _quantized: &[Option<Codec>],
    _reduce_dims: &[u16],
) -> u64 {
    SIMD_WIDTH
}

/// [`tiled_gemm_threadgroup_width`]'s own nsg multiplier for the packed
/// row-blocked path -- `PACKED_ROW_NSG` with either nsg2 feature on and
/// `metal-q4k-split-k` off (see that call site's own doc for why split-K
/// must win when both are compiled in), `1` otherwise. Two functions, not a
/// `cfg!()` branch inline, so a feature-off build never references
/// `PACKED_ROW_NSG` from code it does not generate (mirrors
/// [`packed_row_split_factor`]'s own on/off pair).
#[cfg(all(
    any(
        feature = "metal-packed-row-nsg2",
        feature = "metal-q4k-ggml-port",
        feature = "metal-q4_0-native"
    ),
    not(feature = "metal-q4k-split-k")
))]
pub(super) fn packed_row_nsg_factor() -> u64 {
    PACKED_ROW_NSG as u64
}

/// The nsg2-features-off (or `metal-q4k-split-k`-on) arm: nsg widening never
/// engages, so the factor is always `1` -- see [`packed_row_nsg_factor`]'s
/// feature-on twin for the real policy.
#[cfg(not(all(
    any(
        feature = "metal-packed-row-nsg2",
        feature = "metal-q4k-ggml-port",
        feature = "metal-q4_0-native"
    ),
    not(feature = "metal-q4k-split-k")
)))]
pub(super) fn packed_row_nsg_factor() -> u64 {
    1
}

// the emitter threads a bound op's full shape (rank, axes, reduce op, init,
// element type, codec flags) into one kernel body; splitting that into a
// struct would relocate the arguments, not remove them.
#[allow(clippy::too_many_arguments)]
pub(super) fn push_cooperative_reduce_body(
    source: &mut String,
    resolved: &BoundOp,
    reduce_op: ScalarOp,
    init: ReduceInit,
    output_axes: &[u16],
    reduce_dims: &[u16],
    rank: usize,
    quantized: &[Option<Codec>],
    element_type: &str,
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
    is_broadcast_epilogue: bool,
    expert_source_mode: bool,
) -> Result<(), EmitError> {
    let rank_len = rank.max(1);
    let output_rank = output_axes.len();
    let output_rank_len = output_rank.max(1);
    let reduce_rank = reduce_dims.len();
    let reduce_rank_len = reduce_rank.max(1);
    let operand_count = resolved.operands().len();
    let gather_slots = gather_slots(resolved);

    // the tiled GEMM path owns its own preamble entirely (`tiitg`/`sgitg`/
    // `tile_index`, derived straight from `gid` against `TILED_GEMM_NSG *
    // SIMD_WIDTH` threads per threadgroup, ROW 109) -- it needs neither
    // `output_index` nor `lane` the way the row-blocked path below does, see
    // `kernel_cache_key`'s own comment for why the two are mutually
    // exclusive by construction.
    if let Some(block) = tiled_gemm_block(resolved, quantized, reduce_op, init, output_axes) {
        push_tiled_gemm_body(
            source,
            resolved.node,
            output_axes,
            rank,
            &block,
            element_type,
        )?;
        return Ok(());
    }

    // the row-blocked packed path owns its own preamble: `output_index` is a
    // GROUP index there, not an output index, so the guard below would be
    // wrong for it. It always dispatches at SIMD_WIDTH regardless of
    // `metal-wide-cooperative-reduce` -- ROW-BLOCKED is a separate,
    // untouched investigation (see this file's own history), not the
    // reduction-extent-driven scaling below. Covers both the plain
    // single-activation-row shape and the multi-row fold
    // [`push_packed_row_blocked_body`]'s own `token_total > 1` branch emits
    // -- one preamble, since both branches dispatch the identical
    // `output_index`/`lane` pair at `SIMD_WIDTH` (`metal-q4k-split-k`'s own
    // `tptg`-derived preamble below is the only variant on this pair).
    if let Some(block) = packed_row_block(resolved, quantized) {
        if cfg!(feature = "metal-q4k-split-k") {
            // `tptg` is the ACTUAL per-dispatch threadgroup width
            // (`kernel_signature`'s new param, wired on for this exact
            // path -- see its call site's own gate). `split == 1` (the
            // feature-off-equivalent case) makes every line below collapse
            // to the plain `output_index`/`lane` pair the non-split-K arm
            // emits: `tptg_width == SIMD_WIDTH`, so `tiitg == gid % SIMD_WIDTH`
            // (today's `lane`), `sgitg == 0`, and `lane == tiitg` -- the same
            // value, same bits, same order.
            source.push_str("    uint tptg_width = tptg;\n");
            source.push_str("    long output_index = (long)gid / (long)tptg_width;\n");
            source.push_str("    uint tiitg = (uint)((long)gid % (long)tptg_width);\n");
            source.push_str(&format!("    uint sgitg = tiitg / {SIMD_WIDTH}u;\n"));
            source.push_str(&format!("    uint lane = tiitg % {SIMD_WIDTH}u;\n"));
            source.push_str(&format!("    uint split = tptg_width / {SIMD_WIDTH}u;\n"));
        } else {
            source.push_str(&format!(
                "    long output_index = (long)gid / {SIMD_WIDTH};\n"
            ));
            source.push_str(&format!("    uint lane = gid % {SIMD_WIDTH}u;\n"));
        }
        push_packed_row_blocked_body(
            source,
            resolved,
            reduce_op,
            init,
            output_axes,
            rank,
            quantized,
            element_type,
            &block,
            epilogue_body,
            epilogue_operands,
            expert_source_mode,
        )?;
        return Ok(());
    }

    // Single source of truth with the dispatch shape `grid_threads`/
    // `tiled_gemm_threadgroup_width` compute -- `width` here MUST equal
    // `cooperative_reduce_width`'s return for this exact op, or the grid
    // launched and the lane math emitted below disagree.
    let width = cooperative_reduce_width(resolved, quantized, reduce_dims);
    source.push_str(&format!("    long output_index = (long)gid / {width};\n"));
    source.push_str("    if (output_index >= u.output_total) { return; }\n");
    source.push_str(&format!("    uint lane = gid % {width}u;\n"));

    source.push_str(&format!("    long full_coord[{rank_len}];\n"));
    for dim in 0..rank {
        source.push_str(&format!("    full_coord[{dim}] = 0;\n"));
    }

    if output_rank > 0 {
        source.push_str(&format!("    long output_coord[{output_rank_len}];\n"));
        source.push_str("    long remaining = output_index;\n");
        for index in (0..output_rank).rev() {
            source.push_str(&format!(
                "    output_coord[{index}] = remaining % u.output_extents[{index}]; \
                 remaining /= u.output_extents[{index}];\n"
            ));
        }
        for (index, dim) in output_axes.iter().enumerate() {
            source.push_str(&format!("    full_coord[{dim}] = output_coord[{index}];\n"));
        }
    }

    let (init_expr, seeded_init) = fold_init_tokens(init);
    let identity = cooperative_identity_token(resolved.node, reduce_op)?;
    source.push_str(&format!("    {element_type} accumulator;\n"));
    source.push_str("    bool seeded;\n");
    source.push_str("    if (lane == 0u) {\n");
    source.push_str(&format!("        accumulator = {init_expr};\n"));
    source.push_str(&format!("        seeded = {seeded_init};\n"));
    source.push_str("    } else {\n");
    source.push_str(&format!("        accumulator = {identity};\n"));
    source.push_str("        seeded = true;\n");
    source.push_str("    }\n");

    // A SINGLE reduction dim is the shape every matmul takes, and it makes
    // the whole per-element index computation redundant. `r` already IS the
    // reduction coordinate (`r < reduction_total == reduction_extents[0]`),
    // so the unflatten is an identity; and every operand's offset then
    // advances by a CONSTANT stride per step, so the base can be hoisted and
    // the step folded into one add.
    //
    // What the general path below costs per element, measured on the emitted
    // MSL: a 64-bit integer `%` and `/` against a runtime extent (Apple GPUs
    // have no integer divider — that is an emulated multi-instruction
    // sequence), a write into a thread-local `long` array, and `rank`
    // 64-bit multiply-adds per operand. For a 4096x4096 matvec that is all
    // of it: the probe measured 1.6 GB/s against llama.cpp Metal's 214.7.
    if reduce_rank == 1 {
        let reduce_dim = reduce_dims[0] as usize;
        // SUPER-BLOCK TILED PACKED READ. `q4k_element` derives `d`, `dmin` and
        // the 6-bit scale/min per ELEMENT, but all three are constant across a
        // 32-element sub-block, so the strided walk above pays that decode 256
        // times per super-block. Measured: packed marginal 12.3 GB/s = 21.9 G
        // elem/s against llama.cpp Metal's 381 G elem/s, while the f32 kernel on
        // the SAME loop hits 60.5 G elem/s reading 7.1x more bytes — Q4 was
        // compute-bound, not bandwidth-bound (`docs/discipline.md` ROW 72).
        //
        // Giving each lane a CONTIGUOUS run of `Q4K_BLOCK_ELEMENTS / SIMD_WIDTH`
        // elements keeps that run inside one sub-block (lane*8 .. lane*8+7 never
        // crosses a 32 boundary), so the header decodes once per run. Same shape
        // as ggml's `for (short i = 0; i < 8; ++i)`.
        //
        // Requires: exactly one packed operand, contiguous along the reduction
        // dim, and a reduction extent that is a whole number of super-blocks —
        // all known here, from the bound layout, not at runtime.
        // Q4_K-only: the body below calls `q4k_header_for`/`q4k_value` by name,
        // so this fallback requires the packed operand specifically to be that
        // codec — a `Q6_K` operand that somehow reaches here (it never does in
        // practice: `packed_row_block` above already claims every real
        // `Q6_K` matmul this repo's checkpoint carries) falls through to the
        // fully generic scalar path below instead of emitting the wrong codec's
        // unpack call.
        let packed: Vec<usize> = quantized
            .iter()
            .enumerate()
            .filter_map(|(index, codec)| matches!(codec, Some(Codec::Q4K)).then_some(index))
            .collect();
        let run = Q4K_BLOCK_ELEMENTS / SIMD_WIDTH as usize;
        // The broadcast-write tail (`push_cooperative_reduce_tail`'s own doc)
        // only ever walks the plain per-lane strided loop below -- the Q4K
        // super-block-tiled specialization is Q4K-only (packed weight
        // operand), never the shape a broadcast-reduce epilogue's f32
        // activation fold takes, and `render_reduce`'s own gate already
        // rejects a packed-row-block match, so this stays `false` in
        // practice; forced off here rather than relied upon so a future
        // packed operand slipping past that gate still falls through to the
        // supported loop instead of silently skipping the epilogue write.
        let tiled =
            !is_broadcast_epilogue && q4k_super_block_tiled(resolved, quantized, reduce_dims);
        if tiled {
            let weight = packed[0];
            for (index, gather_slot) in gather_slots.iter().copied().enumerate().take(operand_count) {
                source.push_str(&format!(
                    "    long base{index} = u.operand_base[{index}];\n"
                ));
                for dim in 0..rank {
                    if dim == reduce_dim {
                        continue;
                    }
                    source.push_str(&format!(
                    "    base{index} += full_coord[{dim}] * u.operand_strides[{index}][{dim}];\n"
                ));
                }
                if index != weight {
                    source.push_str(&format!(
                        "    long stride{index} = u.operand_strides[{index}][{reduce_dim}];\n"
                    ));
                }
                if let Some(slot) = gather_slot {
                    push_cooperative_gather_fetch(
                        source,
                        index,
                        slot,
                        rank,
                        "full_coord",
                        &format!("base{index}"),
                    );
                }
            }
            source.push_str(&format!("    uint slot = (uint)lane * {run}u;\n"));
            source.push_str(&format!(
            "    for (int block_start = 0; block_start < (int)u.reduction_total; block_start += {Q4K_BLOCK_ELEMENTS}) {{\n"
        ));
            source.push_str(&format!(
            "        device const uchar *blk = in{weight} + (((int)base{weight} + block_start) / {Q4K_BLOCK_ELEMENTS}) * {Q4K_BLOCK_BYTES};\n"
        ));
            source.push_str("        q4k_header hdr = q4k_header_for(blk, slot);\n");
            source.push_str(&format!("        for (int j = 0; j < {run}; ++j) {{\n"));
            source.push_str(&format!(
                "            {element_type} scratch[{}];\n",
                operand_count.max(1)
            ));
            source.push_str(&format!(
                "            scratch[{weight}] = q4k_value(blk, slot + (uint)j, hdr);\n"
            ));
            for index in 0..operand_count {
                if index == weight {
                    continue;
                }
                source.push_str(&format!(
                "            scratch[{index}] = in{index}[base{index} + (long)(block_start + (int)slot + j) * stride{index}];\n"
            ));
            }
            let value_expr = push_body_steps(
                source,
                resolved.element_body(),
                "            ",
                element_type,
            );
            source.push_str(&format!(
                "            {element_type} value = {value_expr};\n"
            ));
            let combine_expr = scalar_op_expr(reduce_op, &["accumulator", "value"]);
            source.push_str(&format!(
                "            accumulator = seeded ? {combine_expr} : value;\n"
            ));
            source.push_str("            seeded = true;\n");
            source.push_str("        }\n");
            source.push_str("    }\n");
            push_cooperative_reduce_tail(
                source,
                resolved.node,
                reduce_op,
                rank,
                width,
                element_type,
                output_rank,
                epilogue_body,
                epilogue_operands,
                reduce_dims,
                is_broadcast_epilogue,
            )?;
            return Ok(());
        }

        for (index, gather_slot) in gather_slots.iter().copied().enumerate().take(operand_count) {
            source.push_str(&format!(
                "    long stride{index} = u.operand_strides[{index}][{reduce_dim}];\n"
            ));
            source.push_str(&format!("    long off{index} = u.operand_base[{index}];\n"));
            for dim in 0..rank {
                if dim == reduce_dim {
                    continue;
                }
                source.push_str(&format!(
                    "    off{index} += full_coord[{dim}] * u.operand_strides[{index}][{dim}];\n"
                ));
            }
            if let Some(slot) = gather_slot {
                push_cooperative_gather_fetch(
                    source,
                    index,
                    slot,
                    rank,
                    "full_coord",
                    &format!("off{index}"),
                );
            }
            source.push_str(&format!("    off{index} += (long)lane * stride{index};\n"));
            // 32-bit from here down. The offsets ABOVE stay `long` because a
            // layout base can legitimately be one; the per-element WALK never
            // needs that range, and Apple GPUs are 32-bit machines where
            // 64-bit integer arithmetic is emulated. `u.walk_fits_int` is the
            // runtime guard — when an operand's span really does exceed
            // `int`, the 64-bit walk below runs instead.
            source.push_str(&format!("    int walk{index} = (int)off{index};\n"));
            source.push_str(&format!(
                "    int advance{index} = (int)(stride{index} * {width});\n"
            ));
        }
        source.push_str(&format!(
            "    for (int r = (int)lane; r < (int)u.reduction_total; r += {width}) {{\n"
        ));
        source.push_str(&format!(
            "        {element_type} scratch[{}];\n",
            operand_count.max(1)
        ));
        for (index, &codec) in quantized.iter().enumerate() {
            source.push_str(&format!(
                "        scratch[{index}] = {};\n",
                operand_read(index, &format!("walk{index}"), codec)
            ));
        }
        let value_expr = push_body_steps(source, resolved.element_body(), "        ", element_type);
        source.push_str(&format!("        {element_type} value = {value_expr};\n"));
        let combine_expr = scalar_op_expr(reduce_op, &["accumulator", "value"]);
        source.push_str(&format!(
            "        accumulator = seeded ? {combine_expr} : value;\n"
        ));
        source.push_str("        seeded = true;\n");
        for index in 0..operand_count {
            source.push_str(&format!("        walk{index} += advance{index};\n"));
        }
        source.push_str("    }\n");
        push_cooperative_reduce_tail(
            source,
            resolved.node,
            reduce_op,
            rank,
            width,
            element_type,
            output_rank,
            epilogue_body,
            epilogue_operands,
            reduce_dims,
            is_broadcast_epilogue,
        )?;
        return Ok(());
    }

    source.push_str(&format!(
        "    for (long r = (long)lane; r < u.reduction_total; r += {width}) {{\n"
    ));
    if reduce_rank > 0 {
        source.push_str(&format!(
            "        long reduction_coord[{reduce_rank_len}];\n"
        ));
        source.push_str("        long remaining_r = r;\n");
        for index in (0..reduce_rank).rev() {
            source.push_str(&format!(
                "        reduction_coord[{index}] = remaining_r % u.reduction_extents[{index}]; \
                 remaining_r /= u.reduction_extents[{index}];\n"
            ));
        }
        for (index, dim) in reduce_dims.iter().enumerate() {
            source.push_str(&format!(
                "        full_coord[{dim}] = reduction_coord[{index}];\n"
            ));
        }
    }

    for (index, gather_slot) in gather_slots.iter().copied().enumerate().take(operand_count) {
        source.push_str(&format!(
            "        long off{index} = u.operand_base[{index}];\n"
        ));
        for dim in 0..rank {
            source.push_str(&format!(
                "        off{index} += full_coord[{dim}] * u.operand_strides[{index}][{dim}];\n"
            ));
        }
        if let Some(slot) = gather_slot {
            push_cooperative_gather_fetch(
                source,
                index,
                slot,
                rank,
                "full_coord",
                &format!("off{index}"),
            );
        }
    }
    source.push_str(&format!(
        "        {element_type} scratch[{}];\n",
        operand_count.max(1)
    ));
    for (index, &codec) in quantized.iter().enumerate() {
        source.push_str(&format!(
            "        scratch[{index}] = {};\n",
            operand_read(index, &format!("off{index}"), codec)
        ));
    }
    let value_expr = push_body_steps(source, resolved.element_body(), "        ", element_type);
    source.push_str(&format!("        {element_type} value = {value_expr};\n"));
    let combine_expr = scalar_op_expr(reduce_op, &["accumulator", "value"]);
    source.push_str(&format!(
        "        accumulator = seeded ? {combine_expr} : value;\n"
    ));
    source.push_str("        seeded = true;\n");
    source.push_str("    }\n");

    push_cooperative_reduce_tail(
        source,
        resolved.node,
        reduce_op,
        rank,
        width,
        element_type,
        output_rank,
        epilogue_body,
        epilogue_operands,
        reduce_dims,
        is_broadcast_epilogue,
    )?;
    Ok(())
}

/// The per-lane `simd_sum` fold and the final store both cooperative loop
/// shapes end with — shared so the strength-reduced single-reduction-dim
/// path, the general path, and the Q4_K super-block-tiled path (which
/// always calls this with `width == SIMD_WIDTH`, see
/// [`q4k_super_block_tiled`]) cannot drift on how the result is written
/// out.
///
/// `width == SIMD_WIDTH` (32, one simdgroup, the byte-identical prior
/// shape): a single `simd_sum`-class fold and a lane-0 store, unchanged.
///
/// `width > SIMD_WIDTH` (`metal-wide-cooperative-reduce` only --
/// [`cooperative_reduce_width`] never returns a wider value with the
/// feature off): a two-level fold. Each simdgroup folds its own 32 lanes
/// with `simd_combine_fn`, its lane 0 stores that partial into a
/// `threadgroup` array sized to the EXACT simdgroup count this kernel
/// dispatches (`width / SIMD_WIDTH`, baked into the source as a literal --
/// not a uniform, so there is no way to index it out of bounds or read an
/// element no lane wrote). A barrier orders the writes before thread 0
/// folds the partials serially and stores the result. Every lane in every
/// simdgroup of a `width`-wide threadgroup is real (the grid this pairs
/// with is always an exact multiple of `width`, `grid_threads`' own
/// invariant), so every partial slot is written before the fold reads it —
/// there is no ragged-tail case here to guard, unlike the per-lane
/// accumulator seed above (which already handles `reduction_total < width`
/// via `cooperative_identity_token`, both before and after this feature).
#[allow(clippy::too_many_arguments)]
pub(super) fn push_cooperative_reduce_tail(
    source: &mut String,
    node: NodeId,
    reduce_op: ScalarOp,
    rank: usize,
    width: u64,
    element_type: &str,
    output_rank: usize,
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
    reduce_dims: &[u16],
    is_broadcast_epilogue: bool,
) -> Result<(), EmitError> {
    let combine_fn = simd_combine_fn(node, reduce_op)?;
    let simdgroups = width / SIMD_WIDTH;
    let coord = |dim: usize| {
        if output_rank > 0 {
            format!("output_coord[{dim}]")
        } else {
            "0".to_string()
        }
    };
    if simdgroups <= 1 {
        // `combine_fn` is a `simd_*` reduction (`simd_combine_fn`'s own
        // doc): Metal broadcasts its result to every lane of the simdgroup
        // already, so a broadcast-reduce epilogue needs no extra barrier
        // here to make `reduced` visible everywhere `push_broadcast_
        // epilogue_write` reads it from.
        source.push_str(&format!(
            "    {element_type} reduced = {combine_fn}(accumulator);\n"
        ));
        if is_broadcast_epilogue {
            push_broadcast_epilogue_write(
                source,
                rank,
                reduce_dims,
                width,
                epilogue_body,
                epilogue_operands,
                element_type,
                "reduced",
            );
            return Ok(());
        }
        source.push_str("    if (lane == 0u) {\n");
        source.push_str("        long out_offset = u.out_base;\n");
        for dim in 0..rank {
            source.push_str(&format!(
                "        out_offset += full_coord[{dim}] * u.out_strides[{dim}];\n"
            ));
        }
        push_reduce_epilogue_write(
            source,
            epilogue_body,
            epilogue_operands,
            output_rank,
            element_type,
            "        ",
            coord,
            "reduced",
            "out_offset",
        );
        source.push_str("    }\n");
        return Ok(());
    }

    source.push_str(&format!(
        "    {element_type} partial = {combine_fn}(accumulator);\n"
    ));
    source.push_str(&format!(
        "    threadgroup {element_type} partials[{simdgroups}];\n"
    ));
    source.push_str(&format!(
        "    if (lane % {SIMD_WIDTH}u == 0u) {{ partials[lane / {SIMD_WIDTH}u] = partial; }}\n"
    ));
    source.push_str("    threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    if is_broadcast_epilogue {
        // Every lane needs `reduced`, not just lane 0 -- fold on lane 0 same
        // as below, publish it back through `partials[0]`, and a second
        // barrier before every lane reads it, matching llama.cpp's own
        // `kernel_rms_norm` shape (`ggml-metal.metal`: cooperative fold,
        // scalar broadcast through shared memory, every lane writes its own
        // elements).
        source.push_str("    if (lane == 0u) {\n");
        source.push_str(&format!("        {element_type} reduced = partials[0];\n"));
        source.push_str(&format!(
            "        for (uint fold_index = 1u; fold_index < {simdgroups}u; ++fold_index) {{\n"
        ));
        let fold_expr = scalar_op_expr(reduce_op, &["reduced", "partials[fold_index]"]);
        source.push_str(&format!("            reduced = {fold_expr};\n"));
        source.push_str("        }\n");
        source.push_str("        partials[0] = reduced;\n");
        source.push_str("    }\n");
        source.push_str("    threadgroup_barrier(mem_flags::mem_threadgroup);\n");
        source.push_str(&format!("    {element_type} reduced = partials[0];\n"));
        push_broadcast_epilogue_write(
            source,
            rank,
            reduce_dims,
            width,
            epilogue_body,
            epilogue_operands,
            element_type,
            "reduced",
        );
        return Ok(());
    }
    source.push_str("    if (lane == 0u) {\n");
    source.push_str(&format!("        {element_type} reduced = partials[0];\n"));
    source.push_str(&format!(
        "        for (uint fold_index = 1u; fold_index < {simdgroups}u; ++fold_index) {{\n"
    ));
    let fold_expr = scalar_op_expr(reduce_op, &["reduced", "partials[fold_index]"]);
    source.push_str(&format!("            reduced = {fold_expr};\n"));
    source.push_str("        }\n");
    source.push_str("        long out_offset = u.out_base;\n");
    for dim in 0..rank {
        source.push_str(&format!(
            "        out_offset += full_coord[{dim}] * u.out_strides[{dim}];\n"
        ));
    }
    push_reduce_epilogue_write(
        source,
        epilogue_body,
        epilogue_operands,
        output_rank,
        element_type,
        "        ",
        coord,
        "reduced",
        "out_offset",
    );
    source.push_str("    }\n");
    Ok(())
}

/// The broadcast-reduce epilogue's own write tail
/// ([`BoundOpKind::Reduce::epilogue_broadcast_axes`]'s own doc): once the
/// fold's scalar (`reduced_expr`, already visible to every lane -- see each
/// [`push_cooperative_reduce_tail`] call site) is available, every lane
/// re-walks the SAME per-lane strided range over the reduction dims the
/// accumulation loop above just folded (`bind::epilogue_broadcast_axes_for`'s
/// own doc: a non-empty value is always exactly this fold's `reduce_dims`,
/// nothing else -- `render_reduce`'s own gate rejects anything that would
/// disagree), this time writing one output element per step instead of
/// reading one. Same shape llama.cpp's `kernel_rms_norm` takes (`ggml-metal.
/// metal`): a cooperative fold, then every thread writes its own share of
/// the row. `epilogue_operand_strides`/`broadcast_out_strides` are the two
/// uniform rows [`pack_reduce_uniforms`] packs at FULL rank for exactly this
/// loop -- `output_rank` handed to [`push_reduce_epilogue_write`] here is
/// `rank` itself (the widened space), and `coord` reads `full_coord[dim]`
/// rather than `output_coord[dim]`.
#[allow(clippy::too_many_arguments)]
pub(super) fn push_broadcast_epilogue_write(
    source: &mut String,
    rank: usize,
    reduce_dims: &[u16],
    width: u64,
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
    element_type: &str,
    reduced_expr: &str,
) {
    let reduce_rank = reduce_dims.len();
    let reduce_rank_len = reduce_rank.max(1);

    // ROW 370: the fold scalar (`reduced_expr`) is the same value on every
    // pass of the loop below -- any epilogue step that only ever reads it
    // (mean_square + eps, sqrt, reciprocal for rmsnorm) is loop-invariant and
    // is declared exactly once here, before the loop, instead of being
    // re-derived on all 16 iterations a [1,4096] row takes at width 256.
    let epilogue_operand_count = epilogue_operands.len();
    let is_identity = reduce_epilogue_is_identity(epilogue_body, epilogue_operands);
    let is_invariant_operand = |operand_index: usize| {
        operand_index == epilogue_operand_count
            || epilogue_operand_is_loop_invariant(epilogue_operands, reduce_dims, operand_index)
    };
    let mut element_body = String::new();
    let epi_value = if is_identity {
        reduced_expr.to_string()
    } else {
        source.push_str(&format!(
            "    {element_type} epi_scratch[{}];\n",
            epilogue_operand_count + 1
        ));
        push_epilogue_operand_reads(
            source,
            (0..epilogue_operand_count).filter(|&index| is_invariant_operand(index)),
            rank,
            "    ",
            |dim| format!("full_coord[{dim}]"),
        );
        source.push_str(&format!(
            "    epi_scratch[{epilogue_operand_count}] = {reduced_expr};\n"
        ));
        crate::epilogue::declare_steps_partitioned(
            source,
            &mut element_body,
            epilogue_body,
            "epi_scratch",
            "epi_step",
            is_invariant_operand,
            scalar_op_expr,
            |source, index, expr| {
                source.push_str(&format!("    {element_type} epi_step{index} = {expr};\n"));
            },
            |source, index, expr| {
                source.push_str(&format!(
                    "        {element_type} epi_step{index} = {expr};\n"
                ));
            },
        )
    };

    source.push_str(&format!(
        "    for (long r = (long)lane; r < u.reduction_total; r += {width}) {{\n"
    ));
    if reduce_rank > 0 {
        source.push_str(&format!(
            "        long reduction_coord[{reduce_rank_len}];\n"
        ));
        source.push_str("        long remaining_r = r;\n");
        for index in (0..reduce_rank).rev() {
            source.push_str(&format!(
                "        reduction_coord[{index}] = remaining_r % u.reduction_extents[{index}]; \
                 remaining_r /= u.reduction_extents[{index}];\n"
            ));
        }
        for (index, dim) in reduce_dims.iter().enumerate() {
            source.push_str(&format!(
                "        full_coord[{dim}] = reduction_coord[{index}];\n"
            ));
        }
    }
    source.push_str("        long out_offset = u.out_base;\n");
    for dim in 0..rank {
        source.push_str(&format!(
            "        out_offset += full_coord[{dim}] * u.broadcast_out_strides[{dim}];\n"
        ));
    }
    if is_identity {
        source.push_str(&format!("        out[out_offset] = {epi_value};\n"));
    } else {
        push_epilogue_operand_reads(
            source,
            (0..epilogue_operand_count).filter(|&index| !is_invariant_operand(index)),
            rank,
            "        ",
            |dim| format!("full_coord[{dim}]"),
        );
        source.push_str(&element_body);
        source.push_str(&format!("        out[out_offset] = {epi_value};\n"));
    }
    source.push_str("    }\n");
}

pub(super) fn render_scan(
    resolved: &BoundOp,
    entry: &str,
    quantized: &[Option<Codec>],
) -> Result<String, EmitError> {
    let BoundOpKind::Reduce {
        reduce_op, init, ..
    } = &resolved.kind
    else {
        return Err(EmitError::RenderKindMismatch {
            node: resolved.node,
            expected: "keep::scan fold",
            found: resolved.kind.name(),
        });
    };
    let rank = resolved.extents.len();
    let rank_len = rank.max(1);
    let outer_rank = rank.saturating_sub(1);
    let outer_rank_len = outer_rank.max(1);
    let last_dim = rank.saturating_sub(1);
    let operand_count = resolved.operands().len();
    let gather_count = gather_count(resolved);
    let gather_slots = gather_slots(resolved);
    let element_type = type_token(resolved.node, resolved.dtype)?;

    let mut source = String::new();
    preamble(&mut source);

    source.push_str("struct Uniforms {\n");
    source.push_str("    long outer_total;\n");
    source.push_str("    long inner_len;\n");
    source.push_str(&format!("    long outer_extents[{outer_rank_len}];\n"));
    source.push_str(&format!("    long operand_base[{operand_count}];\n"));
    source.push_str(&format!(
        "    long operand_strides[{operand_count}][{rank_len}];\n"
    ));
    source.push_str("    long out_base;\n");
    source.push_str(&format!("    long out_strides[{rank_len}];\n"));
    push_gather_uniform_fields(&mut source, gather_count, rank_len);
    source.push_str("};\n\n");

    kernel_signature(
        &mut source,
        quantized,
        0,
        gather_count,
        entry,
        element_type,
        false,
    );
    source.push_str("    if ((long)gid >= u.outer_total) { return; }\n");

    if outer_rank > 0 {
        source.push_str(&format!("    long outer_coord[{outer_rank_len}];\n"));
        source.push_str("    long remaining = (long)gid;\n");
        for dim in (0..outer_rank).rev() {
            source.push_str(&format!(
                "    outer_coord[{dim}] = remaining % u.outer_extents[{dim}]; \
                 remaining /= u.outer_extents[{dim}];\n"
            ));
        }
    }

    for (index, gather_slot) in gather_slots.iter().enumerate() {
        source.push_str(&format!(
            "    long running{index} = u.operand_base[{index}];\n"
        ));
        for dim in 0..outer_rank {
            source.push_str(&format!(
                "    running{index} += outer_coord[{dim}] * u.operand_strides[{index}][{dim}];\n"
            ));
        }
        if let Some(slot) = gather_slot {
            source.push_str(&format!(
                "    long gather_running{index} = u.gather_index_base[{slot}];\n"
            ));
            for dim in 0..outer_rank {
                source.push_str(&format!(
                    "    gather_running{index} += outer_coord[{dim}] * u.gather_index_strides[{slot}][{dim}];\n"
                ));
            }
        }
    }
    source.push_str("    long out_running = u.out_base;\n");
    for dim in 0..outer_rank {
        source.push_str(&format!(
            "    out_running += outer_coord[{dim}] * u.out_strides[{dim}];\n"
        ));
    }

    let (init_expr, seeded_init) = fold_init_tokens(*init);
    source.push_str(&format!("    {element_type} accumulator = {init_expr};\n"));
    source.push_str(&format!("    bool seeded = {seeded_init};\n"));

    source.push_str("    for (long step = 0; step < u.inner_len; step++) {\n");
    source.push_str(&format!(
        "        {element_type} scratch[{}];\n",
        operand_count.max(1)
    ));
    for (index, gather_slot) in gather_slots.iter().enumerate() {
        // the gathered dim's contribution is per-step (the fetched index
        // varies along the scanned dim too, in general), so it is combined
        // into a fresh `read_off` here rather than folded permanently into
        // `running{index}`, which must keep advancing by its own stride
        // alone — see the module doc's Uniforms-packing note for why.
        if let Some(slot) = gather_slot {
            source.push_str(&format!(
                "        long fetched{index} = (long)gather_idx{slot}[gather_running{index}];\n"
            ));
            push_gather_fault_check(&mut source, index, *slot, "        ");
            source.push_str(&format!(
                "        fetched{index} = max((long)0, min(fetched{index}, u.gather_extent[{slot}] - 1));\n"
            ));
            source.push_str(&format!(
                "        long read_off{index} = running{index} + fetched{index} * u.gather_element_stride[{slot}];\n"
            ));
            source.push_str(&format!(
                "        scratch[{index}] = {};\n",
                operand_read(index, &format!("read_off{index}"), quantized[index])
            ));
            source.push_str(&format!(
                "        gather_running{index} += u.gather_index_strides[{slot}][{last_dim}];\n"
            ));
        } else {
            source.push_str(&format!(
                "        scratch[{index}] = {};\n",
                operand_read(index, &format!("running{index}"), quantized[index])
            ));
        }
        source.push_str(&format!(
            "        running{index} += u.operand_strides[{index}][{last_dim}];\n"
        ));
    }
    let value_expr = push_body_steps(
        &mut source,
        resolved.element_body(),
        "        ",
        element_type,
    );
    source.push_str(&format!("        {element_type} value = {value_expr};\n"));
    let combine_expr = scalar_op_expr(*reduce_op, &["accumulator", "value"]);
    source.push_str(&format!(
        "        accumulator = seeded ? {combine_expr} : value;\n"
    ));
    source.push_str("        seeded = true;\n");
    source.push_str("        out[out_running] = accumulator;\n");
    source.push_str(&format!(
        "        out_running += u.out_strides[{last_dim}];\n"
    ));
    source.push_str("    }\n");
    source.push_str("}\n");
    Ok(source)
}

