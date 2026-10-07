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
#[allow(clippy::too_many_arguments)]
pub(super) fn push_tiled_gemm_body(
    source: &mut String,
    node: NodeId,
    output_axes: &[u16],
    rank: usize,
    block: &TiledGemmBlock,
    element_type: &str,
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
    metal: &crate::identity::MetalOnlyExtras,
) -> Result<(), EmitError> {
    let TiledGemmBlock {
        weight,
        other,
        reduce_dim,
        ref token_axes,
        ref feature_axes,
        codec,
        ..
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

    // `PROXIMA_TILED_GEMM_GRID2D` (see `docs/model-interop/discipline.md`
    // ROW C4.19, `MetalOnlyExtras::tiled_gemm_grid2d`'s own doc): swaps the flattened
    // `uint gid [[thread_position_in_grid]]` `kernel_signature` already
    // emitted for ggml's own 2D attribute triple, and derives `row_tile`/
    // `col_tile` straight from `tgpig` instead of dividing/modding a
    // flattened `tile_index` by `num_col_tiles` -- the harness's own
    // `S2_2d_index_attrs` ablation validated this bit-identical before this
    // switch existed.
    let grid2d_active = metal.tiled_gemm_grid2d;
    if grid2d_active {
        let scalar_gid = "uint gid [[thread_position_in_grid]]";
        let threadgroup_argument = if metal.tiled_gemm_dynamic_tgmem {
            ",\n    threadgroup uchar *tg_shared [[threadgroup(0)]]"
        } else {
            ""
        };
        let vector_attrs = format!(
            "uint3 tgpig [[threadgroup_position_in_grid]],\n    \
            ushort tiitg [[thread_index_in_threadgroup]],\n    \
            ushort sgitg [[simdgroup_index_in_threadgroup]]{threadgroup_argument}"
        );
        let gid_offset = source.find(scalar_gid).ok_or(EmitError::RenderKindMismatch {
            node,
            expected: "scalar thread_position_in_grid parameter",
            found: "missing",
        })?;
        source.replace_range(gid_offset..gid_offset + scalar_gid.len(), &vector_attrs);
    }
    source.push_str(&format!(
        "    long feature_extent = {};\n",
        group_extent_expr(feature_axes)?
    ));
    source.push_str(&format!(
        "    long token_extent = {};\n",
        group_extent_expr(token_axes)?
    ));
    if grid2d_active {
        source.push_str("    long row_tile = (long)tgpig.y;\n");
        source.push_str("    long col_tile = (long)tgpig.x;\n");
        source.push_str("    long row_half = sgitg & 1;\n");
        source.push_str("    long col_half = sgitg >> 1;\n");
    } else {
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
    }
    // phase 2 (see `docs/model-interop/discipline.md` ROW C4.10,
    // `PROXIMA_TILED_GEMM_SLIM_TGMEM=1`):
    // `weight_tile`/`act_tile` are dead by the time `out_tile` needs its
    // bytes (the K-loop's own trailing `threadgroup_barrier` below already
    // fences the last read before any thread could reach the aliased
    // write), so a shared backing array sized to the larger of the two
    // phases' needs replaces three separately-sized ones -- mirrors ggml's
    // own `kernel_mul_mm` `shmem` reuse (`ggml-metal.metal:160-161,330`).
    let act_element_type = "float";
    let act_value_cast = "";
    let act_simdgroup_type = "simdgroup_float8x8";
    let slim_tgmem_active = metal.tiled_gemm_slim_tgmem;
    let weight_tile_bytes = weight_tile_elems * 2;
    if slim_tgmem_active {
        if !metal.tiled_gemm_dynamic_tgmem {
            source.push_str(&format!(
                "    threadgroup uchar tg_shared[{}];\n",
                tiled_gemm_shared_bytes()
            ));
        }
        source.push_str("    threadgroup half *weight_tile = (threadgroup half *)tg_shared;\n");
        source.push_str(&format!(
            "    threadgroup {act_element_type} *act_tile = (threadgroup {act_element_type} *)(tg_shared + {weight_tile_bytes});\n"
        ));
    } else {
        source.push_str(&format!(
            "    threadgroup half weight_tile[{weight_tile_elems}];\n"
        ));
        source.push_str(&format!(
            "    threadgroup {act_element_type} act_tile[{act_tile_elems}];\n"
        ));
    }
    source.push_str(&format!("    simdgroup_float8x8 acc[{mc_count}];\n"));
    source.push_str(&format!(
        "    for (int i = 0; i < {mc_count}; ++i) {{ acc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f); }}\n"
    ));
    // ROW 113: weight staging amortizes the Q4_K sub-block header the same
    // way `push_packed_row_blocked_body` and ggml's own `dequantize_q4_K`
    // (ggml-metal.metal:336-352) both do -- one `q4k_header_for` per
    // 32-element sub-block, `q4k_run8` batching the nibble extract 8 at a
    // time -- instead of `operand_read`'s generic `q4k_element`, which
    // rederives the header (two `device` header reads plus the 6-bit
    // scale/min unpack) from scratch on every one of the tile's individual
    // elements.
    // `tiled_gemm_codec_chunk_width` is the SAME definition `classify_
    // tiled_gemm` checks against `tiled_gemm_block_k_chunk_aligned` before
    // admitting an op onto this path (`emit_and_classify.rs`'s own doc on
    // that check) -- an admitted op's `block_k` is always either <= this
    // chunk width or a whole multiple of it, so `num_chunks` below covers
    // `block_k` exactly with no ragged remainder, never a `weight_tile`
    // overrun. The SAME chunking loop serves both codecs; only the
    // per-chunk decode differs (`classify_tiled_gemm` only ever admits
    // `Q4_0`/`Q4_K`).
    let block_elements = u64::try_from(codec_block_elements(codec)).unwrap_or(u64::MAX);
    let block_bytes = u64::try_from(codec_block_bytes(codec)).unwrap_or(u64::MAX);
    let chunk_width = tiled_gemm_codec_chunk_width(codec).min(block_k);
    let num_chunks = block_k.div_ceil(chunk_width);

    // `PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE` (see `docs/model-interop/
    // discipline.md` ROW C4.12, `MetalOnlyExtras::tiled_gemm_wide_weight_stage`'s own doc): admitted
    // only when a codec's own decode contract (8-element runs) still divides
    // evenly into a HALF of one chunk (so a thread's half-chunk assignment
    // never splits a run) AND the block/K-step ratio is a clean whole number
    // in EITHER direction -- `chunk_width == block_elements` (Q4_0 with
    // today's sizing, or any config where the codec's own block is no wider
    // than one K-step: `block_k` a whole multiple of it) or `chunk_width ==
    // block_k` (Q4_K with today's sizing: the K-step is narrower than the
    // codec's super-block, so one super-block spans a whole multiple of
    // K-steps). Both are checked, not assumed, so a future non-default
    // `block_k` that breaks either ratio silently falls back to the
    // existing per-K-step division instead of emitting a wrong pointer
    // advance.
    let wide_weight_stage_eligible = metal.tiled_gemm_wide_weight_stage
        && chunk_width.is_multiple_of(16)
        && if chunk_width == block_elements {
            block_k.is_multiple_of(chunk_width)
        } else {
            chunk_width == block_k && block_elements.is_multiple_of(block_k)
        };
    let half_width = chunk_width / 2;
    let total_halves = block_m * num_chunks * 2;
    let half_units = total_halves.div_ceil(block_threads);

    let mm_layout_schedule = metal.tiled_gemm_mm_layout
        && wide_weight_stage_eligible
        && mm_layout_geometry_supported()
        && half_width == 16;

    if mm_layout_schedule {
        push_mm_layout_k_loop(source, block, token_axis, feature_axis, metal, grid2d_active);
    } else {
        if wide_weight_stage_eligible {
            push_wide_weight_stage_setup(
                source,
                weight,
                feature_axis,
                block_m,
                block_k,
                num_chunks,
                half_width,
                total_halves,
                half_units,
                block_elements,
                block_bytes,
            );
        }

        let k0_counter_type = if grid2d_active { "int" } else { "long" };
        source.push_str(&format!(
            "    for ({k0_counter_type} k0 = 0; k0 < u.reduction_total; k0 += {block_k}) {{\n"
        ));
        if wide_weight_stage_eligible {
            push_wide_weight_stage_body(
                source,
                codec,
                block_k,
                num_chunks,
                half_width,
                half_units,
                chunk_width,
                block_elements,
                block_bytes,
            );
        } else {
            // Staged by ROW rather than by flat index: `block_threads` (128)
            // exceeds `block_m` (64) with the default sizing, so the first
            // `block_m` threads each own exactly one row of the tile for this
            // phase and the rest do no extra weight work (`act_tile`'s own load
            // below still uses every thread) -- see `wide_weight_stage_eligible`
            // above for the schedule that fixes this idle half.
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
            for chunk_index in 0..num_chunks {
                let chunk_offset = chunk_index * chunk_width;
                source.push_str("                {\n");
                source.push_str(&format!(
                    "                    long slot_off = row_base + {chunk_offset};\n"
                ));
                source.push_str(&format!(
                    "                    device const uchar *blk = in{weight} + (slot_off / {block_elements}) * {block_bytes};\n"
                ));
                source.push_str(&format!(
                    "                    uint slot = (uint)(slot_off % {block_elements});\n"
                ));
                match codec {
                    Codec::Q4_0 => {
                        // ROW 113's discipline (read the block's scale ONCE, not
                        // once per element) now applies here too, via `Q4_0`'s own
                        // batched sibling to `q4k_run8` (`Q4_0_RUN8_MSL`, see its
                        // own doc): `q4_0_block_scale` reads `d` once per
                        // 32-element block, `q4_0_run8` batches the raw-nibble
                        // extract 8 at a time, same shape as the Q4_K arm below.
                        source.push_str("                    float q4_0_d = q4_0_block_scale(blk);\n");
                        let runs = chunk_width / 8;
                        for run_index in 0..runs {
                            let run_offset = run_index * 8;
                            source.push_str("                    {\n");
                            source.push_str("                        float levels[8];\n");
                            source.push_str(&format!(
                                "                        q4_0_run8(blk, slot + {run_offset}u, levels);\n"
                            ));
                            let weight_index = format!("w_row * {block_k} + {chunk_offset} + {run_offset} + j");
                            source.push_str(&format!(
                                "                        for (int j = 0; j < 8; ++j) {{ weight_tile[{weight_index}] = (half)((levels[j] - 8.0f) * q4_0_d); }}\n"
                            ));
                            source.push_str("                    }\n");
                        }
                    }
                    _ => {
                        source.push_str("                    q4k_header hdr = q4k_header_for(blk, slot);\n");
                        let runs = chunk_width / 8;
                        for run_index in 0..runs {
                            let run_offset = run_index * 8;
                            source.push_str("                    {\n");
                            source.push_str("                        float levels[8];\n");
                            source.push_str(&format!(
                                "                        q4k_run8(blk, slot + {run_offset}u, levels);\n"
                            ));
                            let weight_index = format!("w_row * {block_k} + {chunk_offset} + {run_offset} + j");
                            source.push_str(&format!(
                                "                        for (int j = 0; j < 8; ++j) {{ weight_tile[{weight_index}] = (half)(hdr.scale * levels[j] - hdr.minimum); }}\n"
                            ));
                            source.push_str("                    }\n");
                        }
                    }
                }
                source.push_str("                }\n");
            }
            source.push_str("            } else {\n");
            let weight_fill_index = format!("w_row * {block_k} + fill_k");
            source.push_str(&format!(
                "                for (long fill_k = 0; fill_k < {block_k}; ++fill_k) {{ weight_tile[{weight_fill_index}] = 0.0h; }}\n"
            ));
            source.push_str("            }\n");
            source.push_str("        }\n");
        }
        // item 3c (`PROXIMA_TILED_GEMM_WIDE_ACT_LOAD=1`, `STAGING.md` §5.3):
        // admitted (by `wide_activation_load_active`, at classification time)
        // only when the activation operand's stride along the reduce dim is
        // exactly 1, so four LOGICALLY consecutive `a_k` values are four
        // PHYSICALLY consecutive device floats -- safe to read as one
        // `float4`. `block_k % 4 == 0` is guaranteed by `build.rs`'s
        // `require_multiple_of_eight` (a multiple of 8 is a multiple of 4), so
        // a 4-wide flat-index run never straddles an `a_col` boundary.
        let wide_act_load_active = metal.tiled_gemm_wide_act_load && block_k.is_multiple_of(4);
        if wide_act_load_active {
            source.push_str(&format!(
                "        bool act_tile_interior = (col_tile * {block_n} + {block_n} <= token_extent);\n"
            ));
            source.push_str("        if (act_tile_interior) {\n");
            source.push_str(&format!(
                "            for (long idx4 = tiitg; idx4 < {}; idx4 += {block_threads}) {{\n",
                act_tile_elems / 4
            ));
            source.push_str("                long flat = idx4 * 4;\n");
            source.push_str(&format!("                long a_col = flat / {block_k};\n"));
            source.push_str(&format!("                long a_k = flat % {block_k};\n"));
            source.push_str(&format!(
                "                long a_tok = col_tile * {block_n} + a_col;\n"
            ));
            source.push_str("                long a_k_global = k0 + a_k;\n");
            source.push_str(&format!(
                "                long aoff = u.operand_base[{other}] + a_tok * u.operand_strides[{other}][{token_axis}] + a_k_global * u.operand_strides[{other}][{reduce_dim}];\n"
            ));
            source.push_str(&format!(
                "                float4 wide = *(const device float4 *)(in{other} + aoff);\n"
            ));
            for (lane, component) in ["x", "y", "z", "w"].into_iter().enumerate() {
                let suffix = if lane == 0 {
                    String::new()
                } else {
                    format!(" + {lane}")
                };
                let index = format!("a_col * {block_k} + a_k{suffix}");
                source.push_str(&format!(
                    "                act_tile[{index}] = wide.{component};\n"
                ));
            }
            source.push_str("            }\n");
            source.push_str("        } else {\n");
        }
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
        let act_scalar_index = format!("a_col * {block_k} + a_k");
        source.push_str(&format!(
            "            act_tile[{act_scalar_index}] = {act_value_cast}a_value;\n"
        ));
        source.push_str("        }\n");
        if wide_act_load_active {
            source.push_str("        }\n");
        }
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
        let (a_frag_offset, a_frag_stride) =
            fragment_load_offset_stride("row_half", "i", thread_mat_m, block_k);
        source.push_str(&format!(
            "                simdgroup_load(a_frag[i], weight_tile + {a_frag_offset}, {a_frag_stride});\n"
        ));
        source.push_str("            }\n");
        source.push_str("            simdgroup_barrier(mem_flags::mem_none);\n");
        source.push_str(&format!(
            "            {act_simdgroup_type} b_frag[{thread_mat_n}];\n"
        ));
        source.push_str(&format!(
            "            for (int j = 0; j < {thread_mat_n}; ++j) {{\n"
        ));
        let (b_frag_offset, b_frag_stride) =
            fragment_load_offset_stride("col_half", "j", thread_mat_n, block_k);
        source.push_str(&format!(
            "                simdgroup_load(b_frag[j], act_tile + {b_frag_offset}, {b_frag_stride}, ulong2(0), true);\n"
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
    }

    if metal.tiled_gemm_slim_tgmem {
        source.push_str("    threadgroup float *out_tile = (threadgroup float *)tg_shared;\n");
    } else {
        source.push_str(&format!(
            "    threadgroup float out_tile[{out_tile_elems}];\n"
        ));
    }
    // lever 2 (see `docs/model-interop/discipline.md` ROW C4.11): a `device float*` direct-store pointer
    // is only valid when `out`'s own element type IS `float` (`out
    // [[buffer(N)]]` is declared `device {element_type}*` --
    // `signature_tokens_prelude.rs:597`); `metal.tiled_gemm_direct_store`
    // alone cannot see `element_type`, so this emitter is the one place that
    // actually decides whether the direct-store text gets rendered at all.
    let direct_store_eligible = metal.tiled_gemm_direct_store && element_type == "float";
    if direct_store_eligible {
        push_tiled_gemm_direct_store_arm(
            source,
            rank,
            &[],
            feature_axis,
            token_axis,
            block_m,
            block_n,
            thread_mat_m,
            thread_mat_n,
            mm_layout_schedule,
        );
        source.push_str("    } else {\n");
    }
    push_tiled_gemm_restage_writeback(
        source,
        rank,
        output_axes,
        &[],
        feature_axis,
        token_axis,
        block_m,
        block_n,
        block_threads,
        thread_mat_m,
        thread_mat_n,
        out_tile_elems,
        element_type,
        epilogue_body,
        epilogue_operands,
        mm_layout_schedule,
    );
    if direct_store_eligible {
        source.push_str("    }\n");
    }
    Ok(())
}

/// `true` when the build-time tile sizing is the geometry ggml's
/// `kernel_mul_mm` is written for (`mul_mm.metal:164-170`: `NR0` 64 weight
/// rows, `NR1` 32 tokens, `NK` 32 deep, four simdgroups): the weight and
/// activation tile layouts [`push_mm_layout_k_loop`] emits hard-code the
/// `64 x 32 x 32` tile and 128 threads, so any other sizing keeps the
/// row-major tile path.
#[cfg(feature = "metal-tiled-gemm")]
pub(super) const fn mm_layout_geometry_supported() -> bool {
    crate::sized::TILED_GEMM_BLOCK_M == 64
        && crate::sized::TILED_GEMM_BLOCK_N == 32
        && crate::sized::TILED_GEMM_BLOCK_K == 32
        && (TILED_GEMM_NSG as u64) * SIMD_WIDTH == 128
}

/// Bytes of the slim threadgroup backing store of [`push_tiled_gemm_body`]: the
/// `half` weight tile and the `float` activation tile side by side, or the `float`
/// output tile aliased over them, whichever is larger. The one number the
/// kernel's declared array and `Grid2DSpec::threadgroup_bytes` both read.
#[cfg(feature = "metal-tiled-gemm")]
pub(super) const fn tiled_gemm_shared_bytes() -> u64 {
    let weight_and_activation = crate::sized::TILED_GEMM_BLOCK_M * crate::sized::TILED_GEMM_BLOCK_K * 2
        + crate::sized::TILED_GEMM_BLOCK_N * crate::sized::TILED_GEMM_BLOCK_K * 4;
    let output = crate::sized::TILED_GEMM_BLOCK_M * crate::sized::TILED_GEMM_BLOCK_N * 4;
    if weight_and_activation > output { weight_and_activation } else { output }
}

/// The K-reduction loop of [`push_tiled_gemm_body`] in ggml's own tile layout
/// (`kernel_mul_mm`, `mul_mm.metal:164-316`), for a packed weight on the wide
/// weight stage's schedule (two threads per weight row, one half-block each):
///
/// - the weight tile is 4 x 8 blocks of 8x8 `half`, each block `[k][feature]`
///   (feature fastest), so a fragment loads untransposed as the `k x feature`
///   operand; the activation tile is 4 x 4 blocks of 8x8, each `[token][k]`,
///   loaded untransposed as the `token x k` operand;
/// - the multiply is `acc += act_frag * weight_frag`, so every accumulator
///   fragment is `token x feature` and the writeback stores it transposed into
///   the `[feature][token]` `out_tile` the copy-out loop already reads;
/// - rows and tokens past the extents are clamped to the last valid one
///   instead of zero-filled; their results are masked at the write.
///
/// A Q4_0 half-block decodes through `q4_0_dequant_half16` (ggml's
/// `dequantize_q4_0` form: one fused multiply-add per element), which rounds
/// to the same `half` as the `(level - 8) * d` expression of the row-major
/// path -- the product is exact in `float`, so the output is bit-identical.
/// With a unit-stride activation the K offset is `k0` itself, not `k0 *
/// stride`.
#[cfg(feature = "metal-tiled-gemm")]
fn push_mm_layout_k_loop(
    source: &mut String,
    block: &TiledGemmBlock,
    token_axis: u16,
    feature_axis: u16,
    metal: &crate::identity::MetalOnlyExtras,
    grid2d_active: bool,
) {
    let weight = block.weight;
    let other = block.other;
    let reduce_dim = block.reduce_dim;
    let codec = block.codec;
    let block_k = crate::sized::TILED_GEMM_BLOCK_K;
    let block_elements = u64::try_from(codec_block_elements(codec)).unwrap_or(u64::MAX);
    let block_bytes = u64::try_from(codec_block_bytes(codec)).unwrap_or(u64::MAX);
    let chunk_width = tiled_gemm_codec_chunk_width(codec).min(block_k);
    let half_width = chunk_width / 2;
    let unit_stride = metal.tiled_gemm_wide_act_load;
    let k0_counter_type = if grid2d_active { "int" } else { "long" };

    source.push_str("    long mm_row = tiitg / 2;\n");
    source.push_str("    long mm_half = tiitg % 2;\n");
    source.push_str("    long mm_feat = min(row_tile * 64 + mm_row, feature_extent - 1);\n");
    source.push_str(&format!(
        "    long mm_wbase = u.operand_base[{weight}] + mm_feat * u.operand_strides[{weight}][{feature_axis}] + mm_half * {half_width};\n"
    ));
    source.push_str(&format!(
        "    device const uchar *wws_blk0 = in{weight} + (mm_wbase / {block_elements}) * {block_bytes};\n"
    ));
    source.push_str(&format!(
        "    uint wws_slot0 = (uint)(mm_wbase % {block_elements});\n"
    ));
    source.push_str("    long mm_weight_store = 64 * (16 * mm_half + mm_row / 8) + mm_row % 8;\n");
    source.push_str("    long mm_token = min(col_tile * 32 + tiitg / 4, token_extent - 1);\n");
    source.push_str(&format!(
        "    long mm_act_base = u.operand_base[{other}] + mm_token * u.operand_strides[{other}][{token_axis}] + (tiitg % 4) * 8 * u.operand_strides[{other}][{reduce_dim}];\n"
    ));
    source.push_str(
        "    long mm_act_store = 64 * (4 * (tiitg % 4) + (tiitg / 4) / 8) + 8 * ((tiitg / 4) % 8);\n",
    );
    if unit_stride {
        source.push_str(&format!(
            "    device const float *mm_act_ptr = in{other} + mm_act_base;\n"
        ));
    }
    source.push_str("    simdgroup_half8x8 ma[4];\n");
    source.push_str("    simdgroup_float8x8 mb[2];\n");
    source.push_str(&format!(
        "    for ({k0_counter_type} k0 = 0; k0 < u.reduction_total; k0 += {block_k}) {{\n"
    ));
    push_mm_layout_weight_decode(source, codec, half_width);
    push_wide_weight_stage_advance(
        source,
        0,
        block_k,
        1,
        chunk_width,
        block_elements,
        block_bytes,
    );
    source.push_str("        threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    for run in 0..half_width / 8 {
        source.push_str(&format!(
            "        for (int j = 0; j < 8; ++j) {{ weight_tile[mm_weight_store + {} + 8 * j] = decoded[{} + j]; }}\n",
            512 * run,
            run * 8
        ));
    }
    push_mm_layout_activation_stage(source, block, unit_stride);
    source.push_str("        threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    push_mm_layout_multiply(source);
    source.push_str("    }\n");
    source.push_str("    threadgroup_barrier(mem_flags::mem_threadgroup);\n");
}

/// Decodes this thread's half-block of the current K-step into `decoded[16]`,
/// ahead of the barrier that fences the previous step's multiplies (ggml's own
/// order: `kernel_mul_mm` dequantizes, then waits, then stores), so the
/// device reads of one simdgroup overlap the multiplies of its slower peers.
#[cfg(feature = "metal-tiled-gemm")]
fn push_mm_layout_weight_decode(source: &mut String, codec: Codec, half_width: u64) {
    source.push_str("        half decoded[16];\n");
    match codec {
        Codec::Q4_0 => {
            source.push_str("        q4_0_dequant_half16(wws_blk0, (uint)mm_half, decoded);\n");
        }
        _ => {
            source.push_str("        q4k_header mm_header = q4k_header_for(wws_blk0, wws_slot0);\n");
            for run in 0..half_width / 8 {
                source.push_str("        {\n");
                source.push_str("            float levels[8];\n");
                source.push_str(&format!(
                    "            q4k_run8(wws_blk0, wws_slot0 + {}u, levels);\n",
                    run * 8
                ));
                source.push_str(&format!(
                    "            for (int j = 0; j < 8; ++j) {{ decoded[{} + j] = (half)(mm_header.scale * levels[j] - mm_header.minimum); }}\n",
                    run * 8
                ));
                source.push_str("        }\n");
            }
        }
    }
}

/// Reads this thread's 8 activation floats of the current K-step and stores them
/// into `act_tile`. A unit-stride activation walks one carried pointer, 32
/// floats per step; any other layout reads each element through the operand
/// accessor at the runtime stride.
#[cfg(feature = "metal-tiled-gemm")]
fn push_mm_layout_activation_stage(source: &mut String, block: &TiledGemmBlock, unit_stride: bool) {
    let other = block.other;
    let reduce_dim = block.reduce_dim;
    if unit_stride {
        source.push_str("        float4 act_low = *(const device float4 *)(mm_act_ptr);\n");
        source.push_str("        float4 act_high = *(const device float4 *)(mm_act_ptr + 4);\n");
        source.push_str(&format!(
            "        mm_act_ptr += {};\n",
            crate::sized::TILED_GEMM_BLOCK_K
        ));
    } else {
        source.push_str(&format!(
            "        long mm_act_offset = mm_act_base + k0 * u.operand_strides[{other}][{reduce_dim}];\n"
        ));
        source.push_str("        float act_scalar[8];\n");
        source.push_str(&format!(
            "        for (int e = 0; e < 8; ++e) {{ long element_offset = mm_act_offset + e * u.operand_strides[{other}][{reduce_dim}]; act_scalar[e] = {}; }}\n",
            operand_read(other, "element_offset", None)
        ));
        source.push_str(
            "        float4 act_low = float4(act_scalar[0], act_scalar[1], act_scalar[2], act_scalar[3]);\n",
        );
        source.push_str(
            "        float4 act_high = float4(act_scalar[4], act_scalar[5], act_scalar[6], act_scalar[7]);\n",
        );
    }
    source.push_str("        *(threadgroup float4 *)&act_tile[mm_act_store] = act_low;\n");
    source.push_str("        *(threadgroup float4 *)&act_tile[mm_act_store + 4] = act_high;\n");
}

/// The four 8-deep multiply steps of one K-step, in `kernel_mul_mm`'s own
/// form (`mul_mm.metal:290-314`): fragment pointers walked by addition, the
/// loads grouped ahead of the multiplies by `simdgroup_barrier`s, every loop
/// fully unrolled. `acc[(i % 4) * 2 + i / 4]` is ggml's flat `mc[i]` renamed to
/// the `[feature][token]` index the write-back reads.
#[cfg(feature = "metal-tiled-gemm")]
fn push_mm_layout_multiply(source: &mut String) {
    let unroll = "_Pragma(\"clang loop unroll(full)\")";
    source.push_str("        threadgroup const half *lsma = weight_tile + 4 * 64 * (sgitg % 2);\n");
    source.push_str("        threadgroup const float *lsmb = act_tile + 2 * 64 * (sgitg / 2);\n");
    source.push_str(&format!("        {unroll} for (short ik = 0; ik < 4; ik++) {{\n"));
    source.push_str("            simdgroup_barrier(mem_flags::mem_none);\n");
    source.push_str(&format!("            {unroll} for (short i = 0; i < 4; i++) {{\n"));
    source.push_str("                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);\n");
    source.push_str("            }\n");
    source.push_str("            simdgroup_barrier(mem_flags::mem_none);\n");
    source.push_str(&format!("            {unroll} for (short i = 0; i < 2; i++) {{\n"));
    source.push_str("                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);\n");
    source.push_str("            }\n");
    source.push_str("            simdgroup_barrier(mem_flags::mem_none);\n");
    source.push_str(&format!("            {unroll} for (short i = 0; i < 8; i++) {{\n"));
    source.push_str(
        "                simdgroup_multiply_accumulate(acc[(i % 4) * 2 + i / 4], mb[i / 4], ma[i % 4], acc[(i % 4) * 2 + i / 4]);\n",
    );
    source.push_str("            }\n");
    source.push_str("            lsma += 8 * 64;\n");
    source.push_str("            lsmb += 4 * 64;\n");
    source.push_str("        }\n");
}

/// `PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE`'s setup half: computed ONCE, before
/// the `k0` reduction loop opens, for each of the (typically one)
/// `half_units` this thread iterates -- a HALF-BLOCK unit `h = tiitg +
/// unit * block_threads` decomposes into `(row, chunk, half)` via division
/// by the compile-time constants `num_chunks * 2` and `2` (the Metal
/// compiler strength-reduces these the same way it would the original
/// per-K-step `slot_off / block_elements`; the difference this switch buys
/// is doing it ONCE per row instead of once per row per `k0` step -- see
/// [`crate::identity::MetalOnlyExtras::tiled_gemm_wide_weight_stage`]'s own
/// doc). `total_halves` (`block_m * num_chunks * 2`) equals `block_threads`
/// under today's default sizing (`block_m`=64, `num_chunks`=1), so every one
/// of the threadgroup's 128 threads owns exactly one half-block -- the fix
/// for the idle-half-of-the-threadgroup defect the non-wide path still has.
#[cfg(feature = "metal-tiled-gemm")]
#[allow(clippy::too_many_arguments)]
fn push_wide_weight_stage_setup(
    source: &mut String,
    weight: usize,
    feature_axis: u16,
    block_m: u64,
    block_k: u64,
    num_chunks: u64,
    half_width: u64,
    total_halves: u64,
    half_units: u64,
    block_elements: u64,
    block_bytes: u64,
) {
    let chunks_times_two = num_chunks * 2;
    for unit in 0..half_units {
        let h_offset = unit * block_threads_const();
        source.push_str(&format!(
            "    long wws_h{unit} = tiitg + {h_offset};\n"
        ));
        source.push_str(&format!(
            "    device const uchar *wws_blk{unit} = in{weight};\n"
        ));
        source.push_str(&format!("    uint wws_slot{unit} = 0u;\n"));
        source.push_str(&format!("    long wws_row{unit} = 0;\n"));
        source.push_str(&format!("    long wws_koff{unit} = 0;\n"));
        source.push_str(&format!("    bool wws_active{unit} = false;\n"));
        source.push_str(&format!(
            "    if (wws_h{unit} < {total_halves}) {{\n"
        ));
        source.push_str(&format!(
            "        wws_row{unit} = wws_h{unit} / {chunks_times_two};\n"
        ));
        source.push_str(&format!(
            "        long wws_within{unit} = wws_h{unit} % {chunks_times_two};\n"
        ));
        source.push_str(&format!(
            "        long wws_chunk{unit} = wws_within{unit} / 2;\n"
        ));
        source.push_str(&format!(
            "        long wws_half{unit} = wws_within{unit} % 2;\n"
        ));
        source.push_str(&format!(
            "        wws_koff{unit} = wws_chunk{unit} * {half_width_x2} + wws_half{unit} * {half_width};\n",
            half_width_x2 = half_width * 2,
        ));
        source.push_str(&format!(
            "        long wws_feat{unit} = row_tile * {block_m} + wws_row{unit};\n"
        ));
        source.push_str(&format!(
            "        if (wws_feat{unit} < feature_extent) {{\n"
        ));
        source.push_str(&format!("            wws_active{unit} = true;\n"));
        source.push_str(&format!(
            "            long wws_base{unit} = u.operand_base[{weight}] + wws_feat{unit} * u.operand_strides[{weight}][{feature_axis}] + wws_koff{unit};\n"
        ));
        source.push_str(&format!(
            "            wws_blk{unit} = in{weight} + (wws_base{unit} / {block_elements}) * {block_bytes};\n"
        ));
        source.push_str(&format!(
            "            wws_slot{unit} = (uint)(wws_base{unit} % {block_elements});\n"
        ));
        source.push_str("        } else {\n");
        let zero_fill_index = format!("wws_row{unit} * {block_k} + wws_koff{unit} + z");
        source.push_str(&format!(
            "            for (long z = 0; z < {half_width}; ++z) {{ weight_tile[{zero_fill_index}] = 0.0h; }}\n"
        ));
        source.push_str("        }\n");
        source.push_str("    }\n");
    }
}

/// `TILED_GEMM_NSG * SIMD_WIDTH` as the exact `u64` literal
/// [`push_wide_weight_stage_setup`]'s own `h_offset` needs -- avoids a
/// second computed-vs-hard-coded copy of [`push_tiled_gemm_body`]'s own
/// `block_threads` local (both read the identical two constants).
#[cfg(feature = "metal-tiled-gemm")]
const fn block_threads_const() -> u64 {
    (TILED_GEMM_NSG as u64) * SIMD_WIDTH
}

/// The `(pointer offset, stride)` pair a `simdgroup_load` call needs to read
/// one `TILE_DIM x TILE_DIM` fragment, shared by [`push_tiled_gemm_body`]
/// and [`push_dense_batched_gemm_body`]'s identical `a_frag`/`b_frag` load
/// sites: the fragment's 8 rows are `block_k` elements apart in the
/// row-major tile (`stride = block_k`).
#[cfg(feature = "metal-tiled-gemm")]
fn fragment_load_offset_stride(
    half_expr: &str,
    unroll_var: &str,
    unroll_count: u64,
    block_k: u64,
) -> (String, String) {
    (
        format!("({half_expr} * {unroll_count} + {unroll_var}) * 8 * {block_k} + sub_k * 8"),
        block_k.to_string(),
    )
}

/// `PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE`'s decode half, emitted inside the
/// `k0` reduction loop: reads `half_width` elements at each unit's current
/// `(wws_blk, wws_slot)`, writes them into `weight_tile` via `half4` vector
/// stores (`half_width` is a multiple of 8, per `wide_weight_stage_eligible`'s
/// own admission -- `koff`'s two summands are each a multiple of `half_width`,
/// so every store offset this function computes is a multiple of 4, the
/// `half4` alignment `simdgroup_load`'s own downstream read needs unchanged),
/// then advances the pointer/slot by exactly ONE addition and, in the case
/// where the codec's block is narrower than a K-step, one compile-time-
/// constant-vs-runtime-counter compare -- never a division, unlike the
/// non-wide path's `slot_off / block_elements` recomputed fresh every `k0`
/// step (see [`crate::identity::MetalOnlyExtras::tiled_gemm_wide_weight_
/// stage`]'s own doc for the two cases this splits on).
#[cfg(feature = "metal-tiled-gemm")]
#[allow(clippy::too_many_arguments)]
fn push_wide_weight_stage_body(
    source: &mut String,
    codec: Codec,
    block_k: u64,
    num_chunks: u64,
    half_width: u64,
    half_units: u64,
    chunk_width: u64,
    block_elements: u64,
    block_bytes: u64,
) {
    let runs = half_width / 8;
    for unit in 0..half_units {
        source.push_str(&format!("        if (wws_active{unit}) {{\n"));
        match codec {
            Codec::Q4_0 => {
                source.push_str(&format!(
                    "            float q4_0_d = q4_0_block_scale(wws_blk{unit});\n"
                ));
                for run_index in 0..runs {
                    let run_offset = run_index * 8;
                    source.push_str("            {\n");
                    source.push_str("                float levels[8];\n");
                    source.push_str(&format!(
                        "                q4_0_run8_wide(wws_blk{unit}, wws_slot{unit} + {run_offset}u, levels);\n"
                    ));
                    push_wide_weight_stage_half4_store(
                        source, unit, run_offset, block_k, "q4_0_d", None,
                    );
                    source.push_str("            }\n");
                }
            }
            _ => {
                source.push_str(&format!(
                    "            q4k_header wws_hdr{unit} = q4k_header_for(wws_blk{unit}, wws_slot{unit});\n"
                ));
                for run_index in 0..runs {
                    let run_offset = run_index * 8;
                    source.push_str("            {\n");
                    source.push_str("                float levels[8];\n");
                    source.push_str(&format!(
                        "                q4k_run8(wws_blk{unit}, wws_slot{unit} + {run_offset}u, levels);\n"
                    ));
                    push_wide_weight_stage_half4_store(
                        source,
                        unit,
                        run_offset,
                        block_k,
                        &format!("wws_hdr{unit}.scale"),
                        Some(&format!("wws_hdr{unit}.minimum")),
                    );
                    source.push_str("            }\n");
                }
            }
        }
        push_wide_weight_stage_advance(
            source,
            unit,
            block_k,
            num_chunks,
            chunk_width,
            block_elements,
            block_bytes,
        );
        source.push_str("        }\n");
    }
}

/// The pointer/slot advance every schedule of the wide weight stage shares:
/// exactly one addition per K-step (plus one compare in the narrow-block
/// case), never a division.
///
/// Case A: this unit's chunk is exactly one codec block wide
/// (`chunk_width == block_elements`) -- the whole `block_k`-wide K-step
/// advances the row window by `num_chunks` whole blocks, and a unit's own
/// slot-within-its-block (`0` for the low half, `half_width` for the high half)
/// never changes across `k0` steps: the window shift always lands this unit
/// back at the SAME relative position in a new block, never a fraction of one.
/// Case B: the codec's block is wider than one K-step (`chunk_width ==
/// block_k`, `num_chunks == 1`) -- `wws_slot` accumulates by `block_k` each
/// step. A unit's own starting slot is NOT necessarily `0` (the `half == 1`
/// unit of a chunk starts at `half_width`, not the block origin), so wrapping
/// compares `>=` and SUBTRACTS `block_elements` rather than resetting to `0` --
/// `wws_slot < block_elements` is a loop invariant and `block_k <=
/// block_elements` in this branch, so `wws_slot + block_k` never exceeds
/// `2 * block_elements` and one subtraction always suffices.
#[cfg(feature = "metal-tiled-gemm")]
fn push_wide_weight_stage_advance(
    source: &mut String,
    unit: u64,
    block_k: u64,
    num_chunks: u64,
    chunk_width: u64,
    block_elements: u64,
    block_bytes: u64,
) {
    if chunk_width == block_elements {
        source.push_str(&format!(
            "            wws_blk{unit} += {num_chunks}u * {block_bytes};\n"
        ));
    } else {
        source.push_str(&format!("            wws_slot{unit} += {block_k}u;\n"));
        source.push_str(&format!(
            "            if (wws_slot{unit} >= {block_elements}u) {{ wws_slot{unit} -= {block_elements}u; wws_blk{unit} += {block_bytes}u; }}\n"
        ));
    }
}

/// Writes 8 already-decoded levels as two `half4` vector stores into
/// `weight_tile` -- `minimum_expr` is `None` for `Q4_0` (fixed midpoint 8.0,
/// no separate minimum term) and `Some(..)` for the K-quant header shape
/// (`scale * level - minimum`), the same two value expressions
/// [`push_tiled_gemm_body`]'s non-wide arms already compute per element,
/// just grouped 4-at-a-time into one vector store instead of 8 separate
/// scalar `half` writes.
#[cfg(feature = "metal-tiled-gemm")]
fn push_wide_weight_stage_half4_store(
    source: &mut String,
    unit: u64,
    run_offset: u64,
    block_k: u64,
    scale_expr: &str,
    minimum_expr: Option<&str>,
) {
    for half in 0..2 {
        let base = run_offset + half * 4;
        let mut components = Vec::with_capacity(4);
        for lane in 0..4 {
            let index = half * 4 + lane;
            let value = match minimum_expr {
                Some(minimum) => {
                    format!("(half)({scale_expr} * levels[{index}] - {minimum})")
                }
                None => format!("(half)((levels[{index}] - 8.0f) * {scale_expr})"),
            };
            components.push(value);
        }
        let target = format!("wws_row{unit} * {block_k} + wws_koff{unit} + {base}");
        source.push_str(&format!(
            "                *(threadgroup half4 *)&weight_tile[{target}] = half4({}, {}, {}, {});\n",
            components[0], components[1], components[2], components[3],
        ));
    }
}

/// Lever 2's fast arm: [`ggml`'s own `kernel_mul_mm`
/// (`mul_mm.metal:317-357`)]'s technique -- `simdgroup_store` the
/// accumulators straight to `device` memory for a tile that is fully inside
/// both extents, skipping the `out_tile` threadgroup restage and the
/// per-element bounds-checked copy-out loop entirely. Opens the `if
/// (direct_store_interior) {` brace; the caller closes it (and appends the
/// `} else { <restage> }` tail) since the restage path is shared with the
/// direct-store-inactive case verbatim.
#[cfg(feature = "metal-tiled-gemm")]
#[allow(clippy::too_many_arguments)]
fn push_tiled_gemm_direct_store_arm(
    source: &mut String,
    rank: usize,
    batch_axes: &[u16],
    feature_axis: u16,
    token_axis: u16,
    block_m: u64,
    block_n: u64,
    thread_mat_m: u64,
    thread_mat_n: u64,
    acc_token_major: bool,
) {
    let rank_len = rank.max(1);
    let store_tail = if acc_token_major { "" } else { ", ulong2(0), true" };
    // A tile is safe to store directly only when NO row or column of it
    // falls outside the real extents (no mask needed) AND the output's own
    // FEATURE axis is unit-stride: `acc`'s fragment is [row=feature,
    // col=token] by construction (`a_frag` is the weight/feature operand,
    // `b_frag` the token operand loaded `transpose_matrix=true` -- this
    // function's own module doc), so a device write with `transpose=true`
    // stores `acc^T` into a row-major destination whose ROW dimension
    // (`elements_per_row`-scaled) is TOKEN and whose COLUMN dimension
    // (contiguous) is FEATURE -- exactly ggml's own `kernel_mul_mm` dst
    // convention (`mul_mm.metal:320-325`: `ne0` (feature/M) is the
    // contiguous axis, `ne1` (token/N) is `elements_per_row`-scaled), which
    // is also this crate's own real production output layout
    // (`real_shaped_tiled_gemm_op`'s `out_map` places the feature axis
    // innermost). `u.out_strides` is a RUNTIME uniform, never baked into
    // source text, so this has to be a runtime branch, not a compile-time
    // admission.
    source.push_str(&format!(
        "    bool direct_store_interior = (row_tile * {block_m} + {block_m} <= feature_extent) && (col_tile * {block_n} + {block_n} <= token_extent) && (u.out_strides[{feature_axis}] == 1);\n"
    ));
    source.push_str("    if (direct_store_interior) {\n");
    source.push_str(&format!("        long direct_coord[{rank_len}];\n"));
    source.push_str(&format!(
        "        for (int d = 0; d < {rank}; ++d) {{ direct_coord[d] = 0; }}\n"
    ));
    source.push_str(&format!(
        "        direct_coord[{feature_axis}] = row_tile * {block_m};\n"
    ));
    source.push_str(&format!(
        "        direct_coord[{token_axis}] = col_tile * {block_n};\n"
    ));
    for &axis in batch_axes {
        source.push_str(&format!(
            "        direct_coord[{axis}] = dense_batch_coord_{axis};\n"
        ));
    }
    source.push_str("        long direct_offset = u.out_base;\n");
    for dim in 0..rank {
        source.push_str(&format!(
            "        direct_offset += direct_coord[{dim}] * u.out_strides[{dim}];\n"
        ));
    }
    source.push_str(&format!(
        "        for (int i = 0; i < {thread_mat_m}; ++i) {{\n"
    ));
    source.push_str(&format!(
        "            for (int j = 0; j < {thread_mat_n}; ++j) {{\n"
    ));
    source.push_str(&format!(
        "                simdgroup_store(acc[i * {thread_mat_n} + j], out + direct_offset + (row_half * {thread_mat_m} + i) * 8 * u.out_strides[{feature_axis}] + (col_half * {thread_mat_n} + j) * 8 * u.out_strides[{token_axis}], u.out_strides[{token_axis}]{store_tail});\n"
    ));
    source.push_str("            }\n");
    source.push_str("        }\n");
}

/// The pre-existing threadgroup-restage write tail, extracted verbatim so
/// [`push_tiled_gemm_body`]/[`push_dense_batched_gemm_body`] can share it
/// between the direct-store-inactive case and the direct-store-active
/// boundary-tile fallback -- see [`push_tiled_gemm_direct_store_arm`]'s own
/// doc for why a boundary tile (or a non-`float` output) still needs this
/// path even with lever 2 on.
#[cfg(feature = "metal-tiled-gemm")]
#[allow(clippy::too_many_arguments)]
fn push_tiled_gemm_restage_writeback(
    source: &mut String,
    rank: usize,
    output_axes: &[u16],
    batch_axes: &[u16],
    feature_axis: u16,
    token_axis: u16,
    block_m: u64,
    block_n: u64,
    block_threads: u64,
    thread_mat_m: u64,
    thread_mat_n: u64,
    out_tile_elems: u64,
    element_type: &str,
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
    acc_token_major: bool,
) {
    let rank_len = rank.max(1);
    let store_tail = if acc_token_major { ", ulong2(0), true" } else { "" };
    source.push_str(&format!(
        "    for (int i = 0; i < {thread_mat_m}; ++i) {{\n"
    ));
    source.push_str(&format!(
        "        for (int j = 0; j < {thread_mat_n}; ++j) {{\n"
    ));
    source.push_str(&format!(
        "            simdgroup_store(acc[i * {thread_mat_n} + j], out_tile + (row_half * {thread_mat_m} + i) * 8 * {block_n} + (col_half * {thread_mat_n} + j) * 8, {block_n}{store_tail});\n"
    ));
    source.push_str("        }\n");
    source.push_str("    }\n");
    source.push_str("    threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    // `coord[]` is allocated and its provably loop-invariant slots (every
    // dim NOT one of feature/token/batch stays 0 for the op's whole rank --
    // it's the reduced axis, never an output axis at all) are set ONCE here,
    // outside the per-element loop below, instead of a fresh zero + rewrite
    // on every one of up to `out_tile_elems / block_threads` iterations per
    // thread -- ROW C4.11's own basic-block read found that re-zero
    // (a 3-store unrolled loop per iteration in the measured 3-axis case) is
    // 21 of the 43 per-element instructions, none of it hoistable by the
    // optimizer because the old code re-wrote the SAME alloca slots fresh
    // every iteration. `out_offset_base` folds in the batch axes' own
    // contribution the same way: their coordinate is a per-thread constant
    // (`dense_batch_coord_{axis}`, set once per z-tile), so summing
    // `coord[axis] * u.out_strides[axis]` for them here and reusing the
    // running total inside the loop removes those terms (and the dead
    // multiply against the never-written reduced-axis slots) from the
    // per-element address computation entirely -- same total value as the
    // old `for dim in 0..rank { out_offset += coord[dim] * u.out_strides[dim]; }`
    // sum, just factored so the loop-invariant part is computed once.
    source.push_str(&format!("    long coord[{rank_len}];\n"));
    source.push_str(&format!(
        "    for (int d = 0; d < {rank}; ++d) {{ coord[d] = 0; }}\n"
    ));
    for &axis in batch_axes {
        source.push_str(&format!(
            "    coord[{axis}] = dense_batch_coord_{axis};\n"
        ));
    }
    source.push_str("    long out_offset_base = u.out_base;\n");
    for &axis in batch_axes {
        source.push_str(&format!(
            "    out_offset_base += coord[{axis}] * u.out_strides[{axis}];\n"
        ));
    }
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
    source.push_str(&format!("            coord[{feature_axis}] = o_feat;\n"));
    source.push_str(&format!("            coord[{token_axis}] = o_tok;\n"));
    source.push_str(&format!(
        "            long out_offset = out_offset_base + o_feat * u.out_strides[{feature_axis}] + o_tok * u.out_strides[{token_axis}];\n"
    ));
    // reuse the SAME fused-epilogue emitter every other reduce renderer
    // funnels its write through (`push_reduce_epilogue_write`'s own doc) --
    // `coord` reads this function's own per-output-element `coord[rank]`
    // array at the axis id `output_axes[dim]` maps to, the identical
    // addressing the `out_offset` computation just above already uses, so a
    // fused GeGLU-shaped (or any other plain, non-broadcast) epilogue reads
    // its extra operands at the exact element `push_tiled_gemm_body` is
    // about to write. `classify_tiled_gemm`'s own `BroadcastEpilogueNotSupported`
    // gate is what keeps a broadcast-reduce epilogue -- the one shape this
    // write tail cannot express -- from ever reaching this function at all.
    let accumulator_expr = format!("({element_type})out_tile[idx]");
    push_reduce_epilogue_write(
        source,
        epilogue_body,
        epilogue_operands,
        output_axes.len(),
        element_type,
        "            ",
        |dim| format!("coord[{}]", output_axes[dim]),
        &accumulator_expr,
        "out_offset",
    );
    source.push_str("        }\n");
    source.push_str("    }\n");
}

/// Never actually invoked: [`classify_tiled_gemm`]'s own `#[cfg(not(feature
/// = "metal-tiled-gemm"))]` arm always returns `None`, so no caller ever
/// holds a `&TiledGemmBlock` to pass here without the feature -- this stub
/// exists only so [`push_cooperative_reduce_body`]'s `if let Some(block) =
/// tiled_gemm_block(...)` arm still type-checks in that build.
#[cfg(not(feature = "metal-tiled-gemm"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn push_tiled_gemm_body(
    source: &mut String,
    node: NodeId,
    output_axes: &[u16],
    rank: usize,
    block: &TiledGemmBlock,
    element_type: &str,
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
    metal: &crate::identity::MetalOnlyExtras,
) -> Result<(), EmitError> {
    let _ = (
        source,
        output_axes,
        rank,
        block,
        element_type,
        epilogue_body,
        epilogue_operands,
        metal,
    );
    Err(EmitError::TiledGemmFeatureDisabled { node })
}

/// The dense (unquantized) counterpart to [`push_tiled_gemm_body`]: BOTH
/// operand tiles load straight `float` reads (no codec decode arm, so
/// `weight_tile` is `float` here, not `half`, and both `simdgroup_load`s
/// build `simdgroup_float8x8` fragments) -- the sliding-pattern family's GQA attention score
/// (`Q.K^T`) and value (`P.V`) folds, ported behind `PROXIMA_TILED_GEMM_DENSE=1`
/// (`classify_dense_batched_gemm`'s own doc).
///
/// Adds ONE thing [`push_tiled_gemm_body`] has no shape for: `block.batch_axes`,
/// dispatched over `[[thread_position_in_grid]]`'s `z` component
/// ([`GridSpec::depth`], [`dense_batched_gemm_depth`]) rather than folded
/// into the `x` tile grid the way `token_axes`/`feature_axes` are --
/// `crate::metal::dispatch`'s own `MTLSize { depth: grid.depth, .. }` with a
/// `threadgroup` depth of `1` means `thread_position_in_grid.z` already IS
/// the flat batch index, one distinct value per z-slice, matching
/// `splice_round_batched_reduce_base_table`'s own `round_gid.z` precedent.
/// Metal rejects a signature mixing a scalar and a vector
/// `thread_position_in_grid` attribute (that splice's own doc), so this
/// widens the EXISTING scalar `uint gid` parameter [`kernel_signature`]
/// already emitted (by the time this function runs, `render_reduce` has
/// already called it) to `uint3` in place, exactly that splice's technique,
/// just applied inline rather than as a separate post-emit pass -- there is
/// no already-rendered kernel to reuse here, so a second function to splice
/// into would only relocate these lines, not remove them.
///
/// Per-axis batch offsets are computed from the SAME generic
/// `u.operand_strides[operand][axis]`/`u.out_strides[axis]` fields every
/// other reduce kernel already reads (no new `Uniforms` field, no runtime
/// table): a broadcast operand's stride is already `0` in that buffer, so
/// adding `coord * stride` is correct whether the axis broadcasts or not --
/// this is what makes ggml's own `r2`/`r3` repeat-ratio recomputation
/// (`kernel_mul_mm`'s own `i12 = im % ne12` shape) unnecessary here: this
/// crate's layout model already expresses a broadcast as a literal zero
/// stride, so the ratio is folded into the stride itself, not recomputed at
/// dispatch time.
#[cfg(feature = "metal-tiled-gemm")]
#[allow(clippy::too_many_arguments)]
pub(super) fn push_dense_batched_gemm_body(
    source: &mut String,
    node: NodeId,
    output_axes: &[u16],
    rank: usize,
    block: &DenseBatchedGemmBlock,
    element_type: &str,
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
    metal: &crate::identity::MetalOnlyExtras,
) -> Result<(), EmitError> {
    // `metal.tiled_gemm_slim_tgmem` (phase 2) IS wired here -- see
    // `push_tiled_gemm_body`'s own doc.
    let DenseBatchedGemmBlock {
        weight,
        other,
        reduce_dim,
        ref token_axes,
        ref feature_axes,
        ref batch_axes,
    } = *block;
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

    let block_m = crate::sized::TILED_GEMM_BLOCK_M;
    let block_n = crate::sized::TILED_GEMM_BLOCK_N;
    let block_k = crate::sized::TILED_GEMM_BLOCK_K;
    let block_threads = (TILED_GEMM_NSG as u64) * SIMD_WIDTH;
    let thread_mat_m = block_m / (TILE_DIM as u64 * 2);
    let thread_mat_n = block_n / (TILE_DIM as u64 * 2);
    let mc_count = thread_mat_m * thread_mat_n;
    let sub_k_steps = block_k / TILE_DIM as u64;
    let weight_tile_elems = block_m * block_k;
    let act_tile_elems = block_n * block_k;
    let out_tile_elems = block_m * block_n;

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

    // `PROXIMA_TILED_GEMM_GRID2D` (see `docs/model-interop/discipline.md`
    // ROW C4.19, `MetalOnlyExtras::tiled_gemm_grid2d`'s own doc): the SAME 2D-attribute
    // swap `push_tiled_gemm_body` performs, with one simplification unique to
    // the dense-batched body -- `tgpig.z` already IS the batch index (the
    // grid's own third axis under `dispatchThreadgroups`), so there is no
    // separate scalar-`gid`-to-`uint3` widening step to bridge through the
    // way the non-`grid2d` form below needs.
    let grid2d_active = metal.tiled_gemm_grid2d;
    let scalar_gid = "uint gid [[thread_position_in_grid]]";
    if grid2d_active {
        let vector_attrs = "uint3 tgpig [[threadgroup_position_in_grid]],\n    \
            ushort tiitg [[thread_index_in_threadgroup]],\n    \
            ushort sgitg [[simdgroup_index_in_threadgroup]]";
        let gid_offset = source.find(scalar_gid).ok_or(EmitError::RenderKindMismatch {
            node,
            expected: "scalar thread_position_in_grid parameter",
            found: "missing",
        })?;
        source.replace_range(gid_offset..gid_offset + scalar_gid.len(), vector_attrs);
        source.push_str("    long dense_batch_index = (long)tgpig.z;\n");
    } else {
        // Widen the scalar `gid` `kernel_signature` already emitted to a
        // `uint3` in place -- see this function's own doc for why splicing an
        // already-rendered body is not the shape here.
        let vector_gid = "uint3 dense_batch_gid [[thread_position_in_grid]]";
        let gid_offset = source
            .find(scalar_gid)
            .ok_or(EmitError::RenderKindMismatch {
                node,
                expected: "scalar thread_position_in_grid parameter",
                found: "missing",
            })?;
        source.replace_range(gid_offset..gid_offset + scalar_gid.len(), vector_gid);
        source.push_str("    long gid = (long)dense_batch_gid.x;\n");
        source.push_str("    long dense_batch_index = (long)dense_batch_gid.z;\n");
    }
    source.push_str("    long dense_weight_batch_base = 0;\n");
    source.push_str("    long dense_other_batch_base = 0;\n");
    if !batch_axes.is_empty() {
        source.push_str("    long dense_batch_remaining = dense_batch_index;\n");
        for &axis in batch_axes.iter().rev() {
            let Some(position) = output_axes.iter().position(|&candidate| candidate == axis) else {
                return Err(EmitError::AxisNotInOutputAxes { node, axis });
            };
            source.push_str(&format!(
                "    long dense_batch_coord_{axis} = dense_batch_remaining % u.output_extents[{position}];\n"
            ));
            source.push_str(&format!(
                "    dense_batch_remaining /= u.output_extents[{position}];\n"
            ));
            source.push_str(&format!(
                "    dense_weight_batch_base += dense_batch_coord_{axis} * u.operand_strides[{weight}][{axis}];\n"
            ));
            source.push_str(&format!(
                "    dense_other_batch_base += dense_batch_coord_{axis} * u.operand_strides[{other}][{axis}];\n"
            ));
        }
    }

    source.push_str(&format!(
        "    long feature_extent = {};\n",
        group_extent_expr(feature_axes)?
    ));
    source.push_str(&format!(
        "    long token_extent = {};\n",
        group_extent_expr(token_axes)?
    ));
    if grid2d_active {
        source.push_str("    long row_tile = (long)tgpig.y;\n");
        source.push_str("    long col_tile = (long)tgpig.x;\n");
        source.push_str("    long row_half = sgitg & 1;\n");
        source.push_str("    long col_half = sgitg >> 1;\n");
    } else {
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
    }
    // see `push_tiled_gemm_body`'s own doc for why this aliasing is safe.
    let dense_element_type = "float";
    let dense_value_cast = "";
    let dense_zero_literal = "0.0f";
    let dense_simdgroup_type = "simdgroup_float8x8";
    let slim_tgmem_active = metal.tiled_gemm_slim_tgmem;
    let dense_element_bytes = 4u64;
    let weight_tile_bytes = weight_tile_elems * dense_element_bytes;
    let act_tile_bytes = act_tile_elems * dense_element_bytes;
    let out_tile_bytes = out_tile_elems * 4;
    let shared_bytes = (weight_tile_bytes + act_tile_bytes).max(out_tile_bytes);
    if slim_tgmem_active {
        source.push_str(&format!(
            "    threadgroup uchar tg_shared[{shared_bytes}];\n"
        ));
        source.push_str(&format!(
            "    threadgroup {dense_element_type} *weight_tile = (threadgroup {dense_element_type} *)tg_shared;\n"
        ));
        source.push_str(&format!(
            "    threadgroup {dense_element_type} *act_tile = (threadgroup {dense_element_type} *)(tg_shared + {weight_tile_bytes});\n"
        ));
    } else {
        source.push_str(&format!(
            "    threadgroup {dense_element_type} weight_tile[{weight_tile_elems}];\n"
        ));
        source.push_str(&format!(
            "    threadgroup {dense_element_type} act_tile[{act_tile_elems}];\n"
        ));
    }
    source.push_str(&format!("    simdgroup_float8x8 acc[{mc_count}];\n"));
    source.push_str(&format!(
        "    for (int i = 0; i < {mc_count}; ++i) {{ acc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f); }}\n"
    ));
    let k0_counter_type = if grid2d_active { "int" } else { "long" };
    source.push_str(&format!(
        "    for ({k0_counter_type} k0 = 0; k0 < u.reduction_total; k0 += {block_k}) {{\n"
    ));
    source.push_str(&format!(
        "        for (long w_row = tiitg; w_row < {block_m}; w_row += {block_threads}) {{\n"
    ));
    source.push_str(&format!(
        "            long w_feat = row_tile * {block_m} + w_row;\n"
    ));
    source.push_str("            if (w_feat < feature_extent) {\n");
    source.push_str(&format!(
        "                long row_base = u.operand_base[{weight}] + dense_weight_batch_base + w_feat * u.operand_strides[{weight}][{feature_axis}] + k0 * u.operand_strides[{weight}][{reduce_dim}];\n"
    ));
    source.push_str(&format!(
        "                for (long w_k = 0; w_k < {block_k}; ++w_k) {{\n"
    ));
    // Dense reduce extents carry no super-block-multiple guarantee the way
    // `PackedRowBlock`'s own `Q4K_BLOCK_ELEMENTS` gate gives the packed path
    // (`push_tiled_gemm_body`'s own doc names that guarantee explicitly) --
    // a ragged `u.reduction_total` (not a whole multiple of `block_k`) would
    // otherwise read `w_k` elements past the operand's real reduce extent on
    // the last `k0` step. Zero-fill out-of-range k, matching the boundary-
    // tile mask this same function already applies on `feature_extent`/
    // `token_extent`.
    source.push_str("                    if (k0 + w_k < u.reduction_total) {\n");
    source.push_str(&format!(
        "                        long w_off = row_base + w_k * u.operand_strides[{weight}][{reduce_dim}];\n"
    ));
    let dense_weight_index = format!("w_row * {block_k} + w_k");
    source.push_str(&format!(
        "                        weight_tile[{dense_weight_index}] = {dense_value_cast}{};\n",
        operand_read(weight, "w_off", None)
    ));
    source.push_str("                    } else {\n");
    source.push_str(&format!(
        "                        weight_tile[{dense_weight_index}] = {dense_zero_literal};\n"
    ));
    source.push_str("                    }\n");
    source.push_str("                }\n");
    source.push_str("            } else {\n");
    let dense_weight_fill_index = format!("w_row * {block_k} + fill_k");
    source.push_str(&format!(
        "                for (long fill_k = 0; fill_k < {block_k}; ++fill_k) {{ weight_tile[{dense_weight_fill_index}] = {dense_zero_literal}; }}\n"
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
    // Same ragged-K guard as the weight tile load above -- `a_k_global` can
    // reach `u.reduction_total` on the last `k0` step when the dense reduce
    // extent is not a whole multiple of `block_k`.
    source.push_str("            if (a_tok < token_extent && a_k_global < u.reduction_total) {\n");
    source.push_str(&format!(
        "                long aoff = u.operand_base[{other}] + dense_other_batch_base + a_tok * u.operand_strides[{other}][{token_axis}] + a_k_global * u.operand_strides[{other}][{reduce_dim}];\n"
    ));
    source.push_str(&format!(
        "                a_value = {};\n",
        operand_read(other, "aoff", None)
    ));
    source.push_str("            }\n");
    let dense_act_index = format!("a_col * {block_k} + a_k");
    source.push_str(&format!(
        "            act_tile[{dense_act_index}] = {dense_value_cast}a_value;\n"
    ));
    source.push_str("        }\n");
    source.push_str("        threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    source.push_str(&format!(
        "        for (int sub_k = 0; sub_k < {sub_k_steps}; ++sub_k) {{\n"
    ));
    source.push_str(&format!(
        "            {dense_simdgroup_type} a_frag[{thread_mat_m}];\n"
    ));
    source.push_str(&format!(
        "            for (int i = 0; i < {thread_mat_m}; ++i) {{\n"
    ));
    let (dense_a_frag_offset, dense_a_frag_stride) =
        fragment_load_offset_stride("row_half", "i", thread_mat_m, block_k);
    source.push_str(&format!(
        "                simdgroup_load(a_frag[i], weight_tile + {dense_a_frag_offset}, {dense_a_frag_stride});\n"
    ));
    source.push_str("            }\n");
    source.push_str("            simdgroup_barrier(mem_flags::mem_none);\n");
    source.push_str(&format!(
        "            {dense_simdgroup_type} b_frag[{thread_mat_n}];\n"
    ));
    source.push_str(&format!(
        "            for (int j = 0; j < {thread_mat_n}; ++j) {{\n"
    ));
    let (dense_b_frag_offset, dense_b_frag_stride) =
        fragment_load_offset_stride("col_half", "j", thread_mat_n, block_k);
    source.push_str(&format!(
        "                simdgroup_load(b_frag[j], act_tile + {dense_b_frag_offset}, {dense_b_frag_stride}, ulong2(0), true);\n"
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
    if slim_tgmem_active {
        source.push_str("    threadgroup float *out_tile = (threadgroup float *)tg_shared;\n");
    } else {
        source.push_str(&format!(
            "    threadgroup float out_tile[{out_tile_elems}];\n"
        ));
    }
    // batch coords go through `coord[]`/`direct_coord[]`, not a separate
    // out-base term: the fused epilogue reads its extra operands at
    // `coord[..]`, so a per-head epilogue operand would otherwise read head
    // 0 for every head. See `push_tiled_gemm_body`'s own doc for why
    // `element_type == "float"` gates the direct-store arm.
    let direct_store_eligible = metal.tiled_gemm_direct_store && element_type == "float";
    if direct_store_eligible {
        push_tiled_gemm_direct_store_arm(
            source,
            rank,
            batch_axes,
            feature_axis,
            token_axis,
            block_m,
            block_n,
            thread_mat_m,
            thread_mat_n,
            false,
        );
        source.push_str("    } else {\n");
    }
    push_tiled_gemm_restage_writeback(
        source,
        rank,
        output_axes,
        batch_axes,
        feature_axis,
        token_axis,
        block_m,
        block_n,
        block_threads,
        thread_mat_m,
        thread_mat_n,
        out_tile_elems,
        element_type,
        epilogue_body,
        epilogue_operands,
        false,
    );
    if direct_store_eligible {
        source.push_str("    }\n");
    }
    Ok(())
}

/// Never actually invoked: [`classify_dense_batched_gemm`]'s own `#[cfg(not(feature
/// = "metal-tiled-gemm"))]` arm always returns `Err`, so no caller ever holds
/// a `&DenseBatchedGemmBlock` to pass here without the feature -- mirrors
/// [`push_tiled_gemm_body`]'s own non-feature stub.
#[cfg(not(feature = "metal-tiled-gemm"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn push_dense_batched_gemm_body(
    source: &mut String,
    node: NodeId,
    output_axes: &[u16],
    rank: usize,
    block: &DenseBatchedGemmBlock,
    element_type: &str,
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
    metal: &crate::identity::MetalOnlyExtras,
) -> Result<(), EmitError> {
    let _ = (
        source,
        output_axes,
        rank,
        block,
        element_type,
        epilogue_body,
        epilogue_operands,
        metal,
    );
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
    // Must agree with `grid_threads`'s own `CachedSoftmaxWeights` arm: one
    // threadgroup per attention row, `width` lanes wide -- the same
    // cooperative-reduce width the surviving 152/157/162 production folds
    // use for this same `cached_key_rows` bucket (`BoundOpKind::
    // CachedSoftmaxWeights::cached_key_rows`'s own doc).
    if let BoundOpKind::CachedSoftmaxWeights {
        cached_key_rows, ..
    } = &resolved.kind
    {
        return Some(wide_cooperative_reduce_width(*cached_key_rows));
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
        // Same form classification as `grid_threads`'s own `CachedAttention`
        // arm -- `two_range_cached_bound` never widens to the compiled cap,
        // since its `context_length` is already the bucket-padded
        // compile-time value. The decode split form is one query head per
        // threadgroup, so its width carries no `query_groups` factor.
        let context_length = *cached_key_rows + *new_key_rows;
        let (dynamic_cached_len, chunks) =
            match cached_attention_form(&resolved.kind, numeric_policy) {
                Some(CachedAttentionForm::SingleRangeDynamic { .. }) => {
                    (true, effective_context_chunk_cap(*query_groups, *head_dim))
                }
                #[cfg(feature = "metal-attn-split-decode")]
                Some(CachedAttentionForm::TwoRangeDecodeSplit { chunks, .. }) => {
                    return Some(chunks * SIMD_WIDTH);
                }
                #[cfg(feature = "metal-attn-split-rows")]
                Some(CachedAttentionForm::TwoRangeRowTiled { simdgroups, .. }) => {
                    return Some(simdgroups * SIMD_WIDTH);
                }
                Some(CachedAttentionForm::Static | CachedAttentionForm::TwoRangeCachedBound)
                | None => (
                    false,
                    context_chunks_for(context_length, *query_groups, *head_dim, numeric_policy),
                ),
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
        if dense_batched_gemm_block(resolved, quantized, *reduce_op, *init, output_axes).is_some() {
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
    Some(with_reduction_dims(resolved, output_axes, |reduce_dims| {
        cooperative_reduce_width(resolved, quantized, reduce_dims)
    }))
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
    // this Q4-only specialization cannot represent that ABI. Shared with
    // `emit_and_classify`'s three call sites -- one std-vs-no_std gate, not
    // four.
    if super::emit_and_classify::unsafe_metal_expert_sources_enabled() {
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
/// `extents`, so a caller with no real reduce `BoundOp` to point at
/// (`crate::metal::prepare_uniforms_pack`'s own `CachedSoftmaxWeights`
/// uniform packer, which bakes every shape constant at emit time instead of
/// packing a `Uniforms::reduction_total` field) computes the SAME width
/// production's real per-node reduce kernels do, so the two can never drift
/// onto different topologies for the same `reduction_total`.
/// `cooperative_reduce_width` itself calls straight through to this after
/// resolving its own `reduction_total` from `reduce_dims`.
#[cfg(feature = "metal-wide-cooperative-reduce")]
pub(crate) fn wide_cooperative_reduce_width(reduction_total: u64) -> u64 {
    let quarter = reduction_total.div_ceil(4).max(1);
    quarter
        .next_multiple_of(SIMD_WIDTH)
        .clamp(SIMD_WIDTH, crate::sized::WIDE_COOPERATIVE_REDUCE_MAX_WIDTH)
}

/// Feature off: always `SIMD_WIDTH`, matching [`cooperative_reduce_width`]'s
/// own feature-off arm -- see that function's doc.
#[cfg(not(feature = "metal-wide-cooperative-reduce"))]
pub(crate) fn wide_cooperative_reduce_width(_reduction_total: u64) -> u64 {
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
    metal: &MetalOnlyExtras,
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
        if is_expert_grouped(&block) {
            push_expert_grouped_gemm_body(
                source,
                resolved,
                output_axes,
                &block,
                element_type,
                epilogue_body,
                epilogue_operands,
            )?;
            return Ok(());
        }
        push_tiled_gemm_body(
            source,
            resolved.node,
            output_axes,
            rank,
            &block,
            element_type,
            epilogue_body,
            epilogue_operands,
            metal,
        )?;
        return Ok(());
    }

    // The dense (unquantized) counterpart, mutually exclusive with the arm
    // above by construction: `classify_dense_batched_gemm` requires BOTH
    // operands unquantized, `classify_tiled_gemm` requires exactly ONE
    // packed. Same preamble ownership as the arm above.
    if let Some(block) = dense_batched_gemm_block(resolved, quantized, reduce_op, init, output_axes) {
        push_dense_batched_gemm_body(
            source,
            resolved.node,
            output_axes,
            rank,
            &block,
            element_type,
            epilogue_body,
            epilogue_operands,
            metal,
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
            metal,
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
        // narrows the coordinate decomposition's divisors to `uint` only
        // when every output extent fits `u32`.
        if coord_index32_extents_fit(resolved, output_axes) && coord_index32_override() {
            source.push_str("    uint remaining = (uint)output_index;\n");
            for index in (0..output_rank).rev() {
                source.push_str(&format!(
                    "    output_coord[{index}] = remaining % (uint)u.output_extents[{index}]; \
                     remaining /= (uint)u.output_extents[{index}];\n"
                ));
            }
        } else {
            source.push_str("    long remaining = output_index;\n");
            for index in (0..output_rank).rev() {
                source.push_str(&format!(
                    "    output_coord[{index}] = remaining % u.output_extents[{index}]; \
                     remaining /= u.output_extents[{index}];\n"
                ));
            }
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

        if is_broadcast_epilogue {
            push_broadcast_epilogue_preload(
                source,
                rank,
                reduce_dims,
                width,
                epilogue_body,
                epilogue_operands,
                element_type,
            );
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
        let unroll = crate::sized::COOPERATIVE_REDUCE_UNROLL;
        let loads_are_plain =
            quantized.iter().all(Option::is_none) && gather_slots.iter().all(Option::is_none);
        if unroll > 1 && loads_are_plain {
            push_batched_accumulate_loop(
                source,
                resolved,
                reduce_op,
                width,
                unroll,
                element_type,
                operand_count,
            );
        } else {
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
            let value_expr =
                push_body_steps(source, resolved.element_body(), "        ", element_type);
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
        }
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
    let prefetch_unroll =
        broadcast_epilogue_prefetch_unroll(reduce_dims, epilogue_body, epilogue_operands);
    let mut element_body = String::new();
    let epi_value = if is_identity {
        reduced_expr.to_string()
    } else {
        if prefetch_unroll.is_none() {
            push_epilogue_scratch_and_invariant_reads(
                source,
                rank,
                reduce_dims,
                epilogue_operands,
                element_type,
            );
        }
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

    if let Some(unroll) = prefetch_unroll {
        let variant_operands: Vec<usize> = (0..epilogue_operand_count)
            .filter(|&index| !is_invariant_operand(index))
            .collect();
        push_prefetched_broadcast_write_loop(
            source,
            rank,
            reduce_dims[0],
            width,
            unroll,
            &variant_operands,
            &operand_aliases(epilogue_operands),
            &element_body,
            &epi_value,
            element_type,
        );
        return;
    }

    source.push_str(&format!(
        "    for (long r = (long)lane; r < u.reduction_total; r += {width}) {{\n"
    ));
    if reduce_rank == 1 {
        source.push_str(&format!("        full_coord[{}] = r;\n", reduce_dims[0]));
    } else if reduce_rank > 1 {
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
            &operand_aliases(epilogue_operands),
            rank,
            "        ",
            |dim| format!("full_coord[{dim}]"),
        );
        source.push_str(&element_body);
        source.push_str(&format!("        out[out_offset] = {epi_value};\n"));
    }
    source.push_str("    }\n");
}

// the slot loops index per-thread arrays, which only stay in registers when
// every trip is unrolled; the trip count is a literal, so full unroll is legal
const FULL_UNROLL: &str = "#pragma unroll";

/// The cooperative fold's accumulate loop with its loads issued ahead of its
/// folds: each trip loads `unroll` strided elements of every operand into
/// `batch`, then folds slot 0, 1, ... in that order -- the order the
/// one-element-per-trip loop folds them in, so the sum is bit-identical. The
/// per-trip loop paid one memory round trip per element (a `[1, 1536]` row
/// over 256 lanes is six); this pays one per `unroll`. Only for plain loads:
/// a packed codec decode or a gather fetch carries its own address chain.
fn push_batched_accumulate_loop(
    source: &mut String,
    resolved: &BoundOp,
    reduce_op: ScalarOp,
    width: u64,
    unroll: u64,
    element_type: &str,
    operand_count: usize,
) {
    let operand_slots = operand_count.max(1);
    source.push_str("    int total_r = (int)u.reduction_total;\n");
    source.push_str(&format!(
        "    for (int r = (int)lane; r < total_r; r += {}) {{\n",
        width * unroll
    ));
    source.push_str(&format!(
        "        {element_type} batch[{unroll}][{operand_slots}];\n"
    ));
    source.push_str(&format!(
        "        {FULL_UNROLL}\n        for (int slot = 0; slot < {unroll}; ++slot) {{\n"
    ));
    source.push_str(&format!(
        "            bool in_range = (r + slot * {width}) < total_r;\n"
    ));
    for index in 0..operand_count {
        let read = operand_read(index, &format!("walk{index} + slot * advance{index}"), None);
        source.push_str(&format!(
            "            batch[slot][{index}] = in_range ? {read} : ({element_type})0;\n"
        ));
    }
    source.push_str("        }\n");
    source.push_str(&format!(
        "        {FULL_UNROLL}\n        for (int slot = 0; slot < {unroll}; ++slot) {{\n"
    ));
    source.push_str(&format!(
        "            if ((r + slot * {width}) < total_r) {{\n"
    ));
    source.push_str(&format!(
        "                {element_type} scratch[{operand_slots}];\n"
    ));
    for index in 0..operand_count {
        source.push_str(&format!(
            "                scratch[{index}] = batch[slot][{index}];\n"
        ));
    }
    let value_expr = push_body_steps(
        source,
        resolved.element_body(),
        "                ",
        element_type,
    );
    source.push_str(&format!(
        "                {element_type} value = {value_expr};\n"
    ));
    let combine_expr = scalar_op_expr(reduce_op, &["accumulator", "value"]);
    source.push_str(&format!(
        "                accumulator = seeded ? {combine_expr} : value;\n"
    ));
    source.push_str("                seeded = true;\n");
    source.push_str("            }\n");
    source.push_str("        }\n");
    for index in 0..operand_count {
        source.push_str(&format!("        walk{index} += {unroll} * advance{index};\n"));
    }
    source.push_str("    }\n");
}

/// The slots per lane [`push_broadcast_epilogue_preload`] and
/// [`push_broadcast_epilogue_write`] both key on, or `None` when the write
/// keeps its plain per-trip loop: a multi-dim reduce, an identity epilogue
/// (nothing to read), a gathered epilogue operand (its address hangs on an
/// index buffer), or [`crate::sized::COOPERATIVE_REDUCE_UNROLL`] of 1. The
/// slot count is the unroll, cut so every per-element operand's slots fit
/// [`crate::sized::COOPERATIVE_REDUCE_PREFETCH_REGISTERS`] together: an
/// epilogue with eight such operands cannot hold eight slots each in
/// registers. One pure function of the same arguments, so the preload and the
/// write loop cannot disagree on which shape was emitted.
pub(super) fn broadcast_epilogue_prefetch_unroll(
    reduce_dims: &[u16],
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
) -> Option<u64> {
    let unroll = crate::sized::COOPERATIVE_REDUCE_UNROLL;
    let applies = unroll > 1
        && reduce_dims.len() == 1
        && !reduce_epilogue_is_identity(epilogue_body, epilogue_operands)
        && epilogue_operands.iter().all(|(_, _, lookup)| lookup.is_none());
    if !applies {
        return None;
    }
    let variant_count = prefetch_loaded_operands(reduce_dims, epilogue_operands).len() as u64;
    let budgeted = crate::sized::COOPERATIVE_REDUCE_PREFETCH_REGISTERS / variant_count.max(1);
    Some(unroll.min(budgeted.max(1)))
}

/// The per-element epilogue operands a prefetch holds in `epi_pre{index}`:
/// every operand that varies along the reduce axis and is the first mention of
/// its data. A later mention of the same data (see [`operand_aliases`]) reads
/// the first one's `epi_scratch` slot instead of holding a second copy.
fn prefetch_loaded_operands(
    reduce_dims: &[u16],
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
) -> Vec<usize> {
    let aliases = operand_aliases(epilogue_operands);
    (0..epilogue_operands.len())
        .filter(|&index| aliases[index] == index)
        .filter(|&index| !epilogue_operand_is_loop_invariant(epilogue_operands, reduce_dims, index))
        .collect()
}

fn push_epilogue_scratch_and_invariant_reads(
    source: &mut String,
    rank: usize,
    reduce_dims: &[u16],
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
    element_type: &str,
) {
    let operand_count = epilogue_operands.len();
    source.push_str(&format!(
        "    {element_type} epi_scratch[{}];\n",
        operand_count + 1
    ));
    push_epilogue_operand_reads(
        source,
        (0..operand_count)
            .filter(|&index| epilogue_operand_is_loop_invariant(epilogue_operands, reduce_dims, index)),
        &operand_aliases(epilogue_operands),
        rank,
        "    ",
        |dim| format!("full_coord[{dim}]"),
    );
}

// emitter helper threading one kernel's loop geometry; a struct would relocate
// the arguments, not remove them
#[allow(clippy::too_many_arguments)]
fn push_epilogue_prefetch_slots(
    source: &mut String,
    indent: &str,
    rank: usize,
    reduce_dim: u16,
    width: u64,
    unroll: u64,
    variant_operands: &[usize],
    first_element: &str,
    element_type: &str,
) {
    for &index in variant_operands {
        source.push_str(&format!(
            "{indent}{FULL_UNROLL}\n{indent}for (int slot = 0; slot < {unroll}; ++slot) {{\n"
        ));
        source.push_str(&format!(
            "{indent}    long epi_pre_r = {first_element} + (long)slot * {width};\n"
        ));
        source.push_str(&format!(
            "{indent}    long epi_pre_off{index} = u.epilogue_operand_base[{index}];\n"
        ));
        for dim in 0..rank {
            let coord = if dim == usize::from(reduce_dim) {
                "epi_pre_r".to_string()
            } else {
                format!("full_coord[{dim}]")
            };
            source.push_str(&format!(
                "{indent}    epi_pre_off{index} += {coord} * u.epilogue_operand_strides[{index}][{dim}];\n"
            ));
        }
        source.push_str(&format!(
            "{indent}    epi_pre{index}[slot] = epi_pre_r < u.reduction_total ? epi{index}[epi_pre_off{index}] : ({element_type})0;\n"
        ));
        source.push_str(&format!("{indent}}}\n"));
    }
}

/// Issues a broadcast epilogue's operand loads BEFORE the fold: none of them
/// depends on the folded scalar, so emitted ahead of the accumulate loop they
/// overlap its memory wait instead of starting a second round trip after the
/// threadgroup barrier (measured on the `[1, 1536]` rmsnorm: 8.6 us with the
/// loads after the barrier, 4.6 us with them hoisted and the loops batched).
/// Each lane keeps its first `unroll` strided elements of every per-element
/// operand in `epi_pre{index}`; [`push_broadcast_epilogue_write`] consumes
/// them. A no-op when [`broadcast_epilogue_prefetch_unroll`] is `None`.
#[allow(clippy::too_many_arguments)]
pub(super) fn push_broadcast_epilogue_preload(
    source: &mut String,
    rank: usize,
    reduce_dims: &[u16],
    width: u64,
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
    element_type: &str,
) {
    let Some(unroll) =
        broadcast_epilogue_prefetch_unroll(reduce_dims, epilogue_body, epilogue_operands)
    else {
        return;
    };
    push_epilogue_scratch_and_invariant_reads(
        source,
        rank,
        reduce_dims,
        epilogue_operands,
        element_type,
    );
    let loaded_operands = prefetch_loaded_operands(reduce_dims, epilogue_operands);
    for &index in &loaded_operands {
        source.push_str(&format!("    {element_type} epi_pre{index}[{unroll}];\n"));
    }
    push_epilogue_prefetch_slots(
        source,
        "    ",
        rank,
        reduce_dims[0],
        width,
        unroll,
        &loaded_operands,
        "(long)lane",
        element_type,
    );
}

#[allow(clippy::too_many_arguments)]
fn push_prefetched_broadcast_write_loop(
    source: &mut String,
    rank: usize,
    reduce_dim: u16,
    width: u64,
    unroll: u64,
    variant_operands: &[usize],
    aliases: &[usize],
    element_body: &str,
    epi_value: &str,
    element_type: &str,
) {
    let block = width * unroll;
    let loaded_operands: Vec<usize> = variant_operands
        .iter()
        .copied()
        .filter(|&index| aliases[index] == index)
        .collect();
    source.push_str(&format!(
        "    for (long block_start = (long)lane; block_start < u.reduction_total; block_start += {block}) {{\n"
    ));
    source.push_str("        if (block_start != (long)lane) {\n");
    push_epilogue_prefetch_slots(
        source,
        "            ",
        rank,
        reduce_dim,
        width,
        unroll,
        &loaded_operands,
        "block_start",
        element_type,
    );
    source.push_str("        }\n");
    source.push_str(&format!(
        "        {FULL_UNROLL}\n        for (int slot = 0; slot < {unroll}; ++slot) {{\n"
    ));
    source.push_str(&format!(
        "            long r = block_start + (long)slot * {width};\n"
    ));
    source.push_str("            if (r < u.reduction_total) {\n");
    source.push_str(&format!("                full_coord[{reduce_dim}] = r;\n"));
    source.push_str("                long out_offset = u.out_base;\n");
    for dim in 0..rank {
        source.push_str(&format!(
            "                out_offset += full_coord[{dim}] * u.broadcast_out_strides[{dim}];\n"
        ));
    }
    for &index in variant_operands {
        let source_slot = if aliases[index] == index {
            format!("epi_pre{index}[slot]")
        } else {
            format!("epi_scratch[{}]", aliases[index])
        };
        source.push_str(&format!(
            "                epi_scratch[{index}] = {source_slot};\n"
        ));
    }
    source.push_str(element_body);
    source.push_str(&format!("                out[out_offset] = {epi_value};\n"));
    source.push_str("            }\n");
    source.push_str("        }\n");
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
    preamble(&mut source, false);

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

