use super::*;

/// the float-only scale-deferred shape the K-quant pair-dot bodies need; shared by the emitter and
/// [`packed_row_geometry`] so both select the body from the same inputs
pub(super) fn packed_row_plain_product(
    resolved: &BoundOp,
    reduce_op: ScalarOp,
    block: &PackedRowBlock,
    expert_source_mode: bool,
) -> bool {
    !expert_source_mode
        && codec_supports_pair_dot(block.codec)
        && resolved.dtype == DType::FLOAT32
        && is_plain_product_reduce(resolved, reduce_op, block.weight, block.other)
}

// ONE exhaustive selection instead of four chained predicates. every axis
// that used to be a boolean at the dispatch site (which codec family,
// whether the op is a plain product, which reference body) is a variant
// here, so a codec or epilogue shape whose body was removed is a
// non-exhaustive match at compile time rather than a silent fallthrough
// to the per-element arm -- which is how a missing fast body has cost us
// an unmeasured regression before.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PackedBodySelect {
    /// ggml's Q5_K body, kept because the rendered Q5_K measured slower (see its doc)
    Q5KGgmlPort,
    /// the sub-block matvec rendered from the codec's own block layout (Q4_K, Q6_K)
    SubBlockRendered,
    /// the level-set matvec rendered from the format's block layout and level table (IQ4_XS)
    LevelsRendered,
    /// the flat-block matvec rendered from a whole-byte-code layout (Q8_0), one block per lane
    FlatRendered,
    /// ggml's own Q4_0 lane geometry, kept as a REFERENCE/Oracle body.
    Q4_0Reference,
    /// the flat-block matvec reading the layout's coded planes (Q5_0, Q5_1, Q4_1), one
    /// block per lane, extent a multiple of the block and not of 256
    BlockRendered,
    /// the generic lane-spread preamble plus its per-codec decode arms
    Generic,
}

impl PackedBodySelect {
    /// Total function of the op's own numeric configuration. Precedence is
    /// exactly the previous chain's: ggml-port arms win over the Q4_0
    /// reference body, which wins over the generic arm.
    pub(super) fn select(
        codec: Codec,
        plain_product: bool,
        plain_product_reduce: bool,
        real_float: bool,
        expert_source_mode: bool,
    ) -> Self {
        let levels_rendered = matches!(codec, Codec::Levels(_))
            && !expert_source_mode
            && level_codec_row_body_rendered(plain_product_reduce, real_float);
        let ggml_port = plain_product
            && matches!(codec, Codec::Q4K | Codec::Q5K | Codec::Q6K)
            && cfg!(feature = "metal")
            && !cfg!(feature = "metal-q4k-split-k");
        // the reference body transcribes the Q4_0 LAYOUT, so it is an
        // implementation of that single cell rather than an axis of the
        // selection: its admission is a feature check here, never a
        // parameter a caller could set for a different codec.
        let q4_0_reference = matches!(codec, Codec::Q4_0)
            && plain_product_reduce
            && real_float
            && cfg!(feature = "metal-q4_0-native")
            && !cfg!(feature = "metal-q4k-split-k");
        let flat_rendered = codec
            .block_layout()
            .is_some_and(|layout| FlatBlockForm::plan(&layout).is_ok())
            && plain_product_reduce
            && real_float
            && !expert_source_mode;
        let block_rendered = block_rendered_elements(codec).is_some()
            && block_codec_row_body_rendered(plain_product_reduce, real_float)
            && !expert_source_mode;
        match (ggml_port, codec, q4_0_reference) {
            (true, Codec::Q5K, _) => Self::Q5KGgmlPort,
            (true, _, _) => Self::SubBlockRendered,
            (false, _, false) if flat_rendered => Self::FlatRendered,
            (false, _, true) => Self::Q4_0Reference,
            (false, _, false) if block_rendered => Self::BlockRendered,
            (false, _, false) if levels_rendered => Self::LevelsRendered,
            (false, _, false) => Self::Generic,
        }
    }

    /// The selection the emitter makes for `resolved`, from the same inputs the emitter reads.
    pub(super) fn for_op(
        resolved: &BoundOp,
        reduce_op: ScalarOp,
        block: &PackedRowBlock,
        expert_source_mode: bool,
    ) -> Self {
        Self::select(
            block.codec,
            packed_row_plain_product(resolved, reduce_op, block, expert_source_mode),
            is_plain_product_reduce(resolved, reduce_op, block.weight, block.other),
            resolved.dtype == DType::FLOAT32,
            expert_source_mode,
        )
    }

    /// Whether the body's lanes own whole units (a block or a scalar run), so simdgroups can
    /// split the reduction between them. The sub-block and lane-spread bodies share a unit across
    /// lanes and cannot. Exhaustive, so a new body variant must decide. Necessary, not sufficient:
    /// `packed_row_geometry` additionally requires the lane unit to be a scalar run.
    pub(super) fn splits_k(self) -> bool {
        match self {
            Self::FlatRendered | Self::BlockRendered => true,
            Self::Q5KGgmlPort
            | Self::SubBlockRendered
            | Self::LevelsRendered
            | Self::Q4_0Reference
            | Self::Generic => false,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn push_packed_row_blocked_body(
    source: &mut String,
    resolved: &BoundOp,
    reduce_op: ScalarOp,
    init: ReduceInit,
    output_axes: &[u16],
    rank: usize,
    quantized: &[Option<Codec>],
    element_type: &str,
    block: &PackedRowBlock,
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
    expert_source_mode: bool,
    geometry: PackedRowGeometry,
    metal: &MetalOnlyExtras,
) -> Result<(), EmitError> {
    let PackedRowGeometry {
        rows_per_simdgroup: rows,
        k_split,
    } = geometry;
    if packed_row_block_token_total(block, &resolved.extents) > 1 {
        push_packed_row_multi_row_body(
            source,
            resolved,
            reduce_op,
            init,
            rank,
            quantized,
            element_type,
            block,
            epilogue_body,
            epilogue_operands,
            expert_source_mode,
            metal,
        )?;
        return Ok(());
    }
    let PackedRowBlock {
        weight,
        other,
        reduce_dim,
        codec,
        ..
    } = *block;
    // `codec_row_block_step_bytes`, not plain `codec_block_bytes`: the
    // preamble below steps `blk_ptr[q]` by `Q4K_BLOCK_ELEMENTS`-wide spans
    // (`weight_base[q] / Q4K_BLOCK_ELEMENTS`) regardless of codec -- equal
    // to `codec_block_bytes` for every K-quant codec, but `Q8_0`'s own block
    // is 32 elements, so its step must be eight real blocks' worth of bytes.
    // The ggml-port bodies below that also take a `block_bytes` parameter
    // never see `Q8_0` (gated to `Q4K`/`Q5K`/`Q6K`), so this substitution is
    // a no-op for every codec but `Q8_0`.
    let block_bytes = codec_row_block_step_bytes(codec);
    let rank_len = rank.max(1);
    let operand_count = resolved.operands().len();
    // seeded on lane 0 only, exactly as the general cooperative path does:
    // the true seed folds in once and every other lane starts at the
    // algebraic identity, so `simd_*` can combine them unconditionally.
    let (init_expr, _) = fold_init_tokens(init);
    let identity = cooperative_identity_token(resolved.node, reduce_op)?;
    // ROW-BLOCKED PACKED PATH. One SIMD group folds `codec.rows_per_
    // simdgroup()` output rows at once so the activation's run of 8 values
    // is loaded into registers ONCE and reused across all of them — ggml's
    // `float sumf[nr0]` (`N_R0_Q4_K 4`, `N_R0_Q6_K 1`). Combined with the
    // super-block header amortization below, the per-element cost becomes
    // one byte load, one mask, one fma (`docs/discipline.md` ROW 74).
    {
        let run = Q4K_BLOCK_ELEMENTS / SIMD_WIDTH as usize;
        source.push_str(&format!("    long group_first = output_index * {rows};\n"));
        source.push_str(&format!("    {element_type} sumf[{rows}];\n"));
        // when the seed and the algebraic identity are the textually same
        // token (true for every op this path actually reaches: Add/Zero,
        // Multiply/One, Maximum/NegativeInfinity, Minimum/PositiveInfinity),
        // every lane's `sumf[q]` starts at the same value regardless of
        // which lane it is -- the per-lane ternary is dead, a select over
        // two identical constants, so emit a plain assignment instead.
        if init_expr == identity {
            source.push_str(&format!(
                "    for (int q = 0; q < {rows}; ++q) {{ sumf[q] = {identity}; }}\n"
            ));
        } else if k_split > 1 || cfg!(feature = "metal-q4k-split-k") {
            // the true seed folds in exactly ONCE across the WHOLE
            // threadgroup, not once per simdgroup -- at split == 1 `sgitg`
            // is always `0`, so this collapses to the feature-off condition
            // (`lane == 0u`) exactly.
            source.push_str(&format!(
                "    for (int q = 0; q < {rows}; ++q) {{ sumf[q] = (lane == 0u && sgitg == 0u) ? ({init_expr}) : ({identity}); }}\n"
            ));
        } else {
            source.push_str(&format!(
                "    for (int q = 0; q < {rows}; ++q) {{ sumf[q] = (lane == 0u) ? ({init_expr}) : ({identity}); }}\n"
            ));
        }
        source.push_str(&format!("    long weight_base[{rows}];\n"));
        source.push_str(&format!("    long other_base[{rows}];\n"));
        // `coord_q_cache[q]` -- kept past this loop, not a scratch local --
        // is the SAME `flat = group_first + q` decode
        // `push_packed_row_combine_and_write`'s epilogue needs for the
        // output-write coordinate; caching it here instead of re-running the
        // `%`/`/` chain against `u.output_extents` a second time in the
        // epilogue is the fix for the ladder-vs-production kernel-body gap
        // this op's own diff found (`docs/bench-campaigns/2026-09-03-gpu-one-
        // risc/design-2026-09-04/kernel-body-diff.md`): the ladder's hand-
        // written dispatch never had a second uniform-driven coordinate
        // decode to pay, because it never had a first one either.
        source.push_str(&format!("    long coord_q_cache[{rows}][{rank_len}];\n"));
        push_packed_row_group_bases(source, resolved, output_axes, rank, rows, weight, other);
        // STRIDE-FREE SPECIALIZATION (`docs/discipline.md` perf/packed-row-
        // addressing row): ggml's own row-blocked kernel assumes a
        // contiguous activation and addresses it with pure element offsets
        // (`ggml-metal.metal:5132`'s `y4 += 4*QK_K`, no per-element stride
        // multiply anywhere). This op's activation is not always
        // contiguous along the reduce axis, but `resolved`'s `Layout`
        // already carries the answer here at EMIT time -- this function
        // renders one op's kernel text once, not once per dispatch -- so
        // every activation address below drops the runtime `other_stride`
        // multiply entirely in the source text whenever the layout proves
        // it would multiply by 1.
        let other_stride_is_one = resolved.operands()[other].1.stride(reduce_dim as u16) == 1;
        source.push_str(&format!(
            "    long other_stride = u.operand_strides[{other}][{reduce_dim}];\n"
        ));
        // LANE SPREAD, ggml's `ix = tiisg/8`. Putting all 32 lanes on ONE
        // super-block gives each lane 8 of its 256 elements, so the header
        // decode is amortized over 8. Putting EIGHT lanes on a super-block
        // and letting the 32 lanes span FOUR at once gives each lane a whole
        // 32-element sub-block per decode — 4x the amortization, and the
        // sub-block is exactly the granularity the header is constant over.
        //
        // `it` selects the sub-block, so `slot = it * 32` and every one of
        // that lane's 32 elements shares a group and a nibble half. Levels
        // are still pulled 8 at a time (`q4k_run8`) rather than 32, to keep
        // the live register count near ggml's `yl[16]+yh[16]+sumf[4]`.
        // eight lanes per super-block (ggml's `tiisg/8`), so the 32 lanes of
        // a SIMD group span four super-blocks and each lane owns exactly one
        // 32-element sub-block — the granularity the header is constant over.
        let lanes_per_block = 8;
        let sub = Q4K_BLOCK_ELEMENTS / lanes_per_block;
        // Structural, not feature-gated: the paired-nibble/paired-lane body
        // applies to any codec whose block layout has one
        // ([`Codec::supports_pair_dot`]) when the dtype is real
        // `Float32` -- a `DType` match, not the `element_type == "float"`
        // MSL-type-token comparison this replaced, which also admitted
        // `Int32`/`UInt32`/`Bool`/`Int8`/`UInt8` (every dtype `type_token`
        // happens to lower to the same MSL `float` storage type) and would
        // have run the scale/minimum float algebra below on integer data.
        let plain_product = packed_row_plain_product(resolved, reduce_op, block, expert_source_mode);
        // With `metal`: the ggml-geometry bodies. `Q4_K` and `Q6_K` are RENDERED from their
        // `BlockLayout`s ([`sub_block_matvec_msl`]) at the launch geometry their
        // `SubBlockMatvecGeometry` constructor names; `Q5_K` keeps
        // [`push_q5k_ggml_port_body`] (its rendering measured slower, see that doc). Requires
        // `plain_product` (float-only scale-deferred shape, the same gate
        // `q4k_pair_dot`'s own arm below needs) and takes priority over it --
        // both target the exact same design point. Not `metal-q4k-split-k`-aware:
        // its `ib` loop has no `sgitg`/`split` stride of its own. ALSO requires the codec be one of `Q4K`, `Q5K`, `Q6K`:
        // `plain_product` alone is not codec-specific (every codec with
        // `supports_pair_dot` reaches this arm), and routing a `Q5_K` matmul
        // through a body that never reads its `qh` plane gave relative=0.977
        // (`metal_matmul_on_packed_q5k_weights_matches_the_dequantized_f32_cpu_path`).
        // (the gate itself now lives in `PackedBodySelect::select` below)
        // `metal-q4_0-native` (NOT in omega's `metal`; `proxima-model-interop`'s
        // `metal` enables it, proxima-model-interop/Cargo.toml:153): ggml's OWN `Q4_0` lane geometry
        // (`push_q4_0_native_body`'s own doc) rather than this preamble's
        // K-quant-shaped `ix = lane/8` split -- `Q4_0` has no `codec_
        // supports_pair_dot` entry (its own doc: only the four K-quants have
        // a paired-nibble body), so it never sets `plain_product` above and
        // needs its own gate mirroring `Codec::Q4_0 if is_plain_product_
        // reduce`'s existing per-element/pair-dot arms below. Not `metal-
        // q4k-split-k`-aware, same posture as `use_ggml_port`: this body owns
        // its own complete `ib` loop with no `sgitg`/`split` stride of its own.
        // (the gate itself now lives in `PackedBodySelect::select` below)
        let packed_body = PackedBodySelect::select(
            codec,
            plain_product,
            // the two facts the reference arm needs, both structural: an identity
            // epilogue and real f32 activations. Neither names a codec.
            is_plain_product_reduce(resolved, reduce_op, weight, other),
            resolved.dtype == DType::FLOAT32,
            expert_source_mode,
        );
        if packed_body == PackedBodySelect::Q5KGgmlPort {
            push_q5k_ggml_port_body(
                source,
                weight,
                other,
                rows,
                block_bytes,
                other_stride_is_one,
                metal,
            );
        } else if packed_body == PackedBodySelect::SubBlockRendered {
            let layout = codec
                .block_layout()
                .ok_or(EmitError::PackedCodecNotSupported {
                    node: resolved.node,
                    codec: "no block layout",
                })?;
            let (simdgroups, lanes) = (packed_row_nsg_factor() as usize, SIMD_WIDTH as usize);
            let geometry = match codec {
                Codec::Q6K => {
                    SubBlockMatvecGeometry::decode_signed_scales(rows, simdgroups, lanes)
                }
                _ => SubBlockMatvecGeometry::decode(rows, simdgroups, lanes),
            };
            let reduction_bound = packed_row_reduction_bound_token(metal);
            let body = sub_block_matvec_msl(
                &layout,
                &geometry,
                weight,
                other,
                other_stride_is_one,
                &reduction_bound,
            )
            .map_err(|refusal| EmitError::PackedCodecNotSupported {
                node: resolved.node,
                codec: refusal.0,
            })?;
            source.push_str(&body);
        } else if packed_body == PackedBodySelect::FlatRendered {
            let layout = codec
                .block_layout()
                .ok_or(EmitError::PackedCodecNotSupported {
                    node: resolved.node,
                    codec: "no block layout",
                })?;
            let form = FlatBlockForm::plan(&layout).map_err(|refusal| EmitError::PackedCodecNotSupported {
                node: resolved.node,
                codec: refusal.0,
            })?;
            let reduction_bound = packed_row_reduction_bound_token(metal);
            let launch = FlatLaunch {
                rows,
                lanes: SIMD_WIDTH as usize,
                simdgroups: k_split,
                weight,
                other,
                other_stride_is_one,
            };
            source.push_str(
                &form
                    .matvec_msl(&launch, &reduction_bound, element_type, Fold::ScaleEachTerm)
                    .map_err(|refusal| EmitError::PackedCodecNotSupported {
                        node: resolved.node,
                        codec: refusal.0,
                    })?,
            );
        } else if packed_body == PackedBodySelect::BlockRendered {
            let refused = |refusal: sub_block_matvec::SubBlockMatvecRefusal| EmitError::PackedCodecNotSupported {
                node: resolved.node,
                codec: refusal.0,
            };
            let (form, fold) = match scalar_run_read(codec) {
                Some(scalar) => (
                    FlatBlockForm::plan_scalar(scalar, crate::sized::PACKED_ROW_SCALAR_RUN).map_err(refused)?,
                    Fold::Plain,
                ),
                None => {
                    let layout = codec
                        .block_layout()
                        .ok_or(EmitError::PackedCodecNotSupported {
                            node: resolved.node,
                            codec: "no block layout",
                        })?;
                    (FlatBlockForm::plan_planes(&layout).map_err(refused)?, Fold::ScaleEachTerm)
                }
            };
            let reduction_bound = packed_row_reduction_bound_token(metal);
            let launch = FlatLaunch {
                rows,
                lanes: SIMD_WIDTH as usize,
                simdgroups: k_split,
                weight,
                other,
                other_stride_is_one,
            };
            source.push_str(&form.matvec_msl(&launch, &reduction_bound, element_type, fold).map_err(refused)?);
        } else if packed_body == PackedBodySelect::LevelsRendered {
            let Codec::Levels(format) = codec else {
                return Err(EmitError::PackedCodecNotSupported {
                    node: resolved.node,
                    codec: "not a level format",
                });
            };
            let layout = format.packing().ok_or(EmitError::PackedCodecNotSupported {
                node: resolved.node,
                codec: "no block layout",
            })?;
            let codes = LevelCodes {
                table: &format.levels,
                element_fn: codec_element_fn_name(codec),
            };
            let (simdgroups, lanes) = (packed_row_nsg_factor() as usize, SIMD_WIDTH as usize);
            let geometry = SubBlockMatvecGeometry::for_levels(&format.levels, rows, simdgroups, lanes);
            let reduction_bound = packed_row_reduction_bound_token(metal);
            let body = level_matvec_msl(
                &layout,
                &codes,
                &geometry,
                weight,
                other,
                other_stride_is_one,
                &reduction_bound,
            )
            .map_err(|refusal| EmitError::PackedCodecNotSupported {
                node: resolved.node,
                codec: refusal.0,
            })?;
            source.push_str(&body);
        } else if packed_body == PackedBodySelect::Q4_0Reference {
            push_q4_0_native_body(source, weight, other, rows, other_stride_is_one, metal);
        } else {
            source.push_str(&format!("    uint ix = (uint)lane / {lanes_per_block}u;\n"));
            source.push_str(&format!("    uint it = (uint)lane % {lanes_per_block}u;\n"));
            source.push_str(&format!("    uint slot = it * {sub}u;\n"));
            if plain_product {
                source.push_str(
                    "    uint iq = it / 4u; uint ir = it % 4u;\n    float yl[16]; float yh[16];\n",
                );
            } else {
                source.push_str(&format!("    {element_type} acts[{sub}];\n"));
            }
            let reduction_bound = packed_row_reduction_bound_token(metal);
            source.push_str(&format!(
                "    int super_blocks = (int){reduction_bound} / {Q4K_BLOCK_ELEMENTS};\n"
            ));
            let ix_stride = SIMD_WIDTH as usize / lanes_per_block;
            if cfg!(feature = "metal-q4k-split-k") {
                // each simdgroup (`sgitg`, 0 at split == 1) owns a disjoint
                // interleaved slice of super-blocks -- a plain strided loop, so a
                // `super_blocks` not evenly divisible by `split` is handled by
                // construction (some simdgroups simply run one fewer iteration),
                // never a separate ragged-tail branch.
                source.push_str(&format!(
                "    int ib_first = (int)(ix + sgitg * {ix_stride}u);\n    int ib_step = (int)({ix_stride}u * split);\n"
            ));
            } else {
                source.push_str(&format!(
                    "    int ib_first = (int)ix;\n    int ib_step = {ix_stride};\n"
                ));
            }
            // HOIST + POINTER INCREMENT (`docs/discipline.md` perf/packed-row-
            // addressing row): `weight_base[q]/Q4K_BLOCK_ELEMENTS` and the y4
            // lane offset (`64*iq + 8*ir`) are invariant across every `ib` this
            // thread visits -- only `ib` itself varies. The prior form
            // recomputed `(weight_base[q]/256 + ib) * block_bytes` and
            // `ib*256*other_stride` from scratch every iteration (a 64-bit
            // multiply-add per row per iteration); ggml's own row/`y4` pointers
            // instead advance by a CONSTANT per iteration
            // (`ggml-metal.metal:5132,5182`'s `q1 += nb01/2`, `y4 += 4*QK_K`).
            // This computes each row's starting byte pointer and the
            // per-iteration byte step ONCE before the loop, then the loop body
            // only adds.
            source.push_str(&format!(
                "    long blk_step = (long)ib_step * {block_bytes};\n"
            ));
            source.push_str(&format!("    device const uchar *blk_ptr[{rows}];\n"));
            source.push_str(&format!("    for (int q = 0; q < {rows}; ++q) {{\n"));
            source.push_str(&format!(
            "        blk_ptr[q] = in{weight} + ((long)((int)weight_base[q] / {Q4K_BLOCK_ELEMENTS}) + (long)ib_first) * {block_bytes};\n"
        ));
            source.push_str("    }\n");
            if plain_product {
                push_q4k_plain_product_y4_address(source, other, other_stride_is_one);
            } else if other_stride_is_one {
                // SAME HOIST, generic (non-plain-product) arm: `elem0 = ib*256 +
                // slot` was rebuilt every `ib` purely to feed `(elem0+j)*
                // other_stride` -- the identical 64-bit multiply-add-per-
                // iteration shape arm1 already removed from the plain-product
                // `y4` pointer above. `other_stride_is_one` additionally drops
                // the multiply itself, same as the `y4` arm just above.
                source.push_str(&format!(
                    "    long acts_step = (long)ib_step * {Q4K_BLOCK_ELEMENTS};\n"
                ));
                source.push_str(&format!("    device const {element_type} *acts_row = in{other} + other_base[0] + (long)ib_first * {Q4K_BLOCK_ELEMENTS} + (long)slot;\n"));
            } else {
                source.push_str(&format!(
                    "    long acts_step = (long)ib_step * {Q4K_BLOCK_ELEMENTS} * other_stride;\n"
                ));
                source.push_str(&format!("    device const {element_type} *acts_row = in{other} + other_base[0] + (long)ib_first * {Q4K_BLOCK_ELEMENTS} * other_stride + (long)slot * other_stride;\n"));
            }
            source.push_str("    for (int ib = ib_first; ib < super_blocks; ib += ib_step) {\n");
            if plain_product && other_stride_is_one {
                source.push_str("        for (uint i = 0u; i < 8u; ++i) { yl[i] = y4[i]; yl[i + 8u] = y4[i + 32u]; yh[i] = y4[i + 128u]; yh[i + 8u] = y4[i + 160u]; }\n");
            } else if plain_product {
                // CORRECTNESS FIX, not part of the stride-free specialization
                // above: this arm's `y4[i]`/`y4[i+32]`/... reads were pure
                // element offsets regardless of `other_stride` before this
                // landing -- correct only by accident, for every caller that
                // happened to hand this path a contiguous activation. Ported
                // from `sub_block_matvec_msl`'s own already-stride-aware
                // form (`y4_base + (long)i * other_stride`, below in this
                // file), the one sibling body that already got this right.
                source.push_str("        for (uint i = 0u; i < 8u; ++i) { yl[i] = y4[(long)i * other_stride]; yl[i + 8u] = y4[(long)(i + 32u) * other_stride]; yh[i] = y4[(long)(i + 128u) * other_stride]; yh[i + 8u] = y4[(long)(i + 160u) * other_stride]; }\n");
            } else {
                source.push_str(&format!("        for (int j = 0; j < {sub}; ++j) {{\n"));
                if other_stride_is_one {
                    source.push_str("            acts[j] = acts_row[j];\n");
                } else {
                    source.push_str("            acts[j] = acts_row[(long)j * other_stride];\n");
                }
                source.push_str("        }\n");
            }
            source.push_str(&format!("        for (int q = 0; q < {rows}; ++q) {{\n"));
            source.push_str("            device const uchar *blk = blk_ptr[q];\n");
            match codec {
                Codec::Q2K => {
                    source.push_str(&format!("            for (int e = 0; e < {sub}; ++e) {{\n"));
                    source.push_str(&format!(
                        "                {element_type} scratch[{}];\n",
                        operand_count.max(1)
                    ));
                    source.push_str(&format!(
                        "                scratch[{weight}] = q2k_element(blk, slot + (uint)e);\n"
                    ));
                    source.push_str(&format!("                scratch[{other}] = acts[e];\n"));
                    let value_expr = push_body_steps(
                        source,
                        resolved.element_body(),
                        "                ",
                        element_type,
                    );
                    source.push_str(&format!(
                        "                {element_type} value = {value_expr};\n"
                    ));
                    let combine_expr = scalar_op_expr(reduce_op, &["sumf[q]", "value"]);
                    source.push_str(&format!("                sumf[q] = {combine_expr};\n"));
                    source.push_str("            }\n");
                }
                Codec::Q3K if plain_product => {
                    // `plain_product` is codec-agnostic (the `yl`/`yh` gather
                    // above is built once, shared by `Q4_K`/`Q3_K` and, when
                    // `metal-q5k-pair-dot` is on, `Q5_K` too) -- no separate
                    // activation load path needed here, same posture as
                    // `Q5_K`'s own `plain_product` arm.
                    source.push_str(
                        "            sumf[q] = sumf[q] + q3k_pair_dot(blk, iq, ir, yl, yh);\n",
                    );
                }
                Codec::Q3K => {
                    // `Q3_K`'s sub-block width (16) is narrower than this
                    // loop's 32-element `sub` slot, unlike `Q5_K`'s matching
                    // 32-element sub-block -- amortizing one header decode
                    // across the whole slot the way the `Q5_K` arm below does
                    // would silently span two different sub-block scales. Each
                    // call to `q3k_element` decodes its own header, the same
                    // posture `Q6_K`'s per-element path takes for a different
                    // reason (its scale bytes are plain, not bit-packed, so the
                    // per-call cost is small either way). A follow-up
                    // optimization (a two-headers-per-slot amortization), not a
                    // correctness gap.
                    source.push_str(&format!("            for (int e = 0; e < {sub}; ++e) {{\n"));
                    source.push_str(&format!(
                        "                {element_type} scratch[{}];\n",
                        operand_count.max(1)
                    ));
                    source.push_str(&format!(
                        "                scratch[{weight}] = q3k_element(blk, slot + (uint)e);\n"
                    ));
                    source.push_str(&format!("                scratch[{other}] = acts[e];\n"));
                    let value_expr = push_body_steps(
                        source,
                        resolved.element_body(),
                        "                ",
                        element_type,
                    );
                    source.push_str(&format!(
                        "                {element_type} value = {value_expr};\n"
                    ));
                    let combine_expr = scalar_op_expr(reduce_op, &["sumf[q]", "value"]);
                    source.push_str(&format!("                sumf[q] = {combine_expr};\n"));
                    source.push_str("            }\n");
                }
                Codec::Q4K => {
                    if plain_product {
                        source.push_str(
                            "            sumf[q] = sumf[q] + q4k_pair_dot(blk, iq, ir, yl, yh);\n",
                        );
                    } else if is_plain_product_reduce(resolved, reduce_op, weight, other) {
                        // SCALE-DEFERRED PATH (`docs/discipline.md` ROW 106).
                        // Accumulate the raw nibble x activation product and the
                        // activation sum UNSCALED across the whole sub-block, then
                        // apply `hdr.scale`/`hdr.minimum` ONCE at the end instead
                        // of once per element — legal here because
                        // `is_plain_product_reduce` already proved reduce_op is
                        // `Add` and the body is exactly `weight * other`, so
                        // `sum_j (scale*nibble_j - min)*act_j == scale*sum(nibble_j
                        // *act_j) - min*sum(act_j)`. Mirrors
                        // `ggml-metal.metal:5157-5175`'s `acc1`/`dall` split.
                        // Two bodies behind `metal-q4k-mask-fma`: off, the
                        // shift-then-mask `q4k_run8` extraction into a `dot`
                        // reduce; on, ggml's actual mask-without-shift technique,
                        // fused with the accumulate -- see
                        // `push_q4k_product_reduce_body`'s own doc.
                        push_q4k_header_decode(source);
                        push_q4k_product_reduce_body(source, sub, run, element_type);
                    } else {
                        // the per-element dequant below reads `hdr`, which only
                        // the scale-deferred arm above declared
                        push_q4k_header_decode(source);
                        source.push_str(&format!(
                            "            for (int c = 0; c < {}; ++c) {{\n",
                            sub / run
                        ));
                        // raw 4-bit levels (0..15) are exact in float regardless of
                        // the kernel's element type; q4k_run8 takes `thread float
                        // *out`, and the narrowing to element_type happens where
                        // levels combine into scratch below, same as every other
                        // operand read.
                        source.push_str(&format!("                float levels[{run}];\n"));
                        source.push_str(&format!(
                            "                q4k_run8(blk, slot + (uint)(c * {run}), levels);\n"
                        ));
                        source.push_str(&format!(
                            "                for (int j = 0; j < {run}; ++j) {{\n"
                        ));
                        source.push_str(&format!(
                            "                    {element_type} scratch[{}];\n",
                            operand_count.max(1)
                        ));
                        source.push_str(&format!(
                        "                    scratch[{weight}] = hdr.scale * levels[j] - hdr.minimum;\n"
                    ));
                        source.push_str(&format!(
                            "                    scratch[{other}] = acts[c * {run} + j];\n"
                        ));
                        let value_expr = push_body_steps(
                            source,
                            resolved.element_body(),
                            "                    ",
                            element_type,
                        );
                        source.push_str(&format!(
                            "                    {element_type} value = {value_expr};\n"
                        ));
                        let combine_expr = scalar_op_expr(reduce_op, &["sumf[q]", "value"]);
                        source
                            .push_str(&format!("                    sumf[q] = {combine_expr};\n"));
                        source.push_str("                }\n");
                        source.push_str("            }\n");
                    }
                }
                Codec::Q5K if plain_product => {
                    // See [`Q5K_PAIR_DOT_MSL`]: the same `yl`/`yh` two-word-load pairing `Q4_K`'s own
                    // `plain_product` arm above uses, extended with `Q5_K`'s `qh`
                    // high-bit plane. Reads the SAME `yl`/`yh` activation gather
                    // this preamble already built for `Q4_K` (`plain_product`
                    // is codec-agnostic there), so no separate activation load
                    // path is needed for this codec.
                    source.push_str(
                        "            sumf[q] = sumf[q] + q5k_pair_dot(blk, iq, ir, yl, yh);\n",
                    );
                }
                Codec::Q5K => {
                    // No `q5k_run8`-style batched unpack yet — `Q5_K`'s `qh`
                    // high-bit plane means each element needs a `qs` nibble AND
                    // a `qh` bit from a DIFFERENT byte, the same shape gap
                    // `Q6_K`'s own arm below documents. `d` and this sub-block's
                    // scale/min/mask ARE decoded once per 32-element run via
                    // `q5k_header_for` (the same granularity `q4k_header_for`
                    // amortizes over) — a follow-up optimization, not a
                    // correctness gap; see this landing's discipline row (ROW
                    // 92) for the measured cost of skipping it. The `plain_product`
                    // arm above replaces this whole per-element loop with the
                    // paired-nibble body whenever the reduce is a plain product.
                    source.push_str("            q5k_header hdr = q5k_header_for(blk, slot);\n");
                    source.push_str(&format!("            for (int e = 0; e < {sub}; ++e) {{\n"));
                    source.push_str(&format!(
                        "                {element_type} scratch[{}];\n",
                        operand_count.max(1)
                    ));
                    source.push_str(&format!(
                        "                scratch[{weight}] = q5k_value(blk, slot + (uint)e, hdr);\n"
                    ));
                    source.push_str(&format!("                scratch[{other}] = acts[e];\n"));
                    let value_expr = push_body_steps(
                        source,
                        resolved.element_body(),
                        "                ",
                        element_type,
                    );
                    source.push_str(&format!(
                        "                {element_type} value = {value_expr};\n"
                    ));
                    let combine_expr = scalar_op_expr(reduce_op, &["sumf[q]", "value"]);
                    source.push_str(&format!("                sumf[q] = {combine_expr};\n"));
                    source.push_str("            }\n");
                }
                Codec::Q6K if plain_product => {
                    // See [`Q6K_PAIR_DOT_MSL`]: the same paired-lane body `Q4_K`/`Q5_K`'s own
                    // `plain_product` arms use above, ported to `Q6_K`'s
                    // ql/qh/signed-scale layout. Reads the SAME `yl`/`yh`
                    // activation gather this preamble already built for
                    // `Q4_K` (`plain_product` is codec-agnostic there), so no
                    // separate activation load path is needed for this codec.
                    source.push_str(
                        "            sumf[q] = sumf[q] + q6k_pair_dot(blk, iq, ir, yl, yh);\n",
                    );
                }
                Codec::Q6K => {
                    // No `q6k_run8`-style batched unpack yet — `Q6_K`'s bit
                    // layout does not reduce to two word loads the way `Q4_K`'s
                    // does (each element needs a `ql` byte, a `qh` byte, AND a
                    // sub-block scale byte, not one nibble out of an
                    // already-loaded word). Correct, one element at a time; `d`
                    // is still decoded ONCE per super-block via
                    // `q6k_header_for` rather than per element. The `plain_product`
                    // arm above replaces this whole per-element loop with the
                    // paired-lane body whenever the reduce is a plain product.
                    source.push_str("            q6k_header hdr = q6k_header_for(blk);\n");
                    source.push_str(&format!("            for (int e = 0; e < {sub}; ++e) {{\n"));
                    source.push_str(&format!(
                        "                {element_type} scratch[{}];\n",
                        operand_count.max(1)
                    ));
                    source.push_str(&format!(
                        "                scratch[{weight}] = q6k_value(blk, slot + (uint)e, hdr);\n"
                    ));
                    source.push_str(&format!("                scratch[{other}] = acts[e];\n"));
                    let value_expr = push_body_steps(
                        source,
                        resolved.element_body(),
                        "                ",
                        element_type,
                    );
                    source.push_str(&format!(
                        "                {element_type} value = {value_expr};\n"
                    ));
                    let combine_expr = scalar_op_expr(reduce_op, &["sumf[q]", "value"]);
                    source.push_str(&format!("                sumf[q] = {combine_expr};\n"));
                    source.push_str("            }\n");
                }
                Codec::Q8_0
                    if cfg!(feature = "metal-q4k-split-k")
                        && is_plain_product_reduce(resolved, reduce_op, weight, other)
                        && resolved.dtype == DType::FLOAT32 =>
                {
                    // BATCHED ARM, mirroring the K-quant codecs' own
                    // scale-deferred `is_plain_product_reduce` arms
                    // (`Codec::Q4K`'s below, `push_q4k_product_reduce_body`):
                    // `Q8_0`'s whole 32-element slot IS one real block,
                    // already matching this preamble's `acts[sub]` gather
                    // one-for-one, so `q8_0_pair_dot` decodes that block's
                    // one `d` scale ONCE and walks it against `acts` directly
                    // -- see [`Q8_0_PAIR_DOT_MSL`]'s own doc for why this
                    // replaces `q8_0_super_element`'s per-element re-decode
                    // instead of extending `push_q4k_plain_product_y4_
                    // address`'s `yl`/`yh` pairing (`Q8_0` has no nibble
                    // packing to pair across). The `DType::FLOAT32` guard
                    // mirrors `plain_product`'s own gate above (line ~146):
                    // `acts` is declared `{element_type} acts[sub]`, and
                    // `q8_0_pair_dot`'s MSL signature is fixed to `float
                    // *acts` -- a `half`-typed `acts` array would not bind
                    // to it (found by `metal_matmul_parity_across_codec_and_
                    // dtype::q4_0_at_float16`'s sibling case, same gap,
                    // same fix, for `Codec::Q4_0` below).
                    source.push_str(
                        "            sumf[q] = sumf[q] + q8_0_pair_dot(blk, slot, acts);\n",
                    );
                }
                Codec::Q8_0 => {
                    // `sub` (32) is exactly `Q8_0_BLOCK_ELEMENTS`: this
                    // lane's whole 32-element slot IS one real `Q8_0` block,
                    // so `q8_0_super_element` decodes it a scale-load-plus-
                    // level-load at a time. The `is_plain_product_reduce`
                    // arm above replaces this per-element loop whenever the
                    // reduce is a plain product; this fallback stays for
                    // reduce shapes that identity does not cover.
                    source.push_str(&format!("            for (int e = 0; e < {sub}; ++e) {{\n"));
                    source.push_str(&format!(
                        "                {element_type} scratch[{}];\n",
                        operand_count.max(1)
                    ));
                    source.push_str(&format!(
                        "                scratch[{weight}] = q8_0_super_element(blk, slot + (uint)e);\n"
                    ));
                    source.push_str(&format!("                scratch[{other}] = acts[e];\n"));
                    let value_expr = push_body_steps(
                        source,
                        resolved.element_body(),
                        "                ",
                        element_type,
                    );
                    source.push_str(&format!(
                        "                {element_type} value = {value_expr};\n"
                    ));
                    let combine_expr = scalar_op_expr(reduce_op, &["sumf[q]", "value"]);
                    source.push_str(&format!("                sumf[q] = {combine_expr};\n"));
                    source.push_str("            }\n");
                }
                Codec::Q4_0
                    if is_plain_product_reduce(resolved, reduce_op, weight, other)
                        && resolved.dtype == DType::FLOAT32 =>
                {
                    // BATCHED ARM -- the #1 GPU-time bucket in the E2B checkpoint
                    // decode (`reduce-packed-row-blocked`, 39.6%, 275
                    // dispatches/step) ran through the per-element fallback
                    // below on every one of those dispatches. Same posture as
                    // the `Codec::Q8_0 if is_plain_product_reduce` arm above
                    // and the K-quant codecs' own scale-deferred arms
                    // (`Codec::Q4K`'s below): `Q4_0`'s whole 32-element slot
                    // IS one real block, already matching this preamble's
                    // `acts[sub]` gather one-for-one, so `q4_0_pair_dot`
                    // decodes that block's one `d` scale ONCE and walks its
                    // 16 packed-nibble bytes against `acts` directly -- see
                    // [`Q4_0_PAIR_DOT_MSL`]'s own doc. The `DType::FLOAT32`
                    // guard mirrors `plain_product`'s own gate above (line
                    // ~146): `acts` is declared `{element_type} acts[sub]`,
                    // and `q4_0_pair_dot`'s MSL signature is fixed to `float
                    // *acts` -- a `half`-typed `acts` array does not bind to
                    // it (`metal_matmul_parity_across_codec_and_dtype::
                    // q4_0_at_float16` caught this: "no matching function
                    // for call to 'q4_0_pair_dot'" against a `half[32]`
                    // argument).
                    source.push_str(
                        "            sumf[q] = sumf[q] + q4_0_pair_dot(blk, slot, acts);\n",
                    );
                }
                Codec::Q4_0 => {
                    // `sub` (32) is exactly `Q4_0_BLOCK_ELEMENTS`: this
                    // lane's whole 32-element slot IS one real `Q4_0` block,
                    // so `q4_0_super_element` decodes it a scale-load-plus-
                    // nibble-load at a time. The `is_plain_product_reduce`
                    // arm above replaces this per-element loop whenever the
                    // reduce is a plain product; this fallback stays for
                    // reduce shapes that identity does not cover.
                    source.push_str(&format!("            for (int e = 0; e < {sub}; ++e) {{\n"));
                    source.push_str(&format!(
                        "                {element_type} scratch[{}];\n",
                        operand_count.max(1)
                    ));
                    source.push_str(&format!(
                        "                scratch[{weight}] = q4_0_super_element(blk, slot + (uint)e);\n"
                    ));
                    source.push_str(&format!("                scratch[{other}] = acts[e];\n"));
                    let value_expr = push_body_steps(
                        source,
                        resolved.element_body(),
                        "                ",
                        element_type,
                    );
                    source.push_str(&format!(
                        "                {element_type} value = {value_expr};\n"
                    ));
                    let combine_expr = scalar_op_expr(reduce_op, &["sumf[q]", "value"]);
                    source.push_str(&format!("                sumf[q] = {combine_expr};\n"));
                    source.push_str("            }\n");
                }
                Codec::Fp6E3M2
                    if is_plain_product_reduce(resolved, reduce_op, weight, other)
                        && resolved.dtype == DType::FLOAT32 =>
                {
                    source.push_str(
                        "            sumf[q] = sumf[q] + scaled_e3m2_quad_dot(blk, slot, acts);\n",
                    );
                }
                Codec::Fp6E3M2 => push_super_element_fallback(
                    source,
                    resolved,
                    reduce_op,
                    block,
                    element_type,
                    "scaled_e3m2_super_element",
                ),
                Codec::Q5_1 => {
                    return Err(EmitError::NonKQuantCodec {
                        node: resolved.node,
                        codec: "q5_1",
                    });
                }
                Codec::Q5_0 => {
                    return Err(EmitError::NonKQuantCodec {
                        node: resolved.node,
                        codec: "q5_0",
                    });
                }
                Codec::Float16 => {
                    return Err(EmitError::NonKQuantCodec {
                        node: resolved.node,
                        codec: "float16",
                    });
                }
                Codec::BFloat16 => {
                    return Err(EmitError::NonKQuantCodec {
                        node: resolved.node,
                        codec: "bfloat16",
                    });
                }
                Codec::BFloat8 => {
                    return Err(EmitError::NonKQuantCodec {
                        node: resolved.node,
                        codec: "bfloat8",
                    });
                }
                Codec::Bf8E5M2 | Codec::Bf4E2M1 => {
                    return Err(EmitError::NonKQuantCodec {
                        node: resolved.node,
                        codec: "brain-float scalar",
                    });
                }
                // Unreachable by construction (see `operand_read`'s doc on
                // `PackedOperands`'s closed population), kept exhaustive.
                Codec::Q4_1
                | Codec::Q8_1
                | Codec::Q8_1Grouped
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
                | Codec::Q2_0
                | Codec::Pq2_0
                | Codec::Ptq1_0 => {
                    return Err(EmitError::NonKQuantCodec {
                        node: resolved.node,
                        codec: "unsupported",
                    });
                }
                _ => {
                    return Err(EmitError::NonKQuantCodec {
                        node: resolved.node,
                        codec: "unsupported",
                    });
                }
            }
            source.push_str("        }\n");
            source.push_str(&format!(
                "        for (int q = 0; q < {rows}; ++q) {{ blk_ptr[q] += blk_step; }}\n"
            ));
            if plain_product {
                source.push_str("        y4 += y4_step;\n");
            } else {
                source.push_str("        acts_row += acts_step;\n");
            }
            source.push_str("    }\n");
        }
        push_packed_row_combine_and_write(
            source,
            resolved.node,
            reduce_op,
            rows,
            k_split,
            rank,
            output_axes,
            packed_row_direct_output_axis(resolved, output_axes),
            element_type,
            epilogue_body,
            epilogue_operands,
        )?;
    }
    Ok(())
}

/// The plain-product row-blocked bodies' shared activation address: the
/// `y4` pointer's base and per-iteration byte step. `push_packed_row_blocked_
/// body`'s own `plain_product` arm and `sub_block_matvec_msl` both read
/// `weight_base`/`other_base`/`other_stride`/`iq`/`ir` from the same
/// preamble and must have already declared `ib_first`/`ib_step` in scope --
/// this is the ONE place either renders the pointer, so the stride-free
/// specialization (drop the runtime `other_stride` multiply when the layout
/// proves it is 1, see the caller's own `other_stride_is_one` doc) applies
/// identically to both instead of drifting.
pub(super) fn push_q4k_plain_product_y4_address(
    source: &mut String,
    other: usize,
    other_stride_is_one: bool,
) {
    push_packed_row_plain_product_y4_address(
        source,
        other,
        other_stride_is_one,
        "64u * iq + 8u * ir",
    );
}

/// [`push_q4k_plain_product_y4_address`] generalized to any ggml-port body's
/// own lane-derived activation offset -- the Q4_0 reference body shares
/// this instead of a second copy of the address arithmetic, passing
/// `"128u * ip + l0"` (llama.cpp's `y_offset`, `ggml-metal.metal:5383`) in place
/// of `Q4_K`'s `"64u * iq + 8u * ir"`.
pub(super) fn push_packed_row_plain_product_y4_address(
    source: &mut String,
    other: usize,
    other_stride_is_one: bool,
    lane_offset: &str,
) {
    if other_stride_is_one {
        source.push_str(&format!(
            "    long y4_step = (long)ib_step * {Q4K_BLOCK_ELEMENTS};\n"
        ));
        source.push_str(&format!("    device const float *y4 = in{other} + other_base[0] + (long)ib_first * {Q4K_BLOCK_ELEMENTS} + (long)({lane_offset});\n"));
    } else {
        source.push_str(&format!(
            "    long y4_step = (long)ib_step * {Q4K_BLOCK_ELEMENTS} * other_stride;\n"
        ));
        source.push_str(&format!("    device const float *y4 = in{other} + other_base[0] + (long)ib_first * {Q4K_BLOCK_ELEMENTS} * other_stride + (long)({lane_offset}) * other_stride;\n"));
    }
}

// ggml (llama.cpp, MIT license: https://github.com/ggml-org/llama.cpp/blob/
// master/LICENSE) `mul_vec_q_n_f32_impl<block_q4_0, N_R0_Q4_0=4>` +
// `block_q_n_dot_y(device const block_q4_0 *, ...)`
// (ggml/src/ggml-metal/kernels/mul_mv.metal), transcribed line-for-line onto
// this crate's operand-base/stride addressing.
//
// Copyright (c) 2023-2024 The ggml authors. MIT-licensed; see THIRD_PARTY.md.
//
/// `metal-q4_0-native` (NOT in omega's `metal`; `proxima-model-interop`'s `metal`
/// enables it, proxima-model-interop/Cargo.toml:153): a VERBATIM port of ggml's own `Q4_0`
/// matvec lane geometry -- TWO threads per REAL 32-element block (`ix =
/// tiisg/2`, 16 distinct block-owning lane-PAIRS per simdgroup; `il =
/// (tiisg%2)*8`, the two lanes of a pair split one block's 16 packed-nibble
/// bytes into adjacent 8-byte halves) -- NOT the K-quant-shaped `ix =
/// lane/8` split [`push_packed_row_blocked_body`]'s generic arm and
/// [`Q4_0_PAIR_DOT_MSL`] both inherit from the 256-element "virtual
/// super-block" grouping [`codec_row_block_step_bytes`] imposes on every
/// flat 32-element codec in this file so they can share one preamble with
/// the real K-quant super-block codecs. `Q4_0` has no super-block of its
/// own; ggml's kernel instead walks REAL blocks directly (`nb =
/// reduction_total / 32`, `weight_base[q] / 32`, one real
/// [`Q4_0_BLOCK_BYTES`] (18) per step), so 16 lane-PAIRS of one 32-lane
/// simdgroup read 16 contiguous blocks (288 bytes) every outer-loop
/// iteration -- see this landing's discipline row for the measured effect
/// on the E2B checkpoint's real `Q4_0` weights.
///
/// Activation gather (`yl[16]`, `sumy0`/`sumy1`): four registers are filled
/// per iteration of the inner 4-step loop (`i = 0, 2, 4, 6`), pre-scaled by
/// ggml's fixed `1/16`/`1/256`/`1/4096` factors so the nibble dot below
/// needs no runtime shift -- ggml's own comment: "we assume the yl's have
/// been multiplied with the appropriate scale factor that corresponds to
/// the missing bit shifts".
///
/// Dot product (`block_q_n_dot_y<block_q4_0>`): `d` (this block's one
/// `half` scale) read once; `qs` is the block's 16 packed-nibble bytes
/// viewed as 8 `uint16_t` words, offset by `il/2` words (0 or 4) so each
/// lane's `qs` already lands on its own half; the fixed masks
/// `0x000F`/`0x0F00`/`0x00F0`/`0xF000` pull all four nibbles a lane needs
/// with no runtime shift, and `sumy * -8.0f` folds the fixed nibble
/// zero-point (stored levels are `0..15`, the real value is `level - 8`) in
/// algebraically instead of per element.
///
/// SIMD reduction: unchanged from every other row-blocked arm, handled by
/// the shared `push_packed_row_combine_and_write` tail this function's
/// caller still invokes after it returns.
pub(super) fn push_q4_0_native_body(
    source: &mut String,
    weight: usize,
    other: usize,
    rows: usize,
    other_stride_is_one: bool,
    metal: &MetalOnlyExtras,
) {
    source.push_str("    uint ix = (uint)lane / 2u;\n");
    source.push_str("    uint il = ((uint)lane % 2u) * 8u;\n");
    let reduction_bound = packed_row_reduction_bound_token(metal);
    source.push_str(&format!(
        "    int nb = (int){reduction_bound} / {Q4_0_BLOCK_ELEMENTS};\n"
    ));
    source.push_str("    float yl[16];\n");
    source.push_str("    int ib_first = (int)ix;\n    int ib_step = 16;\n");
    source.push_str(&format!(
        "    long blk_step = (long)ib_step * {Q4_0_BLOCK_BYTES};\n"
    ));
    source.push_str(&format!("    device const uchar *blk_ptr[{rows}];\n"));
    push_full_unroll(source);
    source.push_str(&format!("    for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str(&format!(
        "        blk_ptr[q] = in{weight} + ((long)((int)weight_base[q] / {Q4_0_BLOCK_ELEMENTS}) + (long)ib_first) * {Q4_0_BLOCK_BYTES};\n"
    ));
    source.push_str("    }\n");
    if other_stride_is_one {
        source.push_str(&format!(
            "    long y4_step = (long)ib_step * {Q4_0_BLOCK_ELEMENTS};\n"
        ));
        source.push_str(&format!(
            "    device const float *y4 = in{other} + other_base[0] + (long)ib_first * {Q4_0_BLOCK_ELEMENTS} + (long)il;\n"
        ));
    } else {
        source.push_str(&format!(
            "    long y4_step = (long)ib_step * {Q4_0_BLOCK_ELEMENTS} * other_stride;\n"
        ));
        source.push_str(&format!(
            "    device const float *y4 = in{other} + other_base[0] + (long)ib_first * {Q4_0_BLOCK_ELEMENTS} * other_stride + (long)il * other_stride;\n"
        ));
    }
    source.push_str("    for (int ib = ib_first; ib < nb; ib += ib_step) {\n");
    source.push_str("        float sumy0 = 0.0f; float sumy1 = 0.0f;\n");
    if other_stride_is_one {
        push_full_unroll(source);
        source.push_str("        for (uint i = 0u; i < 8u; i += 2u) {\n");
        source.push_str("            sumy0 += y4[i + 0u] + y4[i + 1u];\n");
        source.push_str("            yl[i + 0u] = y4[i + 0u];\n");
        source.push_str("            yl[i + 1u] = y4[i + 1u] * (1.0f / 256.0f);\n");
        source.push_str("            sumy1 += y4[i + 16u] + y4[i + 17u];\n");
        source.push_str("            yl[i + 8u] = y4[i + 16u] * (1.0f / 16.0f);\n");
        source.push_str("            yl[i + 9u] = y4[i + 17u] * (1.0f / 4096.0f);\n");
        source.push_str("        }\n");
    } else {
        push_full_unroll(source);
        source.push_str("        for (uint i = 0u; i < 8u; i += 2u) {\n");
        source.push_str("            float y0 = y4[(long)(i + 0u) * other_stride];\n");
        source.push_str("            float y1 = y4[(long)(i + 1u) * other_stride];\n");
        source.push_str("            float y16 = y4[(long)(i + 16u) * other_stride];\n");
        source.push_str("            float y17 = y4[(long)(i + 17u) * other_stride];\n");
        source.push_str("            sumy0 += y0 + y1;\n");
        source.push_str("            yl[i + 0u] = y0;\n");
        source.push_str("            yl[i + 1u] = y1 * (1.0f / 256.0f);\n");
        source.push_str("            sumy1 += y16 + y17;\n");
        source.push_str("            yl[i + 8u] = y16 * (1.0f / 16.0f);\n");
        source.push_str("            yl[i + 9u] = y17 * (1.0f / 4096.0f);\n");
        source.push_str("        }\n");
    }
    source.push_str("        float sumy = sumy0 + sumy1;\n");
    push_full_unroll(source);
    source.push_str(&format!("        for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str("            device const uchar *blk = blk_ptr[q];\n");
    source.push_str("            device const half *dh = (device const half *)blk;\n");
    source.push_str("            float d = (float)dh[0];\n");
    source.push_str(
        "            device const ushort *qs = (device const ushort *)(blk + 2) + (il / 2u);\n",
    );
    source.push_str(
        "            float acc0 = 0.0f; float acc1 = 0.0f; float acc2 = 0.0f; float acc3 = 0.0f;\n",
    );
    push_full_unroll(source);
    source.push_str("            for (uint i = 0u; i < 8u; i += 2u) {\n");
    source.push_str("                ushort word = qs[i / 2u];\n");
    source.push_str("                acc0 += yl[i + 0u] * (float)(word & (ushort)0x000Fu);\n");
    source.push_str("                acc1 += yl[i + 1u] * (float)(word & (ushort)0x0F00u);\n");
    source.push_str("                acc2 += yl[i + 8u] * (float)(word & (ushort)0x00F0u);\n");
    source.push_str("                acc3 += yl[i + 9u] * (float)(word & (ushort)0xF000u);\n");
    source.push_str("            }\n");
    source.push_str(
        "            sumf[q] = sumf[q] + d * (sumy * -8.0f + acc0 + acc1 + acc2 + acc3);\n",
    );
    source.push_str("        }\n");
    push_full_unroll(source);
    source.push_str(&format!(
        "        for (int q = 0; q < {rows}; ++q) {{ blk_ptr[q] += blk_step; }}\n"
    ));
    source.push_str("        y4 += y4_step;\n");
    source.push_str("    }\n");
}

// ggml (llama.cpp, MIT license: https://github.com/ggml-org/llama.cpp/blob/
// master/LICENSE) `kernel_mul_mv_q5_K_f32_impl<2,2,32>`
// (ggml-metal.metal:5209-5324), transcribed line-for-line onto this crate's
// operand-base/stride addressing.
//
// Copyright (c) 2023-2024 The ggml authors. MIT-licensed; see THIRD_PARTY.md.
//
/// A VERBATIM port of ggml's `kernel_mul_mv_q5_K_f32_impl<nr0=2, nsg=2, nw=32>`
/// (`ggml-metal.metal:5209-5324`), selected for `Q5_K` by `PackedBodySelect::Q5KGgmlPort`.
///
/// This is the one K-quant matvec NOT rendered from its `BlockLayout` by default: the rendered
/// `Q5_K` (`SubBlockMatvecGeometry::decode_high_plane`, bit-identical to this body on four real
/// tensors, commit e436879f) measured `new/old` of 1.0110 (an MoE checkpoint's ffn_gate_exps, fast clock mode)
/// and 1.0051/1.0072 (ollama attn_v / that ffn_gate_exps, slow mode) against this body, so the
/// card's "at least as fast" rule keeps this port. The gaps sit inside 12-16% CoV.
///
/// Lane assignment (`ggml-metal.metal:5244-5250`): `tid = tiisg/4`, `ix = tiisg%4`, `iq = tid/4`,
/// `ir = tid%4`; `l0 = 8*ir`; `y_offset = 64*iq + l0` shares
/// [`push_q4k_plain_product_y4_address`]. Super-block stride `4` (`ggml-metal.metal:5263`).
/// High-bit plane (`ggml-metal.metal:5253-5256,5289-5297`): `hm1 = 1 << (2*iq)`, `hm2 = hm1<<1`,
/// `hm3 = hm1<<4`, `hm4 = hm2<<4`; `acc1[k]` sums the activation times the masked nibble byte,
/// `acc2[k]` sums the activation gated by `qh[l] & hm_k`. Final combine
/// (`ggml-metal.metal:5299-5305`): `dall`/`dmin` applied once per sub-block pair.
#[allow(clippy::too_many_arguments)]
pub(super) fn push_q5k_ggml_port_body(
    source: &mut String,
    weight: usize,
    other: usize,
    rows: usize,
    block_bytes: usize,
    other_stride_is_one: bool,
    metal: &MetalOnlyExtras,
) {
    source.push_str("    uint tid = (uint)lane / 4u;\n");
    source.push_str("    uint ix = (uint)lane % 4u;\n");
    source.push_str("    uint iq = tid / 4u;\n");
    source.push_str("    uint ir = tid % 4u;\n");
    source.push_str("    uint l0 = 8u * ir;\n");
    source.push_str("    uint q_offset = 32u * iq + l0;\n");
    source.push_str("    uchar hm1 = (uchar)(1u << (2u * iq));\n");
    source.push_str("    uchar hm2 = (uchar)(hm1 << 1u);\n");
    source.push_str("    uchar hm3 = (uchar)(hm1 << 4u);\n");
    source.push_str("    uchar hm4 = (uchar)(hm2 << 4u);\n");
    let reduction_bound = packed_row_reduction_bound_token(metal);
    source.push_str(&format!(
        "    int super_blocks = (int){reduction_bound} / {Q4K_BLOCK_ELEMENTS};\n"
    ));
    source.push_str("    float yl[16];\n");
    source.push_str("    float yh[16];\n");
    source.push_str("    int ib_first = (int)ix;\n    int ib_step = 4;\n");
    source.push_str(&format!(
        "    long blk_step = (long)ib_step * {block_bytes};\n"
    ));
    source.push_str(&format!("    device const uchar *blk_ptr[{rows}];\n"));
    source.push_str(&format!("    for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str(&format!(
        "        blk_ptr[q] = in{weight} + ((long)((int)weight_base[q] / {Q4K_BLOCK_ELEMENTS}) + (long)ib_first) * {block_bytes};\n"
    ));
    source.push_str("    }\n");
    push_q4k_plain_product_y4_address(source, other, other_stride_is_one);
    source.push_str("    for (int ib = ib_first; ib < super_blocks; ib += ib_step) {\n");
    source.push_str(
        "        float sumy0 = 0.0f; float sumy1 = 0.0f; float sumy2 = 0.0f; float sumy3 = 0.0f;\n",
    );
    if other_stride_is_one {
        source.push_str(
            "        for (uint i = 0u; i < 8u; ++i) {\n            yl[i] = y4[i]; sumy0 += yl[i];\n",
        );
        source.push_str("            yl[i + 8u] = y4[i + 32u]; sumy1 += yl[i + 8u];\n");
        source.push_str("            yh[i] = y4[i + 128u]; sumy2 += yh[i];\n");
        source.push_str("            yh[i + 8u] = y4[i + 160u]; sumy3 += yh[i + 8u];\n");
    } else {
        source.push_str(
            "        for (uint i = 0u; i < 8u; ++i) {\n            yl[i] = y4[(long)i * other_stride]; sumy0 += yl[i];\n",
        );
        source.push_str(
            "            yl[i + 8u] = y4[(long)(i + 32u) * other_stride]; sumy1 += yl[i + 8u];\n",
        );
        source
            .push_str("            yh[i] = y4[(long)(i + 128u) * other_stride]; sumy2 += yh[i];\n");
        source.push_str(
            "            yh[i + 8u] = y4[(long)(i + 160u) * other_stride]; sumy3 += yh[i + 8u];\n",
        );
    }
    source.push_str("        }\n");
    source.push_str(&format!("        for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str("            device const uchar *blk = blk_ptr[q];\n");
    source
        .push_str("            device const ushort *sc = (device const ushort *)(blk + 4) + iq;\n");
    source.push_str("            device const uchar *qh = blk + 16u + l0;\n");
    source.push_str("            device const uchar *q1 = blk + 48u + q_offset;\n");
    source.push_str("            device const uchar *q2 = q1 + 64u;\n");
    source.push_str("            device const half *dh = (device const half *)blk;\n");
    source.push_str("            ushort sc16_0 = sc[0] & (ushort)0x3f3fu;\n");
    source.push_str("            ushort sc16_1 = sc[2] & (ushort)0x3f3fu;\n");
    source.push_str(
        "            ushort sc16_2 = (ushort)(((sc[4] >> 0) & (ushort)0x0f0fu) | ((sc[0] & (ushort)0xc0c0u) >> 2));\n",
    );
    source.push_str(
        "            ushort sc16_3 = (ushort)(((sc[4] >> 4) & (ushort)0x0f0fu) | ((sc[2] & (ushort)0xc0c0u) >> 2));\n",
    );
    source.push_str(
        "            uchar sc8_0 = (uchar)(sc16_0 & 0xffu); uchar sc8_1 = (uchar)(sc16_0 >> 8);\n",
    );
    source.push_str(
        "            uchar sc8_2 = (uchar)(sc16_1 & 0xffu); uchar sc8_3 = (uchar)(sc16_1 >> 8);\n",
    );
    source.push_str(
        "            uchar sc8_4 = (uchar)(sc16_2 & 0xffu); uchar sc8_5 = (uchar)(sc16_2 >> 8);\n",
    );
    source.push_str(
        "            uchar sc8_6 = (uchar)(sc16_3 & 0xffu); uchar sc8_7 = (uchar)(sc16_3 >> 8);\n",
    );
    source.push_str(
        "            float acc1_0 = 0.0f; float acc1_1 = 0.0f; float acc1_2 = 0.0f; float acc1_3 = 0.0f;\n",
    );
    source.push_str(
        "            float acc2_0 = 0.0f; float acc2_1 = 0.0f; float acc2_2 = 0.0f; float acc2_3 = 0.0f;\n",
    );
    source.push_str("            for (uint l = 0u; l < 8u; ++l) {\n");
    source.push_str("                uchar h = qh[l];\n");
    source.push_str("                acc1_0 += yl[l] * (float)(q1[l] & 0x0Fu);\n");
    source.push_str("                acc1_1 += yl[l + 8u] * (float)(q1[l] & 0xF0u);\n");
    source.push_str("                acc1_2 += yh[l] * (float)(q2[l] & 0x0Fu);\n");
    source.push_str("                acc1_3 += yh[l + 8u] * (float)(q2[l] & 0xF0u);\n");
    source.push_str("                acc2_0 += (h & hm1) != 0u ? yl[l] : 0.0f;\n");
    source.push_str("                acc2_1 += (h & hm2) != 0u ? yl[l + 8u] : 0.0f;\n");
    source.push_str("                acc2_2 += (h & hm3) != 0u ? yh[l] : 0.0f;\n");
    source.push_str("                acc2_3 += (h & hm4) != 0u ? yh[l + 8u] : 0.0f;\n");
    source.push_str("            }\n");
    source.push_str("            float dall = (float)dh[0];\n");
    source.push_str("            float dmin = (float)dh[1];\n");
    source.push_str(
        "            sumf[q] = sumf[q] + dall * ((float)sc8_0 * (acc1_0 + 16.0f * acc2_0) +\n",
    );
    source.push_str(
        "                                       (float)sc8_1 * (acc1_1 * (1.0f/16.0f) + 16.0f * acc2_1) +\n",
    );
    source.push_str(
        "                                       (float)sc8_4 * (acc1_2 + 16.0f * acc2_2) +\n",
    );
    source.push_str(
        "                                       (float)sc8_5 * (acc1_3 * (1.0f/16.0f) + 16.0f * acc2_3)) -\n",
    );
    source.push_str(
        "                      dmin * (sumy0 * (float)sc8_2 + sumy1 * (float)sc8_3 + sumy2 * (float)sc8_6 + sumy3 * (float)sc8_7);\n",
    );
    source.push_str("        }\n");
    source.push_str(&format!(
        "        for (int q = 0; q < {rows}; ++q) {{ blk_ptr[q] += blk_step; }}\n"
    ));
    source.push_str("        y4 += y4_step;\n");
    source.push_str("    }\n");
}

/// Emits the pragma that makes the next `for` unroll completely. ggml spells
/// this `FOR_UNROLL` on every row, lane and nibble loop of its matvec bodies:
/// a loop that stays rolled indexes `sumf`, `yl` and `blk_ptr` with a runtime
/// value, and a private array indexed that way is placed in memory instead of
/// registers. Unrolling reorders no floating-point operation.
fn push_full_unroll(source: &mut String) {
    source.push_str(FULL_UNROLL);
    source.push('\n');
}

/// The per-element row-blocked loop for a flat 32-element-block codec when the reduce is
/// not a plain product: the same text the `Q8_0`/`Q4_0` fallback arms emit, with the
/// codec's `(superblock, index)` accessor named by `element_fn`.
fn push_super_element_fallback(
    source: &mut String,
    resolved: &BoundOp,
    reduce_op: ScalarOp,
    block: &PackedRowBlock,
    element_type: &str,
    element_fn: &str,
) {
    let (weight, other) = (block.weight, block.other);
    let elements = codec_block_elements(block.codec);
    let scratch_len = resolved.operands().len().max(1);
    source.push_str(&format!(
        "            for (int e = 0; e < {elements}; ++e) {{\n"
    ));
    source.push_str(&format!(
        "                {element_type} scratch[{scratch_len}];\n"
    ));
    source.push_str(&format!(
        "                scratch[{weight}] = {element_fn}(blk, slot + (uint)e);\n"
    ));
    source.push_str(&format!("                scratch[{other}] = acts[e];\n"));
    let value_expr = push_body_steps(
        source,
        resolved.element_body(),
        "                ",
        element_type,
    );
    source.push_str(&format!(
        "                {element_type} value = {value_expr};\n"
    ));
    let combine_expr = scalar_op_expr(reduce_op, &["sumf[q]", "value"]);
    source.push_str(&format!("                sumf[q] = {combine_expr};\n"));
    source.push_str("            }\n");
}
