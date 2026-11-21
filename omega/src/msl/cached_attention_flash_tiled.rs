use super::flash_refusal::FlashRefusal;
use super::layout_render::{hardware_type, readable_scalar_cache};
use super::*;
use proxima_numeric::float_layout::FloatLayout;

/// Simdgroups per flash threadgroup: llama.cpp's `nsg` for a head of fewer than 512 elements,
/// read from `[attention_rows].flash_simdgroups` in `omega-runtime.toml`.
pub(super) const FLASH_SIMDGROUPS: u64 = crate::sized::ATTENTION_ROWS_FLASH_SIMDGROUPS;

/// The head widths flash is verified against Legacy for: the three served attention widths. Each is
/// a whole number of float4 per lane of one simdgroup (`head_dim / 128` per lane), which is what
/// the body's O rescale and row write step by.
const FLASH_HEAD_DIMS: [u64; 3] = [128, 256, 512];

/// Keys one flash block walks for `head_dim`: the head's row-tiled block, because bit parity with
/// Legacy needs the same block boundaries (the online-softmax rescale lands per block).
fn flash_block(head_dim: u64) -> u64 {
    row_tiled_block(head_dim)
}

/// Whether the flash body stages Q in threadgroup memory for `head_dim` under `budget`: it does
/// while the stage `sq` (`16 * (head_dim / 2 + 8)` floats) fits beside the output and scores, and
/// otherwise reads Q from device f32, as Legacy does once its query tile is over
/// `query_tile_staged`'s limits (`signature_tokens_prelude.rs`). Both are binary32 reads of the same
/// values, so the choice moves no rounding.
fn flash_stages_query(head_dim: u64, budget: u64) -> bool {
    flash_bytes(head_dim, true) <= budget
}

fn flash_bytes(head_dim: u64, staged: bool) -> u64 {
    let stage = if staged { 16 * (head_dim / 2 + 8) } else { 0 };
    let floats = stage + 8 * head_dim + 8 * flash_block(head_dim);
    floats * core::mem::size_of::<f32>() as u64
}

/// Static threadgroup bytes the flash body declares for `head_dim` within `budget`: the Q stage
/// `sq` when [`flash_stages_query`] (`16 * (head_dim / 2 + 8)` floats), the output `so`
/// (`8 * head_dim`) and the scores `ss` (`8 * block`), each one accumulator scalar wide.
pub(super) fn flash_threadgroup_bytes_within(head_dim: u64, budget: u64) -> u64 {
    flash_bytes(head_dim, flash_stages_query(head_dim, budget))
}

/// The query heads per kv head the flash body's fragment map assumes.
const FLASH_QUERY_GROUPS: u64 = 8;

/// Admits `kind` for the flash body, or names the one condition it fails: a sinked, causal op,
/// full or sliding-window over the cached keys, of eight query heads per kv head. Conditions are checked in the order of [`FlashRefusal`]; the first
/// failure is the reason. Head dims 128, 256 and 512 are the shapes verified; 512 reads Q
/// from device (its Q stage would not fit); a shape whose tiles exceed the budget even so fails
/// `ThreadgroupBytes`.
///
/// Any query row count is admitted: the keys of the new rows past the last whole tile run the
/// scalar `flash_tail` step, as the row-tiled kernel's `mode == 2`. Tile loads are guarded by the
/// block's live column count, so a tile never starts past the live keys; the K and V buffers are
/// bound from the cached extent (`proxima-tensor/src/physical.rs:47`, `cached_key_width =
/// cached_key_rows * head_dim`), so a tile that ends past the live keys loads only the rows inside them (`load_k_rows` clamps the key row as the row-tiled kernel's `omega_load_key_matrix`, `load_v_rows` zeroes the rest), whatever the extent. Keys between the live length and the tile end
/// are masked to -inf by `visible` and their V rows zeroed, as the row-tiled kernel.
pub(super) fn flash_admission(kind: &BoundOpKind) -> Result<(), FlashRefusal> {
    flash_admission_within(kind, crate::sized::CACHED_ATTENTION_THREADGROUP_MEMORY_BYTES)
}

/// [`flash_admission`] against an explicit threadgroup-memory `budget`.
pub(super) fn flash_admission_within(kind: &BoundOpKind, budget: u64) -> Result<(), FlashRefusal> {
    let BoundOpKind::CachedAttention {
        query_rows,
        new_key_rows,
        query_groups,
        head_dim,
        new_upper_inclusive,
        attention_sink,
        ..
    } = kind
    else {
        return Err(FlashRefusal::NotCachedAttention);
    };
    let checks = [
        (*attention_sink, FlashRefusal::NoAttentionSink),
        (*query_groups == FLASH_QUERY_GROUPS, FlashRefusal::QueryGroups),
        (FLASH_HEAD_DIMS.contains(head_dim), FlashRefusal::HeadDim),
        (*new_upper_inclusive == 0, FlashRefusal::NewKeysNotCausal),
        (query_rows == new_key_rows, FlashRefusal::QueryRowsDifferFromNewKeys),
        (
            flash_threadgroup_bytes_within(*head_dim, budget) <= budget,
            FlashRefusal::ThreadgroupBytes,
        ),
    ];
    checks
        .into_iter()
        .find_map(|(holds, refusal)| (!holds).then_some(refusal))
        .map_or_else(|| legacy_split_refusal(kind), Err)
}

/// Refuses an op whose Legacy row-tiled form splits the key range: the same `rows_per_threadgroup_for` and
/// `row_tiled_splits` Legacy's `row_tiled_form` calls, over Legacy's simdgroup count. One rule for `Flash` and
/// `FlashWhereAdmitted`, so a flash op never runs one pass where Legacy runs several.
fn legacy_split_refusal(kind: &BoundOpKind) -> Result<(), FlashRefusal> {
    let BoundOpKind::CachedAttention {
        query_rows,
        cached_key_rows,
        new_key_rows,
        kv_heads,
        query_groups,
        head_dim,
        ..
    } = kind
    else {
        return Err(FlashRefusal::NotCachedAttention);
    };
    let simdgroups = row_tiled_simdgroups(*head_dim);
    let capacity = cached_key_rows + new_key_rows;
    let rows_per_threadgroup =
        rows_per_threadgroup_for(*query_rows, *kv_heads, *query_groups, *head_dim, capacity, simdgroups);
    let tiles = query_rows.div_ceil(rows_per_threadgroup);
    match row_tiled_splits(capacity, *kv_heads, tiles, simdgroups) {
        1 => Ok(()),
        splits => Err(FlashRefusal::LegacySplits { splits }),
    }
}

/// The flash-decomposition kernel (`AttentionDecomposition::Flash`) for the cached-attention
/// contract the row-tiled form dispatches: same buffers, same `Uniforms`, grid `kv_heads * rows`
/// threadgroups of `FLASH_SIMDGROUPS` simdgroups. One vector block (the eight heads of one query
/// row) per threadgroup, Q staged once in threadgroup memory, the online-softmax state in
/// registers, the output in threadgroup memory rescaled by the row's owning simdgroup.
///
/// Operand and accumulator types are read off proxima-numeric layouts through
/// [`hardware_type`], the way [`super::layout_render`] spells every other kernel type; no scalar
/// type is written in the template.
pub(super) fn render_cached_attention_flash_tiled(
    resolved: &BoundOp,
    entry: &str,
    cached_kv_codec: Option<Codec>,
    qk_operand: Option<FloatLayout>,
) -> Result<String, EmitError> {
    let BoundOpKind::CachedAttention {
        kv_heads,
        query_groups,
        head_dim,
        scale,
        cached_lower_inclusive,
        ..
    } = &resolved.kind
    else {
        return Err(EmitError::RenderKindMismatch {
            node: resolved.node,
            expected: "cached_attention_flash_tiled",
            found: resolved.kind.name(),
        });
    };
    flash_admission(&resolved.kind).map_err(|refusal| {
        EmitError::CachedAttentionVariantAxisNotSupported {
            axis: "decomposition",
            value: refusal.reason(),
        }
    })?;
    let accumulator = hardware_type(&FloatLayout::BF32).ok_or(
        EmitError::CachedAttentionKvCodecNotSupported {
            node: resolved.node,
            reason: "the binary32 layout has no Metal hardware type",
        },
    )?;
    let cache_layout = cached_kv_codec
        .as_ref()
        .and_then(readable_scalar_cache)
        .map(|(layout, _)| layout);
    let element = cache_layout
        .and_then(|layout| hardware_type(&layout))
        .filter(|element| *element != accumulator)
        .ok_or(EmitError::CachedAttentionKvCodecNotSupported {
            node: resolved.node,
            reason: "flash reads a narrow scalar cache through the matrix hardware type of its layout",
        })?;
    let cached_lower = if *cached_lower_inclusive == i64::MIN {
        "-9223372036854775807L".to_string()
    } else {
        format!("{cached_lower_inclusive}L")
    };
    let half_dim = head_dim / 2;
    let staged = flash_stages_query(*head_dim, crate::sized::CACHED_ATTENTION_THREADGROUP_MEMORY_BYTES);
    let (q_space, q_stride, q_decl, q_setup) = if staged {
        ("threadgroup", half_dim + 8, STAGED_Q_DECL, STAGED_Q_SETUP)
    } else {
        ("device", half_dim, "", DEVICE_Q_SETUP)
    };
    let operand_layout = if staged {
        qk_operand.or(cache_layout).unwrap_or(FloatLayout::BF32)
    } else {
        FloatLayout::BF32
    };
    let operand = hardware_type(&operand_layout).ok_or(EmitError::CachedAttentionKvCodecNotSupported {
        node: resolved.node,
        reason: "the flash QK operand layout has no Metal hardware type",
    })?;
    let widened = operand == accumulator;
    let (qk_helpers, qk_frags, qk_load, qk_load_rows, qe_read, qo_read) = if widened {
        (
            "",
            "@ACC_FRAG@ qe, qo; typename k_frag<T>::type kef, kof;",
            "load_k",
            "load_k_rows",
            "qe4[index]".to_string(),
            "qo4[index]".to_string(),
        )
    } else {
        (
            QK_OPERAND_HELPERS,
            "@QOP_FRAG@ qe, qo, kef, kof;",
            "load_qk",
            "load_qk_rows",
            format!("{accumulator}4(qe4[index])"),
            format!("{accumulator}4(qo4[index])"),
        )
    };
    let simdgroups = FLASH_SIMDGROUPS;
    let block = flash_block(*head_dim);
    let substitutions = [
        ("@QK_HELPERS@", qk_helpers.to_string()),
        ("@QK_FRAGS@", qk_frags.to_string()),
        ("@QK_LOAD_ROWS@", qk_load_rows.to_string()),
        ("@QK_LOAD@", qk_load.to_string()),
        ("@QE4_READ@", qe_read),
        ("@QO4_READ@", qo_read),
        ("@ENTRY@", entry.to_string()),
        ("@QSPACE@", q_space.to_string()),
        ("@QSTRIDE@", q_stride.to_string()),
        ("@Q_DECL@", q_decl.to_string()),
        ("@Q_SETUP@", q_setup.to_string()),
        ("@KV_HEADS@", kv_heads.to_string()),
        ("@GROUPS@", query_groups.to_string()),
        ("@HEAD_DIM@", head_dim.to_string()),
        ("@HALF_DIM@", half_dim.to_string()),
        ("@NSG@", simdgroups.to_string()),
        ("@BLOCK@", block.to_string()),
        ("@NQ@", (query_groups / simdgroups).to_string()),
        ("@NO@", (head_dim / 8 / simdgroups).to_string()),
        ("@LANE_VECTORS@", (head_dim / 4 / 32).to_string()),
        ("@QK_PER_SG@", (block / 8 / simdgroups).to_string()),
        ("@ITEMS@", (block / 32).to_string()),
        ("@THREADS@", (simdgroups * 32).to_string()),
        ("@SCALE@", msl_literal(*scale)),
        ("@CACHED_LOWER@", cached_lower),
        ("@ALIGN_ROWS@", flash_row_alignment(&resolved.kind).to_string()),
        ("@QOP_FRAG@", format!("simdgroup_{operand}8x8")),
        ("@QOP@", operand.to_string()),
        ("@ELEM_FRAG@", format!("simdgroup_{element}8x8")),
        ("@ACC_FRAG@", format!("simdgroup_{accumulator}8x8")),
        ("@ELEM@", element.to_string()),
        ("@ACC@", accumulator.to_string()),
    ];
    let mut source = String::new();
    preamble(&mut source);
    let mut body = FLASH_TILED_KERNEL.to_string();
    for (token, value) in &substitutions {
        body = body.replace(token, value);
    }
    source.push_str(&body);
    Ok(source)
}

const QK_OPERAND_HELPERS: &str = r#"
template <typename T> inline void load_qk(thread @QOP_FRAG@& dst, device const T* src, int stride) {
    simdgroup_matrix<T, 8, 8> wide; simdgroup_load(wide, src, (ulong)stride, ulong2(0, 0), true);
    dst.thread_elements()[0] = @QOP@(wide.thread_elements()[0]); dst.thread_elements()[1] = @QOP@(wide.thread_elements()[1]);
}
inline void load_qk(thread @QOP_FRAG@& dst, device const @QOP@* src, int stride) { simdgroup_load(dst, src, (ulong)stride, ulong2(0, 0), true); }
template <typename T> inline void load_qk_rows(thread @QOP_FRAG@& dst, device const T* src, int stride, ushort lane, int rows_left) {
    short quad = (short)(lane / 4); short row = (short)((quad & 4) + ((lane / 2) % 4)); short column = (short)((quad & 2) * 2 + (lane % 2) * 2);
    int last_key = max(rows_left - 1, 0);
    dst.thread_elements()[0] = @QOP@(src[min((int)column, last_key) * stride + row]);
    dst.thread_elements()[1] = @QOP@(src[min((int)column + 1, last_key) * stride + row]);
}"#;

const STAGED_Q_DECL: &str = "threadgroup @QOP@ sq[16 * (half_dim + 8)];";

const STAGED_Q_SETUP: &str = r#"for (int i = (int)sg * 32 + (int)lane; i < 8 * half_dim; i += @THREADS@) {
        int v = i / half_dim, c = i % half_dim;
        sq[v * qstride + c] = @QOP@(in0[qbase + i]); sq[(8 + v) * qstride + c] = @QOP@(in1[qbase + i]);
    }
    threadgroup const @QOP@* qe_base = sq; threadgroup const @QOP@* qo_base = sq + 8 * qstride;"#;

const DEVICE_Q_SETUP: &str = "device const @ACC@* qe_base = in0 + qbase; device const @ACC@* qo_base = in1 + qbase;";

const FLASH_TILED_KERNEL: &str = r#"#define FOR_UNROLL _Pragma("clang loop unroll(full)")
struct Uniforms { long total_elements; long splits; };
constant constexpr int kv_heads = @KV_HEADS@, groups = @GROUPS@, head_dim = @HEAD_DIM@, half_dim = @HALF_DIM@;
constant constexpr int nsg = @NSG@, block = @BLOCK@, nq = @NQ@, no = @NO@, lane_vectors = @LANE_VECTORS@, qstride = @QSTRIDE@, tiles = block / 8;
constant constexpr @ACC@ scale = @SCALE@f;

template <typename T> struct k_frag;
template <> struct k_frag<@ELEM@> { typedef @ACC_FRAG@ type; };
template <> struct k_frag<@ACC@> { typedef @ACC_FRAG@ type; };

inline void load_k(thread @ACC_FRAG@& dst, device const @ELEM@* src, int stride) {
    @ELEM_FRAG@ narrow; simdgroup_load(narrow, src, (ulong)stride, ulong2(0, 0), true);
    dst.thread_elements()[0] = @ACC@(narrow.thread_elements()[0]); dst.thread_elements()[1] = @ACC@(narrow.thread_elements()[1]);
}
inline void load_k(thread @ACC_FRAG@& dst, device const @ACC@* src, int stride) { simdgroup_load(dst, src, (ulong)stride, ulong2(0, 0), true); }
inline void load_v(thread @ACC_FRAG@& dst, device const @ELEM@* src, int stride) {
    @ELEM_FRAG@ narrow; simdgroup_load(narrow, src, (ulong)stride);
    dst.thread_elements()[0] = @ACC@(narrow.thread_elements()[0]); dst.thread_elements()[1] = @ACC@(narrow.thread_elements()[1]);
}
inline void load_v(thread @ACC_FRAG@& dst, device const @ACC@* src, int stride) { simdgroup_load(dst, src, (ulong)stride); }@QK_HELPERS@

// a tile that ends past the live keys reads only the rows inside them, as the row-tiled kernel's clamped key load and zero-padded value rows: the cache extent need not be a whole number of tiles
template <typename T>
inline void load_k_rows(thread @ACC_FRAG@& dst, device const T* src, int stride, ushort lane, int rows_left) {
    short quad = (short)(lane / 4); short row = (short)((quad & 4) + ((lane / 2) % 4)); short column = (short)((quad & 2) * 2 + (lane % 2) * 2);
    int last_key = max(rows_left - 1, 0);
    dst.thread_elements()[0] = @ACC@(src[min((int)column, last_key) * stride + row]);
    dst.thread_elements()[1] = @ACC@(src[min((int)column + 1, last_key) * stride + row]);
}
template <typename T>
inline void load_v_rows(thread @ACC_FRAG@& dst, device const T* src, int stride, ushort lane, int rows_left) {
    short quad = (short)(lane / 4); short row = (short)((quad & 4) + ((lane / 2) % 4)); short column = (short)((quad & 2) * 2 + (lane % 2) * 2);
    dst.thread_elements()[0] = row < rows_left ? @ACC@(src[row * stride + column]) : 0.0f;
    dst.thread_elements()[1] = row < rows_left ? @ACC@(src[row * stride + column + 1]) : 0.0f;
}

inline void zero_rows(thread @ACC_FRAG@& frag, int key_start, int valid_end, ushort lane) {
    short quad = (short)(lane / 4); short frag_row = (short)((quad & 4) + ((lane / 2) % 4));
    if (key_start + frag_row >= valid_end) { frag.thread_elements()[0] = 0.0f; frag.thread_elements()[1] = 0.0f; }
}

inline void flash_softmax(threadgroup @ACC@* ss, threadgroup @ACC@* so, thread @ACC@ (&M)[nq], thread @ACC@ (&S)[nq], int key0, int columns, int valid_end,
                          bool cached_phase, int row, long lower, ushort lane, ushort sg) {
    FOR_UNROLL for (int jj = 0; jj < nq; ++jj) {                      // softmax: this simdgroup owns vectors jj*nsg + sg
        int v = jj * nsg + sg;
        @ACC@ local[@ITEMS@]; @ACC@ bm = -INFINITY;
        FOR_UNROLL for (int item = 0; item < @ITEMS@; ++item) {
            int column = (int)lane + item * 32; @ACC@ raw = -INFINITY;
            if (item * 32 < columns) {
                int key = key0 + column;
                bool visible = cached_phase ? (key < valid_end && (long)(key - valid_end - row) >= lower)
                                            : (column < columns && key <= row && (long)(key - row) >= lower);
                if (visible) { raw = ss[v * block + column] * scale; }
            }
            local[item] = raw; bm = max(bm, raw);
        }
        bm = simd_max(bm);
        @ACC@ prev = M[jj]; @ACC@ next = max(prev, bm);
        @ACC@ rescale = (prev == -INFINITY) ? 0.0f : exp(prev - next);
        @ACC@ bs = 0.0f;
        FOR_UNROLL for (int item = 0; item < @ITEMS@; ++item) {
            @ACC@ w = (local[item] == -INFINITY) ? 0.0f : exp(local[item] - next);
            ss[v * block + (int)lane + item * 32] = w; bs += w;
        }
        bs = simd_sum(bs);
        M[jj] = next; S[jj] = S[jj] * rescale + bs;
        threadgroup @ACC@4* o4 = (threadgroup @ACC@4*)(so + v * head_dim);
        FOR_UNROLL for (int part = 0; part < lane_vectors; ++part) { o4[part * 32 + lane] *= rescale; }   // O rescale by the row owner, 32 lanes x float4 x lane_vectors dims
    }
}

template <typename T, bool partial>
inline void flash_block(device const T* ke, device const T* ko, device const T* vv, int key0, int columns, int valid_end, bool cached_phase, int row, long lower,
                        @QSPACE@ const @QOP@* qe_base, @QSPACE@ const @QOP@* qo_base, threadgroup @ACC@* ss, threadgroup @ACC@* so, thread @ACC@ (&M)[nq], thread @ACC@ (&S)[nq],
                        int kv_head, ushort lane, ushort sg) {
    const int kstride = kv_heads * half_dim, vstride = kv_heads * head_dim;
    FOR_UNROLL for (int cc = 0; cc < @QK_PER_SG@; ++cc) {                       // QK: key tiles sg and sg + 4
        int kt = cc * nsg + sg;
        {
            @ACC_FRAG@ acc = make_filled_simdgroup_matrix<@ACC@, 8, 8>(0.0f);
            if (kt * 8 < columns) {                                   // a tile wholly past the live keys loads nothing, as the row-tiled kernel
                int key_rows_left = valid_end - (key0 + kt * 8);
                device const T* kep = ke + (key0 + kt * 8) * kstride + kv_head * half_dim;
                device const T* kop = ko + (key0 + kt * 8) * kstride + kv_head * half_dim;
                FOR_UNROLL for (int j = 0; j < half_dim / 8; ++j) {   // chain e0,o0,e1,o1,... as the row-tiled kernel
                    simdgroup_barrier(mem_flags::mem_none);
                    @QK_FRAGS@
                    simdgroup_load(qe, qe_base + 8 * j, (ulong)qstride);
                    simdgroup_load(qo, qo_base + 8 * j, (ulong)qstride);
                    if (!partial || key_rows_left >= 8) {
                        @QK_LOAD@(kef, kep + 8 * j, kstride);
                        @QK_LOAD@(kof, kop + 8 * j, kstride);
                    } else {
                        @QK_LOAD_ROWS@(kef, kep + 8 * j, kstride, lane, key_rows_left);
                        @QK_LOAD_ROWS@(kof, kop + 8 * j, kstride, lane, key_rows_left);
                    }
                    simdgroup_barrier(mem_flags::mem_none);
                    simdgroup_multiply_accumulate(acc, qe, kef, acc);
                    simdgroup_multiply_accumulate(acc, qo, kof, acc);
                }
            }
            simdgroup_store(acc, ss + kt * 8, (ulong)block);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    flash_softmax(ss, so, M, S, key0, columns, valid_end, cached_phase, row, lower, lane, sg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    @ACC_FRAG@ lo[no];                                       // PV: this simdgroup owns dimension blocks sg + nsg * i
    FOR_UNROLL for (int i = 0; i < no; ++i) { simdgroup_load(lo[i], so + 8 * (sg + nsg * i), (ulong)head_dim); }
    FOR_UNROLL for (int kt = 0; kt < tiles; ++kt) {
        if (kt * 8 < columns) {
            @ACC_FRAG@ p; simdgroup_load(p, ss + kt * 8, (ulong)block);
            FOR_UNROLL for (int i = 0; i < no; ++i) {
                @ACC_FRAG@ vf;
                device const T* vp = vv + (key0 + kt * 8) * vstride + kv_head * head_dim + 8 * (sg + nsg * i);
                int value_rows_left = key0 + columns - (key0 + kt * 8);
                if (!partial || value_rows_left >= 8) { load_v(vf, vp, vstride); } else { load_v_rows(vf, vp, vstride, lane, value_rows_left); }
                zero_rows(vf, key0 + kt * 8, key0 + columns, lane);
                simdgroup_multiply_accumulate(lo[i], p, vf, lo[i]);
            }
        }
    }
    FOR_UNROLL for (int i = 0; i < no; ++i) { simdgroup_store(lo[i], so + 8 * (sg + nsg * i), (ulong)head_dim); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void flash_tail(device const @ACC@* ke, device const @ACC@* ko, device const @ACC@* vv, int key0, int columns, int row, long lower,
                       @QSPACE@ const @QOP@* qe_base, @QSPACE@ const @QOP@* qo_base, threadgroup @ACC@* ss, threadgroup @ACC@* so, thread @ACC@ (&M)[nq], thread @ACC@ (&S)[nq],
                       int kv_head, ushort lane, ushort sg) {
    const int kstride = kv_heads * half_dim, vstride = kv_heads * head_dim;
    const int pairs = 8 * columns;                                    // the new keys past the last whole 8-key tile: the row-tiled scalar step
    short ty = (short)(lane / 8), tx = (short)(lane % 8);
    for (int group = (int)sg; group * 4 < pairs; group += nsg) {
        int pair = group * 4 + (int)ty;
        int v = pair / columns, column = pair % columns;
        @ACC@ partial = 0.0f;
        if (pair < pairs) {
            @QSPACE@ const @QOP@4* qe4 = (@QSPACE@ const @QOP@4*)(qe_base + v * qstride);
            @QSPACE@ const @QOP@4* qo4 = (@QSPACE@ const @QOP@4*)(qo_base + v * qstride);
            device const @ACC@4* ke4 = (device const @ACC@4*)(ke + (key0 + column) * kstride + kv_head * half_dim);
            device const @ACC@4* ko4 = (device const @ACC@4*)(ko + (key0 + column) * kstride + kv_head * half_dim);
            for (short index = tx; index < (short)(half_dim / 4); index += 8) {
                partial += dot(ke4[index], @QE4_READ@);
                partial += dot(ko4[index], @QO4_READ@);
            }
        }
        partial += simd_shuffle_down(partial, 4);
        partial += simd_shuffle_down(partial, 2);
        partial += simd_shuffle_down(partial, 1);
        if (tx == 0 && pair < pairs) { ss[v * block + column] = partial; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    flash_softmax(ss, so, M, S, key0, columns, key0 + columns, false, row, lower, lane, sg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int element = (int)sg * 32 + (int)lane; element < 8 * head_dim; element += @THREADS@) {
        int v = element / head_dim, dim = element % head_dim;
        @ACC@ sum = 0.0f;                                             // sum then add, as the row-tiled scalar step
        for (int column = 0; column < columns; ++column) { sum += ss[v * block + column] * vv[(key0 + column) * vstride + kv_head * head_dim + dim]; }
        so[element] += sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void @ENTRY@(device const @ACC@* in0 [[buffer(0)]], device const @ACC@* in1 [[buffer(1)]],
    device const @ELEM@* in2 [[buffer(2)]], device const @ELEM@* in3 [[buffer(3)]],
    device const @ACC@* in4 [[buffer(4)]], device const @ACC@* in5 [[buffer(5)]],
    device const @ELEM@* in6 [[buffer(6)]], device const @ACC@* in7 [[buffer(7)]],
    device const @ACC@* in8 [[buffer(8)]], device const @ACC@* in9 [[buffer(9)]],
    device @ACC@* out [[buffer(10)]], constant Uniforms& u [[buffer(11)]],
    uint tgid [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    int rows = (int)(u.total_elements / (kv_heads * groups));
    if ((int)tgid >= kv_heads * rows) { return; }
    int row = rows - 1 - ((int)tgid % rows);                          // heaviest first, as the row-tiled form
    int kv_head = (int)tgid / rows;
    int live = (int)in8[0];
    @Q_DECL@ threadgroup @ACC@4 so4[8 * head_dim / 4]; threadgroup @ACC@ ss[8 * block];
    threadgroup @ACC@* so = (threadgroup @ACC@*)so4;
    int qbase = (row * (kv_heads * groups) + kv_head * groups) * half_dim;
    @Q_SETUP@
    for (int i = (int)sg * 32 + (int)lane; i < 8 * head_dim; i += @THREADS@) { so[i] = 0.0f; }
    @ACC@ M[nq], S[nq];
    FOR_UNROLL for (int jj = 0; jj < nq; ++jj) {
        @ACC@ sink = in9[kv_head * groups + jj * nsg + sg];
        M[jj] = sink; S[jj] = (sink == -INFINITY) ? 0.0f : 1.0f;      // sink = softmax initial state, as SINK_INIT
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr long cached_lower = @CACHED_LOWER@;                     // in the kernel body so the index narrowing rewrites it with the arithmetic below
    constexpr long tile_rows = @ALIGN_ROWS@L;                          // legacy's rows per threadgroup; flash launches one row but keeps its block grid
    long row0 = (long)row - (long)row % tile_rows;
    long first_key = max(0L, (long)live + cached_lower + row0) & ~7L;   // legacy block boundaries start at the window, not at 0
    for (long key0 = first_key; key0 < live; key0 += block) {
        int columns = min(block, live - (int)key0);                    // only the last block can end inside a tile: whole-tile blocks take the instance with no per-tile branch
        if ((columns & 7) == 0) { flash_block<@ELEM@, false>(in2, in3, in6, (int)key0, columns, live, true, row, cached_lower, qe_base, qo_base, ss, so, M, S, kv_head, lane, sg); }
        else { flash_block<@ELEM@, true>(in2, in3, in6, (int)key0, columns, live, true, row, cached_lower, qe_base, qo_base, ss, so, M, S, kv_head, lane, sg); }
    }
    int whole = rows & ~7, mma_end = min(row + 1, whole);
    long new_start = max(0L, row0 + cached_lower) & ~7L;
    for (long key0 = new_start; key0 < mma_end; key0 += block) {
        flash_block<@ACC@, false>(in4, in5, in7, (int)key0, min(block, mma_end - (int)key0), mma_end, false, row, cached_lower, qe_base, qo_base, ss, so, M, S, kv_head, lane, sg);
    }
    if (row >= whole) { flash_tail(in4, in5, in7, whole, rows - whole, row, cached_lower, qe_base, qo_base, ss, so, M, S, kv_head, lane, sg); }
    FOR_UNROLL for (int jj = 0; jj < nq; ++jj) {
        int v = jj * nsg + sg;
        @ACC@ inverse = S[jj] == 0.0f ? 0.0f : 1.0f / S[jj];
        FOR_UNROLL for (int part = 0; part < lane_vectors; ++part) {
            ((device @ACC@4*)(out + (row * (kv_heads * groups) + kv_head * groups + v) * head_dim))[part * 32 + lane] = so4[v * (head_dim / 4) + part * 32 + lane] * inverse;
        }
    }
}
"#;
