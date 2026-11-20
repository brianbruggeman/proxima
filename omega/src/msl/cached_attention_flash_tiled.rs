use super::flash_refusal::FlashRefusal;
use super::layout_render::{hardware_type, readable_scalar_cache};
use super::*;
use proxima_numeric::float_layout::FloatLayout;

/// Simdgroups per flash threadgroup: llama.cpp's `nsg` for a head of fewer than 512 elements.
pub(super) const FLASH_SIMDGROUPS: u64 = 4;

/// Keys one flash block walks: llama.cpp's `OP_FLASH_ATTN_EXT_NCPSG`. Equal to the row-tiled
/// block of the same head so the entry name and scratch layout keep meaning one thing.
pub(super) const FLASH_KEYS_PER_BLOCK: u64 = 64;

/// The head width the flash body's row write assumes: one float4 per lane of one simdgroup.
const FLASH_HEAD_DIM: u64 = 128;

/// The query heads per kv head the flash body's fragment map assumes.
const FLASH_QUERY_GROUPS: u64 = 8;

/// Admits `kind` for the flash body, or names the one condition it fails: a sinked,
/// full-attention, causal op of eight query heads per kv head whose cached extent and new-row
/// extent are whole key blocks. Conditions are checked in the order of [`FlashRefusal`]; the
/// first failure is the reason. Head dims 256 and 512 are `HeadDim` refusals until the template
/// generalises past 128.
///
/// The whole-block rule is what lets the body drop every edge-tile load guard. The K and V
/// buffers are bound from these same extents (`proxima-tensor/src/physical.rs:47`,
/// `cached_key_width = cached_key_rows * head_dim`), so a cached extent that is a whole number
/// of blocks is a buffer a block-aligned read never leaves; the keys between the live length and
/// the extent are masked to -inf by the softmax's `visible` predicate, not by the loads.
pub(super) fn flash_admission(kind: &BoundOpKind) -> Result<(), FlashRefusal> {
    let BoundOpKind::CachedAttention {
        query_rows,
        cached_key_rows,
        new_key_rows,
        query_groups,
        head_dim,
        cached_lower_inclusive,
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
        (*head_dim == FLASH_HEAD_DIM, FlashRefusal::HeadDim),
        (*cached_lower_inclusive == i64::MIN, FlashRefusal::SlidingWindow),
        (*new_upper_inclusive == 0, FlashRefusal::NewKeysNotCausal),
        (query_rows == new_key_rows, FlashRefusal::QueryRowsDifferFromNewKeys),
        (
            query_rows.is_multiple_of(FLASH_KEYS_PER_BLOCK),
            FlashRefusal::QueryRowsNotWholeBlocks,
        ),
        (
            cached_key_rows.is_multiple_of(FLASH_KEYS_PER_BLOCK),
            FlashRefusal::CachedRowsNotWholeBlocks,
        ),
        (
            row_tiled_block(*head_dim) == FLASH_KEYS_PER_BLOCK,
            FlashRefusal::RowTiledBlockDiffers,
        ),
    ];
    checks
        .into_iter()
        .find_map(|(holds, refusal)| (!holds).then_some(refusal))
        .map_or(Ok(()), Err)
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
) -> Result<String, EmitError> {
    let BoundOpKind::CachedAttention {
        kv_heads,
        query_groups,
        head_dim,
        scale,
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
    let element = cached_kv_codec
        .as_ref()
        .and_then(readable_scalar_cache)
        .and_then(|(layout, _)| hardware_type(&layout))
        .filter(|element| *element != accumulator)
        .ok_or(EmitError::CachedAttentionKvCodecNotSupported {
            node: resolved.node,
            reason: "flash reads a narrow scalar cache through the matrix hardware type of its layout",
        })?;
    let half_dim = head_dim / 2;
    let simdgroups = FLASH_SIMDGROUPS;
    let block = FLASH_KEYS_PER_BLOCK;
    let substitutions = [
        ("@ENTRY@", entry.to_string()),
        ("@KV_HEADS@", kv_heads.to_string()),
        ("@GROUPS@", query_groups.to_string()),
        ("@HEAD_DIM@", head_dim.to_string()),
        ("@HALF_DIM@", half_dim.to_string()),
        ("@NSG@", simdgroups.to_string()),
        ("@BLOCK@", block.to_string()),
        ("@NQ@", (query_groups / simdgroups).to_string()),
        ("@NO@", (head_dim / 8 / simdgroups).to_string()),
        ("@QK_PER_SG@", (block / 8 / simdgroups).to_string()),
        ("@ITEMS@", (block / 32).to_string()),
        ("@THREADS@", (simdgroups * 32).to_string()),
        ("@SCALE@", msl_literal(*scale)),
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

const FLASH_TILED_KERNEL: &str = r#"#define FOR_UNROLL _Pragma("clang loop unroll(full)")
struct Uniforms { long total_elements; long splits; };
constant constexpr int kv_heads = @KV_HEADS@, groups = @GROUPS@, head_dim = @HEAD_DIM@, half_dim = @HALF_DIM@;
constant constexpr int nsg = @NSG@, block = @BLOCK@, nq = @NQ@, no = @NO@, stage = half_dim + 8, tiles = block / 8;
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
inline void load_v(thread @ACC_FRAG@& dst, device const @ACC@* src, int stride) { simdgroup_load(dst, src, (ulong)stride); }

template <typename T>
inline void flash_block(device const T* ke, device const T* ko, device const T* vv, int key0, int columns, int valid_end, bool cached_phase, int row,
                        threadgroup const @ACC@* sq, threadgroup @ACC@* ss, threadgroup @ACC@* so, thread @ACC@ (&M)[nq], thread @ACC@ (&S)[nq],
                        int kv_head, ushort lane, ushort sg) {
    const int kstride = kv_heads * half_dim, vstride = kv_heads * head_dim;
    FOR_UNROLL for (int cc = 0; cc < @QK_PER_SG@; ++cc) {                       // QK: key tiles sg and sg + 4
        int kt = cc * nsg + sg;
        {
            @ACC_FRAG@ acc = make_filled_simdgroup_matrix<@ACC@, 8, 8>(0.0f);
            device const T* kep = ke + (key0 + kt * 8) * kstride + kv_head * half_dim;
            device const T* kop = ko + (key0 + kt * 8) * kstride + kv_head * half_dim;
            FOR_UNROLL for (int j = 0; j < half_dim / 8; ++j) {       // chain e0,o0,e1,o1,... as the row-tiled kernel
                simdgroup_barrier(mem_flags::mem_none);
                @ACC_FRAG@ qe, qo; typename k_frag<T>::type kef, kof;
                simdgroup_load(qe, sq + 8 * j, (ulong)stage);
                simdgroup_load(qo, sq + 8 * stage + 8 * j, (ulong)stage);
                load_k(kef, kep + 8 * j, kstride);
                load_k(kof, kop + 8 * j, kstride);
                simdgroup_barrier(mem_flags::mem_none);
                simdgroup_multiply_accumulate(acc, qe, kef, acc);
                simdgroup_multiply_accumulate(acc, qo, kof, acc);
            }
            simdgroup_store(acc, ss + kt * 8, (ulong)block);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    FOR_UNROLL for (int jj = 0; jj < nq; ++jj) {                      // softmax: this simdgroup owns vectors jj*nsg + sg
        int v = jj * nsg + sg;
        @ACC@ local[@ITEMS@]; @ACC@ bm = -INFINITY;
        FOR_UNROLL for (int item = 0; item < @ITEMS@; ++item) {
            int column = (int)lane + item * 32; @ACC@ raw = -INFINITY;
            if (item * 32 < columns) {
                int key = key0 + column;
                bool visible = cached_phase ? (key < valid_end) : (column < columns && key <= row);
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
        o4[lane] *= rescale;                                          // O rescale by the row owner, 32 lanes x 4 = 128 dims
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    @ACC_FRAG@ lo[no];                                       // PV: this simdgroup owns dimension blocks sg + nsg * i
    FOR_UNROLL for (int i = 0; i < no; ++i) { simdgroup_load(lo[i], so + 8 * (sg + nsg * i), (ulong)head_dim); }
    FOR_UNROLL for (int kt = 0; kt < tiles; ++kt) {
        @ACC_FRAG@ p; simdgroup_load(p, ss + kt * 8, (ulong)block);
        FOR_UNROLL for (int i = 0; i < no; ++i) {
            @ACC_FRAG@ vf;
            load_v(vf, vv + (key0 + kt * 8) * vstride + kv_head * head_dim + 8 * (sg + nsg * i), vstride);
            simdgroup_multiply_accumulate(lo[i], p, vf, lo[i]);
        }
    }
    FOR_UNROLL for (int i = 0; i < no; ++i) { simdgroup_store(lo[i], so + 8 * (sg + nsg * i), (ulong)head_dim); }
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
    threadgroup @ACC@ sq[16 * stage]; threadgroup @ACC@4 so4[8 * head_dim / 4]; threadgroup @ACC@ ss[8 * block];
    threadgroup @ACC@* so = (threadgroup @ACC@*)so4;
    int qbase = (row * (kv_heads * groups) + kv_head * groups) * half_dim;
    for (int i = (int)sg * 32 + (int)lane; i < 8 * half_dim; i += @THREADS@) {
        int v = i / half_dim, c = i % half_dim;
        sq[v * stage + c] = @ACC@(in0[qbase + i]); sq[(8 + v) * stage + c] = @ACC@(in1[qbase + i]);
    }
    for (int i = (int)sg * 32 + (int)lane; i < 8 * head_dim; i += @THREADS@) { so[i] = 0.0f; }
    @ACC@ M[nq], S[nq];
    FOR_UNROLL for (int jj = 0; jj < nq; ++jj) {
        @ACC@ sink = in9[kv_head * groups + jj * nsg + sg];
        M[jj] = sink; S[jj] = (sink == -INFINITY) ? 0.0f : 1.0f;      // sink = softmax initial state, as SINK_INIT
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int key0 = 0; key0 < live; key0 += block) {
        flash_block<@ELEM@>(in2, in3, in6, key0, min(block, live - key0), live, true, row, sq, ss, so, M, S, kv_head, lane, sg);
    }
    int mma_end = min(row + 1, rows & ~7);                            // v0: rows % 8 == 0, so no scalar tail
    for (int key0 = 0; key0 < mma_end; key0 += block) {
        flash_block<@ACC@>(in4, in5, in7, key0, min(block, mma_end - key0), mma_end, false, row, sq, ss, so, M, S, kv_head, lane, sg);
    }
    FOR_UNROLL for (int jj = 0; jj < nq; ++jj) {
        int v = jj * nsg + sg;
        @ACC@ inverse = S[jj] == 0.0f ? 0.0f : 1.0f / S[jj];
        ((device @ACC@4*)(out + (row * (kv_heads * groups) + kv_head * groups + v) * head_dim))[lane] = so4[v * (head_dim / 4) + lane] * inverse;
    }
}
"#;
