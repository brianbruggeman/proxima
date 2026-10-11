use std::env;
use std::fs;
use std::path::PathBuf;

const BLOCK_64: &str = "constexpr long block = 64;";
const BLOCK_32: &str = "constexpr long block = 32;";
const DEPTH_LOOP: &str = "            for (int depth = 0; depth < (int)half_dim; depth += 8 * (int)depth_unroll) {";
const KEY_EVEN_LOAD: &str = "simdgroup_load(key_even_tile[group][step_index], key_even_ptr + depth + 8 * step_index, (ulong)(kv_heads * half_dim), ulong2(0, 0), true);";
const KEY_ODD_LOAD: &str = "simdgroup_load(key_odd_tile[group][step_index], key_odd_ptr + depth + 8 * step_index, (ulong)(kv_heads * half_dim), ulong2(0, 0), true);";
const VALUE_LOAD: &str = "simdgroup_load(value_float, value_ptr + (long)key_tile * 8L * (kv_heads * head_dim) + dimension_block * 8, (ulong)(kv_heads * head_dim));";
const WEIGHTS_LINE: &str = "                    simdgroup_float8x8 weights[tile_blocks];";
const QUERY_STAGE_DECL: &str = "    threadgroup float query_stage[stage_query ? tile_blocks * 2L * 8L * query_stage_stride : 1L];";
const QUERY_EVEN_LOAD: &str = "simdgroup_load(query_even[step_index], stage_even, (ulong)query_stage_stride);";
const QUERY_ODD_LOAD: &str = "simdgroup_load(query_odd[step_index], stage_even + 8L * query_stage_stride, (ulong)query_stage_stride);";
const SOFTMAX_LOOP: &str = "        for (long vector = (long)simdgroup_slot; vector < tile_vectors; vector += simdgroups) {";
const BARRIER_A: &str = "        threadgroup_barrier(mem_flags::mem_threadgroup);\n        for (long vector = (long)simdgroup_slot; vector < tile_vectors; vector += simdgroups) {";
const BARRIER_B: &str = "        threadgroup_barrier(mem_flags::mem_threadgroup);\n        if (false) {\n            for (int vector_block = (int)simdgroup_slot; vector_block < (int)tile_blocks; vector_block += (int)simdgroups) {\n                int vector_slot = vector_block / (int)simdgroups;\n                float row_scale";
const MAX_REDUCE: &str = "block_maximum = simd_max(block_maximum);";
const SUM_REDUCE: &str = "block_sum = simd_sum(block_sum);";
const RESCALE_EXP: &str = "exp(previous_maximum - next_maximum)";
const WEIGHT_EXP: &str = "exp(local_scores[item] - next_maximum)";
const ROW_SCALE: &str = "float row_scale = rescale_tile[vector_block * 8 + fragment_row];";
const FILLED: &str = "make_filled_simdgroup_matrix<float, 8, 8>(1.0f)";
const SCORE_READ: &str = "score_tile[vector * block + column] * scale";
const MASK_NEW: &str = "if (column < columns && relative <= new_upper && relative >= cached_lower) {";
const MASK_CACHED: &str = "if (key < slice_end && (key - live - query_row) >= cached_lower) {";
const WEIGHT_STORE: &str = "score_tile[vector * block + (long)lane + item * 32L] = weight;";
const STATS_STORE: &str = "if (lane == 0) { row_maximum[vector] = next_maximum; row_sum[vector] = row_sum[vector] * rescale + block_sum; rescale_tile[vector] = rescale; }";

struct Source(String);

impl Source {
    fn replace_all(&mut self, anchor: &str, replacement: &str, expected: usize) {
        let found = self.0.matches(anchor).count();
        assert_eq!(found, expected, "anchor `{}` found {found} times, expected {expected}", anchor.lines().next().unwrap_or(""));
        self.0 = self.0.replace(anchor, replacement);
    }

    fn replace_last(&mut self, anchor: &str, replacement: &str, expected_total: usize) {
        let found = self.0.matches(anchor).count();
        assert_eq!(found, expected_total, "anchor `{anchor}` found {found} times, expected {expected_total}");
        let at = self.0.rfind(anchor).expect("anchor present");
        self.0.replace_range(at..at + anchor.len(), replacement);
    }

    fn insert_after(&mut self, anchor: &str, addition: &str) {
        self.replace_all(anchor, &format!("{anchor}\n{addition}"), 1);
    }

    fn insert_before(&mut self, anchor: &str, addition: &str) {
        self.replace_all(anchor, &format!("{addition}\n{anchor}"), 1);
    }
}

#[derive(Clone, Copy)]
enum KStage {
    None,
    DepthChunk { transposed: bool },
    FullBlock { transposed: bool },
}

#[derive(Clone, Copy)]
struct Arm {
    block: u32,
    k_stage: KStage,
    k_pad: u32,
    v_stage: bool,
    v_pad: u32,
}

impl Arm {
    const fn direct(block: u32) -> Arm {
        Arm { block, k_stage: KStage::None, k_pad: 0, v_stage: false, v_pad: 0 }
    }

    fn key_chunk_depth(&self) -> u32 {
        match self.k_stage {
            KStage::DepthChunk { .. } => 16,
            _ => 32,
        }
    }

    fn k_plane_floats(&self) -> u32 {
        match self.k_stage {
            KStage::None => 0,
            KStage::DepthChunk { transposed: false } | KStage::FullBlock { transposed: false } => self.block * (self.key_chunk_depth() + self.k_pad),
            KStage::DepthChunk { transposed: true } | KStage::FullBlock { transposed: true } => self.key_chunk_depth() * (self.block + self.k_pad),
        }
    }

    fn v_tiles(&self) -> u32 {
        if self.block == 64 { 4 } else { self.block / 8 }
    }

    fn v_floats(&self) -> u32 {
        if self.v_stage { self.v_tiles() * 8 * (64 + self.v_pad) } else { 0 }
    }

    fn union_floats(&self) -> u32 {
        (2 * self.k_plane_floats()).max(self.v_floats())
    }
}

fn k_stage_code(arm: &Arm) -> String {
    let block = arm.block;
    let depth = arm.key_chunk_depth();
    let parts = depth / 4;
    let pad = arm.k_pad;
    let (transposed, source_depth) = match arm.k_stage {
        KStage::DepthChunk { transposed } => (transposed, "depth"),
        KStage::FullBlock { transposed } => (transposed, "0"),
        KStage::None => unreachable!("no key stage"),
    };
    let store = if transposed {
        let stride = block + pad;
        format!(
            "                        for (int lane_part = 0; lane_part < 4; lane_part++) {{\n                            kstage_even[(copy_part * 4 + lane_part) * {stride} + copy_key] = even4[lane_part];\n                            kstage_odd[(copy_part * 4 + lane_part) * {stride} + copy_key] = odd4[lane_part];\n                        }}\n"
        )
    } else {
        let stride = depth + pad;
        format!(
            "                        *(threadgroup float4*)(kstage_even + copy_key * {stride} + copy_part * 4) = even4;\n                        *(threadgroup float4*)(kstage_odd + copy_key * {stride} + copy_part * 4) = odd4;\n"
        )
    };
    format!(
        "                threadgroup_barrier(mem_flags::mem_threadgroup);\n                for (int copy = (int)thread_id; copy < {block} * {parts}; copy += (int)threads) {{\n                    int copy_key = copy / {parts}; int copy_part = copy % {parts};\n                    if (copy_key < fragments * 8) {{\n                        long source_offset = (key0 + (long)copy_key) * (kv_heads * half_dim) + kv_head * half_dim + {source_depth} + copy_part * 4;\n                        float4 even4 = *(device const float4*)(key_even + source_offset);\n                        float4 odd4 = *(device const float4*)(key_odd + source_offset);\n{store}                    }}\n                }}\n                threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
}

fn k_load_code(arm: &Arm, plane: &str) -> String {
    let depth_term = match arm.k_stage {
        KStage::FullBlock { .. } => "depth + 8 * step_index",
        _ => "8 * step_index",
    };
    let name = if plane == "even" { "key_even_tile" } else { "key_odd_tile" };
    let stage = format!("kstage_{plane}");
    match arm.k_stage {
        KStage::DepthChunk { transposed: false } | KStage::FullBlock { transposed: false } => {
            let stride = arm.key_chunk_depth() + arm.k_pad;
            format!("simdgroup_load({name}[group][step_index], (threadgroup const float*)({stage}) + (key_tile * 8) * {stride} + {depth_term}, (ulong){stride}, ulong2(0, 0), true);")
        }
        KStage::DepthChunk { transposed: true } | KStage::FullBlock { transposed: true } => {
            let stride = arm.block + arm.k_pad;
            format!("simdgroup_load({name}[group][step_index], (threadgroup const float*)({stage}) + ({depth_term}) * {stride} + key_tile * 8, (ulong){stride}, ulong2(0, 0), false);")
        }
        KStage::None => unreachable!("no key stage"),
    }
}

fn v_stage_code(arm: &Arm) -> String {
    let tiles = arm.v_tiles();
    let keys = tiles * 8;
    let stride = 64 + arm.v_pad;
    format!(
        "                    if (key_tile % {tiles} == 0) {{\n                        threadgroup_barrier(mem_flags::mem_threadgroup);\n                        for (int copy = (int)thread_id; copy < {keys} * 16; copy += (int)threads) {{\n                            int copy_key = copy / 16; int copy_part = copy % 16;\n                            if (key_tile * 8 + copy_key < fragments * 8) {{\n                                *(threadgroup float4*)(vstage + copy_key * {stride} + copy_part * 4) = *(device const float4*)(value_ptr + (long)(key_tile * 8 + copy_key) * (kv_heads * head_dim) + copy_part * 4);\n                            }}\n                        }}\n                        threadgroup_barrier(mem_flags::mem_threadgroup);\n                    }}"
    )
}

fn v_load_code(arm: &Arm) -> String {
    let tiles = arm.v_tiles();
    let stride = 64 + arm.v_pad;
    format!("simdgroup_load(value_float, (threadgroup const float*)(vstage) + ((key_tile % {tiles}) * 8) * {stride} + dimension_block * 8, (ulong){stride});")
}

fn staged(base: &str, arm: &Arm) -> String {
    let mut source = Source(base.to_string());
    if arm.block == 32 {
        source.replace_all(BLOCK_64, BLOCK_32, 1);
    }
    let union = arm.union_floats();
    if union > 0 {
        let plane = arm.k_plane_floats();
        let mut declaration = format!("    threadgroup float stage_union[{union}];\n");
        if plane > 0 {
            declaration.push_str(&format!("    threadgroup float* kstage_even = stage_union; threadgroup float* kstage_odd = stage_union + {plane};\n"));
        }
        if arm.v_stage {
            declaration.push_str("    threadgroup float* vstage = stage_union;");
        }
        source.insert_after(QUERY_STAGE_DECL, declaration.trim_end());
    }
    if !matches!(arm.k_stage, KStage::None) {
        match arm.k_stage {
            KStage::DepthChunk { .. } => source.insert_after(DEPTH_LOOP, &k_stage_code(arm)),
            KStage::FullBlock { .. } => source.insert_before(DEPTH_LOOP, &k_stage_code(arm)),
            KStage::None => {}
        }
        source.replace_all(KEY_EVEN_LOAD, &k_load_code(arm, "even"), 1);
        source.replace_all(KEY_ODD_LOAD, &k_load_code(arm, "odd"), 1);
    }
    if arm.v_stage {
        source.insert_after(WEIGHTS_LINE, &v_stage_code(arm));
        source.replace_last(VALUE_LOAD, &v_load_code(arm), 3);
    }
    source.0
}


fn exp_off(source: &mut Source) {
    source.replace_all(RESCALE_EXP, "(previous_maximum - next_maximum)", 1);
    source.replace_all(WEIGHT_EXP, "(local_scores[item] - next_maximum)", 1);
}

fn reduce_off(source: &mut Source) {
    source.replace_all(MAX_REDUCE, "block_maximum = block_maximum;", 1);
    source.replace_all(SUM_REDUCE, "block_sum = block_sum;", 1);
}

fn barrier_a_off(source: &mut Source) {
    source.replace_all(BARRIER_A, &format!("{}", &BARRIER_A["        threadgroup_barrier(mem_flags::mem_threadgroup);\n".len()..]), 1);
}

fn barrier_b_off(source: &mut Source) {
    source.replace_all(BARRIER_B, &format!("{}", &BARRIER_B["        threadgroup_barrier(mem_flags::mem_threadgroup);\n".len()..]), 1);
}

fn pad_probe(base: &str, bytes: u32) -> String {
    let floats = bytes / 4;
    let mut source = Source(base.to_string());
    source.insert_after(QUERY_STAGE_DECL, &format!("    threadgroup float pad_probe[{floats}];\n    pad_probe[thread_id % {floats}] = 0.0f; if (live < 0L) {{ out[0] = pad_probe[(thread_id * 7L + 1L) % {floats}]; }}"));
    source.0
}

fn all_loads_off(base: &str) -> String {
    let mut source = Source(base.to_string());
    source.replace_all(KEY_EVEN_LOAD, &format!("key_even_tile[group][step_index] = {FILLED};"), 1);
    source.replace_all(KEY_ODD_LOAD, &format!("key_odd_tile[group][step_index] = {FILLED};"), 1);
    source.replace_last(VALUE_LOAD, &format!("value_float = {FILLED};"), 3);
    source.replace_all(QUERY_EVEN_LOAD, &format!("query_even[step_index] = {FILLED};"), 2);
    source.replace_all(QUERY_ODD_LOAD, &format!("query_odd[step_index] = {FILLED};"), 2);
    source.0
}

fn arms(base: &str) -> Vec<(String, String)> {
    let mut list: Vec<(String, String)> = Vec::new();
    list.push(("ctrl".into(), base.to_string()));
    list.push(("all_loads_off".into(), all_loads_off(base)));
    let d16 = Arm { k_stage: KStage::DepthChunk { transposed: false }, ..Arm::direct(64) };
    list.push(("k_d16".into(), staged(base, &d16)));
    list.push(("k_d16_pad4".into(), staged(base, &Arm { k_pad: 4, ..d16 })));
    let d16t = Arm { k_stage: KStage::DepthChunk { transposed: true }, k_pad: 4, ..Arm::direct(64) };
    list.push(("kt_d16_pad4".into(), staged(base, &d16t)));
    let v64 = Arm { v_stage: true, ..Arm::direct(64) };
    list.push(("v_t4".into(), staged(base, &v64)));
    list.push(("v_t4_pad4".into(), staged(base, &Arm { v_pad: 4, ..v64 })));
    list.push(("kv_d16_t4".into(), staged(base, &Arm { v_stage: true, ..d16 })));
    list.push(("kvt_d16_t4".into(), staged(base, &Arm { v_stage: true, ..d16t })));
    list.push(("b32".into(), staged(base, &Arm::direct(32))));
    let k32 = Arm { k_stage: KStage::FullBlock { transposed: false }, ..Arm::direct(32) };
    list.push(("b32_k".into(), staged(base, &k32)));
    list.push(("b32_kt".into(), staged(base, &Arm { k_stage: KStage::FullBlock { transposed: true }, k_pad: 4, ..Arm::direct(32) })));
    list.push(("b32_v".into(), staged(base, &Arm { v_stage: true, ..Arm::direct(32) })));
    list.push(("b32_kv".into(), staged(base, &Arm { v_stage: true, ..k32 })));
    list.push(("b32_all_loads_off".into(), {
        let mut source = Source(all_loads_off(base));
        source.replace_all(BLOCK_64, BLOCK_32, 1);
        source.0
    }));
    for bytes in [8192u32, 10240, 16384] {
        list.push((format!("tgpad_{}k", bytes / 1024), pad_probe(base, bytes)));
    }
    list.push(("softmax_loop_off".into(), {
        let mut source = Source(base.to_string());
        source.replace_all(SOFTMAX_LOOP, "        for (long vector = (long)simdgroup_slot; vector < 0L; vector += simdgroups) {", 1);
        source.0
    }));
    list.push(("exp_off".into(), {
        let mut source = Source(base.to_string());
        exp_off(&mut source);
        source.0
    }));
    list.push(("reduce_off".into(), {
        let mut source = Source(base.to_string());
        reduce_off(&mut source);
        source.0
    }));
    list.push(("exp_reduce_off".into(), {
        let mut source = Source(base.to_string());
        exp_off(&mut source);
        reduce_off(&mut source);
        source.0
    }));
    list.push(("rescale_off".into(), {
        let mut source = Source(base.to_string());
        source.replace_last(ROW_SCALE, "float row_scale = 1.0f;", 2);
        source.0
    }));
    list.push(("barrier_a_off".into(), {
        let mut source = Source(base.to_string());
        barrier_a_off(&mut source);
        source.0
    }));
    list.push(("barrier_b_off".into(), {
        let mut source = Source(base.to_string());
        barrier_b_off(&mut source);
        source.0
    }));
    list.push(("barrier_off".into(), {
        let mut source = Source(base.to_string());
        barrier_a_off(&mut source);
        barrier_b_off(&mut source);
        source.0
    }));
    for (tag, edits) in [
        ("scores_read_off", vec![(SCORE_READ, "(scale * (float)column)", 2usize)]),
        ("mask_off", vec![(MASK_NEW, "if (true) {", 1), (MASK_CACHED, "if (true) {", 1)]),
        ("weights_store_off", vec![(WEIGHT_STORE, "(void)weight;", 1)]),
        ("stats_store_off", vec![(STATS_STORE, "if (lane == 99) { }", 1)]),
        (
            "softmax_body_min",
            vec![
                (SCORE_READ, "(scale * (float)column)", 2),
                (MASK_NEW, "if (true) {", 1),
                (MASK_CACHED, "if (true) {", 1),
                (WEIGHT_STORE, "(void)weight;", 1),
                (STATS_STORE, "if (lane == 99) { }", 1),
                (RESCALE_EXP, "(previous_maximum - next_maximum)", 1),
                (WEIGHT_EXP, "(local_scores[item] - next_maximum)", 1),
                (MAX_REDUCE, "block_maximum = block_maximum;", 1),
                (SUM_REDUCE, "block_sum = block_sum;", 1),
            ],
        ),
    ] {
        let mut source = Source(base.to_string());
        for (anchor, replacement, count) in edits {
            source.replace_all(anchor, replacement, count);
        }
        list.push((tag.to_string(), source.0));
    }
    for total in [10880u32, 11008, 16320, 16512] {
        list.push((format!("tgtotal_{total}"), pad_probe(base, total - 9600)));
    }
    list.push(("qstage_off".into(), {
        let mut source = Source(base.to_string());
        source.replace_all("constexpr bool stage_query = true;", "constexpr bool stage_query = false;", 1);
        source.0
    }));
    list
}

fn main() {
    let arguments: Vec<String> = env::args().collect();
    assert!(arguments.len() == 4, "usage: make_prefill_variants <base.metal> <out dir> <sha16>");
    let base = fs::read_to_string(&arguments[1]).expect("read base emitted source");
    let out_dir = PathBuf::from(&arguments[2]);
    fs::create_dir_all(&out_dir).expect("create out dir");
    let sha = &arguments[3];
    let list = arms(&base);
    assert!(!list.is_empty(), "N==0 arms");
    for (tag, text) in &list {
        let path = out_dir.join(format!("{sha}.{tag}.metal"));
        fs::write(&path, text).expect("write variant");
        let delta = text.len() as i64 - base.len() as i64;
        println!("{tag}: {} bytes ({delta:+} against base)", text.len());
    }
    println!("wrote {} variants to {}", list.len(), out_dir.display());
}
