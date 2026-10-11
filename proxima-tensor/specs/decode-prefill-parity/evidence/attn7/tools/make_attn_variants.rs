use std::fs;
use std::path::{Path, PathBuf};

const KEY_LOAD: &str = "if (live && index < plane_vectors) { key_real[step][slot] = float4(kr4[index]); key_imag[step][slot] = float4(ki4[index]); }";
const VALUE_LOAD: &str = "if (live && index < row_vectors) { value_row[step][slot] = float4(v4[index]); }";
const KEY_SYNTH: &str = "if (live && index < plane_vectors) { key_real[step][slot] = float4((float)((key + (long)index) & 15L) * 0.0625f); key_imag[step][slot] = float4((float)((key + (long)index + 3L) & 15L) * 0.0625f); }";
const VALUE_SYNTH: &str = "if (live && index < row_vectors) { value_row[step][slot] = float4((float)((key + (long)index) & 15L) * 0.0625f); }";
const QUERY_LOAD: &str = "if (index < plane_vectors) { query_real[slot] = float4(qr4[index]); query_imag[slot] = float4(qi4[index]); }";
const QUERY_SYNTH: &str = "if (index < plane_vectors) { query_real[slot] = float4((float)(index & 15) * 0.0625f); query_imag[slot] = float4((float)((index + 3) & 15) * 0.0625f); }";
const DOT_REAL: &str = "partial_score += dot(key_real[step][slot], query_real[slot]);";
const DOT_IMAG: &str = "partial_score += dot(key_imag[step][slot], query_imag[slot]);";
const DOT_SYNTH: &str = "partial_score += (float)((batch_start + (long)lane + (long)slot + (long)step) & 7L) * 0.125f;";
const SHUFFLES: [&str; 5] = [
    "if (16 < lanes_per_key) { partial_score += simd_shuffle_xor(partial_score, (ushort)16); }",
    "if (8 < lanes_per_key) { partial_score += simd_shuffle_xor(partial_score, (ushort)8); }",
    "if (4 < lanes_per_key) { partial_score += simd_shuffle_xor(partial_score, (ushort)4); }",
    "if (2 < lanes_per_key) { partial_score += simd_shuffle_xor(partial_score, (ushort)2); }",
    "if (1 < lanes_per_key) { partial_score += simd_shuffle_xor(partial_score, (ushort)1); }",
];
const NEXT_MAX: &str = "float next_max = max(maximum, batch_max);";
const RESCALE: &str = "float rescale = (maximum == -INFINITY) ? 0.0f : exp(maximum - next_max);";
const WEIGHT: &str = "float weight = (score[step] == -INFINITY) ? 0.0f : exp(score[step] - next_max);";
const PV: &str = "weighted[slot] += value_row[step][slot] * weight;";
const STATS_WRITE: &str = "if (splits != 1L && lane == 0) { attn_scratch[stats_index] = merged_max; attn_scratch[stats_index + 1L] = merged_sum; }";
const PARTIAL_WRITE: &str = "((device float4*)attn_scratch)[(value_base + index) * splits + split] = merged;";
const CHUNK_MERGE: &str = "OMEGA_UNROLL for (long c = 0; c < cap; c++) { merged += (c < chunks) ? shared_o[c * row_vectors + index] * chunk_scale[c] : float4(0.0f); }";
const BARRIER: &str = "threadgroup_barrier(mem_flags::mem_threadgroup);";
const SPLITS_UNIFORM: &str = "long splits = u.splits;";
const SPLIT_DECODE: &str = "long split = (long)tgid % splits;\n    long query_head = ((long)tgid / splits) % (kv_heads * query_groups);\n    long query_row = (long)tgid / (splits * kv_heads * query_groups);";
const SPLIT_DECODE_32: &str = "uint splits32 = (uint)splits; long split = (long)(tgid % splits32);\n    long query_head = (long)((tgid / splits32) % (uint)(kv_heads * query_groups));\n    long query_row = (long)(tgid / (splits32 * (uint)(kv_heads * query_groups)));";
const SLICE_LEN: &str = "long slice_len = (band + splits - 1L) / splits;";
const SLICE_LEN_32: &str = "long slice_len = (long)(((uint)band + (uint)splits - 1u) / (uint)splits);";
const LOCAL_KEYS: &str = "long num_local_keys = (slice_start < hi) ? ((hi - 1L - slice_start) / chunks) + 1L : 0L;";
const LOCAL_KEYS_32: &str = "long num_local_keys = (slice_start < hi) ? (long)((uint)(hi - 1L - slice_start) / (uint)chunks) + 1L : 0L;";
const CHUNK_EXP: &str = "chunk_scale[c] = live_chunk ? exp(shared_m[c] - merged_max) : 0.0f;";
const CHUNKS_UNIFORM: &str = "long chunks = u.context_chunks;";
const CAP_BASE: &str = "constexpr long cap = ";
const LANES_BASE: &str = "constexpr short lanes_per_key = 32;";
const BATCH_BASE: &str = "constexpr short batch = 1;";
const PTR_KR4: &str = "device const float4* kr4 = (device const float4*)((cached ? in2 : in4) + kbase);";
const PTR_KI4: &str = "device const float4* ki4 = (device const float4*)((cached ? in3 : in5) + kbase);";
const PTR_V4: &str = "device const float4* v4 = (device const float4*)((cached ? in6 : in7) + kbase * 2);";
const SIG_IN2: &str = "device const float* in2 [[buffer(2)]]";
const SIG_IN3: &str = "device const float* in3 [[buffer(3)]]";
const SIG_IN6: &str = "device const float* in6 [[buffer(6)]]";

const BF16_HELPER: &str = "static inline float4 omega_bf16x4_to_float4(ushort4 value) { return float4(as_type<float>((uint)value.x << 16u), as_type<float>((uint)value.y << 16u), as_type<float>((uint)value.z << 16u), as_type<float>((uint)value.w << 16u)); }\n";
const BF8_HELPER: &str = "static inline float omega_bf8_to_float(uchar value) { uint sign = ((uint)value & 0x80u) << 24u; uint exponent = ((uint)value >> 2u) & 0x1fu; uint mantissa = (uint)value & 0x3u; if (exponent == 0u) { float magnitude = ldexp((float)mantissa, -16); return sign == 0u ? magnitude : -magnitude; } if (exponent == 31u) { uint bits = 0x7f800000u | (mantissa << 21u); return as_type<float>(sign | bits); } uint bits = ((exponent + 112u) << 23u) | (mantissa << 21u); return as_type<float>(sign | bits); }\nstatic inline float4 omega_bf8x4_to_float4(uchar4 value) { return float4(omega_bf8_to_float(value.x), omega_bf8_to_float(value.y), omega_bf8_to_float(value.z), omega_bf8_to_float(value.w)); }\n";

fn narrow_pointers(narrow: &str) -> String {
    format!(
        "device const {narrow}4* kr4_cached = (device const {narrow}4*)(in2 + kbase);\n                device const {narrow}4* ki4_cached = (device const {narrow}4*)(in3 + kbase);\n                device const {narrow}4* v4_cached = (device const {narrow}4*)(in6 + kbase * 2);\n                device const float4* kr4_new = (device const float4*)(in4 + kbase);\n                device const float4* ki4_new = (device const float4*)(in5 + kbase);\n                device const float4* v4_new = (device const float4*)(in7 + kbase * 2);"
    )
}

fn narrow_key_load(convert: &str) -> String {
    format!(
        "if (live && index < plane_vectors) {{ if (cached) {{ key_real[step][slot] = {convert}(kr4_cached[index]); key_imag[step][slot] = {convert}(ki4_cached[index]); }} else {{ key_real[step][slot] = float4(kr4_new[index]); key_imag[step][slot] = float4(ki4_new[index]); }} }}"
    )
}

fn narrow_value_load(convert: &str) -> String {
    format!(
        "if (live && index < row_vectors) {{ if (cached) {{ value_row[step][slot] = {convert}(v4_cached[index]); }} else {{ value_row[step][slot] = float4(v4_new[index]); }} }}"
    )
}

fn replace_once(source: &str, from: &str, to: &str) -> String {
    let found = source.matches(from).count();
    assert_eq!(found, 1, "pattern must occur exactly once, found {found}: {from}");
    source.replacen(from, to, 1)
}

fn replace_each(source: &str, patterns: &[&str], to: &str) -> String {
    patterns.iter().fold(source.to_string(), |text, from| replace_once(&text, from, to))
}

fn insert_before_kernel(source: &str, helper: &str) -> String {
    let at = source.find("kernel void ").expect("a kernel entry");
    format!("{}{}{}", &source[..at], helper, &source[at..])
}

struct Output {
    directory: PathBuf,
    sha: String,
    prefix: String,
}

impl Output {
    fn metal(&self, tag: &str, source: &str) {
        fs::write(self.directory.join(format!("{}.{}{}.metal", self.sha, self.prefix, tag)), source).expect("write variant");
    }

    fn side(&self, tag: &str, extension: &str, text: &str) {
        fs::write(self.directory.join(format!("{}.{}{}.{}", self.sha, self.prefix, tag, extension)), text).expect("write side file");
    }
}

fn sha_of(path: &Path) -> String {
    let stem = path.file_stem().and_then(|stem| stem.to_str()).expect("file stem");
    stem.strip_prefix("pipeline_").expect("pipeline_<sha16>.metal").to_string()
}

fn write_ablations(output: &Output, base: &str, with_division_arms: bool) {
    if with_division_arms {
        output.metal("div32", &div32(base));
        output.metal("div32_split", &replace_once(base, SPLIT_DECODE, SPLIT_DECODE_32));
        output.metal("div32_slice", &replace_once(base, SLICE_LEN, SLICE_LEN_32));
        output.metal("div32_keys", &replace_once(base, LOCAL_KEYS, LOCAL_KEYS_32));
    }
    output.metal("ctrl", base);
    output.metal("empty", &replace_once(base, SPLITS_UNIFORM, "long splits = u.splits; if (splits > 0L) { return; }"));
    output.metal("k_loads_off", &replace_once(base, KEY_LOAD, KEY_SYNTH));
    output.metal("v_loads_off", &replace_once(base, VALUE_LOAD, VALUE_SYNTH));
    output.metal("q_loads_off", &replace_once(base, QUERY_LOAD, QUERY_SYNTH));
    output.metal("kv_loads_off", &replace_once(&replace_once(base, KEY_LOAD, KEY_SYNTH), VALUE_LOAD, VALUE_SYNTH));
    output.metal("all_loads_off", &all_loads_off(base));
    output.metal("qk_off", &qk_off(base));
    output.metal("qk_shuffle_off", &replace_each(base, &SHUFFLES, ""));
    output.metal("exp_off", &exp_off(base));
    output.metal("softmax_off", &softmax_off(base));
    output.metal("rescale_off", &replace_once(base, RESCALE, "float rescale = 1.0f;"));
    output.metal("pv_off", &replace_once(base, PV, "weighted[slot] = value_row[step][slot];"));
    output.metal("write_off", &write_off(base));
    output.metal("tgmerge_off", &replace_once(base, CHUNK_MERGE, "merged = shared_o[index] * chunk_scale[0];"));
    output.metal("barrier_off", &replace_once(base, BARRIER, ""));
    let local_keys = if with_division_arms { LOCAL_KEYS } else { LOCAL_KEYS_32 };
    output.metal("loop_off", &replace_once(base, local_keys, "long num_local_keys = 0L;"));
    output.metal("merge_exp_off", &replace_once(base, CHUNK_EXP, "chunk_scale[c] = live_chunk ? fma(shared_m[c] - merged_max, 0.0009765625f, 1.0f) : 0.0f;"));
    output.metal("skeleton", &skeleton(base));
}

fn div32(base: &str) -> String {
    let split = replace_once(base, SPLIT_DECODE, SPLIT_DECODE_32);
    let slice = replace_once(&split, SLICE_LEN, SLICE_LEN_32);
    replace_once(&slice, LOCAL_KEYS, LOCAL_KEYS_32)
}

fn all_loads_off(base: &str) -> String {
    let keys = replace_once(base, KEY_LOAD, KEY_SYNTH);
    let values = replace_once(&keys, VALUE_LOAD, VALUE_SYNTH);
    replace_once(&values, QUERY_LOAD, QUERY_SYNTH)
}

fn qk_off(base: &str) -> String {
    let dots = replace_once(base, DOT_REAL, DOT_SYNTH);
    replace_once(&dots, DOT_IMAG, "")
}

fn exp_off(base: &str) -> String {
    let rescale = replace_once(base, RESCALE, "float rescale = (maximum == -INFINITY) ? 0.0f : fma(maximum - next_max, 0.0009765625f, 1.0f);");
    replace_once(&rescale, WEIGHT, "float weight = (score[step] == -INFINITY) ? 0.0f : fma(score[step] - next_max, 0.0009765625f, 1.0f);")
}

fn softmax_off(base: &str) -> String {
    let next_max = replace_once(base, NEXT_MAX, "float next_max = maximum;");
    let rescale = replace_once(&next_max, RESCALE, "float rescale = 1.0f;");
    replace_once(&rescale, WEIGHT, "float weight = (score[step] == -INFINITY) ? 0.0f : fma(score[step], 0.0009765625f, 1.0f);")
}

fn write_off(base: &str) -> String {
    let stats = replace_once(base, STATS_WRITE, "if (merged_sum == -777.0f && lane == 0) { attn_scratch[stats_index] = merged_max; }");
    replace_once(&stats, PARTIAL_WRITE, "if (merged.x == -777.0f) { ((device float4*)attn_scratch)[(value_base + index) * splits + split] = merged; }")
}

fn skeleton(base: &str) -> String {
    let loads = all_loads_off(base);
    let dots = qk_off(&loads);
    let shuffles = replace_each(&dots, &SHUFFLES, "");
    let softmax = softmax_off(&shuffles);
    let pv = replace_once(&softmax, PV, "weighted[slot] = value_row[step][slot];");
    let merged = replace_once(&pv, CHUNK_MERGE, "merged = shared_o[index] * chunk_scale[0];");
    write_off(&merged)
}

struct Setting {
    keys_per_split: u64,
    keys_per_simdgroup: u64,
    simdgroups_max: u64,
    keys_in_flight: u64,
    keys_per_batch: u64,
    splits_max: u64,
}

struct Shape {
    capacity: u64,
    head_dim: u64,
    base_splits: u64,
    base_chunks: u64,
}

impl Setting {
    fn production() -> Self {
        Self { keys_per_split: 32, keys_per_simdgroup: 8, simdgroups_max: 4, keys_in_flight: 1, keys_per_batch: 1, splits_max: 32 }
    }

    fn tag(&self) -> String {
        format!(
            "kps{}_ks{}_sm{}_kif{}_kpb{}_mx{}",
            self.keys_per_split, self.keys_per_simdgroup, self.simdgroups_max, self.keys_in_flight, self.keys_per_batch, self.splits_max
        )
    }

    fn splits(&self, shape: &Shape) -> u64 {
        shape.capacity.div_ceil(self.keys_per_split).clamp(1, self.splits_max)
    }

    fn simdgroup_cap(&self, shape: &Shape) -> u64 {
        let budget = 32768 / (4 * (shape.head_dim + 2));
        self.simdgroups_max.min(budget.max(1))
    }

    fn chunks(&self, shape: &Shape) -> u64 {
        let splits = self.splits(shape);
        shape.capacity.div_ceil(splits).div_ceil(self.keys_per_simdgroup).clamp(1, self.simdgroup_cap(shape))
    }
}

fn base_cap_text(source: &str) -> String {
    let at = source.find(CAP_BASE).expect("cap constant") + CAP_BASE.len();
    let end = source[at..].find(';').expect("cap terminator") + at;
    source[at..end].to_string()
}

fn apply_setting(base: &str, setting: &Setting, shape: &Shape) -> String {
    let splits = setting.splits(shape);
    let chunks = setting.chunks(shape);
    let lanes = 32 / setting.keys_in_flight;
    let cap_text = base_cap_text(base);
    let source = replace_once(base, SPLITS_UNIFORM, &format!("long splits = {splits}L;"));
    let source = replace_once(&source, CHUNKS_UNIFORM, &format!("long chunks = {chunks}L;"));
    let source = replace_once(&source, &format!("{CAP_BASE}{cap_text};"), &format!("{CAP_BASE}{};", setting.simdgroup_cap(shape)));
    let source = replace_once(&source, LANES_BASE, &format!("constexpr short lanes_per_key = {lanes};"));
    replace_once(&source, BATCH_BASE, &format!("constexpr short batch = {};", setting.keys_per_batch))
}

fn write_setting(partial: &Output, merge: &Output, partial_base: &str, merge_base: &str, setting: &Setting, shape: &Shape) {
    let tag = setting.tag();
    let splits = setting.splits(shape);
    let chunks = setting.chunks(shape);
    partial.metal(&tag, &apply_setting(partial_base, setting, shape));
    partial.side(&tag, "width", &format!("{}", 32 * chunks));
    partial.side(&tag, "scale", &format!("{}/{}", splits, shape.base_splits));
    merge.metal(&tag, &replace_once(merge_base, SPLITS_UNIFORM, &format!("long splits = {splits}L;")));
    if setting.splits_max != 32 {
        merge.side(&tag, "width", &format!("{}", 32 * setting.splits_max));
    }
    println!("{tag} splits={splits} chunks={chunks} lanes={} batch={} cap={} merge_width={}", 32 / setting.keys_in_flight, setting.keys_per_batch, setting.simdgroup_cap(shape), 32 * setting.splits_max);
}

fn sweep_settings() -> Vec<Setting> {
    let production = Setting::production;
    let mut settings = vec![production()];
    for keys_per_split in [64, 96, 128, 192, 256, 512, 1024] {
        settings.push(Setting { keys_per_split, ..production() });
    }
    for splits_max in [16, 8, 4, 2, 1] {
        settings.push(Setting { splits_max, ..production() });
    }
    for keys_per_simdgroup in [1, 2, 4, 16, 32, 64] {
        settings.push(Setting { keys_per_simdgroup, ..production() });
    }
    for simdgroups_max in [1, 2, 8, 16, 32] {
        settings.push(Setting { simdgroups_max, ..production() });
    }
    for (keys_per_simdgroup, simdgroups_max) in [(4, 8), (2, 16), (1, 32), (4, 16), (2, 8)] {
        settings.push(Setting { keys_per_simdgroup, simdgroups_max, ..production() });
    }
    for keys_in_flight in [2, 4] {
        settings.push(Setting { keys_in_flight, ..production() });
    }
    for keys_per_batch in [2, 4, 8] {
        settings.push(Setting { keys_per_batch, ..production() });
    }
    for (keys_in_flight, keys_per_batch) in [(2, 2), (2, 4), (4, 2), (4, 4)] {
        settings.push(Setting { keys_in_flight, keys_per_batch, ..production() });
    }
    for (keys_in_flight, keys_per_simdgroup, simdgroups_max) in [(4, 4, 4), (4, 16, 4), (4, 4, 8), (4, 2, 8), (4, 8, 8), (4, 2, 16), (2, 4, 8), (2, 16, 4)] {
        settings.push(Setting { keys_in_flight, keys_per_simdgroup, simdgroups_max, ..production() });
    }
    settings
}

fn is_integer_literal_suffix(characters: &[char], at: usize) -> bool {
    let run_start = (0..at).rev().take_while(|index| characters[*index].is_ascii_digit()).last();
    let Some(start) = run_start else { return false };
    let named = start > 0 && (characters[start - 1].is_alphabetic() || characters[start - 1] == '_');
    let word_follows = characters.get(at + 1).is_some_and(|next| next.is_alphanumeric() || *next == '_');
    !named && !word_follows
}

fn strip_literal_suffix(text: &str) -> String {
    let characters: Vec<char> = text.chars().collect();
    let kept = characters
        .iter()
        .enumerate()
        .filter(|(index, character)| !(**character == 'L' && is_integer_literal_suffix(&characters, *index)))
        .map(|(_, character)| *character);
    kept.collect()
}

fn int32_body(base: &str) -> String {
    let at = base.find("kernel void ").expect("a kernel entry");
    let (head, body) = base.split_at(at);
    let body = body.replace("-9223372036854775807L", "-2147483647");
    let body = body.replace("(long)", "(int)").replace("long ", "int ");
    let body = strip_literal_suffix(&body);
    let body = body
        .replace("u.total_elements", "(int)u.total_elements")
        .replace("u.splits", "(int)u.splits")
        .replace("u.context_chunks", "(int)u.context_chunks");
    format!("{head}{body}")
}

fn write_kv_storage(output: &Output, base: &str) {
    let signature = replace_once(&replace_once(&replace_once(base, SIG_IN2, "device const ushort* in2 [[buffer(2)]]"), SIG_IN3, "device const ushort* in3 [[buffer(3)]]"), SIG_IN6, "device const ushort* in6 [[buffer(6)]]");
    let bf16 = narrow_variant(&signature, "ushort", "omega_bf16x4_to_float4", BF16_HELPER);
    output.metal("kv_bf16", &bf16);
    output.side("kv_bf16", "narrow", "2,3,6:bf16");
    let signature_bf8 = replace_once(&replace_once(&replace_once(base, SIG_IN2, "device const uchar* in2 [[buffer(2)]]"), SIG_IN3, "device const uchar* in3 [[buffer(3)]]"), SIG_IN6, "device const uchar* in6 [[buffer(6)]]");
    let bf8 = narrow_variant(&signature_bf8, "uchar", "omega_bf8x4_to_float4", BF8_HELPER);
    output.metal("kv_bf8", &bf8);
    output.side("kv_bf8", "narrow", "2,3,6:bf8");
}

fn narrow_variant(signature_patched: &str, narrow: &str, convert: &str, helper: &str) -> String {
    let pointers = format!("{PTR_KR4}\n                {PTR_KI4}\n                {PTR_V4}");
    let source = replace_once(signature_patched, &pointers, &narrow_pointers(narrow));
    let source = replace_once(&source, KEY_LOAD, &narrow_key_load(convert));
    let source = replace_once(&source, VALUE_LOAD, &narrow_value_load(convert));
    insert_before_kernel(&source, helper)
}

fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    let set = arguments[1].as_str();
    let partial_path = PathBuf::from(&arguments[2]);
    let merge_path = PathBuf::from(&arguments[3]);
    let directory = PathBuf::from(&arguments[4]);
    let shape = Shape {
        capacity: arguments[5].parse().expect("capacity"),
        head_dim: arguments[6].parse().expect("head_dim"),
        base_splits: arguments[7].parse().expect("base splits"),
        base_chunks: arguments[8].parse().expect("base chunks"),
    };
    fs::create_dir_all(&directory).expect("create output directory");
    let partial_base = fs::read_to_string(&partial_path).expect("read partial");
    let merge_base = fs::read_to_string(&merge_path).expect("read merge");
    let partial = Output { directory: directory.clone(), sha: sha_of(&partial_path), prefix: String::new() };
    let merge = Output { directory, sha: sha_of(&merge_path), prefix: String::new() };
    let production = Setting::production();
    assert_eq!(production.splits(&shape), shape.base_splits, "capacity must reproduce the captured splits");
    assert_eq!(production.chunks(&shape), shape.base_chunks, "capacity must reproduce the captured chunks");
    match set {
        "ablate" => write_ablations(&partial, &partial_base, true),
        "ablate32" => {
            let narrowed = Output { prefix: "d32_".to_string(), directory: partial.directory.clone(), sha: partial.sha.clone() };
            write_ablations(&narrowed, &div32(&partial_base), false);
        }
        "sweep" => {
            for setting in sweep_settings() {
                write_setting(&partial, &merge, &partial_base, &merge_base, &setting, &shape);
            }
            partial.metal("ctrl", &partial_base);
            merge.metal("ctrl", &merge_base);
        }
        "kv" => {
            write_kv_storage(&partial, &partial_base);
            let production_tag = production.tag();
            write_setting(&partial, &merge, &partial_base, &merge_base, &production, &shape);
            println!("kv arms use production splits; tag {production_tag} is the substitution control");
        }
        "int32" => {
            partial.metal("ctrl", &partial_base);
            merge.metal("ctrl", &merge_base);
            partial.metal("div32", &div32(&partial_base));
            partial.metal("i32_all", &int32_body(&partial_base));
            merge.metal("i32_all", &int32_body(&merge_base));
        }
        other => panic!("set `{other}`: expected ablate, ablate32, sweep, kv or int32"),
    }
}
