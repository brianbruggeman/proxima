//! Experiment: int8 tensor-core prefill GEMM over block-packed weights, timed against the
//! fp16 mma plan kernel. Activation is quantized to int8 once per launch (one f32 scale per 32
//! K elements), the weight codes are unpacked to int8 into shared memory once per scale group,
//! and `mma.sync.m8n8k16.s32.s8.s8.s32` accumulates each 32-K chunk exactly before the fold.
//!
//! args: <features> <depth> <tokens> <runs>

use std::error::Error;
use std::sync::Arc;

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use omega::Codec;
use omega::backend::{Engine, GpuDriver, execute_plan_named, plan_named};
use proxima_gguf::quant::block_layout;
use proxima_numeric::block_layout::BlockLayout;
use proxima_numeric::float_layout::FloatLayout;
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp, append, projection};
use proxima_numeric::{DType, NumericPolicy};

type Failure = Box<dyn Error>;

const FEATURE_TILE: usize = 128;
const TOKEN_TILE: usize = 64;
const THREADS: u32 = 256;
const ACT_GROUP: usize = 32;
const ROW_PAD: usize = 16;
const PARITY_TOLERANCE: f64 = 2e-5;
const CHECKED_FEATURES: usize = 4096;

struct GroupLayout {
    code_bits: u32,
    group_elements: usize,
    block_bytes: usize,
    scale_offset: usize,
    codes_offset: usize,
    code_bias: i32,
}

fn group_layout(layout: &BlockLayout) -> Result<GroupLayout, Failure> {
    let element = &layout.element;
    let bits = u32::from(element.low.bits);
    let base = element.low.address(0);
    let contiguous = (0..usize::from(layout.elements))
        .all(|index| element.low.address(index) == base + index * bits as usize);
    let plain = element.high.is_none() && element.packed_radix == 0 && layout.multiplier == 1;
    let unscaled = layout.min.is_none() && layout.sub.is_none() && layout.scale.layout == FloatLayout::F16;
    if !(contiguous && plain && unscaled && bits == 2 && base.is_multiple_of(8)) {
        return Err("block layout is not a single contiguous 2-bit code run with one f16 scale".into());
    }
    Ok(GroupLayout {
        code_bits: bits,
        group_elements: usize::from(layout.elements),
        block_bytes: usize::from(layout.block_bytes),
        scale_offset: usize::from(layout.scale.byte_offset),
        codes_offset: base / 8,
        code_bias: i32::from(element.bias),
    })
}

impl GroupLayout {
    fn code_bytes(&self) -> usize {
        self.group_elements * self.code_bits as usize / 8
    }

    fn spread_text(&self) -> String {
        let mask = (1u32 << self.code_bits) - 1;
        (0..8 / self.code_bits)
            .map(|slot| {
                format!(
                    "    spread |= ((packed_byte >> {}u) & {mask}u) << {}u;\n",
                    self.code_bits * slot,
                    8 * slot
                )
            })
            .collect()
    }

    fn check_tile_fit(&self) -> Result<(), Failure> {
        let fits = self.group_elements == 4 * ACT_GROUP
            && self.code_bytes().is_multiple_of(16)
            && FEATURE_TILE * (self.code_bytes() / 16) == THREADS as usize
            && self.block_bytes.is_multiple_of(2)
            && self.codes_offset.is_multiple_of(2)
            && self.scale_offset.is_multiple_of(2);
        fits.then_some(()).ok_or_else(|| "layout does not fit the 128x64 tile staging".into())
    }

    fn defines(&self) -> String {
        format!(
            "#define SLAB {}\n#define ROW_STRIDE {}\n#define BLOCK_BYTES {}\n#define SCALE_OFFSET {}\n\
             #define CODES_OFFSET {}\n#define CODE_BIAS {}\n#define CODE_BYTES {}\n\
             #define FEATURE_TILE {FEATURE_TILE}\n#define TOKEN_TILE {TOKEN_TILE}\n#define ACT_GROUP {ACT_GROUP}\n",
            self.group_elements,
            self.group_elements + ROW_PAD,
            self.block_bytes,
            self.scale_offset,
            self.codes_offset,
            self.code_bias,
            self.code_bytes(),
        )
    }
}

const GEMM_PROLOGUE: &str = r#"
#define CHUNKS (SLAB / ACT_GROUP)
#define WEIGHT_PARTS (CODE_BYTES / 16)
#define ACT_PART_BYTES (SLAB / 4)
#define WEIGHT_TILE_BYTES (FEATURE_TILE * ROW_STRIDE)
#define ACT_TILE_BYTES (TOKEN_TILE * ROW_STRIDE)
#define WSCALE_BYTES (FEATURE_TILE * 4)
#define ASCALE_BYTES (TOKEN_TILE * CHUNKS * 4)
#define BUFFER_BYTES (WEIGHT_TILE_BYTES + ACT_TILE_BYTES + WSCALE_BYTES + ASCALE_BYTES)

struct Stage { unsigned int codes[4]; unsigned int scale_bits; uint4 act[2]; float4 act_scale; };

__device__ __forceinline__ float half_bits_to_float(unsigned int bits) {
    float value;
    asm("cvt.f32.f16 %0, %1;" : "=f"(value) : "h"((unsigned short)bits));
    return value;
}
__device__ __forceinline__ void mma_s8(int *acc, int fragment_a, int fragment_b) {
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
        : "+r"(acc[0]), "+r"(acc[1]) : "r"(fragment_a), "r"(fragment_b));
}
"#;

const GEMM_STAGING: &str = r#"
__device__ __forceinline__ void load_stage(Stage &stage, const unsigned char *weight, const signed char *act_q,
        const float *act_scales, int feature_base, int token_base, int depth, int slab) {
    int tid = threadIdx.x;
    int feature = feature_base + tid / WEIGHT_PARTS;
    int part = tid % WEIGHT_PARTS;
    size_t row_bytes = (size_t)(depth / SLAB) * BLOCK_BYTES;
    const unsigned char *block = weight + (size_t)feature * row_bytes + (size_t)slab * BLOCK_BYTES;
    const unsigned short *words = (const unsigned short *)(block + CODES_OFFSET + part * 16);
    #pragma unroll
    for (int word = 0; word < 4; word++) {
        stage.codes[word] = (unsigned int)words[2 * word] | ((unsigned int)words[2 * word + 1] << 16);
    }
    stage.scale_bits = *(const unsigned short *)(block + SCALE_OFFSET);
    int token = token_base + (tid >> 2);
    const uint4 *source = (const uint4 *)(act_q + (size_t)token * depth + (size_t)slab * SLAB + (tid & 3) * ACT_PART_BYTES);
    stage.act[0] = source[0];
    stage.act[1] = source[1];
    stage.act_scale = *(const float4 *)(act_scales + (size_t)token * (depth / ACT_GROUP) + (size_t)slab * CHUNKS);
}
__device__ __forceinline__ void store_stage(const Stage &stage, unsigned char *buffer) {
    int tid = threadIdx.x;
    int feature_local = tid / WEIGHT_PARTS;
    int part = tid % WEIGHT_PARTS;
    uint4 *weight_dst = (uint4 *)(buffer + feature_local * ROW_STRIDE + part * (SLAB / WEIGHT_PARTS));
    #pragma unroll
    for (int word = 0; word < 4; word++) {
        unsigned int packed = stage.codes[word];
        weight_dst[word] = make_uint4(spread_codes(packed & 0xFFu), spread_codes((packed >> 8) & 0xFFu),
            spread_codes((packed >> 16) & 0xFFu), spread_codes(packed >> 24));
    }
    if (part == 0) {
        ((float *)(buffer + WEIGHT_TILE_BYTES + ACT_TILE_BYTES))[feature_local] = half_bits_to_float(stage.scale_bits);
    }
    int token_local = tid >> 2;
    int act_part = tid & 3;
    uint4 *act_dst = (uint4 *)(buffer + WEIGHT_TILE_BYTES + token_local * ROW_STRIDE + act_part * ACT_PART_BYTES);
    act_dst[0] = stage.act[0];
    act_dst[1] = stage.act[1];
    if (act_part == 0) {
        ((float4 *)(buffer + WEIGHT_TILE_BYTES + ACT_TILE_BYTES + WSCALE_BYTES))[token_local] = stage.act_scale;
    }
}
"#;

const GEMM_COMPUTE: &str = r#"
__device__ __forceinline__ void compute_slab(const unsigned char *buffer, int warp_feature, int warp_token, int lane,
        float (&facc)[4][4][2]) {
    const signed char *weight_tile = (const signed char *)buffer;
    const signed char *act_tile = (const signed char *)(buffer + WEIGHT_TILE_BYTES);
    const float *weight_scales = (const float *)(buffer + WEIGHT_TILE_BYTES + ACT_TILE_BYTES);
    const float *act_scales = (const float *)(buffer + WEIGHT_TILE_BYTES + ACT_TILE_BYTES + WSCALE_BYTES);
    #pragma unroll
    for (int chunk = 0; chunk < CHUNKS; chunk++) {
        int acc[4][4][2] = {};
        #pragma unroll
        for (int step = 0; step < ACT_GROUP / 16; step++) {
            int depth_at = chunk * ACT_GROUP + step * 16 + (lane & 3) * 4;
            int fragment_a[4];
            int fragment_b[4];
            #pragma unroll
            for (int tile = 0; tile < 4; tile++) {
                fragment_a[tile] = *(const int *)(weight_tile + (warp_feature * 32 + tile * 8 + (lane >> 2)) * ROW_STRIDE + depth_at);
                fragment_b[tile] = *(const int *)(act_tile + (warp_token * 32 + tile * 8 + (lane >> 2)) * ROW_STRIDE + depth_at);
            }
            #pragma unroll
            for (int row = 0; row < 4; row++) {
                #pragma unroll
                for (int col = 0; col < 4; col++) { mma_s8(acc[row][col], fragment_a[row], fragment_b[col]); }
            }
        }
        #pragma unroll
        for (int row = 0; row < 4; row++) {
            float weight_scale = weight_scales[warp_feature * 32 + row * 8 + (lane >> 2)];
            #pragma unroll
            for (int col = 0; col < 4; col++) {
                #pragma unroll
                for (int pair = 0; pair < 2; pair++) {
                    int token_local = warp_token * 32 + col * 8 + (lane & 3) * 2 + pair;
                    float scale = weight_scale * act_scales[token_local * CHUNKS + chunk];
                    facc[row][col][pair] += scale * (float)acc[row][col][pair];
                }
            }
        }
    }
}
"#;

const GEMM_ENTRY_HEAD: &str = r#"
extern "C" __global__ void __launch_bounds__(256) int8_gemm(const unsigned char *weight, const signed char *act_q,
        const float *act_scales, float *out, int depth, int tokens) {
    extern __shared__ __align__(16) unsigned char smem[];
    int feature_base = blockIdx.x * FEATURE_TILE;
    int token_base = blockIdx.y * TOKEN_TILE;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int warp_feature = warp & 3;
    int warp_token = warp >> 2;
    int slabs = depth / SLAB;
    float facc[4][4][2] = {};
    Stage stage;
"#;

const SERIAL_LOOP: &str = r#"
    for (int slab = 0; slab < slabs; slab++) {
        load_stage(stage, weight, act_q, act_scales, feature_base, token_base, depth, slab);
        store_stage(stage, smem);
        __syncthreads();
        compute_slab(smem, warp_feature, warp_token, lane, facc);
        __syncthreads();
    }
"#;

const PIPELINED_LOOP: &str = r#"
    load_stage(stage, weight, act_q, act_scales, feature_base, token_base, depth, 0);
    store_stage(stage, smem);
    __syncthreads();
    for (int slab = 0; slab < slabs; slab++) {
        bool more = slab + 1 < slabs;
        if (more) { load_stage(stage, weight, act_q, act_scales, feature_base, token_base, depth, slab + 1); }
        compute_slab(smem + (slab & 1) * BUFFER_BYTES, warp_feature, warp_token, lane, facc);
        if (more) { store_stage(stage, smem + ((slab + 1) & 1) * BUFFER_BYTES); }
        __syncthreads();
    }
"#;

const GEMM_ENTRY_TAIL: &str = r#"
    #pragma unroll
    for (int row = 0; row < 4; row++) {
        #pragma unroll
        for (int col = 0; col < 4; col++) {
            int feature = feature_base + warp_feature * 32 + row * 8 + (lane >> 2);
            int token = token_base + warp_token * 32 + col * 8 + (lane & 3) * 2;
            *(float2 *)(out + (size_t)feature * tokens + token) = make_float2(facc[row][col][0], facc[row][col][1]);
        }
    }
}
"#;

const QUANTIZE_SOURCE: &str = r#"
extern "C" __global__ void quantize_activation(const float *activation, signed char *quantized, float *scales, long long groups) {
    long long group = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (group >= groups) { return; }
    float value = activation[group * 32 + lane];
    float amax = fabsf(value);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset)); }
    float step = amax / 127.0f;
    float level = step == 0.0f ? 0.0f : roundf(value / step);
    quantized[group * 32 + lane] = (signed char)level;
    if (lane == 0) { scales[group] = step; }
}
"#;

fn gemm_source(group: &GroupLayout, pipelined: bool) -> String {
    let spread = format!(
        "__device__ __forceinline__ unsigned int spread_codes(unsigned int packed_byte) {{\n    unsigned int spread = 0;\n{}    return __vsub4(spread, (unsigned int)CODE_BIAS * 0x01010101u);\n}}\n",
        group.spread_text()
    );
    let main_loop = if pipelined { PIPELINED_LOOP } else { SERIAL_LOOP };
    [
        group.defines().as_str(),
        GEMM_PROLOGUE,
        spread.as_str(),
        GEMM_STAGING,
        GEMM_COMPUTE,
        GEMM_ENTRY_HEAD,
        main_loop,
        GEMM_ENTRY_TAIL,
    ]
    .concat()
}

fn compile(context: &Arc<CudaContext>, source: &str, entry: &str) -> Result<CudaFunction, Failure> {
    let options = CompileOptions { arch: Some("sm_75"), ..CompileOptions::default() };
    let ptx = compile_ptx_with_opts(source, options)?;
    Ok(context.load_module(ptx)?.load_function(entry)?)
}

fn median_micros(
    context: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    runs: usize,
    mut launch: impl FnMut() -> Result<(), Failure>,
) -> Result<f64, Failure> {
    let mut millis = Vec::with_capacity(runs);
    for _ in 0..runs + 2 {
        let start = context.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        let end = context.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        start.record(stream)?;
        launch()?;
        end.record(stream)?;
        end.synchronize()?;
        millis.push(start.elapsed_ms(&end)?);
    }
    let mut measured = millis.split_off(2);
    measured.sort_by(f32::total_cmp);
    Ok(f64::from(measured[measured.len() / 2]) * 1000.0)
}

fn matmul_program(features: u32, depth: u32, tokens: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let input = |program: &mut Vec<Op>, dtype, rows, name: &str| {
        append(
            program,
            Op::Input {
                dtype,
                shape: vec![Extent::Static(rows), Extent::Static(depth)],
                name: Some(name.into()),
            },
        )
    };
    let weight = input(&mut program, DType::UINT8, features, "weight");
    let activation = input(&mut program, DType::FLOAT32, tokens, "activation");
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(projection(3, &[0, 2]))),
                (activation, IndexMap::Affine(projection(3, &[1, 2]))),
            ],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("prefill_gemm".into()),
        }),
    );
    (program, sum)
}

fn arbitrary_blocks(layout: &BlockLayout, blocks: usize, seed: u64) -> Vec<u8> {
    let mut lcg = Lcg(seed);
    let block_bytes = usize::from(layout.block_bytes);
    let scale_at = usize::from(layout.scale.byte_offset);
    let mut bytes: Vec<u8> = (0..blocks * block_bytes)
        .map(|_| (lcg.next_unit() * 256.0) as u8)
        .collect();
    for block in bytes.chunks_exact_mut(block_bytes) {
        let exponent = 9 + (lcg.next_unit() * 4.0) as u16;
        block[scale_at..scale_at + 2].copy_from_slice(&(exponent << 10).to_le_bytes());
    }
    bytes
}

fn fake_quantize(activation: &[f32]) -> Vec<f32> {
    activation
        .as_chunks::<ACT_GROUP>().0.iter()
        .flat_map(|group| {
            let step = group.iter().fold(0.0f32, |amax, value| amax.max(value.abs())) / 127.0;
            group.iter().map(move |value| if step == 0.0 { 0.0 } else { (value / step).round() * step })
        })
        .collect()
}

fn chunk_dot(left: &[f32], right: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    for (left_part, right_part) in left.as_chunks::<8>().0.iter().zip(right.as_chunks::<8>().0) {
        for (lane, (a_value, b_value)) in lanes.iter_mut().zip(left_part.iter().zip(right_part)) {
            *lane += a_value * b_value;
        }
    }
    lanes.iter().sum()
}

fn dot(left: &[f32], right: &[f32], group: usize) -> f64 {
    left.chunks_exact(group)
        .zip(right.chunks_exact(group))
        .map(|(left_chunk, right_chunk)| f64::from(chunk_dot(left_chunk, right_chunk)))
        .sum()
}

struct References {
    exact: Vec<f32>,
    fake_quantized: Vec<f32>,
}

fn references(
    weights: &[f32],
    activation: &[f32],
    fake: &[f32],
    checked: &[usize],
    depth: usize,
    tokens: usize,
    group: usize,
) -> References {
    let mut exact = vec![0.0f32; checked.len() * tokens];
    let mut fake_quantized = vec![0.0f32; checked.len() * tokens];
    let threads = std::thread::available_parallelism().map_or(4, usize::from);
    let rows_per_thread = checked.len().div_ceil(threads);
    std::thread::scope(|scope| {
        let parts = exact
            .chunks_mut(rows_per_thread * tokens)
            .zip(fake_quantized.chunks_mut(rows_per_thread * tokens))
            .zip(checked.chunks(rows_per_thread));
        for ((exact_part, fake_part), rows) in parts {
            scope.spawn(move || {
                for (row_at, &feature) in rows.iter().enumerate() {
                    let weight_row = &weights[feature * depth..][..depth];
                    for token in 0..tokens {
                        let at = row_at * tokens + token;
                        exact_part[at] = dot(weight_row, &activation[token * depth..][..depth], group) as f32;
                        fake_part[at] = dot(weight_row, &fake[token * depth..][..depth], group) as f32;
                    }
                }
            });
        }
    });
    References { exact, fake_quantized }
}

struct ErrorStats {
    max_abs: f64,
    rms_err: f64,
    rms_ref: f64,
    peak: f64,
}

fn compare(got: &[f32], want: &[f32]) -> ErrorStats {
    let count = want.len() as f64;
    let diffs = got.iter().zip(want).map(|(a_value, b_value)| f64::from(a_value - b_value).abs());
    let max_abs = diffs.clone().fold(0.0, f64::max);
    let rms_err = (diffs.map(|diff| diff * diff).sum::<f64>() / count).sqrt();
    let rms_ref = (want.iter().map(|value| f64::from(*value).powi(2)).sum::<f64>() / count).sqrt();
    let peak = want.iter().fold(0.0f64, |largest, value| largest.max(f64::from(value.abs())));
    ErrorStats { max_abs, rms_err, rms_ref, peak }
}

fn report(label: &str, stats: &ErrorStats) {
    let relative = stats.max_abs / stats.peak;
    let verdict = if relative < PARITY_TOLERANCE { "PASS" } else { "FAIL" };
    println!(
        "{label}: max_abs_err={:.6e} rms_err={:.6e} rms_ref={:.6e} max_abs/peak={relative:.3e} vs {PARITY_TOLERANCE:e} (cuda_parity.rs:4698): {verdict}",
        stats.max_abs, stats.rms_err, stats.rms_ref
    );
}

fn gather_rows(output: &[f32], checked: &[usize], tokens: usize) -> Vec<f32> {
    checked.iter().flat_map(|&feature| output[feature * tokens..][..tokens].iter().copied()).collect()
}

struct Geometry {
    features: usize,
    depth: usize,
    tokens: usize,
    runs: usize,
}

fn parse_geometry() -> Result<Geometry, Failure> {
    let args: Vec<String> = std::env::args().collect();
    let number = |at: usize| -> Result<usize, Failure> {
        Ok(args.get(at).ok_or("args: <features> <depth> <tokens> <runs>")?.parse()?)
    };
    let geometry = Geometry { features: number(1)?, depth: number(2)?, tokens: number(3)?, runs: number(4)? };
    let aligned = geometry.features.is_multiple_of(FEATURE_TILE) && geometry.tokens.is_multiple_of(TOKEN_TILE) && geometry.depth.is_multiple_of(128);
    aligned.then_some(geometry).ok_or_else(|| "features %128, tokens %64, depth %128 must be 0".into())
}

fn fp16_plan_output(layout_codec: Codec, weight: &[u8], activation: &[f32], geometry: &Geometry) -> Result<Vec<f32>, Failure> {
    let (program, root) = matmul_program(geometry.features as u32, geometry.depth as u32, geometry.tokens as u32);
    let named: Vec<(&str, QuantizedBlock<'_>)> = vec![
        ("weight", QuantizedBlock::Packed { codec: layout_codec, bytes: weight }),
        ("activation", QuantizedBlock::Float32(activation)),
    ];
    let policy = NumericPolicy::bit_exact().with_reassociation(true).with_activation_quantization(true);
    let mut plan = plan_named(Engine::Gpu, Some(GpuDriver::Cuda), &program, &[], &named, &[root], policy)?;
    Ok(execute_plan_named(&mut plan, &named)?.root().to_vec())
}

fn run_variant(
    context: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    group: &GroupLayout,
    pipelined: bool,
    buffers: &DeviceBuffers,
    geometry: &Geometry,
) -> Result<(f64, Vec<f32>), Failure> {
    let function = compile(context, &gemm_source(group, pipelined), "int8_gemm")?;
    let buffer_bytes = FEATURE_TILE * (group.group_elements + ROW_PAD)
        + TOKEN_TILE * (group.group_elements + ROW_PAD)
        + FEATURE_TILE * 4
        + TOKEN_TILE * (group.group_elements / ACT_GROUP) * 4;
    let shared = if pipelined { 2 * buffer_bytes } else { buffer_bytes };
    function.set_attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared as i32)?;
    let config = LaunchConfig {
        grid_dim: ((geometry.features / FEATURE_TILE) as u32, (geometry.tokens / TOKEN_TILE) as u32, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: shared as u32,
    };
    let mut output = stream.alloc_zeros::<f32>(geometry.features * geometry.tokens)?;
    let (depth, tokens) = (geometry.depth as i32, geometry.tokens as i32);
    let micros = median_micros(context, stream, geometry.runs, || {
        let mut args = stream.launch_builder(&function);
        args.arg(&buffers.weight).arg(&buffers.quantized).arg(&buffers.scales).arg(&mut output).arg(&depth).arg(&tokens);
        unsafe { args.launch(config) }?;
        Ok(())
    })?;
    Ok((micros, stream.clone_dtoh(&output)?))
}

struct DeviceBuffers {
    weight: cudarc::driver::CudaSlice<u8>,
    quantized: cudarc::driver::CudaSlice<i8>,
    scales: cudarc::driver::CudaSlice<f32>,
}

fn quantize_on_device(
    context: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    activation: &[f32],
    weight: &[u8],
    runs: usize,
) -> Result<(DeviceBuffers, f64), Failure> {
    let function = compile(context, QUANTIZE_SOURCE, "quantize_activation")?;
    let device_activation = stream.clone_htod(activation)?;
    let groups = activation.len() / ACT_GROUP;
    let mut quantized = stream.alloc_zeros::<i8>(activation.len())?;
    let mut scales = stream.alloc_zeros::<f32>(groups)?;
    let config = LaunchConfig {
        grid_dim: ((groups * 32).div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let group_count = groups as i64;
    let micros = median_micros(context, stream, runs, || {
        let mut args = stream.launch_builder(&function);
        args.arg(&device_activation).arg(&mut quantized).arg(&mut scales).arg(&group_count);
        unsafe { args.launch(config) }?;
        Ok(())
    })?;
    let weight = stream.clone_htod(weight)?;
    Ok((DeviceBuffers { weight, quantized, scales }, micros))
}

fn main() -> Result<(), Failure> {
    let geometry = parse_geometry()?;
    let codec = Codec::Pq2_0;
    let layout = codec.block_layout().ok_or("codec has no block layout")?;
    let group = group_layout(&layout)?;
    group.check_tile_fit()?;
    let blocks = geometry.features * geometry.depth / group.group_elements;
    let weight = arbitrary_blocks(&layout, blocks, 41);
    let mut lcg = Lcg(59);
    let activation: Vec<f32> = (0..geometry.depth * geometry.tokens).map(|_| lcg.next_unit() * 4.0 - 2.0).collect();
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let (buffers, quantize_micros) = quantize_on_device(&context, &stream, &activation, &weight, geometry.runs)?;
    let flop = 2.0 * (geometry.features * geometry.depth * geometry.tokens) as f64;
    println!("shape features={} depth={} tokens={} quantize_median_us={quantize_micros:.1}", geometry.features, geometry.depth, geometry.tokens);
    let mut outputs = Vec::new();
    for (name, pipelined) in [("serial", false), ("pipelined", true)] {
        let (micros, output) = run_variant(&context, &stream, &group, pipelined, &buffers, &geometry)?;
        println!("variant={name} gemm_median_us={micros:.1} tflops_equiv={:.2}", flop / micros / 1e6);
        outputs.push((name, output));
    }
    let mut decoded = vec![0.0f32; geometry.features * geometry.depth];
    block_layout::dequantize(&layout, &weight, &mut decoded)?;
    let fake = fake_quantize(&activation);
    let stride = geometry.features.div_ceil(CHECKED_FEATURES);
    let checked: Vec<usize> = (0..geometry.features).step_by(stride).collect();
    let refs = references(&decoded, &activation, &fake, &checked, geometry.depth, geometry.tokens, group.group_elements);
    for (name, output) in &outputs {
        let rows = gather_rows(output, &checked, geometry.tokens);
        report(&format!("int8 {name} vs fake-quantized-activation reference"), &compare(&rows, &refs.fake_quantized));
        report(&format!("int8 {name} vs unquantized f32 reference"), &compare(&rows, &refs.exact));
    }
    let fp16 = fp16_plan_output(codec, &weight, &activation, &geometry)?;
    let fp16_rows = gather_rows(&fp16, &checked, geometry.tokens);
    report("fp16 plan vs unquantized f32 reference", &compare(&fp16_rows, &refs.exact));
    println!("checked_features={} of {}", checked.len(), geometry.features);
    Ok(())
}
