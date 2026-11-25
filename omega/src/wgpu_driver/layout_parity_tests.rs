//! Executes the WGSL the render steps produce on a real wgpu device and compares every result with
//! the host codec, bit for bit.
//!
//! Teaching pointer: `render::{load, store, unpack}(&Wgsl, ..)` render the steps and
//! `proxima_numeric::float_layout_codec::{decode, encode}` is the CPU algorithm they must equal.
//! The naga tests under `backend::wgpu` prove the text validates; these prove it computes.
//!
//! A host without a wgpu adapter fails these tests on purpose: a skipped parity test would read as
//! a passing one.

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use proxima_numeric::float_layout::{FloatLayout, NAMED_LAYOUTS};
use proxima_numeric::float_layout_codec::{decode, encode};

use super::{acquire_device, map_read, read_u32_bytes, storage_buffer};
use crate::backend::wgpu::dialect::Wgsl;
use crate::render::{self, Dialect, HardwareScalar};

const WORKGROUP_SIZE: usize = 64;
const ENTRY: &str = "probe";
const ENUMERABLE_WIDTH: u32 = 16;
const REPORT_LIMIT: usize = 32;

/// Codes of a 32-bit layout, read as wide-scalar patterns: zero, the smallest subnormals, the
/// subnormal/normal seam, one, the largest normals, infinities, quiet NaNs, and each neighbour.
const BOUNDARY_CODES: &[u32] = &[
    0x0000_0000,
    0x0000_0001,
    0x0000_0002,
    0x007f_ffff,
    0x0080_0000,
    0x0080_0001,
    0x3f7f_ffff,
    0x3f80_0000,
    0x3f80_0001,
    0x7f7f_fffe,
    0x7f7f_ffff,
    0x7f80_0000,
    0x7f80_0001,
    0x7fc0_0000,
    0x7fff_ffff,
    0x8000_0000,
    0x8000_0001,
    0x807f_ffff,
    0x8080_0000,
    0xbf7f_ffff,
    0xbf80_0000,
    0xbf80_0001,
    0xff7f_fffe,
    0xff7f_ffff,
    0xff80_0000,
    0xffc0_0000,
    0xffff_fffe,
    0xffff_ffff,
    0x1234_5678,
    0xdead_beef,
];

struct Spelling {
    wide: String,
    word: String,
}

fn wide_row() -> HardwareScalar {
    HardwareScalar::widest(&Wgsl).expect("the wgsl dialect spells one hardware row")
}

fn spelling(row: HardwareScalar) -> Spelling {
    Spelling {
        wide: Wgsl
            .scalar(row)
            .expect("the widest row has a wgsl spelling")
            .into_owned(),
        word: Wgsl
            .unsigned(row.bytes())
            .expect("the widest row's width has a wgsl unsigned integer")
            .into_owned(),
    }
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Gpu {
    fn acquire() -> Self {
        let (device, queue, _caps, _identity) = acquire_device().expect(
            "layout parity needs a wgpu adapter; this test fails instead of skipping so a host without one cannot pass silently",
        );
        Self { device, queue }
    }

    /// Runs `source`'s `probe` entry once per element of `inputs[output]`, one workgroup per
    /// `WORKGROUP_SIZE` elements, with `inputs[i]` bound at `@binding(i)`; returns `inputs[output]`
    /// after the dispatch.
    fn run(&self, source: &str, inputs: &[&[u32]], output: usize) -> Vec<u32> {
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(ENTRY),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(ENTRY),
                layout: None,
                module: &module,
                entry_point: Some(ENTRY),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let buffers: Vec<wgpu::Buffer> = inputs.iter().map(|words| self.upload(words)).collect();
        let entries: Vec<wgpu::BindGroupEntry<'_>> = buffers
            .iter()
            .enumerate()
            .map(|(index, buffer)| wgpu::BindGroupEntry {
                binding: index as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(ENTRY),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let elements = inputs[output].len();
        let bytes = (elements * size_of_word()) as u64;
        let staged = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("layout-parity-readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("layout-parity-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(ENTRY),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(elements.div_ceil(WORKGROUP_SIZE).max(1) as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&buffers[output], 0, &staged, 0, bytes);
        self.queue.submit(core::iter::once(encoder.finish()));
        let raw = map_read(&self.device, &staged).expect("the readback buffer maps for reading");
        let mut words = read_u32_bytes(&raw).expect("the readback is whole words");
        words.truncate(elements);
        words
    }

    fn upload(&self, words: &[u32]) -> wgpu::Buffer {
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_ne_bytes()).collect();
        let buffer = storage_buffer(
            &self.device,
            "layout-parity",
            bytes.len(),
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        );
        self.queue.write_buffer(&buffer, 0, &bytes);
        buffer
    }
}

const fn size_of_word() -> usize {
    core::mem::size_of::<u32>()
}

fn code_mask(layout: &FloatLayout) -> u32 {
    match u32::from(layout.width()) {
        32.. => u32::MAX,
        width => (1 << width) - 1,
    }
}

/// Every code of a layout of 16 bits or fewer; the boundary samples, masked to the width, above.
fn codes_for(layout: &FloatLayout) -> Vec<u32> {
    let width = u32::from(layout.width());
    if width <= ENUMERABLE_WIDTH {
        return (0..1u32 << width).collect();
    }
    let mask = code_mask(layout);
    let mut codes: Vec<u32> = BOUNDARY_CODES.iter().map(|code| code & mask).collect();
    codes.sort_unstable();
    codes.dedup();
    codes
}

fn bits_are_nan(bits: u32) -> bool {
    decode(bits, wide_row().layout()).is_nan()
}

/// A wide NaN is compared by NaN-ness: the payload a NaN carries through shader arithmetic is not
/// something WGSL defines, while every other value, subnormals and signed zeros included, must
/// match bit for bit.
fn load_matches(expected: u32, observed: u32) -> bool {
    if bits_are_nan(expected) {
        bits_are_nan(observed)
    } else {
        expected == observed
    }
}

fn describe(differing: &[(&str, u32)]) -> String {
    let shown = &differing[..differing.len().min(REPORT_LIMIT)];
    format!("{} differ, first {:x?}", differing.len(), shown)
}

fn module_source(spelling: &Spelling, declarations: &str, functions: &str, body: &str) -> String {
    let word = &spelling.word;
    format!(
        "{declarations}\n{functions}\n@compute @workgroup_size({WORKGROUP_SIZE})\nfn {ENTRY}(@builtin(global_invocation_id) id: vec3<{word}>) {{\n    if (id.x >= arrayLength(&data)) {{\n        return;\n    }}\n    {body}\n}}\n"
    )
}

fn data_declaration(spelling: &Spelling, binding: usize, access: &str) -> String {
    format!(
        "@group(0) @binding({binding}) var<storage, {access}> data: array<{}>;",
        spelling.word
    )
}

/// The inputs a store is held to for `codes`: each code's decoded value, one bit either side of
/// it, and the exact midpoint to the next code with one bit either side of that, plus the
/// boundary patterns themselves.
fn store_inputs(layout: &FloatLayout, codes: &[u32]) -> Vec<u32> {
    let mask = code_mask(layout);
    let mut inputs = Vec::new();
    for &code in codes {
        let value = decode(code, *layout).to_bits();
        inputs.extend([value.wrapping_sub(1), value, value.wrapping_add(1)]);
        if code < mask {
            let midpoint = (decode(code, *layout) + decode(code + 1, *layout)) * 0.5;
            if midpoint.is_finite() {
                let tie = midpoint.to_bits();
                inputs.extend([tie.wrapping_sub(1), tie, tie.wrapping_add(1)]);
            }
        }
    }
    inputs.extend_from_slice(BOUNDARY_CODES);
    inputs.sort_unstable();
    inputs.dedup();
    inputs
}

/// Packs `codes` densely, `width` bits each, low bits first, into little-endian words with one
/// spare word so a straddling read of the last element stays inside the payload.
fn pack(codes: &[u32], width: u32) -> Vec<u32> {
    let width = width as usize;
    let words = (codes.len() * width).div_ceil(32) + 1;
    let mut bytes = vec![0u8; words * size_of_word()];
    for (index, &code) in codes.iter().enumerate() {
        for bit in 0..width {
            if (code >> bit) & 1 == 1 {
                let position = index * width + bit;
                bytes[position / 8] |= 1 << (position % 8);
            }
        }
    }
    let (chunks, _) = bytes.as_chunks::<4>();
    chunks.iter().map(|chunk| u32::from_le_bytes(*chunk)).collect()
}

#[test]
fn wgsl_load_equals_the_host_decode_bit_for_bit_for_every_code() {
    let gpu = Gpu::acquire();
    let wide = wide_row();
    let spelling = spelling(wide);
    let mut compared = 0usize;
    let mut differing: Vec<(&str, u32)> = Vec::new();

    for (layout, name) in NAMED_LAYOUTS.iter() {
        let Ok(function) = render::load(&Wgsl, wide, *layout, "probe_load") else {
            continue;
        };
        let codes = codes_for(layout);
        let body = format!(
            "data[id.x] = {};",
            Wgsl.bitcast(&spelling.word, "probe_load(data[id.x])")
        );
        let source = module_source(
            &spelling,
            &data_declaration(&spelling, 0, "read_write"),
            &function,
            &body,
        );

        let observed = gpu.run(&source, &[&codes], 0);

        for (code, got) in codes.iter().zip(&observed) {
            compared += 1;
            if !load_matches(decode(*code, *layout).to_bits(), *got) {
                differing.push((name, *code));
            }
        }
    }

    assert!(compared > 0, "no layout rendered a load");
    assert!(differing.is_empty(), "{}", describe(&differing));
}

#[test]
fn wgsl_store_equals_the_host_encode_bit_for_bit_on_boundaries_and_ties() {
    let gpu = Gpu::acquire();
    let wide = wide_row();
    let spelling = spelling(wide);
    let mut compared = 0usize;
    let mut differing: Vec<(&str, u32)> = Vec::new();

    for (layout, name) in NAMED_LAYOUTS.iter() {
        let Ok(function) = render::store(&Wgsl, wide, *layout, "probe_store") else {
            continue;
        };
        let inputs = store_inputs(layout, &codes_for(layout));
        let body = format!(
            "data[id.x] = probe_store({});",
            Wgsl.bitcast(&spelling.wide, "data[id.x]")
        );
        let source = module_source(
            &spelling,
            &data_declaration(&spelling, 0, "read_write"),
            &function,
            &body,
        );

        let observed = gpu.run(&source, &[&inputs], 0);

        for (input, got) in inputs.iter().zip(&observed) {
            compared += 1;
            let expected = encode(decode(*input, wide.layout()), *layout);
            if expected != *got {
                differing.push((name, *input));
            }
        }
    }

    assert!(compared > 0, "no layout rendered a store");
    assert!(differing.is_empty(), "{}", describe(&differing));
}

#[test]
fn wgsl_unpack_equals_the_host_decode_for_every_packed_element() {
    let gpu = Gpu::acquire();
    let wide = wide_row();
    let spelling = spelling(wide);
    let mut compared = 0usize;
    let mut sub_byte = 0usize;
    let mut straddling = 0usize;
    let mut differing: Vec<(&str, u32)> = Vec::new();

    for (layout, name) in NAMED_LAYOUTS.iter() {
        let Ok(function) = render::unpack(&Wgsl, wide, *layout, "probe_unpack") else {
            continue;
        };
        let width = u32::from(layout.width());
        let codes = codes_for(layout);
        let payload = pack(&codes, width);
        let outputs = vec![0u32; codes.len()];
        let declarations = format!(
            "@group(0) @binding(0) var<storage, read> payload: array<{}>;\n{}",
            spelling.word,
            data_declaration(&spelling, 1, "read_write")
        );
        let body = format!(
            "data[id.x] = {};",
            Wgsl.bitcast(&spelling.word, "probe_unpack(id.x)")
        );
        let source = module_source(&spelling, &declarations, &function, &body);

        let observed = gpu.run(&source, &[&payload, &outputs], 1);

        for (code, got) in codes.iter().zip(&observed) {
            compared += 1;
            if !load_matches(decode(*code, *layout).to_bits(), *got) {
                differing.push((name, *code));
            }
        }
        sub_byte += usize::from(width < 8 && 8 % width == 0);
        straddling += usize::from(width % 8 != 0 && 8 % width != 0);
    }

    assert!(compared > 0, "no layout rendered an unpack");
    assert!(sub_byte > 0, "no sub-byte layout was unpacked");
    assert!(straddling > 0, "no byte-straddling layout was unpacked");
    assert!(differing.is_empty(), "{}", describe(&differing));
}
