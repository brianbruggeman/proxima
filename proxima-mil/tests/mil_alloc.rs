//! The writer's allocation budget (SPEC R20), counted by a global allocator
//! that tallies per thread so the test harness's own threads never enter the
//! count: encoding one node and writing one weight allocate nothing, and a
//! package is exactly three buffers (model, weights, manifest) whatever the
//! program size. Programs are real ONNX wire bytes parsed and lowered by
//! `proxima-onnx`, the same path the door takes.

#![cfg(feature = "alloc-count")]
// fixtures are hand-built to succeed; an expect failure here IS the test failing.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use proxima_mil::weights::{BlobWriter, SourceDType, entry_bytes};
use proxima_mil::wire::Encoder;
use proxima_mil::writer::{Context, MilRequest, MilWriter, WeightView, emit_node, measure};
use proxima_onnx::lower::lower_graph;
use proxima_onnx::messages::{
    AttributeProto, Dimension, DimensionValue, GraphProto, ModelProto, NodeProto, TensorProto,
    TensorShapeProto, TypeProto, TypeProtoTensor, TypeValue, ValueInfoProto,
};
use proxima_onnx::pipe::parse_complete;
use proxima_onnx::write_model_proto;
use proxima_tensor::{
    AxisTerm, DType, Extent, IndexMap, NodeId, Op, ScalarOp, Shapes, affine, append, infer,
    projection,
};

const ITERATIONS: usize = 100_000;
const OUTPUT: &str = "y";
const WIDTH: i64 = 4;
const LARGE_LAYERS: usize = 250;

thread_local! {
    static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
}

struct Counting;

fn tally() {
    ALLOCATIONS.with(|count| count.set(count.get() + 1));
}

// forwards to System unchanged; the tally is the only addition.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        tally();
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        tally();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        tally();
        unsafe { System.realloc(pointer, layout, size) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

fn allocations_during<T>(work: impl FnOnce() -> T) -> (T, u64) {
    let before = ALLOCATIONS.with(Cell::get);
    let produced = work();
    (produced, ALLOCATIONS.with(Cell::get) - before)
}

struct Program {
    ops: Vec<Op>,
    shapes: Shapes,
    encoded: Vec<(String, Vec<u8>)>,
}

impl Program {
    fn views(&self) -> Vec<WeightView<'_>> {
        self.encoded
            .iter()
            .map(|(name, bytes)| WeightView {
                name,
                dtype: SourceDType::Float32,
                bytes,
            })
            .collect()
    }
}

fn fixed(extent: i64) -> Dimension<'static> {
    Dimension {
        value: Some(DimensionValue::Value(extent)),
        denotation: "",
    }
}

fn float_input(name: &str) -> ValueInfoProto<'_> {
    let tensor = TypeProtoTensor {
        elem_type: 1,
        shape: Some(TensorShapeProto {
            dim: vec![fixed(2), fixed(WIDTH)],
        }),
    };
    ValueInfoProto {
        name,
        r#type: Some(TypeProto {
            value: Some(TypeValue::Tensor(tensor)),
            denotation: "",
        }),
        doc_string: "",
    }
}

fn node<'a>(op_type: &'a str, inputs: &[&'a str], output: &'a str) -> NodeProto<'a> {
    NodeProto {
        input: inputs.to_vec(),
        output: vec![output],
        name: output,
        op_type,
        ..NodeProto::default()
    }
}

fn softmax<'a>(input: &'a str, output: &'a str) -> NodeProto<'a> {
    NodeProto {
        attribute: vec![AttributeProto {
            name: "axis",
            type_raw: 2,
            i: 1,
            ..AttributeProto::default()
        }],
        ..node("Softmax", &[input], output)
    }
}

fn weights<'a>(name: &'a str, dims: &[i64], raw: &'a [u8]) -> TensorProto<'a> {
    TensorProto {
        dims: dims.to_vec(),
        data_type: 1,
        name,
        raw_data: Some(raw),
        ..TensorProto::default()
    }
}

fn little_endian(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn trained_like(count: usize, layer: usize) -> Vec<f32> {
    (0..count)
        .map(|index| ((index * 7 + layer * 13) % 29) as f32 / 29.0 - 0.5)
        .collect()
}

fn layer_names(layers: usize) -> Vec<[String; 4]> {
    (0..layers)
        .map(|layer| {
            [
                format!("W{layer}"),
                format!("b{layer}"),
                format!("h{layer}"),
                format!("r{layer}"),
            ]
        })
        .collect()
}

fn chain_model_bytes(layers: usize) -> Vec<u8> {
    let names = layer_names(layers);
    let matrices: Vec<Vec<u8>> = (0..layers)
        .map(|layer| little_endian(&trained_like((WIDTH * WIDTH) as usize, layer)))
        .collect();
    let biases: Vec<Vec<u8>> = (0..layers)
        .map(|layer| little_endian(&trained_like(WIDTH as usize, layer + 1)))
        .collect();
    let mut nodes = Vec::new();
    let mut initializer = Vec::new();
    for (layer, [weight, bias, hidden, relu]) in names.iter().enumerate() {
        let feed = if layer == 0 {
            "x"
        } else {
            names[layer - 1][3].as_str()
        };
        nodes.push(node("Gemm", &[feed, weight, bias], hidden));
        if layer + 1 < layers {
            nodes.push(node("Relu", &[hidden], relu));
        }
        initializer.push(weights(weight, &[WIDTH, WIDTH], &matrices[layer]));
        initializer.push(weights(bias, &[WIDTH], &biases[layer]));
    }
    nodes.push(softmax(names[layers - 1][2].as_str(), OUTPUT));
    let graph = GraphProto {
        node: nodes,
        name: "chain",
        initializer,
        input: vec![float_input("x")],
        output: vec![ValueInfoProto {
            name: OUTPUT,
            ..ValueInfoProto::default()
        }],
        ..GraphProto::default()
    };
    write_model_proto(&ModelProto {
        ir_version: 8,
        graph: Some(graph),
        ..ModelProto::default()
    })
}

fn program(layers: usize) -> Program {
    let bytes = chain_model_bytes(layers);
    let model = parse_complete(&bytes).expect("the chain bytes parse");
    let graph = model.graph.as_ref().expect("the chain has a graph");
    let lowered = lower_graph(graph).expect("the chain lowers");
    let shapes = infer(&lowered.program, &[]).expect("the chain has shapes");
    let encoded = lowered
        .initializers
        .iter()
        .map(|(name, values)| (name.clone(), little_endian(values)))
        .collect();
    Program {
        ops: lowered.program,
        shapes,
        encoded,
    }
}

fn package_allocations(program: &Program) -> u64 {
    let views = program.views();
    let request = MilRequest {
        program: &program.ops,
        shapes: &program.shapes,
        weights: &views,
        output: OUTPUT,
    };
    let (package, count) = allocations_during(|| MilWriter::new().write(&request));
    package.expect("the writer emits the chain");
    count
}

fn embedding_lookup_program() -> Program {
    let table: Vec<u8> = (0..15_u8)
        .flat_map(|value| f32::from(value).to_le_bytes())
        .collect();
    let rows = Op::Input {
        dtype: DType::Int32,
        shape: vec![Extent::Static(4)],
        name: Some("rows".to_owned()),
    };
    let weight = Op::Input {
        dtype: DType::Float32,
        shape: vec![Extent::Static(5), Extent::Static(3)],
        name: Some("table".to_owned()),
    };
    let mut ops = vec![weight, rows];
    append(
        &mut ops,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Identity,
            operands: vec![(
                NodeId(0),
                IndexMap::Computed {
                    indices: NodeId(1),
                    index_map: projection(2, &[0]),
                    base: affine(2, &[(&[], 0), (&[AxisTerm::projection(1)], 0)]),
                    gathered_dim: 0,
                },
            )],
            name: None,
        },
    );
    let shapes = infer(&ops, &[]).expect("the lookup has shapes");
    Program {
        ops,
        shapes,
        encoded: vec![("table".to_owned(), table)],
    }
}

fn emitted_in_loop(program: &Program) -> (usize, u64) {
    let views = program.views();
    let request = MilRequest {
        program: &program.ops,
        shapes: &program.shapes,
        weights: &views,
        output: OUTPUT,
    };
    let measured = measure(&request).expect("the program measures");
    let mut model = vec![0_u8; measured.model];
    let mut blob_bytes = vec![0_u8; measured.weights];
    let context = Context {
        program: &program.ops,
        shapes: &program.shapes,
        weights: &views,
    };
    allocations_during(|| {
        (0..ITERATIONS)
            .filter(|iteration| {
                let node = NodeId((iteration % program.ops.len()) as u32);
                let mut encoder = Encoder::new(&mut model);
                let mut blob = BlobWriter::new(&mut blob_bytes);
                emit_node(&context, node, &mut encoder, &mut blob).is_ok()
            })
            .count()
    })
}

#[test]
fn mil_alloc_emit_node_100k_iterations_zero_allocations() {
    let mlp = program(2);
    let lookup = embedding_lookup_program();

    let (mlp_emitted, mlp_allocations) = emitted_in_loop(&mlp);
    let (lookup_emitted, lookup_allocations) = emitted_in_loop(&lookup);

    assert_eq!((mlp_emitted, lookup_emitted), (ITERATIONS, ITERATIONS));
    assert_eq!((mlp_allocations, lookup_allocations), (0, 0));
}

#[test]
fn mil_alloc_write_weight_100k_iterations_zero_allocations() {
    let source = little_endian(&trained_like(256, 0));
    let mut blob_bytes = vec![0_u8; 64 + entry_bytes(256)];

    let (written, allocations) = allocations_during(|| {
        (0..ITERATIONS)
            .filter(|_| {
                let mut blob = BlobWriter::new(&mut blob_bytes);
                blob.write_weight(&source, SourceDType::Float32).is_ok() && blob.finish().is_ok()
            })
            .count()
    });

    assert_eq!(source.len(), 1024);
    assert_eq!(written, ITERATIONS);
    assert_eq!(allocations, 0);
}

#[test]
fn mil_alloc_package_assembly_allocates_exactly_three_buffers_for_a_two_layer_mlp() {
    let small = program(2);

    let allocations = package_allocations(&small);

    assert_eq!(allocations, 3);
}

#[test]
fn mil_alloc_package_assembly_allocates_exactly_three_buffers_for_a_program_100_times_larger() {
    let small = program(2);
    let large = program(LARGE_LAYERS);

    let allocations = package_allocations(&large);

    assert!(large.ops.len() >= 100 * small.ops.len());
    assert_eq!(allocations, 3);
}

#[test]
fn mil_alloc_counter_control_a_vec_push_loop_reports_more_than_zero() {
    let (_, allocations) = allocations_during(|| {
        let mut grown: Vec<u64> = Vec::new();
        (0..ITERATIONS as u64).for_each(|value| grown.push(value));
        grown.len()
    });

    assert_ne!(allocations, 0);
}
