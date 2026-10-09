#![cfg(all(feature = "metal", target_os = "macos"))]

use omega::emit;
use proxima_primitives::Codec as PrimitiveCodec;
use proxima_tensor::cpu::{QuantizedBlock, evaluate_quantized};
use proxima_tensor::spec::{gathered_expert_product, input_leaf};
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, Reduce, ReduceInit, ScalarOp, append,
    bind, infer, map,
};
use std::collections::{BTreeMap, BTreeSet};

const BF8_BYTES: [u8; 12] = [
    0x3c, 0x38, 0xbc, 0x40, 0x3e, 0xb8, 0xc0, 0x3e, 0x38, 0xb8, 0x40, 0xbe,
];
const BF4_BYTES: [u8; 6] = [0x12, 0x4a, 0x93, 0x3c, 0x91, 0xb4];
const ROUTE: [f32; 1] = [1.0];
const ACTIVATION: [f32; 3] = [2.0, -1.0, 2.0];
const EXPECTED: [f32; 2] = [-4.5, -6.0];

fn computed_gather_program() -> (Vec<Op>, NodeId, NodeId) {
    let mut program = Vec::new();
    let stack = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(2), Extent::Static(3), Extent::Static(2)],
        "expert_stack",
    );
    let route = input_leaf(&mut program, DType::Int32, vec![Extent::Static(1)], "route");
    let activation = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(1), Extent::Static(3)],
        "activation",
    );
    let product = gathered_expert_product(&mut program, stack, route, activation);
    let output = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 2])),
            keep: Keep::Reduce,
            name: Some("packed_brain_float_gather".into()),
        }),
    );
    (program, stack, output)
}

fn decoded_logical_weights(codec: PrimitiveCodec, bytes: &[u8]) -> Vec<f32> {
    let native_values = match codec {
        PrimitiveCodec::Bf8E5M2 => bytes
            .iter()
            .copied()
            .map(proxima_gguf::quant::bf8_e5m2::decode)
            .collect::<Vec<_>>(),
        PrimitiveCodec::Bf4E2M1 => bytes
            .iter()
            .flat_map(|byte| proxima_gguf::quant::bf4_e2m1::unpack_pair(*byte))
            .collect::<Vec<_>>(),
        _ => panic!("fixture only accepts BF8 and BF4 codecs"),
    };
    let mut logical_values = Vec::with_capacity(native_values.len());
    for expert in 0..2 {
        let expert_base = expert * 6;
        for input_feature in 0..3 {
            for output_feature in 0..2 {
                logical_values
                    .push(native_values[expert_base + output_feature * 3 + input_feature]);
            }
        }
    }
    logical_values
}

fn cpu_reference(codec: PrimitiveCodec, bytes: &[u8]) -> Vec<f32> {
    let (program, _, output) = computed_gather_program();
    let decoded_weights = decoded_logical_weights(codec, bytes);
    let blocks = [
        QuantizedBlock::Float32(&decoded_weights),
        QuantizedBlock::Float32(&ROUTE),
        QuantizedBlock::Float32(&ACTIVATION),
    ];
    evaluate_quantized(&program, &[], &blocks, &[output])
        .expect("CPU computes the gathered dense oracle from the packed bytes")
        .get(output)
        .expect("CPU output exists")
        .0
        .to_vec()
}

fn assert_direct_packed_kernel(codec: PrimitiveCodec, program: &[Op], output: NodeId) -> String {
    let shapes = infer(program, &[]).expect("MoE fixture infers");
    let mut bound = bind(program, &shapes, &[output], NumericPolicy::default())
        .expect("MoE fixture binds")
        .into_iter()
        .find(|operation| operation.node == output)
        .expect("root reduce binds");
    let packed_nodes = BTreeSet::from([NodeId(0)]);
    proxima_tensor::correct_packed_matmul_layouts(core::slice::from_mut(&mut bound), &packed_nodes);
    let packed_layout = &bound
        .operands()
        .iter()
        .find(|(node, _, _)| *node == NodeId(0))
        .expect("packed weight remains a reduce operand")
        .1;
    assert_eq!(
        packed_layout.strides.as_ref(),
        &[0_i64, 1, 3],
        "native [out,in] storage maps p = expert_base + output * in_dim + input"
    );
    let packed_operands = BTreeMap::from([(NodeId(0), codec)]);
    let kernel = emit(&bound, &packed_operands, NumericPolicy::default())
        .expect("packed gather kernel emits");
    assert!(
        kernel.source.contains("device const uchar* in0"),
        "kernel must bind the original packed bytes directly"
    );
    assert!(
        !kernel.source.contains("device const float* in0"),
        "packed weights must not be widened into an FP32 input buffer"
    );
    let decoder = match codec {
        PrimitiveCodec::Bf8E5M2 => "bf8_e5m2_element(in0 +",
        PrimitiveCodec::Bf4E2M1 => "bf4_e2m1_element(in0 +",
        _ => panic!("fixture only accepts BF8 and BF4 codecs"),
    };
    assert!(
        kernel.source.contains(decoder),
        "kernel must decode from the packed input at its computed offset"
    );
    kernel.source
}

#[test]
fn packed_computed_gather_matches_cpu() {
    for (codec, bytes) in [
        (PrimitiveCodec::Bf8E5M2, BF8_BYTES.as_slice()),
        (PrimitiveCodec::Bf4E2M1, BF4_BYTES.as_slice()),
    ] {
        let (program, _, output) = computed_gather_program();
        let kernel_source = assert_direct_packed_kernel(codec, &program, output);
        let cpu = cpu_reference(codec, bytes);
        let metal = omega::execute(
            &program,
            &[],
            &[
                QuantizedBlock::Packed { codec, bytes },
                QuantizedBlock::Float32(&ROUTE),
                QuantizedBlock::Float32(&ACTIVATION),
            ],
            &[output],
            NumericPolicy::default(),
        )
        .expect("Metal executes the packed computed gather on a real device")
        .get(output)
        .expect("Metal output exists")
        .0
        .to_vec();
        assert_eq!(
            cpu, EXPECTED,
            "CPU fixture mismatch: codec={codec:?}, full_bytes={bytes:?}, route={ROUTE:?}"
        );
        assert_eq!(
            metal, cpu,
            "Metal fixture mismatch: codec={codec:?}, full_bytes={bytes:?}, route={ROUTE:?}, cpu={cpu:?}, metal={metal:?}, emitted_kernel={kernel_source}"
        );
    }
}
