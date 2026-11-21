//! First tensor on omega: load a `RuntimeConfig` from TOML, plan one
//! elementwise add, execute it, print the result.
//!
//! ```text
//! cargo run -p proxima-omega --example omega_hello_tensor
//! ```

use anyhow::Result;
use omega::backend::{execute_plan_named, plan_named_with_config};
use omega::config::RuntimeConfig;
use proxima_numeric::DType;
use proxima_tensor::map::{self, IndexMap};
use proxima_tensor::{Extent, NodeId, Op, QuantizedBlock, ScalarOp};

fn main() -> Result<()> {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/examples/omega_hello_tensor.toml").into()
    });
    let config = RuntimeConfig::from_file(&path)?;

    let input = |name: &str| Op::Input {
        dtype: DType::FLOAT32,
        shape: vec![Extent::Static(4)],
        name: Some(name.into()),
    };
    let identity = || IndexMap::Affine(map::projection(1, &[0]));
    let program = vec![
        input("a"),
        input("b"),
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            operands: vec![(NodeId(0), identity()), (NodeId(1), identity())],
            name: None,
        },
    ];
    let left = [1.0_f32, 2.0, 3.0, 4.0];
    let right = [10.0_f32, 20.0, 30.0, 40.0];
    let named = [
        ("a", QuantizedBlock::Float32(&left)),
        ("b", QuantizedBlock::Float32(&right)),
    ];

    let (mut plan, report) = plan_named_with_config(&config, &program, &[], &named, &[NodeId(2)])?;
    let result = execute_plan_named(&mut plan, &named)?;

    println!("backend = {} (config: {path})", report[0].device_identity);
    println!("shape   = {:?}", result.shape());
    println!("a + b   = {:?}", result.root());
    Ok(())
}
