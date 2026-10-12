use std::io;

use omega::backend::{self, Engine, GpuDriver, Plan};
use omega::config::RuntimeConfig;
use proxima_tensor::{NodeId, NumericPolicy, Op, QuantizedBlock, ScalarOp};
use serde_json::{Value, json};

use super::graph::Graph;

type NamedTaps = Vec<(&'static str, Vec<f32>)>;

pub struct Executor {
    pub graph: Graph,
    pub plan: Plan,
    pub planning_records: Vec<Value>,
}

impl Executor {
    pub fn prepare(
        graph: Graph,
        backend_name: &str,
        positions: usize,
        bindings: &[(&str, QuantizedBlock<'_>)],
        diagnostic: bool,
    ) -> io::Result<Self> {
        Self::prepare_with_config(graph, backend_name, positions, bindings, diagnostic, None)
    }

    pub fn prepare_with_config(
        graph: Graph,
        backend_name: &str,
        positions: usize,
        bindings: &[(&str, QuantizedBlock<'_>)],
        diagnostic: bool,
        config: Option<&RuntimeConfig>,
    ) -> io::Result<Self> {
        let mut outputs = vec![graph.root];
        if diagnostic {
            outputs.extend(graph.taps.iter().map(|(_, node)| *node));
        }
        for (index, operation) in graph.program.iter().enumerate() {
            let Op::Reduce(reduction) = operation else {
                continue;
            };
            let source = graph.program.get(reduction.operand.0 as usize);
            let statistic = reduction.body == ScalarOp::Maximum
                || matches!(
                    source,
                    Some(Op::Elementwise {
                        body: ScalarOp::Exponential,
                        ..
                    })
                )
                || matches!(source,Some(Op::Elementwise { body:ScalarOp::Multiply,operands,.. }) if operands.len() == 2 && operands[0].0 == operands[1].0);
            if statistic {
                let node = NodeId(index as u32);
                outputs.push(node);
            }
        }
        let engine = if backend_name == "cpu" {
            Engine::Cpu
        } else {
            Engine::Gpu
        };
        let planned = match config {
            Some(config) => backend::plan_named_with_config(
                config,
                &graph.program,
                &[positions as u64],
                bindings,
                &outputs,
            ),
            None => backend::plan_named_exact(
                engine,
                Some(GpuDriver::Metal),
                &graph.program,
                &[positions as u64],
                bindings,
                &outputs,
                NumericPolicy::bit_exact(),
            )
            .map(|plan| (plan, Vec::new())),
        };
        let (plan, measurements) = planned.map_err(|error| {
            io::Error::other(format!(
                "native planning failed: {error}; requested_outputs={outputs:?}; program={:?}",
                graph.program
            ))
        })?;
        let planning_records = measurements
            .iter()
            .map(|measurement| {
                json!({
                    "engine": measurement.engine.name(),
                    "driver": measurement.driver.map(GpuDriver::name),
                    "device_identity": measurement.device_identity,
                    "cold_plan_ns": measurement.cold_plan_ns,
                    "warmup_ns": measurement.warmup_ns,
                    "execution_ns": measurement.execution_ns,
                    "execution_sum_ns": measurement.execution_sum_ns,
                    "selected": measurement.selected,
                    "rejection": measurement.rejection,
                    "host_tensor_reserved_bytes": measurement.host_tensor_reserved_bytes,
                    "device_buffer_reserved_bytes": measurement.device_buffer_reserved_bytes,
                    "largest_device_buffer_bytes": measurement.largest_device_buffer_bytes,
                    "resolved_op_count": measurement.resolved_op_count,
                    "kernel_identities": measurement.kernel_identities,
                })
            })
            .collect();
        Ok(Self {
            graph,
            plan,
            planning_records,
        })
    }

    pub fn run(
        &mut self,
        bindings: &[(&str, QuantizedBlock<'_>)],
        diagnostic: bool,
    ) -> io::Result<(Vec<f32>, NamedTaps)> {
        let evaluated =
            backend::execute_plan_named(&mut self.plan, bindings).map_err(io::Error::other)?;
        let output = evaluated
            .get(self.graph.root)
            .ok_or_else(|| io::Error::other("requested graph root absent"))?
            .0
            .to_vec();
        if output.iter().any(|value| !value.is_finite()) {
            return Err(io::Error::other("graph root contains nonfinite values"));
        }
        let mut taps = Vec::new();
        if diagnostic {
            for (name, node) in &self.graph.taps {
                let values = evaluated
                    .get(*node)
                    .ok_or_else(|| io::Error::other(format!("requested tap {name} absent")))?
                    .0;
                if values.iter().any(|value| !value.is_finite()) {
                    return Err(io::Error::other(format!(
                        "tap {name} contains nonfinite values"
                    )));
                }
                taps.push((*name, values.to_vec()));
            }
        }
        Ok((output, taps))
    }
}
