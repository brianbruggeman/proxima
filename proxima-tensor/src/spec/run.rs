use alloc::string::{String, ToString};
use alloc::vec::Vec;

use conflaguration::Validate;

use super::program_spec::{NodeSpec, ProgramSpec};
use crate::cpu;
use crate::error::TensorError;
use crate::op::{NodeId, Op};

type NamedTensor = (String, Vec<u64>, Vec<f32>);

impl ProgramSpec {
    /// Evaluates the plan on the cpu. Each `[[bind]]` takes its tensor from
    /// `inputs` by the bind's `name`; each `[[output]]` returns
    /// `(name, shape, values)` in declaration order.
    pub fn evaluate_cpu(&self, inputs: &[(&str, &[f32])]) -> Result<Vec<NamedTensor>, TensorError> {
        self.validate()
            .map_err(|error| TensorError::InvalidPlan(error.to_string()))?;
        let program = Vec::<Op>::try_from(self)?;
        let named = self.bound_inputs(inputs)?;
        let outputs = self.output_nodes()?;
        let evaluated = cpu::evaluate_named(&program, &[], &named, &outputs)?;
        self.outputs
            .iter()
            .zip(&outputs)
            .map(|(output, node)| {
                let (values, shape) = evaluated
                    .get(*node)
                    .ok_or(TensorError::UnknownOutput(*node))?;
                Ok((output.name.clone(), shape.to_vec(), values.to_vec()))
            })
            .collect()
    }

    fn bound_inputs<'a>(
        &'a self,
        inputs: &[(&str, &'a [f32])],
    ) -> Result<Vec<(&'a str, &'a [f32])>, TensorError> {
        self.binds
            .iter()
            .map(|bind| {
                let name = self.input_name(&bind.node)?;
                let values = inputs
                    .iter()
                    .find(|(key, _)| *key == bind.name)
                    .map(|(_, values)| *values)
                    .ok_or_else(|| TensorError::UnboundInputName(bind.name.clone()))?;
                Ok((name, values))
            })
            .collect()
    }

    fn output_nodes(&self) -> Result<Vec<NodeId>, TensorError> {
        self.outputs
            .iter()
            .map(|output| self.node_id(&output.node))
            .collect()
    }

    fn input_name(&self, id: &str) -> Result<&str, TensorError> {
        match self.node.get(self.node_position(id)?) {
            Some(NodeSpec::Input {
                name: Some(name), ..
            }) => Ok(name),
            _ => Err(TensorError::UnnamedInput(self.node_id(id)?)),
        }
    }

    fn node_id(&self, id: &str) -> Result<NodeId, TensorError> {
        let position = self.node_position(id)?;
        u32::try_from(position)
            .map(NodeId)
            .map_err(|_| TensorError::UnknownNode(id.to_string()))
    }

    fn node_position(&self, id: &str) -> Result<usize, TensorError> {
        self.node
            .iter()
            .position(|entry| entry.id() == id)
            .ok_or_else(|| TensorError::UnknownNode(id.to_string()))
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::test_support::Lcg;

    const MATMUL_SILU_SOFTMAX_TOML: &str = include_str!("fixtures/matmul_silu_softmax.toml");

    fn matmul_silu_softmax_spec() -> ProgramSpec {
        toml::from_str(MATMUL_SILU_SOFTMAX_TOML).expect("matmul-silu-softmax fixture parses")
    }

    fn matmul_inputs() -> (Vec<f32>, Vec<f32>) {
        let mut lcg = Lcg(0x5eed_0003);
        let x: Vec<f32> = (0..8).map(|_| lcg.next_unit() - 0.5).collect();
        let w: Vec<f32> = (0..12).map(|_| lcg.next_unit() - 0.5).collect();
        (x, w)
    }

    #[test]
    fn evaluate_cpu_matmul_silu_softmax_names_the_softmax_output() {
        let spec = matmul_silu_softmax_spec();
        let (x, w) = matmul_inputs();
        let outputs = spec
            .evaluate_cpu(&[("x", &x), ("w", &w)])
            .expect("matmul-silu-softmax plan evaluates through evaluate_cpu");
        assert_eq!(outputs.len(), 1, "the fixture declares one output");
        let (name, shape, values) = outputs.first().expect("one output was checked above");

        let program = Vec::<Op>::try_from(&spec).expect("fixture lowers");
        let root = NodeId(13);
        let direct = cpu::evaluate_named(&program, &[], &[("x", &x), ("w", &w)], &[root])
            .expect("direct evaluation succeeds");
        let (direct_values, direct_shape) = direct.get(root).expect("direct output present");

        assert_eq!(name, "softmax", "the output is keyed by its plan name");
        assert_eq!(shape.as_slice(), &[2_u64, 3], "output is the 2x3 product");
        assert_eq!(shape.as_slice(), direct_shape, "shape matches direct run");
        let run_bits: Vec<u32> = values.iter().map(|value| value.to_bits()).collect();
        let direct_bits: Vec<u32> = direct_values.iter().map(|value| value.to_bits()).collect();
        assert_eq!(run_bits.len(), 6, "six output elements are compared");
        assert_eq!(
            run_bits, direct_bits,
            "evaluate_cpu is bit-identical to direct"
        );
    }

    #[test]
    fn evaluate_cpu_rejects_missing_bind_input_by_name() {
        let spec = matmul_silu_softmax_spec();
        let (x, _) = matmul_inputs();
        let error = spec
            .evaluate_cpu(&[("x", &x)])
            .expect_err("w is bound but not supplied");
        assert_eq!(error, TensorError::UnboundInputName("w".to_string()));
    }

    #[test]
    fn evaluate_cpu_rejects_version_two_plan() {
        let text = MATMUL_SILU_SOFTMAX_TOML.replacen("version = 1", "version = 2", 1);
        let spec: ProgramSpec = toml::from_str(&text).expect("version 2 still parses");
        let (x, w) = matmul_inputs();
        let error = spec
            .evaluate_cpu(&[("x", &x), ("w", &w)])
            .expect_err("version 2 is not a plan this build reads");
        assert!(
            matches!(&error, TensorError::InvalidPlan(message) if message.contains("unsupported plan version 2")),
            "error names the rejected version: {error}"
        );
    }

    #[test]
    fn evaluate_cpu_rejects_output_naming_undefined_node() {
        let text = MATMUL_SILU_SOFTMAX_TOML.replacen("node = \"n13\"", "node = \"n99\"", 1);
        let spec: ProgramSpec = toml::from_str(&text).expect("undefined output still parses");
        let (x, w) = matmul_inputs();
        let error = spec
            .evaluate_cpu(&[("x", &x), ("w", &w)])
            .expect_err("the output names a node the plan never defines");
        assert!(
            matches!(&error, TensorError::InvalidPlan(message) if message.contains("output node is not defined")),
            "error names the undefined output node: {error}"
        );
    }
}
