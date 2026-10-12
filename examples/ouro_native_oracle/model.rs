use std::io;

use crate::checkpoint::Checkpoint;

use super::math::{self, Mode};

pub struct LayerWeights {
    scales: [Vec<f32>; 4],
    matrices: [Vec<f32>; 7],
}

impl LayerWeights {
    pub fn read(checkpoint: &Checkpoint, layer: usize) -> io::Result<Self> {
        let prefix = format!("model.layers.{layer}.");
        let mut scales = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for (index, name) in [
            "input_layernorm.weight",
            "input_layernorm_2.weight",
            "post_attention_layernorm.weight",
            "post_attention_layernorm_2.weight",
        ]
        .iter()
        .enumerate()
        {
            scales[index] = math::decode(checkpoint.bytes(&format!("{prefix}{name}"))?)?;
        }
        let mut matrices = [
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ];
        for (index, name) in [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ]
        .iter()
        .enumerate()
        {
            matrices[index] = math::decode(checkpoint.bytes(&format!("{prefix}{name}"))?)?;
        }
        Ok(Self { scales, matrices })
    }
}

pub struct Model {
    pub positions: usize,
    pub hidden_width: usize,
    intermediate_width: usize,
    heads: usize,
    head_width: usize,
    epsilon: f32,
    mode: Mode,
    cosine: Vec<f32>,
    sine: Vec<f32>,
    final_scale: Vec<f32>,
}

impl Model {
    pub fn prepare(checkpoint: &Checkpoint, positions: usize, mode: Mode) -> io::Result<Self> {
        let hidden_width = checkpoint.shape.hidden as usize;
        let head_width = checkpoint.shape.head_width as usize;
        let (cosine, sine) =
            math::rope_tables(positions, head_width, checkpoint.hf_config.rope_theta, mode)?;
        Ok(Self {
            positions,
            hidden_width,
            intermediate_width: checkpoint.shape.intermediate as usize,
            heads: checkpoint.shape.heads as usize,
            head_width,
            epsilon: checkpoint.hf_config.rms_norm_eps,
            mode,
            cosine,
            sine,
            final_scale: math::decode(checkpoint.bytes("model.norm.weight")?)?,
        })
    }

    pub fn embedding(&self, checkpoint: &Checkpoint, prefix: &[u32]) -> io::Result<Vec<f32>> {
        if prefix.len() != self.positions || prefix.is_empty() {
            return Err(io::Error::other("embedding prefix dimensions differ"));
        }
        let bytes = checkpoint.bytes("model.embed_tokens.weight")?;
        let row_bytes = self.hidden_width * 2;
        let mut output = Vec::with_capacity(self.positions * self.hidden_width);
        for token in prefix {
            if *token >= checkpoint.shape.vocab {
                return Err(io::Error::other("oracle prefix token exceeds vocabulary"));
            }
            let start = *token as usize * row_bytes;
            output.extend(math::decode(
                bytes
                    .get(start..start + row_bytes)
                    .ok_or_else(|| io::Error::other("embedding row range absent"))?,
            )?);
        }
        Ok(output)
    }

    pub fn layer(&self, hidden: &[f32], weights: &LayerWeights) -> io::Result<Vec<f32>> {
        let normalized = math::norm(hidden, &weights.scales[0], self.epsilon, self.mode)?;
        let query = math::linear(
            &normalized,
            &weights.matrices[0],
            self.positions,
            self.hidden_width,
            self.hidden_width,
            self.mode,
        )?;
        let key = math::linear(
            &normalized,
            &weights.matrices[1],
            self.positions,
            self.hidden_width,
            self.hidden_width,
            self.mode,
        )?;
        let value = math::linear(
            &normalized,
            &weights.matrices[2],
            self.positions,
            self.hidden_width,
            self.hidden_width,
            self.mode,
        )?;
        let query = math::rope(
            &query,
            &self.cosine,
            &self.sine,
            self.positions,
            self.heads,
            self.head_width,
            self.mode,
        )?;
        let key = math::rope(
            &key,
            &self.cosine,
            &self.sine,
            self.positions,
            self.heads,
            self.head_width,
            self.mode,
        )?;
        let attended = math::attention(
            &query,
            &key,
            &value,
            self.positions,
            self.heads,
            self.head_width,
            self.mode,
        )?;
        let projected = math::linear(
            &attended,
            &weights.matrices[3],
            self.positions,
            self.hidden_width,
            self.hidden_width,
            self.mode,
        )?;
        let normalized = math::norm(&projected, &weights.scales[1], self.epsilon, self.mode)?;
        let attention_hidden = math::residual(hidden, &normalized, self.mode)?;
        let normalized = math::norm(
            &attention_hidden,
            &weights.scales[2],
            self.epsilon,
            self.mode,
        )?;
        let gate = math::linear(
            &normalized,
            &weights.matrices[4],
            self.positions,
            self.hidden_width,
            self.intermediate_width,
            self.mode,
        )?;
        let up_projection = math::linear(
            &normalized,
            &weights.matrices[5],
            self.positions,
            self.hidden_width,
            self.intermediate_width,
            self.mode,
        )?;
        let gated = math::gated_silu(&gate, &up_projection, self.mode)?;
        let down = math::linear(
            &gated,
            &weights.matrices[6],
            self.positions,
            self.intermediate_width,
            self.hidden_width,
            self.mode,
        )?;
        let normalized = math::norm(&down, &weights.scales[3], self.epsilon, self.mode)?;
        math::residual(&attention_hidden, &normalized, self.mode)
    }

    pub fn normalize_loop(&self, hidden: &[f32]) -> io::Result<Vec<f32>> {
        math::norm(hidden, &self.final_scale, self.epsilon, self.mode)
    }

    pub fn readout(&self, checkpoint: &Checkpoint, hidden: &[f32]) -> io::Result<Vec<f32>> {
        let start = (self.positions - 1) * self.hidden_width;
        let final_position = hidden
            .get(start..)
            .ok_or_else(|| io::Error::other("readout position absent"))?;
        self.readout_last(checkpoint, final_position)
    }

    pub fn normalize_hidden(&self, hidden: &[f32]) -> io::Result<Vec<f32>> {
        math::norm(hidden, &self.final_scale, self.epsilon, self.mode)
    }

    pub fn readout_last(&self, checkpoint: &Checkpoint, hidden: &[f32]) -> io::Result<Vec<f32>> {
        let weights = math::decode(checkpoint.bytes("lm_head.weight")?)?;
        math::linear(
            hidden,
            &weights,
            1,
            self.hidden_width,
            checkpoint.shape.vocab as usize,
            self.mode,
        )
    }
}
