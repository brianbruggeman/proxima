use std::io;

use omega::MathMode;
use omega::config::RuntimeConfig;
use proxima_model_interop::{
    LoadedModel, PromptCacheConfig, RecurrentConfig, ServingConfig, SpeculativeConfig,
    ouro_descriptor_from_hf_config,
};
use proxima_tensor::NumericPolicy;
use proxima_tensor::op::NodeId;
use proxima_tensor::spec::LoopCdMode;
use serde_json::{Value, json};

use super::checkpoint::{Checkpoint, DIGEST, REVISION};
use super::guidance::{self, Adaptive, Guidance};

#[derive(Clone, Copy)]
pub struct Arm {
    pub name: &'static str,
    pub depth: usize,
    pub guidance: Guidance,
}

impl Arm {
    pub const fn strength(self) -> Option<f32> {
        self.guidance.strength()
    }

    pub fn recurrent_config(self) -> io::Result<RecurrentConfig> {
        let iterations = u32::try_from(self.depth).map_err(io::Error::other)?;
        if iterations == 0
            || self
                .strength()
                .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(io::Error::other(
                "guidance requires positive depth and finite nonnegative strength",
            ));
        }
        let mode = match self.guidance {
            Guidance::Baseline => LoopCdMode::Disabled,
            Guidance::FixedLogits { .. } => LoopCdMode::FixedLogits,
            Guidance::AdaptiveLogits { .. } => LoopCdMode::AdaptiveLogits,
            Guidance::FixedHidden { .. } => LoopCdMode::FixedHidden,
        };
        Ok(RecurrentConfig {
            iterations: Some(iterations),
            mode,
            strength: self.strength().unwrap_or(0.0),
            reference_iteration: 1,
        })
    }
}

pub const ARMS: [Arm; 4] = [
    Arm {
        name: "baseline4",
        depth: 4,
        guidance: Guidance::Baseline,
    },
    Arm {
        name: "baseline2",
        depth: 2,
        guidance: Guidance::Baseline,
    },
    Arm {
        name: "loopcd2_zero",
        depth: 2,
        guidance: Guidance::FixedLogits { strength: 0.0 },
    },
    Arm {
        name: "loopcd2_half",
        depth: 2,
        guidance: Guidance::FixedLogits { strength: 0.5 },
    },
];
pub const PANEL_ARMS: [Arm; 6] = [
    ARMS[0],
    ARMS[1],
    Arm {
        name: "loopcd4_zero",
        depth: 4,
        guidance: Guidance::FixedLogits { strength: 0.0 },
    },
    Arm {
        name: "loopcd4_half",
        depth: 4,
        guidance: Guidance::FixedLogits { strength: 0.5 },
    },
    ARMS[2],
    ARMS[3],
];

pub struct Readouts {
    pub logits: Vec<f32>,
    pub reference: Option<Vec<f32>>,
    pub head_logits: Vec<f32>,
    pub head_executions: usize,
    pub layer_executions: usize,
    pub branch_norm_sites: usize,
    pub loop_norm_sites: usize,
    pub trace: Vec<Value>,
    pub raw_reference_hidden: Option<Vec<f32>>,
    pub raw_terminal_hidden: Option<Vec<f32>>,
    pub contrasted_hidden: Option<Vec<f32>>,
    pub normalized_hidden: Option<Vec<f32>>,
    pub adaptive: Option<Adaptive>,
    pub effective_strength: Option<f32>,
}
pub struct Output {
    pub logits: Vec<f32>,
    pub selected: u32,
    pub reference: Option<Vec<f32>>,
    pub terminal: Vec<f32>,
    pub head_executions: usize,
    pub layer_executions: usize,
    pub branch_norm_sites: usize,
    pub loop_norm_sites: usize,
    pub trace: Vec<Value>,
}
pub struct Model<'checkpoint> {
    pub checkpoint: &'checkpoint Checkpoint,
    pub backend: String,
    pub prefix: Vec<u32>,
    runtime_config: Option<RuntimeConfig>,
    models: Vec<(RecurrentConfig, LoadedModel<'checkpoint>)>,
}

pub fn pick(values: &[f32]) -> io::Result<u32> {
    let mut chosen = None;
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(io::Error::other("logits contain nonfinite values"));
        }
        if chosen.is_none_or(|(_, maximum)| value > maximum) {
            chosen = Some((index, value));
        }
    }
    u32::try_from(chosen.ok_or_else(|| io::Error::other("empty logits"))?.0)
        .map_err(io::Error::other)
}
fn extracted(
    node: Option<NodeId>,
    requested: &[NodeId],
    values: &[Vec<f32>],
) -> io::Result<Option<Vec<f32>>> {
    node.map(|node| {
        requested
            .iter()
            .position(|candidate| *candidate == node)
            .and_then(|index| values.get(index))
            .cloned()
            .ok_or_else(|| io::Error::other(format!("requested readout {} absent", node.0)))
    })
    .transpose()
}
fn last_row(values: Option<Vec<f32>>, width: usize) -> io::Result<Option<Vec<f32>>> {
    values
        .map(|values| {
            if width == 0 || values.is_empty() || !values.len().is_multiple_of(width) {
                return Err(io::Error::other(
                    "readout width differs from model embedding",
                ));
            }
            Ok(values[values.len() - width..].to_vec())
        })
        .transpose()
}

impl<'checkpoint> Model<'checkpoint> {
    pub fn prepare(
        checkpoint: &'checkpoint Checkpoint,
        prefix: Vec<u32>,
        backend_name: &str,
    ) -> io::Result<Self> {
        Self::prepare_with_config(checkpoint, prefix, backend_name, None)
    }
    pub fn prepare_with_config(
        checkpoint: &'checkpoint Checkpoint,
        prefix: Vec<u32>,
        backend_name: &str,
        config: Option<&RuntimeConfig>,
    ) -> io::Result<Self> {
        if prefix.is_empty() || prefix.iter().any(|token| *token >= checkpoint.shape.vocab) {
            return Err(io::Error::other("prefix must contain valid token IDs"));
        }
        if !matches!(backend_name, "cpu" | "metal") {
            return Err(io::Error::other("unknown model backend"));
        }
        Ok(Self {
            checkpoint,
            backend: backend_name.to_owned(),
            prefix,
            runtime_config: config.cloned(),
            models: Vec::new(),
        })
    }
    pub fn loaded(&mut self, arm: Arm) -> io::Result<&LoadedModel<'checkpoint>> {
        let recurrent = arm.recurrent_config()?;
        let index = if let Some(index) = self
            .models
            .iter()
            .position(|(candidate, _)| *candidate == recurrent)
        {
            index
        } else {
            let (architecture, descriptor) =
                ouro_descriptor_from_hf_config(&self.checkpoint.hf_config, recurrent)
                    .map_err(io::Error::other)?;
            let loaded = if let Some((_, first)) = self.models.first() {
                first.with_forward_descriptor(&descriptor)
            } else {
                LoadedModel::load_from_safetensors_with_descriptor(
                    &self.checkpoint.manifest,
                    self.checkpoint.mapped_bytes(),
                    self.checkpoint.data_start,
                    architecture,
                    self.checkpoint.vocab.clone(),
                    &descriptor,
                )
            }
            .map_err(io::Error::other)?;
            self.models.push((recurrent, loaded));
            self.models.len() - 1
        };
        Ok(&self.models[index].1)
    }
    pub fn serving_config(&self, arm: Arm) -> io::Result<ServingConfig<'_>> {
        let numeric_policy = self
            .runtime_config
            .as_ref()
            .map_or(NumericPolicy::bit_exact(), |config| config.numeric);
        Ok(ServingConfig {
            gpu_layers: if self.backend == "metal" { -1 } else { 0 },
            numeric_policy,
            math_mode: MathMode::for_numeric_policy(numeric_policy),
            exact_activations: true,
            temperature: 0.0,
            repeat_penalty: 1.0,
            prompt_cache: PromptCacheConfig::off(),
            speculative: SpeculativeConfig::none(),
            recurrent: arm.recurrent_config()?,
            omega_runtime: self.runtime_config.as_ref(),
            ..ServingConfig::default()
        })
    }
    pub fn context(&self, arm: Arm) -> Value {
        json!({
            "model":"ByteDance/Ouro-1.4B", "checkpoint_revision":REVISION,
            "checkpoint_expected_sha256":DIGEST, "checkpoint_observed_sha256":DIGEST,
            "digest_verified_during_setup":true, "backend":self.backend,
            "checkpoint_mapping_mode":self.checkpoint.mapping_mode.label(),
            "numeric_policy":"f32 activation arithmetic; author BF16 intermediate rounding not reproduced",
            "backend_numeric_policy":self.runtime_config.as_ref().map_or(NumericPolicy::bit_exact(), |config| config.numeric),
            "omega_config":self.runtime_config, "execution_owner":"proxima_model_interop::LoadedModel",
            "build_dense_float_storage":omega::sized::DENSE_FLOAT_STORAGE_ENABLED,
            "build_tiled_min_tokens":omega::sized::DENSE_TILED_MIN_TOKENS,
            "prefix_ids":self.prefix, "positions":self.prefix.len(), "depth":arm.depth,
            "strength":arm.strength(), "method":arm.guidance.name(), "arm":arm.name,
            "use_cache":false, "forward_scope":"mainline fresh-prefix diagnostic evaluation",
            "readout":"explicit completed recurrence roots", "author_activation_oracle_status":"unmeasured",
            "execution_counts_status":"derived from requested shared graph; not observed backend dispatch counts",
            "trace_scope":"completed-depth readout vectors; no synthetic layer traces",
            "adaptive_statistics_provenance":"top probabilities and margin derived from captured terminal logits; effective strength captured from graph root",
        })
    }
    pub fn forward(&mut self, arm: Arm, diagnostic: bool) -> io::Result<Output> {
        let readouts = self.forward_internal(arm, diagnostic, diagnostic)?;
        Ok(Output {
            selected: pick(&readouts.logits)?,
            logits: readouts.logits,
            reference: readouts.reference,
            terminal: readouts.head_logits,
            head_executions: readouts.head_executions,
            layer_executions: readouts.layer_executions,
            branch_norm_sites: readouts.branch_norm_sites,
            loop_norm_sites: readouts.loop_norm_sites,
            trace: readouts.trace,
        })
    }
    pub fn forward_readouts(&mut self, arm: Arm, capture_reference: bool) -> io::Result<Readouts> {
        self.forward_internal(arm, false, capture_reference)
    }
    fn forward_internal(
        &mut self,
        arm: Arm,
        diagnostic: bool,
        capture_reference: bool,
    ) -> io::Result<Readouts> {
        let effective_arm = if capture_reference && matches!(arm.guidance, Guidance::Baseline) {
            Arm {
                guidance: Guidance::FixedLogits { strength: 0.0 },
                ..arm
            }
        } else {
            arm
        };
        self.loaded(effective_arm)?;
        let recurrent = effective_arm.recurrent_config()?;
        let loaded = &self
            .models
            .iter()
            .find(|(candidate, _)| *candidate == recurrent)
            .ok_or_else(|| io::Error::other("prepared shared model absent"))?
            .1;
        let roots = *loaded
            .recurrent_readout_roots()
            .ok_or_else(|| io::Error::other("recurrent readout roots absent"))?;
        let logits_root = loaded
            .logits_root()
            .ok_or_else(|| io::Error::other("generation root absent"))?;
        let head_root = roots
            .terminal_logits
            .ok_or_else(|| io::Error::other("terminal head absent"))?;
        let mut requested = vec![logits_root, head_root];
        let need_reference = arm.guidance.uses_logit_reference() || capture_reference;
        if need_reference {
            requested.extend(roots.reference_logits);
        }
        let hidden_mode = matches!(arm.guidance, Guidance::FixedHidden { .. });
        if hidden_mode || diagnostic {
            requested.extend(
                [
                    roots.reference_raw,
                    roots.terminal_raw,
                    roots.reference_normalized,
                    roots.terminal_normalized,
                    roots.guided_raw,
                ]
                .into_iter()
                .flatten(),
            );
        }
        requested.extend(roots.adaptive_strength);
        requested.sort_unstable_by_key(|node| node.0);
        requested.dedup();
        let config = self.serving_config(effective_arm)?;
        let values = loaded
            .forward_node_values_from_ids(&self.prefix, &requested, &config)
            .map_err(io::Error::other)?;
        let head_logits = extracted(Some(head_root), &requested, &values)?
            .ok_or_else(|| io::Error::other("head readout absent"))?;
        let logits = if matches!(arm.guidance, Guidance::Baseline) {
            head_logits.clone()
        } else {
            extracted(Some(logits_root), &requested, &values)?
                .ok_or_else(|| io::Error::other("guided readout absent"))?
        };
        let reference = if need_reference {
            extracted(roots.reference_logits, &requested, &values)?
        } else {
            None
        };
        let mut trace = Vec::new();
        if diagnostic {
            for (node, values) in requested.iter().zip(&values) {
                trace.push(json!({"kind":"mainline_readout","node":node.0,"values":values,"roots":format!("{roots:?}"),"diagnostic_only":true}));
            }
        }
        let width = self.checkpoint.shape.hidden as usize;
        let raw_reference_hidden = if hidden_mode {
            last_row(extracted(roots.reference_raw, &requested, &values)?, width)?
        } else {
            None
        };
        let raw_terminal_hidden = if hidden_mode {
            last_row(extracted(roots.terminal_raw, &requested, &values)?, width)?
        } else {
            None
        };
        let contrasted_hidden = if hidden_mode {
            last_row(extracted(roots.guided_raw, &requested, &values)?, width)?
        } else {
            None
        };
        let normalized_hidden = if hidden_mode {
            last_row(
                extracted(roots.terminal_normalized, &requested, &values)?,
                width,
            )?
        } else {
            None
        };
        let adaptive = match arm.guidance {
            Guidance::AdaptiveLogits { max_strength } => {
                let mut statistics = guidance::adaptive(&head_logits, max_strength)?;
                let measured = extracted(roots.adaptive_strength, &requested, &values)?
                    .ok_or_else(|| io::Error::other("actual adaptive strength root absent"))?;
                let [strength] = measured.as_slice() else {
                    return Err(io::Error::other(
                        "adaptive readout requires one row strength",
                    ));
                };
                statistics.effective_strength = *strength;
                Some(statistics)
            }
            _ => None,
        };
        let effective_strength = adaptive
            .map(|value| value.effective_strength)
            .or(arm.strength());
        let layer_executions = arm.depth * self.checkpoint.hf_config.num_hidden_layers as usize;
        let head_executions =
            if reference.is_some() && roots.reference_logits != roots.terminal_logits {
                2
            } else {
                1
            };
        Ok(Readouts {
            logits,
            reference,
            head_logits,
            head_executions,
            layer_executions,
            branch_norm_sites: layer_executions * 4,
            loop_norm_sites: arm.depth,
            trace,
            raw_reference_hidden,
            raw_terminal_hidden,
            contrasted_hidden,
            normalized_hidden,
            adaptive,
            effective_strength,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mainline_caller_guidance_mapping() {
        assert_eq!(
            MathMode::for_numeric_policy(NumericPolicy::bit_exact()),
            MathMode::Safe
        );
        assert_eq!(
            MathMode::for_numeric_policy(NumericPolicy::llama_relaxed()),
            MathMode::Relaxed
        );
        assert_eq!(
            MathMode::for_numeric_policy(NumericPolicy::fast()),
            MathMode::Fast
        );
        let mut partial = NumericPolicy::bit_exact();
        partial.contraction = true;
        assert_eq!(MathMode::for_numeric_policy(partial), MathMode::Safe);
        for (guidance, expected) in [
            (Guidance::Baseline, LoopCdMode::Disabled),
            (
                Guidance::FixedLogits { strength: 0.5 },
                LoopCdMode::FixedLogits,
            ),
            (
                Guidance::AdaptiveLogits { max_strength: 1.0 },
                LoopCdMode::AdaptiveLogits,
            ),
            (
                Guidance::FixedHidden { strength: 0.5 },
                LoopCdMode::FixedHidden,
            ),
        ] {
            let config = Arm {
                name: "fixture",
                depth: 4,
                guidance,
            }
            .recurrent_config()
            .expect("literal mode config");
            assert_eq!(config.mode, expected);
            assert_eq!(config.iterations, Some(4));
            assert_eq!(config.reference_iteration, 1);
        }
        assert!(
            Arm {
                depth: 0,
                ..ARMS[0]
            }
            .recurrent_config()
            .is_err()
        );
        assert!(
            Arm {
                guidance: Guidance::FixedLogits { strength: f32::NAN },
                ..ARMS[0]
            }
            .recurrent_config()
            .is_err()
        );
    }
    #[test]
    fn mainline_caller_root_extraction() {
        let requested = [NodeId(7), NodeId(3)];
        let payloads = vec![vec![3.0, 4.0], vec![1.0, 2.0, 5.0, 6.0]];
        assert_eq!(
            extracted(Some(NodeId(7)), &requested, &payloads).expect("reference root"),
            Some(vec![3.0, 4.0])
        );
        assert_eq!(
            last_row(
                extracted(Some(NodeId(3)), &requested, &payloads).expect("terminal root"),
                2
            )
            .expect("last row"),
            Some(vec![5.0, 6.0])
        );
        assert!(extracted(Some(NodeId(8)), &requested, &payloads).is_err());
        assert!(last_row(Some(vec![1.0, 2.0, 3.0]), 2).is_err());
        assert_ne!(
            payloads[0],
            vec![5.0, 6.0],
            "wrong boundary fails literal equality"
        );
        println!("requested={requested:?} payloads={payloads:?}");
    }
}
