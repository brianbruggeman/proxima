//! Host-side assembly of the generic encoder-decoder builders: read a
//! safetensors checkpoint and its config, build the mel/encoder/cross-KV/
//! decoder graphs, and run a greedy transcript. This is the std half; the
//! graph builders themselves stay alloc-only and model-neutral.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use proxima_safetensors::{Manifest, SafetensorsParser};
use proxima_tensor::cpu::{Evaluated, QuantizedBlock};
use proxima_tensor::op::{Extent, NodeId, Op};
use proxima_tensor::spec::{
    CachedLayerRoots, STFT_COS_INPUT, STFT_SIN_INPUT, STFT_WINDOW_INPUT, StftLogMelConfig,
    cross_kv, input_leaf, stft_log_mel, stft_tables,
};
use serde_json::Value;

use crate::bind::{BoundWeights, retarget_recoded_leaves};
use crate::decode_policy::{DecodeConfig, DecodePolicy};
use crate::families::encoder_decoder::{
    DecoderLayer, DecoderStem, EncoderLayer, EncoderStem, Linear, encoder,
};
use crate::generate::{BackendRuntime, cached_kv_extent, program_hash};
use crate::hf_bind::bind_safetensors_program_leaves;
use crate::profiles::BindingProfile;
use crate::serving::apply_serving_config;
use crate::serving_settings::ServingSettings;
use crate::source::CheckpointMapping;

/// The dimensions a checkpoint states, read from its `config.json`.
#[derive(Debug, Clone)]
pub struct Shape {
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub encoder_heads: u32,
    pub encoder_head_dim: u32,
    pub decoder_heads: u32,
    pub decoder_head_dim: u32,
    pub width: u32,
    pub mels: u32,
    pub encoder_ffn: u32,
    pub decoder_ffn: u32,
    pub source_positions: u32,
    pub target_positions: u32,
    pub vocab: u32,
    pub eps: f32,
}

impl Shape {
    pub fn from_config(config: &Value) -> Result<Self> {
        let int = |key: &str| -> Result<u32> {
            config
                .get(key)
                .and_then(Value::as_u64)
                .map(|value| value as u32)
                .with_context(|| format!("config.json lacks {key}"))
        };
        let encoder_layers = int("encoder_layers")? as usize;
        let decoder_layers = int("decoder_layers")? as usize;
        let encoder_heads = int("encoder_attention_heads")?;
        let decoder_heads = int("decoder_attention_heads")?;
        let width = int("d_model")?;
        let mels = int("num_mel_bins")?;
        let encoder_ffn = int("encoder_ffn_dim")?;
        let decoder_ffn = int("decoder_ffn_dim")?;
        let source_positions = int("max_source_positions")?;
        let target_positions = int("max_target_positions")?;
        let vocab = int("vocab_size")?;
        if width % encoder_heads != 0 || width % decoder_heads != 0 {
            bail!("d_model {width} is not a multiple of its head counts");
        }
        Ok(Self {
            encoder_layers,
            decoder_layers,
            encoder_heads,
            encoder_head_dim: width / encoder_heads,
            decoder_heads,
            decoder_head_dim: width / decoder_heads,
            width,
            mels,
            encoder_ffn,
            decoder_ffn,
            source_positions,
            target_positions,
            vocab,
            eps: 1e-5,
        })
    }
}

const CACHE_PLANES: [&str; 3] = ["k_even", "k_odd", "v"];

fn cache_plane_name(plane: &str, layer: usize) -> String {
    format!("kv_cache.{layer}.{plane}")
}

fn cache_names(layers: usize) -> impl Iterator<Item = String> {
    (0..layers).flat_map(|layer| CACHE_PLANES.map(|part| cache_plane_name(part, layer)))
}

/// What `Model` keeps of a checkpoint file: its mapping, which the shared
/// binder borrows weights from in place, the tensor directory, and where the
/// tensor data starts.
type Checkpoint = (CheckpointMapping, Manifest, u64);

/// Maps a safetensors file read-only through the same [`CheckpointMapping`] the
/// diffusion loader opens its checkpoints with, then parses its header.
fn read_checkpoint(path: &Path) -> Result<Checkpoint> {
    let mapping = CheckpointMapping::open_file(path).with_context(|| path.display().to_string())?;
    let (manifest, data_start) = parse_checkpoint(mapping.bytes())?;
    Ok((mapping, manifest, data_start))
}

fn parse_checkpoint(bytes: &[u8]) -> Result<(Manifest, u64)> {
    let prefix: [u8; 8] = bytes
        .get(..8)
        .and_then(|head| head.try_into().ok())
        .context("safetensors file is shorter than its length prefix")?;
    let header_len = u64::from_le_bytes(prefix) as usize;
    let header = bytes
        .get(..8 + header_len)
        .context("safetensors header runs past the end of the file")?;
    let mut parser = SafetensorsParser::new();
    parser.feed(header);
    parser
        .poll()
        .map_err(|error| anyhow::anyhow!("safetensors header does not parse: {error:?}"))?;
    let SafetensorsParser::TensorData { manifest, .. } = parser else {
        return Err(anyhow::anyhow!("safetensors header did not complete"));
    };
    Ok((manifest, (8 + header_len) as u64))
}

struct Bound<'a> {
    program: Vec<Op>,
    named: BTreeMap<String, Vec<f32>>,
    weights: BoundWeights<'a>,
    /// leaves the host feeds each run; every other named leaf is a checkpoint tensor
    runtime_inputs: Vec<String>,
    shape: Shape,
}

impl<'a> Bound<'a> {
    fn new(shape: Shape) -> Self {
        Self {
            program: Vec::new(),
            named: BTreeMap::new(),
            weights: BoundWeights::new(&[]),
            runtime_inputs: Vec::new(),
            shape,
        }
    }

    fn leaf(&mut self, name: &str, shape: &[u32]) -> NodeId {
        self.runtime_inputs.push(name.into());
        super::encoder_decoder::leaf(&mut self.program, shape, name)
    }

    fn weight(&mut self, name: &str, shape: &[u32]) -> NodeId {
        super::encoder_decoder::leaf(&mut self.program, shape, name)
    }

    /// Binds every checkpoint leaf the program names through the shared
    /// safetensors binder, then retargets the leaves a precision rule recoded.
    fn bind(&mut self, model: &'a Model) -> Result<()> {
        let rules = model.settings.weight_precision_rules()?;
        self.weights = bind_safetensors_program_leaves(
            &model.manifest,
            model.mapping.bytes(),
            model.data_start,
            &self.program,
            &BindingProfile::default(),
            &self.runtime_inputs,
            &rules,
        )?;
        retarget_recoded_leaves(&mut self.program, &self.weights);
        Ok(())
    }

    fn symbolic_leaf(&mut self, name: &str) -> NodeId {
        self.runtime_inputs.push(name.into());
        input_leaf(
            &mut self.program,
            super::encoder_decoder::ACTIVATION,
            vec![Extent::Symbolic(0)],
            name,
        )
    }

    fn cross_leaves(&mut self) -> Vec<(NodeId, NodeId)> {
        let dims = [
            self.shape.source_positions,
            self.shape.decoder_heads,
            self.shape.decoder_head_dim,
        ];
        (0..self.shape.decoder_layers)
            .map(|layer| {
                (
                    self.leaf(&format!("cross_k_{layer}"), &dims),
                    self.leaf(&format!("cross_v_{layer}"), &dims),
                )
            })
            .collect()
    }

    fn cache_leaves(&mut self) -> Vec<CachedLayerRoots> {
        let heads = self.shape.decoder_heads;
        let half = self.shape.decoder_head_dim / 2;
        let full = self.shape.decoder_head_dim;
        let plane = |this: &mut Self, name: String, width: u32| {
            this.named.insert(name.clone(), Vec::new());
            this.runtime_inputs.push(name.clone());
            let extents = vec![
                Extent::Symbolic(1),
                Extent::Static(heads),
                Extent::Static(width),
            ];
            input_leaf(
                &mut this.program,
                super::encoder_decoder::ACTIVATION,
                extents,
                &name,
            )
        };
        (0..self.shape.decoder_layers)
            .map(|layer| {
                let [first, second, value] = CACHE_PLANES.map(|part| cache_plane_name(part, layer));
                (
                    plane(self, first, half),
                    plane(self, second, half),
                    plane(self, value, full),
                )
            })
            .collect()
    }

    fn cached_len_leaf(&mut self) -> NodeId {
        self.runtime_inputs.push("cached_len".into());
        input_leaf(
            &mut self.program,
            super::encoder_decoder::ACTIVATION,
            Vec::new(),
            "cached_len",
        )
    }

    fn cache_row_widths(&self) -> [usize; 3] {
        let heads = self.shape.decoder_heads as usize;
        let half = self.shape.decoder_head_dim as usize / 2;
        let full = self.shape.decoder_head_dim as usize;
        [heads * half, heads * half, heads * full]
    }

    fn fit_cache(&mut self, extent: usize) {
        let widths = self.cache_row_widths();
        for (index, name) in cache_names(self.shape.decoder_layers).enumerate() {
            if let Some(plane) = self.named.get_mut(&name) {
                plane.resize(extent * widths[index % widths.len()], Default::default());
            }
        }
    }

    fn trim_cache(&mut self, kept: usize) {
        let widths = self.cache_row_widths();
        for (index, name) in cache_names(self.shape.decoder_layers).enumerate() {
            if let Some(plane) = self.named.get_mut(&name) {
                plane.truncate(kept * widths[index % widths.len()]);
            }
        }
    }

    fn stage_step(&mut self, ids: &[u32], cached: usize) {
        let tokens = ids.iter().map(|id| *id as _).collect();
        let positions = (cached..cached + ids.len()).map(|at| at as _).collect();
        let live = (cached..=cached).map(|at| at as _).collect();
        self.named.insert("tokens".into(), tokens);
        self.named.insert("positions".into(), positions);
        self.named.insert("cached_len".into(), live);
    }

    fn matrix(&mut self, name: &str, input: u32, output: u32) -> Linear {
        let weight = self.weight(&format!("{name}.weight"), &[output, input]);
        let bias = self.weight(&format!("{name}.bias"), &[output]);
        Linear {
            weight,
            bias: Some(bias),
        }
    }

    /// A head-major input projection over the stored `[heads*head_dim, width]`
    /// matrix, read in the stored order.
    fn heads(&mut self, name: &str, bias: bool) -> Linear {
        let (heads, head_dim) = (self.shape.encoder_heads, self.shape.encoder_head_dim);
        let weight = self.weight(
            &format!("{name}.weight"),
            &[heads, head_dim, self.shape.width],
        );
        let bias = bias.then(|| self.weight(&format!("{name}.bias"), &[heads, head_dim]));
        Linear { weight, bias }
    }

    /// The cross-attention K and V projections `cross_kv` reads `[width, heads,
    /// head_dim]`, a contraction-first order the binder reads as `InOut`.
    fn memory_heads(&mut self, name: &str, bias: bool) -> Linear {
        let (heads, head_dim) = (self.shape.encoder_heads, self.shape.encoder_head_dim);
        let weight = self.weight(
            &format!("{name}.weight"),
            &[self.shape.width, heads, head_dim],
        );
        let bias = bias.then(|| self.weight(&format!("{name}.bias"), &[heads, head_dim]));
        Linear { weight, bias }
    }

    fn heads_out(&mut self, name: &str) -> Linear {
        let (heads, head_dim) = (self.shape.encoder_heads, self.shape.encoder_head_dim);
        let weight = self.weight(
            &format!("{name}.weight"),
            &[self.shape.width, heads, head_dim],
        );
        let bias = self.weight(&format!("{name}.bias"), &[self.shape.width]);
        Linear {
            weight,
            bias: Some(bias),
        }
    }

    fn norm(&mut self, name: &str) -> Linear {
        let weight = self.weight(&format!("{name}.weight"), &[self.shape.width]);
        let bias = self.weight(&format!("{name}.bias"), &[self.shape.width]);
        Linear {
            weight,
            bias: Some(bias),
        }
    }

    fn encoder_layer(&mut self, layer: usize) -> EncoderLayer {
        let prefix = format!("model.encoder.layers.{layer}");
        EncoderLayer {
            attn_ln: self.norm(&format!("{prefix}.self_attn_layer_norm")),
            attn: super::encoder_decoder::Attention {
                q: self.heads(&format!("{prefix}.self_attn.q_proj"), true),
                k: self.heads(&format!("{prefix}.self_attn.k_proj"), false),
                v: self.heads(&format!("{prefix}.self_attn.v_proj"), true),
                out: self.heads_out(&format!("{prefix}.self_attn.out_proj")),
            },
            mlp_ln: self.norm(&format!("{prefix}.final_layer_norm")),
            fc1: self.matrix(
                &format!("{prefix}.fc1"),
                self.shape.width,
                self.shape.encoder_ffn,
            ),
            fc2: self.matrix(
                &format!("{prefix}.fc2"),
                self.shape.encoder_ffn,
                self.shape.width,
            ),
        }
    }

    fn decoder_layer(&mut self, layer: usize) -> DecoderLayer {
        let prefix = format!("model.decoder.layers.{layer}");
        DecoderLayer {
            self_ln: self.norm(&format!("{prefix}.self_attn_layer_norm")),
            self_attn: super::encoder_decoder::Attention {
                q: self.heads(&format!("{prefix}.self_attn.q_proj"), true),
                k: self.heads(&format!("{prefix}.self_attn.k_proj"), false),
                v: self.heads(&format!("{prefix}.self_attn.v_proj"), true),
                out: self.heads_out(&format!("{prefix}.self_attn.out_proj")),
            },
            cross_ln: self.norm(&format!("{prefix}.encoder_attn_layer_norm")),
            cross_q: self.heads(&format!("{prefix}.encoder_attn.q_proj"), true),
            cross_out: self.heads_out(&format!("{prefix}.encoder_attn.out_proj")),
            mlp_ln: self.norm(&format!("{prefix}.final_layer_norm")),
            fc1: self.matrix(
                &format!("{prefix}.fc1"),
                self.shape.width,
                self.shape.decoder_ffn,
            ),
            fc2: self.matrix(
                &format!("{prefix}.fc2"),
                self.shape.decoder_ffn,
                self.shape.width,
            ),
        }
    }

    fn encoder_stem(&mut self) -> EncoderStem {
        let width = self.shape.width;
        let mels = self.shape.mels;
        let conv1 = self.weight("model.encoder.conv1.weight", &[width, mels, 3]);
        let conv1_bias = self.weight("model.encoder.conv1.bias", &[width]);
        let conv2 = self.weight("model.encoder.conv2.weight", &[width, width, 3]);
        let conv2_bias = self.weight("model.encoder.conv2.bias", &[width]);
        let position = self.weight(
            "model.encoder.embed_positions.weight",
            &[self.shape.source_positions, width],
        );
        EncoderStem {
            conv1: Linear {
                weight: conv1,
                bias: Some(conv1_bias),
            },
            conv2: Linear {
                weight: conv2,
                bias: Some(conv2_bias),
            },
            position,
            layer_norm: self.norm("model.encoder.layer_norm"),
        }
    }

    /// The tied head reads the embedding leaf itself: one node, one bound buffer.
    fn decoder_stem(&mut self) -> DecoderStem {
        let tokens = self.weight(
            "model.decoder.embed_tokens.weight",
            &[self.shape.vocab, self.shape.width],
        );
        let position = self.weight(
            "model.decoder.embed_positions.weight",
            &[self.shape.target_positions, self.shape.width],
        );
        DecoderStem {
            tokens,
            position,
            layer_norm: self.norm("model.decoder.layer_norm"),
            head: Linear {
                weight: tokens,
                bias: None,
            },
        }
    }

    /// The cached one-step decoder program; returns the logits followed by the
    /// appended cache planes, the outputs every step reads.
    fn decoder_graph(&mut self) -> Result<Vec<NodeId>> {
        let token_leaf = self.symbolic_leaf("tokens");
        let positions = self.symbolic_leaf("positions");
        let stem = self.decoder_stem();
        let layers: Vec<_> = (0..self.shape.decoder_layers)
            .map(|layer| self.decoder_layer(layer))
            .collect();
        let cross_nodes = self.cross_leaves();
        let cache_leaves = self.cache_leaves();
        self.cached_len_leaf();
        let (logits, appended) = super::encoder_decoder::decoder(
            &mut self.program,
            token_leaf,
            positions,
            &stem,
            &layers,
            &cross_nodes,
            Some(&cache_leaves),
            self.shape.decoder_heads,
            self.shape.decoder_head_dim,
            self.shape.eps,
        )?;
        let mut outputs = vec![logits];
        outputs.extend(
            appended
                .iter()
                .flat_map(|(first, second, value)| [*first, *second, *value]),
        );
        Ok(outputs)
    }

    /// Runs the program and returns `outputs`; `stage` (mel, encode, decode)
    /// names the program in the span and in any error.
    #[cfg_attr(
        feature = "instrument",
        proxima_telemetry::instrument(
            level = "debug",
            fields(stage = stage, rows = symbols.first().copied().unwrap_or_default())
        )
    )]
    fn run(
        &self,
        stage: &'static str,
        outputs: &[NodeId],
        symbols: &[u64],
        runtime: &mut BackendRuntime,
    ) -> Result<Vec<Vec<f32>>> {
        let named: Vec<(&str, QuantizedBlock<'_>)> = self
            .named
            .iter()
            .map(|(name, data)| (name.as_str(), QuantizedBlock::from(data.as_slice())))
            .chain(self.weights.blocks()?)
            .collect();
        let resident: BTreeSet<&str> = self.weights.names();
        let evaluated = runtime
            .evaluate_segment(&self.program, symbols, &named, outputs, &resident, None)
            .map_err(|error| anyhow::anyhow!("{stage} segment does not evaluate: {error:?}"))?;
        collect_outputs(&evaluated, outputs)
    }
}

#[cfg_attr(feature = "instrument", proxima_telemetry::instrument(level = "debug"))]
fn collect_outputs(evaluated: &Evaluated, outputs: &[NodeId]) -> Result<Vec<Vec<f32>>> {
    outputs
        .iter()
        .map(|node| {
            evaluated
                .get(*node)
                .map(|(values, _)| values.to_vec())
                .with_context(|| format!("evaluated program has no output for node {node:?}"))
        })
        .collect()
}

/// Makes every closed `#[instrument]` span emit a `debug!` carrying its name and
/// `duration_ns`, so span durations reach the same console and `RUST_LOG` filter
/// as the rest of the telemetry. Uses the recorder's own duration observer
/// ([`proxima_telemetry::recorder::Recorder::set_duration_observer`]); no timer
/// of its own.
#[cfg(feature = "instrument-metrics")]
pub fn log_span_durations(recorder: &proxima_telemetry::recorder::Recorder) {
    recorder.set_duration_observer(|name, duration_ns| {
        proxima_telemetry::debug!(name = name, duration_ns = duration_ns, "span closed");
    });
}

/// One transcription and the values the caller can check.
pub struct Transcript {
    pub tokens: Vec<u32>,
    pub text: String,
    pub ended_on_eot: bool,
    pub mel: std::time::Duration,
    pub encode: std::time::Duration,
    pub decode: std::time::Duration,
    pub prompt_tokens: usize,
    /// transcription start to the end of the decode step that picked the first generated token
    pub first_token: Duration,
    /// transcription start to the end of the last decode step
    pub last_token: Duration,
    /// one entry per decode step, each from the step's start to its token pick
    pub step_times: Vec<Duration>,
    /// bytes the device had allocated when the transcription ended; `None` on the host
    pub gpu_allocated_bytes: Option<u64>,
    /// largest allocation sampled after each stage and each decode step; `None` on the host
    pub gpu_allocated_peak_bytes: Option<u64>,
}

fn sample_peak(peak: &mut Option<u64>, runtime: &BackendRuntime) {
    *peak = (*peak).max(runtime.allocated_bytes());
}

/// A loaded checkpoint ready to transcribe.
pub struct Model {
    /// the mapped checkpoint file; every bound weight borrows from it in place
    mapping: CheckpointMapping,
    manifest: Manifest,
    data_start: u64,
    shape: Shape,
    generation: Value,
    config: Value,
    dir: PathBuf,
    settings: ServingSettings,
}

impl Model {
    pub fn load(dir: &Path) -> Result<Self> {
        let config: Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("config.json")).context("config.json")?,
        )?;
        let generation: Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("generation_config.json"))
                .context("generation_config.json")?,
        )?;
        let shape = Shape::from_config(&config)?;
        let checkpoint = read_checkpoint(&dir.join("model.safetensors"))?;
        Ok(Self::from_parts(checkpoint, shape, generation, config, dir))
    }

    fn from_parts(
        (mapping, manifest, data_start): Checkpoint,
        shape: Shape,
        generation: Value,
        config: Value,
        dir: &Path,
    ) -> Self {
        Self {
            mapping,
            manifest,
            data_start,
            shape,
            generation,
            config,
            dir: dir.to_path_buf(),
            settings: ServingSettings::default(),
        }
    }

    /// Replaces the serving settings every stage lowers into its runtime.
    #[must_use]
    pub fn with_settings(mut self, settings: ServingSettings) -> Self {
        self.settings = settings;
        self
    }

    /// The cached-range rows a step binds, decided exactly as the llm decides
    /// them: the runtime probes the program, and pads to the capacity only where
    /// the fused cached attention's runtime bound excludes the pad rows.
    fn cache_extent(
        &self,
        runtime: &mut BackendRuntime,
        decoder: &Bound<'_>,
        outputs: &[NodeId],
        (program_hash, cached, new_count): (u64, usize, usize),
    ) -> usize {
        let capacity = self.shape.target_positions as usize;
        let probe_extent = capacity.max(1) as u64;
        runtime.settle_kv_padding(
            &decoder.program,
            program_hash,
            &[new_count as u64, probe_extent],
            || outputs.to_vec(),
        );
        cached_kv_extent(
            runtime.pads_cached_kv(program_hash, new_count),
            cached,
            new_count,
            self.settings.kv_bucket_tokens,
            capacity,
        )
        .max(cached)
    }

    #[cfg_attr(feature = "instrument", proxima_telemetry::instrument(level = "debug"))]
    fn runtime(&self) -> Result<BackendRuntime> {
        let config = self.settings.as_serving_config(&[]);
        apply_serving_config(&config, 0)
            .map_err(|error| anyhow::anyhow!("serving settings are not admitted: {error:?}"))?;
        Ok(BackendRuntime::new(&config))
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Transcribes on the device the serving settings select: `gpu_layers` of `GPU_LAYERS_ALL` runs the stages on the device.
    pub fn transcribe(&self, wav: &[f32], language: &str) -> Result<Transcript> {
        self.transcribe_with(wav, language)
    }

    pub fn transcribe_with(&self, wav: &[f32], language: &str) -> Result<Transcript> {
        let start = Instant::now();
        let policy = DecodePolicy::from_checkpoint(
            &self.generation,
            &self.config,
            &DecodeConfig {
                language: language.into(),
                ..DecodeConfig::default()
            },
        )?;

        let mut runtime = self.runtime()?;
        let mut gpu_peak = None;
        let padded = pad(wav)?;
        let front = front_end(&self.dir)?;
        if front.mels != self.shape.mels {
            bail!(
                "preprocessor_config.json says {} mels, config.json says {}",
                front.mels,
                self.shape.mels
            );
        }
        let frames = ((padded.len() as u32 - front.n_fft) / front.hop) + 1;
        let mut mel_bound = Bound::new(self.shape.clone());
        let signal = mel_bound.leaf("signal", &[padded.len() as u32]);
        let filters = front.filters;
        let bank = mel_bound.leaf("bank", &[front.mels, front.n_fft / 2 + 1]);
        let config = StftLogMelConfig {
            n_fft: front.n_fft,
            hop: front.hop,
            ..StftLogMelConfig::default()
        };
        let mel = stft_log_mel(
            &mut mel_bound.program,
            &config,
            signal,
            padded.len() as u32,
            bank,
        )?;
        let tables = stft_tables(&config, crate::families::encoder_decoder::ACTIVATION)?;
        let decode = |bytes: &[u8]| {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| f32::from_le_bytes(*word))
                .collect()
        };
        mel_bound.named.insert("signal".into(), padded);
        mel_bound.named.insert("bank".into(), filters);
        mel_bound
            .named
            .insert(STFT_WINDOW_INPUT.into(), decode(&tables.window));
        mel_bound
            .named
            .insert(STFT_COS_INPUT.into(), decode(&tables.cos));
        mel_bound
            .named
            .insert(STFT_SIN_INPUT.into(), decode(&tables.sin));
        let mel_start = std::time::Instant::now();
        let mel_values = mel_bound.run("mel", &[mel], &[], &mut runtime)?.remove(0);
        let mel = mel_start.elapsed();
        sample_peak(&mut gpu_peak, &runtime);
        // The encoder consumes the first 2*source_positions frames.
        let needed = (self.shape.source_positions * 2) as usize;
        let mut mel_input = vec![0.0; (self.shape.mels as usize) * needed];
        for row in 0..self.shape.mels as usize {
            for frame in 0..needed {
                mel_input[row * needed + frame] = mel_values[row * frames as usize + frame];
            }
        }

        let mut bound = Bound::new(self.shape.clone());
        let mel_leaf = bound.leaf("mel", &[1, self.shape.mels, needed as u32]);
        bound.named.insert("mel".into(), mel_input);
        let stem = bound.encoder_stem();
        let layers: Vec<_> = (0..self.shape.encoder_layers)
            .map(|layer| bound.encoder_layer(layer))
            .collect();
        let memory = encoder(
            &mut bound.program,
            mel_leaf,
            &stem,
            &layers,
            self.shape.encoder_heads,
            self.shape.encoder_head_dim,
            self.shape.eps,
            self.shape.mels,
            needed as u32,
        )?;
        let mut cross = Vec::new();
        for layer in 0..self.shape.decoder_layers {
            let prefix = format!("model.decoder.layers.{layer}.encoder_attn");
            let k = bound.memory_heads(&format!("{prefix}.k_proj"), false);
            let v = bound.memory_heads(&format!("{prefix}.v_proj"), true);
            let (k, v) = cross_kv(
                &mut bound.program,
                memory,
                k.weight,
                k.bias,
                v.weight,
                v.bias,
            )?;
            cross.push((k, v));
        }
        let mut outputs = vec![memory];
        for (k, v) in &cross {
            outputs.push(*k);
            outputs.push(*v);
        }
        bound.bind(self)?;
        let encode_start = std::time::Instant::now();
        let mut values = bound.run("encode", &outputs, &[], &mut runtime)?;
        let encode = encode_start.elapsed();
        sample_peak(&mut gpu_peak, &runtime);

        let mut decoder = Bound::new(self.shape.clone());
        let step_outputs = decoder.decoder_graph()?;
        for layer in 0..self.shape.decoder_layers {
            let k = std::mem::take(&mut values[1 + layer * 2]);
            let v = std::mem::take(&mut values[2 + layer * 2]);
            decoder.named.insert(format!("cross_k_{layer}"), k);
            decoder.named.insert(format!("cross_v_{layer}"), v);
        }
        decoder.bind(self)?;
        let program_hash = program_hash(&decoder.program);

        let mut tokens: Vec<u32> = policy.prompt().to_vec();
        let mut cached = 0;
        let mut ended = false;
        let mut step_times = Vec::new();
        let mut first_token = Duration::ZERO;
        let mut last_token = Duration::ZERO;
        let decode_start = Instant::now();
        for step in 0..policy.step_cap() as usize {
            let step_start = Instant::now();
            let fresh = &tokens[cached..];
            decoder.stage_step(fresh, cached);
            let extent = self.cache_extent(
                &mut runtime,
                &decoder,
                &step_outputs,
                (program_hash, cached, fresh.len()),
            );
            decoder.fit_cache(extent);
            let symbols = [fresh.len() as u64, extent as u64];
            let mut rows = decoder.run("decode", &step_outputs, &symbols, &mut runtime)?;
            let flat = rows.remove(0);
            decoder.trim_cache(cached);
            for (name, added) in cache_names(self.shape.decoder_layers).zip(rows) {
                if let Some(plane) = decoder.named.get_mut(&name) {
                    plane.extend(added);
                }
            }
            cached = tokens.len();
            let vocab = self.shape.vocab as usize;
            let mut row = flat[flat.len() - vocab..].to_vec();
            let chosen = policy.pick(&mut row, step)?;
            tokens.push(chosen);
            sample_peak(&mut gpu_peak, &runtime);
            step_times.push(step_start.elapsed());
            last_token = start.elapsed();
            if step == 0 {
                first_token = last_token;
            }
            if chosen == policy.end_token() {
                ended = true;
                break;
            }
        }
        let decode = decode_start.elapsed();
        let generated = &tokens[policy.prompt().len()..];
        let text = tokenizer_text(&self.dir, generated)?;
        Ok(Transcript {
            tokens: generated.to_vec(),
            text,
            ended_on_eot: ended,
            mel,
            encode,
            decode,
            prompt_tokens: policy.prompt().len(),
            first_token,
            last_token,
            step_times,
            gpu_allocated_bytes: runtime.allocated_bytes(),
            gpu_allocated_peak_bytes: gpu_peak,
        })
    }
}

struct FrontEnd {
    filters: Vec<f32>,
    n_fft: u32,
    hop: u32,
    mels: u32,
}

fn front_end(dir: &Path) -> Result<FrontEnd> {
    let text = std::fs::read_to_string(dir.join("preprocessor_config.json"))?;
    let value: Value = serde_json::from_str(&text)?;
    let integer = |key: &str| -> Result<u32> {
        value
            .get(key)
            .and_then(Value::as_u64)
            .map(|v| v as u32)
            .with_context(|| format!("preprocessor_config.json lacks {key}"))
    };
    let mut filters = Vec::new();
    if let Some(rows) = value["mel_filters"].as_array() {
        for row in rows {
            for cell in row.as_array().context("filter row")? {
                filters.push(cell.as_f64().context("filter cell")? as f32);
            }
        }
    } else {
        // Some published checkpoints omit the filterbank from
        // preprocessor_config.json; the checkpoint's own ggml file carries it.
        let bytes = std::fs::read(dir.join("mel_filters.f32"))
            .context("preprocessor_config.json has no mel_filters and mel_filters.f32 is absent")?;
        filters = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect();
    }
    Ok(FrontEnd {
        filters,
        n_fft: integer("n_fft")?,
        hop: integer("hop_length")?,
        mels: integer("feature_size")?,
    })
}

fn tokenizer_text(dir: &Path, ids: &[u32]) -> Result<String> {
    let bytes = std::fs::read(dir.join("tokenizer.json")).context("tokenizer.json")?;
    let config: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let id = |key: &str| config.get(key).and_then(Value::as_u64).map(|v| v as u32);
    let vocab = proxima_tokenizer::tokenizer_json::load_tokenizer_json(
        &bytes,
        id("bos_token_id"),
        id("eos_token_id"),
        None,
    )?;
    proxima_tokenizer::pipe::decode(ids, &vocab).map_err(Into::into)
}

fn pad(wav: &[f32]) -> Result<Vec<f32>> {
    if wav.len() < 200 || wav.len() > 480_040 {
        bail!("one 30 s window is 200..=480040 samples; got {}", wav.len());
    }
    let mut out = vec![0.0; 480_240];
    for index in 0..200 {
        out[199 - index] = wav[index + 1];
    }
    out[200..200 + wav.len()].copy_from_slice(wav);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use conflaguration::Settings;
    use proxima_numeric::{NumericPolicy, decode_blocks, encode_blocks};
    use proxima_safetensors::{SafetensorsModel, TensorPayload, write_complete};

    use super::*;

    const BIT_EXACT_JSON: &str = r#"{"contraction":false,"reassociation":false,"nan_assumptions":false,"signed_zero":false,"approx_functions":false,"epilogue_sources":false}"#;

    const EMBEDDING: &str = "model.decoder.embed_tokens.weight";

    fn decoder_tensor_shapes() -> Vec<(String, Vec<u64>)> {
        let layer = "model.decoder.layers.0";
        let matrix = vec![4, 4];
        let vector = vec![4];
        let attention = |name: &str| {
            [
                (format!("{layer}.{name}.q_proj.weight"), matrix.clone()),
                (format!("{layer}.{name}.q_proj.bias"), vector.clone()),
                (format!("{layer}.{name}.out_proj.weight"), matrix.clone()),
                (format!("{layer}.{name}.out_proj.bias"), vector.clone()),
            ]
        };
        let norm = |name: &str| {
            [
                (format!("{name}.weight"), vector.clone()),
                (format!("{name}.bias"), vector.clone()),
            ]
        };
        let mut shapes = vec![
            (EMBEDDING.to_string(), vec![6, 4]),
            (
                "model.decoder.embed_positions.weight".to_string(),
                vec![3, 4],
            ),
            (format!("{layer}.self_attn.k_proj.weight"), matrix.clone()),
            (format!("{layer}.self_attn.v_proj.weight"), matrix.clone()),
            (format!("{layer}.self_attn.v_proj.bias"), vector.clone()),
            (format!("{layer}.fc1.weight"), vec![8, 4]),
            (format!("{layer}.fc1.bias"), vec![8]),
            (format!("{layer}.fc2.weight"), vec![4, 8]),
            (format!("{layer}.fc2.bias"), vec![4]),
        ];
        shapes.extend(attention("self_attn"));
        shapes.extend(attention("encoder_attn"));
        shapes.extend(norm(&format!("{layer}.self_attn_layer_norm")));
        shapes.extend(norm(&format!("{layer}.encoder_attn_layer_norm")));
        shapes.extend(norm(&format!("{layer}.final_layer_norm")));
        shapes.extend(norm("model.decoder.layer_norm"));
        shapes
    }

    /// a real safetensors file holding every tensor the decoder program reads,
    /// each element distinct from every other tensor's
    fn decoder_checkpoint() -> Checkpoint {
        let dtype = super::super::encoder_decoder::ACTIVATION;
        let format = dtype
            .format()
            .expect("the activation dtype has a numeric format");
        let payloads: Vec<(String, Vec<u64>, Vec<u8>)> = decoder_tensor_shapes()
            .into_iter()
            .enumerate()
            .map(|(index, (name, shape))| {
                let count = shape.iter().product::<u64>() as usize;
                let values: Vec<_> = (0..count)
                    .map(|element| (index * 1000 + element) as _)
                    .collect();
                let mut data = vec![0; count * format.size_bytes()];
                encode_blocks(&format, values.as_slice(), &mut data).expect("encodes the tensor");
                (name, shape, data)
            })
            .collect();
        let file = write_complete(&SafetensorsModel {
            tensors: payloads
                .iter()
                .map(|(name, shape, data)| TensorPayload {
                    name: name.clone(),
                    dtype,
                    shape: shape.clone(),
                    data,
                })
                .collect(),
            metadata: BTreeMap::new(),
        })
        .expect("writes a real safetensors buffer");
        let (manifest, data_start) = parse_checkpoint(&file).expect("the written header parses");
        (mapped(&file), manifest, data_start)
    }

    /// maps `bytes` through a real file, as `Model::load` maps a checkpoint; the
    /// mapping outlives the unlinked file on unix
    fn mapped(bytes: &[u8]) -> CheckpointMapping {
        let directory = tempfile::tempdir().expect("creates a checkpoint directory");
        let path = directory.path().join("model.safetensors");
        std::fs::write(&path, bytes).expect("writes the checkpoint file");
        CheckpointMapping::open_file(&path).expect("maps the checkpoint file")
    }

    fn decoder_model() -> Model {
        let config = serde_json::json!({
            "encoder_layers": 1, "decoder_layers": 1,
            "encoder_attention_heads": 2, "decoder_attention_heads": 2,
            "d_model": 4, "num_mel_bins": 2, "encoder_ffn_dim": 8, "decoder_ffn_dim": 8,
            "max_source_positions": 2, "max_target_positions": 3, "vocab_size": 6,
        });
        let shape = Shape::from_config(&config).expect("the decoder config is complete");
        Model::from_parts(
            decoder_checkpoint(),
            shape,
            Value::Null,
            config,
            Path::new(""),
        )
    }

    fn bound_decoder(model: &Model) -> Bound<'_> {
        let mut decoder = Bound::new(model.shape.clone());
        decoder.decoder_graph().expect("the decoder graph builds");
        decoder
            .bind(model)
            .expect("the shared binder binds the decoder");
        decoder
    }

    #[test]
    fn decoder_binds_with_no_flipped_names_and_no_decoded_copies() {
        let model = decoder_model();

        let decoder = bound_decoder(&model);

        let flipped = decoder
            .weights
            .names()
            .into_iter()
            .filter(|name| name.contains("#flipped"))
            .count();
        assert_eq!(flipped, 0, "a stored-order leaf binds under its own name");
        assert_eq!(
            decoder.weights.owned().len(),
            0,
            "no weight is decoded into a copy"
        );
    }

    #[test]
    fn tied_head_binds_the_embedding_as_one_buffer() {
        let model = decoder_model();
        let entry = model
            .manifest
            .tensor(EMBEDDING)
            .expect("the embedding is stored");
        let (start, end) = entry.data_offsets;
        let file = model.mapping.bytes();
        let raw = &file[(model.data_start + start) as usize..(model.data_start + end) as usize];
        let format = entry
            .dtype
            .format()
            .expect("the embedding has a numeric format");
        let mut values = vec![0.0; entry.shape.iter().product::<u64>() as usize];
        decode_blocks(&format, raw, &mut values).expect("decodes the embedding");
        let embedding = format!("{:?}", QuantizedBlock::from(values.as_slice()));

        let decoder = bound_decoder(&model);

        let blocks = decoder.weights.blocks().expect("the bound blocks resolve");
        let carrying = blocks
            .iter()
            .filter(|(_, block)| format!("{block:?}") == embedding)
            .count();
        assert_eq!(carrying, 1, "the head reads the embedding leaf, not a copy");
        assert!(!decoder.weights.names().contains("lm_head"));
    }

    fn empty_model() -> Model {
        let tensors = (mapped(b"-"), Manifest::default(), 0);
        let config = serde_json::json!({
            "encoder_layers": 1, "decoder_layers": 1,
            "encoder_attention_heads": 1, "decoder_attention_heads": 1,
            "d_model": 4, "num_mel_bins": 1, "encoder_ffn_dim": 4, "decoder_ffn_dim": 4,
            "max_source_positions": 1, "max_target_positions": 1, "vocab_size": 1,
        });
        let shape = Shape::from_config(&config).expect("the tiny config is complete");
        Model::from_parts(tensors, shape, Value::Null, config, Path::new(""))
    }

    #[test]
    fn model_defaults_to_the_serving_settings_policy() {
        let model = empty_model();

        assert_eq!(
            model.settings.numeric_policy,
            ServingSettings::default().numeric_policy
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn runtime_uses_the_gpu_exactly_when_settings_offload_every_layer() {
        let cpu = empty_model().with_settings(ServingSettings {
            gpu_layers: 0,
            ..ServingSettings::default()
        });
        let offloaded = empty_model().with_settings(ServingSettings {
            gpu_layers: crate::serving::GPU_LAYERS_ALL,
            ..ServingSettings::default()
        });

        assert!(!cpu.runtime().expect("cpu settings are admitted").uses_gpu());
        assert!(
            offloaded
                .runtime()
                .expect("offload settings are admitted")
                .uses_gpu()
        );
    }

    #[cfg(not(feature = "gpu"))]
    #[test]
    fn every_layer_offload_is_refused_without_a_gpu_build() {
        let model = empty_model().with_settings(ServingSettings {
            gpu_layers: crate::serving::GPU_LAYERS_ALL,
            ..ServingSettings::default()
        });

        let refusal = model.runtime().err().expect("a build without a gpu backend refuses -ngl all");

        assert!(format!("{refusal}").contains("no GPU backend"), "{refusal}");
    }

    #[test]
    fn serving_env_spells_the_bit_exact_policy() {
        let loaded = temp_env::with_vars(
            [("PROXIMA_SERVING_NUMERIC_POLICY", Some(BIT_EXACT_JSON))],
            || ServingSettings::from_env().expect("the bit-exact env parses"),
        );

        assert_eq!(loaded.numeric_policy, NumericPolicy::bit_exact());
        assert_ne!(
            loaded.numeric_policy,
            ServingSettings::default().numeric_policy
        );
    }
}
