//! Host-side assembly of the generic encoder-decoder builders: read a
//! safetensors checkpoint and its config, build the mel/encoder/cross-KV/
//! decoder graphs, and run a greedy transcript. This is the std half; the
//! graph builders themselves stay alloc-only and model-neutral.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result, bail};
use proxima_numeric::decode_blocks;
use proxima_safetensors::{Manifest, SafetensorsParser, TensorEntry};
use proxima_tensor::cpu::{Evaluated, QuantizedBlock, evaluate_quantized_named};
use proxima_tensor::op::{NodeId, Op};
use proxima_tensor::spec::{
    STFT_COS_INPUT, STFT_SIN_INPUT, STFT_WINDOW_INPUT, StftLogMelConfig, stft_log_mel, stft_tables,
};
use serde_json::Value;

use crate::bind::transpose_out_in_to_in_out;
use crate::decode_policy::{DecodeConfig, DecodePolicy};
use crate::families::encoder_decoder::{
    DecoderLayer, DecoderStem, EncoderLayer, EncoderStem, Linear, encoder,
};
use crate::hf_bind::safetensors_tensor_as_packed_block;

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

/// suffix of the cache key holding a stored `[out, in]` matrix as `[in, out]`
const FLIPPED: &str = "#flipped";

/// The checkpoint's tensors, in the layout the builders read.
pub struct Tensors {
    bytes: Vec<u8>,
    manifest: Manifest,
    data_start: u64,
    /// stored half-width leaves a builder reads decoded, filled on first use and kept across decode steps
    decoded: BTreeMap<String, OnceLock<Vec<f32>>>,
}

impl Tensors {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| path.display().to_string())?;
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
        let decoded = manifest
            .tensors
            .iter()
            .flat_map(|entry| [entry.name.clone(), format!("{}{FLIPPED}", entry.name)])
            .map(|key| (key, OnceLock::new()))
            .collect();
        Ok(Self {
            bytes,
            manifest,
            data_start: (8 + header_len) as u64,
            decoded,
        })
    }

    fn block(&self, name: &str) -> QuantizedBlock<'_> {
        safetensors_tensor_as_packed_block(&self.manifest, &self.bytes, self.data_start, name)
            .unwrap_or_else(|error| panic!("{name} binds in its stored format: {error}"))
    }

    fn entry(&self, name: &str) -> &TensorEntry {
        self.manifest
            .tensors
            .iter()
            .find(|entry| entry.name == name)
            .unwrap_or_else(|| panic!("no tensor {name}"))
    }

    fn decoded(&self, name: &str) -> &[f32] {
        self.decoded
            .get(name)
            .unwrap_or_else(|| panic!("no tensor {name}"))
            .get_or_init(|| {
                let entry = self.entry(name);
                let (start, end) = entry.data_offsets;
                let raw = &self.bytes
                    [(self.data_start + start) as usize..(self.data_start + end) as usize];
                let format = entry
                    .dtype
                    .format()
                    .unwrap_or_else(|| panic!("{name} has no numeric format ({:?})", entry.dtype));
                let mut tensor = vec![0.0; entry.shape.iter().product::<u64>() as usize];
                decode_blocks(&format, raw, &mut tensor)
                    .unwrap_or_else(|error| panic!("decoding {name} as {format:?}: {error}"));
                tensor
            })
    }

    fn dims(&self, name: &str) -> Vec<usize> {
        self.entry(name)
            .shape
            .iter()
            .map(|&dim| dim as usize)
            .collect()
    }
}

struct Bound<'a> {
    program: Vec<Op>,
    named: BTreeMap<String, Vec<f32>>,
    blocks: Vec<(String, QuantizedBlock<'a>)>,
    shape: Shape,
}

impl<'a> Bound<'a> {
    fn new(shape: Shape) -> Self {
        Self {
            program: Vec::new(),
            named: BTreeMap::new(),
            blocks: Vec::new(),
            shape,
        }
    }

    fn leaf(&mut self, name: &str, shape: &[u32]) -> NodeId {
        super::encoder_decoder::leaf(&mut self.program, shape, name)
    }

    fn weight(&mut self, name: &str, shape: &[u32], stored: &str, tensors: &'a Tensors) -> NodeId {
        let node = self.leaf(name, shape);
        let block = tensors.block(stored);
        if block.packed_bytes().is_some() {
            self.blocks.push((name.into(), block));
        } else {
            let flipped = tensors
                .decoded
                .get(&format!("{stored}{FLIPPED}"))
                .unwrap_or_else(|| panic!("no tensor {stored}"))
                .get_or_init(|| {
                    let dims = tensors.dims(stored);
                    transpose_out_in_to_in_out(tensors.decoded(stored), stored, dims[0], dims[1])
                        .unwrap_or_else(|error| panic!("{stored} transposes: {error}"))
                });
            self.blocks
                .push((name.into(), QuantizedBlock::from(flipped.as_slice())));
        }
        node
    }

    fn plain(&mut self, name: &str, shape: &[u32], tensors: &'a Tensors) -> NodeId {
        let node = self.leaf(name, shape);
        let block = tensors.block(name);
        let block = if block.packed_bytes().is_some() {
            QuantizedBlock::from(tensors.decoded(name))
        } else {
            block
        };
        self.blocks.push((name.into(), block));
        node
    }

    fn matrix(&mut self, name: &str, input: u32, output: u32, tensors: &'a Tensors) -> Linear {
        let weight_name = format!("{name}.weight");
        let weight = self.weight(&weight_name, &[input, output], &weight_name, tensors);
        let bias = self.plain(&format!("{name}.bias"), &[output], tensors);
        Linear {
            weight,
            bias: Some(bias),
        }
    }

    fn heads(&mut self, name: &str, tensors: &'a Tensors, bias: bool) -> Linear {
        let weight_name = format!("{name}.weight");
        let weight = self.weight(
            &weight_name,
            &[
                self.shape.width,
                self.shape.encoder_heads,
                self.shape.encoder_head_dim,
            ],
            &weight_name,
            tensors,
        );
        let bias = bias.then(|| {
            self.plain(
                &format!("{name}.bias"),
                &[self.shape.encoder_heads, self.shape.encoder_head_dim],
                tensors,
            )
        });
        Linear { weight, bias }
    }

    fn heads_out(&mut self, name: &str, tensors: &'a Tensors) -> Linear {
        let weight_name = format!("{name}.weight");
        let weight = self.weight(
            &weight_name,
            &[
                self.shape.encoder_heads,
                self.shape.encoder_head_dim,
                self.shape.width,
            ],
            &weight_name,
            tensors,
        );
        let bias = self.plain(&format!("{name}.bias"), &[self.shape.width], tensors);
        Linear {
            weight,
            bias: Some(bias),
        }
    }

    fn norm(&mut self, name: &str, tensors: &'a Tensors) -> Linear {
        let weight = self.plain(&format!("{name}.weight"), &[self.shape.width], tensors);
        let bias = self.plain(&format!("{name}.bias"), &[self.shape.width], tensors);
        Linear {
            weight,
            bias: Some(bias),
        }
    }

    fn encoder_layer(&mut self, layer: usize, tensors: &'a Tensors) -> EncoderLayer {
        let prefix = format!("model.encoder.layers.{layer}");
        EncoderLayer {
            attn_ln: self.norm(&format!("{prefix}.self_attn_layer_norm"), tensors),
            attn: super::encoder_decoder::Attention {
                q: self.heads(&format!("{prefix}.self_attn.q_proj"), tensors, true),
                k: self.heads(&format!("{prefix}.self_attn.k_proj"), tensors, false),
                v: self.heads(&format!("{prefix}.self_attn.v_proj"), tensors, true),
                out: self.heads_out(&format!("{prefix}.self_attn.out_proj"), tensors),
            },
            mlp_ln: self.norm(&format!("{prefix}.final_layer_norm"), tensors),
            fc1: self.matrix(
                &format!("{prefix}.fc1"),
                self.shape.width,
                self.shape.encoder_ffn,
                tensors,
            ),
            fc2: self.matrix(
                &format!("{prefix}.fc2"),
                self.shape.encoder_ffn,
                self.shape.width,
                tensors,
            ),
        }
    }

    fn decoder_layer(&mut self, layer: usize, tensors: &'a Tensors) -> DecoderLayer {
        let prefix = format!("model.decoder.layers.{layer}");
        DecoderLayer {
            self_ln: self.norm(&format!("{prefix}.self_attn_layer_norm"), tensors),
            self_attn: super::encoder_decoder::Attention {
                q: self.heads(&format!("{prefix}.self_attn.q_proj"), tensors, true),
                k: self.heads(&format!("{prefix}.self_attn.k_proj"), tensors, false),
                v: self.heads(&format!("{prefix}.self_attn.v_proj"), tensors, true),
                out: self.heads_out(&format!("{prefix}.self_attn.out_proj"), tensors),
            },
            cross_ln: self.norm(&format!("{prefix}.encoder_attn_layer_norm"), tensors),
            cross_q: self.heads(&format!("{prefix}.encoder_attn.q_proj"), tensors, true),
            cross_out: self.heads_out(&format!("{prefix}.encoder_attn.out_proj"), tensors),
            mlp_ln: self.norm(&format!("{prefix}.final_layer_norm"), tensors),
            fc1: self.matrix(
                &format!("{prefix}.fc1"),
                self.shape.width,
                self.shape.decoder_ffn,
                tensors,
            ),
            fc2: self.matrix(
                &format!("{prefix}.fc2"),
                self.shape.decoder_ffn,
                self.shape.width,
                tensors,
            ),
        }
    }

    fn encoder_stem(&mut self, tensors: &'a Tensors) -> EncoderStem {
        let width = self.shape.width;
        let mels = self.shape.mels;
        let conv1 = self.plain("model.encoder.conv1.weight", &[width, mels, 3], tensors);
        let conv1_bias = self.plain("model.encoder.conv1.bias", &[width], tensors);
        let conv2 = self.plain("model.encoder.conv2.weight", &[width, width, 3], tensors);
        let conv2_bias = self.plain("model.encoder.conv2.bias", &[width], tensors);
        let position = self.plain(
            "model.encoder.embed_positions.weight",
            &[self.shape.source_positions, width],
            tensors,
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
            layer_norm: self.norm("model.encoder.layer_norm", tensors),
        }
    }

    fn decoder_stem(&mut self, tensors: &'a Tensors) -> DecoderStem {
        let tokens_name = "model.decoder.embed_tokens.weight";
        let tokens = self.plain(tokens_name, &[self.shape.vocab, self.shape.width], tensors);
        let position = self.plain(
            "model.decoder.embed_positions.weight",
            &[self.shape.target_positions, self.shape.width],
            tensors,
        );
        let head = self.weight(
            "lm_head",
            &[self.shape.width, self.shape.vocab],
            tokens_name,
            tensors,
        );
        DecoderStem {
            tokens,
            position,
            layer_norm: self.norm("model.decoder.layer_norm", tensors),
            head: Linear {
                weight: head,
                bias: None,
            },
        }
    }

    fn run(&self, outputs: &[NodeId], device: Device) -> Result<Vec<Vec<f32>>> {
        let named: Vec<(&str, QuantizedBlock<'_>)> = self
            .named
            .iter()
            .map(|(name, data)| (name.as_str(), QuantizedBlock::from(data.as_slice())))
            .chain(
                self.blocks
                    .iter()
                    .map(|(name, block)| (name.as_str(), *block)),
            )
            .collect();
        match device {
            Device::Cpu => {
                let evaluated = evaluate_quantized_named(&self.program, &[], &named, outputs)
                    .map_err(|error| anyhow::anyhow!("program does not evaluate: {error:?}"))?;
                collect_outputs(&evaluated, outputs)
            }
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Device::Metal(policy) => {
                let plan = omega::plan_named(&self.program, &[], &named, outputs, policy)
                    .map_err(|error| anyhow::anyhow!("metal plan: {error:?}"))?;
                let evaluated = omega::execute_plan_named(&plan, &named)
                    .map_err(|error| anyhow::anyhow!("metal executes: {error:?}"))?;
                collect_outputs(&evaluated, outputs)
            }
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            Device::Metal(_) => bail!("the metal backend needs --features metal on macOS"),
        }
    }
}

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

/// Which backend executes a stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Metal(proxima_numeric::NumericPolicy),
}

/// One transcription and the values the caller can check.
pub struct Transcript {
    pub tokens: Vec<u32>,
    pub text: String,
    pub ended_on_eot: bool,
    pub mel: std::time::Duration,
    pub encode: std::time::Duration,
    pub decode: std::time::Duration,
}

/// A loaded checkpoint ready to transcribe.
pub struct Model {
    tensors: Tensors,
    shape: Shape,
    generation: Value,
    config: Value,
    dir: PathBuf,
    device: Device,
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
        let tensors = Tensors::load(&dir.join("model.safetensors"))?;
        Ok(Self::from_parts(tensors, shape, generation, config, dir))
    }

    fn from_parts(
        tensors: Tensors,
        shape: Shape,
        generation: Value,
        config: Value,
        dir: &Path,
    ) -> Self {
        Self {
            tensors,
            shape,
            generation,
            config,
            dir: dir.to_path_buf(),
            device: Device::Cpu,
        }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Runs every stage on `device`; a Metal device carries the run's numeric policy.
    #[must_use]
    pub const fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    pub fn transcribe(&self, wav: &[f32], language: &str) -> Result<Transcript> {
        let device = self.device;
        let policy = DecodePolicy::from_checkpoint(
            &self.generation,
            &self.config,
            &DecodeConfig {
                language: language.into(),
                ..DecodeConfig::default()
            },
        )?;

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
        let mel_values = mel_bound.run(&[mel], device)?.remove(0);
        let mel = mel_start.elapsed();
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
        let stem = bound.encoder_stem(&self.tensors);
        let layers: Vec<_> = (0..self.shape.encoder_layers)
            .map(|layer| bound.encoder_layer(layer, &self.tensors))
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
            let k = bound.heads(&format!("{prefix}.k_proj"), &self.tensors, false);
            let v = bound.heads(&format!("{prefix}.v_proj"), &self.tensors, true);
            let (k, v) = super::encoder_decoder::cross_kv(
                &mut bound.program,
                memory,
                k,
                v,
                self.shape.decoder_head_dim,
            )?;
            cross.push((k, v));
        }
        let mut outputs = vec![memory];
        for (k, v) in &cross {
            outputs.push(*k);
            outputs.push(*v);
        }
        let encode_start = std::time::Instant::now();
        let values = bound.run(&outputs, device)?;
        let encode = encode_start.elapsed();
        let cross_values: Vec<(Vec<f32>, Vec<f32>)> = (0..self.shape.decoder_layers)
            .map(|layer| (values[1 + layer * 2].clone(), values[2 + layer * 2].clone()))
            .collect();

        let mut tokens: Vec<u32> = policy.prompt().to_vec();
        let mut ended = false;
        let decode_start = std::time::Instant::now();
        for step in 0..policy.step_cap() as usize {
            let mut decoder = Bound::new(self.shape.clone());
            let n = tokens.len() as u32;
            let token_leaf = decoder.leaf("tokens", &[n]);
            let positions = decoder.leaf("positions", &[n]);
            let _ = &[0_u8; 0];
            let stem = decoder.decoder_stem(&self.tensors);
            let layers: Vec<_> = (0..self.shape.decoder_layers)
                .map(|layer| decoder.decoder_layer(layer, &self.tensors))
                .collect();
            let mut cross_nodes = Vec::new();
            for (layer, (k, v)) in cross_values.iter().enumerate() {
                let k_name = format!("cross_k_{layer}");
                let v_name = format!("cross_v_{layer}");
                let k_node = decoder.leaf(
                    &k_name,
                    &[
                        self.shape.source_positions,
                        self.shape.decoder_heads,
                        self.shape.decoder_head_dim,
                    ],
                );
                let v_node = decoder.leaf(
                    &v_name,
                    &[
                        self.shape.source_positions,
                        self.shape.decoder_heads,
                        self.shape.decoder_head_dim,
                    ],
                );
                decoder.named.insert(k_name, k.clone());
                decoder.named.insert(v_name, v.clone());
                cross_nodes.push((k_node, v_node));
            }
            let logits = super::encoder_decoder::decoder(
                &mut decoder.program,
                token_leaf,
                positions,
                &stem,
                &layers,
                &cross_nodes,
                self.shape.decoder_heads,
                self.shape.decoder_head_dim,
                self.shape.eps,
            )?;
            decoder.named.insert(
                "tokens".into(),
                tokens.iter().map(|id| *id as f32).collect(),
            );
            decoder
                .named
                .insert("positions".into(), (0..n).map(|i| i as f32).collect());
            let flat = decoder.run(&[logits], device)?.remove(0);
            let vocab = self.shape.vocab as usize;
            let last = &flat[(n as usize - 1) * vocab..n as usize * vocab];
            let mut row = last.to_vec();
            let chosen = policy.pick(&mut row, step)?;
            tokens.push(chosen);
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
    use proxima_numeric::NumericPolicy;

    use super::*;
    use crate::serving_settings::ServingSettings;

    const BIT_EXACT_JSON: &str = r#"{"contraction":false,"reassociation":false,"nan_assumptions":false,"signed_zero":false,"approx_functions":false,"epilogue_sources":false}"#;

    fn empty_model() -> Model {
        let tensors = Tensors {
            bytes: Vec::new(),
            manifest: Manifest::default(),
            data_start: 0,
            decoded: BTreeMap::new(),
        };
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
    fn metal_device_carries_the_serving_settings_policy() {
        let policy = ServingSettings::default().numeric_policy;

        let model = empty_model();
        assert_eq!(model.device, Device::Cpu);
        let model = model.with_device(Device::Metal(policy));

        assert_eq!(model.device, Device::Metal(policy));
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
