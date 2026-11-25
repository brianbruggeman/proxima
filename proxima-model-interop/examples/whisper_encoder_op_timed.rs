//! Per-op GPU time of the real whisper-tiny encoder on Metal.
//!
//! Builds the encoder program the way `tests/whisper.rs` (`mod encoder`) does, over the real
//! checkpoint weights and the reference mel, plans it with the serving numeric policy (the policy
//! the encoder runs under at serve time, which grants the reassociation the row-tiled attention
//! form needs), and runs it through `omega::backend::execute_plan_named_metal_op_timed`, the
//! named wrapper of `omega::metal::execute_plan_op_timed`. Prints `gpu_ns` per op with its kind
//! and node, then a total per kind.
//!
//! `PROXIMA_WHISPER_DIR` names the checkpoint directory (`model.safetensors`);
//! `PROXIMA_WHISPER_REF` names the oracle directory whose `mel` dump is the input. Every value is
//! decoded through `proxima_numeric::decode_blocks` into the host element, so no scalar type is
//! spelled here.

#[cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
mod run {
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    use anyhow::{Context, Result, bail};
    use omega::backend::{Engine, execute_plan_named_metal_op_timed, plan_named};
    use proxima_model_interop::ServingSettings;
    use proxima_model_interop::families::encoder_decoder::{
        self, Attention, EncoderLayer, EncoderStem, Linear,
    };
    use proxima_numeric::{NumericFormat, decode_blocks};
    use proxima_safetensors::SafetensorsParser;
    use proxima_tensor::cpu::QuantizedBlock;
    use proxima_tensor::op::{NodeId, Op};

    const HEADS: u32 = 6;
    const HEAD_DIM: u32 = 64;
    const WIDTH: u32 = HEADS * HEAD_DIM;
    const MELS: u32 = 80;
    const FRAMES: u32 = 3000;
    const REFERENCE_FRAMES: usize = 4100;
    const FFN: u32 = 1536;
    const LAYERS: usize = 4;
    const POSITIONS: u32 = 1500;

    struct Tensors<Element> {
        values: BTreeMap<String, Vec<Element>>,
    }

    impl<Element: Clone + Default> Tensors<Element> {
        fn load(dir: &Path, decode: &impl Fn(&[u8], &mut [Element]) -> Result<()>) -> Result<Self> {
            let path = dir.join("model.safetensors");
            let bytes = std::fs::read(&path).with_context(|| path.display().to_string())?;
            let header_len = u64::from_le_bytes(
                bytes[..8]
                    .try_into()
                    .context("the safetensors header length is eight bytes")?,
            ) as usize;
            let mut parser = SafetensorsParser::new();
            parser.feed(&bytes[..8 + header_len]);
            parser.poll().context("the safetensors header parses")?;
            let SafetensorsParser::TensorData { manifest, .. } = parser else {
                bail!("safetensors header did not complete");
            };
            let data = &bytes[8 + header_len..];
            let mut values = BTreeMap::new();
            for entry in &manifest.tensors {
                let (start, end) = entry.data_offsets;
                let count: u64 = entry.shape.iter().product();
                let mut decoded = vec![Element::default(); count as usize];
                decode(&data[start as usize..end as usize], &mut decoded)
                    .with_context(|| entry.name.clone())?;
                values.insert(entry.name.clone(), decoded);
            }
            Ok(Self { values })
        }

        fn data(&self, name: &str) -> Result<Vec<Element>> {
            self.values
                .get(name)
                .cloned()
                .with_context(|| format!("no tensor {name}"))
        }
    }

    struct Bound<Element> {
        program: Vec<Op>,
        named: BTreeMap<String, Vec<Element>>,
    }

    impl<Element: Clone + Default> Bound<Element> {
        fn leaf(
            &mut self,
            name: &str,
            shape: &[u32],
            tensors: &Tensors<Element>,
        ) -> Result<NodeId> {
            let node = encoder_decoder::leaf(&mut self.program, shape, name);
            self.named.insert(name.to_string(), tensors.data(name)?);
            Ok(node)
        }

        fn weight_and_bias(
            &mut self,
            name: &str,
            tensors: &Tensors<Element>,
            weight_shape: &[u32],
            bias_shape: Option<&[u32]>,
        ) -> Result<Linear> {
            let weight = self.leaf(&format!("{name}.weight"), weight_shape, tensors)?;
            let bias = match bias_shape {
                Some(shape) => Some(self.leaf(&format!("{name}.bias"), shape, tensors)?),
                None => None,
            };
            Ok(Linear { weight, bias })
        }

        fn layer(&mut self, layer: usize, tensors: &Tensors<Element>) -> Result<EncoderLayer> {
            let prefix = format!("model.encoder.layers.{layer}");
            let norm = [WIDTH];
            let heads_weight = [HEADS, HEAD_DIM, WIDTH];
            let heads_bias = [HEADS, HEAD_DIM];
            Ok(EncoderLayer {
                attn_ln: self.weight_and_bias(
                    &format!("{prefix}.self_attn_layer_norm"),
                    tensors,
                    &norm,
                    Some(&norm),
                )?,
                attn: Attention {
                    q: self.weight_and_bias(
                        &format!("{prefix}.self_attn.q_proj"),
                        tensors,
                        &heads_weight,
                        Some(&heads_bias),
                    )?,
                    k: self.weight_and_bias(
                        &format!("{prefix}.self_attn.k_proj"),
                        tensors,
                        &heads_weight,
                        None,
                    )?,
                    v: self.weight_and_bias(
                        &format!("{prefix}.self_attn.v_proj"),
                        tensors,
                        &heads_weight,
                        Some(&heads_bias),
                    )?,
                    out: self.weight_and_bias(
                        &format!("{prefix}.self_attn.out_proj"),
                        tensors,
                        &[WIDTH, HEADS, HEAD_DIM],
                        Some(&norm),
                    )?,
                },
                mlp_ln: self.weight_and_bias(
                    &format!("{prefix}.final_layer_norm"),
                    tensors,
                    &norm,
                    Some(&norm),
                )?,
                fc1: self.weight_and_bias(
                    &format!("{prefix}.fc1"),
                    tensors,
                    &[FFN, WIDTH],
                    Some(&[FFN]),
                )?,
                fc2: self.weight_and_bias(
                    &format!("{prefix}.fc2"),
                    tensors,
                    &[WIDTH, FFN],
                    Some(&norm),
                )?,
            })
        }

        fn stem(&mut self, tensors: &Tensors<Element>) -> Result<EncoderStem> {
            let conv1 = self.weight_and_bias(
                "model.encoder.conv1",
                tensors,
                &[WIDTH, MELS, 3],
                Some(&[WIDTH]),
            )?;
            let conv2 = self.weight_and_bias(
                "model.encoder.conv2",
                tensors,
                &[WIDTH, WIDTH, 3],
                Some(&[WIDTH]),
            )?;
            let position = self.leaf(
                "model.encoder.embed_positions.weight",
                &[POSITIONS, WIDTH],
                tensors,
            )?;
            let layer_norm = self.weight_and_bias(
                "model.encoder.layer_norm",
                tensors,
                &[WIDTH],
                Some(&[WIDTH]),
            )?;
            Ok(EncoderStem {
                conv1,
                conv2,
                position,
                layer_norm,
            })
        }
    }

    /// The first `FRAMES` columns of the reference's `REFERENCE_FRAMES`-column mel dump, found by
    /// its `mel` file stem so no extension is spelled.
    fn reference_mel<Element: Copy + Default>(
        reference: &Path,
        decode: &impl Fn(&[u8], &mut [Element]) -> Result<()>,
    ) -> Result<Vec<Element>> {
        let path = std::fs::read_dir(reference)
            .with_context(|| reference.display().to_string())?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .find(|path| path.file_stem().is_some_and(|stem| stem == "mel"))
            .with_context(|| format!("{} holds no mel dump", reference.display()))?;
        let bytes = std::fs::read(&path).with_context(|| path.display().to_string())?;
        let mut dump = vec![Element::default(); MELS as usize * REFERENCE_FRAMES];
        decode(&bytes, &mut dump).with_context(|| path.display().to_string())?;
        let frames = FRAMES as usize;
        let mut mel = vec![Element::default(); MELS as usize * frames];
        for row in 0..MELS as usize {
            let source = &dump[row * REFERENCE_FRAMES..row * REFERENCE_FRAMES + frames];
            mel[row * frames..(row + 1) * frames].copy_from_slice(source);
        }
        Ok(mel)
    }

    fn directory(variable: &str) -> Result<PathBuf> {
        Ok(PathBuf::from(
            std::env::var(variable).with_context(|| format!("{variable} must be set"))?,
        ))
    }

    pub fn main() -> Result<()> {
        let decode = |bytes: &[u8], out: &mut [_]| {
            decode_blocks(&NumericFormat::host_element(), bytes, out)
                .map(drop)
                .map_err(anyhow::Error::from)
        };
        let tensors = Tensors::load(&directory("PROXIMA_WHISPER_DIR")?, &decode)?;
        let mut bound = Bound {
            program: Vec::new(),
            named: BTreeMap::new(),
        };
        let mel = encoder_decoder::leaf(&mut bound.program, &[1, MELS, FRAMES], "mel");
        let stem = bound.stem(&tensors)?;
        let layers = (0..LAYERS)
            .map(|layer| bound.layer(layer, &tensors))
            .collect::<Result<Vec<_>>>()?;
        let encoded = encoder_decoder::encoder(
            &mut bound.program,
            mel,
            &stem,
            &layers,
            HEADS,
            HEAD_DIM,
            1e-5,
            MELS,
            FRAMES,
        )?;
        let mel_values = reference_mel(&directory("PROXIMA_WHISPER_REF")?, &decode)?;
        bound.named.insert("mel".into(), mel_values);

        let named: Vec<(&str, QuantizedBlock<'_>)> = bound
            .named
            .iter()
            .map(|(name, data)| (name.as_str(), QuantizedBlock::from(data.as_slice())))
            .collect();
        let policy = ServingSettings::default().numeric_policy;
        let plan = plan_named(
            Engine::Gpu,
            None,
            &bound.program,
            &[],
            &named,
            &[encoded],
            policy,
        )?;
        let (_, timings) = execute_plan_named_metal_op_timed(&plan, &named, None)?;
        if timings.is_empty() {
            bail!("the timed run returned no per-op timings");
        }

        let mut per_kind: BTreeMap<&'static str, (usize, u64)> = BTreeMap::new();
        for timing in &timings {
            println!(
                "op node={} kind={} extents={:?} gpu_ns={}",
                timing.node.0, timing.kind, timing.extents, timing.gpu_ns
            );
            let entry = per_kind.entry(timing.kind).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += timing.gpu_ns;
        }
        for (kind, (count, nanos)) in &per_kind {
            println!("kind={kind} ops={count} gpu_ns_total={nanos}");
        }
        let total: u64 = timings.iter().map(|timing| timing.gpu_ns).sum();
        println!("ops={} gpu_ns_total={total}", timings.len());
        Ok(())
    }
}

#[cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
fn main() -> anyhow::Result<()> {
    run::main()
}

#[cfg(not(all(feature = "metal", feature = "instrument", target_os = "macos")))]
fn main() {
    eprintln!("whisper_encoder_op_timed needs the metal and instrument features on macos");
}
