use std::env;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

#[path = "ouro_loopcd_lab/checkpoint.rs"]
pub mod checkpoint;
#[path = "ouro_native_oracle/compare.rs"]
mod compare;
#[path = "ouro_native_oracle/fixtures.rs"]
mod fixtures;
#[path = "ouro_loopcd_lab/graph.rs"]
pub mod graph;
#[path = "ouro_native_oracle/math.rs"]
mod math;
#[path = "ouro_native_oracle/model.rs"]
mod model;
#[path = "ouro_native_oracle/verify.rs"]
mod verify;

use checkpoint::{Checkpoint, DIGEST, REVISION, WeightMappingMode};
use compare::{Archive, Comparisons, JsonLines};
use math::Mode;
use model::{LayerWeights, Model};

struct Arguments {
    checkpoint: PathBuf,
    output: Option<PathBuf>,
    prefix: Vec<u32>,
    mode: Mode,
    depth: usize,
    strength: f32,
    compare: Option<PathBuf>,
    fixtures_only: bool,
    negative_controls: bool,
    audit: bool,
    verify_artifacts: Option<PathBuf>,
    verify_contrast: Option<PathBuf>,
    compare_modes: Option<(PathBuf, PathBuf)>,
    paper_method: Option<String>,
    compare_paper: Option<PathBuf>,
    reference_oracle: Option<PathBuf>,
    scope_controls: Option<PathBuf>,
}

impl Arguments {
    fn parse() -> io::Result<Self> {
        let mut arguments = Self {
            checkpoint: PathBuf::from("/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/models/ouro-1.4b"),
            output: None,
            prefix: vec![19556, 905, 30],
            mode: Mode::F32,
            depth: 4,
            strength: 0.5,
            compare: None,
            fixtures_only: false,
            negative_controls: false,
            audit: false,
            verify_artifacts: None,
            verify_contrast: None,
            compare_modes: None,
            paper_method: None,
            compare_paper: None,
            reference_oracle: None,
            scope_controls: None,
        };
        let mut source = env::args().skip(1);
        while let Some(flag) = source.next() {
            let mut next = || {
                source
                    .next()
                    .ok_or_else(|| io::Error::other(format!("argument {flag} needs a value")))
            };
            match flag.as_str() {
                "--checkpoint" => arguments.checkpoint = PathBuf::from(next()?),
                "--output" => arguments.output = Some(PathBuf::from(next()?)),
                "--prefix-ids" => {
                    arguments.prefix = next()?
                        .split(',')
                        .map(|value| value.parse::<u32>().map_err(io::Error::other))
                        .collect::<io::Result<_>>()?
                }
                "--mode" => {
                    arguments.mode = match next()?.as_str() {
                        "f32" => Mode::F32,
                        "bf16" => Mode::Bf16,
                        _ => return Err(io::Error::other("mode must be f32 or bf16")),
                    }
                }
                "--depth" => arguments.depth = next()?.parse().map_err(io::Error::other)?,
                "--strength" => arguments.strength = next()?.parse().map_err(io::Error::other)?,
                "--compare" => arguments.compare = Some(PathBuf::from(next()?)),
                "--fixtures-only" => arguments.fixtures_only = true,
                "--negative-controls" => arguments.negative_controls = true,
                "--audit-independence" => arguments.audit = true,
                "--verify-artifacts" => arguments.verify_artifacts = Some(PathBuf::from(next()?)),
                "--verify-contrast" => arguments.verify_contrast = Some(PathBuf::from(next()?)),
                "--compare-modes" => {
                    arguments.compare_modes = Some((PathBuf::from(next()?), PathBuf::from(next()?)))
                }
                "--paper-method" => arguments.paper_method = Some(next()?),
                "--compare-paper" => arguments.compare_paper = Some(PathBuf::from(next()?)),
                "--reference-oracle" => arguments.reference_oracle = Some(PathBuf::from(next()?)),
                "--scope-negative-controls" => {
                    arguments.scope_controls = Some(PathBuf::from(next()?))
                }
                "--help" => {
                    println!(
                        "native Ouro oracle: --fixtures-only [--negative-controls] | --mode f32|bf16 [--compare ARCHIVE] | --paper-method baseline|fixed-logits|adaptive-logits|fixed-hidden --prefix-ids IDS --depth 4 --strength .5 | --verify-artifacts DIR | --verify-contrast DIR | --compare-modes F32DIR BF16DIR | --compare-paper FILE --reference-oracle DIR | --audit-independence; use --output FRESH-DIR for runs"
                    );
                    return Err(io::Error::other("help requested"));
                }
                _ => return Err(io::Error::other(format!("unknown oracle argument {flag}"))),
            }
        }
        if arguments.prefix.is_empty()
            || arguments.depth == 0
            || arguments.depth > 4
            || !arguments.strength.is_finite()
            || arguments.strength < 0.0
        {
            return Err(io::Error::other(
                "prefix, depth or strength outside admitted values",
            ));
        }
        if let Some(method) = &arguments.paper_method {
            if ![
                "baseline",
                "fixed-logits",
                "adaptive-logits",
                "fixed-hidden",
            ]
            .contains(&method.as_str())
            {
                return Err(io::Error::other("unknown paper method"));
            }
            if arguments.compare.is_some() {
                return Err(io::Error::other(
                    "archive comparison is the four-loop baseline oracle mode",
                ));
            }
        }
        if arguments.compare.is_some() && (arguments.mode != Mode::F32 || arguments.depth != 4) {
            return Err(io::Error::other(
                "archived F32 comparison requires f32 and depth four",
            ));
        }
        Ok(arguments)
    }

    fn output(&self) -> io::Result<&Path> {
        self.output
            .as_deref()
            .ok_or_else(|| io::Error::other("fresh --output directory required"))
    }
}

fn save(path: &Path, record: &Value) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut writer, record).map_err(io::Error::other)?;
    writer.write_all(b"\n")?;
    writer.flush()
}

fn output_directory(path: &Path) -> io::Result<()> {
    if path.exists() {
        return Err(io::Error::other(format!(
            "output already exists: {}",
            path.display()
        )));
    }
    fs::create_dir_all(path)
}

fn capture(
    writer: &mut JsonLines,
    comparisons: &mut Option<Comparisons>,
    archive: Option<&Archive>,
    key: &str,
    values: &[f32],
    shape: &[usize],
    mode: Mode,
) -> io::Result<()> {
    let actual = compare::vector(writer, key, values, shape, mode.label())?;
    if let (Some(comparisons), Some(archive)) = (comparisons, archive) {
        comparisons.compare(key, &actual, archive.get(key)?, "end_to_end_f32")?;
    }
    Ok(())
}

fn dtype_table(directory: &Path) -> io::Result<usize> {
    let mut records = JsonLines::create(&directory.join("dtype-sites.jsonl"))?;
    for (operation, policy, line) in [
        ("embeddings_weights", "exact bf16 byte decode", 585),
        ("linear", "f32 accumulation then bf16 RNE output", 304),
        (
            "rmsnorm",
            "f32 normalization; bf16 cast before learned scale; bf16 product",
            357,
        ),
        (
            "rope_tables",
            "f32 frequency and cos/sin then bf16 cast",
            467,
        ),
        ("rope_application", "bf16 products then bf16 sum", 110),
        (
            "attention_scores",
            "bf16 dot output then scaled output then mask sum",
            241,
        ),
        ("softmax", "f32 softmax then bf16 cast", 246),
        ("attention_values", "f32 accumulation then bf16 output", 253),
        (
            "silu_ffn",
            "bf16 fused silu output then product then down output",
            76,
        ),
        ("residuals", "bf16 sum output", 415),
        ("loop_norm_head", "source norm and linear policies", 606),
        ("loopcd", "explicit f32 promoted logits equation", 0),
    ] {
        records.emit(&json!({"kind":"dtype_site","operation":operation,"policy":policy,"author_source":"proxima-tensor/specs/decode-performance-lab/references/ouro-modeling_ouro.py","author_line":line,"runtime_bit_equality":"unmeasured"}))?;
    }
    records.flush()?;
    Ok(records.line)
}

fn run(arguments: &Arguments, directory: &Path) -> io::Result<Value> {
    let checkpoint = Checkpoint::open(&arguments.checkpoint, WeightMappingMode::CpuUnmapped)?;
    let model = Model::prepare(&checkpoint, arguments.prefix.len(), arguments.mode)?;
    let archive = arguments
        .compare
        .as_deref()
        .map(|path| Archive::open(path, &arguments.prefix))
        .transpose()?;
    let context = json!({"model":"ByteDance/Ouro-1.4B","checkpoint_revision":REVISION,"checkpoint_expected_sha256":DIGEST,"checkpoint_observed_sha256":DIGEST,"prefix_ids":arguments.prefix,"positions":arguments.prefix.len(),"depth":arguments.depth,"mode":arguments.mode.label(),"numeric_policy":if arguments.mode==Mode::F32 {"independent Accelerate and Rust; BF16 checkpoint values, F32 activations"} else {"independent BF16 RNE source-semantics emulator; author runtime bit equality unmeasured"},"use_cache":false,"readout":"explicit terminal recurrent step","source_archive":arguments.compare,"paper_method":arguments.paper_method,"strength":if arguments.paper_method.as_deref()==Some("baseline"){None}else{Some(arguments.strength)}});
    save(&directory.join("context.json"), &context)?;
    let mut vectors = JsonLines::create(&directory.join("vectors.jsonl"))?;
    let mut replays = JsonLines::create(&directory.join("replay-vectors.jsonl"))?;
    let mut comparisons = if archive.is_some() {
        Some(Comparisons::create(directory)?)
    } else {
        None
    };
    let mut hidden = model.embedding(&checkpoint, &arguments.prefix)?;
    let hidden_shape = [model.positions, model.hidden_width];
    let mut reference_logits = None;
    let mut terminal_logits = None;
    let mut raw_reference = Vec::new();
    let mut raw_terminal = Vec::new();
    let mut bf16_violations = 0usize;
    let mut heads = 0usize;
    let mut loop_norm_sites = 0usize;
    let mut check_bf16 = |values: &[f32]| {
        if arguments.mode == Mode::Bf16 {
            bf16_violations += values
                .iter()
                .filter(|value| value.to_bits() & 0xffff != 0)
                .count();
        }
    };
    for iteration in 1..=arguments.depth {
        for layer in 0..24 {
            let key = compare::layer_key(iteration, layer, "input");
            check_bf16(&hidden);
            capture(
                &mut vectors,
                &mut comparisons,
                archive.as_ref(),
                &key,
                &hidden,
                &hidden_shape,
                arguments.mode,
            )?;
            let weights = LayerWeights::read(&checkpoint, layer)?;
            let output = model.layer(&hidden, &weights)?;
            let key = compare::layer_key(iteration, layer, "output");
            check_bf16(&output);
            capture(
                &mut vectors,
                &mut comparisons,
                archive.as_ref(),
                &key,
                &output,
                &hidden_shape,
                arguments.mode,
            )?;
            if let (Some(archive), Some(comparisons)) = (archive.as_ref(), comparisons.as_mut()) {
                let observed_input = archive.get(&compare::layer_key(iteration, layer, "input"))?;
                let replay = model.layer(&observed_input.values, &weights)?;
                let actual = compare::vector(
                    &mut replays,
                    &key,
                    &replay,
                    &hidden_shape,
                    arguments.mode.label(),
                )?;
                comparisons.compare(
                    &key,
                    &actual,
                    archive.get(&key)?,
                    "archived_input_layer_replay",
                )?;
            }
            hidden = output;
        }
        let start = (model.positions - 1) * model.hidden_width;
        if iteration == 1 {
            raw_reference = hidden[start..].to_vec();
        }
        if iteration == arguments.depth {
            raw_terminal = hidden[start..].to_vec();
        }
        capture(
            &mut vectors,
            &mut comparisons,
            archive.as_ref(),
            &compare::loop_key(iteration, "pre_norm"),
            &hidden,
            &hidden_shape,
            arguments.mode,
        )?;
        let method = arguments.paper_method.as_deref();
        if !(iteration == arguments.depth && method == Some("fixed-hidden")) {
            hidden = model.normalize_loop(&hidden)?;
            loop_norm_sites += 1;
            check_bf16(&hidden);
            capture(
                &mut vectors,
                &mut comparisons,
                archive.as_ref(),
                &compare::loop_key(iteration, "post_norm"),
                &hidden,
                &hidden_shape,
                arguments.mode,
            )?;
        }
        let reference_needed =
            method.is_none() || method == Some("fixed-logits") || method == Some("adaptive-logits");
        if iteration == 1 && reference_needed {
            let logits = model.readout(&checkpoint, &hidden)?;
            check_bf16(&logits);
            capture(
                &mut vectors,
                &mut comparisons,
                archive.as_ref(),
                "reference_logits",
                &logits,
                &[49152],
                arguments.mode,
            )?;
            reference_logits = Some(logits);
            heads += 1;
        }
        if iteration == arguments.depth && method != Some("fixed-hidden") {
            let logits = model.readout(&checkpoint, &hidden)?;
            check_bf16(&logits);
            capture(
                &mut vectors,
                &mut comparisons,
                archive.as_ref(),
                "terminal_logits",
                &logits,
                &[49152],
                arguments.mode,
            )?;
            terminal_logits = Some(logits);
            heads += 1;
        }
        eprintln!(
            "oracle mode={} iteration={} layers=24 captured",
            arguments.mode.label(),
            iteration
        );
    }
    vectors.flush()?;
    replays.flush()?;
    let comparison_summary = comparisons.as_mut().map(Comparisons::finish).transpose()?;
    if archive.is_some()
        && (vectors.line != 202
            || replays.line != 96
            || comparisons
                .as_ref()
                .is_none_or(|result| result.count != 298))
    {
        return Err(io::Error::other("full oracle comparison coverage differs"));
    }
    let mut contrasts = JsonLines::create(&directory.join("contrasts.jsonl"))?;
    if let (Some(reference), Some(terminal)) = (&reference_logits, &terminal_logits) {
        for strength in [0.0, 0.5] {
            let guided = math::contrast(reference, terminal, strength)?;
            contrasts.emit(&json!({"kind":"contrast","strength":strength,"shape":[49152],"values":guided,"reference_key":"reference_logits","terminal_key":"terminal_logits"}))?;
        }
    }
    contrasts.flush()?;
    let mut method_payload = Value::Null;
    if let Some(method) = arguments.paper_method.as_deref() {
        let mut adaptive = Value::Null;
        let mut effective = if method == "baseline" {
            0.0
        } else {
            arguments.strength
        };
        let mut contrasted_hidden = None;
        let mut normalized_hidden = None;
        let guided = if method == "fixed-hidden" {
            let contrasted = math::contrast(&raw_reference, &raw_terminal, effective)?;
            let normalized = model.normalize_hidden(&contrasted)?;
            loop_norm_sites += 1;
            let logits = model.readout_last(&checkpoint, &normalized)?;
            contrasted_hidden = Some(contrasted);
            normalized_hidden = Some(normalized);
            heads += 1;
            logits
        } else {
            let terminal = terminal_logits
                .as_deref()
                .ok_or_else(|| io::Error::other("paper terminal logits absent"))?;
            if method == "baseline" {
                terminal.to_vec()
            } else {
                if method == "adaptive-logits" {
                    let (strength, margin) = fixtures::adaptive(terminal, arguments.strength)?;
                    effective = strength;
                    adaptive = margin;
                }
                math::contrast(
                    reference_logits
                        .as_deref()
                        .ok_or_else(|| io::Error::other("paper reference logits absent"))?,
                    terminal,
                    effective,
                )?
            }
        };
        let selected = verify::pick(&guided)?;
        method_payload = json!({"kind":"paper_selected_output","method":method,"context":context,"strength":if method=="baseline"{None}else{Some(arguments.strength)},"reference_logits":reference_logits,"terminal_logits":terminal_logits,"guided_logits":guided,"raw_reference_hidden":if method=="fixed-hidden" {Some(&raw_reference)} else {None},"raw_terminal_hidden":if method=="fixed-hidden" {Some(&raw_terminal)} else {None},"contrasted_hidden":contrasted_hidden,"normalized_hidden":normalized_hidden,"adaptive":adaptive,"effective_strength":if method=="baseline"{None}else{Some(effective)},"head_executions":heads,"layer_executions":arguments.depth*24,"branch_norm_sites":arguments.depth*24*4,"loop_norm_sites":loop_norm_sites,"selected_id":selected,"selected_bytes":checkpoint.vocab.token_bytes(selected)});
        let mut writer = JsonLines::create(&directory.join("paper-output.jsonl"))?;
        writer.emit(&method_payload)?;
        writer.flush()?;
    }
    let dtype_sites = if arguments.mode == Mode::Bf16 {
        dtype_table(directory)?
    } else {
        0
    };
    if bf16_violations != 0 {
        return Err(io::Error::other(
            "stored bf16 activation low-bit violations",
        ));
    }
    Ok(
        json!({"context":context,"iterations":arguments.depth,"layer_executions":arguments.depth*24,"branch_norm_sites":arguments.depth*24*4,"loop_norm_sites":loop_norm_sites,"heads":heads,"boundary_vectors":vectors.line,"replay_vectors":replays.line,"comparisons":comparison_summary,"nonfinite_coordinates":0,"bf16_low_bit_violations":bf16_violations,"dtype_site_records":dtype_sites,"paper_output":if method_payload.is_null(){Value::Null}else{json!({"method":method_payload["method"],"selected_id":method_payload["selected_id"],"effective_strength":method_payload["effective_strength"]})},"numerical_policy_status":"preregistered diagnostic tolerance; author runtime bit equality unmeasured"}),
    )
}

fn main() -> io::Result<()> {
    let arguments = Arguments::parse()?;
    if let Some(directory) = &arguments.verify_artifacts {
        return verify::artifacts(directory);
    }
    if let Some(directory) = &arguments.verify_contrast {
        return verify::contrasts(directory);
    }
    let directory = arguments.output()?;
    output_directory(directory)?;
    let summary = if arguments.fixtures_only {
        let mut summary = fixtures::run(directory, arguments.negative_controls)?;
        summary["paper_method_fixtures"] = fixtures::method_fixtures(directory)?;
        summary
    } else if let Some(archive) = &arguments.scope_controls {
        verify::scope_controls(archive, &arguments.prefix, directory)?
    } else if arguments.audit {
        verify::independence(directory)?
    } else if let Some((f32_directory, bf16_directory)) = &arguments.compare_modes {
        verify::modes(f32_directory, bf16_directory, directory)?
    } else if let Some(candidate) = &arguments.compare_paper {
        verify::paper(
            candidate,
            arguments
                .reference_oracle
                .as_deref()
                .ok_or_else(|| io::Error::other("--reference-oracle required"))?,
            directory,
        )?
    } else {
        run(&arguments, directory)?
    };
    save(&directory.join("summary.json"), &summary)?;
    println!(
        "{}",
        serde_json::to_string(&summary).map_err(io::Error::other)?
    );
    let vector_exceeding = if arguments.compare.is_some() {
        summary["comparisons"]["exceeding_coordinates"]
            .as_u64()
            .unwrap_or(0)
    } else if arguments.compare_paper.is_some() {
        summary["exceeding_coordinates"].as_u64().unwrap_or(0)
    } else {
        0
    };
    let scalar_exceeding = if arguments.compare_paper.is_some() {
        summary["scalar_exceeding"].as_u64().unwrap_or(0)
    } else {
        0
    };
    let selection_differences = if arguments.compare_paper.is_some() {
        summary["selection_differences"].as_u64().unwrap_or(0)
    } else {
        0
    };
    if vector_exceeding != 0 || scalar_exceeding != 0 || selection_differences != 0 {
        return Err(io::Error::other(format!(
            "preregistered comparison differs: vectors={vector_exceeding}, scalars={scalar_exceeding}, selections={selection_differences}; complete artifacts retained"
        )));
    }
    Ok(())
}
