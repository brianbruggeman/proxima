//! Family inference over a real checkpoint header (PROXNUM-610 S1, PROXNUM-611).
//!
//! `PROXIMA_WHISPER_DIR` names the tiny checkpoint directory; a missing
//! fixture fails the test, it never skips.

#![cfg(feature = "toml")]

use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use proxima_model_interop::families::{
    CheckpointSha, FamilyTemplate, GoldenRecord, InferError, Inferred, Refusal, Source,
    TensorShape, Value, accept, infer, parse_ledger,
};
use proxima_safetensors::SafetensorsParser;

const TINY_SHA: &str = "7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395";

fn fixture_dir() -> Result<PathBuf> {
    let dir = std::env::var("PROXIMA_WHISPER_DIR")
        .context("PROXIMA_WHISPER_DIR must name the tiny checkpoint directory")?;
    Ok(PathBuf::from(dir))
}

fn tiny_tensors() -> Result<Vec<TensorShape>> {
    let mut file = File::open(fixture_dir()?.join("model.safetensors")).context("tiny weights")?;
    let mut prefix = [0_u8; 8];
    file.read_exact(&mut prefix)?;
    let mut header = prefix.to_vec();
    header.resize(8 + usize::try_from(u64::from_le_bytes(prefix))?, 0);
    file.read_exact(&mut header[8..])?;
    let mut parser = SafetensorsParser::new();
    parser.feed(&header);
    parser
        .poll()
        .context("production parser reads the header")?;
    let SafetensorsParser::TensorData { manifest, .. } = parser else {
        bail!("header did not complete");
    };
    Ok(manifest.tensors.iter().map(TensorShape::from).collect())
}

fn tiny_config() -> Result<serde_json::Value> {
    let text = std::fs::read_to_string(fixture_dir()?.join("config.json"))?;
    Ok(serde_json::from_str(&text)?)
}
/// The outer `Result` is fixture trouble; the inner one is the inference outcome under test.
fn infer_with(
    tensors: &[TensorShape],
    config: &serde_json::Value,
) -> Result<Result<Inferred, InferError>> {
    let templates = FamilyTemplate::builtin()?;
    let sha = CheckpointSha::parse(TINY_SHA)?;
    Ok(infer(tensors, config, &templates, sha))
}

fn int(inferred: &Inferred, key: &str) -> u64 {
    match inferred.facts.get(key).map(|fact| &fact.value) {
        Some(Value::Int(value)) => *value,
        other => panic!("{key} is not an int fact: {other:?}"),
    }
}

fn flag(inferred: &Inferred, key: &str) -> bool {
    match inferred.facts.get(key).map(|fact| &fact.value) {
        Some(Value::Flag(value)) => *value,
        other => panic!("{key} is not a flag fact: {other:?}"),
    }
}

fn scale(dim: u64) -> u64 {
    match dim {
        384 => 512,
        1536 => 2048,
        80 => 128,
        other => other,
    }
}

fn layer_zero_to(tensors: &[TensorShape], stack: &str, layers: u64) -> Vec<TensorShape> {
    let marker = format!("{stack}.layers.0.");
    tensors
        .iter()
        .filter(|tensor| tensor.name.contains(&marker))
        .flat_map(|tensor| {
            (0..layers).map(|layer| TensorShape {
                name: tensor
                    .name
                    .replace(&marker, &format!("{stack}.layers.{layer}.")),
                shape: tensor.shape.clone(),
            })
        })
        .collect()
}

fn synthetic_second_size(tiny: &[TensorShape]) -> Vec<TensorShape> {
    let outside = tiny
        .iter()
        .filter(|tensor| !tensor.name.contains(".layers."));
    let stacks = layer_zero_to(tiny, "model.encoder", 6)
        .into_iter()
        .chain(layer_zero_to(tiny, "model.decoder", 2));
    outside
        .cloned()
        .chain(stacks)
        .map(|tensor| TensorShape {
            shape: tensor.shape.iter().map(|dim| scale(*dim)).collect(),
            ..tensor
        })
        .collect()
}

mod infer {
    use super::*;

    #[test]
    fn tiny_real_header_and_config_infer_the_encoder_decoder_record() {
        let tensors = tiny_tensors().expect("tiny header");
        let config = tiny_config().expect("tiny config");
        let inferred = infer_with(&tensors, &config)
            .expect("fixtures")
            .expect("tiny infers");

        assert_eq!(inferred.family, "encoder_decoder");
        let sizes = [
            int(&inferred, "encoder_layers"),
            int(&inferred, "decoder_layers"),
            int(&inferred, "d_model"),
            int(&inferred, "mel_bins"),
            int(&inferred, "conv1d_kernel"),
        ];
        assert_eq!(sizes, [4, 4, 384, 80, 3]);
        assert!(flag(&inferred, "cross_attention"));
        assert!(!flag(&inferred, "k_proj_bias"));
        assert_eq!(
            inferred.required_stages,
            ["mel", "encoder", "logits", "tokens"]
        );
    }

    #[test]
    fn synthetic_second_size_derived_from_tiny_names_goes_through_the_same_function() {
        let tensors = synthetic_second_size(&tiny_tensors().expect("tiny header"));
        let config = serde_json::json!({
            "encoder_layers": 6, "decoder_layers": 2, "d_model": 512, "num_mel_bins": 128,
            "encoder_attention_heads": 8, "decoder_attention_heads": 8,
            "activation_function": "gelu",
        });

        let inferred = infer_with(&tensors, &config)
            .expect("fixtures")
            .expect("synthetic size infers");

        let sizes = [
            int(&inferred, "encoder_layers"),
            int(&inferred, "decoder_layers"),
            int(&inferred, "d_model"),
            int(&inferred, "mel_bins"),
        ];
        assert_eq!(sizes, [6, 2, 512, 128]);
    }

    #[test]
    fn shape_beats_config_and_a_contradiction_is_refused() {
        let tensors = tiny_tensors().expect("tiny header");
        let mut silent = tiny_config().expect("tiny config");
        silent.as_object_mut().expect("object").remove("d_model");
        let mut wrong = tiny_config().expect("tiny config");
        wrong["d_model"] = serde_json::json!(512);

        let from_shape = infer_with(&tensors, &silent)
            .expect("fixtures")
            .expect("shape alone fills d_model");
        let refused = infer_with(&tensors, &wrong)
            .expect("fixtures")
            .expect_err("config contradicts shape");

        let source = &from_shape.facts["d_model"].source;
        assert!(
            matches!(source, Source::Tensor { axis: Some(0), .. }),
            "{source:?}"
        );
        assert!(
            matches!(refused, InferError::ConfigContradicts { .. }),
            "{refused:?}"
        );
    }

    #[test]
    fn no_cross_attention_is_no_family_naming_the_unmet_pattern() {
        let decoder_only: Vec<_> = tiny_tensors()
            .expect("tiny header")
            .into_iter()
            .filter(|tensor| !tensor.name.contains("encoder_attn"))
            .collect();

        let config = tiny_config().expect("tiny config");
        let error = infer_with(&decoder_only, &config)
            .expect("fixtures")
            .expect_err("no cross-attention");

        let InferError::NoFamily { unmet } = error else {
            panic!("expected NoFamily");
        };
        assert_eq!(unmet.len(), 1);
        assert_eq!(
            unmet[0].pattern,
            "model.decoder.layers.{L}.encoder_attn.k_proj.weight"
        );
    }
}

mod accept {
    use super::*;

    fn tiny_inferred() -> Result<Inferred> {
        let tensors = tiny_tensors()?;
        let config = tiny_config()?;
        Ok(infer_with(&tensors, &config)??)
    }

    // synthetic ledger lines: real golden records arrive with S6
    fn ledger_for(inferred: &Inferred, measured: f64) -> Result<Vec<GoldenRecord>> {
        let line = |stage: &str| {
            let sha = inferred.sha.as_str();
            let digest = inferred.digest();
            format!(
                r#"{{"sha":"{sha}","digest":"{digest}","stage":"{stage}","measured":{measured:e},"ceiling":5e-4}}"#
            )
        };
        let lines: Vec<String> = inferred
            .required_stages
            .iter()
            .map(|stage| line(stage))
            .collect();
        let text = lines.join("\n");
        Ok(parse_ledger(&text)?)
    }

    #[test]
    fn missing_golden_is_refused_with_golden_missing_for_every_way_it_can_be_absent() {
        let inferred = tiny_inferred().expect("tiny infers");
        let good = ledger_for(&inferred, 1e-6).expect("synthetic ledger");
        let other_sha = GoldenRecord {
            sha: "0".repeat(64),
            ..good[0].clone()
        };
        let other_digest = GoldenRecord {
            digest: "stale".into(),
            ..good[0].clone()
        };

        let none = accept(&inferred, &[]).expect_err("no records");
        let sha = accept(&inferred, &[other_sha]).expect_err("other sha");
        let digest = accept(&inferred, &[other_digest]).expect_err("other digest");
        let stage = accept(&inferred, &good[..1]).expect_err("one stage missing");

        for refusal in [&none, &sha, &digest] {
            assert!(
                matches!(refusal, Refusal::GoldenMissing { stage: None, .. }),
                "{refusal}"
            );
        }
        let named = Some("encoder".to_string());
        assert!(
            matches!(&stage, Refusal::GoldenMissing { stage, .. } if *stage == named),
            "{stage}"
        );
    }

    #[test]
    fn failed_golden_is_refused_with_golden_failed_naming_the_stage() {
        let inferred = tiny_inferred().expect("tiny infers");
        let above = ledger_for(&inferred, 1e-2).expect("synthetic ledger");
        let mut not_a_number = ledger_for(&inferred, 0.0).expect("synthetic ledger");
        not_a_number
            .iter_mut()
            .for_each(|record| record.measured = f64::NAN);

        let over = accept(&inferred, &above).expect_err("above ceiling");
        let nan = accept(&inferred, &not_a_number).expect_err("NaN measurement");

        assert!(
            matches!(&over, Refusal::GoldenFailed { stage, .. } if stage == "mel"),
            "{over}"
        );
        assert!(
            matches!(&nan, Refusal::GoldenFailed { stage, .. } if stage == "mel"),
            "{nan}"
        );
    }

    #[test]
    fn golden_within_every_ceiling_is_accepted() {
        let inferred = tiny_inferred().expect("tiny infers");
        let records = ledger_for(&inferred, 1e-6).expect("synthetic ledger");

        let accepted = accept(&inferred, &records).expect("all stages pass");

        assert_eq!(accepted.inferred().digest(), inferred.digest());
    }

    #[test]
    fn latest_record_per_stage_is_authoritative_in_an_append_only_ledger() {
        let inferred = tiny_inferred().expect("tiny infers");
        let passing = ledger_for(&inferred, 1e-6).expect("synthetic ledger");
        let failing: Vec<_> = passing
            .iter()
            .map(|record| GoldenRecord {
                measured: 1e-2,
                ..record.clone()
            })
            .collect();
        let pass_then_fail = [passing.clone(), failing.clone()].concat();
        let fail_then_pass = [failing, passing].concat();

        let revoked = accept(&inferred, &pass_then_fail).expect_err("a later failure revokes");
        let recovered = accept(&inferred, &fail_then_pass).expect("a later pass supersedes");

        assert!(matches!(revoked, Refusal::GoldenFailed { .. }), "{revoked}");
        assert_eq!(recovered.inferred().digest(), inferred.digest());
    }
}
