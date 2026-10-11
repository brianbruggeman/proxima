//! The real-model fixtures of the ONNX door: the pinned bge-small file and the 64-sentence corpus.
//! The model is 133 MB and is never vendored; the repository holds its first 4096 bytes and its sha256.

#![cfg(feature = "onnx-door")]

#[path = "support/bge.rs"]
mod bge;

use anyhow::{Context, Result, anyhow, ensure};
use omega::onnx_door::{DoorBatch, DoorConfig, OnnxDoor};
use proxima_primitives::pipe::Pipe;

use bge::RecordBatch;

const MODEL_BYTES: usize = 133_093_490;
const MODEL_SHA256: &str = "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35";
const HEADER_BYTES: usize = 4096;
const CORPUS_ROWS: usize = 64;
const MAX_TOKENS: usize = 64;
const CLS_ID: i64 = 101;
const SEP_ID: i64 = 102;

const BGE_EVAL_ROWS: [(&str, &[i64]); 3] = [
    (
        "the cat sat on the mat",
        &[101, 1996, 4937, 2938, 2006, 1996, 13523, 102],
    ),
    (
        "a cat is sitting on a mat",
        &[101, 1037, 4937, 2003, 3564, 2006, 1037, 13523, 102],
    ),
    (
        "quantum physics explains atomic energy",
        &[101, 8559, 5584, 7607, 9593, 2943, 102],
    ),
];

#[test]
fn bge_fixture_model_file_length_is_133093490() -> Result<()> {
    let model = bge::read_model()?;

    ensure!(
        model.len() == MODEL_BYTES,
        "model is {} bytes, expected {MODEL_BYTES}",
        model.len()
    );
    Ok(())
}

#[test]
fn bge_fixture_model_file_sha256_equals_the_recorded_hash() -> Result<()> {
    let model = bge::read_model()?;

    let computed = bge::sha256_hex(&model);

    ensure!(
        computed == bge::recorded_sha256()?,
        "model sha256 {computed} differs from model.sha256"
    );
    ensure!(
        computed == MODEL_SHA256,
        "model sha256 {computed} differs from the spec hash {MODEL_SHA256}"
    );
    Ok(())
}

#[test]
fn bge_fixture_altered_byte_changes_the_sha256() -> Result<()> {
    let mut model = bge::read_model()?;
    model[0] ^= 0x01;

    let computed = bge::sha256_hex(&model);

    ensure!(
        computed != bge::recorded_sha256()?,
        "an altered first byte kept the recorded sha256"
    );
    Ok(())
}

#[test]
fn bge_fixture_vendored_header_equals_the_first_4096_bytes_of_the_model() -> Result<()> {
    let model = bge::read_model()?;
    let header = bge::read_fixture("model_header.bin")?;

    ensure!(
        header.len() == HEADER_BYTES,
        "header is {} bytes, expected {HEADER_BYTES}",
        header.len()
    );
    ensure!(
        header == model[..HEADER_BYTES],
        "vendored header differs from the model's first {HEADER_BYTES} bytes"
    );
    Ok(())
}

#[test]
fn bge_fixture_corpus_has_64_sentences() -> Result<()> {
    let corpus = bge::read_corpus()?;

    ensure!(
        corpus.len() == CORPUS_ROWS,
        "corpus has {} rows, expected {CORPUS_ROWS}",
        corpus.len()
    );
    ensure!(
        corpus.iter().all(|row| !row.text.trim().is_empty()),
        "a corpus row has empty text"
    );
    Ok(())
}

#[test]
fn bge_fixture_corpus_ids_start_101_end_102_and_fit_64_tokens() -> Result<()> {
    let corpus = bge::read_corpus()?;

    for row in &corpus {
        ensure!(
            row.ids.first() == Some(&CLS_ID),
            "row {:?} does not start with 101",
            row.text
        );
        ensure!(
            row.ids.last() == Some(&SEP_ID),
            "row {:?} does not end with 102",
            row.text
        );
        ensure!(
            row.ids.len() <= MAX_TOKENS,
            "row {:?} has {} tokens",
            row.text,
            row.ids.len()
        );
    }
    Ok(())
}

#[test]
fn bge_fixture_first_three_corpus_rows_equal_the_hard_coded_id_arrays_of_bge_eval_rs() -> Result<()>
{
    let corpus = bge::read_corpus()?;

    for (row, (text, ids)) in corpus.iter().zip(BGE_EVAL_ROWS) {
        ensure!(
            row.text == text,
            "row text {:?} differs from {text:?}",
            row.text
        );
        ensure!(
            row.ids == ids,
            "row {text:?} ids {:?} differ from {ids:?}",
            row.ids
        );
    }
    Ok(())
}

const EMBEDDING_DIMENSION: usize = bge::EMBEDDING_DIMENSION;
const ONNXRUNTIME_COSINES: [f64; 3] = [0.936_311, 0.378_777, 0.334_176];
const COSINE_TOLERANCE: f64 = 1e-4;
const PADDING_TOLERANCE: f32 = 1e-5;

fn record_rows(cell: &str) -> Result<Vec<Vec<f32>>> {
    let record = bge::read_record(cell)?;
    Ok(record
        .batches
        .into_iter()
        .flat_map(|batch| batch.cls)
        .collect())
}

fn recorded_cosines() -> Result<[f64; 3]> {
    let rows = record_rows("b1")?;
    Ok([
        bge::cosine(&rows[0], &rows[1]),
        bge::cosine(&rows[0], &rows[2]),
        bge::cosine(&rows[1], &rows[2]),
    ])
}

#[test]
fn tract_record_b1_record_has_64_rows_of_384_finite_floats() -> Result<()> {
    let rows = record_rows("b1")?;

    ensure!(
        rows.len() == CORPUS_ROWS,
        "b1 record has {} rows",
        rows.len()
    );
    ensure!(
        rows.iter().all(|row| row.len() == EMBEDDING_DIMENSION),
        "a b1 row is not {EMBEDDING_DIMENSION} floats"
    );
    ensure!(
        rows.iter().flatten().all(|value| value.is_finite()),
        "a b1 value is not finite"
    );
    Ok(())
}

#[test]
fn tract_record_b32_record_has_2_batches_of_32_rows_of_384() -> Result<()> {
    let record = bge::read_record("b32")?;

    ensure!(
        record.batches.len() == 2,
        "b32 record has {} batches",
        record.batches.len()
    );
    for batch in &record.batches {
        ensure!(batch.cls.len() == 32, "batch has {} rows", batch.cls.len());
        ensure!(
            batch.cls.iter().all(|row| row.len() == EMBEDDING_DIMENSION),
            "a b32 row is not {EMBEDDING_DIMENSION} floats"
        );
        ensure!(
            batch.input_ids.iter().all(|row| row.len() == batch.seq)
                && batch
                    .attention_mask
                    .iter()
                    .all(|row| row.len() == batch.seq)
                && batch
                    .token_type_ids
                    .iter()
                    .all(|row| row.len() == batch.seq),
            "an input tensor row is not seq {} wide",
            batch.seq
        );
    }
    Ok(())
}

#[test]
fn tract_record_header_names_tract_onnx_0_22_4_and_the_model_sha256() -> Result<()> {
    for header in [
        bge::read_record("b1")?.header,
        bge::read_record("b32")?.header,
    ] {
        ensure!(
            header["tract_onnx"] == "0.22.4",
            "header names tract {}",
            header["tract_onnx"]
        );
        ensure!(
            header["model_sha256"] == MODEL_SHA256,
            "header names model {}",
            header["model_sha256"]
        );
    }
    Ok(())
}

#[test]
fn tract_record_padding_invariance_each_b32_row_equals_its_b1_row_within_1e_5_after_normalization()
-> Result<()> {
    let single = record_rows("b1")?;
    let batched = record_rows("b32")?;

    ensure!(single.len() == batched.len(), "row counts differ");
    for (index, (single_row, batched_row)) in single.iter().zip(&batched).enumerate() {
        let difference = bge::max_abs_difference(
            &bge::l2_normalize(single_row),
            &bge::l2_normalize(batched_row),
        );
        ensure!(
            difference <= PADDING_TOLERANCE,
            "row {index} differs by {difference} between b1 and b32"
        );
    }
    Ok(())
}

#[test]
fn tract_record_three_sentence_cosines_equal_the_recorded_onnxruntime_values_0_936311_0_378777_0_334176_within_1e_4()
-> Result<()> {
    let cosines = recorded_cosines()?;

    ensure!(
        bge::all_within(&cosines, &ONNXRUNTIME_COSINES, COSINE_TOLERANCE),
        "tract cosines {cosines:?} differ from onnxruntime {ONNXRUNTIME_COSINES:?}"
    );
    Ok(())
}

#[test]
fn tract_record_identity_control_the_records_cosines_compared_with_themselves_pass() -> Result<()> {
    let cosines = recorded_cosines()?;

    ensure!(
        bge::all_within(&cosines, &cosines, COSINE_TOLERANCE),
        "the record's cosines differ from themselves"
    );
    Ok(())
}

#[test]
fn tract_record_degenerate_control_cosine_a_b_altered_by_1e_3_fails() -> Result<()> {
    let cosines = recorded_cosines()?;
    let mut altered = ONNXRUNTIME_COSINES;
    altered[0] += 1e-3;

    ensure!(
        !bge::all_within(&cosines, &altered, COSINE_TOLERANCE),
        "an alteration of 1e-3 stayed within {COSINE_TOLERANCE}"
    );
    Ok(())
}

const CPU_TOLERANCE: f32 = 1e-5;
const METAL_TOLERANCE: f32 = 1e-3;
const OUTPUT_NAME: &str = "last_hidden_state";
const UNIT_NORM_TOLERANCE: f32 = 1e-6;

#[derive(Clone, Copy)]
enum MaskPolicy {
    Recorded,
    PadPositionsUnmasked,
}

#[derive(Clone, Copy)]
enum Target {
    Cpu,
    Metal,
}

impl Target {
    fn label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
        }
    }

    fn config(self) -> DoorConfig {
        match self {
            Self::Cpu => DoorConfig::default(),
            Self::Metal => DoorConfig::builder()
                .engine("gpu")
                .gpu_driver("metal")
                .build(),
        }
    }
}

fn build_door(target: Target) -> Result<OnnxDoor> {
    OnnxDoor::from_bytes(bge::read_model()?, &target.config())
        .map_err(|error| anyhow!("{} door config: {error}", target.label()))
}

fn door_batch(batch: &RecordBatch, policy: MaskPolicy) -> DoorBatch {
    let mask = match policy {
        MaskPolicy::Recorded => bge::flatten_as_float(&batch.attention_mask),
        MaskPolicy::PadPositionsUnmasked => vec![1.0; batch.input_ids.len() * batch.seq],
    };
    DoorBatch {
        pins: [
            ("batch_size".to_owned(), batch.input_ids.len() as u64),
            ("sequence_length".to_owned(), batch.seq as u64),
        ]
        .into(),
        inputs: vec![
            (
                "input_ids".to_owned(),
                bge::flatten_as_float(&batch.input_ids),
            ),
            ("attention_mask".to_owned(), mask),
            (
                "token_type_ids".to_owned(),
                bge::flatten_as_float(&batch.token_type_ids),
            ),
        ],
    }
}

async fn door_cls_rows(
    door: &OnnxDoor,
    batch: &RecordBatch,
    policy: MaskPolicy,
) -> Result<Vec<Vec<f32>>> {
    let rows = batch.input_ids.len();
    let output = door
        .call(door_batch(batch, policy))
        .await
        .map_err(|error| anyhow!("door call at batch {rows} seq {}: {error}", batch.seq))?;
    let (values, shape) = output
        .get(OUTPUT_NAME)
        .with_context(|| format!("door output has no {OUTPUT_NAME}"))?;
    ensure!(
        shape == [rows as u64, batch.seq as u64, EMBEDDING_DIMENSION as u64],
        "{OUTPUT_NAME} has shape {shape:?}"
    );
    bge::cls_rows(values, rows)
}

async fn door_cls_for_cell(
    target: Target,
    cell: &str,
    policy: MaskPolicy,
) -> Result<Vec<Vec<f32>>> {
    let door = build_door(target)?;
    let mut rows = Vec::new();
    for batch in &bge::read_record(cell)?.batches {
        rows.extend(door_cls_rows(&door, batch, policy).await?);
    }
    Ok(rows)
}

fn normalized(rows: &[Vec<f32>]) -> Vec<Vec<f32>> {
    rows.iter().map(|row| bge::l2_normalize(row)).collect()
}

fn recorded_embeddings(cell: &str) -> Result<Vec<Vec<f32>>> {
    Ok(normalized(&record_rows(cell)?))
}

fn report_differences(target: Target, cell: &str, differences: &[f32]) {
    for (sentence, difference) in differences.iter().enumerate() {
        println!(
            "bge_{}_parity cell={cell} sentence={sentence} max_abs={difference:e}",
            target.label()
        );
    }
}

fn ensure_within(cell: &str, differences: &[f32], tolerance: f32) -> Result<()> {
    let exceeding: Vec<String> = differences
        .iter()
        .enumerate()
        .filter(|(_, difference)| difference.is_nan() || **difference > tolerance)
        .map(|(sentence, difference)| format!("{sentence}:{difference:e}"))
        .collect();
    ensure!(
        exceeding.is_empty(),
        "cell {cell}: {} of {} sentences exceed {tolerance:e}: {exceeding:?}",
        exceeding.len(),
        differences.len()
    );
    Ok(())
}

async fn parity_against_tract(target: Target, cell: &str, tolerance: f32) -> Result<()> {
    let rows = door_cls_for_cell(target, cell, MaskPolicy::Recorded).await?;
    let differences = bge::sentence_differences(&normalized(&rows), &recorded_embeddings(cell)?)?;
    ensure!(
        differences.len() == CORPUS_ROWS,
        "{} sentences compared",
        differences.len()
    );
    report_differences(target, cell, &differences);
    ensure_within(cell, &differences, tolerance)
}

fn identity_control(tolerance: f32) -> Result<()> {
    for cell in ["b1", "b32"] {
        let record = recorded_embeddings(cell)?;

        let differences = bge::sentence_differences(&record, &record)?;

        ensure!(
            differences.len() == CORPUS_ROWS,
            "{cell} compared {} rows",
            differences.len()
        );
        ensure_within(cell, &differences, tolerance)?;
    }
    Ok(())
}

fn degenerate_control(alteration: f32, tolerance: f32) -> Result<()> {
    let record = recorded_embeddings("b1")?;
    let mut altered = record.clone();
    altered[17][200] += alteration;

    let differences = bge::sentence_differences(&altered, &record)?;

    ensure!(
        ensure_within("b1", &differences, tolerance).is_err(),
        "an element altered by {alteration:e} stayed within {tolerance:e}"
    );
    Ok(())
}

async fn mask_leak_control(target: Target, tolerance: f32) -> Result<()> {
    let rows = door_cls_for_cell(target, "b32", MaskPolicy::PadPositionsUnmasked).await?;

    let differences = bge::sentence_differences(&normalized(&rows), &recorded_embeddings("b32")?)?;

    let exceeding = differences
        .iter()
        .filter(|difference| **difference > tolerance)
        .count();
    println!(
        "bge_{}_parity mask_leak sentences_exceeding={exceeding}",
        target.label()
    );
    ensure!(
        exceeding >= 1,
        "unmasked pad positions left every sentence within {tolerance:e}"
    );
    Ok(())
}

async fn finite_unit_norm_check(target: Target) -> Result<()> {
    let raw = door_cls_for_cell(target, "b32", MaskPolicy::Recorded).await?;

    ensure!(
        raw.iter().flatten().all(|value| value.is_finite()),
        "a door output value is not finite"
    );
    for (sentence, row) in normalized(&raw).iter().enumerate() {
        let deviation = (bge::norm(row) - 1.0).abs();
        ensure!(
            deviation <= UNIT_NORM_TOLERANCE,
            "sentence {sentence} norm deviates from 1 by {deviation:e}"
        );
    }
    Ok(())
}

#[proxima::test]
async fn bge_cpu_parity_b1_all_64_sentences_within_1e_5_of_tract() -> Result<()> {
    parity_against_tract(Target::Cpu, "b1", CPU_TOLERANCE).await
}

#[proxima::test]
async fn bge_cpu_parity_b32_all_64_sentences_within_1e_5_of_tract() -> Result<()> {
    parity_against_tract(Target::Cpu, "b32", CPU_TOLERANCE).await
}

#[test]
fn bge_cpu_parity_identity_control_tract_record_compared_with_itself_passes() -> Result<()> {
    identity_control(CPU_TOLERANCE)
}

#[test]
fn bge_cpu_parity_degenerate_control_one_element_altered_by_1e_3_fails() -> Result<()> {
    degenerate_control(1e-3, CPU_TOLERANCE)
}

#[proxima::test]
async fn bge_cpu_parity_mask_leak_control_pad_positions_unmasked_exceed_the_tolerance() -> Result<()>
{
    mask_leak_control(Target::Cpu, CPU_TOLERANCE).await
}

#[proxima::test]
async fn bge_cpu_parity_outputs_are_finite_and_unit_norm() -> Result<()> {
    finite_unit_norm_check(Target::Cpu).await
}

#[cfg(feature = "metal")]
#[proxima::test]
async fn bge_metal_parity_b1_all_64_sentences_within_1e_3_of_tract() -> Result<()> {
    parity_against_tract(Target::Metal, "b1", METAL_TOLERANCE).await
}

#[cfg(feature = "metal")]
#[proxima::test]
async fn bge_metal_parity_b32_all_64_sentences_within_1e_3_of_tract() -> Result<()> {
    parity_against_tract(Target::Metal, "b32", METAL_TOLERANCE).await
}

#[cfg(feature = "metal")]
#[test]
fn bge_metal_parity_identity_control_tract_record_compared_with_itself_passes() -> Result<()> {
    identity_control(METAL_TOLERANCE)
}

#[cfg(feature = "metal")]
#[test]
fn bge_metal_parity_degenerate_control_one_element_altered_by_1e_2_fails() -> Result<()> {
    degenerate_control(1e-2, METAL_TOLERANCE)
}

#[cfg(feature = "metal")]
#[proxima::test]
async fn bge_metal_parity_mask_leak_control_pad_positions_unmasked_exceed_the_tolerance()
-> Result<()> {
    mask_leak_control(Target::Metal, METAL_TOLERANCE).await
}

#[cfg(feature = "metal")]
#[proxima::test]
async fn bge_metal_parity_outputs_are_finite_and_unit_norm() -> Result<()> {
    finite_unit_norm_check(Target::Metal).await
}

#[cfg(feature = "metal")]
struct DeviceRun {
    allocated_bytes: u64,
    decoded_bytes: u64,
    nocopy_bytes: u64,
    copied_bytes: u64,
}

#[cfg(feature = "metal")]
async fn device_run_of_first_b1_sentence(target: Target) -> Result<DeviceRun> {
    let record = bge::read_record("b1")?;
    let batch = record.batches.first().context("b1 record has no batch")?;
    let allocated_before = omega::metal::current_allocated_size().unwrap_or(0);
    let nocopy_before = omega::metal::BLOCK_NOCOPY_BOUND_BYTES.get();
    let copied_before = omega::metal::BLOCK_COPIED_BYTES.get();
    let door = build_door(target)?;

    door_cls_rows(&door, batch, MaskPolicy::Recorded).await?;

    Ok(DeviceRun {
        allocated_bytes: omega::metal::current_allocated_size()
            .unwrap_or(0)
            .saturating_sub(allocated_before),
        decoded_bytes: door.stats().decoded_bytes,
        nocopy_bytes: omega::metal::BLOCK_NOCOPY_BOUND_BYTES.get() - nocopy_before,
        copied_bytes: omega::metal::BLOCK_COPIED_BYTES.get() - copied_before,
    })
}

#[cfg(feature = "metal")]
#[proxima::test]
async fn bge_metal_parity_gpu_ran_control_metal_binds_every_decoded_byte_and_cpu_binds_none()
-> Result<()> {
    let cpu = device_run_of_first_b1_sentence(Target::Cpu).await?;
    let metal = device_run_of_first_b1_sentence(Target::Metal).await?;

    for (label, run) in [("cpu", &cpu), ("metal", &metal)] {
        println!(
            "bge_metal_parity gpu_ran {label} allocated={} decoded={} nocopy_bytes={} copied_bytes={}",
            run.allocated_bytes, run.decoded_bytes, run.nocopy_bytes, run.copied_bytes
        );
    }
    ensure!(
        cpu.nocopy_bytes + cpu.copied_bytes == 0,
        "the cpu door bound {} bytes to metal buffers",
        cpu.nocopy_bytes + cpu.copied_bytes
    );
    ensure!(
        metal.nocopy_bytes + metal.copied_bytes >= metal.decoded_bytes,
        "metal bound {} bytes, below the {} decoded initializer bytes",
        metal.nocopy_bytes + metal.copied_bytes,
        metal.decoded_bytes
    );
    ensure!(
        metal.allocated_bytes > 0,
        "metal reports no allocated bytes"
    );
    Ok(())
}
