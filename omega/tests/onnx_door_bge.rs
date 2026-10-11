//! The real-model fixtures of the ONNX door: the pinned bge-small file and the 64-sentence corpus.
//! The model is 133 MB and is never vendored; the repository holds its first 4096 bytes and its sha256.

#![cfg(feature = "onnx-door")]

#[path = "support/bge.rs"]
mod bge;

use anyhow::{Result, ensure};

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

const EMBEDDING_DIMENSION: usize = 384;
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
