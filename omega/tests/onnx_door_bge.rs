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
