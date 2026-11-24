#![allow(clippy::expect_used)]

use proxima_gguf::parse_complete;

use super::chunked_prefill_tests::dense_checkpoint;
use super::warm_load::{LoadWarm, warm_chunk_rows};
use super::{LoadedModel, ServingConfig};
use crate::serving::ContextLength;

const UBATCH_ROWS: u32 = 24;

fn config(context_length: ContextLength) -> ServingConfig<'static> {
    ServingConfig {
        gpu_layers: 0,
        ubatch_size: UBATCH_ROWS,
        context_length,
        temperature: 0.0,
        ..ServingConfig::default()
    }
}

#[test]
fn warm_at_load_declares_no_buffers_and_warms_two_chunks_on_the_cpu_engine() {
    let bytes = dense_checkpoint();
    let parsed = parse_complete(&bytes).expect("parse the dense fixture");
    let model = LoadedModel::load(&parsed, &bytes).expect("load the dense fixture");

    let warm = model.warm_at_load(&config(ContextLength::Within(128))).expect("warm at load");

    let chunk = UBATCH_ROWS as usize;
    assert_eq!(warm, LoadWarm { buffers: 0, rows: 2 * chunk });
}

#[test]
fn the_warm_chunk_shrinks_to_leave_room_for_two_chunks_in_a_short_context() {
    assert_eq!(warm_chunk_rows(&config(ContextLength::Within(20))), 9);
    assert_eq!(warm_chunk_rows(&config(ContextLength::Within(128))), UBATCH_ROWS as usize);
    assert_eq!(warm_chunk_rows(&config(ContextLength::Native)), 11);
}
