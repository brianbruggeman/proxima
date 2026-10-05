#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_model_interop::InteropError;
use proxima_model_interop::block_file::{BlockFileHeader, BlockFileLayer, encode_block};

#[path = "support/block_write.rs"]
mod block_write;

use block_write::write_block_file;

fn fixture_bytes() -> Vec<u8> {
    let header = BlockFileHeader {
        descriptor_digest: [0x11; 16],
        content_key: 0x0102030405060708,
        base_position: 0,
        layers: vec![BlockFileLayer {
            rows: 2,
            k_even_row_bytes: 8,
            k_odd_row_bytes: 8,
            v_row_bytes: 4,
            ring_window: 0,
            ring_capacity: 0,
        }],
    };
    let k_even = [0.0_f32, 0.5, 1.0, 1.5];
    let k_odd = [100.0_f32, 100.5, 101.0, 101.5];
    let v = [200.0_f32, 200.5];
    let mut bytes = Vec::new();
    encode_block(&header, &[&k_even, &k_odd, &v], &mut bytes).expect("encode the fixture");
    bytes
}

#[test]
fn blockfile_write_creates_the_final_name_only() {
    let bytes = fixture_bytes();
    let dir = tempfile::tempdir().expect("create a temp directory");

    let written = write_block_file(dir.path(), 0x0102030405060708, &bytes).expect("write the block file");

    assert_eq!(written.file_name().unwrap(), "0102030405060708.pxkv");
    assert_eq!(std::fs::read(&written).expect("read the block file back"), bytes);
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);

    let absent = dir.path().join("absent");
    let failure = write_block_file(&absent, 1, &bytes);
    assert!(matches!(
        failure,
        Err(InteropError::BlockFileIo { ref path, .. }) if path.ends_with("0000000000000001.pxkv.tmp")
    ));
}
