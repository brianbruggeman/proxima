#![cfg(feature = "std")]

use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use futures::executor::block_on;
use proxima_primitives::pipe::{ProximaError, SendPipe};
use proxima_process::capabilities::CapFilesystem;
use proxima_process::host_grounds::{FixedClock, HostRead, HostWrite, OsEntropy, RealClock};
use proxima_process::protocol::{ChildRequest, ChildResponse, ReadResponse, WriteResponse};

fn read_request(max_bytes: u32, offset: u64) -> ChildRequest {
    ChildRequest::Read {
        handle: 7,
        max_bytes,
        offset,
    }
}

fn bytes(response: ChildResponse) -> ReadResponse {
    match response {
        ChildResponse::Read(response) => response,
        other => panic!("expected actual read bytes, got {other:?}"),
    }
}

#[test]
fn host_ground_filesystem_payloads() {
    let directory = tempfile::tempdir().expect("create host-ground directory");
    let path = directory.path().join("café-日本.txt");
    let capability = CapFilesystem::grant();
    let writer = HostWrite::new(&path, &capability);
    for payload in [b"proxima host ".as_slice(), b"append payload".as_slice()] {
        let response = block_on(writer.call(ChildRequest::Write {
            handle: 7,
            bytes: payload.to_vec(),
        }))
        .expect("write actual host bytes");
        assert_eq!(
            response,
            ChildResponse::Write(WriteResponse {
                bytes_written: payload.len() as u32
            })
        );
    }
    let expected = b"proxima host append payload";
    assert_eq!(fs::read(&path).expect("independent std read"), expected);
    let reader = HostRead::new(&path, &capability);
    let prefix = bytes(block_on(reader.call(read_request(4, 8))).expect("read bounded offset"));
    assert_eq!(prefix.bytes, b"host");
    assert!(!prefix.eof);
    let tail = bytes(block_on(reader.call(read_request(128, 13))).expect("read final slice"));
    assert_eq!(tail.bytes, b"append payload");
    assert!(tail.eof);
    let empty =
        bytes(block_on(reader.call(read_request(10, u32::MAX as u64))).expect("read beyond EOF"));
    assert!(empty.bytes.is_empty());
    assert!(empty.eof);
    let metadata = block_on(reader.call(ChildRequest::Stat { handle: 7 })).expect("stat response");
    #[cfg(unix)]
    assert!(
        matches!(metadata, ChildResponse::Stat { size, is_directory: false, .. } if size == expected.len() as u64)
    );
    #[cfg(windows)]
    assert_eq!(
        metadata,
        ChildResponse::Error {
            errno: libc::ENOSYS
        }
    );
}

#[test]
fn host_ground_filesystem_errors() {
    let directory = tempfile::tempdir().expect("create error fixture directory");
    let capability = CapFilesystem::grant();
    let missing = directory.path().join("missing.txt");
    let oracle = fs::File::open(&missing).expect_err("native missing-file error");
    let reader = HostRead::new(&missing, &capability);
    match block_on(reader.call(read_request(32, 0))).expect_err("host missing-file error") {
        ProximaError::Io(error) => {
            assert_eq!(error.kind(), oracle.kind());
            assert_eq!(error.raw_os_error(), oracle.raw_os_error());
        }
        other => panic!("expected retained OS error, got {other:?}"),
    }
    let writer = HostWrite::new(directory.path(), &capability);
    assert!(
        block_on(writer.call(ChildRequest::Write {
            handle: 7,
            bytes: b"must not acknowledge".to_vec()
        }))
        .is_err()
    );
    assert_eq!(
        block_on(reader.call(ChildRequest::Write {
            handle: 7,
            bytes: b"denied".to_vec()
        }))
        .expect("read-only response"),
        ChildResponse::Error { errno: libc::EROFS }
    );
    assert_eq!(
        block_on(writer.call(read_request(10, 0))).expect("unsupported response"),
        ChildResponse::Error {
            errno: libc::ENOSYS
        }
    );
}

#[test]
fn host_ground_entropy_reads_os() {
    let source = OsEntropy::new();
    let first = bytes(block_on(source.call(read_request(256, 0))).expect("first OS entropy"));
    let second = bytes(block_on(source.call(read_request(256, 0))).expect("second OS entropy"));
    assert_eq!(first.bytes.len(), 256);
    assert_eq!(second.bytes.len(), 256);
    assert!(!first.eof);
    assert!(
        first.bytes.iter().any(|byte| *byte != 0),
        "reject the zero-buffer stub"
    );
    assert_ne!(first.bytes, second.bytes, "reject a canned repeated buffer");
    assert_eq!(
        block_on(source.call(ChildRequest::Write {
            handle: 7,
            bytes: vec![1]
        }))
        .expect("read-only entropy response"),
        ChildResponse::Error { errno: libc::EROFS }
    );
}

#[test]
fn host_ground_clock_epoch_and_slices() {
    let before = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("wall clock epoch")
        .as_secs();
    let observed =
        bytes(block_on(RealClock::new().call(read_request(32, 0))).expect("real clock read"));
    let after = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("wall clock epoch")
        .as_secs();
    let seconds: u64 = std::str::from_utf8(&observed.bytes)
        .expect("decimal clock UTF-8")
        .parse()
        .expect("decimal clock seconds");
    assert!((before..=after).contains(&seconds));
    assert!(observed.eof);
    let fixed = FixedClock::new(1_700_000_123);
    let prefix = bytes(block_on(fixed.call(read_request(4, 0))).expect("fixed prefix"));
    assert_eq!(prefix.bytes, b"1700");
    assert!(!prefix.eof);
    let tail = bytes(block_on(fixed.call(read_request(32, 7))).expect("fixed tail"));
    assert_eq!(tail.bytes, b"123");
    assert!(tail.eof);
    let empty = bytes(block_on(fixed.call(read_request(32, u64::MAX))).expect("fixed beyond EOF"));
    assert!(empty.bytes.is_empty());
    assert!(empty.eof);
}
