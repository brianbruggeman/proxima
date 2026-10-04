#![cfg(all(feature = "std", feature = "runtime-prime-bgpool"))]
// expect() with a message is the test-edge failure report; production code stays denied
#![allow(clippy::expect_used)]

use std::fs::{self, File, OpenOptions};
use std::future::Future;
use std::io::{self, Read, Seek, Write};
#[cfg(unix)]
use std::os::fd::OwnedFd as OwnedFileHandle;
#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;
#[cfg(windows)]
use std::os::windows::io::OwnedHandle as OwnedFileHandle;
use std::pin::pin;
use std::sync::mpsc;
use std::task::{Context, Poll};
use std::thread;

use futures::executor::block_on;
use futures::task::noop_waker;
use prime::os::background::ProximaBackgroundPool;
use proxima_core::ProximaError;

const PAYLOAD: &[u8] = b"proxima windows file io\n";

#[test]
fn windows_fs_background_round_trip_unicode_path() {
    let directory = tempfile::tempdir().expect("temporary filesystem root");
    let path = directory
        .path()
        .join("records")
        .join("caf\u{e9}-\u{65e5}\u{672c}.jsonl");
    let oracle_parent = directory.path().join("oracle");
    let oracle_path = oracle_parent.join(path.file_name().expect("unicode filename"));
    let oracle = fs::create_dir_all(&oracle_parent)
        .and_then(|()| fs::write(&oracle_path, PAYLOAD))
        .and_then(|()| fs::read(&oracle_path));
    let worker_path = path.clone();
    let pool = ProximaBackgroundPool::with_threads(1).expect("filesystem worker");
    let caller = thread::current().id();
    let (release, wait_for_release) = mpsc::channel();
    let future = pool.spawn(move || {
        wait_for_release
            .recv()
            .expect("test releases filesystem worker");
        let parent = worker_path.parent().expect("nested path parent");
        fs::create_dir_all(parent).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("create directory {}: {error}", parent.display()),
            )
        })?;
        let mut file = File::create(&worker_path).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("create file {}: {error}", worker_path.display()),
            )
        })?;
        file.write_all(PAYLOAD).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("write file {}: {error}", worker_path.display()),
            )
        })?;
        file.sync_all().map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("sync file {}: {error}", worker_path.display()),
            )
        })?;
        drop(file);
        let bytes = fs::read(&worker_path).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("read file {}: {error}", worker_path.display()),
            )
        })?;
        Ok((thread::current().id(), bytes))
    });
    let mut future = pin!(future);
    let waker = noop_waker();
    let mut context = Context::from_waker(&waker);
    assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
    release.send(()).expect("release filesystem worker");
    let outcome = block_on(future);
    assert!(
        oracle.is_ok(),
        "direct std::fs oracle at {}: {oracle:?}; background result: {outcome:?}",
        oracle_path.display()
    );
    assert_eq!(oracle.expect("std filesystem oracle"), PAYLOAD);
    let (worker, bytes) = outcome.expect("filesystem operation completes");
    assert_ne!(worker, caller, "file IO must run off the calling thread");
    assert_eq!(bytes, PAYLOAD);
    assert_eq!(fs::read(path).expect("independent file read"), PAYLOAD);
}

#[test]
fn windows_fs_owned_handle_rename_and_remove() {
    let directory = tempfile::tempdir().expect("temporary filesystem root");
    let path = directory.path().join("source.txt");
    let renamed = directory.path().join("renamed.txt");
    let pool = ProximaBackgroundPool::with_threads(1).expect("filesystem worker");
    let result = block_on(pool.spawn(move || {
        let mut file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&path)?;
        file.write_all(PAYLOAD)?;
        file.rewind()?;
        let owned = OwnedFileHandle::from(file);
        let mut restored = File::from(owned);
        let mut actual = Vec::new();
        restored.read_to_end(&mut actual)?;
        let length = restored.metadata()?.len();
        drop(restored);
        fs::rename(&path, &renamed)?;
        let renamed_bytes = fs::read(&renamed)?;
        fs::remove_file(&renamed)?;
        Ok((
            actual,
            renamed_bytes,
            length,
            path.exists(),
            renamed.exists(),
        ))
    }))
    .expect("owned file handle transfers and closes");
    assert_eq!(result.0, PAYLOAD);
    assert_eq!(result.1, PAYLOAD);
    assert_eq!(result.2, PAYLOAD.len() as u64);
    assert!(
        !result.3 && !result.4,
        "both filenames are absent after removal"
    );
}

#[test]
fn windows_fs_os_file_extensions_preserve_platform_cursor_contract() {
    let directory = tempfile::tempdir().expect("temporary filesystem root");
    let path = directory.path().join("offsets.txt");
    fs::write(&path, PAYLOAD).expect("initial payload");
    let pool = ProximaBackgroundPool::with_threads(1).expect("filesystem worker");
    let (written, read, actual, position, bytes) = block_on(pool.spawn(move || {
        let mut file = OpenOptions::new().read(true).write(true).open(&path)?;
        let mut actual = [0; 4];
        #[cfg(windows)]
        let written = file.seek_write(b"port", 8)?;
        #[cfg(unix)]
        let written = file.write_at(b"port", 8)?;
        #[cfg(windows)]
        let read = file.seek_read(&mut actual, 8)?;
        #[cfg(unix)]
        let read = file.read_at(&mut actual, 8)?;
        let position = file.stream_position()?;
        file.sync_all()?;
        drop(file);
        Ok((written, read, actual, position, fs::read(path)?))
    }))
    .expect("native file extension operations");
    assert_eq!(written, 4);
    assert_eq!(read, 4);
    assert_eq!(&actual, b"port");
    assert_eq!(bytes, b"proxima portows file io\n");
    #[cfg(windows)]
    assert_eq!(position, 12, "Windows FileExt advances the shared cursor");
    #[cfg(unix)]
    assert_eq!(
        position, 0,
        "Unix positional IO preserves the shared cursor"
    );
}

#[test]
fn windows_fs_errors_survive_background_completion() {
    let directory = tempfile::tempdir().expect("temporary filesystem root");
    let path = directory.path().join("present.txt");
    fs::write(&path, PAYLOAD).expect("initial payload");
    let missing = directory.path().join("missing.txt");
    let pool = ProximaBackgroundPool::with_threads(1).expect("filesystem worker");
    let (missing_kind, exists_kind, read_only_kind) = block_on(pool.spawn(move || {
        let missing_kind = File::open(missing)
            .expect_err("missing file must fail")
            .kind();
        let exists_kind = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .expect_err("create_new must not overwrite")
            .kind();
        let mut read_only = File::open(path)?;
        let read_only_kind = read_only
            .write_all(b"must not write")
            .expect_err("read-only handle rejects write")
            .kind();
        Ok((missing_kind, exists_kind, read_only_kind))
    }))
    .expect("filesystem error values cross worker boundary");
    assert_eq!(missing_kind, io::ErrorKind::NotFound);
    assert_eq!(exists_kind, io::ErrorKind::AlreadyExists);
    #[cfg(windows)]
    assert_eq!(read_only_kind, io::ErrorKind::PermissionDenied);
    #[cfg(unix)]
    assert_eq!(
        read_only_kind,
        io::Error::from_raw_os_error(libc::EBADF).kind()
    );
    let missing = directory.path().join("still-missing.txt");
    let error = block_on(pool.spawn(move || {
        let _file = File::open(missing)?;
        Ok(())
    }))
    .expect_err("filesystem error must reach the background caller");
    assert!(matches!(error, ProximaError::Io(error) if error.kind() == io::ErrorKind::NotFound));
}
