use core::convert::Infallible;
use core::sync::atomic::{AtomicUsize, Ordering};

use proxima_primitives::sync::blocking::Mutex;

use super::code_object::{FunctionCache, HIPRTC_OPTIONS, cached_by_source, code_object_key};
use super::device::compile_options;
use super::error::HipDriverError;
use super::window::{WindowRefusal, element_window};
use crate::cuda_driver::{CompileCounters, stored_or_compiled};
use crate::cuda_ptx_store::PtxStore;

const SOURCE: &str =
    "#include <hip/hip_runtime.h>\nextern \"C\" __global__ void touch_first() { }\n";
const COMPILER: (i32, i32) = (6, 2);
const UPGRADED_COMPILER: (i32, i32) = (6, 3);
const CODE_OBJECT: &[u8] = b"\x7fELF\x02\x01\x01\x40amdgcn-amd-amdhsa--gfx1100 code object bytes";

fn compiled(compiles: &AtomicUsize) -> Result<Vec<u8>, Infallible> {
    compiles.fetch_add(1, Ordering::Relaxed);
    Ok(CODE_OBJECT.to_vec())
}

#[test]
fn a_code_object_compiles_once_then_reads_back_from_the_disk_store() {
    let root = tempfile::tempdir().expect("tempdir");
    let counters = CompileCounters::new(PtxStore::Directory(root.path().to_path_buf()));
    let compiles = AtomicUsize::new(0);
    let key = code_object_key(SOURCE, "gfx1100", COMPILER, HIPRTC_OPTIONS);

    let first = stored_or_compiled(&counters, &key, || compiled(&compiles));
    let second = stored_or_compiled(&counters, &key, || compiled(&compiles));

    assert_eq!(first, Ok(CODE_OBJECT.to_vec()));
    assert_eq!(
        second,
        Ok(CODE_OBJECT.to_vec()),
        "the stored bytes are the compiled bytes"
    );
    assert_eq!(
        compiles.load(Ordering::Relaxed),
        1,
        "the second call reads the store and compiles nothing"
    );
    assert_eq!(
        counters.disk_cache_hits.load(Ordering::Relaxed),
        1,
        "the read-back is counted as a disk hit"
    );
}

#[test]
fn a_different_compiler_version_compiles_again_instead_of_reading_the_stored_object() {
    let root = tempfile::tempdir().expect("tempdir");
    let counters = CompileCounters::new(PtxStore::Directory(root.path().to_path_buf()));
    let compiles = AtomicUsize::new(0);
    let before = code_object_key(SOURCE, "gfx1100", COMPILER, HIPRTC_OPTIONS);
    let after = code_object_key(SOURCE, "gfx1100", UPGRADED_COMPILER, HIPRTC_OPTIONS);

    stored_or_compiled(&counters, &before, || compiled(&compiles)).expect("compiles");
    stored_or_compiled(&counters, &after, || compiled(&compiles)).expect("compiles");
    stored_or_compiled(&counters, &after, || compiled(&compiles)).expect("reads back");

    assert_eq!(
        compiles.load(Ordering::Relaxed),
        2,
        "the upgraded compiler compiles once, then reads its own entry"
    );
    assert_eq!(counters.disk_cache_hits.load(Ordering::Relaxed), 1);
}

#[test]
fn another_arch_or_another_source_compiles_again() {
    let root = tempfile::tempdir().expect("tempdir");
    let counters = CompileCounters::new(PtxStore::Directory(root.path().to_path_buf()));
    let compiles = AtomicUsize::new(0);
    let keys = [
        code_object_key(SOURCE, "gfx1100", COMPILER, HIPRTC_OPTIONS),
        code_object_key(SOURCE, "gfx90a", COMPILER, HIPRTC_OPTIONS),
        code_object_key(
            "extern \"C\" __global__ void other() {}\n",
            "gfx1100",
            COMPILER,
            HIPRTC_OPTIONS,
        ),
        code_object_key(SOURCE, "gfx1100", COMPILER, &[]),
        code_object_key(SOURCE, "gfx1100", UPGRADED_COMPILER, HIPRTC_OPTIONS),
    ];

    for key in &keys {
        stored_or_compiled(&counters, key, || compiled(&compiles)).expect("compiles");
    }

    assert_eq!(
        compiles.load(Ordering::Relaxed),
        keys.len(),
        "no key reads another key's code object"
    );
    assert_eq!(counters.disk_cache_hits.load(Ordering::Relaxed), 0);
}

#[test]
fn a_corrupt_stored_code_object_is_recompiled_not_loaded() {
    let root = tempfile::tempdir().expect("tempdir");
    let counters = CompileCounters::new(PtxStore::Directory(root.path().to_path_buf()));
    let compiles = AtomicUsize::new(0);
    let key = code_object_key(SOURCE, "gfx1100", COMPILER, HIPRTC_OPTIONS);
    std::fs::write(root.path().join(key.to_hex()), b"PXKC truncated").expect("corrupt the entry");

    let image = stored_or_compiled(&counters, &key, || compiled(&compiles));

    assert_eq!(image, Ok(CODE_OBJECT.to_vec()));
    assert_eq!(
        compiles.load(Ordering::Relaxed),
        1,
        "a rejected entry compiles"
    );
    assert_eq!(counters.disk_cache_hits.load(Ordering::Relaxed), 0);
}

fn build_counted(builds: &AtomicUsize, tag: &str) -> Result<String, Infallible> {
    builds.fetch_add(1, Ordering::Relaxed);
    Ok(tag.to_owned())
}

#[test]
fn two_sources_sharing_an_entry_get_distinct_functions() {
    let cache = Mutex::new(FunctionCache::new());
    let builds = AtomicUsize::new(0);

    let first = cached_by_source(&cache, "k_add", "source a", || {
        build_counted(&builds, "built from a")
    });
    let second = cached_by_source(&cache, "k_add", "source b", || {
        build_counted(&builds, "built from b")
    });
    let first_again = cached_by_source(&cache, "k_add", "source a", || {
        build_counted(&builds, "rebuilt")
    });

    assert_eq!(first, Ok(String::from("built from a")));
    assert_eq!(second, Ok(String::from("built from b")));
    assert_eq!(
        first_again,
        Ok(String::from("built from a")),
        "the first source still resolves to its own function"
    );
    assert_eq!(
        builds.load(Ordering::Relaxed),
        2,
        "a repeated source builds nothing"
    );
}

#[test]
fn every_kernel_compiles_with_the_warp_sync_builtins_enabled() {
    let options = compile_options(Some("gfx1100"), HIPRTC_OPTIONS);

    assert_eq!(
        options,
        vec![
            String::from("--offload-arch=gfx1100"),
            String::from("-DHIP_ENABLE_WARP_SYNC_BUILTINS")
        ]
    );
}

#[test]
fn a_window_inside_the_buffer_is_scaled_to_bytes() {
    assert_eq!(element_window(100, &(10..30), 4), Ok((40, 80)));
}

#[test]
fn a_window_past_the_end_is_refused_with_its_byte_extent() {
    let refusal = element_window(100, &(90..130), 4)
        .expect_err("a window past the end is refused, not clamped");

    assert_eq!(
        refusal,
        WindowRefusal {
            offset: 360,
            len: 160,
            bytes: 400
        }
    );
    assert!(
        matches!(
            refusal.error(),
            HipDriverError::ViewOutOfRange {
                offset: 360,
                len: 160,
                bytes: 400
            }
        ),
        "the refusal is the typed view error"
    );
}

#[test]
fn a_window_starting_past_the_end_is_refused() {
    assert!(
        element_window(100, &(120..130), 4).is_err(),
        "a start past the end would hand the kernel an end pointer"
    );
}

#[test]
fn a_reversed_range_is_refused() {
    let reversed = core::ops::Range { start: 30, end: 10 };

    assert!(
        element_window(100, &reversed, 4).is_err(),
        "a reversed range names no window"
    );
}

#[test]
fn an_empty_range_is_an_empty_window_at_its_start() {
    assert_eq!(element_window(100, &(30..30), 1), Ok((30, 0)));
}
