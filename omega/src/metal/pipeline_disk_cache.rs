//! Compiled Metal pipelines persisted across processes.
//!
//! [`compile_pipeline`] used to pay the backend compile of every kernel on
//! every process start. A pipeline is now stored as a one-pipeline
//! `MTLBinaryArchive` file named by the digest of everything its binary
//! depends on, and a later process loads that file instead of recompiling.
//! The MSL front end still runs (an `MTLFunction` is what the archive lookup
//! is keyed against); what a hit skips is the pipeline backend compile.
//!
//! Composes: `MTLBinaryArchive` (Apple's persisted-pipeline primitive),
//! [`PIPELINE_CACHE`] (the in-process cache this sits underneath: a hit there
//! never reaches this module) and [`RuntimeConfig`] (the `OMEGA_PIPELINE_CACHE*`
//! knobs). One file per pipeline, not one archive per process, because a
//! thread-local Metal object cannot be shared across threads while a file
//! can: concurrent threads and processes publish with an atomic rename and
//! never write the same bytes in place.
//!
//! The digest covers the kernel source, entry point, math mode, device name,
//! and the OS version and build string. The Metal compiler ships inside the
//! OS, so a compiler update changes the build string; an entry from another
//! device, OS build or option set is therefore never opened. A file that
//! fails to load is deleted and the pipeline is compiled and stored again.

use super::*;

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU64;
use std::time::SystemTime;

use objc2_foundation::{NSProcessInfo, NSURL};
use objc2_metal::{
    MTLBinaryArchive, MTLBinaryArchiveDescriptor, MTLComputePipelineDescriptor, MTLFunction,
    MTLPipelineOption,
};
use sha2::{Digest, Sha256};

use crate::config::RuntimeConfig;

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type Archive = Retained<ProtocolObject<dyn MTLBinaryArchive>>;
type Function = Retained<ProtocolObject<dyn MTLFunction>>;

const IDENTITY_VERSION: &str = "omega-pipeline-archive-1";
const ARCHIVE_EXTENSION: &str = "metalar";
const PARTIAL_EXTENSION: &str = "partial";
const DEFAULT_CACHE_SUBDIRECTORY: &str = "Library/Caches/proxima/omega-pipelines";

static RUNTIME_CONFIG: OnceLock<RuntimeConfig> = OnceLock::new();
static DISK_CACHE: OnceLock<Option<DiskCacheLocation>> = OnceLock::new();
static ARCHIVE_HITS: AtomicU64 = AtomicU64::new(0);
static ARCHIVE_STORES: AtomicU64 = AtomicU64::new(0);
static PARTIAL_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Where the process keeps its pipeline archives and how many bytes it may hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DiskCacheLocation {
    pub(super) directory: PathBuf,
    pub(super) max_bytes: u64,
}

/// What [`archived_pipeline`] did for one kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ArchiveOutcome {
    /// The pipeline came out of an archive file; no backend compile ran.
    Loaded,
    /// The pipeline was compiled and its archive file published.
    Stored,
    /// The pipeline was compiled; publishing its archive file failed.
    StoreFailed,
}

#[derive(Debug, thiserror::Error)]
enum ArchiveError {
    #[error("binary archive {operation} failed: {log}")]
    Metal { operation: &'static str, log: String },
    #[error("pipeline cache path {path} {operation} failed: {source}")]
    Io {
        operation: &'static str,
        path: PathBuf,
        source: io::Error,
    },
}

impl ArchiveError {
    fn metal(operation: &'static str, error: &NSError) -> Self {
        Self::Metal {
            operation,
            log: nserror_description(error),
        }
    }

    fn io(operation: &'static str, path: &Path, source: io::Error) -> Self {
        Self::Io {
            operation,
            path: path.to_path_buf(),
            source,
        }
    }
}

/// Installs the policy the pipeline cache reads, before the first kernel
/// compiles. Returns the config back when the policy was already fixed, by an
/// earlier call or by the first compile reading `OMEGA_*` itself.
pub fn set_runtime_config(config: RuntimeConfig) -> Result<(), RuntimeConfig> {
    RUNTIME_CONFIG.set(config)
}

/// `(archive hits, archive files published)` since process start. A hit is a
/// pipeline that skipped the backend compile; both are zero when the cache is off.
#[must_use]
pub fn pipeline_disk_cache_counts() -> (u64, u64) {
    (
        ARCHIVE_HITS.load(Ordering::Relaxed),
        ARCHIVE_STORES.load(Ordering::Relaxed),
    )
}

fn runtime_config() -> &'static RuntimeConfig {
    RUNTIME_CONFIG.get_or_init(|| {
        RuntimeConfig::from_env().unwrap_or_else(|error| {
            proxima_telemetry::warn!(%error, "OMEGA_* did not parse, pipeline cache keeps its defaults");
            RuntimeConfig::default()
        })
    })
}

/// The cache directory and byte budget `config` selects, or `None` when the
/// cache is off: switched off, a zero budget, or no directory to default to.
pub(super) fn resolve_location(
    config: &RuntimeConfig,
    home: Option<&Path>,
) -> Option<DiskCacheLocation> {
    if !config.pipeline_cache || config.pipeline_cache_max_bytes == 0 {
        return None;
    }
    let directory = match config.pipeline_cache_dir.as_str() {
        "" => home?.join(DEFAULT_CACHE_SUBDIRECTORY),
        explicit => PathBuf::from(explicit),
    };
    Some(DiskCacheLocation {
        directory,
        max_bytes: config.pipeline_cache_max_bytes,
    })
}

fn disk_cache() -> Option<&'static DiskCacheLocation> {
    DISK_CACHE
        .get_or_init(|| {
            let home = std::env::var_os("HOME").map(PathBuf::from);
            let location = resolve_location(runtime_config(), home.as_deref())?;
            match fs::create_dir_all(&location.directory) {
                Ok(()) => Some(location),
                Err(error) => {
                    proxima_telemetry::warn!(
                        directory = ?location.directory,
                        %error,
                        "pipeline cache directory could not be created, compiling without it"
                    );
                    None
                }
            }
        })
        .as_ref()
}

/// Hex SHA-256 over `parts`, each prefixed by its byte length so two field
/// lists that concatenate to the same text still digest differently.
pub(super) fn archive_digest(parts: &[&str]) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update((part.len() as u64).to_le_bytes());
        hasher.update(part.as_bytes());
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// The content name of `kernel`'s archive on this device and OS build.
pub(super) fn archive_key(
    device: &ProtocolObject<dyn MTLDevice>,
    kernel: &Kernel,
    math_mode: MathMode,
) -> String {
    let device_name = device.name().to_string();
    let operating_system = NSProcessInfo::processInfo()
        .operatingSystemVersionString()
        .to_string();
    let math_token = math_mode.cache_token().to_string();
    archive_digest(&[
        IDENTITY_VERSION,
        &device_name,
        &operating_system,
        &math_token,
        &kernel.entry,
        &kernel.source,
    ])
}

fn archive_path(directory: &Path, key: &str) -> PathBuf {
    directory.join(format!("{key}.{ARCHIVE_EXTENSION}"))
}

fn is_cache_file(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension == ARCHIVE_EXTENSION || extension == PARTIAL_EXTENSION)
}

/// Removes the oldest cache files in `directory` until the rest fit in
/// `max_bytes`; returns how many were removed. Files that are not archives or
/// partial writes are never touched.
pub(super) fn evict_to_budget(directory: &Path, max_bytes: u64) -> io::Result<usize> {
    let mut entries: Vec<(SystemTime, PathBuf, u64)> = fs::read_dir(directory)?
        .filter_map(Result::ok)
        .filter(|entry| is_cache_file(&entry.path()))
        .filter_map(|entry| {
            let metadata = entry.metadata().ok().filter(fs::Metadata::is_file)?;
            Some((metadata.modified().ok()?, entry.path(), metadata.len()))
        })
        .collect();
    entries.sort();
    let mut total: u64 = entries.iter().map(|(_, _, length)| length).sum();
    let mut removed = 0;
    for (_, path, length) in entries {
        if total <= max_bytes {
            break;
        }
        match fs::remove_file(&path) {
            Ok(()) => removed += 1,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
        total = total.saturating_sub(length);
    }
    Ok(removed)
}

fn file_url(path: &Path) -> Retained<NSURL> {
    NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()))
}

fn open_archive(
    device: &ProtocolObject<dyn MTLDevice>,
    source: Option<&Path>,
) -> Result<Archive, ArchiveError> {
    let descriptor = MTLBinaryArchiveDescriptor::new();
    if let Some(path) = source {
        descriptor.setUrl(Some(&file_url(path)));
    }
    device
        .newBinaryArchiveWithDescriptor_error(&descriptor)
        .map_err(|error| ArchiveError::metal("open", &error))
}

fn compute_descriptor(
    function: &ProtocolObject<dyn MTLFunction>,
    archive: Option<&Archive>,
) -> Retained<MTLComputePipelineDescriptor> {
    let descriptor = MTLComputePipelineDescriptor::new();
    descriptor.setComputeFunction(Some(function));
    if let Some(archive) = archive {
        let archives = NSArray::from_retained_slice(core::slice::from_ref(archive));
        descriptor.setBinaryArchives(Some(&archives));
    }
    descriptor
}

fn load_archived(
    device: &ProtocolObject<dyn MTLDevice>,
    function: &ProtocolObject<dyn MTLFunction>,
    path: &Path,
) -> Result<Pipeline, ArchiveError> {
    let archive = open_archive(device, Some(path))?;
    let descriptor = compute_descriptor(function, Some(&archive));
    device
        .newComputePipelineStateWithDescriptor_options_reflection_error(
            &descriptor,
            MTLPipelineOption::FailOnBinaryArchiveMiss,
            None,
        )
        .map_err(|error| ArchiveError::metal("lookup", &error))
}

fn store_archived(
    device: &ProtocolObject<dyn MTLDevice>,
    function: &ProtocolObject<dyn MTLFunction>,
    location: &DiskCacheLocation,
    path: &Path,
) -> Result<(), ArchiveError> {
    let archive = open_archive(device, None)?;
    archive
        .addComputePipelineFunctionsWithDescriptor_error(&compute_descriptor(function, None))
        .map_err(|error| ArchiveError::metal("add", &error))?;
    let partial = path.with_extension(format!(
        "{}-{}.{PARTIAL_EXTENSION}",
        std::process::id(),
        PARTIAL_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    archive
        .serializeToURL_error(&file_url(&partial))
        .map_err(|error| ArchiveError::metal("serialize", &error))?;
    fs::rename(&partial, path).map_err(|error| {
        let _ = fs::remove_file(&partial);
        ArchiveError::io("publish", path, error)
    })?;
    evict_to_budget(&location.directory, location.max_bytes)
        .map_err(|error| ArchiveError::io("evict", &location.directory, error))?;
    Ok(())
}

fn refresh_modified(path: &Path) {
    let refreshed = fs::File::options()
        .write(true)
        .open(path)
        .and_then(|file| file.set_modified(SystemTime::now()));
    if let Err(error) = refreshed {
        debug!(path = ?path, %error, "pipeline archive recency not refreshed");
    }
}

pub(super) fn pipeline_from_function(
    device: &ProtocolObject<dyn MTLDevice>,
    function: &ProtocolObject<dyn MTLFunction>,
) -> Result<Pipeline, MetalError> {
    device
        .newComputePipelineStateWithFunction_error(function)
        .map_err(|error| MetalError::CompileFailed {
            log: nserror_description(&error),
        })
}

/// `function`'s pipeline, from `location` when its archive file is there and
/// loads, otherwise compiled and then published to `location`. A file that
/// exists and fails to load is deleted. Publishing is best effort: its
/// failure is logged and the compiled pipeline is returned.
pub(super) fn archived_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    function: &ProtocolObject<dyn MTLFunction>,
    key: &str,
    location: &DiskCacheLocation,
) -> Result<(Pipeline, ArchiveOutcome), MetalError> {
    let path = archive_path(&location.directory, key);
    if path.is_file() {
        match load_archived(device, function, &path) {
            Ok(pipeline) => {
                refresh_modified(&path);
                ARCHIVE_HITS.fetch_add(1, Ordering::Relaxed);
                trace!(key = %key, "pipeline archive loaded");
                return Ok((pipeline, ArchiveOutcome::Loaded));
            }
            Err(error) => {
                debug!(key = %key, %error, "pipeline archive rejected, recompiling");
                let _ = fs::remove_file(&path);
            }
        }
    }
    let pipeline = pipeline_from_function(device, function)?;
    match store_archived(device, function, location, &path) {
        Ok(()) => {
            ARCHIVE_STORES.fetch_add(1, Ordering::Relaxed);
            Ok((pipeline, ArchiveOutcome::Stored))
        }
        Err(error) => {
            proxima_telemetry::warn!(key = %key, %error, "pipeline archive not published");
            Ok((pipeline, ArchiveOutcome::StoreFailed))
        }
    }
}

/// [`archived_pipeline`] against the process's configured cache, or a plain
/// compile when the cache is off.
pub(super) fn pipeline_for_function(
    device: &ProtocolObject<dyn MTLDevice>,
    function: &Function,
    kernel: &Kernel,
    math_mode: MathMode,
) -> Result<Pipeline, MetalError> {
    let Some(location) = disk_cache() else {
        return pipeline_from_function(device, function);
    };
    let key = archive_key(device, kernel, math_mode);
    archived_pipeline(device, function, &key, location).map(|(pipeline, _)| pipeline)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    use crate::msl::GridSpec;

    const PROBE_SOURCE: &str = "#include <metal_stdlib>\n\
        using namespace metal;\n\
        kernel void archive_probe(device float* output [[buffer(0)]], uint gid [[thread_position_in_grid]]) {\n\
            output[gid] = float(gid) * 2.0f + 1.0f;\n\
        }\n";
    const PROBE_ELEMENTS: usize = 64;

    fn config(enabled: bool, directory: &str, max_bytes: u64) -> RuntimeConfig {
        RuntimeConfig {
            pipeline_cache: enabled,
            pipeline_cache_dir: directory.to_string(),
            pipeline_cache_max_bytes: max_bytes,
            ..RuntimeConfig::default()
        }
    }

    fn probe_kernel(source: &str) -> Kernel {
        Kernel {
            source: source.to_string(),
            entry: String::from("archive_probe"),
            bindings: Vec::new(),
            grid: GridSpec {
                threads: PROBE_ELEMENTS as u64,
                threadgroup_width: None,
                depth: 1,
                grid2d: None,
            },
        }
    }

    fn probe_function(
        device: &ProtocolObject<dyn MTLDevice>,
        kernel: &Kernel,
        math_mode: MathMode,
    ) -> Function {
        let options = MTLCompileOptions::new();
        options.setMathMode(math_mode.as_mtl());
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(&kernel.source), Some(&options))
            .expect("the probe kernel source must compile");
        library
            .newFunctionWithName(&NSString::from_str(&kernel.entry))
            .expect("the probe library must define its entry point")
    }

    fn run_probe(pipeline: &ProtocolObject<dyn MTLComputePipelineState>) -> Vec<f32> {
        let (device, queue) = device_and_queue().expect("a Metal device is required");
        let length = PROBE_ELEMENTS * size_of::<f32>();
        let buffer = device
            .newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)
            .expect("device must allocate the probe output");
        let command_buffer = queue.commandBuffer().expect("queue must hand out a command buffer");
        let encoder = command_buffer
            .computeCommandEncoder()
            .expect("command buffer must open a compute encoder");
        encoder.setComputePipelineState(pipeline);
        unsafe { encoder.setBuffer_offset_atIndex(Some(&buffer), 0, 0) };
        encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize { width: PROBE_ELEMENTS, height: 1, depth: 1 },
            MTLSize { width: PROBE_ELEMENTS, height: 1, depth: 1 },
        );
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        let contents = buffer.contents().as_ptr().cast::<f32>();
        (0..PROBE_ELEMENTS)
            .map(|index| unsafe { contents.add(index).read() })
            .collect()
    }

    fn expected_probe_output() -> Vec<f32> {
        (0..PROBE_ELEMENTS).map(|index| index as f32 * 2.0 + 1.0).collect()
    }

    #[test]
    fn digest_is_stable_and_changes_with_every_field() {
        let base = ["v1", "Apple M1 Max", "Version 15.6 (Build 24G84)", "R", "entry", "source"];

        let baseline = archive_digest(&base);

        assert_eq!(baseline, archive_digest(&base), "same fields must give the same name");
        assert_eq!(baseline.len(), 64, "a hex sha256 is 64 characters");
        for index in 0..base.len() {
            let mut changed = base;
            changed[index] = "different";
            assert_ne!(baseline, archive_digest(&changed), "field {index} must be part of the name");
        }
    }

    #[test]
    fn digest_separates_fields_that_concatenate_alike() {
        assert_ne!(archive_digest(&["ab", "c"]), archive_digest(&["a", "bc"]));
    }

    #[test]
    fn location_follows_the_config_switches() {
        let home = Path::new("/Users/someone");

        let off = resolve_location(&config(false, "", 1024), Some(home));
        let zero_budget = resolve_location(&config(true, "", 0), Some(home));
        let explicit = resolve_location(&config(true, "/var/cache/omega", 4096), Some(home));
        let defaulted = resolve_location(&config(true, "", 4096), Some(home));
        let homeless = resolve_location(&config(true, "", 4096), None);
        let homeless_explicit = resolve_location(&config(true, "/var/cache/omega", 4096), None);

        assert_eq!(off, None);
        assert_eq!(zero_budget, None);
        assert_eq!(
            explicit,
            Some(DiskCacheLocation { directory: PathBuf::from("/var/cache/omega"), max_bytes: 4096 })
        );
        assert_eq!(
            defaulted,
            Some(DiskCacheLocation {
                directory: PathBuf::from("/Users/someone/Library/Caches/proxima/omega-pipelines"),
                max_bytes: 4096,
            })
        );
        assert_eq!(homeless, None);
        assert_eq!(homeless_explicit.map(|location| location.max_bytes), Some(4096));
    }

    fn write_aged(directory: &Path, name: &str, bytes: usize, age_seconds: u64) -> PathBuf {
        let path = directory.join(name);
        fs::write(&path, vec![7u8; bytes]).expect("fixture file must be written");
        let modified = SystemTime::now() - std::time::Duration::from_secs(age_seconds);
        fs::File::options()
            .write(true)
            .open(&path)
            .and_then(|file| file.set_modified(modified))
            .expect("fixture mtime must be set");
        path
    }

    #[test]
    fn eviction_removes_the_oldest_cache_files_until_the_budget_holds() {
        let directory = tempfile::tempdir().expect("tempdir");
        let oldest = write_aged(directory.path(), "aaa.metalar", 400, 300);
        let middle = write_aged(directory.path(), "bbb.metalar", 400, 200);
        let newest = write_aged(directory.path(), "ccc.metalar", 400, 100);
        let foreign = write_aged(directory.path(), "notes.txt", 5000, 900);

        let removed = evict_to_budget(directory.path(), 800).expect("eviction must scan the directory");

        assert_eq!(removed, 1);
        assert!(!oldest.exists(), "the oldest file must go first");
        assert!(middle.exists() && newest.exists(), "newer files fit the budget and stay");
        assert!(foreign.exists(), "a file that is not a cache file is never removed");
    }

    #[test]
    fn eviction_keeps_everything_when_the_files_fit() {
        let directory = tempfile::tempdir().expect("tempdir");
        let first = write_aged(directory.path(), "aaa.metalar", 100, 20);
        let partial = write_aged(directory.path(), "bbb.1-0.partial", 100, 10);

        let removed = evict_to_budget(directory.path(), 200).expect("eviction must scan the directory");

        assert_eq!(removed, 0);
        assert!(first.exists() && partial.exists());
    }

    #[test]
    fn eviction_counts_partial_writes_against_the_budget() {
        let directory = tempfile::tempdir().expect("tempdir");
        let stale_partial = write_aged(directory.path(), "aaa.9-0.partial", 500, 3000);
        let published = write_aged(directory.path(), "bbb.metalar", 500, 10);

        let removed = evict_to_budget(directory.path(), 600).expect("eviction must scan the directory");

        assert_eq!(removed, 1);
        assert!(!stale_partial.exists(), "an abandoned partial write is the oldest and goes first");
        assert!(published.exists());
    }

    #[test]
    fn eviction_of_a_missing_directory_is_an_error_not_a_silent_zero() {
        let directory = tempfile::tempdir().expect("tempdir");

        let result = evict_to_budget(&directory.path().join("absent"), 0);

        assert!(result.is_err());
    }

    #[test]
    fn a_compiled_pipeline_is_published_then_loaded_and_computes_the_same_bytes() {
        let (device, _queue) = device_and_queue().expect("a Metal device is required");
        let directory = tempfile::tempdir().expect("tempdir");
        let location = DiskCacheLocation { directory: directory.path().to_path_buf(), max_bytes: 64 << 20 };
        let kernel = probe_kernel(PROBE_SOURCE);
        let key = archive_key(&device, &kernel, MathMode::Relaxed);
        let function = probe_function(&device, &kernel, MathMode::Relaxed);

        let (compiled, first) = archived_pipeline(&device, &function, &key, &location).expect("first compile");
        let (loaded, second) = archived_pipeline(&device, &function, &key, &location).expect("second lookup");

        assert_eq!(first, ArchiveOutcome::Stored);
        assert_eq!(second, ArchiveOutcome::Loaded);
        assert!(archive_path(directory.path(), &key).is_file());
        assert_eq!(run_probe(&compiled), expected_probe_output());
        assert_eq!(run_probe(&loaded), expected_probe_output());
        assert_eq!(
            loaded.maxTotalThreadsPerThreadgroup(),
            compiled.maxTotalThreadsPerThreadgroup(),
            "an archived pipeline must carry the compiled pipeline's limits"
        );
    }

    #[test]
    fn a_corrupt_archive_file_is_replaced_by_a_fresh_compile() {
        let (device, _queue) = device_and_queue().expect("a Metal device is required");
        let directory = tempfile::tempdir().expect("tempdir");
        let location = DiskCacheLocation { directory: directory.path().to_path_buf(), max_bytes: 64 << 20 };
        let kernel = probe_kernel(PROBE_SOURCE);
        let key = archive_key(&device, &kernel, MathMode::Relaxed);
        let function = probe_function(&device, &kernel, MathMode::Relaxed);
        let path = archive_path(directory.path(), &key);
        let garbage = b"not a metal binary archive";
        fs::write(&path, garbage).expect("garbage fixture");

        let (pipeline, outcome) = archived_pipeline(&device, &function, &key, &location).expect("fallback compile");
        let (_, after) = archived_pipeline(&device, &function, &key, &location).expect("lookup after repair");

        assert_eq!(outcome, ArchiveOutcome::Stored, "the bad file is dropped and a good one published");
        assert_eq!(after, ArchiveOutcome::Loaded);
        assert_ne!(fs::read(&path).expect("repaired file"), garbage);
        assert_eq!(run_probe(&pipeline), expected_probe_output());
    }

    #[test]
    fn math_mode_and_source_each_name_a_separate_archive() {
        let (device, _queue) = device_and_queue().expect("a Metal device is required");
        let directory = tempfile::tempdir().expect("tempdir");
        let location = DiskCacheLocation { directory: directory.path().to_path_buf(), max_bytes: 64 << 20 };
        let relaxed_kernel = probe_kernel(PROBE_SOURCE);
        let edited_kernel = probe_kernel(&PROBE_SOURCE.replace("2.0f", "3.0f"));
        let variants = [
            (&relaxed_kernel, MathMode::Relaxed),
            (&relaxed_kernel, MathMode::Safe),
            (&edited_kernel, MathMode::Relaxed),
        ];

        for (kernel, math_mode) in variants {
            let key = archive_key(&device, kernel, math_mode);
            let function = probe_function(&device, kernel, math_mode);
            let (_, outcome) = archived_pipeline(&device, &function, &key, &location).expect("compile");
            assert_eq!(outcome, ArchiveOutcome::Stored, "a new option set must not reuse another's archive");
        }

        let published = fs::read_dir(directory.path())
            .expect("cache directory")
            .filter_map(Result::ok)
            .filter(|entry| is_cache_file(&entry.path()))
            .count();
        assert_eq!(published, 3);
    }

    #[test]
    fn an_archive_larger_than_the_budget_is_evicted_but_the_pipeline_still_runs() {
        let (device, _queue) = device_and_queue().expect("a Metal device is required");
        let directory = tempfile::tempdir().expect("tempdir");
        let location = DiskCacheLocation { directory: directory.path().to_path_buf(), max_bytes: 1 };
        let kernel = probe_kernel(PROBE_SOURCE);
        let key = archive_key(&device, &kernel, MathMode::Relaxed);
        let function = probe_function(&device, &kernel, MathMode::Relaxed);

        let (pipeline, _) = archived_pipeline(&device, &function, &key, &location).expect("compile");

        assert!(!archive_path(directory.path(), &key).exists(), "a file over the budget is evicted at once");
        assert_eq!(run_probe(&pipeline), expected_probe_output());
    }
}
