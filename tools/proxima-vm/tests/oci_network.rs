//! `traefik/whoami:v1.10.2` pulled from docker hub through proxima's own prime HTTP/1 client and
//! TLS. Needs the network, so `Cargo.toml` gates it behind the `oci-network` feature; the default
//! suite stays offline.

#![cfg(all(target_os = "linux", target_arch = "x86_64"))]
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::cell::Cell;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use bytes::Bytes;
use http::{Request, Response};
use prime::os::core_shard::run_on_this_thread;
use proxima_runtime::CoreId;
use proxima_vm::oci::error::OciError;
use proxima_vm::oci::reference::ImageRef;
use proxima_vm::oci::registry::{PulledImage, pull};
use proxima_vm::oci::transport;

const WHOAMI: &str = "traefik/whoami:v1.10.2";

fn cache_root(name: &str) -> PathBuf {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::create_dir_all(&root).expect("create cache root");
    root
}

fn evidence(line: &str) {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("oci-evidence.log");
    let mut log = std::fs::OpenOptions::new().create(true).append(true).open(path).expect("open evidence log");
    writeln!(log, "{line}").expect("write evidence");
}

fn counted_pull(cache_root: PathBuf, image: ImageRef) -> (Result<PulledImage, OciError>, usize, std::time::Duration) {
    run_on_this_thread(CoreId(0), async move {
        let requests = Cell::new(0_usize);
        let counting = async |request: Request<Bytes>| -> Result<Response<Bytes>, OciError> {
            requests.set(requests.get() + 1);
            evidence(&format!("request {}: {}", requests.get(), request.uri()));
            transport::fetch(request).await
        };
        let started = Instant::now();
        let outcome = pull(&counting, &cache_root, &image, "linux", "amd64").await;
        (outcome, requests.get(), started.elapsed())
    })
    .expect("prime worker")
}


mod common;

const WHOAMI_INDEX: &str = "traefik/whoami@sha256:1474027c316661cdec87df2623e13a41e7e1ce0ba99c24917631de8f300b5420";
const SAMPLES: usize = 5;

fn summarize(label: &str, mut samples: Vec<std::time::Duration>) {
    samples.sort();
    let (median, maximum) = (samples[samples.len() / 2], samples[samples.len() - 1]);
    evidence(&format!("{label:<34} n={} p50={:.3}ms max={:.3}ms", samples.len(), median.as_secs_f64() * 1e3, maximum.as_secs_f64() * 1e3));
}

fn loadavg() -> String {
    std::fs::read_to_string("/proc/loadavg").expect("read /proc/loadavg").trim().to_string()
}

#[test]
fn whoami_pulls_by_tag_then_by_digest_with_zero_requests_n5() {
    let cache = cache_root("oci-whoami-cache-fresh");
    let by_tag = ImageRef::parse(WHOAMI).expect("reference");
    evidence(&format!("== pull run loadavg={}", loadavg()));
    let (mut network, mut cached, mut network_requests) = (Vec::new(), Vec::new(), Vec::new());

    for sample in 0..SAMPLES {
        std::fs::remove_dir_all(&cache).expect("clear cache");
        std::fs::create_dir_all(&cache).expect("recreate cache");
        let (first, first_requests, first_elapsed) = counted_pull(cache.clone(), by_tag.clone());
        let first = first.expect("network pull");
        let (second, second_requests, second_elapsed) = counted_pull(cache.clone(), by_tag.with_digest(first.root.clone()));
        let second = second.expect("cached pull");
        if sample == 0 {
            evidence(&format!("root digest {}", first.root));
            evidence(&format!("manifest digest {}", first.manifest));
            for layer in &first.layers {
                evidence(&format!("layer {} {} {} bytes", layer.digest, layer.media_type, layer.bytes.len()));
            }
            evidence(&format!("run {:?}", first.run));
        }
        assert!(first_requests > 0);
        assert_eq!(second_requests, 0, "a pull by digest from a warm cache must not touch the network");
        assert_eq!(second.layers.len(), first.layers.len());
        network.push(first_elapsed);
        cached.push(second_elapsed);
        network_requests.push(first_requests);
    }

    summarize("pull by tag, network, cold cache", network);
    summarize("pull by digest, warm cache, 0 req", cached);
    evidence(&format!("network pull request counts {network_requests:?}; warm-cache request count 0 on all {SAMPLES}"));
}

#[test]
fn whoami_pulled_by_digest_boots_and_prints_its_startup_line_n5() {
    let cache = cache_root("oci-whoami-cache");
    let image = ImageRef::parse(WHOAMI_INDEX).expect("reference");
    evidence(&format!("== boot run loadavg={}", loadavg()));
    let (pulled, requests, pull_elapsed) = counted_pull(cache.clone(), image);
    let pulled = pulled.expect("pull by digest");
    evidence(&format!("boot-run pull by digest: {requests} requests in {pull_elapsed:?}"));
    let init = common::guest("oci-init");
    let mut flatten = Vec::new();
    let mut initramfs = Vec::new();
    for _ in 0..SAMPLES {
        let flatten_started = Instant::now();
        initramfs = proxima_vm::oci::initramfs_for(&pulled, &init).expect("initramfs");
        flatten.push(flatten_started.elapsed());
    }
    summarize("flatten + newc cpio (debug build)", flatten);
    evidence(&format!("initramfs {} bytes", initramfs.len()));

    let (mut init_exec, mut entrypoint, mut totals) = (Vec::new(), Vec::new(), Vec::new());
    let mut startup_lines = Vec::new();
    for _ in 0..SAMPLES {
        let run = common::boot_image(&initramfs, 400, 20_000);
        init_exec.push(run.timings.init_exec.expect("init marker"));
        entrypoint.push(run.timings.entrypoint.expect("entrypoint marker"));
        totals.push(run.timings.total);
        let console = String::from_utf8_lossy(&run.console).into_owned();
        if startup_lines.is_empty() {
            for line in console.lines().filter(|line| ["Linux version", "Kernel command line", "Unknown kernel", "PCI:", "virtio", "NET:", "Trying to unpack", "Freeing initrd"].iter().any(|needle| line.contains(needle))) {
                evidence(&format!("kernel console: {line}"));
            }
        }
        startup_lines.push(console.lines().filter(|line| !line.starts_with('[')).collect::<Vec<_>>().join(" | "));
        assert_eq!(run.output, b"running", "whoami is a server: still alive when the linger ends\n{console}");
    }

    summarize("create -> /init exec (host clock)", init_exec);
    summarize("create -> entrypoint spawn (host)", entrypoint);
    summarize("create -> end, 400ms linger (host)", totals);
    for line in &startup_lines {
        evidence(&format!("guest console (non-kernel lines): {line}"));
    }
    assert!(startup_lines.iter().all(|line| line.contains("Starting up on port 80")), "{startup_lines:?}");
    let guest_year: u32 = startup_lines[0].split('/').next().and_then(|year| year.trim().parse().ok()).expect("whoami logs a year first");
    assert!(guest_year >= 2025, "the guest wall clock reads year {guest_year}: the CMOS RTC must hand the kernel the host date");
}

#[test]
fn a_layer_tampered_in_the_warm_cache_is_rejected_before_any_boot() {
    let cache = cache_root("oci-whoami-cache-tampered");
    std::fs::remove_dir_all(&cache).expect("clear the cache a previous run tampered with");
    std::fs::create_dir_all(&cache).expect("recreate cache");
    let image = ImageRef::parse(WHOAMI_INDEX).expect("reference");
    let (seeded, _, _) = counted_pull(cache.clone(), image.clone());
    let seeded = seeded.expect("seed the cache over the network");
    let victim = &seeded.layers[2].digest;
    let path = cache.join("blobs").join("sha256").join(victim.hex());
    let mut bytes = std::fs::read(&path).expect("read the binary layer");
    let middle = bytes.len() / 2;
    bytes[middle] ^= 0xff;
    std::fs::write(&path, &bytes).expect("tamper");

    let (outcome, requests, _) = counted_pull(cache, image);

    let error = outcome.expect_err("a tampered layer must be rejected");
    let message = error.to_string();
    evidence(&format!("tampered layer rejected: {message}"));
    assert!(message.contains(&format!("expected {victim}")) && message.contains("actual sha256:"), "{message}");
    assert_eq!(requests, 0, "the rejection happened from the cache, before any boot and without the network");
}
