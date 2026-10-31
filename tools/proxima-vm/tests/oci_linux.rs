//! The OCI path into the linux flavor, offline: vendored real whoami layers for flatten, a probe
//! entrypoint for what the image config hands the process, and a tampered blob for the control.
//! Boots `/boot/vmlinuz-linux` (or `PROXIMA_VM_KERNEL`) like `linux_guest.rs`.

#![cfg(all(target_os = "linux", target_arch = "x86_64"))]
#![allow(clippy::expect_used, clippy::unwrap_used)]

mod common;

use std::cell::Cell;
use std::io::Cursor;
use std::io::Write;
use std::path::PathBuf;

use bytes::Bytes;
use flate2::Compression;
use flate2::write::GzEncoder;
use http::{Request, Response};
use proxima_vm::oci::digest::Digest;
use proxima_vm::oci::error::OciError;
use proxima_vm::oci::manifest::{self, RunConfig};
use proxima_vm::oci::reference::ImageRef;
use proxima_vm::oci::registry::{Layer, PulledImage, pull};
use proxima_vm::oci::rootfs::{self, Entry, Node};
use proxima_vm::oci::{cache, initramfs_for};
use tar::{Builder, EntryType, Header};

const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/oci/whoami-v1.10.2");
const INDEX_DIGEST: &str = "sha256:1474027c316661cdec87df2623e13a41e7e1ce0ba99c24917631de8f300b5420";
const MANIFEST_DIGEST: &str = "sha256:7000846753fcc36eb2a1a3a21fe897da09c71e1d3b5381f912ea6e6e1c8871b6";
const CONFIG_DIGEST: &str = "sha256:ab0ca2603e47e42f7d45e094b6f4cecc5d5b7b0ef6ca89170ce1a771ed1a0066";
const ZONEINFO_LAYER: &str = "sha256:1b5c60a1be3bf9cb17566bdad4cb1f36387367e56798bc307d394169f1e5d72d";
const CERTS_LAYER: &str = "sha256:1d12c9005bc5cd6824edcfe37f412c35922508f76cb4b854beb52f85c12a929b";
const DOCKER_LAYER: &str = "application/vnd.docker.image.rootfs.diff.tar.gzip";

fn vendored(digest: &str) -> Vec<u8> {
    let name = digest.strip_prefix("sha256:").expect("sha256 digest");
    let raw: PathBuf = [FIXTURE, "blobs", "sha256", name].iter().collect();
    match std::fs::read(&raw) {
        Ok(bytes) => bytes,
        Err(_) => {
            let hex_path = raw.with_extension("hex");
            let text = std::fs::read_to_string(&hex_path).unwrap_or_else(|error| panic!("vendored blob {} must exist raw or as .hex: {error}", raw.display()));
            decode_hex(&text)
        }
    }
}

fn decode_hex(text: &str) -> Vec<u8> {
    text.trim()
        .as_bytes()
        .chunks(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).expect("ascii hex"), 16).expect("hex digit pair"))
        .collect()
}

fn vendored_layer(digest: &str) -> Layer {
    Layer { digest: Digest::parse(digest).expect("digest"), media_type: DOCKER_LAYER.into(), bytes: vendored(digest) }
}

fn file_data(tree: &rootfs::Rootfs, path: &str) -> Vec<u8> {
    match tree.get(path) {
        Some(Entry { node: Node::File(data), .. }) => data.clone(),
        other => panic!("{path} should be a file in the flattened rootfs, found {other:?}"),
    }
}

#[test]
fn every_vendored_blob_verifies_against_the_digest_it_is_named_by() {
    for digest in [INDEX_DIGEST, MANIFEST_DIGEST, CONFIG_DIGEST, ZONEINFO_LAYER, CERTS_LAYER] {
        Digest::parse(digest).expect("digest").verify("vendored blob", &vendored(digest)).expect("fixture bytes match their name");
    }
}

#[test]
fn the_vendored_manifest_and_config_name_whoami_as_the_entrypoint() {
    let document = manifest::parse_document(&vendored(MANIFEST_DIGEST)).expect("manifest");
    let config = manifest::parse_image_config(&vendored(CONFIG_DIGEST)).expect("config");

    assert_eq!(document.layers.len(), 3);
    assert_eq!(document.layers[0].digest, ZONEINFO_LAYER);
    assert_eq!(config.config.argv().expect("argv"), ["/whoami"]);
    assert_eq!(config.config.working_dir.as_deref(), Some("/"));
}

#[test]
fn the_vendored_index_selects_the_linux_amd64_manifest() {
    let index = manifest::parse_document(&vendored(INDEX_DIGEST)).expect("index");

    let chosen = index.select("linux", "amd64").expect("amd64 is listed");

    assert_eq!(chosen.as_str(), MANIFEST_DIGEST);
}

#[test]
fn the_real_zoneinfo_layer_flattens_with_its_hard_links_resolved_to_the_same_bytes() {
    let tree = rootfs::flatten(&[vendored_layer(ZONEINFO_LAYER), vendored_layer(CERTS_LAYER)]).expect("flatten real layers");

    assert_eq!(tree["usr/share/zoneinfo"].node, Node::Directory);
    assert_eq!(file_data(&tree, "usr/share/zoneinfo/Africa/Accra"), file_data(&tree, "usr/share/zoneinfo/Africa/Abidjan"));
    assert_eq!(file_data(&tree, "usr/share/zoneinfo/Africa/Abidjan").len(), 148);
    assert_eq!(file_data(&tree, "etc/ssl/certs/ca-certificates.crt").len(), 214_509);
}

fn probe_layer(program: &[u8]) -> Layer {
    let mut builder = Builder::new(Vec::new());
    let mut append = |kind: EntryType, mode: u32, path: &str, contents: &[u8]| {
        let mut header = Header::new_gnu();
        header.set_entry_type(kind);
        header.set_mode(mode);
        header.set_size(contents.len() as u64);
        header.set_uid(0);
        header.set_gid(0);
        header.set_mtime(0);
        builder.append_data(&mut header, path, contents).expect("append");
    };
    append(EntryType::Directory, 0o755, "srv/", b"");
    append(EntryType::Directory, 0o755, "bin/", b"");
    append(EntryType::Regular, 0o755, "bin/entry-probe", program);
    let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
    encoder.write_all(&builder.into_inner().expect("tar")).expect("gzip");
    let bytes = encoder.finish().expect("gzip finish");
    Layer { digest: Digest::of(&bytes), media_type: DOCKER_LAYER.into(), bytes }
}

fn probe_image() -> PulledImage {
    let run = RunConfig {
        entrypoint: Some(vec!["/bin/entry-probe".into()]),
        cmd: Some(vec!["--serve".into(), "8080".into()]),
        env: Some(vec!["OCI_PROBE=from-image-config".into()]),
        working_dir: Some("/srv".into()),
        user: None,
    };
    PulledImage {
        root: Digest::of(b"root"),
        manifest: Digest::of(b"manifest"),
        run,
        layers: vec![probe_layer(&common::guest("entry-probe"))],
    }
}

#[test]
fn the_image_config_reaches_the_process_as_argv_env_and_cwd() {
    let initramfs = initramfs_for(&probe_image(), &common::guest("oci-init")).expect("initramfs");

    let run = common::boot_image(&initramfs, 3000, 15_000);

    let console = String::from_utf8_lossy(&run.console);
    assert!(
        console.contains(r#"entry-probe argv=["/bin/entry-probe", "--serve", "8080"] cwd=/srv OCI_PROBE=from-image-config variables=2"#),
        "console lacks the probe line:\n{console}"
    );
    assert_eq!(run.output, b"exited:0", "oci-init reports the entrypoint's fate over COM2");
    assert!(run.timings.entrypoint.is_some(), "oci-init writes the entrypoint marker before spawning");
}

#[test]
fn a_long_lived_entrypoint_is_reported_running_when_the_linger_ends() {
    let mut image = probe_image();
    image.layers = vec![probe_layer(&common::guest("spin-init"))];
    image.run.entrypoint = Some(vec!["/bin/entry-probe".into()]);
    image.run.cmd = None;
    let initramfs = initramfs_for(&image, &common::guest("oci-init")).expect("initramfs");

    let run = common::boot_image(&initramfs, 300, 15_000);

    assert_eq!(run.output, b"running", "the entrypoint was still alive when the linger ended");
}

#[test]
fn an_entrypoint_that_is_not_in_the_rootfs_is_reported_by_init_on_the_console() {
    let mut image = probe_image();
    image.run.entrypoint = Some(vec!["/bin/absent".into()]);
    let initramfs = initramfs_for(&image, &common::guest("oci-init")).expect("initramfs");

    let error = common::try_boot_image(&initramfs, 200, 15_000).expect_err("init cannot spawn what is not there");

    let message = error.to_string();
    assert!(message.contains(r#"oci-init: spawn ["/bin/absent", "--serve", "8080"]: No such file or directory"#), "{message}");
}

#[test]
fn a_tampered_cached_layer_is_rejected_before_any_boot() {
    let cache_root = tempfile::tempdir().expect("tempdir");
    let hex = |digest: &str| digest.strip_prefix("sha256:").expect("sha256").to_string();
    for digest in [INDEX_DIGEST, MANIFEST_DIGEST, CONFIG_DIGEST, ZONEINFO_LAYER, CERTS_LAYER] {
        cache::write(cache_root.path(), &Digest::parse(digest).expect("digest"), &vendored(digest)).expect("seed cache");
    }
    let tampered = cache_root.path().join("blobs").join("sha256").join(hex(ZONEINFO_LAYER));
    let mut bytes = std::fs::read(&tampered).expect("read layer");
    let middle = bytes.len() / 2;
    bytes[middle] ^= 0xff;
    std::fs::write(&tampered, &bytes).expect("tamper");
    let booted = Cell::new(false);
    let no_network = async |_: Request<Bytes>| -> Result<Response<Bytes>, OciError> { panic!("a cached pull must not use the network") };
    let image = ImageRef::parse(&format!("traefik/whoami@{INDEX_DIGEST}")).expect("reference");

    let outcome = futures::executor::block_on(pull(&no_network, cache_root.path(), &image, "linux", "amd64"));
    if let Ok(pulled) = &outcome {
        booted.set(initramfs_for(pulled, b"init").is_ok());
    }

    let error = outcome.expect_err("tampered layer");
    let message = error.to_string();
    assert!(message.contains(&format!("expected {ZONEINFO_LAYER}")), "{message}");
    assert!(message.contains("actual sha256:"), "{message}");
    assert!(!booted.get(), "nothing may be built from a rejected pull");
}

#[test]
fn the_probe_image_archive_holds_the_entrypoint_as_an_executable_regular_file() {
    let initramfs = initramfs_for(&probe_image(), &common::guest("oci-init")).expect("initramfs");
    let mut cursor = Cursor::new(initramfs);
    let mut listing = Vec::new();
    loop {
        let reader = cpio::newc::Reader::new(&mut cursor).expect("newc header");
        if reader.entry().is_trailer() {
            break;
        }
        listing.push((reader.entry().name().to_string(), reader.entry().mode(), reader.entry().file_size()));
        reader.finish().expect("entry end");
    }

    let probe = listing.iter().find(|entry| entry.0 == "bin/entry-probe").unwrap_or_else(|| panic!("entrypoint missing from {listing:?}"));
    assert_eq!(probe.1, 0o100_755, "{listing:?}");
    assert!(probe.2 > 100_000, "the static probe is hundreds of KiB: {listing:?}");
}
