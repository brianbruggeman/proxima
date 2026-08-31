//! End-to-end proof for the wire-to-weights combo demo: serves the real
//! composed pipe (`examples/support/wire_to_weights_pipeline.rs`) over a
//! real tokio-free h1 listener (`PrimeServeExt::serve_http`, the same
//! primitive `tests/h1_native_prime.rs` proves), then drives it with real
//! HTTP/1.1 requests over a plain blocking `std::net::TcpStream` — the
//! same client shape `tests/h1_native_prime.rs` and
//! `examples/h1_native_prime_round_trip.rs` both use, since neither needs
//! anything a raw socket doesn't already give a real client (no cookies,
//! no redirects, no connection pooling to exercise here).
//!
//! Sends real `t10k` MNIST test images (not synthetic pixels), normalized
//! exactly as `proxima-onnx/tests/real_mnist_accuracy.rs` normalizes them,
//! and checks the served argmax against the real label.
//!
//! `#[ignore]`d and skips cleanly when either the real `mnist.onnx`
//! checkout or the real MNIST idx dataset is absent — the same convention
//! `real_mnist_accuracy.rs` uses for its own host-local fixtures.

#![cfg(feature = "wire-to-weights-demo")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

#[path = "../examples/support/wire_to_weights_pipeline.rs"]
mod pipeline;

use std::fs;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use proxima::prime::PrimeRuntime;
use proxima::runtime::PrimeServeExt;

const MODEL_PATH: &str = "/Users/brianbruggeman/repos/others/burn/examples/onnx-inference/src/model/mnist.onnx";
const DATASET_DIR: &str = "/Users/brianbruggeman/.cache/burn-dataset/mnist";
const SAMPLE_COUNT: usize = 50;

fn test_images_path() -> PathBuf {
    Path::new(DATASET_DIR).join("test/t10k-images-idx3-ubyte")
}

fn test_labels_path() -> PathBuf {
    Path::new(DATASET_DIR).join("test/t10k-labels-idx1-ubyte")
}

fn dataset_present() -> bool {
    test_images_path().exists() && test_labels_path().exists()
}

/// Same idx big-endian header walk `real_mnist_accuracy.rs::idx_header`
/// uses — magic, item count, then per-axis extents.
fn idx_header(bytes: &[u8]) -> (usize, Vec<usize>) {
    let dimension_count = bytes[3] as usize;
    let item_count = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let mut extents = Vec::with_capacity(dimension_count - 1);
    for axis in 1..dimension_count {
        let offset = 4 + axis * 4;
        extents.push(u32::from_be_bytes([bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]]) as usize);
    }
    (item_count, extents)
}

/// Every sampled test image, normalized exactly as `real_mnist_accuracy.rs`
/// normalizes: `(pixel/255 - 0.1307)/0.3081`, serialized as raw
/// little-endian `f32` bytes — the wire shape `pipeline::ParseImage`
/// expects.
fn load_normalized_image_bodies(path: &Path, limit: usize) -> Vec<Vec<u8>> {
    let bytes = fs::read(path).expect("read idx3 image file");
    let (item_count, extents) = idx_header(&bytes);
    let pixel_count = extents.iter().product::<usize>();
    let take = item_count.min(limit);
    let header_length = 4 + extents.len() * 4 + 4;
    (0..take)
        .map(|image_index| {
            let start = header_length + image_index * pixel_count;
            let raw = &bytes[start..start + pixel_count];
            let mut body = Vec::with_capacity(pixel_count * 4);
            for &pixel in raw {
                let normalized = ((pixel as f32 / 255.0) - 0.1307) / 0.3081;
                body.extend_from_slice(&normalized.to_le_bytes());
            }
            body
        })
        .collect()
}

fn load_labels(path: &Path, limit: usize) -> Vec<u8> {
    let bytes = fs::read(path).expect("read idx1 label file");
    let (item_count, _extents) = idx_header(&bytes);
    let take = item_count.min(limit);
    bytes[8..8 + take].to_vec()
}

/// Sends one real HTTP/1.1 `POST /classify` over a blocking TCP socket
/// and returns the parsed `(status, body)`.
fn post_classify(addr: std::net::SocketAddr, body: &[u8]) -> (u16, String) {
    let mut stream = TcpStream::connect(addr).expect("connect to the served listener");
    let head = format!("POST /classify HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n", body.len());
    stream.write_all(head.as_bytes()).expect("write request head");
    stream.write_all(body).expect("write request body");
    stream.set_read_timeout(Some(Duration::from_secs(5))).expect("set read timeout");

    let mut raw = Vec::new();
    let mut buffer = [0_u8; 8192];
    let (status, header_length, content_length) = loop {
        let read = stream.read(&mut buffer).expect("read response bytes");
        assert!(read > 0, "connection closed before a full response head arrived");
        raw.extend_from_slice(&buffer[..read]);
        let text = String::from_utf8_lossy(&raw);
        let Some(head_end) = text.find("\r\n\r\n") else {
            continue;
        };
        let status_line = text.lines().next().expect("status line present");
        let status: u16 = status_line.split_whitespace().nth(1).expect("status code present").parse().expect("status code parses");
        let content_length: usize = text
            .lines()
            .find_map(|line| line.to_ascii_lowercase().strip_prefix("content-length:").map(|value| value.trim().to_string()))
            .expect("content-length header present")
            .parse()
            .expect("content-length parses");
        break (status, head_end + 4, content_length);
    };

    while raw.len() < header_length + content_length {
        let read = stream.read(&mut buffer).expect("read response body bytes");
        assert!(read > 0, "connection closed before the full body arrived");
        raw.extend_from_slice(&buffer[..read]);
    }

    let body_text = String::from_utf8_lossy(&raw[header_length..header_length + content_length]).into_owned();
    (status, body_text)
}

fn parse_digit(body: &str) -> usize {
    let marker = "\"digit\":";
    let start = body.find(marker).expect("response carries a digit field") + marker.len();
    let end = body[start..].find('}').map_or(body.len(), |offset| start + offset);
    body[start..end].trim().parse().expect("digit field parses as an integer")
}

#[test]
#[ignore = "depends on a real .onnx checkout and the real MNIST idx dataset outside this repo"]
fn wire_to_weights_classifies_real_mnist_images_over_real_http() {
    let model_path = Path::new(MODEL_PATH);
    if !model_path.exists() {
        eprintln!("skipping: no host-local mnist.onnx checkout at {MODEL_PATH}");
        return;
    }
    if !dataset_present() {
        eprintln!("skipping: no host-local MNIST idx dataset under {DATASET_DIR}");
        return;
    }

    let model = pipeline::load_model(model_path).expect("load the real mnist.onnx checkpoint").expect("checkpoint present, load_model returns Some");
    let handler = pipeline::build_handler(Arc::new(model));

    let runtime = Arc::new(PrimeRuntime::builder().cores(1).background_inline().build().expect("build prime runtime"));
    let bind = "127.0.0.1:0".parse().expect("bind addr parses");
    let handle = runtime.serve_http(bind, handler).expect("serve_http binds through the prime AcceptorFactory");
    let addr = handle.bind_addr().expect("listener reports its bound address");

    let bodies = load_normalized_image_bodies(&test_images_path(), SAMPLE_COUNT);
    let labels = load_labels(&test_labels_path(), SAMPLE_COUNT);
    assert_eq!(bodies.len(), labels.len(), "same number of images and labels");
    assert!(bodies.len() >= SAMPLE_COUNT, "expected at least {SAMPLE_COUNT} real test images, got {}", bodies.len());

    let mut correct = 0_usize;
    let mut total_wall_clock = Duration::ZERO;
    let mut transcript: Option<(Vec<u8>, u16, String)> = None;

    for (index, (body, &label)) in bodies.iter().zip(labels.iter()).enumerate() {
        let start = Instant::now();
        let (status, response_body) = post_classify(addr, body);
        total_wall_clock += start.elapsed();
        assert_eq!(status, 200, "request {index} expected 200, got {status}: {response_body}");

        if transcript.is_none() {
            transcript = Some((body.clone(), status, response_body.clone()));
        }

        let predicted = parse_digit(&response_body);
        if predicted == label as usize {
            correct += 1;
        }
    }

    let accuracy = correct as f64 / bodies.len() as f64;
    let average_latency = total_wall_clock / bodies.len() as u32;
    eprintln!("wire_to_weights: accuracy {accuracy:.4} ({correct}/{}) over real HTTP, average {average_latency:?}/request", bodies.len());

    if let Some((request_body, status, response_body)) = &transcript {
        eprintln!(
            "wire_to_weights transcript: POST /classify ({} bytes) -> {status} {response_body}",
            request_body.len()
        );
    }

    handle.shutdown();

    assert!(accuracy >= 0.90, "expected wire_to_weights to classify at least 90% of {} real test images over real HTTP, got {accuracy:.4}", bodies.len());
}
