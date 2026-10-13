#![cfg(unix)]
#![allow(clippy::expect_used, clippy::unwrap_used)]

//! End-to-end `proxima tensor run` over a real proximad UDS. The remote
//! softmax output must be bit-equal to `ProgramSpec::evaluate_cpu` run
//! in-process on the same plan and inputs.

use std::collections::BTreeMap;
use std::io::BufRead;
use std::io::BufReader;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Output, Stdio};

use proxima_cli::tensor_wire::decode_response;
use proxima_safetensors::{SafetensorsModel, TensorPayload, write_complete};
use proxima_tensor::DType;
use proxima_tensor::spec::ProgramSpec;
use proxima_tensor::test_support::Lcg;
use tempfile::{TempDir, tempdir};

const PLAN_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../proxima-tensor/src/spec/fixtures/matmul_silu_softmax.toml"
);
const LISTENER_POLL_LIMIT: usize = 1_000;

struct Daemon {
    child: Child,
    stdout: BufReader<ChildStdout>,
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn spawn_daemon(socket: &Path, state_dir: &Path) -> Daemon {
    let mut child = Command::new(env!("CARGO_BIN_EXE_proximad"))
        .arg("serve")
        .arg("--unix")
        .arg(socket)
        .arg("--state-dir")
        .arg(state_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn proximad");
    let stdout = child.stdout.take().expect("proximad stdout is piped");
    let mut daemon = Daemon {
        child,
        stdout: BufReader::new(stdout),
    };
    let mut ready_line = String::new();
    daemon
        .stdout
        .read_line(&mut ready_line)
        .expect("read proximad READY line");
    assert!(
        ready_line.starts_with("READY "),
        "first line must be READY, got: {ready_line:?}"
    );
    daemon
}

fn await_listener(socket: &Path) {
    let accepted = (0..LISTENER_POLL_LIMIT).any(|_| UnixStream::connect(socket).is_ok());
    assert!(
        accepted,
        "proximad never accepted on {socket:?} within {LISTENER_POLL_LIMIT} polls"
    );
}

fn run_cli(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_proxima"))
        .args(args)
        .output()
        .expect("spawn proxima cli")
}

fn plan_text() -> String {
    std::fs::read_to_string(PLAN_PATH).expect("matmul-silu-softmax fixture is readable")
}

fn matmul_inputs() -> (Vec<f32>, Vec<f32>) {
    let mut lcg = Lcg(0x5eed_0003);
    let x: Vec<f32> = (0..8).map(|_| lcg.next_unit() - 0.5).collect();
    let w: Vec<f32> = (0..12).map(|_| lcg.next_unit() - 0.5).collect();
    (x, w)
}

fn f32_le(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn write_inputs(path: &Path, x: &[f32], w: &[f32]) {
    let x_bytes = f32_le(x);
    let w_bytes = f32_le(w);
    let bytes = write_complete(&SafetensorsModel {
        tensors: vec![
            TensorPayload {
                name: "x".to_string(),
                dtype: DType::Float32,
                shape: vec![2, 4],
                data: &x_bytes,
            },
            TensorPayload {
                name: "w".to_string(),
                dtype: DType::Float32,
                shape: vec![4, 3],
                data: &w_bytes,
            },
        ],
        metadata: BTreeMap::new(),
    })
    .expect("inputs body writes");
    std::fs::write(path, bytes).expect("inputs file writes");
}

fn socket_in(dir: &TempDir) -> PathBuf {
    dir.path().join("proximad.sock")
}

#[test]
fn tensor_run_returns_softmax_bit_equal_to_in_process() {
    let state = tempdir().expect("state dir");
    let sockets = tempdir().expect("socket dir");
    let socket = socket_in(&sockets);
    let _daemon = spawn_daemon(&socket, state.path());
    await_listener(&socket);
    let inputs_path = sockets.path().join("inputs.safetensors");
    let output_path = sockets.path().join("outputs.safetensors");
    let (x, w) = matmul_inputs();
    write_inputs(&inputs_path, &x, &w);

    let output = run_cli(&[
        "tensor",
        "run",
        "--socket",
        socket.to_str().expect("socket path is utf-8"),
        PLAN_PATH,
        "--inputs",
        inputs_path.to_str().expect("inputs path is utf-8"),
        "--output",
        output_path.to_str().expect("output path is utf-8"),
    ]);

    assert_eq!(
        output.status.code(),
        Some(0),
        "cli exits 0; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stdout).trim_end(),
        "softmax [2, 3] 6 floats",
        "stdout names the one output, its shape and its float count"
    );
    let remote = decode_response(&std::fs::read(&output_path).expect("output file exists"))
        .expect("output file decodes");
    assert_eq!(remote.len(), 1, "the fixture declares one output");
    let (name, shape, remote_values) = &remote[0];
    assert_eq!(name, "softmax");
    assert_eq!(shape.as_slice(), &[2_u64, 3]);
    let spec: ProgramSpec = toml::from_str(&plan_text()).expect("plan parses");
    let local = spec
        .evaluate_cpu(&[("x", &x), ("w", &w)])
        .expect("in-process evaluation succeeds");
    let local_values = &local[0].2;
    assert_eq!(remote_values.len(), 6, "six output floats came back");
    let remote_bits: Vec<u32> = remote_values.iter().map(|value| value.to_bits()).collect();
    let local_bits: Vec<u32> = local_values.iter().map(|value| value.to_bits()).collect();
    assert_eq!(
        remote_bits, local_bits,
        "remote softmax is bit-equal to in-process evaluate_cpu"
    );
}

#[test]
fn tensor_run_version_two_plan_exits_nonzero_with_named_error() {
    let state = tempdir().expect("state dir");
    let sockets = tempdir().expect("socket dir");
    let socket = socket_in(&sockets);
    let _daemon = spawn_daemon(&socket, state.path());
    await_listener(&socket);
    let original = plan_text();
    let version_two = original.replace("version = 1", "version = 2");
    assert_ne!(version_two, original, "the fixture declares version = 1");
    let plan_path = sockets.path().join("version_two.toml");
    std::fs::write(&plan_path, version_two).expect("plan writes");

    let output = run_cli(&[
        "tensor",
        "run",
        "--socket",
        socket.to_str().expect("socket path is utf-8"),
        plan_path.to_str().expect("plan path is utf-8"),
    ]);

    assert!(!output.status.success(), "a version 2 plan must not exit 0");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("version"),
        "stderr names the version field: {stderr}"
    );
}

#[test]
fn tensor_run_unreachable_socket_reports_connect_error() {
    let sockets = tempdir().expect("socket dir");
    let missing = socket_in(&sockets);

    let output = run_cli(&[
        "tensor",
        "run",
        "--socket",
        missing.to_str().expect("socket path is utf-8"),
        PLAN_PATH,
    ]);

    assert!(
        !output.status.success(),
        "a socket nobody listens on must not exit 0"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("connect"),
        "stderr names the failed connect: {stderr}"
    );
}
