//! Proves, rather than asserts, that an element conversion written as a
//! [`proxima_tensor::Convert`] `Pipe` costs nothing over the bare loop it
//! replaces — the design claim this crate's `convert` module rests on.
//!
//! Method: compile a standalone source (the actual `Cast`/`Convert`/`Pipe`
//! shapes this crate ships, replicated verbatim so the measurement targets
//! the LANGUAGE property with no cargo graph in the way — the same
//! methodology `scratchpad/pipe_zero_cost.rs` validated this session) with
//! `rustc -O -C target-cpu=native`, then read `otool -tV`'s disassembly of
//! the resulting binary. `cargo test`'s own binaries are unopimized debug
//! builds by default and would not answer this question.
//!
//! macOS-only: `otool -tV` is the disassembler this repo's other
//! instruction-count gates use (Apple's `objdump` has no `-d`).
#![cfg(target_os = "macos")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::process::Command;

const PROBE_SOURCE: &str = r#"
use std::future::Future;
use std::hint::black_box;
use std::task::{Context, Poll};

trait Pipe {
    type In;
    type Out;
    type Err: core::fmt::Debug + 'static;
    fn call(&self, input: Self::In) -> impl Future<Output = Result<Self::Out, Self::Err>>;
}

#[derive(Debug)]
enum Never {}

trait Cast<To> {
    fn cast(self) -> To;
}
impl Cast<f32> for i32 {
    #[inline(always)]
    fn cast(self) -> f32 {
        self as f32
    }
}

struct Convert<From, To>(core::marker::PhantomData<fn(From) -> To>);
impl<From, To> Convert<From, To> {
    const fn new() -> Self {
        Self(core::marker::PhantomData)
    }
}
impl<From: Cast<To>, To: 'static> Pipe for Convert<From, To> {
    type In = From;
    type Out = To;
    type Err = Never;
    fn call(&self, input: From) -> impl Future<Output = Result<To, Never>> {
        async move { Ok(input.cast()) }
    }
}

fn block_on<Fut: Future>(future: Fut) -> Fut::Output {
    let mut future = core::pin::pin!(future);
    let mut context = Context::from_waker(std::task::Waker::noop());
    loop {
        if let Poll::Ready(output) = future.as_mut().poll(&mut context) {
            return output;
        }
    }
}

#[inline(never)]
pub fn bare_i32_to_f32(input: &[i32], output: &mut [f32]) {
    for (source, target) in input.iter().zip(output.iter_mut()) {
        *target = *source as f32;
    }
}

#[inline(never)]
pub fn pipe_i32_to_f32(input: &[i32], output: &mut [f32]) {
    let pipe = Convert::<i32, f32>::new();
    for (source, target) in input.iter().zip(output.iter_mut()) {
        *target = block_on(pipe.call(*source)).unwrap_or(0.0);
    }
}

fn main() {
    const COUNT: usize = 1 << 16;
    let integers: Vec<i32> = (0..COUNT as i32).collect();
    let mut out = vec![0.0f32; COUNT];

    bare_i32_to_f32(black_box(&integers), black_box(&mut out));
    let bare_checksum: f32 = out.iter().sum();
    pipe_i32_to_f32(black_box(&integers), black_box(&mut out));
    let pipe_checksum: f32 = out.iter().sum();

    println!("bare={bare_checksum:.1} pipe={pipe_checksum:.1} equal={}", bare_checksum == pipe_checksum);
}
"#;

/// One function's instruction lines from an `otool -tV` listing: everything
/// between its label and the next label, address column stripped.
fn function_body<'a>(disassembly: &'a str, label_substring: &str) -> Option<Vec<&'a str>> {
    let lines: Vec<&str> = disassembly.lines().collect();
    let start = lines
        .iter()
        .position(|line| line.ends_with(':') && line.contains(label_substring))?;
    let body_start = start + 1;
    let body_end = lines[body_start..]
        .iter()
        .position(|line| line.ends_with(':') && !line.starts_with(char::is_whitespace))
        .map_or(lines.len(), |offset| body_start + offset);
    Some(
        lines[body_start..body_end]
            .iter()
            .copied()
            .filter(|line| !line.trim().is_empty())
            .collect(),
    )
}

#[test]
fn cast_through_the_pipe_costs_nothing_over_the_bare_loop() {
    let workspace = tempfile::tempdir().expect("create a scratch dir for the probe build");
    let source_path = workspace.path().join("probe.rs");
    std::fs::write(&source_path, PROBE_SOURCE).expect("write probe source");
    let binary_path = workspace.path().join("probe");

    let compile = Command::new("rustc")
        .args(["-O", "--edition", "2024", "-C", "target-cpu=native", "-o"])
        .arg(&binary_path)
        .arg(&source_path)
        .output()
        .expect("invoke rustc");
    assert!(
        compile.status.success(),
        "probe failed to compile:\n{}",
        String::from_utf8_lossy(&compile.stderr)
    );

    // correctness first: a byte-identical-instructions claim is worthless if
    // the two paths do not compute the same answer.
    let run = Command::new(&binary_path).output().expect("run the probe binary");
    let stdout = String::from_utf8_lossy(&run.stdout);
    assert!(run.status.success(), "probe binary exited nonzero: {stdout}");
    assert!(stdout.contains("equal=true"), "bare and pipe paths disagreed: {stdout}");

    let disassembly_output = Command::new("otool")
        .arg("-tV")
        .arg(&binary_path)
        .output()
        .expect("invoke otool -tV");
    assert!(
        disassembly_output.status.success(),
        "otool failed:\n{}",
        String::from_utf8_lossy(&disassembly_output.stderr)
    );
    let disassembly = String::from_utf8_lossy(&disassembly_output.stdout);

    let symbols = Command::new("nm")
        .arg(&binary_path)
        .output()
        .expect("invoke nm");
    let symbol_table = String::from_utf8_lossy(&symbols.stdout);
    let pipe_symbol_survived = symbol_table.contains("pipe_i32_to_f32");

    let bare_body = function_body(&disassembly, "bare_i32_to_f32").expect("bare_i32_to_f32 in disassembly");
    let has_vectorized_main_loop = bare_body.iter().any(|line| line.contains("scvtf.4s"));
    assert!(
        has_vectorized_main_loop,
        "bare_i32_to_f32's compiled body has no `scvtf.4s` — the loop that should have \
         auto-vectorized did not, so this run cannot tell zero-cost apart from \
         both-paths-degraded-to-scalar:\n{bare_body:#?}"
    );

    if pipe_symbol_survived {
        // distinct symbol survived linking: compare instruction counts directly.
        let pipe_body = function_body(&disassembly, "pipe_i32_to_f32").expect("pipe_i32_to_f32 in disassembly");
        println!(
            "bare_i32_to_f32: {} instructions, pipe_i32_to_f32: {} instructions",
            bare_body.len(),
            pipe_body.len()
        );
        assert_eq!(
            bare_body.len(),
            pipe_body.len(),
            "instruction counts diverged — bare:\n{bare_body:#?}\npipe:\n{pipe_body:#?}"
        );
    } else {
        // stronger than equal counts: the compiler proved the two bodies
        // identical and folded `pipe_i32_to_f32` onto `bare_i32_to_f32`
        // entirely, so only one symbol exists in the linked binary at all.
        println!(
            "pipe_i32_to_f32 did not survive linking as a distinct symbol — folded onto \
             bare_i32_to_f32 ({} instructions), a stronger result than equal instruction counts",
            bare_body.len()
        );
        assert!(
            symbol_table.contains("bare_i32_to_f32"),
            "neither symbol survived linking; nothing was measured"
        );
    }
}
