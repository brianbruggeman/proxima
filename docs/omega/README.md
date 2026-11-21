# omega tutorials

Omega runs tensor programs on a CPU or a GPU. These pages take you from one vector addition to a measured LLM decode, in the order you actually learn it: run something, change where it runs, describe a bigger program, train it, run inference, then compare against the tools you already use.

If you have ten minutes, read [00](./00-hello-tensor.md). If you want the numbers, jump to [07](./07-comparisons.md). Every page is runnable; how they were produced is at the end.

## The path

- **[00: hello, tensor](./00-hello-tensor.md)** — the whole first result in two commands: a program written in TOML, a daemon, and `sum [4] [10.0, 11.0, 12.0, 13.0]`. Then the same three ops in Rust, printing which device ran them, and the one-line config change from CPU to Metal. Start here even if you only came for inference; everything after assumes that a program is data.
- **[01: configuration in depth](./01-config.md)** — the four settings you will touch in your first week, then the other twenty-four. You will layer defaults, a file, the environment and explicit setters and see which wins per field, and read three validation failures that name the offending field. Ends with the three nested config types and the two standalone ones: the CUDA kernel cache and the CPU execution policy.
- **[02: programs as data](./02-programs-as-data.md)** — how a program file is actually structured: nodes and ids, static and symbolic shapes, the map grammar that says which element feeds which, the two leaf kinds that need no input data, and a bind list for when you do have data (`[[bind]]`, which names the tensors you will pass in). You run a bound plan, see both of its failure messages, and run a real spec from `proxima-tensor/specs/`: a rotary position embedding (RoPE).
- **[03: training](./03-train.md)** — the four lines of a PyTorch loop and the four calls that replace them. An optimizer loaded from `adam.toml`, one training step and then ten on Metal checked against the CPU, and four epochs on real MNIST: loss 0.4619 → 0.1352, 918 of 1000 test images correct.
- **[04: inference with an LLM](./04-infer-an-llm.md)** — loading a real GGUF, the three architecture facts a checkpoint does not carry, the serving config that mirrors llama.cpp's flags, and a decode on Llama-3.2-1B-Instruct f16 at 20.9 ms per token warm.
- **[05: speech, whisper](./05-speech-whisper.md)** — which pieces of a whisper pipeline exist today (log-mel front end, cross-attention, conv1d, rope tables), each shown with the test that proves it, and the exact point where an audio file would fall off the end: there is no model path yet.
- **[06: diffusion](./06-diffusion.md)** — the same treatment for image generation: adaLN modulation, non-causal attention, norm without affine and layer taps exist and are differentiable; the scheduler, the VAE decode and the image path do not.
- **[07: comparisons](./07-comparisons.md)** — omega against burn, PyTorch, llama.cpp, whisper.cpp and stable-diffusion.cpp. Every row is a measured number with its command, hardware, run count and variation, or a named gap with the missing artifact. The first number is the one that hurts: on the same 1B f16 checkpoint, omega decodes 2.2x slower than llama.cpp.

## Dependency graph

```mermaid
graph TD
  P00["00 hello, tensor"]
  P01["01 configuration in depth"]
  P02["02 programs as data"]
  P03["03 training"]
  P04["04 inference with an LLM"]
  P05["05 speech: whisper"]
  P06["06 diffusion"]
  P07["07 comparisons"]
  P00 --> P01
  P01 --> P02
  P02 --> P03
  P03 --> P04
  P02 --> P05
  P03 --> P05
  P05 --> P06
  P02 --> P06
  P04 --> P07
  P05 --> P07
  P06 --> P07
```

## Relationship to the rest of the tree

This is the operations track: tensors, training, inference. The runtime track is [`docs/tutorials/`](../tutorials/README.md), which teaches the pipe algebra and serving on the same rules. If you want the compiler internals instead — how a plan becomes a kernel, how the Metal and CUDA emitters are built — read `proxima-tensor/specs/omega-compiler-teaching/SPEC.md`.

## Gaps

The gaps a reader hits first, plus the ones that affect more than one page. Every page ends with its own full list; each row here is a fact with the line that proves it.

| gap | proof |
|---|---|
| The daemon path has no runtime-TOML surface; config is `OMEGA_*` per request | `tools/proxima-cli/src/tensor_run.rs:59` |
| `proxima tensor run` does not report which device served a plan | `tools/proxima-cli/src/tensor.rs:105` |
| Build `proximad` without `--features metal` and `OMEGA_ENGINE=gpu` is refused, not silently CPU | `tools/proxima-cli/Cargo.toml:28` |
| `numerics` is recorded and not read by lowering | `omega/src/config.rs:102` |
| Three of four optimizers have no file loader | `proxima-autograd/src/optimizer.rs:632`, `:755`, `:985` |
| A plan cannot include another plan | `proxima-tensor/src/spec/program_spec.rs:26-45` |
| The eleven production specs are not reachable from the CLI (they carry no `[[bind]]`) | `tools/proxima-cli/src/tensor.rs:41-45` |
| No whisper model path; the spec is a draft and R2 forbids a per-model profile | `proxima-model-interop/specs/whisper/SPEC.md:3`, `:26` |
| No diffusion model path; the spec is a draft, no scheduler driver or VAE decode | `proxima-model-interop/specs/flux2-klein/SPEC.md:3` |
| MLX and Core ML exist but cannot be selected by config | `omega/src/backend.rs:113`, `omega/src/config.rs:298` |
| `fit` takes every batch in memory; no streaming loader | `proxima-autograd/src/train.rs:171-178` |
| Training is a library call; `proximad` has no training route | `tools/proxima-cli/src/tensor_run.rs:94` |

## How these pages were produced

- **Nothing is used before it is taught.** The graph above is the order; a page may only use what an earlier page introduced.
- **Every fact has a line.** A type, a function or a default is cited `file:line`, and the line was opened before the sentence was written.
- **Every command shows its output.** The `text` block under a command is what that command printed here. Where nothing could be run, the page names the missing artifact instead of estimating.

`cargo val` also reports doctests in its `SUMMARY`; where a page shows two passes for a one-test filter, the second is a doctest, and the named test is the first line. Every command on every page was run on an Apple M1 Max, in `rustc`'s debug profile unless the line says `--release`. A debug wall clock is a debugging aid, never a performance claim; page 07 is the only page that makes performance claims, and each of its rows names the build profile and the load it was taken under.

Run one cargo job at a time, and check the test count rather than the exit code: `cargo val` prints `SUMMARY passed=N failed=0`, and a run of zero tests is a failure, not a pass.
