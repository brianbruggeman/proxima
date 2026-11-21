# Omega 03: training

**Prerequisites:** [page 02](./02-programs-as-data.md) (a program as data, `Reduce` as a matmul, outputs by `NodeId`).

**You will:** map the four lines of a PyTorch training loop onto four calls, load the optimizer from a TOML file, run ten steps on Metal at CPU parity, and run four epochs on real MNIST with the loss curve and the test accuracy it reached.

**New concepts (in order):** forward is a program · loss as a node: `mse`, `cross_entropy`, `softmax_cross_entropy` · backward as `differentiate` · the step as `adam_step`, `train_step` and `fit` · `train::State` · `AdamConfig` from a TOML file · the same step on Metal · what omega does not do yet.

## 1. The loop, mapped

A PyTorch training loop is four lines. Here is the same four, with the crate that owns each:

```text
logits = model(x)              forward     the program you already built
loss   = cross_entropy(...)    loss        proxima_autograd::loss::softmax_cross_entropy
loss.backward()                backward    proxima_autograd::adjoint::differentiate
opt.step()                     step        proxima_autograd::optimizer::adam_step, or fit
```

There is no `Module`, no `Parameter`, and no `.backward()` on a tensor. A model is a `Vec<Op>` whose leaves are `Op::Input`s, and a parameter is an input you keep re-binding by name.

The three losses are `mse` (`proxima-autograd/src/loss.rs:32`), `cross_entropy` (`:103`) and `softmax_cross_entropy` (`:153`). Each takes the program under construction and the ids it needs, appends its own nodes, and returns the `NodeId` of the scalar it built — so a loss is a node in the same graph, not a separate layer.

`differentiate` (`adjoint.rs:214`) takes the program and the loss node and returns a `Differentiated` (`:137`), which is the backward program plus the map from a forward node to its gradient node (`gradient_of`, `:163`). It is a graph transform: the backward pass is built once, not recorded per step.

The step is nine elementwise nodes over `(param, grad, m, v)` (`optimizer.rs:1-6`). `adam_step` (`:368`) appends them; `step_input` (`:336`) gives you the rank-0 `step` input that bias correction needs; `AdamOperands` (`:354`) is the four ids.

## 2. The optimizer, as a TOML file

`AdamConfig` (`optimizer.rs:233`) has six hyper-parameters and three layouts, and it is a config in both senses: a builder and a file. `proxima-autograd/examples/adam.toml` is the file this page uses:

```toml
learning_rate = 0.01
beta1 = 0.95
beta2 = 0.999
epsilon = 1.0e-8
first_moment_layout = "float32"
second_moment_layout = "float32"
```

`AdamConfig::from_file` (`optimizer.rs:293`) loads it and validates it, naming the field it refuses. The call is the one the test makes, `AdamConfig::from_file(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/adam.toml"))` — the shipped file, so the file and the test cannot drift apart.

A zero step size never reaches a step: `Validate` (`optimizer.rs:317`) rejects it with a message that names `learning_rate`, which the test asserts by substring rather than by trusting the whole sentence.

The test that proves both halves is `adam_config_from_file_loads_the_shipped_example_and_refuses_a_zero_step_size`:

```sh
cargo val -p proxima-autograd -E 'test(adam_config_from_file)'
```

```text
PASS proxima-autograd optimizer_config::adam_config_from_file_loads_the_shipped_example_and_refuses_a_zero_step_size
SUMMARY  passed=3 failed=0 ignored=0 not_run=0  wall=3.493s discovered=203 run=3 filtered=200 binaries=24
```

The other two passes in that summary are doctests; the first line is the test.

The layouts matter for the same reason they matter in PyTorch's `foreach` fused optimizers: `first_moment_layout` and `second_moment_layout` decide the storage of `m` and `v` between steps, and `hyper_layout` decides the width the hyper-parameters are rounded to before they become graph constants. All three are config values over `proxima-numeric` layouts, so a bf16 moment is a file edit, not a type change.

## 3. One step on the GPU, checked against the CPU

`train_step` (`train.rs:69`) and `train_step_with_arena` (`train.rs:110`) run one step; `fit` (`train.rs:171`) runs epochs over a batch list and returns the final `State` plus the loss per batch. `State` is `Vec<(String, Vec<f32>)>` (`train.rs:49`) — named buffers, the same binding-by-name idea as a plan's `[[bind]]`.

Training reaches a GPU through the same `plan_named`/`execute_plan_named` pair a forward pass uses; nothing in the training path knows about a device. `omega/tests/training_step_parity.rs` is the proof, and CPU is the oracle every GPU arm is compared against:

```sh
cargo val -p proxima-omega --features metal \
  -E 'test(a_training_step_runs_on_metal_at_cpu_parity) or test(ten_training_steps_rebind_state_and_the_loss_drops_on_metal)'
```

```text
PASS proxima-omega::training_step_parity a_training_step_runs_on_metal_at_cpu_parity
PASS proxima-omega::training_step_parity ten_training_steps_rebind_state_and_the_loss_drops_on_metal
SUMMARY  passed=4 failed=0 ignored=0 not_run=0  wall=2.760s discovered=1569 run=4 filtered=1565 binaries=166
```

The second test is the one that matters for a real loop: it rebinds `m`, `v`, `param` and `step` across ten steps and asserts the loss drops ([`training_step_parity.rs:652`](../../omega/tests/training_step_parity.rs)).

## 4. Four epochs on real MNIST

`proxima-autograd/tests/real_mnist_training.rs` trains a 784-128-10 MLP with `fit` on the idx files in `~/.cache/burn-dataset/mnist` — the same normalized data burn's own MNIST example reads. It runs 250 batches per epoch at batch 32 over 8000 training images, for four epochs, then classifies 1000 test images.

```sh
cargo nextest run -p proxima-autograd \
  -E 'test(real_mnist_mlp_trains_and_classifies_at_reference_accuracy)' --no-capture
```

```text
real_mnist_training: 250 batches/epoch x 4 epochs, batch=32, train_examples=8000
real_mnist_training loss curve: first-epoch-avg=0.4619 last-epoch-avg=0.1352 wall_clock=108.479117375s
real_mnist_training test accuracy: 0.9180 (918/1000 images), total wall_clock=109.199250708s
```

`cargo val` is the usual runner here, but it does not forward `--no-capture`, and this test prints its numbers on stderr — so the command above calls nextest directly to keep the output. The test itself is the same one `cargo val -p proxima-autograd` runs.

Read it honestly: the loss fell from 0.4619 to 0.1352 averaged over the first and last epoch, and 918 of 1000 held-out images were classified correctly. The wall clock is a debug build and is not a performance claim; page 07 is where timing is measured, against an incumbent, in release.

The dataset is a host-local path constant (`real_mnist_training.rs:65`); the test skips rather than fails when the files are absent, which is why a green run also means the data was there.

## 5. What omega does not do yet

- **There is no optimizer trait.** `adam_step` and its three sibling configs are functions, not an `impl Optimizer`; adding a new optimizer is a new graph-building function (`optimizer.rs:1-14` explains why: the step is a graph, not an object).
- **Only Adam has a file loader.** `AdamConfig::from_file` exists (`optimizer.rs:293`); `SgdConfig` (`:632`), `RmspropConfig` (`:755`) and `AdamwConfig` (`:985`) have builders and env layering but no `from_file`. Adding one is the same six lines.
- **`fit` takes every batch in memory.** Its signature is `batches: &[Vec<(&str, &[f32])>]` (`train.rs:177`), so a dataset larger than RAM needs a caller-driven loop around `train_step` instead. There is no streaming data loader.
- **No mixed-precision training path.** The moments and the hyper-parameters have layouts, but the activations and gradients in the graph are float32; `numerics` is still recorded and not read (page 01, section 6).
- **Training is a library call, not a daemon route.** `proximad`'s tensor route only evaluates a plan (`tools/proxima-cli/src/tensor_run.rs:94`); nothing in the CLI submits a training step.

Next: [04: inference with an LLM](./04-infer-an-llm.md) loads a real GGUF checkpoint, reads its architecture as data, generates tokens, and measures milliseconds per token.
