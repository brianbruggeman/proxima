# fp6-qat-ablation

status: audited
owner: Brian Bruggeman
created: 2026-10-11

## problem

On this Mac's GPU (Metal, the only machine), a 256-step quantization-aware fine-tune of granite 3.1 1b A400M (bf16-grid masters dequantized from the Q8_0 checkpoint, f32 compute, Adam learning rate 1e-4, seed 17, 4 sequences of 256 tokens per step, 1,024 tokens per step) with e3m2 fake-quant (OCP MX element s1e3m2, f16 absmax scale per 32, straight-through backward) on the 168 expert and attention projection tensors, trained on the first 90% of the paragraphs of `war_and_peace.txt` and evaluated on the last 10%, is measured by these bounds: final held-out loss (64 sequences of 256 tokens) no more than 0.05 nats per token above the same-seed fine-tune with fake-quant off (the control) and below the control's weights quantized to e3m2 after training with no QAT (the ptq arm); positional token agreement of the converted QAT checkpoint against the recorded llama ids (224 positions) no more than `floor(0.02 x 224) = 4` positions below the unconverted control checkpoint's own count; and, before any of that, the fake-quant-off training throughput (ms per step, tokens per second, peak RSS, peak GPU bytes, CoV) measured at one layer and at all 24 layers, which replaces every derivation of how long the fine-tune takes.

## refutation condition

The ablation is refuted as a basis for deciding whether e3m2 can be trained into granite on this machine if any one of these observations holds:

- the AC12 relation `final held-out loss(qat) - final held-out loss(control) <= 0.05` prints `holds=false`; the 0.05 nats per token is a pre-registered gap, status plausible, not derived from any measurement in the tree;
- the AC12 relation `final held-out loss(qat) < final held-out loss(ptq_eval)` prints `holds=false` (QAT recovers nothing over quantizing after training);
- the AC10 identity control prints `within=false` (two fake-quant-off runs with the same seed disagree by more than 0.001 nats per token, so the run-to-run noise exceeds the effect being measured and every other relation is void);
- the AC14 token-agreement example prints `within=false` for the `qat` cell;
- AC1 prints a nonzero `metal_reject`, or AC2 prints a `step_bench_refused` line for 24 layers (the full model does not run on this GPU); these are recorded outcomes that move the work to the rejected ops or to a memory design, they are not failures of the measurement.

Any one refutes the relation it names. Refuting output is recorded, never edited. The throughput and memory numbers of AC2, AC15 and AC16 stand on their own whatever the quality relations print.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | The Metal backward census: a new example lowers the granite config (one layer, then 24) through `build_forward`, appends softmax cross-entropy over 256 target ids, differentiates it with `proxima_autograd::adjoint::differentiate`, classifies every node of the resulting program (forward and adjoint) through omega's Metal emit, and prints one line per node plus one census line per cell; the count of nodes Metal rejects is zero for both cells | yes |
| R2 | The measurement first: a 64-step Metal training run at granite's real dims, fake-quant off, bf16-grid masters, 1,024 tokens per step, at 1 layer and at 24 layers, 3 processes per cell, step 1 excluded as plan build, recording median ms per step with MAD, CoV, tokens per second, peak RSS, peak footprint and peak GPU bytes; a cell that cannot allocate prints a refusal line naming the failed allocation | yes |
| R3 | The e3m2 fake-quant function: quantize then dequantize through `scaled_float.rs` (`quantize_block`, `dequantize_block`, `FloatLayout::E3M2`) over 32-element blocks along the contiguous axis of every tensor `policy_selects` names (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`, `attn_q`, `attn_k`, `attn_v`, `attn_output`), with no branch on a model or tensor name beyond that policy | yes |
| R4 | The straight-through backward: the training program computes the effective weight as `Add(master, delta)` over two `Input` leaves, `delta = fake_quant(master) - master` bound each step, so the existing adjoint of `Add` delivers the effective weight's gradient unchanged to the master | yes |
| R5 | The Metal fake-quant program: the same function expressed as a graph over existing `ScalarOp`s and executed on Metal (block absmax by `Reduce(Maximum)`, f16 scale by an `Elementwise` of dtype `Float16`, grid rounding by constant add and subtract, no new op and no new kernel) equals the R3 function bit for bit | yes |
| R6 | The bf16-grid rounding program: the master update's rounding to the bf16 grid, expressed as a graph over existing `ScalarOp`s and executed on Metal, equals `half::bf16::from_f32` bit for bit | yes |
| R7 | The corpus pin: `proxima-model-interop/examples/data/war_and_peace.txt` (3,274,130 bytes, sha256 pinned in the test), split by paragraph at 90% (train) and 10% (held-out) with the two splits exhaustive and disjoint, tokenized by the granite tokenizer whose output on the four parity prompts reproduces the llama-recorded `prompt_ids`, with bytes and token counts printed per split and the held-out split holding at least the 16,384 tokens the evaluation consumes | yes |
| R8 | The token-agreement prompts are disjoint from the training text: none of the four prompts in the granite parity fixtures occurs in the corpus | yes |
| R9 | The control is the same experiment minus fake-quant: same seed, same initial masters, same data order, and programs that differ only by the `delta` leaves | yes |
| R10 | The identity control: two fake-quant-off runs with seed 17 print 256 train-loss lines and 9 held-out-loss lines each and agree on final held-out loss within 0.001 nats per token | yes |
| R11 | The QAT run and the ptq arm: the same fine-tune with fake-quant on prints 256 train-loss and 9 held-out-loss lines, and the ptq arm evaluates the control's final masters under fake-quant with no training | yes |
| R12 | The pre-registered loss relations evaluate on the saved outputs: `qat - control <= 0.05`, `qat < ptq_eval`, zero non-finite values in every curve, and the degenerate control `control_repeat - control <= 0.05` (the reference run through the same check, which must hold) | yes |
| R13 | Checkpoint export: final masters written as an F32 GGUF (exact widening of the bf16 grid) that reloads with every tensor present | yes |
| R14 | Positional token agreement against the recorded llama ids (224 positions) of the converted QAT checkpoint (`qat` cell) and of the control checkpoint converted after training with no QAT (`ptq` cell), each with the unconverted control checkpoint's own count as the reference, plus the control checkpoint against itself (`ctl` cell, gap 0), through the fp6 spec's `codec_token_agreement` example unchanged | yes |
| R15 | Duration: wall time per training run and tokens per second, the example's own clock agreeing with `/usr/bin/time -l` within 2% | yes |
| R16 | Memory: peak RSS, peak footprint and peak GPU bytes per training run with `memory_cap=none`, the example's RSS agreeing with `/usr/bin/time -l` within 5% | yes |
| R17 | The ablation result section of this SPEC, holding one row per arm and the list of unexplained observations | yes |
| R18 | No new type in any library crate unless R1 forces one | yes |
| R19 | The loss is the incumbent's: the teacher-forced negative log-probability the step graph's forward assigns at step 0 (fake-quant off, the checkpoint's masters) to the 96 generated ids recorded in `n_probs.json` (3 cases x 32 steps) agrees with llama.cpp's recorded `logprob` at each position, mean absolute difference at most 0.02 nats and maximum at most 0.1 nats (both pre-registered, status plausible) | yes |

## architecture

The owner's two questions have one answer each, bound by reading, not by estimate.

How long: UNMEASURED. Every training record in the tree is a 32-parameter toy: two experts of `[4,4]` (`proxima-tensor/specs/low_precision_moe_training/SPEC.md:35`). The CPU pilots recorded `wall_time_ns` per step (mean 2,088,894 ns for the fp32 arm, 9,021,112 to 9,084,544 ns for the six packed arms, averaged over the step records of `proxima-tensor/specs/low_precision_moe_training/results/tiny-pilot.json` and `proxima-tensor/specs/packed_pilot_moe_training/results/packed-tiny-pilot.json` with `jq`), and the record keys of `proxima-tensor/specs/metal_packed_pilot_moe_64step/results/metal-packed-pilot-64step.json` hold 0 matches for wall, time, ms or rss. Nothing predicts granite; R2 measures it.

GPU: yes. `Differentiated.program` is an ordinary `proxima_tensor` program, and omega executes it on Metal through `execute_plan_named` (`proxima-tensor/specs/metal_packed_pilot_moe_64step/SPEC.md`, admitted, 64 steps). `proxima-autograd` has no Metal code and needs none: `git grep -i metal origin/main -- proxima-autograd` prints 0 lines. The backward is a graph transform (`proxima-autograd/src/adjoint.rs`, module doc) producing `Elementwise`, `Reduce`, `Input`, `Constant` and `Iota` nodes, so "Metal backward coverage" is whether omega's emit accepts every node of the adjoint program, which is the R1 census. Parts that stay on the host by existing design: the gathered-operand gradient, coalesced by `proxima_autograd::sparse::dedupe_and_sum_rows` (`adjoint.rs` module doc, `GatheredContribution`). CPU training is out of scope.

```
qat_census / qat_step_bench / qat_finetune / qat_relations   proxima-model-interop/examples/ (feature qat), shared module examples/qat/mod.rs (the tests under proxima-model-interop/tests/ include it through a path attribute, precedent omega/tests/packed_brain_float_moe_training_metal.rs:24)
  descriptor  = conflaguration::from_file(tests/fixtures/model-configs/granite_moe.toml), block_count overridden for the 1-layer cell
  forward     = proxima_tensor::spec::descriptor::build_forward(&descriptor).program        (descriptor.rs:679)
  loss        = proxima_autograd::loss::softmax_cross_entropy(logits, targets)
  step graph  = differentiate(program, loss)  -> Differentiated { program, gradients, gathered }
  per tensor selected by convert::policy_selects:   effective = Add(master_leaf, delta_leaf)      // STE, no new op
  delta       = fake_quant_program(master) - master       // Metal, existing ScalarOps, evaluated before the step graph
  update      = proxima_autograd::optimizer::adam_step(...) then bf16_grid_round(master')          // Metal, existing ScalarOps
  masters     = f32 buffers holding bf16-grid values
fake_quant (CPU function, the oracle)      proxima-gguf/src/quant/scaled_float.rs  quantize_block / dequantize_block, FloatLayout::E3M2
eval        = mean softmax cross-entropy over 64 held-out sequences of 256 tokens, step graph forward only
export      = masters -> F32 GGUF (transform writer);  convert_codec (policy default, fp6_e3m2) runs on the export
agreement   = codec_token_agreement (fp6-e3m2-codec slice 16), unchanged
```

Fake-quant is a function, `fake_quant_e3m2(values: &[f32]) -> Vec<f32>` over whole blocks, composed from `quantize_block` and `dequantize_block`; it wears the transform pipe form through the closure bridge in the example and adds no type to a library crate (R18). The pipe question for it: a transform `[f32] -> [f32]`, expressed as the function above. The second question (what can a caller do that they could not before) is answered by R5: the same transform as a Metal graph, which the CPU function cannot give.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| STE as `Add(master, delta)` with `delta` a bound leaf | yes | an in-graph quantizer would be differentiated by the adjoint: `Greater` and `Equal` have no gradient (`adjoint.rs:602`) and `Reduce(Maximum)` mask-routes, so the gradient would be the quantizer's true derivative, not the identity; a leaf has no inbound derivative, so the existing `Add` rule is exactly STE and needs no stop-gradient op |
| fake-quant on Metal from existing `ScalarOp`s | yes | a rendered encode kernel is a new kernel; the existing ops can express absmax (`Reduce(Maximum)`), the f16 scale (an `Elementwise` of dtype `Float16`, which Metal accepts, `omega/src/metal/prepare_uniforms_pack.rs:409`), and round-to-grid (constant add and subtract). If R5 or R6 print a mismatch the recorded outcome is a rendered kernel, via an amendment, not a looser bound |
| `bf16` masters are the bf16 grid in f32 buffers | yes | Metal rejects every dtype but `Float32` and `Float16` outside a gather's indices (`omega/src/metal/prepare_uniforms_pack.rs:409`), so a width-true bf16 master needs an omega dtype path that this spec does not scope; the grid is what changes training dynamics (an update smaller than half a bf16 ulp is lost), and R16 reports the memory this costs |
| learning rate 1e-4, no search | yes | Adam's first step is about the learning rate; a bf16 weight of magnitude below 0.03125 has half-ulp at most 6.1e-5, so 1e-4 survives rounding there, and a 1e-5 step would be lost for every weight above 0.0078; each run prints `updates_lost_to_rounding`, the count of elements whose master did not change while the Adam update was nonzero, so the loss is measured, not assumed |
| masters start from the Q8_0 checkpoint | yes | the fp32 original is not in the tree; the starting point carries Q8_0's error, stated in the result section |
| ptq arm is eval-only on the control's masters | yes | "recovers" needs the quantized-after-training baseline; it costs one evaluation, not a run; its checkpoint is `convert_codec` over the control export |
| exports are F32, not bf16 | yes | `convert_codec` sources are F32, F16, Q8_0 and Q4_0 (fp6-e3m2-codec R13); F32 holds the bf16 grid exactly |
| data order from `Lcg(seed)` | yes | the packed pilots use the same generator (`low_precision_moe_training/SPEC.md:35`); the order digest is printed so control and qat are compared by value |
| agreement cells reuse `codec_token_agreement` with the converted checkpoint on the fp6 side and the unconverted control export on the "shipped" side | yes | the example's gap is `shipped_match - fp6_match` against recorded ids; the llama ids are the common ground truth and the unconverted control is the reference the problem statement names, so the gap is what conversion costs after the fine-tune; the `ptq` cell supplies the no-QAT conversion and the `ctl` cell the control against itself |

## acceptance criteria

Variables bound by this assignment block (`GRANITE` and the llama fixture directory are the fp6-e3m2-codec bindings; `OUT` is a durable path, never `/tmp`; `QAT_BASE` is the origin/main commit this spec was written against):

```sh
GRANITE=/Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80
OUT=/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/qat/out
IDS=proxima-model-interop/tests/fixtures/llama-parity
CONFIG=proxima-model-interop/tests/fixtures/model-configs/granite_moe.toml
CORPUS=proxima-model-interop/examples/data/war_and_peace.txt
QAT_BASE=07f7b8f702d16e128b46ebd7696b3686a44b5f62
FEAT=std,metal,qat
COMMON="--config $CONFIG --checkpoint $GRANITE --corpus $CORPUS --seed 17 --steps 256 --seq 256 --batch 4 --lr 0.0001 --master bf16 --eval-every 32 --eval-sequences 64 --memory-cap none"
mkdir -p $OUT
```

Every GPU row runs behind `gpu.lock` (`mkdir` the lock directory, release after) and runs alone: no other agent, cargo or Metal suite while it runs, box load read before it (`uptime`, `pgrep`, `ioreg` utilization), one model-loading process at a time. Rows marked TIMING additionally need the quiet box; their `box_before` files are kept beside the output. Test names carry layouts and shapes, never model names; a test that needs the checkpoint fails with its path and the env override `PROXIMA_ARCH_GRANITE_MOE_GGUF` when absent, it never skips. Every test count is the exact number the slice adds. Commands run from the repo root. In the table below `\|` stands for a literal pipe; the command blocks section holds the same commands with plain pipes and is the form to run.

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1 | command block AC1 | `3`: one census line per cell, `census layers=<1\|24> nodes=N node_lines=N adjoint_nodes=A metal_ok=N metal_reject=0`, the same N three times in one line, plus the rejection control `control census nodes=1 metal_ok=0 metal_reject=1` (the example's `--control bfloat16_input` mode classifies a one-node program whose `Input` has dtype `BFloat16`, which the dtype gate at `omega/src/metal/prepare_uniforms_pack.rs:409` rejects, so an inert classifier that accepts everything fails this line). GPU, behind gpu.lock. A nonzero `metal_reject` on a real cell or a count below 3 is the recorded outcome; the rejected node lines (`node id=... metal=reject:<reason>`) name the ops that must land before slice 2 |
| AC2 | R2 | command block AC2 | `8`: 2 summary lines and 6 per-process lines; a 24-layer refusal prints `step_bench_refused layers=24 reason=...` and the count is then 4 (1 summary, 3 process lines), recorded as the outcome. GPU, behind gpu.lock, TIMING, box load checked before each process |
| AC3 | R3 | `PROXIMA_ARCH_GRANITE_MOE_GGUF=$GRANITE cargo nextest run -p proxima-model-interop --features $FEAT -E 'test(/qat_fake_quant_cpu_/)'` | `4 passed`: `qat_fake_quant_cpu_error_within_block_bound_1024x512` (rows of the checkpoint's 1024x512 tensors dequantized then rounded to bf16; for every element with `q = abs(v) / scale_f16 <= 28.0`, `abs(v - fq(v)) <= scale_f16 x h(q)` with `h` the half-spacing function of fp6-e3m2-codec AC7), `qat_fake_quant_cpu_idempotent_on_grid_values_1024x512` (`fq(fq(v)) == fq(v)` bitwise), `qat_fake_quant_cpu_policy_selects_168_of_242_tensors` (the checkpoint header lists 242 tensors, `policy_selects` names exactly 168: 24 layers x 7 projections), `qat_fake_quant_cpu_control_perturbed_output_exceeds_bound` (one flipped element makes the bound check fail) |
| AC4 | R4 | `cargo nextest run -p proxima-model-interop --features $FEAT -E 'test(/qat_ste_/)'` | `3 passed`: `qat_ste_gradient_equals_identity_path_f64_loop_64x32` (a linear layer `y = x . fq(W)^T`, loss `sum(0.5 y^2)`, `x` of shape `[8,32]`: the master gradient from the differentiated program equals the f64 loop `x^T . dy` with `dy` computed from the quantized forward, within `K x 2^-24 x sum(abs(term))`), `qat_ste_program_adds_one_delta_leaf_per_selected_tensor` (for a 2-layer descriptor the program holds 14 `Add(master, delta)` nodes and 14 `delta` leaves), `qat_ste_control_true_quantizer_derivative_fails` (a gradient forced to zero wherever fake-quant moved the value differs from the oracle by more than the bound) |
| AC5 | R5 | `PROXIMA_ARCH_GRANITE_MOE_GGUF=$GRANITE cargo nextest run -p proxima-model-interop --features $FEAT -E 'test(/qat_fake_quant_metal_/)'` | `3 passed`: `qat_fake_quant_metal_equals_cpu_function_bitwise_1024x1024` and `..._1024x512` (real attention projection rows dequantized and rounded to bf16, so exact midpoints occur: each test first asserts the midpoint count is above 0, then asserts mismatching elements equal 0), `qat_fake_quant_metal_control_perturbed_constant_differs` (changing the rounding constant makes mismatches exceed 0). GPU, behind gpu.lock |
| AC6 | R6 | `cargo nextest run -p proxima-model-interop --features $FEAT -E 'test(/qat_bf16_grid_metal_/)'` | `3 passed`: `qat_bf16_grid_metal_equals_half_bf16_from_f32_over_corpus` (f32 bit patterns `index x 4093` for `index` in `0..1_048_576`, finite values only, the length asserted before comparing, mismatches equal 0), `qat_bf16_grid_metal_ties_to_even_matches_half` (the 4 values whose low 16 bits are 0x8000 at two exponents), `qat_bf16_grid_metal_control_truncation_differs` (round-toward-zero disagrees with `half` on more than 0 corpus values). GPU, behind gpu.lock |
| AC7 | R7 | `cargo nextest run -p proxima-model-interop --features $FEAT -E 'test(/qat_corpus_/)'` | `5 passed`: `qat_corpus_sha256_and_bytes_match_pinned_digest` (3,274,130 bytes and sha256 `d7a1d4c8d7427d526f8dc1fb89c4806c6f3276292190dc6fb1ff20a88b9bb7b5`, the digest computed outside the example by `shasum -a 256` on the file at `QAT_BASE`), `qat_corpus_tokenizer_reproduces_llama_prompt_ids` (the granite tokenizer from the checkpoint tokenizes the 4 fixture prompts to exactly the recorded `prompt_ids`, 5 + 7 + 72 + 1000 ids: llama.cpp is the oracle), `qat_corpus_split_is_paragraph_aligned_and_exhaustive` (train bytes + held-out bytes = 3,274,130; the held-out text starts at a paragraph start; the train paragraph count equals `floor(0.9 x P)` with `P` counted in the test by a separate blank-line scan), `qat_corpus_held_out_covers_the_16384_evaluation_tokens` (prints `corpus split=<train\|held_out> bytes=B tokens=T`, read by slice 17 with `--no-capture`, and asserts held-out `T` is at least 16,384), `qat_corpus_control_altered_prompt_id_fails` (one changed recorded id makes the tokenizer check fail) |
| AC8 | R8 | `cargo nextest run -p proxima-model-interop --features $FEAT -E 'test(/qat_disjoint_/)'` | `3 passed`: `qat_disjoint_four_parity_prompts_absent_from_corpus` (the 3 prompts of `$IDS/granite_moe/llama_ids.json` plus the 1 of `long_prompt_llama_ids.json`, asserted to be 4 before checking, each searched as a substring of the corpus text: 0 hits), `qat_disjoint_control_corpus_sentence_is_found` (a sentence cut from the corpus is found by the same search: 1 hit), `qat_disjoint_prompt_count_matches_224_positions` (3 x 32 + 1 x 128 generated ids) |
| AC9 | R9 | `cargo nextest run -p proxima-model-interop --features $FEAT -E 'test(/qat_control_/)'` | `3 passed`: `qat_control_order_matches_independent_lcg` (the first 8 start offsets for seed 17 equal an independent re-implementation inside the test using the pilot generator constants multiplier 6364136223846793005, increment 1, top 32 bits), `qat_control_and_qat_share_order_and_initial_masters` (both arms print the same order digest and initial-master digest), `qat_control_and_qat_programs_differ_only_by_delta_leaves` (node count of the qat program minus the control program equals 2 x 168 = 336: one `delta` leaf and one `Add` per selected tensor) |
| AC10 | R10 | command block AC10 | `531`: 2 arms x (256 train + 9 held-out) = 530 `loss` lines, plus the one `identity` line with `within=true`. GPU, behind gpu.lock, TIMING |
| AC11 | R11 | command block AC11 | `267`: 256 train + 9 held-out `loss` lines for `qat` (265), the one `ptq_eval` held-out `loss` line (1) and the one `update_loss` line (1). GPU, behind gpu.lock, TIMING |
| AC12 | R12 | command block AC12 | `relations: 4 of 4 hold` (refutation AC: a lower N is the recorded outcome; one line per relation with its value, bound and `holds=<bool>` precedes the summary; relation 4 is the degenerate control and must hold or the check itself is void) |
| AC13 | R13 | `PROXIMA_ARCH_GRANITE_MOE_GGUF=$GRANITE cargo nextest run -p proxima-model-interop --features $FEAT -E 'test(/qat_export_/)'` | `3 passed`: `qat_export_reload_has_242_tensors_equal_to_source_header`, `qat_export_tensors_equal_widened_masters_bitwise`, `qat_export_nonfinite_master_rejected_with_tensor_name` |
| AC14 | R14 | command block AC14 | `9`: 3 cells x (2 `match` lines + 1 `bound` line); the `qat` cell's bound line must read `within=true`, the `ptq` cell's may read either, the `ctl` reference cell (the control checkpoint against itself) must read `gap=0 within=true`. GPU, behind gpu.lock |
| AC15 | R15 | command block AC15 | `3`: one `duration` line per training arm (`control`, `control_repeat`, `qat`) with `within_2pct=true` |
| AC16 | R16 | command block AC16 | `3`: one `memory` line per training arm with `within_5pct=true` and `memory_cap=none` |
| AC17 | R17 | command block AC17 | `6`: the section heading, the unexplained heading, and one table row per arm; the section is written by slice 17 from the saved outputs only |
| AC18 | R18 | command block AC18 | `0 2`: the first number counts added type definitions in library `src/` directories since `QAT_BASE`, the second is the control, the same grep over `proxima-primitives/src/float_layout.rs` at `QAT_BASE` (which defines `FloatLayout` and `Specials`) |
| AC19 | R19 | `PROXIMA_ARCH_GRANITE_MOE_GGUF=$GRANITE cargo nextest run -p proxima-model-interop --features $FEAT -E 'test(/qat_loss_/)'` | `3 passed`: `qat_loss_step0_matches_recorded_logprob_96_positions` (for each of the 3 cases of `$IDS/granite_moe/n_probs.json`, the forward over `prompt_ids` followed by `generated_ids` gives `-log softmax` at each of the 32 generated positions; the 96 absolute differences against `-logprob` have mean at most 0.02 and maximum at most 0.1), `qat_loss_recorded_position_count_is_96` (the file holds 3 cases of 32 steps), `qat_loss_control_shifted_targets_fail` (comparing each position to the recorded value of the next position makes the mean exceed 0.02). GPU, behind gpu.lock |

### command blocks

Each block is one command; the count it prints is the asserted N, and a count of 0 is red. Plain pipes below.

```sh
# AC1
{ cargo run --release -p proxima-model-interop --features $FEAT --example qat_census -- --config $CONFIG --layers 1,24 --seq 256; cargo run --release -p proxima-model-interop --features $FEAT --example qat_census -- --control bfloat16_input; } | grep -c -E '^census layers=(1|24) nodes=([1-9][0-9]*) node_lines=\2 adjoint_nodes=[1-9][0-9]* metal_ok=\2 metal_reject=0$|^control census nodes=1 metal_ok=0 metal_reject=1$'
# AC2
for layers in 1 24; do cargo run --release -p proxima-model-interop --features $FEAT --example qat_step_bench -- --config $CONFIG --checkpoint $GRANITE --layers $layers --steps 64 --seq 256 --batch 4 --processes 3 --master bf16 --fake-quant off --memory-cap none; done | grep -c -E '^step_bench layers=(1|24) processes=3 steps=64 warm_excluded=1 fake_quant=off master=bf16 tokens_per_step=1024 ms_per_step_median=[0-9]+\.[0-9]+ ms_per_step_mad=[0-9]+\.[0-9]+ cov_pct=[0-9]+\.[0-9]+ tokens_per_s=[0-9]+\.[0-9]+ peak_rss_bytes=[1-9][0-9]* peak_footprint_bytes=[1-9][0-9]* peak_gpu_bytes=[1-9][0-9]*$|^step_bench_process layers=(1|24) process=[123] ms_per_step_median=[0-9]+\.[0-9]+ '
# AC10
{ /usr/bin/time -l -o $OUT/time_control.log cargo run --release -p proxima-model-interop --features $FEAT --example qat_finetune -- $COMMON --arms control --export $OUT/control_f32.gguf | tee $OUT/ablation_control.out; /usr/bin/time -l -o $OUT/time_control_repeat.log cargo run --release -p proxima-model-interop --features $FEAT --example qat_finetune -- $COMMON --arms control_repeat | tee $OUT/ablation_control_repeat.out; cargo run --release -p proxima-model-interop --features $FEAT --example qat_relations -- --identity $OUT/ablation_control.out $OUT/ablation_control_repeat.out --bound 0.001 | tee $OUT/identity.out; } | grep -c -E '^loss arm=(control|control_repeat) step=[0-9]+ split=(train|held_out) value=[0-9]+\.[0-9]+$|^identity abs_diff=[0-9.e-]+ bound=0\.001 within=true$'
# AC11
{ /usr/bin/time -l -o $OUT/time_qat.log cargo run --release -p proxima-model-interop --features $FEAT --example qat_finetune -- $COMMON --arms qat --export $OUT/qat_f32.gguf | tee $OUT/ablation_qat.out; cargo run --release -p proxima-model-interop --features $FEAT --example qat_finetune -- $COMMON --arms ptq_eval --from-masters $OUT/control_f32.gguf | tee $OUT/ablation_ptq.out; } | grep -c -E '^loss arm=(qat|ptq_eval) step=[0-9]+ split=(train|held_out) value=[0-9]+\.[0-9]+$|^update_loss arm=qat updates_lost_to_rounding=[0-9]+ updates_total=[1-9][0-9]*$'
# AC12
cargo run --release -p proxima-model-interop --features $FEAT --example qat_relations -- --control $OUT/ablation_control.out --qat $OUT/ablation_qat.out --ptq $OUT/ablation_ptq.out --repeat $OUT/ablation_control_repeat.out --max-gap 0.05 | tee $OUT/relations.out | grep -E '^relations: [0-9] of 4 hold$'
# AC14
cargo run --release -p proxima-model-interop --example codec_token_agreement -- --fixtures llama_ids.json,long_prompt_llama_ids.json --allowed-gap-rate 0.02 --cell qat=$OUT/qat_fp6.gguf,$OUT/control_f32.gguf,$IDS/granite_moe --cell ptq=$OUT/ptq_fp6.gguf,$OUT/control_f32.gguf,$IDS/granite_moe --cell ctl=$OUT/control_f32.gguf,$OUT/control_f32.gguf,$IDS/granite_moe | grep -c -E '^cell=(qat|ptq|ctl) codec=(fp6|shipped) match=[1-9][0-9]* of 224$|^cell=qat bound allowed_gap=4 gap=[0-9]+ within=true$|^cell=ctl bound allowed_gap=4 gap=0 within=true$|^cell=ptq bound allowed_gap=4 gap=[0-9]+ within=(true|false)$'
# AC15
cargo run --release -p proxima-model-interop --features $FEAT --example qat_relations -- --duration control=$OUT/ablation_control.out,$OUT/time_control.log control_repeat=$OUT/ablation_control_repeat.out,$OUT/time_control_repeat.log qat=$OUT/ablation_qat.out,$OUT/time_qat.log --tolerance-pct 2 | grep -c -E '^duration arm=(control|control_repeat|qat) steps=256 tokens=262144 wall_s_internal=[0-9]+\.[0-9]+ wall_s_external=[0-9]+\.[0-9]+ tokens_per_s=[0-9]+\.[0-9]+ within_2pct=true$'
# AC16
cargo run --release -p proxima-model-interop --features $FEAT --example qat_relations -- --memory control=$OUT/ablation_control.out,$OUT/time_control.log control_repeat=$OUT/ablation_control_repeat.out,$OUT/time_control_repeat.log qat=$OUT/ablation_qat.out,$OUT/time_qat.log --tolerance-pct 5 | grep -c -E '^memory arm=(control|control_repeat|qat) peak_rss_bytes=[1-9][0-9]* peak_rss_external=[1-9][0-9]* peak_footprint_bytes=[1-9][0-9]* peak_gpu_bytes=[1-9][0-9]* memory_cap=none within_5pct=true$'
# AC17
grep -c -E '^(## ablation result|### unexplained|\| (control|control_repeat|qat|ptq_eval) \|)' proxima-tensor/specs/fp6-qat-ablation/SPEC.md
# AC18
echo "$(git diff $QAT_BASE -- 'proxima-*/src' 'omega/src' | grep -c -E '^\+[[:space:]]*(pub(\([a-z]+\))? )?(struct|enum|trait|union|type) ') $(git show $QAT_BASE:proxima-primitives/src/float_layout.rs | grep -c -E '^[[:space:]]*(pub(\([a-z]+\))? )?(struct|enum|trait|union|type) ')"
```

AC14 setup, run once before the command (the fp6 spec's `convert_codec` example, unchanged; it prints two `converted: ...` lines matching the pattern of fp6-e3m2-codec AC14, count 2, which the setup slice asserts): `cargo run --release -p proxima-model-interop --example convert_codec -- --codec fp6_e3m2 --policy default --job $OUT/qat_f32.gguf=$OUT/qat_fp6.gguf --job $OUT/control_f32.gguf=$OUT/ptq_fp6.gguf`.

## ablation result

Written by slice 17 from the saved `$OUT` files only. Until then this section holds no arm rows and AC17 prints 2.

### unexplained

Written by slice 17. Every observation without a traced mechanism stays here as unexplained; none is dropped or averaged over.

## out of scope

- pretraining, any training that starts from random weights
- LoRA and every other parameter-efficient variant; the ablation trains the full selected tensors
- other codecs: a second codec is a parameter of `fake_quant` and the policy, not a new spec
- CPU training: the CPU pilots stay as they are, no CPU arm is added
- a width-true bf16 master (needs an omega `BFloat16` dtype path, `omega/src/metal/prepare_uniforms_pack.rs:409`); the grid-in-f32 master is the decision above
- activations or KV cache in any float layout; weights only
- the 500 MB serving cap: lifted for training by the owner on 2026-10-11; it still binds serving
- any Ollama or llama-server run; the token oracle is the recorded llama ids and the control's own checkpoint
- changing the serving default of e3m2; the owner decides on the AC12 and AC14 outputs
- hyperparameter search, multiple seeds, learning-rate sweeps: one pre-registered setting, final-step checkpoint for every arm

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| the full 24-layer step does not fit this GPU's memory (masters, Adam moments, gradients and `delta` leaves are all f32 buffers) | unmeasured | AC2 prints `step_bench_refused` and no full-model fine-tune can run | the refusal names the failed allocation; the next work is a memory design from that record, not an estimate |
| Metal rejects ops in the granite adjoint | unmeasured; known gates are `ScatterNotSupported` (`omega/src/error.rs:79`) and the adjoint's own refusals `ScanAdjointUnsupported`, `NonProjectionOperandMap`, `ReduceOverGatherUnsupported` (`proxima-autograd/src/error.rs:29,52,71`) | AC1 prints a nonzero `metal_reject` | each rejected node line names the op; ops land as their own slices before slice 2, and TASKS is amended with the struck order |
| the graph-expressed rounding (constant add and subtract) differs from round-to-nearest-even on ties | medium; bf16-widened weights hit exact e3m2 midpoints often | AC5 or AC6 prints mismatches above 0 | the amendment is a rendered encode kernel from `FloatLayout` constants; the bound is not loosened |
| updates smaller than half a bf16 ulp are lost | likely for weights above 0.03125 at lr 1e-4 | the control barely moves and the held-out gap measures nothing | `updates_lost_to_rounding` is printed per run (counted by AC11 for qat); a control whose held-out loss does not drop below its step-0 value is recorded as unexplained, not tuned away |
| masters start from Q8_0, so the control already carries Q8_0 error | certain | the fp6 gap is measured from a quantized origin | stated in the result section; the control, not the original fp32, is the comparison |
| 256 steps of 1,024 tokens (262,144 tokens) is too few to move held-out loss | unmeasured | the identity and gap relations both sit inside noise | AC10's identity bound is the test of that; no step count is changed after seeing a result |
| the 0.05 nats gap and the 0.001 identity bound are pre-registered values with no measurement behind them | plausible | a relation flips on the bound, not the codec | both stay labelled plausible; the result section prints the raw difference beside each bound |
| the box is loud during a timing cell | medium | AC2, AC10, AC11 numbers are contaminated | the quiet-box protocol and the `box_before` files of the parity campaign; a loud cell is rerun, the loud output is kept and labelled |

## context

- `proxima-tensor/specs/fp6-e3m2-codec/SPEC.md` and `TASKS.md`: the codec, `convert_codec` (slice 13, landed), the `convert_codec` example (slice 14) and `codec_token_agreement` (slice 16), both of which this spec requires landed
- `proxima-gguf/src/quant/scaled_float.rs` (`quantize_block`, `dequantize_block`, scale `f16(absmax / max_normal)`); `proxima-primitives/src/float_layout.rs` (`FloatLayout::E3M2`)
- `proxima-model-interop/src/convert.rs:35` (`policy_selects`), `:54` (`convert_codec`)
- `proxima-tensor/src/spec/descriptor.rs:679` (`build_forward`); `proxima-model-interop/tests/fixtures/model-configs/granite_moe.toml`
- `proxima-autograd/src/adjoint.rs` (module doc: per-op rules, `GatheredContribution`), `proxima-autograd/src/optimizer.rs` (`adam_step`), `proxima-autograd/src/loss.rs`, `proxima-autograd/src/sparse.rs` (`dedupe_and_sum_rows`), `proxima-autograd/src/error.rs:29,52,71` (the refusals)
- `omega/src/metal/prepare_uniforms_pack.rs:409` (dtype gate), `omega/src/error.rs:79` (`ScatterNotSupported`), `omega/tests/packed_brain_float_moe_training_metal.rs` (the Metal training fixture)
- the training pilots: `proxima-tensor/specs/low_precision_moe_training/SPEC.md`, `proxima-tensor/specs/packed_pilot_moe_training/SPEC.md`, `proxima-tensor/specs/metal_packed_pilot_moe_training/SPEC.md`, `proxima-tensor/specs/metal_packed_pilot_moe_64step/SPEC.md`, `proxima-tensor/specs/metal_packed_brain_float_moe_training/SPEC.md`, `proxima-tensor/specs/metal_packed_brain_float_moe_training_loop/SPEC.md`; CPU timing in `proxima-tensor/specs/low_precision_moe_training/results/tiny-pilot.json` and `proxima-tensor/specs/packed_pilot_moe_training/results/packed-tiny-pilot.json` (`wall_time_ns`, `tokens_per_second`)
- `proxima-model-interop/examples/data/war_and_peace.txt` and `proxima-model-interop/examples/data/war_and_peace.LICENSE.txt` (Project Gutenberg ebook 2600); `proxima-model-interop/tests/fixtures/llama-parity/granite_moe/llama_ids.json`, `long_prompt_llama_ids.json` and `gguf_kv.txt` (24 blocks, 1024 embedding, 16 heads, 8 kv heads, 32 experts, 8 used, 242 tensors)
- `proxima-model-interop/examples/decode_arms.rs` and `proxima-tensor/specs/decode-prefill-parity/SPEC.md` (the quiet-box protocol, `max(MAD, 2% of base)` time bound, `peak_gpu_bytes`)
