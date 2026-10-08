# combine: llama-parity digest changes

Fixtures: proxima-model-interop/tests/fixtures/llama-parity/*.digest (consistency baseline, not an oracle). Old copies: combine/digests_old/.

## recaptured (1 of 8)

gemma4_e2b.digest, command
`PROXIMA_ARCH_DATA_CAPTURE=1 cargo nextest run -p proxima-model-interop --features std,metal --cargo-profile gate -E 'binary(arch_data_baseline) and test(=arch_data_digest_gemma4_e2b)'`
(commit "test(interop): recapture the e2b digest for the one-reduce per-layer norm"). Explained by r9 (perf(tensor): norm the per-layer input
once over the layer axis, 63f4bca6):

| line | old | new |
|---|---|---|
| bind.ops | 6074 | 5667 |
| bind.ops_sha256 | 5765164066d0...9164 | 289de46eb2e4...e2a |
| bind.logits_root | NodeId(6073) | NodeId(5666) |
| bind.layer_roots sha256 | 437f443619d6...4f11 | 262cdfa82767...e3ff |
| verify.ops | 6072 | 5665 |
| verify.ops_sha256 | 3141af6ec4e0...a89 | c8078f4ba6e6...0b3 |
| verify.logits_root | NodeId(6071) | NodeId(5664) |
| verify.layer_roots sha256 | 437f443619d6...4f11 | 262cdfa82767...e3ff |

Unchanged lines: checkpoint, architecture, kv_layout, hidden_root=None, residual_roots, router_roots, single_position_step (both programs).
The eight changed lines are the eight r9's spec section lists ("digests expected to change").

Op-count mechanism, measured with a scratch unit test over the real builders (proxima-tensor, `ple_single_reduce_parity_tests` fixture):
flat projections add 11 ops, the shared layer-axis view adds 13, a per-layer slice adds 1 (`ple_layer_input`), the old windowed per-layer form
adds 13. Old preamble 11 + 35 * 13 = 466; new 11 + 13 + 35 * 1 = 59; delta -407 = 6074 - 5667 = 6072 - 5665. r9's written delta was -442 (derived
from "14 ops per layer"; the builder adds 13), so the recapture is 35 ops above r9's derived 5632 / 5630. The 35 is r9's miscount of the
old form, not an unexplained change: the same -407 is the sum of the three bound-op kinds in the attention census (see below).

Cross-check from a second artifact: `gemma4_attention_chain_census` prints the bound op histogram (base run:
combine/base_census_nocapture.log; tip run: combine/tip_interop_5fail.log): reduce 903 -> 869, elementwise 488 -> 454, constant 262 -> 228,
iota 8 -> 8: -34 each = 34 per-layer norms (35 -> 1), total 1661 -> 1559.

## not recaptured

- granite_moe.digest: unchanged at the integrated tip (arch_data_digest_granite_moe and model_config_roundtrip_granite_moe pass). Before
  the revert of r2 0013 (stacked experts in the `metal` set) the run printed bind.ops 6394 -> 5602 (-792 = 24 layers * 33); the per-block
  difference was measured with a scratch test at 32 experts / 8 used / 1024 x 512: PerRoute 179 ops, Stacked 146 ops (-33). The digest
  is therefore tied to whether `moe-stacked-experts` is on, and stays at the base value while the default is off.
- gemma4_26b, openchat, qwen2, qwen3, qwen35, qwen35moe (and lfm2, which has no digest): not run. The order restricts test models to
  gemma4 E2B and granite moe 1b and forbids loading gemma4 26B, so these seven fixtures were neither checked nor recaptured. Slices that can
  move them: r2's stacked strategy only when `moe-stacked-experts` is enabled (off at the tip); r9's per-layer-input form applies where
  `embedding_length_per_layer_input > 0` (gemma4 26B declares 0, fixtures/llama-parity/gemma4_26b/gguf_kv.txt:37); r3, r4, r5, r6, r7 do
  not change the lowered op list. To check them: `cargo nextest run -p proxima-model-interop --features std,metal --cargo-profile gate
  -E 'binary(arch_data_baseline) and test(~arch_data_digest)'` on a box allowed to open those checkpoints.
- every `.bound` file: unchanged (no weight leaf changed).
