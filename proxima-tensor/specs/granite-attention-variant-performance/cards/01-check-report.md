# Card 01: check the saved replay report

**Owner:** GPT-6 Luna
**Dependency:** 00
**Commit:** `test(interop): check granite attention replay report`
**Budget:** at most 30 minutes front-to-back, including AC1a/AC1b, review, and TASKS.md update.

## Purpose

Check Card 00's saved payload independently, with `--expected-shapes` accepting one now and two after Card 02. The checker reports what the artifact contains and refuses false pairs or incomplete timing.

## Read

- `../SPEC.md` sample-statistics contract and Card 00's actual `/private/tmp/granite-attention-ab.json`; open the report before implementing the parser.
- `../../granite-attention-numeric-matrix/check_bf8_contract.py` for this repo's small Python checker style.

## Edit

- `proxima-tensor/specs/granite-attention-variant-performance/check_report.py`: JSON parser, counts, recomputation, and twelve in-memory negative controls.
- `proxima-tensor/specs/granite-attention-variant-performance/TASKS.md`: tick Card 01 after both commands print their exact counts.

## Steps

1. Parse `version=1` and require exactly `--expected-shapes` distinct actual tokenizer counts, each at least its nominal target. Pin the Granite checkpoint SHA256 to `cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80` and byte length to `1422239776`, in addition to the GGUF metadata. Require host arch and OS as nonempty text, hostname as nonempty text or null, a present `device_description` with nonempty text or null, and all reported serving fields including `reasoning_budget=0`. Require exactly two arms per shape labeled legacy and shared_k, both with the same node/extents, different captured entry and SHA, `fault_binding_present=false`, equal nonempty output byte counts/hashes, and equal nonempty generated IDs. Each arm's first dispatch extent must equal the shape's actual tokenizer count. Require prompt-ID digest, grid with positive active dimensions, named pipeline resources, bound bytes, and a process resource string with the exact `cell label=<arm>` tokens followed by exactly one each of `wall_ms`, `cpu_pct`, `rss_peak_mb`, `footprint_mb`, `gpu_alloc_mb`, `load_before`, and `load_after` in that order. Unrelated tokens such as `cpu_ms` may occur between these fields. Treat resource values as opaque; never parse a zero footprint as measured zero.
2. Require each shape's `rounds` to enumerate `0..19` exactly once, in order. Require its exact two-string `arm_order` to alternate even=`["legacy","shared_k"]`, odd=`["shared_k","legacy"]`. Require each arm's 20 finite positive `gpu_ns` samples to enumerate those rounds exactly once with `position` matching its one slot in `arm_order`; `timing_attempts=20`, `resource_replay_attempts=5`, and `replay_errors=0` are required. Recompute nearest-rank p50/p90/p99, min/max, population CoV, and all 20 signed round differences with `SPEC.md`'s exact indexes/formulas/tolerances. Print counts and positive/negative round-difference counts; neither sign is mandatory.
3. With `--negative-controls`, make twelve in-memory copies of the loaded report: remove an arm, remove a sample, copy legacy SHA to selected, set output equality false, set errors=1, reverse one even round's `arm_order` without moving either sample's `position`, replace the checkpoint SHA and matching path suffix, set actual tokenizer count below its nominal target, suffix one resource arm label, change actual count to a different value still above its nominal target while leaving both dispatch extents unchanged, remove `device_description`, and swap `wall_ms` with `cpu_pct` without dropping either token. Each copy must fail the same validator. The original file's bytes must remain unchanged.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC1a | `python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py --expected-shapes 1 /private/tmp/granite-attention-ab.json` | prints `prompt_shapes=1 arms=2 samples_per_arm=20 resource_cells=2 matching_pairs=1 errors=0` |
| AC1b | `python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py --expected-shapes 1 --negative-controls /private/tmp/granite-attention-ab.json` | prints `negative_controls=12 rejected=12`; original report bytes unchanged |

## Residual

Schema consistency cannot establish that capture inputs were immutable or that an isolated replay time predicts whole-layer latency.
