# spec admission record

Auditor: `/root/spec_auditor` (independent re-audit)
Date: 2026-10-03
Decision: ADMIT

| Gate | Result |
|---|---|
| G1 | pass |
| G2 | pass |
| G3 | pass |
| G4 | pass |
| G5 | pass |
| G6 | pass |
| G7 | pass |

| Register | Count |
|---|---:|
| Requirements | 15 / 15 discharged |
| Acceptance criteria | 17 / 17, 0 orphaned |
| Slices | 15 / 15, 0 unvalidated |
| Context paths | 32 / 32, 0 dangling |

## Gate artifact register

| Gate | Opened artifact and observation |
|---|---|
| G1 | `SPEC.md:9` binds the shared path and requested/effective evidence; `SPEC.md:13` names control, state, hazard/resource and duplicate-pipeline refutations. |
| G2 | `SPEC.md:19-33` contains fifteen independently testable requirements: configuration shape, precedence, consumption, execution limits, stream scheduling, identity, kernels, residency, verification, conversion, routing, evidence, model comparison, build tier and feature discipline. |
| G3 | `SPEC.md:199-215` names all fifteen requirements across seventeen ACs; native text recount found fifteen discharged requirements and zero orphaned ACs. |
| G4 | `SPEC.md:199-215` supplies executable commands and expected counts. `SPEC.md:200` requires twenty-six distinct execution-result rows, all six required fields, actual consumer/rejection sources, zero missing/duplicate/extra/owner_mapping/unaccounted rows and a literal positive/mutation control. `fixtures.json:57-146` separates the twenty-six owner inventory rows from twenty-six marked literal schema controls. |
| G4 sampling | `fixtures.json:194-222` binds synthetic checkpoint digest/bytes, prompt, sampling settings, four vectors of 257 logits, fastrand version/state encoding/transition, literal tokens/states, and retained payload/oracle hashes. `results/serving/raw/sampling-logits-payload.log:7-10` contains all four vectors and CDF/draw/state records; `:11-15` contains the corresponding token/yield/state records. `results/serving/sampling-oracle-source.rs:565-583` computes WyRand multiplication/XOR draws and an independent f64 softmax/CDF; `:560-564` checks model identity/vector shape, and `:599-626` checks continuation states. |
| G5 | `TASKS.md:5-19` contains fifteen slices, each naming validation commands and ACs. `SPEC.md:191-195`, `TASKS.md:16` and `TASKS.md:27-30` require dense/MoE controls before the next source integration and the full final matrix. The four work packets bind owner paths, dependency order, controls and evidence destinations. |
| G6 | `SPEC.md:219-229` binds a nonempty scope boundary, including publication, duplicate execution owners and unavailable hardware claims. |
| G7 | Native existence probes resolved all thirty-two context paths at `SPEC.md:243-274`; zero were dangling. Both additional sampling fixture artifact pointers resolve to retained files and the referenced source/payload lines. |

## Sampling artifact hashes

Native SHA-256 capture matched the fixture values:

- `results/serving/raw/sampling-logits-payload.log`:
  `6e0c388b1401602d86abaa70aa747d861238e5cced617bde3fe0a1f2fd80631c`.
- `results/serving/sampling-oracle-source.rs`:
  `2e0419a5998014e8d22fc7225cccc5e028ff65270019788eb8f45cdc6e06d968`.

The canonical local fastrand 2.5.0 source was opened at
`/Users/brianbruggeman/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/fastrand-2.5.0/src/lib.rs:149`,
`:368` and `:398`. The retained oracle spells the same state increment,
multiply/XOR output and f32 conversion independently of the production sampler.
All four retained draws are below one and lie inside their recorded f64 CDF
intervals; their IDs and little-endian state records match the literal fixture.

This admission authorizes implementation against the spec, task commands and
literal fixtures. It does not discharge implementation ACs or report a runtime
or performance verdict. Owner inventory and synthetic controls are not actual
policy-consumption or real-model checkpoint evidence. No build, test, benchmark
or model process was launched during this re-audit.
