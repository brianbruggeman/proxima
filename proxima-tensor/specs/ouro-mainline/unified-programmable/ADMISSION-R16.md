# admission amendment: process footprint and allocation lifecycle

Auditor: `/root/spec_auditor` (independent narrowed re-audit)
Date: 2026-10-03
Decision: ADMIT

| Gate | Result |
|---|---|
| G1 proposition/refutation | pass |
| G2 requirements | pass |
| G3 AC mapping | pass |
| G4 executable fixture/command | pass |
| G5 packet/slice | pass |
| G6 scope | pass |
| G7 context paths | pass |

| Register | Count |
|---|---:|
| Requirements | 16 / 16 discharged |
| Acceptance criteria | 18 / 18, 0 orphaned |
| Slices | 16 / 16, 0 unvalidated |
| Context paths | 32 / 32, 0 dangling |

## Source register

| Opened artifact | Observation |
|---|---|
| `SPEC.md:34` | R16 binds a 500,000,000-byte process physical-footprint ceiling, owner/backing/lifetime/fence accounting and pre-allocation refusal. |
| `SPEC.md:217` | AC18 names the real-model command, lifetime high-water capture, allocation joins, cap and missing-sample rejection. |
| `TASKS.md:20` | S16 maps to AC18 and preserves the same command and expected checks. |
| `fixtures.json:380-409` | Exact fit and one-byte-over controls plus three literal allocation/physical-region records. Independent byte sums equal 500,000,000. |
| `fixtures.json:411-415` | Five literal lifecycle mutations, typed error names and one expected rejection each. |
| `WORK-PACKET-RESIDENCY.md:95-113` | Required lifecycle, graph, KV, numeric, route and cancellation controls point to fixture keys that resolve. |
| `results/checkpoints/e2e-checkpoint-9/RUN-STATUS.json:1` | C9 is partial; Granite was not started and llama has zero cohort rows. |
| `results/checkpoints/e2e-checkpoint-9/vmmap-summary-pid1142.txt:17-18` | C9 current physical footprint is 2.6G and peak is 4.8G. |
| `results/checkpoints/e2e-checkpoint-9/vmmap-pid1142.txt:1003,1733-1744` | Named Qwen mapping exists; MALLOC_LARGE and graphics ranges still lack owner joins. |

C9 is retained failure evidence, not AC18 completion. No source implementation,
model execution, test, or performance conclusion is admitted by this amendment.

## fixture arithmetic and join re-audit

Auditor: `/root/ac18_review` (independent current-source check)
Date: 2026-10-04
Decision: ADMIT

The reservation ledger now has three rows that exactly join the three literal
allocation rows on allocation id, owner, backing id, bytes, generation and
fence id. Both reservation and physical-region totals are 500,000,000 bytes;
the declared 65-byte mutation exceeds the cap by one byte. Sources:
`fixtures.json:381-404` and `SPEC.md:217`. The independent check found 3/3
reservation-to-allocation joins and 3/3 allocation-to-region joins. This
re-audit closes the fixture arithmetic/join correction only; AC18 runtime
admission, owner allocation events, fences and model evidence remain open.
