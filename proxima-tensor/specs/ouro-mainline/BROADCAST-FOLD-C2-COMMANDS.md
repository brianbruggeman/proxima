# fresh C2 cells after broadcast fold repair

Build once and freeze the candidate executable after the source checkpoint.
The independent executable/references are the same as the original owner cells.
Do not overwrite original r1 cells. All commands run from main, natively and
serially. These are untimed numerical observations.

```text
cargo build --release --example ouro_loopcd_lab --features proxima-tensor --target-dir /private/tmp/proxima-ouro-mainline-target
cp /private/tmp/proxima-ouro-mainline-target/release/examples/ouro_loopcd_lab /private/tmp/ouro-mainline-broadcast-paper-r2
shasum -a 256 /private/tmp/ouro-mainline-broadcast-paper-r2 /private/tmp/ouro-mainline-oracle-owner-paper
/private/tmp/ouro-mainline-broadcast-paper-r2 --method baseline --backend metal --prefix-ids 504,3575,282,4649,314 --depth 4 --strength 0.5 --max-new-tokens 1 --output proxima-tensor/specs/ouro-mainline/results/c2-broadcast-france-baseline-r2
/private/tmp/ouro-mainline-broadcast-paper-r2 --method fixed-logits --backend metal --prefix-ids 504,3575,282,4649,314 --depth 4 --strength 0.5 --max-new-tokens 1 --output proxima-tensor/specs/ouro-mainline/results/c2-broadcast-france-fixed-logits-r2
/private/tmp/ouro-mainline-broadcast-paper-r2 --method adaptive-logits --backend metal --prefix-ids 504,3575,282,4649,314 --depth 4 --strength 0.5 --max-new-tokens 1 --output proxima-tensor/specs/ouro-mainline/results/c2-broadcast-france-adaptive-logits-r2
/private/tmp/ouro-mainline-broadcast-paper-r2 --method fixed-hidden --backend metal --prefix-ids 504,3575,282,4649,314 --depth 4 --strength 0.5 --max-new-tokens 1 --output proxima-tensor/specs/ouro-mainline/results/c2-broadcast-france-fixed-hidden-r2
/private/tmp/ouro-mainline-broadcast-paper-r2 --method baseline --backend metal --prefix-ids 6340,260,5307,5054,30,533,451,1977,260,3985,314,4461,284,260,4405,314,2382,30,378,2380,282,260,4405,314 --depth 4 --strength 0.5 --max-new-tokens 1 --output proxima-tensor/specs/ouro-mainline/results/c2-broadcast-ball-baseline-r2
/private/tmp/ouro-mainline-broadcast-paper-r2 --method fixed-logits --backend metal --prefix-ids 6340,260,5307,5054,30,533,451,1977,260,3985,314,4461,284,260,4405,314,2382,30,378,2380,282,260,4405,314 --depth 4 --strength 0.5 --max-new-tokens 1 --output proxima-tensor/specs/ouro-mainline/results/c2-broadcast-ball-fixed-logits-r2
/private/tmp/ouro-mainline-broadcast-paper-r2 --method adaptive-logits --backend metal --prefix-ids 6340,260,5307,5054,30,533,451,1977,260,3985,314,4461,284,260,4405,314,2382,30,378,2380,282,260,4405,314 --depth 4 --strength 0.5 --max-new-tokens 1 --output proxima-tensor/specs/ouro-mainline/results/c2-broadcast-ball-adaptive-logits-r2
/private/tmp/ouro-mainline-broadcast-paper-r2 --method fixed-hidden --backend metal --prefix-ids 6340,260,5307,5054,30,533,451,1977,260,3985,314,4461,284,260,4405,314,2382,30,378,2380,282,260,4405,314 --depth 4 --strength 0.5 --max-new-tokens 1 --output proxima-tensor/specs/ouro-mainline/results/c2-broadcast-ball-fixed-hidden-r2
/private/tmp/ouro-mainline-oracle-owner-paper --compare-paper proxima-tensor/specs/ouro-mainline/results/c2-broadcast-france-baseline-r2/records.jsonl --reference-oracle /private/tmp/ouro-mainline-oracle-reference/ouro-oracle-paper-baseline-p0-v1 --output proxima-tensor/specs/ouro-mainline/results/c2-compare-broadcast-france-baseline-r2
/private/tmp/ouro-mainline-oracle-owner-paper --compare-paper proxima-tensor/specs/ouro-mainline/results/c2-broadcast-france-fixed-logits-r2/records.jsonl --reference-oracle /private/tmp/ouro-mainline-oracle-reference/ouro-oracle-paper-fixed-logits-p0-v1 --output proxima-tensor/specs/ouro-mainline/results/c2-compare-broadcast-france-fixed-logits-r2
/private/tmp/ouro-mainline-oracle-owner-paper --compare-paper proxima-tensor/specs/ouro-mainline/results/c2-broadcast-france-adaptive-logits-r2/records.jsonl --reference-oracle /private/tmp/ouro-mainline-oracle-reference/ouro-oracle-paper-adaptive-logits-p0-v1 --output proxima-tensor/specs/ouro-mainline/results/c2-compare-broadcast-france-adaptive-logits-r2
/private/tmp/ouro-mainline-oracle-owner-paper --compare-paper proxima-tensor/specs/ouro-mainline/results/c2-broadcast-france-fixed-hidden-r2/records.jsonl --reference-oracle /private/tmp/ouro-mainline-oracle-reference/ouro-oracle-paper-fixed-hidden-p0-v1 --output proxima-tensor/specs/ouro-mainline/results/c2-compare-broadcast-france-fixed-hidden-r2
/private/tmp/ouro-mainline-oracle-owner-paper --compare-paper proxima-tensor/specs/ouro-mainline/results/c2-broadcast-ball-baseline-r2/records.jsonl --reference-oracle proxima-tensor/specs/ouro-mainline/results/native-ball-baseline-r2 --output proxima-tensor/specs/ouro-mainline/results/c2-compare-broadcast-ball-baseline-r2
/private/tmp/ouro-mainline-oracle-owner-paper --compare-paper proxima-tensor/specs/ouro-mainline/results/c2-broadcast-ball-fixed-logits-r2/records.jsonl --reference-oracle proxima-tensor/specs/ouro-mainline/results/native-ball-fixed-logits-r2 --output proxima-tensor/specs/ouro-mainline/results/c2-compare-broadcast-ball-fixed-logits-r2
/private/tmp/ouro-mainline-oracle-owner-paper --compare-paper proxima-tensor/specs/ouro-mainline/results/c2-broadcast-ball-adaptive-logits-r2/records.jsonl --reference-oracle proxima-tensor/specs/ouro-mainline/results/native-ball-adaptive-logits-r1 --output proxima-tensor/specs/ouro-mainline/results/c2-compare-broadcast-ball-adaptive-logits-r2
/private/tmp/ouro-mainline-oracle-owner-paper --compare-paper proxima-tensor/specs/ouro-mainline/results/c2-broadcast-ball-fixed-hidden-r2/records.jsonl --reference-oracle proxima-tensor/specs/ouro-mainline/results/native-ball-fixed-hidden-r1 --output proxima-tensor/specs/ouro-mainline/results/c2-compare-broadcast-ball-fixed-hidden-r2
```

Expect exactly8 candidate selected records and8 comparison directories. Open
their actual summaries and signed difference files:26 vectors,901120
coordinates,12 scalar comparisons,8 selections, original abs/rel bounds .003.
Retain both signs, exact coordinates, maxima/indices, actual/reference fields
and source row numbers. A failing bound remains a failed cell to investigate.
The original wrong-prefix refusal is repeated with fresh ball baseline against
the France baseline reference, expecting one typed missing-match refusal.

Native count and negative-control commands actually executed:
```text
jq -s '{cells:length,vectors:([.[].comparison_records]|add),coordinates:([.[].coordinate_comparisons]|add),scalars:([.[].scalar_comparisons]|add),selections:([.[].selection_comparison_records]|add),exceeding:([.[].exceeding_coordinates]|add),scalar_exceeding:([.[].scalar_exceeding]|add),selection_differences:([.[].selection_differences]|add),structural_errors:([.[].structural_errors]|add),nonfinite:([.[].nonfinite_coordinates]|add)} | .cells==8 and .vectors==26 and .coordinates==901120 and .scalars==12 and .selections==8 and .exceeding==0 and .scalar_exceeding==0 and .selection_differences==0 and .structural_errors==0 and .nonfinite==0' proxima-tensor/specs/ouro-mainline/results/c2-compare-broadcast-*-r2/summary.json
/private/tmp/ouro-mainline-oracle-owner-paper --compare-paper proxima-tensor/specs/ouro-mainline/results/c2-broadcast-ball-baseline-r2/records.jsonl --reference-oracle /private/tmp/ouro-mainline-oracle-reference/ouro-oracle-paper-baseline-p0-v1 --output proxima-tensor/specs/ouro-mainline/results/c2-broadcast-wrong-prefix-control-r2
```
Expected count assertion true; wrong-prefix returned1 with matching candidate
record absent. Failure text retained in omega-broadcast-c2-wrong-prefix-r2.txt.
