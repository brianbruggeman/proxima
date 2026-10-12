# policy-preserving broadcast fold -- slices

| # | slice | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|
| 1 | extract original ordered loop; compose cooperative scalar publication plus existing broadcast tail; add3 emitter/2 Metal literals | AC1,AC2,AC6,AC7 | exact native commands in BROADCAST-FOLD-SPEC | 3 emitter+3 specialized+5 target cases;1 alloc artifact;0 strict lib/all-target diagnostics | [x] | raw logs in results; initial unused binding and four all-target diagnostics retained; fixed before freeze |
| 2 | saved-input adaptive replay plus strict live48 and relaxed cached12 checkpoints | AC3,AC4,AC8 | exact replay/live/cached commands in BROADCAST-FOLD-SPEC and BROADCAST-FOLD-CHECKPOINT-COMMANDS | 2 replay cases/38nodes/4 controls;48 strict joins/168vectors/5996544coords/8cells;12 relaxed requests/12 profile pairs;25 identical source rows per bracket | [x] | actual replay-r3 MSL captured after missing-parent r2 capture failure; strict/relaxed conditions separate |
| 3 | eight fresh C2 candidate/reference cells, full ledger and source audit | AC5 | literal invocations in BROADCAST-FOLD-C2-COMMANDS | 8 cells/26vectors/901120coords/12scalars/8selections | [x] | 890894 non-exact coordinates,0 exceed unchanged .003 bounds; wrong-prefix refusal retained; frozen binaries unchanged |

## resume

Last executed slice:3. No commits made.
Source frozen through both timed checkpoints and eight C2 cells.
All raw payloads/count ledgers, failure records, command logs and source diffs
are indexed in results/BROADCAST-FOLD-COMMANDS.md; global evidence is root-owned.

## struck

Global fusion disablement and routing broadcast epilogues through the unsupported
ordinary serial renderer were rejected before implementation.
