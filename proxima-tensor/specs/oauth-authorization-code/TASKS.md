# oauth-authorization-code -- slices

Each slice: one commit, one behaviour change, one validation command, under ~30 minutes.
Update the checkbox and the note IN THE SAME COMMIT as the slice.

| # | slice | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|
| 1 | Add no_std + alloc authorization-code and PKCE transaction FSM | AC1, AC2, AC3, AC4, AC7, AC8, AC9 | `cargo nextest run -p proxima-auth --no-default-features --features alloc,oauth-authorization-code` | 21 tests run, 21 passed, 0 skipped; 13 authorization-code cases plus 8 token lifecycle cases | [x] | AC1/2/3/4/7/8/9 filters reported 1/13/1/1/13/1/1 tests passed respectively |
| 2 | Add optional configured browser-launch edge while retaining manual URL mode | AC5 | `cargo nextest run -p proxima-auth --features oauth-browser browser_cases` | 2 browser setting tests run, 2 passed; manual invokes 0 times and launch invokes once | [x] | the full browser feature run reported 23 tests passed, 0 skipped |
| 3 | Preserve existing client-credentials behavior and include it in the auth feature gate | AC6, AC10 | `bash scripts/proxima-auth-gate.sh` | 22 gate cells passed, 0 failed; four existing OAuth middleware cases passed | [x] | gate command ran with `RUSTC_WRAPPER` and `CARGO_BUILD_RUSTC_WRAPPER` unset because sccache could not launch rustc |

## resume

Last completed slice: 3 (working tree; no commit created)
Next action: none; the acceptance commands have executed
Open question, if any: none; scope is the OAuth authorization-code client flow

## struck

-
