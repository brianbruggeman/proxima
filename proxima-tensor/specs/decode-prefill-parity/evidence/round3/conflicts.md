# round three integration (main 0688ae5b)

1. `git am --3way` of headchunk 0001-0002, capture 0001-0004, foldrouter 0002-0003: 8 commits, no textual conflict (am.log). foldrouter 0001 was skipped: the diff bodies (everything from the first `diff --git`) of `headchunk/0001` and `foldrouter/0001` were compared with `diff` and the output was empty, so the patches are the same change.
2. The route series (`par3/route/patches/`, 5 files, all older than 2 minutes when applied) applied with `git am --3way` after the first gate pass, no conflict. The directory held the full series (segmented scan, its tests, compacted prepass, tile location from compacted offsets, tests).
3. Integration fixes, one change each, are the commits listed in the SPEC section "round three result".
4. A shared cargo target directory between the main checkout and an export of 0688ae5b reused a stale `omega` build script (same relative package path, older source mtimes): the first sweep builds failed on `[grouped_gemm].col_parts`. The base binary is now built in its own target directory; the tip binary was rebuilt from clean and its sha256 matched the first build.
