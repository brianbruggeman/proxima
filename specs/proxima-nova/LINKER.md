# proxima-nova linker (c13) — inventory and slice ladder

status: inventory complete 2026-11-21 (card PROXIMANOVA-52). Every value below was read from
this machine, not recalled. The measured artifacts are in `~/.cache/proxima-nova/PROXIMANOVA-50/`
(`fdz_nova.o`, `div_nova.bin`, `fail_div_zero_nova.bin`); the commands are in the last section.
goal: `pm/GOALS.md` `native-linker:c13`; requirement shape in `SPEC.md` "linker (c13)".

## 0. SPEC element coverage (the card-52 acceptance)

One row per element of `SPEC.md`'s "linker (c13)" table, each with the section here and the
measured evidence. The close counts rows whose evidence cell is non-empty.

| SPEC element | section | evidence |
|---|---|---|
| inputs | §1, §2 | `otool -hv`, `-r`, `-l` on `fdz_nova.o`, `div_nova.bin`, `fail_div_zero_nova.bin` |
| symbol resolution | §2, §3 | `nm -m` on both linked binaries (`_main`, undefined `_exit`, `dyld_stub_binder`) |
| layout | §2, §3 | `otool -l`: vmaddr, vmsize, fileoff, align on both |
| relocations | §1, §2 | `otool -r fdz_nova.o`; dyld rebase/bind sizes in `LC_DYLD_INFO_ONLY` |
| external calls | §2, §3, §4 (L3) | `otool -Iv`: one `_exit` stub, one `__la_symbol_ptr`, one `dyld_stub_binder` got |
| load commands | §1, §2, §3 | `otool -l` command lists, object and executable |
| output | §4 | contract, not yet built: the L1 card's `nova-link-corpus.sh` prints `linked=`, `spawned=`, `exits=`, `codesign=` |
| oracle | §5 | contract: the same script's `exits=N of=N` and `codesign=1`, beside `otool -l`/`-r` and `nm -m` on the result |

## 1. what the compiler emits today

`ObjectBuilder` (`proxima-nova/src/object.rs:90` `new`, `:117` `feed`, `:138` `feed_data`,
`:185` `finish`) writes one arm64 Mach-O **relocatable object** per invocation:

| element | measured value (`otool -hv fdz_nova.o`) |
|---|---|
| magic / type | `MH_MAGIC_64`, `filetype OBJECT` |
| cpu | `ARM64 ALL` |
| load commands | 4: one `LC_SEGMENT_64` (312 bytes, 3 sections: `__TEXT,__text`, `__TEXT,__const`, `__DATA,__const`), `LC_BUILD_VERSION` (24, `platform 1 minos 11.0 sdk n/a`), `LC_SYMTAB` (24), `LC_DYSYMTAB` (80); 312+24+24+80 = the header's 440 `sizeofcmds` |
| text relocations | 7, all `extern`, length 2: `BRANCH26` (type 2, pcrel 1), `PAGE21` (3, pcrel 1), `PAGEOFF12` (4, pcrel 0) |
| data relocations | 2 in `__DATA,__const`, length 3 (`ARM64_RELOC_UNSIGNED`) |

The relocation set is closed in code: `RelocationKind` has three variants only
(`proxima-nova/src/pipeline.rs:18`: `Branch26`, `Page21`, `PageOff12`), emitted by
`proxima-nova/src/aarch64/lower.rs:359`; the writer maps them at `object.rs:222`
(`branch26` `:274`, `page` `:233`), and pointer slots at `object.rs:247` (`unsigned64`).
A linker for nova therefore starts with **four** relocation kinds: the three code kinds above
plus `ARM64_RELOC_UNSIGNED` for a data pointer slot (`object.rs:247`). No subtractor relocation
is emitted (`grep -rn -e SUBTRACTOR -e subtractor proxima-nova/src` prints 0), so it is not in
slice 1.

## 2. what ld64 produces for the same input

`div_nova.bin` (rustc + nova object + `cc`, the current path; built while card 50's phase-B
battery ran, logs in `PROXIMANOVA-50/phaseB/`, no build script kept — `ret.sh` builds the
`ret42`/`ret7` fixtures, not this one) and `fail_div_zero_nova.bin` (same, with an external
`_exit` call).

| element | measured value |
|---|---|
| segments | `__PAGEZERO`, `__TEXT` (vmaddr `0x100000000`, vmsize `0x4000`, fileoff 0), `__DATA_CONST` (`0x100004000`), `__DATA` in the `_exit` binary, `__LINKEDIT` (`0x100008000`, fileoff 32768) |
| sections, `div` | `__TEXT.__text` (addr `0x100000440`, off 1088, align 2^6), `__TEXT.__const` (`0x100000700`), `__TEXT.__unwind_info`, `__DATA_CONST.__const` |
| sections, `_exit` | adds `__TEXT.__stubs`, `__TEXT.__stub_helper`, `__DATA_CONST.__got`, `__DATA.__la_symbol_ptr`, `__DATA.__data` |
| load commands | `div`: 16 (4 `LC_SEGMENT_64` + 12); `fail_div_zero`: 17 (5 segments; adds `__DATA`), `__LINKEDIT` at `0x10000c000`/fileoff 49152 instead of `0x100008000`/32768. The 12 non-segment ones: `LC_DYLD_INFO_ONLY`, `LC_SYMTAB`, `LC_DYSYMTAB`, `LC_LOAD_DYLINKER` (`/usr/lib/dyld`), `LC_UUID`, `LC_BUILD_VERSION` (`platform 1 minos 11.0 sdk 26.2`), `LC_SOURCE_VERSION`, `LC_MAIN` (`entryoff 1112`), `LC_LOAD_DYLIB` (`/usr/lib/libSystem.B.dylib`), `LC_FUNCTION_STARTS`, `LC_DATA_IN_CODE`, `LC_CODE_SIGNATURE` |
| dyld info, `div` | `rebase_off 32768 rebase_size 8`; `bind_size 0` |
| dyld info, `_exit` | `rebase_size 16`, `bind_size 24`, `lazy_bind_size 16` |
| stub | `__TEXT.__stubs` 1 entry: addr `0x10000087c` → `_exit`; `bl 0x10000087c ; symbol stub for: _exit` |
| indirect | `__DATA_CONST.__got` → `dyld_stub_binder`; `__DATA.__la_symbol_ptr` → `_exit` |
| undefined | `div`: `dyld_stub_binder` only; `fail_div_zero`: `_exit` and `dyld_stub_binder` (both from libSystem) |
| signature | `codesign -dv`: `flags=0x20002(adhoc,linker-signed)`, `Signature=adhoc`, embedded; `codesign -v` exits 0 |
| entry | `_main` at `0x100000458` in `div`; `LC_MAIN entryoff` points at the `__TEXT` file offset, not the vmaddr |

Three facts the design turns on:

1. **The ad-hoc signature is mandatory.** Every arm64 macOS executable ld64 writes is
   linker-signed (measured above); an unsigned Mach-O is killed at exec. Slice L1 cannot
   defer it.
2. **`__unwind_info` is written even for `panic=abort` fixtures** (`div` has one, 0x60 bytes).
   A first slice may emit an empty or absent `__unwind_info` only if the binary still runs;
   that is a measured question for L1, not an assumption.
3. **An external call costs the full lazy-binding apparatus**, not just a `bl`: stubs,
   `__stub_helper`, `__got` for `dyld_stub_binder`, `__la_symbol_ptr`, indirect symbol
   table, bind and lazy-bind opcodes. That is why L3 is separate from L1.

## 3. symbol resolution and layout rules (slice scope)

| rule | slice |
|---|---|
| a defined global wins; two definitions of the same name is an error naming the symbol | L1 |
| an undefined symbol is an error naming it, unless it is in a loaded dylib | L1 (error) / L3 (libSystem) |
| text starts at `__TEXT` + header and load-command size, 64-byte aligned (ld64 chose off 1088 for `div`; nova must choose its own, recorded and compared) | L1 |
| vmaddr base `0x100000000`; arm64 pages are 16 KiB (`vmsize 0x4000` for a 16384-byte `__TEXT`) | L1 |
| `LC_MAIN.entryoff` is the file offset of `_main`, not its vmaddr | L1 |
| `__PAGEZERO` covers the low 4 GiB with no file bytes | L1 |
| pointer slots in `__DATA,__const` become rebase opcodes in `LC_DYLD_INFO_ONLY` | L2 |
| external calls go through a 12-byte stub (`adrp x16, __la_symbol_ptr; ldr x16, [x16]; br x16`, measured target `0x100008000`) and lazy binding | L3 |
| archives resolve members in order, only for undefined symbols, once per member | L4 |
| dylib output adds `LC_ID_DYLIB`, exports trie, `-install_name` | L5 |
| other architectures/formats reuse the same layout/binding core with per-format writers | after c12 |

## 4. slice ladder (each one card, ≤20 minutes)

Each `linked=`, `spawned=`, `codesign=` and `exits=` token below is printed by
`proxima-nova/scripts/nova-link-corpus.sh`, which the L1 card delivers. Until that script
exists, c13 stays open; the tokens are the contract it must print, not a measurement today.

| slice | content | falsifiable check |
|---|---|---|
| **L1** | **`_main` only**: parse nova objects, resolve the symbol table, map `__TEXT.__text`, write `__PAGEZERO`, `__TEXT`, `__LINKEDIT`, `LC_LOAD_DYLINKER` (`/usr/lib/dyld`; a user-space executable runs through dyld even with no dylibs — measured 2026-11-22: without it the kernel kills the exec), `LC_DYLD_INFO_ONLY` with zero sizes, `LC_MAIN`, `LC_BUILD_VERSION`, `LC_SYMTAB`, `LC_DYSYMTAB`, `LC_CODE_SIGNATURE`, ad-hoc signature. No data, no stubs, no libSystem. | `nova-link-corpus.sh` prints `linked=1 of=1 spawned=0 exits=1 of=1 codesign=1`; `ret42` by nova = 42; the system-linker control prints `linked=1 of=1 spawned=1 exits=1 of=1`. `spawned=` counts platform-linker execs, not a source grep |
| **L2** | data objects: `__DATA_CONST.__const`, `ARM64_RELOC_UNSIGNED`, rebase opcodes, `LC_DYLD_INFO_ONLY` | `linked=4 of=4 exits=4 of=4` for `static_read` (81), `static_pair` (101), `static_nested` (91), `ptr_offset_const` (80); `otool -l` on each shows `__DATA_CONST,__const`, `LC_DYLD_INFO_ONLY`, and `rebase_size` equal to that object's `ARM64_RELOC_UNSIGNED` slot count |
| **L3** | external calls: stubs, `__stub_helper`, `__got` (`dyld_stub_binder`), `__la_symbol_ptr`, lazy bind, indirect symbol table, `LC_LOAD_DYLIB` libSystem | `linked=4 of=4 exits=4 of=4` for `fail_div_zero` (19), `fail_div_overflow` (36), `call_extern` (56), `fail_index` (22); `otool -Iv` shows one `_exit` stub and one `__la_symbol_ptr` entry, and `nm -u` lists `_exit` once |
| **L4** | multiple objects and static archives: member selection, section merging, symbol ordering, duplicate-definition error | the c2 corpus linked from per-function objects; `linked=N of=N` matches LLVM's N; duplicate definition prints `symbol=...` and exits non-zero |
| **L5** | dylib output, then ELF/COFF for c12's targets | `LC_ID_DYLIB` + a `dlopen`ed symbol; `targets_green=6 of=6` with c12 |

L1 is the card after this one. Its kill criterion: if a hand-written signature is not accepted
by `codesign -v`/exec, stop and record the CodeDirectory fields (`v=20400`, `flags=0x20002`,
`hashes=9+0`, `location=embedded`) that must change; the signature is generated by
`proxima-link`, not by shelling out to `codesign`.

## 5. oracle

The oracle for every slice is the incumbent, not a reference document: for each fixture, the
binary nova linked must exit with the same code as the `cc`/ld64-linked binary of the same
source. Structural checks accompany the exit so a wrong-but-lucky layout cannot pass:
`otool -l` (segment and section vmaddrs/fileoffs, load-command set, `LC_MAIN.entryoff`),
`otool -r` (relocations consumed), `nm -m` (defined/undefined symbol sets and their sections)
and `codesign -v` (exit 0). Where ld64's choice is free (text alignment, `LC_UUID`), the
check compares the property, not the byte; where it is not free (entryoff, rebase/bind
opcodes, signature validity), the check compares the value. The control arm repeats the
same runs through the system linker and must print the same exits plus `spawned=1`.

## 6. open questions and their owners

| question | how it is settled | owner slice |
|---|---|---|
| is `__unwind_info` required for a running binary, or does an empty one suffice? | build L1 both ways and run | L1 |
| which CodeDirectory hash type and page size does an ad-hoc arm64 signature need? | `codesign -d --verbose=4` plus a hand-written and a known-good directory compared byte for byte | L1 |
| does an absent `LC_LOAD_DYLINKER` still give a running binary? | L1 with and without it | L1 |
| lazy vs non-lazy binding for `_exit` | bind both ways; compare `dyld` behaviour and the bind opcodes to ld64's | L3 |
| does an external `_exit` work when bound non-lazily at launch? | run `fail_div_zero` both ways | L3 |
| archive member order and one-pass vs multi-pass resolution | L4 with two members defining the same symbol order | L4 |

## 7. provenance (commands run 2026-11-21)

    otool -hv  ~/.cache/proxima-nova/PROXIMANOVA-50/fdz_nova.o
    otool -r   ~/.cache/proxima-nova/PROXIMANOVA-50/fdz_nova.o
    otool -l   ~/.cache/proxima-nova/PROXIMANOVA-50/fdz_nova.o
    otool -l   ~/.cache/proxima-nova/PROXIMANOVA-50/div_nova.bin
    otool -l   ~/.cache/proxima-nova/PROXIMANOVA-50/fail_div_zero_nova.bin
    otool -Iv  ~/.cache/proxima-nova/PROXIMANOVA-50/fail_div_zero_nova.bin
    otool -tvV ~/.cache/proxima-nova/PROXIMANOVA-50/fail_div_zero_nova.bin
    nm -m      ~/.cache/proxima-nova/PROXIMANOVA-50/fail_div_zero_nova.bin
    codesign -dv ~/.cache/proxima-nova/PROXIMANOVA-50/div_nova.bin
    codesign -v  ~/.cache/proxima-nova/PROXIMANOVA-50/div_nova.bin
    ld -v                          # ld-1230.1
    xcrun --show-sdk-path          # MacOSX.sdk/usr/lib/libSystem.tbd exists

The measured exec baseline for c13's process check is the current path: `ret.sh` invokes
`rustc` with `-C link-arg=-lSystem`, and rustc spawns `cc` (one exec) which spawns `ld`
(one exec). The c13 check must count those to 0 on the nova arm and to at least 1 on the
control arm; it is a spawn trace, not a source grep.
