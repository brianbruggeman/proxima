#!/usr/bin/env bash
# no flag: compile each tests/corpus member with the nova backend and stock rustc, run both, compare exit codes
# --control: nova's arith against llvm's match_int must mismatch, and a corpus copy with a dead fn must warn twice
# --refused: each tests/refused file must pass refusal_verdict (nova fails, leaves no binary, and the adapter's
#   refusal line names the file's construct, case-sensitive); two detector controls must then print refused=0:
#   arith.rs (accepted, fails the status test) and shl.rs with token F64 (refused, fails the reason test)
# --layout: the adapter's struct layout lines against stock rustc's -Zprint-type-sizes
# --casts: the adapter's nova_ir lines for cast_grid and wide_isize, counting Sextend, Uextend and Ireduce; arith is the control
# --traps: Unreachable counts in the nova_ir lines of call_unit (finish), enum_match and enum_option (main); arith is the control
# --derefs: the nova_deref line of ref_widths and ref_guarded (main) against counts from their MIR; arith is the control
# --data: each static's segment,section in nova's and LLVM's --emit=obj output, and static_nested's __DATA,__const relocation count
set -uo pipefail

mode="${1:-}"
root="$(cd "$(dirname "$0")/.." && pwd)"

# fail closed: every tool this script shells out to must exist before anything builds. Without this,
# a missing `timeout` makes every member read build-failed and the run prints `match=0 of=N` with a
# zero exit (the 2026-11-21 verifier finding, ~/.cache/proxima-nova/PROXIMANOVA-51/verify/).
for tool in timeout rustc cargo; do
    command -v "$tool" >/dev/null 2>&1 || { echo "missing=$tool"; exit 2; }
done
# --data is the only mode that reads Mach-O, so it is the only one that needs these two
if [[ "$mode" == "--data" ]]; then
    for tool in nm otool; do
        command -v "$tool" >/dev/null 2>&1 || { echo "missing=$tool"; exit 2; }
    done
fi
toolchain="nightly-2026-11-15"
cache="$HOME/.cache/proxima-nova/PROXIMANOVA-28"
mkdir -p "$cache"
run="$(mktemp -d "$cache/run.XXXXXX")"
trap 'rm -rf "$run"' EXIT

case "$(uname -s)" in
    Darwin) pattern='libproxima_nova_rustc\.dylib' ;;
    Linux) pattern='libproxima_nova_rustc\.so' ;;
    *) pattern='proxima_nova_rustc\.dll' ;;
esac

CARGO_TARGET_DIR="$root/target" CARGO_BUILD_JOBS=2 \
    cargo "+$toolchain" build --manifest-path "$root/Cargo.toml" \
    --message-format=json-render-diagnostics >"$run/build.json"
# cargo reports where it uplifted the dylib, so its location is never assumed
dylib="$(grep -m 1 -oE "[^\"]*$pattern" "$run/build.json" || true)"
[[ -n "$dylib" ]] || { echo "cargo reported no dylib" >&2; exit 1; }

rustc_args=(--edition 2024 -C panic=abort -C overflow-checks=off --crate-type bin)
# the fixtures are no_std and rustc links with -nodefaultlibs, so dyld_stub_binder needs libSystem explicitly
[[ "$(uname -s)" == Darwin ]] && rustc_args+=(-C link-arg=-lSystem)

nova_log="$run/nova.log"
: >"$nova_log"

# compile with the nova backend; stderr is appended to the per-run log and kept per source in $run/<stem>.nova.err
compile_nova() {
    local source="$1" out="$2" stem
    stem="$(basename "$source" .rs)"
    timeout -k 5s 60s rustc "+$toolchain" "${rustc_args[@]}" "-Zcodegen-backend=$dylib" "$source" -o "$out" 2>"$run/$stem.nova.err"
    local status=$?
    cat "$run/$stem.nova.err" >>"$nova_log"
    return $status
}

compile_llvm() {
    timeout -k 5s 60s rustc "+$toolchain" "${rustc_args[@]}" "$1" -o "$2" 2>"$run/$(basename "$1" .rs).llvm.err"
}

# prints the exit code of the binary under a 10 s bound, or a marker when it never ran
run_bounded() {
    [[ -x "$1" ]] || { echo "build-failed"; return; }
    timeout -k 2s 10s "$1"
    local code=$?
    [[ $code -eq 124 ]] && { echo "timeout"; return; }
    echo "$code"
}

nova_exit() {
    local source="$1" out="$run/$(basename "$1" .rs).nova.bin"
    compile_nova "$source" "$out"
    run_bounded "$out"
}

llvm_exit() {
    local source="$1" out="$run/$(basename "$1" .rs).llvm.bin"
    compile_llvm "$source" "$out"
    run_bounded "$out"
}

# prints 1 when nova refuses the source, leaves no binary and its adapter line names the token (case-sensitive), else 0
refusal_verdict() {
    local source="$1" token="$2" stem out status named
    stem="$(basename "$source" .rs)"
    out="$run/$stem.nova.bin"
    rm -f "$out"
    compile_nova "$source" "$out"
    status=$?
    named="$(grep -c "proxima-nova: .*$token" "$run/$stem.nova.err" || true)"
    if [[ $status -ne 0 && ! -e "$out" && "$named" -ge 1 ]]; then
        echo 1
    else
        echo 0
    fi
}

count_warnings() {
    grep -c '^warning' "$1" || true
}

# --layout helpers. nova side: the adapter's nova_layout lines; rustc side: stock -Zprint-type-sizes only
layout_line_pattern='^nova_layout type=([^ ]+) slot_size=([0-9]+) slot_align=([0-9]+) fields=(.*)$'

# prints the nova_layout lines of a build that still links
nova_layout_lines() {
    local stem
    stem="$(basename "$1" .rs)"
    timeout -k 5s 60s rustc "+$toolchain" "${rustc_args[@]}" "-Zcodegen-backend=$dylib" -Cllvm-args=nova-layout "$1" -o "$run/$stem.layout.bin" 2>"$run/$stem.layout.err" | grep '^nova_layout '
}

rustc_type_sizes() {
    local stem
    stem="$(basename "$1" .rs)"
    timeout -k 5s 60s rustc "+$toolchain" "${rustc_args[@]}" -Zprint-type-sizes "$1" -o "$run/$stem.sizes.bin" 2>&1
}

# the print-type-size lines of one type: its header line, then its field and padding lines
type_block() {
    awk -v want="$2" '
        /^print-type-size type: / {
            if (inblock) exit
            name = $3; gsub(/[`:]/, "", name); sub(/.*::/, "", name)
            if (name == want) inblock = 1
        }
        inblock && /^print-type-size/ { print }
    ' "$1"
}

# prints "<size> <align> <field>@<offset>,... <field>:<width>,..." with fields in layout order; an offset is the
# sum of the field widths and padding printed before the field
rustc_layout() {
    type_block "$1" "$2" | awk '
        BEGIN { offset = 0 }
        $2 == "type:" { size = $4; align = $7 }
        $2 == "field" {
            name = $3; gsub(/[`:]/, "", name); sub(/^\./, "", name)
            offsets = offsets (offsets == "" ? "" : ",") name "@" offset
            widths = widths (widths == "" ? "" : ",") name ":" $4
            offset += $4
        }
        $2 == "padding:" { offset += $3 }
        END { print size, align, offsets, widths }
    '
}

sorted_fields() {
    echo "$1" | tr ',' '\n' | sort | paste -sd, -
}

# declaration-order offsets, each field aligned to its own width: "$1" is the declared field list, "$2" the widths
declared_offsets() {
    awk -v declared="$1" -v widths="$2" '
        BEGIN {
            count = split(widths, pairs, ",")
            for (index_ = 1; index_ <= count; index_++) { split(pairs[index_], pair, ":"); width[pair[1]] = pair[2] }
            count = split(declared, fields, ",")
            for (index_ = 1; index_ <= count; index_++) {
                split(fields[index_], pair, "@")
                size = width[pair[1]]
                offset = int((offset + size - 1) / size) * size
                out = out (out == "" ? "" : ",") pair[1] "@" offset
                offset += size
            }
            print out
        }
    '
}

# --derefs helper: the adapter's nova_deref lines of a build that still links
nova_deref_lines() {
    local stem
    stem="$(basename "$1" .rs)"
    timeout -k 5s 60s rustc "+$toolchain" "${rustc_args[@]}" "-Zcodegen-backend=$dylib" -Cllvm-args=nova-layout "$1" -o "$run/$stem.deref.bin" 2>"$run/$stem.deref.err" | grep '^nova_deref ' || true
}

# --casts helpers: the adapter's nova_ir lines of a build that still links
nova_ir_lines() {
    local stem
    stem="$(basename "$1" .rs)"
    timeout -k 5s 60s rustc "+$toolchain" "${rustc_args[@]}" "-Zcodegen-backend=$dylib" -Cllvm-args=nova-ir "$1" -o "$run/$stem.ir.bin" 2>"$run/$stem.ir.err" | grep '^nova_ir ' || true
}

# prints "<functions> <sextend> <uextend> <ireduce>" for a nova_ir listing: the line count, then the occurrences
ir_counts() {
    awk '
        /^nova_ir / { functions++ }
        { line = $0
          while (match(line, /Sextend \{/)) { sextend++; line = substr(line, RSTART + RLENGTH) }
          line = $0
          while (match(line, /Uextend \{/)) { uextend++; line = substr(line, RSTART + RLENGTH) }
          line = $0
          while (match(line, /Ireduce \{/)) { ireduce++; line = substr(line, RSTART + RLENGTH) } }
        END { print functions + 0, sextend + 0, uextend + 0, ireduce + 0 }
    ' "$1"
}

ir_report() {
    read -r functions sextend uextend ireduce <<<"$1"
    echo "functions=$functions sextend=$sextend uextend=$uextend ireduce=$ireduce"
}

enum_line_pattern='^nova_layout type=([^ ]+) slot_size=([0-9]+) slot_align=([0-9]+) tag=0:([0-9]+) access=([^ ]*) fields=(.*)$'

# prints "<size> <align> <tag width> <variants> <Variant.i@offset,...> <Variant.i:width,...> <offset:width,...>"; the
# tag is at offset 0, each variant's fields run in layout order from the tag's width, adding padding
rustc_enum() {
    type_block "$1" "$2" | awk '
        $2 == "type:" { size = $4; align = $7 }
        $2 == "discriminant:" { tag = $3; offset = tag; table = "0:" tag }
        $2 == "variant" { name = $3; gsub(/[`:]/, "", name); offset = tag; variants++ }
        $2 == "field" {
            field = $3; gsub(/[`:]/, "", field); sub(/^\./, "", field)
            offsets = offsets (offsets == "" ? "" : ",") name "." field "@" offset
            widths = widths (widths == "" ? "" : ",") name "." field ":" $4
            table = table "," offset ":" $4
            offset += $4
        }
        $2 == "padding:" { offset += $3 }
        END { print size, align, tag, variants + 0, (offsets == "" ? "-" : offsets), (widths == "" ? "-" : widths), table }
    '
}

# declaration-order offsets for an enum: each variant starts after the tag, each field aligned to its width
declared_enum_offsets() {
    awk -v declared="$1" -v widths="$2" -v tag="$3" '
        BEGIN {
            count = split(widths, pairs, ",")
            for (index_ = 1; index_ <= count; index_++) { split(pairs[index_], pair, ":"); width[pair[1]] = pair[2] }
            count = split(declared, fields, ",")
            for (index_ = 1; index_ <= count; index_++) {
                split(fields[index_], pair, "@")
                split(pair[1], parts, ".")
                if (parts[1] != previous) { offset = tag; previous = parts[1] }
                size = width[pair[1]]
                offset = int((offset + size - 1) / size) * size
                out = out (out == "" ? "" : ",") pair[1] "@" offset
                offset += size
            }
            print out
        }
    '
}

# 1 only when every offset in the access list is a tag or field offset rustc has and each width listed at it is the
# size of a tag or field rustc places there; "$1" is "<offset>:<w>/<w>,...", "$2" is rustc's "<offset>:<width>,..."
access_verdict() {
    awk -v access="$1" -v table="$2" '
        BEGIN {
            count = split(table, entries, ",")
            for (index_ = 1; index_ <= count; index_++) { split(entries[index_], pair, ":"); known[pair[1] ":" pair[2]] = 1; seen[pair[1]] = 1 }
            verdict = 1
            count = split(access, entries, ",")
            if (access == "") verdict = 0
            for (index_ = 1; index_ <= count; index_++) {
                split(entries[index_], pair, ":")
                if (!(pair[1] in seen)) verdict = 0
                wide = split(pair[2], widths, "/")
                for (inner = 1; inner <= wide; inner++) if (!((pair[1] ":" widths[inner]) in known)) verdict = 0
            }
            print verdict
        }
    '
}

# 1 only when the field offsets, the size and the alignment all agree
layout_verdict() {
    if [[ -n "$4" && "$1" == "$4" && "$2" == "$5" && "$3" == "$6" ]]; then echo 1; else echo 0; fi
}

# "<segment>,<section>" of the static whose mangled name ends in the identifier $2, empty when absent
data_section() {
    nm -m "$1" 2>/dev/null | grep -E " __R[^ ]*[0-9]$2\$" | head -1 | sed -E 's/.*\((__[A-Z]+,__[a-z_0-9]+)\).*/\1/'
}

# the relocation entries of the object's __DATA,__const section, 0 when it has none
data_relocs() {
    otool -r "$1" 2>/dev/null | awk '/Relocation information \(__DATA,__const\)/ { found = $4 } END { print found + 0 }'
}

corpus="${NOVA_CORPUS_DIR:-$root/tests/corpus}"
refused_dir="$root/tests/refused"

case "$mode" in
    "")
        members=("$corpus"/*.rs)
        echo "members=${#members[@]}"
        matched=0
        for source in "${members[@]}"; do
            nova="$(nova_exit "$source")"
            llvm="$(llvm_exit "$source")"
            echo "$(basename "$source" .rs) nova=$nova llvm=$llvm"
            [[ "$nova" =~ ^[0-9]+$ && "$nova" == "$llvm" ]] && matched=$((matched + 1))
        done
        echo "match=$matched of=${#members[@]}"
        echo "nova_warnings=$(count_warnings "$nova_log")"
        [[ "$matched" -eq "${#members[@]}" ]] || exit 1
        ;;
    --control)
        nova="$(nova_exit "$corpus/arith.rs")"
        llvm="$(llvm_exit "$corpus/match_int.rs")"
        echo "control nova=$nova llvm=$llvm"
        # a control whose two arms both failed to build is vacuous, not a mismatch
        [[ "$nova" =~ ^[0-9]+$ && "$llvm" =~ ^[0-9]+$ ]] || { echo "control both_arms_built=0"; exit 1; }
        matched=0
        [[ "$nova" == "$llvm" ]] && matched=1
        echo "control match=$matched of=1"
        [[ "$matched" -eq 0 ]] || exit 1
        # a member with a dead function must warn, so the counter is shown to see warnings
        mkdir -p "$run/dead"
        cp "$corpus/arith.rs" "$run/dead/arith_unused.rs"
        echo 'fn unused_helper() {}' >>"$run/dead/arith_unused.rs"
        : >"$nova_log"
        nova_exit "$run/dead/arith_unused.rs" >/dev/null
        echo "control_warnings=$(count_warnings "$nova_log")"
        ;;
    --refused)
        refused=0
        total=0
        for source in "$refused_dir"/*.rs; do
            stem="$(basename "$source" .rs)"
            case "$stem" in
                shl) names=Shl ;;
                float) names=F64 ;;
                indirect) names=FnPtr ;;
                struct_arg) names=StructArgument ;;
                struct_ret) names=StructResult ;;
                struct_nested) names=NestedStruct ;;
                narrow_i8) names=I8 ;;
                narrow_u8) names=U8 ;;
                narrow_i16) names=I16 ;;
                narrow_u16) names=U16 ;;
                wide_i128) names=I128 ;;
                wide_u128) names=U128 ;;
                opt_bool) names=NicheEncoding ;;
                order_backend_then_mir) names=F64 ;;
                track_caller_assert) names=TrackCaller ;;
                track_caller_nested) names=TrackCaller ;;
                *) names="" ;;
            esac
            total=$((total + 1))
            verdict="$(refusal_verdict "$source" "$names")"
            [[ "$verdict" == 1 ]] && refused=$((refused + 1))
            echo "$stem refused=$verdict names=$names"
        done
        echo "refused=$refused of=$total"
        # the detector must be able to say no: an accepted member fails its status test, and a refusal
        # naming another construct fails its reason test
        control_refused="$(refusal_verdict "$corpus/arith.rs" F64)"
        control_reason="$(refusal_verdict "$refused_dir/shl.rs" F64)"
        echo "control refused=$control_refused"
        echo "control_reason refused=$control_reason"
        [[ "$refused" -eq "$total" && "$control_refused" == 0 && "$control_reason" == 0 ]] || exit 1
        ;;
    --layout)
        layout_total=0
        layout_matched=0
        for stem in struct_fields struct_mixed struct_update; do
            source="$corpus/$stem.rs"
            nova_layout_lines "$source" >"$run/$stem.layout"
            rustc_type_sizes "$source" >"$run/$stem.sizes"
            while IFS= read -r line; do
                [[ -n "$line" ]] || continue
                layout_total=$((layout_total + 1))
                [[ "$line" =~ $layout_line_pattern ]] || { echo "$stem unparsed: $line"; continue; }
                type="${BASH_REMATCH[1]}"
                slot_size="${BASH_REMATCH[2]}"
                slot_align="${BASH_REMATCH[3]}"
                nova_fields="$(sorted_fields "${BASH_REMATCH[4]}")"
                read -r rustc_size rustc_align rustc_offsets rustc_widths < <(rustc_layout "$run/$stem.sizes" "$type")
                rustc_fields="$(sorted_fields "${rustc_offsets:-}")"
                verdict="$(layout_verdict "$nova_fields" "$slot_size" "$slot_align" "$rustc_fields" "${rustc_size:-}" "${rustc_align:-}")"
                [[ "$verdict" == 1 ]] && layout_matched=$((layout_matched + 1))
                echo "$stem type=$type nova=$nova_fields rustc=$rustc_fields slot=$slot_size:$slot_align rustc_size=${rustc_size:-}:${rustc_align:-} match=$verdict"
                if [[ "$stem" == struct_mixed && "$type" == M ]]; then
                    control_declared="$(declared_offsets "${BASH_REMATCH[4]}" "$rustc_widths")"
                    control_nova="$(sorted_fields "$control_declared")"
                    control_verdict="$(layout_verdict "$control_nova" "$slot_size" "$slot_align" "$rustc_fields" "${rustc_size:-}" "${rustc_align:-}")"
                    control_line="control type=M declared=$control_declared rustc=$rustc_fields match=$control_verdict"
                fi
            done <"$run/$stem.layout"
            [[ "$stem" == struct_mixed ]] && type_block "$run/$stem.sizes" M
        done
        enum_controls=()
        for stem in enum_match enum_color enum_negative enum_unsigned_tag enum_reordered enum_copy_tag enum_copy_fields; do
            source="$corpus/$stem.rs"
            nova_layout_lines "$source" >"$run/$stem.layout"
            rustc_type_sizes "$source" >"$run/$stem.sizes"
            while IFS= read -r line; do
                [[ -n "$line" ]] || continue
                layout_total=$((layout_total + 1))
                [[ "$line" =~ $enum_line_pattern ]] || { echo "$stem unparsed: $line"; continue; }
                type="${BASH_REMATCH[1]}"
                slot_size="${BASH_REMATCH[2]}"
                slot_align="${BASH_REMATCH[3]}"
                nova_tag="${BASH_REMATCH[4]}"
                access="${BASH_REMATCH[5]}"
                declared="${BASH_REMATCH[6]}"
                nova_fields="$(sorted_fields "$declared")"
                read -r rustc_size rustc_align rustc_tag rustc_variants rustc_offsets rustc_widths rustc_table < <(rustc_enum "$run/$stem.sizes" "$type")
                [[ "${rustc_offsets:-}" == "-" ]] && rustc_offsets=""
                [[ "${rustc_widths:-}" == "-" ]] && rustc_widths=""
                rustc_fields="$(sorted_fields "${rustc_offsets:-}")"
                verdict=0
                [[ -n "${rustc_size:-}" && "$nova_fields" == "$rustc_fields" && "$slot_size" == "$rustc_size" && "$slot_align" == "${rustc_align:-}" ]] && verdict=1
                access_ok="$(access_verdict "$access" "${rustc_table:-}")"
                [[ "$nova_tag" == "${rustc_tag:-}" && "$access_ok" == 1 ]] || verdict=0
                [[ "$verdict" == 1 ]] && layout_matched=$((layout_matched + 1))
                echo "$stem type=$type nova=$nova_fields rustc=$rustc_fields slot=$slot_size:$slot_align rustc_size=${rustc_size:-}:${rustc_align:-} tag=0:$nova_tag rustc_tag=0:${rustc_tag:-} access=$access match=$verdict"
                if [[ "$stem" == enum_match && "$type" == Shape && ${#enum_controls[@]} -eq 0 ]]; then
                    small_tag=4
                    [[ "${rustc_variants:-0}" -le 256 ]] && small_tag=1
                    enum_controls+=("control type=Shape tag=0:$small_tag rustc_tag=0:${rustc_tag:-} match=$([[ "$small_tag" == "${rustc_tag:-}" ]] && echo 1 || echo 0)")
                fi
                if [[ "$stem" == enum_reordered && "$type" == Mixed ]]; then
                    control_declared="$(declared_enum_offsets "$declared" "$rustc_widths" "${rustc_tag:-0}")"
                    control_nova="$(sorted_fields "$control_declared")"
                    control_verdict="$(layout_verdict "$control_nova" "$slot_size" "$slot_align" "$rustc_fields" "${rustc_size:-}" "${rustc_align:-}")"
                    enum_controls+=("control type=Mixed declared=$control_declared rustc=$rustc_fields match=$control_verdict")
                fi
            done <"$run/$stem.layout"
        done
        echo "layout match=$layout_matched of=$layout_total"
        echo "${control_line:-control type=M unavailable match=0}"
        printf '%s\n' "${enum_controls[@]}"
        [[ "$layout_total" -gt 0 && "$layout_matched" -eq "$layout_total" ]] || exit 1
        ;;
    --casts)
        casts_total=0
        casts_matched=0
        casts_lines=0
        for stem in cast_grid wide_isize arith; do
            case "$stem" in
                cast_grid) expected="2 2 2 2" ;;
                wide_isize) expected="2 2 0 0" ;;
                *) expected="2 0 0 0" ;;
            esac
            nova_ir_lines "$corpus/$stem.rs" >"$run/$stem.ir"
            counts="$(ir_counts "$run/$stem.ir")"
            read -r functions _ <<<"$counts"
            casts_lines=$((casts_lines + functions))
            echo "$stem $(ir_report "$counts")"
            [[ "$stem" == arith ]] && continue
            casts_total=$((casts_total + 1))
            [[ "$counts" == "$expected" ]] && casts_matched=$((casts_matched + 1))
        done
        if [[ "$casts_lines" -eq 0 ]]; then
            echo "casts match=0 of=0"
            exit 1
        fi
        echo "casts match=$casts_matched of=$casts_total"
        [[ "$casts_matched" -eq "$casts_total" ]] || exit 1
        ;;
    --traps)
        traps_total=0
        traps_matched=0
        traps_lines=0
        for entry in call_unit:finish:1 enum_match:main:1 enum_option:main:1 arith:main:0; do
            IFS=: read -r stem symbol expected <<<"$entry"
            nova_ir_lines "$corpus/$stem.rs" >"$run/$stem.ir"
            line="$(grep -m 1 "^nova_ir fn=[^ ]*$symbol" "$run/$stem.ir" || true)"
            [[ -n "$line" ]] && traps_lines=$((traps_lines + 1))
            count="$(grep -o 'Unreachable' <<<"$line" | wc -l | tr -d ' ')"
            echo "$stem fn=$symbol unreachable=$count"
            [[ "$stem" == arith ]] && continue
            traps_total=$((traps_total + 1))
            [[ -n "$line" && "$count" == "$expected" ]] && traps_matched=$((traps_matched + 1))
        done
        if [[ "$traps_lines" -eq 0 ]]; then
            echo "traps match=0 of=0"
            exit 1
        fi
        echo "traps match=$traps_matched of=$traps_total"
        [[ "$traps_matched" -eq "$traps_total" ]] || exit 1
        ;;
    --derefs)
        derefs_total=0
        derefs_matched=0
        derefs_lines=0
        for entry in "ref_widths|loads=1:1,4:4,8:8 stores=1:1,4:2,8:4" "ref_guarded|loads=1:2,4:5 stores=1:1,4:2" "arith|"; do
            stem="${entry%%|*}"
            expected="${entry#*|}"
            nova_deref_lines "$corpus/$stem.rs" >"$run/$stem.deref"
            line="$(grep -m 1 '^nova_deref fn=[^ ]*main[^ ]* ' "$run/$stem.deref" || true)"
            [[ -n "$line" ]] && derefs_lines=$((derefs_lines + 1))
            echo "$line"
            if [[ "$stem" == arith ]]; then
                echo "arith deref_lines=$(grep -c '^nova_deref ' "$run/$stem.deref" || true)"
                continue
            fi
            derefs_total=$((derefs_total + 1))
            verdict=0
            [[ -n "$line" && "${line#nova_deref fn=* }" == "$expected" ]] && verdict=1
            [[ "$verdict" == 1 ]] && derefs_matched=$((derefs_matched + 1))
            echo "$stem fn=main ${line#nova_deref fn=* } match=$verdict"
        done
        echo "derefs match=$derefs_matched of=$derefs_total"
        [[ "$derefs_lines" -gt 0 && "$derefs_matched" -eq "$derefs_total" ]] || exit 1
        ;;
    --data)
        data_total=0
        data_matched=0
        for entry in "static_read|LIMIT" "static_pair|HIGH LOW" "static_nested|INNER OUTER" "static_wide|BIG" "static_struct|ORIGIN" "arith|"; do
            stem="${entry%%|*}"
            statics="${entry#*|}"
            source="$corpus/$stem.rs"
            nova_obj="$run/$stem.nova.o"
            llvm_obj="$run/$stem.llvm.o"
            timeout -k 5s 60s rustc "+$toolchain" "${rustc_args[@]}" "-Zcodegen-backend=$dylib" --emit=obj "$source" -o "$nova_obj" 2>"$run/$stem.nova.err"
            timeout -k 5s 60s rustc "+$toolchain" "${rustc_args[@]}" --emit=obj "$source" -o "$llvm_obj" 2>"$run/$stem.llvm.err"
            if [[ "$stem" == arith ]]; then
                echo "arith statics=$(nm -m "$nova_obj" 2>/dev/null | grep -c ' __R[^ ]*[0-9]*[A-Z_][A-Z_]*$' || true)"
                continue
            fi
            for name in $statics; do
                nova_section="$(data_section "$nova_obj" "$name")"
                llvm_section="$(data_section "$llvm_obj" "$name")"
                verdict=0
                [[ -n "$nova_section" && "$nova_section" == "$llvm_section" ]] && verdict=1
                data_total=$((data_total + 1))
                [[ "$verdict" == 1 ]] && data_matched=$((data_matched + 1))
                echo "$stem static=$name nova=${nova_section:-none} llvm=${llvm_section:-none} match=$verdict"
            done
            if [[ "$stem" == static_nested ]]; then
                nova_relocs="$(data_relocs "$nova_obj")"
                llvm_relocs="$(data_relocs "$llvm_obj")"
                verdict=0
                [[ "$nova_relocs" == "$llvm_relocs" && "$nova_relocs" != 0 ]] && verdict=1
                data_total=$((data_total + 1))
                [[ "$verdict" == 1 ]] && data_matched=$((data_matched + 1))
                echo "$stem data_relocs nova=$nova_relocs llvm=$llvm_relocs match=$verdict"
            fi
        done
        echo "data match=$data_matched of=$data_total"
        [[ "$data_total" -gt 0 && "$data_matched" -eq "$data_total" ]] || exit 1
        ;;
    *)
        echo "usage: nova-corpus.sh [--control|--refused|--layout|--casts|--traps|--derefs|--data]" >&2
        exit 2
        ;;
esac
