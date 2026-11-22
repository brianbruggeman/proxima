#!/usr/bin/env bash
# usage: nova-obj-u32.sh OBJECT SYMBOL OFFSET
# prints the little-endian u32 at SYMBOL+OFFSET in a Mach-O object; output goes to files, no pipes
set -euo pipefail
obj="$1"
sym="$2"
off="$3"
nm -m "$obj" >"$obj.nm"
sect=""
addr=""
while read -r a s _ name; do
    if [[ "$name" == "$sym" || "$name" == "_$sym" ]]; then
        addr="$a"
        sect="$s"
        break
    fi
done <"$obj.nm"
[[ -n "$sect" ]] || { echo "no symbol $sym" >&2; exit 1; }
sect="${sect#(}"
sect="${sect%)}"
otool -s "${sect%,*}" "${sect#*,}" "$obj" >"$obj.dump"
target=$((16#$addr + off))
while read -r line_addr w0 w1 w2 w3; do
    [[ "$line_addr" =~ ^[0-9a-f]{16}$ ]] || continue
    start=$((16#$line_addr))
    if ((target >= start && target < start + 16)); then
        words=("$w0" "$w1" "$w2" "$w3")
        echo $((16#${words[(target - start) / 4]}))
        exit 0
    fi
done <"$obj.dump"
echo "offset $off not in section" >&2
exit 1
