#!/usr/bin/env bash
# whisper_reference: the committed incumbent arm for the speech lane.
#
# Mirrors scripts/llama_reference: a pinned external checkout, no assumed
# default, real data only, and it prints what actually ran. The flags are the
# spec's reference command (greedy, temperature 0, language en, no timestamps,
# no fallback), not a second benchmark configuration.
#
# usage: ./run.sh
# env:
#   WHISPER_CPP_CHECKOUT  default ~/repos/others/whisper.cpp
#   WHISPER_CLI           default $WHISPER_CPP_CHECKOUT/build/bin/whisper-cli
#   MODEL_PATH            default ~/.cache/kvgate/proxnum610/ggml/ggml-tiny.bin
#   AUDIO                 default $WHISPER_CPP_CHECKOUT/samples/jfk.wav
#   WHISPER_REF_OUT       default ~/.cache/kvgate/proxnum610/ref-tiny-run

set -euo pipefail

checkout="${WHISPER_CPP_CHECKOUT:-$HOME/repos/others/whisper.cpp}"
cli="${WHISPER_CLI:-$checkout/build/bin/whisper-cli}"
model="${MODEL_PATH:-$HOME/.cache/kvgate/proxnum610/ggml/ggml-tiny.bin}"
audio="${AUDIO:-$checkout/samples/jfk.wav}"
out="${WHISPER_REF_OUT:-$HOME/.cache/kvgate/proxnum610/ref-tiny-run}"

for file in "$cli" "$model" "$audio"; do
  if [[ ! -f "$file" ]]; then
    printf 'RED: %s not found\n' "$file" >&2
    exit 1
  fi
done

if [[ ! -f "$checkout/CMakeCache.txt" && ! -f "$checkout/build/CMakeCache.txt" ]]; then
  printf 'RED: %s is not a built whisper.cpp checkout\n' "$checkout" >&2
  exit 1
fi

mkdir -p "$out"
log="$out/ref.log"
set +e
"$cli" -m "$model" -f "$audio" -l en -nt -bo 1 -bs 1 -tp 0 -tpi 0 -nf \
  -ojf -of "$out/ref" >"$log" 2>&1
status=$?
set -e
if [[ $status -ne 0 ]]; then
  printf 'RED: whisper-cli exited %d; log %s\n' "$status" "$log" >&2
  exit 1
fi

metal=0
if grep -q 'MTL : EMBED_LIBRARY = 1' "$log"; then
  metal=1
fi
mel="$(awk -F'= ' '/mel time/ {print $2}' "$log" | awk '{print $1}')"
encode="$(awk -F'= ' '/encode time/ {print $2}' "$log" | awk '{print $1}')"
decode="$(awk -F'= ' '/decode time/ {print $2}' "$log" | awk '{print $1}')"
text="$(python3 - "$out/ref.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
print(value["transcription"][0]["text"])
PY
)"

printf 'checkout=%s\n' "$checkout"
printf 'model=%s\n' "$model"
printf 'audio=%s\n' "$audio"
printf 'metal=%s\n' "$metal"
printf 'transcript=%s\n' "$text"
printf 'mel_ms=%s\n' "$mel"
printf 'encode_ms=%s\n' "$encode"
printf 'decode_ms=%s\n' "$decode"
printf 'log=%s\n' "$log"
