#!/usr/bin/env bash
# regenerates the committed raw-record fixtures from gen-whisper-metal.jq; the test asserts they are current.
#   whisper_metal.jsonl    20 processes per arm, 3 inputs, one discarded warm-up per arm, 4 not-run backends
#   percentile_1_20.jsonl  the same, proxima cold ttft_us replaced by 1..20 (nearest rank 10/18/19/20)
#   cov_5.jsonl            first five proxima cold processes with ttlt_us 10,11,10,12,11 s (sample CoV 7.75%)
#   llm_prefill.jsonl      the speech file recast as llm: proxima prompt_tokens 120 at ttft 400 ms, incumbent prompt_tokens unreported
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE}"

jq -nc -f "$HERE/gen-whisper-metal.jq" > "$OUT/whisper_metal.jsonl"

jq -sc '
  def cold_proxima: .record == "run" and .arm == "proxima" and .repeat == 0 and (.discarded_warmup | not);
  ([.[] | select(cold_proxima) | .index] | sort) as $ranked
  | .[] | if cold_proxima then .index as $i | .timings.ttft_us = (($ranked | index($i)) + 1) else . end
' "$OUT/whisper_metal.jsonl" > "$OUT/percentile_1_20.jsonl"

jq -sc '
  def cold_proxima: .record == "run" and .arm == "proxima" and .repeat == 0 and (.discarded_warmup | not);
  ([.[] | select(cold_proxima) | .index] | sort | .[0:5]) as $keep
  | ([10, 11, 10, 12, 11] | map(. * 1000000)) as $ttlt
  | .[] | select(.record == "run" and .arm == "proxima")
  | select((.index | IN($keep[])) or ((.index - 1) | IN($keep[])))
  | if cold_proxima then .index as $i | .timings.ttlt_us = $ttlt[$keep | index($i)] else . end
' "$OUT/whisper_metal.jsonl" > "$OUT/cov_5.jsonl"

# llm_prefill.jsonl: the speech file recast as llm. every proxima run has prompt_tokens 120 and ttft 400000 us
# (prefill 300 tok/s); the incumbent never reports prompt_tokens.
jq -sc '
  .[] | if .record == "run" then
    .modality = "llm" | .model = "fixture-llm"
    | del(.timings.mel_us, .timings.encode_us, .input.duration_us)
    | if .arm == "proxima"
      then .prompt_tokens = 120 | .timings.ttft_us = 400000 | .timings.prefill_us = (400000 - .timings.load_us)
      else .prompt_tokens = {unreported: "incumbent-does-not-report"} | .timings.prefill_us = (.timings.ttft_us - .timings.load_us) end
  else . end
' "$OUT/whisper_metal.jsonl" > "$OUT/llm_prefill.jsonl"
