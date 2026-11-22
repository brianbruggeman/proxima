#!/usr/bin/env bash
# PROXNUM-588 AC8: three alternating sd.cpp/omega runs per side, median wall
# and per-side CoV. The 5% ratio bar and the owner's <= 84 s bar are printed,
# not enforced: the verdict belongs to the owner.
set -euo pipefail

K="${K:-$HOME/.cache/kvgate/proxnum588}"
M="${M:-$HOME/models/image_gen/flux2-klein-4b}"
SD="${SD:-$HOME/repos/others/stable-diffusion.cpp/build/bin/sd-cli}"
RUNS="${RUNS:-3}"
OUT="$K/ab"
mkdir -p "$OUT"

if [[ ! -x "$SD" ]]; then
  echo "missing unpatched sd-cli at $SD" >&2
  exit 1
fi

sd_once() {
  local run="$1"
  /usr/bin/time -l "$SD" -v \
    --diffusion-model "$M/flux-2-klein-4b.safetensors" \
    --llm "$M/comfy/qwen_3_4b.safetensors" \
    --vae "$M/comfy/flux2-vae.safetensors" --vae-format flux2 \
    -p "a red fox sitting in fresh snow, morning light, photograph" \
    -s 42 -W 1024 -H 1024 --steps 4 --cfg-scale 1.0 \
    -o "$OUT/sd-$run.png" >"$OUT/sd-$run.log" 2>"$OUT/sd-$run.time"
}

omega_once() {
  local run="$1"
  /usr/bin/time -l cargo run --release -q -p proxima-model-interop \
    --features std,metal --example flux2_klein -- \
    --seed 42 --out "$OUT/omega-$run.png" >"$OUT/omega-$run.log" 2>"$OUT/omega-$run.time"
}

for run in $(seq 1 "$RUNS"); do
  sd_once "$run"
  omega_once "$run"
done

field() {
  local file="$1" pattern="$2"
  awk -v pattern="$pattern" 'index($0, pattern) { print $1; exit }' "$file"
}

rss() {
  local file="$1"
  awk '/maximum resident set size/ { print $1; exit }' "$file"
}

median() {
  printf '%s\n' "$@" | sort -n | awk '{ values[NR] = $1 } END {
    if (NR % 2) { print values[(NR + 1) / 2] }
    else { print (values[NR / 2] + values[NR / 2 + 1]) / 2 }
  }'
}

cov() {
  printf '%s\n' "$@" | awk '{ sum += $1; sumsq += $1 * $1; n += 1 }
    END { mean = sum / n; var = sumsq / n - mean * mean; print sqrt(var) / mean * 100 }'
}

sd_walls=()
omega_walls=()
for run in $(seq 1 "$RUNS"); do
  sd_walls+=("$(field "$OUT/sd-$run.time" "real")")
  omega_walls+=("$(field "$OUT/omega-$run.time" "real")")
done

sd_median="$(median "${sd_walls[@]}")"
omega_median="$(median "${omega_walls[@]}")"

echo "runs=$RUNS per side"
printf 'sd.cpp wall s: %s\n' "${sd_walls[*]}"
printf 'omega  wall s: %s\n' "${omega_walls[*]}"
printf 'sd.cpp median_s=%s cov_pct=%s rss_bytes=%s\n' \
  "$sd_median" "$(cov "${sd_walls[@]}")" "$(rss "$OUT/sd-1.time")"
printf 'omega  median_s=%s cov_pct=%s rss_bytes=%s\n' \
  "$omega_median" "$(cov "${omega_walls[@]}")" "$(rss "$OUT/omega-1.time")"
awk -v omega="$omega_median" -v sd="$sd_median" 'BEGIN {
  printf "wall_ratio=%.4f bar_1.05=%s owner_bar_84s=%s\n", omega / sd,
    (omega / sd <= 1.05 ? "pass" : "miss"), (omega <= 84.0 ? "pass" : "miss")
}'
echo "logs: $OUT"
