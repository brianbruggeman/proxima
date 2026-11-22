# whisper_reference

The committed incumbent arm for the speech lane. It mirrors
`scripts/llama_reference`: a pinned external checkout, every path overridable,
real data only, and a summary that prints what actually ran. The invocation is
the spec's reference command (`proxima-model-interop/specs/whisper/SPEC.md`,
"Reference run"), not a second benchmark configuration.

## External checkout

`~/repos/others/whisper.cpp` at tag `v1.9.5` (`d1be6fde`). Not vendored and not
fetched by this script.

## Build

```sh
cd ~/repos/others/whisper.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON \
  -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=ON
cmake --build build --config Release -j
```

The summary's `metal=1` is `MTL : EMBED_LIBRARY = 1` from the run's own
`system_info` line; a CPU-only build reports `metal=0` and is still a valid
incumbent run, labelled as such.

## Model and audio

`ggml-tiny.bin` and `ggml-large-v3-turbo.bin` from
`https://huggingface.co/ggerganov/whisper.cpp` (the project's own repository),
default under `~/.cache/kvgate/proxnum610/ggml/`. Audio defaults to the
checkout's `samples/jfk.wav` (16 kHz mono).

## Invocation

```sh
./run.sh
```

`MODEL_PATH=.../ggml-large-v3-turbo.bin ./run.sh` runs the large model through
the same flags. The printed lines are `checkout`, `model`, `audio`, `metal`,
`transcript`, `mel_ms`, `encode_ms`, `decode_ms`, `log`.

## The f32 oracle

The published ggml file is ftype 1 (F16 weights). `to_f32_ggml.py` rewrites it
with F32 storage without changing a value, and
`proxima-model-interop/specs/whisper/oracle/whisper-dump.patch` adds the raw
dumps and the for-the-oracle-only F32 im2col/GELU path. Build the patched copy
out of tree:

```sh
cargo run --release -p proxima-model-interop --features std --example ggml_rewrite -- ggml-tiny.bin ggml-tiny-widened.bin
git -C ~/repos/others/whisper.cpp worktree add /tmp/whisper-dump v1.9.5
git -C /tmp/whisper-dump apply \
  "$OLDPWD/proxima-model-interop/specs/whisper/oracle/whisper-dump.patch"
cmake -B /tmp/whisper-dump/build -S /tmp/whisper-dump \
  -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DWHISPER_BUILD_TESTS=OFF
cmake --build /tmp/whisper-dump/build --config Release -j
WHISPER_DUMP_DIR=$HOME/.cache/kvgate/proxnum610/ref-f32 WHISPER_DUMP_F32=1 \
  /tmp/whisper-dump/build/bin/whisper-cli -ng -m ggml-tiny-widened.bin \
  -f ~/repos/others/whisper.cpp/samples/jfk.wav -l en -nt -bo 1 -bs 1 \
  -tp 0 -tpi 0 -nf -ojf -of $HOME/.cache/kvgate/proxnum610/ref-f32/tiny
```

The dump directory holds `mel.f32`, `conv.f32`, `enc.f32`, `cross_k_*.f32`,
`cross_v_*.f32`, `logits0*.f32`, `tokens.i32` and `ref.json`. The tests read it
through `PROXIMA_WHISPER_REF`.
