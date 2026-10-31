# lambda

Two guests behind one HTTP listener: `/elf` boots a bare no_std ELF, `/linux`
boots `/boot/vmlinuz-linux` with a static musl `/init`. linux/x86_64 with `/dev/kvm`.
Relative guest paths in `proxima.toml` resolve against the directory holding
the config file, so the server runs from any working directory.

build (from the repo root):

```
cargo build --release --target x86_64-unknown-none --manifest-path tools/proxima-vm/guests/lambda/Cargo.toml --bin uppercase
cargo build --release --target x86_64-unknown-linux-musl --manifest-path tools/proxima-vm/guests/linux-init/Cargo.toml --bin uppercase-init
cargo build -p proxima-cli --features vm
cp $CARGO_TARGET_DIR/x86_64-unknown-none/release/uppercase examples/lambda/
cp $CARGO_TARGET_DIR/x86_64-unknown-linux-musl/release/uppercase-init examples/lambda/
```

run:

```
$CARGO_TARGET_DIR/debug/proxima serve --config examples/lambda/proxima.toml
```

call:

```
curl -sS -w ' %{http_code} %{time_total}\n' --data 'hello, elf' http://127.0.0.1:8088/elf
curl -sS -w ' %{http_code} %{time_total}\n' --data 'hello, linux' http://127.0.0.1:8088/linux
```
