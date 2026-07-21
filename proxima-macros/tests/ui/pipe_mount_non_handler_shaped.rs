use proxima::{App, ProximaError};
use proxima_macros::piped;

// `#[piped(send)]` on a non-Handler-shaped fn (`In`/`Out`/`Err` don't read as
// `Request<..>`/`Response<..>`/`ProximaError`) produces a struct that
// satisfies none of `App::mount`'s three `IntoMountTarget` arms: not
// `ViaPipe` (not `Handler`-shaped), not `ViaFn` (a unit struct, not an
// `Fn(Request<Bytes>) -> ..`), not `ViaName` (not a `&str`/`String`).
// `#[diagnostic::on_unimplemented]` on `IntoMountTarget` turns this into a
// real, helpful message — not a raw trait-bound dump.
#[piped(send)]
async fn increment(input: u64) -> Result<u64, ProximaError> {
    Ok(input + 1)
}

fn main() {
    let app = App::new().expect("app");
    app.mount("/", increment).expect("mount");
}
