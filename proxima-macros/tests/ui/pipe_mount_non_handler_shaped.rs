use proxima::MountTarget;
use proxima_macros::pipe;

// `#[pipe(send)]` only gets a `From<Struct> for MountTarget` impl when the
// fn is Handler-shaped (`Request<Bytes> -> Response<Bytes>`, `Err =
// ProximaError`). `increment`'s `In`/`Out`/`Err` don't match, so the macro
// never emits the impl, and `Into<MountTarget>` is simply unsatisfied here —
// a real, standard trait-bound error, not a silent miscompile.
#[pipe(send)]
async fn increment(input: u64) -> Result<u64, proxima::ProximaError> {
    Ok(input + 1)
}

fn main() {
    let _target: MountTarget = increment.into();
}
