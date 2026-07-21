use proxima_macros::filter;

// filter! requires a decision shape `In -> Result<In, Err>`: the closure's
// admit type must equal its input type. This closure admits a `bool`, not
// the `u64` it received, so it must be a compile error.
fn main() {
    let _bad = filter!(|input: u64| -> Result<bool, &'static str> { Ok(input > 0) });
}
