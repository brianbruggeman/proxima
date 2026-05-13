use proxima_macros::span;

// applying #[span] to a const fn must be a compile error
#[span]
const fn bad() -> u32 {
    42
}

fn main() {}
