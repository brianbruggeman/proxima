// a timeout that is not "<n>ms" / "<n>s" / "<n>m" must be a compile error.
#[proxima_macros::test(timeout = "soon")]
async fn bad_timeout() {}

fn main() {}
