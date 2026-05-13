use proxima_macros::SpanCarrier;

// no span_id field and no #[span_id] attribute — must be a compile error
#[derive(SpanCarrier)]
struct Bad {
    payload: Vec<u8>,
}

fn main() {}
