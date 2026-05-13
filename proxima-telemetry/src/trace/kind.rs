#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub enum SpanKind {
    #[default]
    Internal,
    Server,
    Client,
    Producer,
    Consumer,
}
