#[derive(Clone, PartialEq, Debug, Default)]
pub enum Status {
    #[default]
    Unset,
    Ok,
    Error {
        reason: &'static str,
    },
}
