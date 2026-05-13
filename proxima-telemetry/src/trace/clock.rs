// re-export the shared clock primitives so c5-trace callers see the same types
// as c8-log callers; both use crate::clock as the canonical location.
pub use crate::clock::{Clock, MonotonicCounter};
