/// Caller-supplied thread identity for a ring lane.
///
/// The caller constructs a `LaneHandle` once per logical producer thread and
/// passes it through to components that need to know which lane to address.
/// No internal TLS is used — the identity is explicit and owned by the caller.
///
/// The handle carries the lane index only; it does not borrow the ring, so it
/// is cheap to clone and pass through call chains without lifetime coupling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LaneHandle {
    index: usize,
}

impl LaneHandle {
    /// Construct a handle for the given lane index.
    ///
    /// Typically `index` is the logical CPU or thread ordinal assigned by the
    /// runtime. The caller is responsible for ensuring the index stays within
    /// the bounds expected by the ring.
    #[must_use]
    pub fn new(index: usize) -> Self {
        Self { index }
    }

    /// The lane index carried by this handle.
    #[must_use]
    pub fn index(self) -> usize {
        self.index
    }
}
