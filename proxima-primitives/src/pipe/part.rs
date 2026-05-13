//! `Part` — the borrowed, discriminated-union primitive for one piece of an
//! HTTP message: method, path, one header, one body chunk, or end. See
//! `docs/proxima-pipe/part-source-sink-design.md` for the locked design this
//! module implements (step 1 of the migration described there).
//!
//! A message is a PROCESS (parts arriving over time, borrowed from a reused
//! buffer), not a VALUE. `Part` never fuses method + path + headers into one
//! owned aggregate; a [`PartSource`] holds exactly one borrowed `Part` at a
//! time. Materializing the aggregate (a [`crate::pipe::request::Request`]) is an
//! opt-in drain (`Request::from_source`, gated `alloc` + this feature), never
//! the default.
//!
//! # Tier
//!
//! `core` only — no `alloc`, no heap, no I/O. `Part` borrows; the traits
//! below don't allocate. This is the tier-3 RISC primitive other proxima
//! leaf crates (h1/h2/h3 codecs) implement against directly.

/// One borrowed piece of an HTTP message. `#[non_exhaustive]` leaves room for
/// future kinds (e.g. trailers) without a breaking change. Borrows only —
/// owns nothing, so it never outlives the buffer it was decoded from.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Part<'a> {
    /// The request method's wire bytes, e.g. `b"GET"`.
    Method(&'a [u8]),
    /// The request path's wire bytes, e.g. `b"/v1/items"`.
    Path(&'a [u8]),
    /// One header name/value pair.
    Header(&'a [u8], &'a [u8]),
    /// One chunk of body bytes.
    Chunk(&'a [u8]),
    /// End of message — no further parts follow.
    End,
}

/// The `() → Part` degenerate [`Pipe`](crate::pipe::primitives::Pipe): a source that
/// steps one borrowed [`Part`] at a time instead of handing back a fused
/// aggregate. A lending step — the borrow returned by `next` is tied to
/// `&mut self`, so no GAT is needed on the `Pipe` trait itself; the lending
/// lives entirely inside this leaf trait, exactly like the existing
/// `proxima-h3-proto` crate's `qpack::decoder::FieldSink` visitor.
pub trait PartSource {
    /// Step to the next part, or `None` once the source is exhausted.
    /// Implementations that model a definite end SHOULD yield exactly one
    /// [`Part::End`] before returning `None`.
    fn next(&mut self) -> Option<Part<'_>>;
}

/// The `Part → ()` degenerate [`Pipe`](crate::pipe::primitives::Pipe): a sink that
/// absorbs one borrowed [`Part`] at a time.
pub trait PartSink {
    /// Error a sink may return, e.g. a fixed-capacity buffer is full or the
    /// parts arrived out of the sink's expected sequence.
    type Err;

    /// Absorb one part.
    ///
    /// # Errors
    ///
    /// Implementations MAY reject a part; the error propagates to the
    /// caller and is never panicked.
    fn push(&mut self, part: Part<'_>) -> Result<(), Self::Err>;
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{Part, PartSink, PartSource};

    struct SliceSource<'a> {
        parts: &'a [Part<'a>],
        cursor: usize,
    }

    impl PartSource for SliceSource<'_> {
        fn next(&mut self) -> Option<Part<'_>> {
            let part = *self.parts.get(self.cursor)?;
            self.cursor += 1;
            Some(part)
        }
    }

    #[derive(Default)]
    struct CountingSink {
        headers_seen: usize,
        ended: bool,
    }

    impl PartSink for CountingSink {
        type Err = &'static str;

        fn push(&mut self, part: Part<'_>) -> Result<(), Self::Err> {
            match part {
                Part::Header(_, _) => {
                    self.headers_seen += 1;
                    Ok(())
                }
                Part::End => {
                    self.ended = true;
                    Ok(())
                }
                Part::Method(_) | Part::Path(_) | Part::Chunk(_) => Ok(()),
            }
        }
    }

    #[test]
    fn part_source_steps_one_borrowed_part_at_a_time() {
        let parts = [
            Part::Method(b"GET"),
            Part::Path(b"/health"),
            Part::Header(b"accept", b"*/*"),
            Part::End,
        ];
        let mut source = SliceSource {
            parts: &parts,
            cursor: 0,
        };

        assert_eq!(source.next(), Some(Part::Method(b"GET")));
        assert_eq!(source.next(), Some(Part::Path(b"/health")));
        assert_eq!(source.next(), Some(Part::Header(b"accept", b"*/*")));
        assert_eq!(source.next(), Some(Part::End));
        assert_eq!(source.next(), None);
    }

    #[test]
    fn part_sink_absorbs_parts_and_reports_counts() {
        let parts = [
            Part::Method(b"POST"),
            Part::Header(b"content-type", b"application/json"),
            Part::Header(b"x-request-id", b"abc123"),
            Part::End,
        ];
        let mut source = SliceSource {
            parts: &parts,
            cursor: 0,
        };
        let mut sink = CountingSink::default();

        while let Some(part) = source.next() {
            sink.push(part).expect("counting sink never rejects");
        }

        assert_eq!(sink.headers_seen, 2);
        assert!(sink.ended);
    }

    #[test]
    fn part_sink_can_reject() {
        struct FullSink;
        impl PartSink for FullSink {
            type Err = &'static str;
            fn push(&mut self, _part: Part<'_>) -> Result<(), Self::Err> {
                Err("capacity exceeded")
            }
        }

        let mut sink = FullSink;
        let error = sink
            .push(Part::Header(b"a", b"b"))
            .expect_err("full sink rejects");
        assert_eq!(error, "capacity exceeded");
    }
}
