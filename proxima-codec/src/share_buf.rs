//! [`ShareBuf`] — the buffer-genericity seam for
//! `proxima_protocols::codec_pipe::OwnFrame::Source` (component C4).
//!
//! `bytes::Buf` is the wrong shape for this seam: it is a CURSOR
//! (`advance`/`remaining`/`chunk`), built to walk forward through a
//! sequence of chunks, not to hand back a cheap, independently-owned
//! sub-slice of the SAME backing allocation the codec parsed from — which
//! is exactly what [`OwnFrame::own_frame`] needs (re-owning a borrowed
//! frame past the `Pipe::call` boundary without copying bytes). `share`
//! names that one operation directly instead of asking every future
//! `Source` (an `Arc<[u8]>`-backed window, a DPDK `rte_mbuf`, ...) to also
//! become a chunk-walking cursor it will never otherwise be driven as.
//!
//! No `Send + Sync + 'static` bound here, deliberately: a DPDK `rte_mbuf`
//! is core-pinned (`!Send` by construction — moving it across cores means
//! moving the NIC descriptor ring ownership, which the hardware does not
//! allow), so baking `Send` into `ShareBuf` would make `type Source =
//! Mbuf` uncompilable and defeat the whole point of a kernel-bypass tier.
//! The cross-core bound belongs at the site that actually crosses cores —
//! `proxima_protocols::codec_pipe`'s `SendPipe` impl for `FrameCodecPipe`
//! already adds `C::Source: Send + Sync` there, not here.
use core::ops::Deref;

/// A buffer that can hand back a cheap, same-allocation sub-slice of
/// itself — the operation [`proxima_protocols::codec_pipe::OwnFrame`]
/// needs to re-own a borrowed frame past a `Pipe::call` boundary without
/// copying bytes (see the module doc for why this is `share`, not
/// `bytes::Buf`).
pub trait ShareBuf: Deref<Target = [u8]> + Clone {
    /// Returns a new `Self` covering exactly `subset`.
    ///
    /// `subset` MUST be a pointer-derived sub-slice of `self` (same
    /// backing allocation) — impls may panic otherwise, mirroring
    /// `Bytes::slice_ref`'s own contract.
    fn share(&self, subset: &[u8]) -> Self;
}

impl ShareBuf for bytes::Bytes {
    fn share(&self, subset: &[u8]) -> Self {
        self.slice_ref(subset)
    }
}
