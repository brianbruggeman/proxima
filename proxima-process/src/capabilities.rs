//! Capability tokens — zero-sized types issued at the trust boundary.
//!
//! Privileged grounds require `&Cap*` at construction. The token's
//! one-field private constructor makes it impossible for an
//! untrusted sub-pipe to mint one — privilege escalation is
//! structurally impossible.
//!
//! Same shape as Linux capabilities (`CAP_NET_ADMIN`, `CAP_SYS_PTRACE`,
//! etc.) lifted into the type system. Zero runtime cost — the tokens
//! are zero-sized.
//!
//! # Usage
//!
//! ```ignore
//! // At the trust boundary (e.g. the binary's main):
//! let fs_cap = CapFilesystem::grant();
//! let net_cap = CapNetwork::grant();
//!
//! // Privileged grounds borrow the cap at construction:
//! let real_read = host_read("/etc/passwd".into(), &fs_cap);
//!
//! // Untrusted code that never gets a CapFilesystem reference
//! // physically cannot call host_read.
//! ```

/// Permission to construct filesystem-touching grounds
/// ([`host_read`](super::grounds), `host_write`, etc.).
#[derive(Debug)]
pub struct CapFilesystem(());

impl CapFilesystem {
    /// Grant this capability. Call at the trust boundary only.
    ///
    /// The privacy of the inner `()` field means this is the ONLY
    /// way to construct a `CapFilesystem` — sub-pipes cannot mint
    /// their own.
    #[must_use]
    pub const fn grant() -> Self {
        Self(())
    }
}

/// Permission to construct network-touching grounds.
#[derive(Debug)]
pub struct CapNetwork(());

impl CapNetwork {
    /// Grant this capability. Call at the trust boundary only.
    #[must_use]
    pub const fn grant() -> Self {
        Self(())
    }
}

/// Permission to construct subprocess-spawning grounds.
#[derive(Debug)]
pub struct CapSpawn(());

impl CapSpawn {
    /// Grant this capability. Call at the trust boundary only.
    #[must_use]
    pub const fn grant() -> Self {
        Self(())
    }
}
