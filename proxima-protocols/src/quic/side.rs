//! Connection role per RFC 9000 §1.2 ("client" / "server").
//!
//! Sans-IO state machines are role-asymmetric: the side that initiated
//! the connection (Client) drives the handshake forward; the responder
//! (Server) reacts. Crypto-secret naming, transport-parameter ID
//! validity, and stream-ID parity all branch on the role.

/// Connection role; lives for the lifetime of the connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Side {
    /// Originator of the connection (RFC 9000 §1.2).
    Client,
    /// Responder to the connection (RFC 9000 §1.2).
    Server,
}

impl Side {
    /// The opposite of `self`.
    #[must_use]
    pub const fn peer(self) -> Self {
        match self {
            Self::Client => Self::Server,
            Self::Server => Self::Client,
        }
    }

    /// Is this side the initiator (client)?
    #[must_use]
    pub const fn is_client(self) -> bool {
        matches!(self, Self::Client)
    }

    /// Is this side the responder (server)?
    #[must_use]
    pub const fn is_server(self) -> bool {
        matches!(self, Self::Server)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn peer_inverts_role() {
        assert_eq!(Side::Client.peer(), Side::Server);
        assert_eq!(Side::Server.peer(), Side::Client);
    }

    #[test]
    fn is_client_and_is_server_are_mutually_exclusive() {
        assert!(Side::Client.is_client() && !Side::Client.is_server());
        assert!(Side::Server.is_server() && !Side::Server.is_client());
    }
}
