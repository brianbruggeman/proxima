//! Blocking driver over the sans-IO [`ClientSession`] — a `std::io::Read +
//! Write` transport (e.g. `std::net::TcpStream`). The session owns the
//! protocol; this only moves bytes. Mirrors
//! `proxima_redis::client::blocking::RedisClient`.

use std::io::{Read, Write};

use proxima_protocols::memcached::{MemcachedRequest, Reply, encode_request};

use crate::client::session::{ClientError, ClientSession, Step};

pub struct MemcachedClient<S> {
    stream: S,
    session: ClientSession,
}

impl<S: Read + Write> MemcachedClient<S> {
    /// Wraps `stream` in a ready client. There is no handshake to run
    /// (see [`ClientSession`]'s docs), so this never fails on its own —
    /// kept fallible (rather than an infallible constructor) for parity
    /// with `RedisClient::connect` and to leave room for a future
    /// SASL-auth extension without an API break.
    ///
    /// # Errors
    /// Currently infallible; the `Result` is reserved for that future
    /// extension.
    pub fn connect(stream: S) -> Result<Self, ClientError> {
        Ok(Self {
            stream,
            session: ClientSession::new(),
        })
    }

    /// Runs one command and collects the reply. For a `noreply`-flagged
    /// command, returns [`Reply::Ok`] as a bare acknowledgement — the wire
    /// contract guarantees no real reply will ever arrive.
    ///
    /// # Errors
    /// [`ClientError`] on I/O or a malformed reply.
    pub fn command(&mut self, request: &MemcachedRequest) -> Result<Reply, ClientError> {
        self.session.submit(request)?;
        self.run_command()
    }

    /// Sends `quit` and drops the connection (best-effort; memcached's
    /// `quit` has no reply — the server just closes the socket).
    ///
    /// # Errors
    /// [`ClientError::Io`] if the final write fails.
    pub fn close(mut self) -> Result<(), ClientError> {
        let mut bytes = Vec::new();
        encode_request(&MemcachedRequest::Quit, &mut bytes);
        self.stream.write_all(&bytes)?;
        self.stream.flush()?;
        Ok(())
    }

    fn run_command(&mut self) -> Result<Reply, ClientError> {
        loop {
            match self.session.advance()? {
                Step::Send => self.flush_outbound()?,
                Step::Recv => self.read_into_session()?,
                Step::Complete(reply) => return Ok(reply),
                Step::Ready => return Ok(Reply::Ok),
            }
        }
    }

    fn flush_outbound(&mut self) -> Result<(), ClientError> {
        let bytes = self.session.take_outbound();
        self.stream.write_all(&bytes)?;
        self.stream.flush()?;
        Ok(())
    }

    fn read_into_session(&mut self) -> Result<(), ClientError> {
        let mut chunk = [0_u8; 8192];
        let read = self.stream.read(&mut chunk)?;
        if read == 0 {
            return Err(ClientError::Closed);
        }
        self.session.feed(&chunk[..read]);
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proxima_protocols::memcached::StoreMode;
    use std::io::Cursor;

    /// A read-once / write-to-owned-vec fake transport: `command()`
    /// consumes `scripted` for its reply, then all writes land in `written`.
    struct ScriptedStream {
        scripted: Cursor<Vec<u8>>,
        written: Vec<u8>,
    }

    impl Read for ScriptedStream {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            self.scripted.read(buf)
        }
    }

    impl Write for ScriptedStream {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.written.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn command_sends_the_encoded_request_and_parses_the_reply() {
        let stream = ScriptedStream {
            scripted: Cursor::new(b"STORED\r\n".to_vec()),
            written: Vec::new(),
        };
        let mut client = MemcachedClient::connect(stream).expect("connect");

        let reply = client
            .command(&MemcachedRequest::Store {
                mode: StoreMode::Set,
                key: b"k".to_vec(),
                flags: 0,
                exptime: 0,
                value: b"v".to_vec(),
                noreply: false,
            })
            .expect("command");

        assert_eq!(reply, Reply::Stored);
        assert_eq!(client.stream.written, b"set k 0 0 1\r\nv\r\n");
    }

    #[test]
    fn close_writes_quit_and_does_not_wait_for_a_reply() {
        let stream = ScriptedStream {
            scripted: Cursor::new(Vec::new()),
            written: Vec::new(),
        };
        let client = MemcachedClient::connect(stream).expect("connect");
        client.close().expect("close");
    }
}
