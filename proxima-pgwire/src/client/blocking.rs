//! Blocking driver over the sans-IO [`ClientSession`] — a `std::io::Read +
//! Write` transport (e.g. `std::net::TcpStream`). Used by the real-PostgreSQL
//! parity harness (capture + differential); the async Pipe driver wraps the
//! same session over a futures-io transport. The session owns the protocol;
//! this only moves bytes.

use std::io::{Read, Write};

use crate::client::session::{ClientError, ClientSession, QueryResult, Step};

pub struct PgClient<S> {
    stream: S,
    session: ClientSession,
    capture: bool,
    /// Every backend byte read, when capture is enabled — the verbatim
    /// server->client stream for fixture vendoring.
    pub captured: Vec<u8>,
}

impl<S: Read + Write> PgClient<S> {
    /// Startup + authentication over `stream`, returning a ready client.
    ///
    /// # Errors
    /// [`ClientError`] on I/O, a server `ErrorResponse`, an unsupported auth
    /// method, or a SCRAM failure.
    pub fn connect(
        stream: S,
        user: &str,
        password: &str,
        database: &str,
    ) -> Result<Self, ClientError> {
        Self::connect_inner(stream, user, password, database, false)
    }

    /// Like [`Self::connect`] but tees every backend byte into `captured`.
    ///
    /// # Errors
    /// See [`Self::connect`].
    pub fn connect_capturing(
        stream: S,
        user: &str,
        password: &str,
        database: &str,
    ) -> Result<Self, ClientError> {
        Self::connect_inner(stream, user, password, database, true)
    }

    fn connect_inner(
        stream: S,
        user: &str,
        password: &str,
        database: &str,
        capture: bool,
    ) -> Result<Self, ClientError> {
        let session = ClientSession::new(user, password, database)?;
        let mut client = Self {
            stream,
            session,
            capture,
            captured: Vec::new(),
        };
        client.drive_until_ready()?;
        Ok(client)
    }

    /// Runs a simple-protocol query and collects the reply.
    ///
    /// # Errors
    /// [`ClientError`] on I/O or a server `ErrorResponse` (the session recovers
    /// to ready so the client stays usable).
    pub fn simple_query(&mut self, sql: &str) -> Result<QueryResult, ClientError> {
        self.session.submit_simple(sql)?;
        self.run_query()
    }

    /// Runs an extended-protocol query (text parameters) and collects the reply.
    ///
    /// # Errors
    /// [`ClientError`] on I/O or a server `ErrorResponse`.
    pub fn extended_query(
        &mut self,
        sql: &str,
        params: &[&str],
    ) -> Result<QueryResult, ClientError> {
        self.session.submit_extended(sql, params)?;
        self.run_query()
    }

    /// Sends Terminate and drops the connection.
    ///
    /// # Errors
    /// [`ClientError::Io`] if the final write fails.
    pub fn close(mut self) -> Result<(), ClientError> {
        self.session.submit_terminate()?;
        self.flush_outbound()
    }

    fn drive_until_ready(&mut self) -> Result<(), ClientError> {
        loop {
            match self.session.advance()? {
                Step::Send => self.flush_outbound()?,
                Step::Recv => self.read_into_session()?,
                Step::Ready => return Ok(()),
                Step::Complete(_) => return Err(ClientError::Protocol("query reply before ready")),
            }
        }
    }

    fn run_query(&mut self) -> Result<QueryResult, ClientError> {
        loop {
            match self.session.advance()? {
                Step::Send => self.flush_outbound()?,
                Step::Recv => self.read_into_session()?,
                Step::Complete(result) => return Ok(result),
                Step::Ready => return Err(ClientError::Protocol("ready without a reply")),
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
        if self.capture {
            self.captured.extend_from_slice(&chunk[..read]);
        }
        self.session.feed(&chunk[..read]);
        Ok(())
    }
}
