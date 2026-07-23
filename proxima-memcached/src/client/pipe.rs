//! Async memcached client `Pipe` over any [`StreamUpstream`] — the same
//! transport seam `proxima_redis::client::pipe::RedisClientUpstream` uses,
//! so the client is agnostic to the wire (prime, tokio, TLS-wrapped). It
//! drives the sans-IO [`ClientSession`] over a futures-io connection and
//! maps the byte-level `Request<Bytes>`/`Response<Bytes>` contract
//! `proxima::Client` speaks to memcached wire commands.
//!
//! Simpler than redis's client: there is no streaming/pub-sub branch (no
//! memcached command switches the connection into a pushed stream), so
//! `exchange` is always one request/reply round trip over the cached
//! connection.
//!
//! Wire convention for `Request<Bytes>`: `Request.method` is the verb
//! (`GET`/`SET`/`DELETE`/...), `Request.payload` (the body) is a
//! NUL-delimited positional-field encoding specific to that verb's shape
//! (mirrors `RedisClientUpstream`'s own NUL-delimited `argv` convention,
//! adapted to memcached's heterogeneous per-verb fields rather than RESP's
//! uniform arg list):
//!
//! | verb                              | body                                          |
//! |------------------------------------|-----------------------------------------------|
//! | `GET` / `GETS`                     | key\[NUL key ...\] (1+ keys)                   |
//! | `SET`/`ADD`/`REPLACE`/`APPEND`/`PREPEND` | key NUL flags NUL exptime NUL value      |
//! | `CAS`                              | key NUL flags NUL exptime NUL cas_unique NUL value |
//! | `DELETE`                           | key                                            |
//! | `INCR` / `DECR`                    | key NUL delta                                  |
//! | `TOUCH`                            | key NUL exptime                                |
//! | `FLUSH_ALL`                        | "" or delay                                    |
//! | `STATS`                            | args (opaque, forwarded raw)                   |
//! | `VERSION` / `QUIT`                 | ignored                                        |

use std::future::Future;
use std::sync::Arc;

use bytes::Bytes;
use futures::io::{AsyncReadExt, AsyncWriteExt};
use futures::lock::Mutex;

use proxima_core::ProximaError;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::request::{Request, Response};
use proxima_primitives::stream::{StreamConnection, StreamUpstream, StreamUpstreamExt};

use proxima_protocols::memcached::{MemcachedRequest, StoreMode, encode_reply};

use crate::client::config::MemcachedClientConfig;
use crate::client::session::{ClientError, ClientSession, Step};

const READ_CHUNK_BYTES: usize = 16 * 1024;

/// memcached client `Pipe` over a `StreamUpstream`. One client owns one
/// upstream binding (host:port) and one cached connection (pool of one)
/// reused across request/reply calls.
pub struct MemcachedClientUpstream<U: StreamUpstream> {
    upstream: Arc<U>,
    cached: Arc<Mutex<Option<U::Conn>>>,
}

impl<U: StreamUpstream> MemcachedClientUpstream<U> {
    /// Builds a client over `upstream`. `config` is accepted for
    /// constructor parity with `RedisClientUpstream::new` (and to carry a
    /// future SASL-auth extension); the base protocol has no handshake, so
    /// it is not otherwise consulted here — only `upstream.connect()`'s
    /// `host`/`port` (already baked into `upstream`) matter.
    ///
    /// `new` never touches the network — `upstream.connect()` only runs
    /// lazily on the first `.call()`, so building one is cheap and
    /// side-effect-free (this doctest never sends a command, and needs no
    /// running memcached server):
    ///
    /// ```
    /// use proxima_memcached::{MemcachedClientConfig, MemcachedClientUpstream};
    /// use proxima_net::prime::PrimeTcpUpstream;
    ///
    /// let addr = "127.0.0.1:11211".parse().expect("valid socket address");
    /// let transport = PrimeTcpUpstream::new(addr);
    /// let client = MemcachedClientUpstream::new(transport, MemcachedClientConfig::default());
    /// # let _ = client;
    /// ```
    pub fn new(upstream: U, _config: MemcachedClientConfig) -> Self {
        Self {
            upstream: Arc::new(upstream),
            cached: Arc::new(Mutex::new(None)),
        }
    }

    async fn exchange(&self, request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
        let verb = String::from_utf8_lossy(request.method.as_bytes()).into_owned();
        let (_request, body) = request.body_bytes().await?;
        let memcached_request = decode_request(&verb, &body).map_err(ProximaError::Config)?;
        self.request_reply(memcached_request).await
    }

    async fn request_reply(
        &self,
        request: MemcachedRequest,
    ) -> Result<Response<Bytes>, ProximaError> {
        let mut guard = self.cached.lock().await;
        if guard.is_none() {
            *guard = Some(self.connect().await?);
        }
        let conn = guard
            .as_mut()
            .ok_or_else(|| ProximaError::Upstream("memcached cache empty".into()))?;

        match run_command(conn, &request).await {
            Ok(reply) => {
                let mut bytes = Vec::new();
                encode_reply(&reply, &mut bytes);
                Ok(Response::ok(Bytes::from(bytes)))
            }
            Err(error) => {
                *guard = None;
                Err(client_error_to_proxima(error))
            }
        }
    }

    async fn connect(&self) -> Result<U::Conn, ProximaError> {
        self.upstream
            .connect()
            .await
            .map_err(|err| ProximaError::Upstream(format!("memcached connect: {err}")))
    }
}

impl<U: StreamUpstream> SendPipe for MemcachedClientUpstream<U> {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move { self.exchange(request).await }
    }
}

async fn run_command<C: StreamConnection>(
    conn: &mut C,
    request: &MemcachedRequest,
) -> Result<proxima_protocols::memcached::Reply, ClientError> {
    let mut session = ClientSession::new();
    session.submit(request)?;
    loop {
        match session.advance()? {
            Step::Send => flush(&mut session, conn).await?,
            Step::Recv => recv(&mut session, conn).await?,
            Step::Complete(reply) => return Ok(reply),
            Step::Ready => {
                // a `noreply`-flagged command has no reply to wait for;
                // the client contract still hands the caller *something*
                // back, so a bare acknowledgement stands in for "sent".
                return Ok(proxima_protocols::memcached::Reply::Ok);
            }
        }
    }
}

async fn flush<C: StreamConnection>(
    session: &mut ClientSession,
    conn: &mut C,
) -> Result<(), ClientError> {
    let bytes = session.take_outbound();
    conn.write_all(&bytes).await?;
    conn.flush().await?;
    Ok(())
}

async fn recv<C: StreamConnection>(
    session: &mut ClientSession,
    conn: &mut C,
) -> Result<(), ClientError> {
    let mut chunk = [0_u8; READ_CHUNK_BYTES];
    let read = conn.read(&mut chunk).await?;
    if read == 0 {
        return Err(ClientError::Closed);
    }
    session.feed(&chunk[..read]);
    Ok(())
}

fn client_error_to_proxima(error: ClientError) -> ProximaError {
    ProximaError::Upstream(format!("memcached client: {error}"))
}

/// `[verb] ++ NUL-split(body)` -> [`MemcachedRequest`], per this module's
/// documented wire convention.
fn decode_request(verb: &str, body: &[u8]) -> Result<MemcachedRequest, String> {
    let mut fields = body.split(|&byte| byte == 0);
    let upper = verb.to_ascii_uppercase();
    match upper.as_str() {
        "GET" | "GETS" => {
            let keys: Vec<Vec<u8>> = if body.is_empty() {
                Vec::new()
            } else {
                fields.map(<[u8]>::to_vec).collect()
            };
            if keys.is_empty() {
                return Err("GET/GETS requires at least one key".to_string());
            }
            Ok(MemcachedRequest::Get {
                keys,
                gets: upper == "GETS",
            })
        }
        "SET" | "ADD" | "REPLACE" | "APPEND" | "PREPEND" => {
            let key = next_field(&mut fields, "key")?;
            let flags = next_u32(&mut fields, "flags")?;
            let exptime = next_u32(&mut fields, "exptime")?;
            let value = next_field(&mut fields, "value")?;
            let mode = match upper.as_str() {
                "SET" => StoreMode::Set,
                "ADD" => StoreMode::Add,
                "REPLACE" => StoreMode::Replace,
                "APPEND" => StoreMode::Append,
                _ => StoreMode::Prepend,
            };
            Ok(MemcachedRequest::Store {
                mode,
                key,
                flags,
                exptime,
                value,
                noreply: false,
            })
        }
        "CAS" => {
            let key = next_field(&mut fields, "key")?;
            let flags = next_u32(&mut fields, "flags")?;
            let exptime = next_u32(&mut fields, "exptime")?;
            let cas_unique = next_u64(&mut fields, "cas_unique")?;
            let value = next_field(&mut fields, "value")?;
            Ok(MemcachedRequest::Cas {
                key,
                flags,
                exptime,
                cas_unique,
                value,
                noreply: false,
            })
        }
        "DELETE" => Ok(MemcachedRequest::Delete {
            key: next_field(&mut fields, "key")?,
            noreply: false,
        }),
        "INCR" | "DECR" => {
            let key = next_field(&mut fields, "key")?;
            let delta = next_u64(&mut fields, "delta")?;
            Ok(MemcachedRequest::Counter {
                increment: upper == "INCR",
                key,
                delta,
                noreply: false,
            })
        }
        "TOUCH" => {
            let key = next_field(&mut fields, "key")?;
            let exptime = next_u32(&mut fields, "exptime")?;
            Ok(MemcachedRequest::Touch {
                key,
                exptime,
                noreply: false,
            })
        }
        "FLUSH_ALL" => {
            let delay = match fields.next() {
                Some(field) if !field.is_empty() => Some(
                    core::str::from_utf8(field)
                        .ok()
                        .and_then(|text| text.parse::<u32>().ok())
                        .ok_or_else(|| "flush_all delay must be a number".to_string())?,
                ),
                _ => None,
            };
            Ok(MemcachedRequest::FlushAll {
                delay,
                noreply: false,
            })
        }
        "STATS" => Ok(MemcachedRequest::Stats {
            args: body.to_vec(),
        }),
        "VERSION" => Ok(MemcachedRequest::Version),
        "QUIT" => Ok(MemcachedRequest::Quit),
        other => Err(format!("unknown memcached verb '{other}'")),
    }
}

fn next_field<'a>(
    fields: &mut impl Iterator<Item = &'a [u8]>,
    name: &str,
) -> Result<Vec<u8>, String> {
    fields
        .next()
        .map(<[u8]>::to_vec)
        .ok_or_else(|| format!("missing field '{name}'"))
}

fn next_u32<'a>(fields: &mut impl Iterator<Item = &'a [u8]>, name: &str) -> Result<u32, String> {
    let field = next_field(fields, name)?;
    core::str::from_utf8(&field)
        .ok()
        .and_then(|text| text.parse::<u32>().ok())
        .ok_or_else(|| format!("field '{name}' must be a number"))
}

fn next_u64<'a>(fields: &mut impl Iterator<Item = &'a [u8]>, name: &str) -> Result<u64, String> {
    let field = next_field(fields, name)?;
    core::str::from_utf8(&field)
        .ok()
        .and_then(|text| text.parse::<u64>().ok())
        .ok_or_else(|| format!("field '{name}' must be a number"))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn decode_request_parses_get_with_one_key() {
        let request = decode_request("GET", b"mykey").expect("decode");
        assert_eq!(
            request,
            MemcachedRequest::Get {
                keys: vec![b"mykey".to_vec()],
                gets: false,
            }
        );
    }

    #[test]
    fn decode_request_parses_multi_get() {
        let mut body = b"a".to_vec();
        body.push(0);
        body.extend_from_slice(b"b");
        let request = decode_request("GET", &body).expect("decode");
        assert_eq!(
            request,
            MemcachedRequest::Get {
                keys: vec![b"a".to_vec(), b"b".to_vec()],
                gets: false,
            }
        );
    }

    #[test]
    fn decode_request_parses_set() {
        let body = b"k\x000\x0060\x00hello";
        let request = decode_request("SET", body).expect("decode");
        assert_eq!(
            request,
            MemcachedRequest::Store {
                mode: StoreMode::Set,
                key: b"k".to_vec(),
                flags: 0,
                exptime: 60,
                value: b"hello".to_vec(),
                noreply: false,
            }
        );
    }

    #[test]
    fn decode_request_rejects_unknown_verb() {
        assert!(decode_request("BOGUS", b"").is_err());
    }

    #[test]
    fn decode_request_rejects_get_with_no_keys() {
        assert!(decode_request("GET", b"").is_err());
    }
}
