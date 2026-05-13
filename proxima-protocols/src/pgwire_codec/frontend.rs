//! Frontend (client → server) message decode and encode.
//!
//! Message shapes follow the PostgreSQL protocol 3.x message formats
//! reference (`https://www.postgresql.org/docs/current/protocol-message-formats.html`).
//! Decode produces borrowed views over the input buffer; encode writes
//! into caller-owned storage via [`MessageWriter`]. Both directions are
//! implemented so the codec is usable for servers, clients, proxies, and
//! round-trip tests.
//!
//! Startup-phase messages (StartupMessage, SSLRequest, GSSENCRequest,
//! CancelRequest) are untagged and parsed by [`parse_initial`]; all other
//! traffic is tagged and parsed by [`parse_frontend`]. The session FSM
//! ([`super::session::Session::wire_phase`]) tells the caller which one
//! applies.

use super::cursor::Reader;
use super::error::{EncodeError, ParseError};
use super::frame::split_tagged;
use super::types::{FormatCode, Oid, PgStr, ProtocolVersion, StatementTarget};
use super::views::{FormatCodes, Oids, StartupParams, Values};
use super::writer::MessageWriter;

/// SSLRequest magic, chosen to collide with no protocol version
/// (1234 << 16 | 5679).
pub const SSL_REQUEST_CODE: i32 = 80877103;
/// GSSENCRequest magic (1234 << 16 | 5680).
pub const GSSENC_REQUEST_CODE: i32 = 80877104;
/// CancelRequest magic (1234 << 16 | 5678).
pub const CANCEL_REQUEST_CODE: i32 = 80877102;

/// Untagged startup-phase message.
#[derive(Debug, Clone, Copy)]
pub enum InitialMessage<'a> {
    Startup(Startup<'a>),
    SslRequest,
    GssEncRequest,
    Cancel(CancelRequest<'a>),
}

/// StartupMessage: protocol version plus parameter pairs (`user` is
/// required by the server; `database`, `options`, and `_pq_.*` extras
/// are optional).
#[derive(Debug, Clone, Copy)]
pub struct Startup<'a> {
    pub version: ProtocolVersion,
    pub parameters: StartupParams<'a>,
}

impl<'a> Startup<'a> {
    #[must_use]
    pub fn user(&self) -> Option<PgStr<'a>> {
        self.parameters.get(b"user")
    }

    #[must_use]
    pub fn database(&self) -> Option<PgStr<'a>> {
        self.parameters.get(b"database")
    }
}

/// CancelRequest: targets another connection by its BackendKeyData.
///
/// Protocol 3.0 fixes the key at 4 bytes; 3.2 (PostgreSQL 17+) allows
/// 4–256 bytes, so the key is exposed as a slice.
#[derive(Debug, Clone, Copy)]
pub struct CancelRequest<'a> {
    pub process_id: i32,
    pub secret_key: &'a [u8],
}

/// Parses one untagged startup-phase message.
///
/// Returns `Ok(None)` when `input` does not yet hold a complete message.
///
/// # Errors
/// [`ParseError`] when the frame or body is structurally invalid, the
/// request code is unknown, or the protocol major version is not 3.
pub fn parse_initial(input: &[u8]) -> Result<Option<(InitialMessage<'_>, usize)>, ParseError> {
    if input.len() < 8 {
        return Ok(None);
    }
    let length = i32::from_be_bytes([input[0], input[1], input[2], input[3]]);
    let Ok(total) = usize::try_from(length) else {
        return Err(ParseError::BadLength { tag: 0, length });
    };
    if total < 8 {
        return Err(ParseError::BadLength { tag: 0, length });
    }
    if input.len() < total {
        return Ok(None);
    }
    let code = i32::from_be_bytes([input[4], input[5], input[6], input[7]]);
    let mut reader = Reader::new(&input[8..total], 0);
    let message = match code {
        SSL_REQUEST_CODE => {
            reader.expect_end()?;
            InitialMessage::SslRequest
        }
        GSSENC_REQUEST_CODE => {
            reader.expect_end()?;
            InitialMessage::GssEncRequest
        }
        CANCEL_REQUEST_CODE => {
            let process_id = reader.take_i32()?;
            let secret_key = reader.take_rest();
            if secret_key.is_empty() || secret_key.len() > 256 {
                return Err(ParseError::InvalidValue {
                    tag: 0,
                    field: "cancel secret key",
                });
            }
            InitialMessage::Cancel(CancelRequest {
                process_id,
                secret_key,
            })
        }
        _ => {
            let version = ProtocolVersion::from_code(code);
            if version.major != 3 {
                if version.major == 1234 {
                    return Err(ParseError::UnknownRequestCode { code });
                }
                return Err(ParseError::UnsupportedProtocol {
                    major: version.major,
                    minor: version.minor,
                });
            }
            let parameters = StartupParams::validate(&mut reader)?;
            reader.expect_end()?;
            InitialMessage::Startup(Startup {
                version,
                parameters,
            })
        }
    };
    Ok(Some((message, total)))
}

/// Bind: binds parameter values to a prepared statement, creating a
/// portal.
#[derive(Debug, Clone, Copy)]
pub struct Bind<'a> {
    pub portal: PgStr<'a>,
    pub statement: PgStr<'a>,
    pub parameter_formats: FormatCodes<'a>,
    pub parameters: Values<'a>,
    pub result_formats: FormatCodes<'a>,
}

impl Bind<'_> {
    /// Effective format of parameter `index` under the Bind rules
    /// (empty list = all text, single code = applies to all).
    #[must_use]
    pub fn parameter_format(&self, index: usize) -> FormatCode {
        self.parameter_formats.resolve(index)
    }

    /// Effective format of result column `index`.
    #[must_use]
    pub fn result_format(&self, index: usize) -> FormatCode {
        self.result_formats.resolve(index)
    }
}

/// Parse: creates a prepared statement from SQL text, with optional
/// parameter type pre-declarations.
#[derive(Debug, Clone, Copy)]
pub struct ParseMessage<'a> {
    pub statement: PgStr<'a>,
    pub sql: PgStr<'a>,
    pub parameter_types: Oids<'a>,
}

/// FunctionCall: legacy fast-path function invocation.
#[derive(Debug, Clone, Copy)]
pub struct FunctionCall<'a> {
    pub object: Oid,
    pub argument_formats: FormatCodes<'a>,
    pub arguments: Values<'a>,
    pub result_format: FormatCode,
}

/// Body of a frontend `p` message. The tag is shared by cleartext
/// passwords, MD5 responses, SASL responses, and GSSAPI responses — the
/// active authentication flow (tracked by the session FSM) determines
/// the interpretation, so the codec exposes the raw body plus typed
/// refinements.
#[derive(Debug, Clone, Copy)]
pub struct AuthData<'a> {
    pub data: &'a [u8],
}

/// SASLInitialResponse refinement of an [`AuthData`] body.
#[derive(Debug, Clone, Copy)]
pub struct SaslInitialResponse<'a> {
    pub mechanism: PgStr<'a>,
    /// initial client response, absent when the mechanism sends none
    pub data: Option<&'a [u8]>,
}

impl<'a> AuthData<'a> {
    /// Cleartext password or MD5 hex digest: a single string.
    ///
    /// # Errors
    /// [`ParseError`] when the body is not one NUL-terminated string.
    pub fn as_password(&self) -> Result<PgStr<'a>, ParseError> {
        let mut reader = Reader::new(self.data, b'p');
        let password = reader.take_cstr()?;
        reader.expect_end()?;
        Ok(password)
    }

    /// SASLInitialResponse: mechanism name plus optional initial data.
    ///
    /// # Errors
    /// [`ParseError`] when the mechanism string or the Int32-prefixed
    /// data section is malformed.
    pub fn as_sasl_initial(&self) -> Result<SaslInitialResponse<'a>, ParseError> {
        let mut reader = Reader::new(self.data, b'p');
        let mechanism = reader.take_cstr()?;
        let length = reader.take_i32()?;
        let data = match usize::try_from(length) {
            Ok(count) => Some(reader.take_bytes(count)?),
            Err(_) if length == -1 => None,
            Err(_) => {
                return Err(ParseError::InvalidValue {
                    tag: b'p',
                    field: "sasl data length",
                });
            }
        };
        reader.expect_end()?;
        Ok(SaslInitialResponse { mechanism, data })
    }

    /// SASLResponse: raw mechanism-specific continuation bytes.
    #[must_use]
    pub const fn as_sasl_response(&self) -> &'a [u8] {
        self.data
    }

    /// GSSResponse: raw GSSAPI/SSPI token bytes.
    #[must_use]
    pub const fn as_gss_response(&self) -> &'a [u8] {
        self.data
    }
}

/// Tagged frontend message.
#[derive(Debug, Clone, Copy)]
pub enum FrontendMessage<'a> {
    /// `p` — authentication response data (see [`AuthData`])
    AuthData(AuthData<'a>),
    /// `Q` — simple query
    Query { sql: PgStr<'a> },
    /// `P` — extended query: create prepared statement
    Parse(ParseMessage<'a>),
    /// `B` — extended query: bind portal
    Bind(Bind<'a>),
    /// `D` — describe statement or portal
    Describe {
        target: StatementTarget,
        name: PgStr<'a>,
    },
    /// `E` — execute portal with optional row limit (0 = unlimited)
    Execute { portal: PgStr<'a>, max_rows: i32 },
    /// `C` — close statement or portal
    Close {
        target: StatementTarget,
        name: PgStr<'a>,
    },
    /// `H` — flush pending output
    Flush,
    /// `S` — end extended-query pipeline, request ReadyForQuery
    Sync,
    /// `d` — COPY data stream chunk
    CopyData { data: &'a [u8] },
    /// `c` — COPY stream finished
    CopyDone,
    /// `f` — COPY failed on the frontend side
    CopyFail { message: PgStr<'a> },
    /// `F` — legacy fast-path function call
    FunctionCall(FunctionCall<'a>),
    /// `X` — graceful termination
    Terminate,
}

/// Parses one tagged frontend message.
///
/// Returns `Ok(None)` when `input` does not yet hold a complete frame.
///
/// # Errors
/// [`ParseError`] when the frame header, tag, or body is invalid.
pub fn parse_frontend(input: &[u8]) -> Result<Option<(FrontendMessage<'_>, usize)>, ParseError> {
    let Some((tag, body, consumed)) = split_tagged(input)? else {
        return Ok(None);
    };
    let mut reader = Reader::new(body, tag);
    let message = match tag {
        b'p' => FrontendMessage::AuthData(AuthData {
            data: reader.take_rest(),
        }),
        b'Q' => {
            let sql = reader.take_cstr()?;
            reader.expect_end()?;
            FrontendMessage::Query { sql }
        }
        b'P' => {
            let statement = reader.take_cstr()?;
            let sql = reader.take_cstr()?;
            let parameter_types = Oids::validate(&mut reader, "parameter type count")?;
            reader.expect_end()?;
            FrontendMessage::Parse(ParseMessage {
                statement,
                sql,
                parameter_types,
            })
        }
        b'B' => {
            let portal = reader.take_cstr()?;
            let statement = reader.take_cstr()?;
            let parameter_formats = FormatCodes::validate(&mut reader, "parameter format count")?;
            let parameters = Values::validate(&mut reader, "parameter count")?;
            let result_formats = FormatCodes::validate(&mut reader, "result format count")?;
            reader.expect_end()?;
            FrontendMessage::Bind(Bind {
                portal,
                statement,
                parameter_formats,
                parameters,
                result_formats,
            })
        }
        b'D' | b'C' => {
            let raw_target = reader.take_u8()?;
            let Some(target) = StatementTarget::from_byte(raw_target) else {
                return Err(ParseError::InvalidValue {
                    tag,
                    field: "describe/close target",
                });
            };
            let name = reader.take_cstr()?;
            reader.expect_end()?;
            if tag == b'D' {
                FrontendMessage::Describe { target, name }
            } else {
                FrontendMessage::Close { target, name }
            }
        }
        b'E' => {
            let portal = reader.take_cstr()?;
            let max_rows = reader.take_i32()?;
            if max_rows < 0 {
                return Err(ParseError::InvalidValue {
                    tag,
                    field: "max rows",
                });
            }
            reader.expect_end()?;
            FrontendMessage::Execute { portal, max_rows }
        }
        b'H' => {
            reader.expect_end()?;
            FrontendMessage::Flush
        }
        b'S' => {
            reader.expect_end()?;
            FrontendMessage::Sync
        }
        b'd' => FrontendMessage::CopyData {
            data: reader.take_rest(),
        },
        b'c' => {
            reader.expect_end()?;
            FrontendMessage::CopyDone
        }
        b'f' => {
            let message = reader.take_cstr()?;
            reader.expect_end()?;
            FrontendMessage::CopyFail { message }
        }
        b'F' => {
            let object = Oid(reader.take_u32()?);
            let argument_formats = FormatCodes::validate(&mut reader, "argument format count")?;
            let arguments = Values::validate(&mut reader, "argument count")?;
            let raw_format = reader.take_i16()?;
            let Some(result_format) = FormatCode::from_i16(raw_format) else {
                return Err(ParseError::InvalidValue {
                    tag,
                    field: "result format code",
                });
            };
            reader.expect_end()?;
            FrontendMessage::FunctionCall(FunctionCall {
                object,
                argument_formats,
                arguments,
                result_format,
            })
        }
        b'X' => {
            reader.expect_end()?;
            FrontendMessage::Terminate
        }
        _ => return Err(ParseError::UnknownTag { tag }),
    };
    Ok(Some((message, consumed)))
}

impl InitialMessage<'_> {
    /// Encodes the startup-phase message into `out`, returning the
    /// encoded size.
    ///
    /// # Errors
    /// [`EncodeError`] when `out` is too small or a field is invalid.
    pub fn encode(&self, out: &mut [u8]) -> Result<usize, EncodeError> {
        let mut writer = MessageWriter::untagged(out)?;
        match self {
            Self::Startup(startup) => {
                writer.put_i32(startup.version.as_code())?;
                for (key, value) in startup.parameters.iter() {
                    writer.put_cstr(key.as_bytes())?;
                    writer.put_cstr(value.as_bytes())?;
                }
                writer.put_u8(0)?;
            }
            Self::SslRequest => {
                writer.put_i32(SSL_REQUEST_CODE)?;
            }
            Self::GssEncRequest => {
                writer.put_i32(GSSENC_REQUEST_CODE)?;
            }
            Self::Cancel(cancel) => {
                writer.put_i32(CANCEL_REQUEST_CODE)?;
                writer.put_i32(cancel.process_id)?;
                writer.put_bytes(cancel.secret_key)?;
            }
        }
        writer.finish()
    }
}

impl FrontendMessage<'_> {
    /// Wire tag byte of this message.
    #[must_use]
    pub const fn tag(&self) -> u8 {
        match self {
            Self::AuthData(_) => b'p',
            Self::Query { .. } => b'Q',
            Self::Parse(_) => b'P',
            Self::Bind(_) => b'B',
            Self::Describe { .. } => b'D',
            Self::Execute { .. } => b'E',
            Self::Close { .. } => b'C',
            Self::Flush => b'H',
            Self::Sync => b'S',
            Self::CopyData { .. } => b'd',
            Self::CopyDone => b'c',
            Self::CopyFail { .. } => b'f',
            Self::FunctionCall(_) => b'F',
            Self::Terminate => b'X',
        }
    }

    /// Encodes the message into `out`, returning the encoded size.
    ///
    /// # Errors
    /// [`EncodeError`] when `out` is too small or a field is invalid.
    pub fn encode(&self, out: &mut [u8]) -> Result<usize, EncodeError> {
        let mut writer = MessageWriter::tagged(out, self.tag())?;
        match self {
            Self::AuthData(auth) => {
                writer.put_bytes(auth.data)?;
            }
            Self::Query { sql } => {
                writer.put_cstr(sql.as_bytes())?;
            }
            Self::Parse(parse) => {
                writer.put_cstr(parse.statement.as_bytes())?;
                writer.put_cstr(parse.sql.as_bytes())?;
                put_count16(
                    &mut writer,
                    parse.parameter_types.len(),
                    "parameter type count",
                )?;
                for oid in parse.parameter_types.iter() {
                    writer.put_u32(oid.0)?;
                }
            }
            Self::Bind(bind) => {
                writer.put_cstr(bind.portal.as_bytes())?;
                writer.put_cstr(bind.statement.as_bytes())?;
                encode_format_codes(&mut writer, &bind.parameter_formats)?;
                encode_values(&mut writer, &bind.parameters)?;
                encode_format_codes(&mut writer, &bind.result_formats)?;
            }
            Self::Describe { target, name } | Self::Close { target, name } => {
                writer.put_u8(target.as_byte())?;
                writer.put_cstr(name.as_bytes())?;
            }
            Self::Execute { portal, max_rows } => {
                writer.put_cstr(portal.as_bytes())?;
                writer.put_i32(*max_rows)?;
            }
            Self::Flush | Self::Sync | Self::CopyDone | Self::Terminate => {}
            Self::CopyData { data } => {
                writer.put_bytes(data)?;
            }
            Self::CopyFail { message } => {
                writer.put_cstr(message.as_bytes())?;
            }
            Self::FunctionCall(call) => {
                writer.put_u32(call.object.0)?;
                encode_format_codes(&mut writer, &call.argument_formats)?;
                encode_values(&mut writer, &call.arguments)?;
                writer.put_i16(call.result_format.as_i16())?;
            }
        }
        writer.finish()
    }
}

fn put_count16(
    writer: &mut MessageWriter<'_>,
    count: usize,
    field: &'static str,
) -> Result<(), EncodeError> {
    let Ok(count) = i16::try_from(count) else {
        return Err(EncodeError::ValueTooLarge { field });
    };
    writer.put_i16(count)?;
    Ok(())
}

fn encode_format_codes(
    writer: &mut MessageWriter<'_>,
    codes: &FormatCodes<'_>,
) -> Result<(), EncodeError> {
    put_count16(writer, codes.len(), "format code count")?;
    for code in codes.iter() {
        writer.put_i16(code.as_i16())?;
    }
    Ok(())
}

fn encode_values(writer: &mut MessageWriter<'_>, values: &Values<'_>) -> Result<(), EncodeError> {
    put_count16(writer, values.len(), "value count")?;
    for value in values.iter() {
        match value {
            Some(bytes) => {
                let Ok(length) = i32::try_from(bytes.len()) else {
                    return Err(EncodeError::ValueTooLarge {
                        field: "value length",
                    });
                };
                writer.put_i32(length)?;
                writer.put_bytes(bytes)?;
            }
            None => {
                writer.put_i32(-1)?;
            }
        }
    }
    Ok(())
}

/// Streaming Bind encoder for client-role construction (the borrowed
/// [`Bind`] view cannot be assembled field-by-field without a buffer to
/// borrow from).
#[derive(Debug)]
pub struct BindWriter<'a> {
    writer: MessageWriter<'a>,
    count_at: usize,
    count: usize,
    stage: BindStage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BindStage {
    Parameters,
    ResultFormats,
}

impl<'a> BindWriter<'a> {
    /// Starts a Bind message with the given portal / statement names and
    /// per-parameter format codes.
    ///
    /// # Errors
    /// [`EncodeError`] when `out` is too small or a name embeds NUL.
    pub fn begin(
        out: &'a mut [u8],
        portal: &[u8],
        statement: &[u8],
        parameter_formats: &[FormatCode],
    ) -> Result<Self, EncodeError> {
        let mut writer = MessageWriter::tagged(out, b'B')?;
        writer.put_cstr(portal)?;
        writer.put_cstr(statement)?;
        put_count16(
            &mut writer,
            parameter_formats.len(),
            "parameter format count",
        )?;
        for code in parameter_formats {
            writer.put_i16(code.as_i16())?;
        }
        let count_at = writer.written();
        writer.put_i16(0)?;
        Ok(Self {
            writer,
            count_at,
            count: 0,
            stage: BindStage::Parameters,
        })
    }

    /// Appends one parameter value (`None` = NULL).
    ///
    /// # Errors
    /// [`EncodeError`] when the buffer is too small, the value exceeds
    /// Int32, or the parameter count exceeds Int16.
    pub fn parameter(&mut self, value: Option<&[u8]>) -> Result<&mut Self, EncodeError> {
        debug_assert_eq!(self.stage, BindStage::Parameters);
        match value {
            Some(bytes) => {
                let Ok(length) = i32::try_from(bytes.len()) else {
                    return Err(EncodeError::ValueTooLarge {
                        field: "parameter length",
                    });
                };
                self.writer.put_i32(length)?;
                self.writer.put_bytes(bytes)?;
            }
            None => {
                self.writer.put_i32(-1)?;
            }
        }
        self.count += 1;
        if i16::try_from(self.count).is_err() {
            return Err(EncodeError::ValueTooLarge {
                field: "parameter count",
            });
        }
        Ok(self)
    }

    /// Closes the parameter list and writes the result-column format
    /// codes, then finishes the message.
    ///
    /// # Errors
    /// [`EncodeError`] when the buffer is too small.
    pub fn finish(mut self, result_formats: &[FormatCode]) -> Result<usize, EncodeError> {
        self.stage = BindStage::ResultFormats;
        let Ok(count) = i16::try_from(self.count) else {
            return Err(EncodeError::ValueTooLarge {
                field: "parameter count",
            });
        };
        self.writer.patch_i16(self.count_at, count);
        put_count16(
            &mut self.writer,
            result_formats.len(),
            "result format count",
        )?;
        for code in result_formats {
            self.writer.put_i16(code.as_i16())?;
        }
        self.writer.finish()
    }
}

// the test helpers build Vec<u8> frames; this crate carries no alloc
// dependency for its no_std tier, so the suite needs std, not just test
#[cfg(all(test, feature = "std"))]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use rstest::rstest;

    use super::*;
    use super::super::error::{EncodeError, ParseError};
    use super::super::types::{FormatCode, ProtocolVersion};

    fn encode_initial(msg: &InitialMessage<'_>) -> Vec<u8> {
        let mut buf = vec![0u8; 1024];
        let written = msg.encode(&mut buf).expect("encode must succeed");
        buf[..written].to_vec()
    }

    fn encode_frontend(msg: &FrontendMessage<'_>) -> Vec<u8> {
        let mut buf = vec![0u8; 1024];
        let written = msg.encode(&mut buf).expect("encode must succeed");
        buf[..written].to_vec()
    }

    #[test]
    fn ssl_request_round_trips() {
        let msg = InitialMessage::SslRequest;
        let bytes = encode_initial(&msg);
        let (parsed, consumed) = parse_initial(&bytes).expect("parse").expect("complete");
        assert_eq!(consumed, bytes.len());
        assert!(matches!(parsed, InitialMessage::SslRequest));
    }

    #[test]
    fn gssenc_request_round_trips() {
        let msg = InitialMessage::GssEncRequest;
        let bytes = encode_initial(&msg);
        let (parsed, consumed) = parse_initial(&bytes).expect("parse").expect("complete");
        assert_eq!(consumed, bytes.len());
        assert!(matches!(parsed, InitialMessage::GssEncRequest));
    }

    #[test]
    fn startup_with_params_round_trips() {
        let raw_params = b"user\0alice\0database\0appdb\0\0";
        let mut reader = super::super::cursor::Reader::new(raw_params, 0);
        let params = super::super::views::StartupParams::validate(&mut reader).expect("params validate");
        let msg = InitialMessage::Startup(Startup {
            version: ProtocolVersion::V3_0,
            parameters: params,
        });
        let bytes = encode_initial(&msg);
        let (parsed, consumed) = parse_initial(&bytes).expect("parse").expect("complete");
        assert_eq!(consumed, bytes.len());
        let InitialMessage::Startup(startup) = parsed else {
            panic!("expected Startup");
        };
        assert_eq!(startup.version, ProtocolVersion::V3_0);
        assert_eq!(startup.user(), Some(PgStr::new(b"alice")));
    }

    #[test]
    fn cancel_request_round_trips() {
        let key = 99999i32.to_be_bytes();
        let msg = InitialMessage::Cancel(CancelRequest {
            process_id: 12345,
            secret_key: &key,
        });
        let bytes = encode_initial(&msg);
        let (parsed, consumed) = parse_initial(&bytes).expect("parse").expect("complete");
        assert_eq!(consumed, bytes.len());
        let InitialMessage::Cancel(cancel) = parsed else {
            panic!("expected Cancel");
        };
        assert_eq!(cancel.process_id, 12345);
        assert_eq!(cancel.secret_key, key.as_slice());
    }

    #[test]
    fn parse_initial_returns_none_on_incomplete_input() {
        let bytes = [0u8; 3];
        let result = parse_initial(&bytes).expect("no error on incomplete");
        assert!(result.is_none(), "incomplete input must return None");
    }

    #[test]
    fn parse_initial_returns_none_when_length_exceeds_buffer() {
        let mut bytes = vec![0u8; 8];
        bytes[0..4].copy_from_slice(&20i32.to_be_bytes());
        bytes[4..8].copy_from_slice(&80877103i32.to_be_bytes());
        let result = parse_initial(&bytes).expect("no error");
        assert!(result.is_none(), "partial frame must return None");
    }

    #[test]
    fn parse_initial_bad_length_below_8_returns_error() {
        let mut bytes = vec![0u8; 8];
        bytes[0..4].copy_from_slice(&7i32.to_be_bytes());
        bytes[4..8].copy_from_slice(&80877103i32.to_be_bytes());
        let result = parse_initial(&bytes);
        assert!(matches!(result, Err(ParseError::BadLength { .. })));
    }

    #[test]
    fn parse_initial_unsupported_protocol_version_returns_error() {
        let version_code = (2u32 << 16) as i32;
        let mut bytes = vec![0u8; 9];
        bytes[0..4].copy_from_slice(&9i32.to_be_bytes());
        bytes[4..8].copy_from_slice(&version_code.to_be_bytes());
        bytes[8] = 0;
        let result = parse_initial(&bytes);
        assert!(matches!(
            result,
            Err(ParseError::UnsupportedProtocol { .. })
        ));
    }

    #[test]
    fn parse_initial_unknown_request_code_returns_error() {
        let code: i32 = (1234 << 16) | 9999;
        let mut bytes = vec![0u8; 8];
        bytes[0..4].copy_from_slice(&8i32.to_be_bytes());
        bytes[4..8].copy_from_slice(&code.to_be_bytes());
        let result = parse_initial(&bytes);
        assert!(matches!(result, Err(ParseError::UnknownRequestCode { .. })));
    }

    #[rstest]
    #[case::query(b"Q")]
    #[case::flush(b"H")]
    #[case::sync(b"S")]
    #[case::terminate(b"X")]
    #[case::copy_done(b"c")]
    fn zero_body_messages_round_trip(#[case] tag: &[u8]) {
        let mut bytes = vec![0u8; 5];
        bytes[0] = tag[0];
        bytes[1..5].copy_from_slice(&4i32.to_be_bytes());

        if tag[0] == b'Q' {
            let mut buf = vec![0u8; 1024];
            let msg = FrontendMessage::Query {
                sql: PgStr::new(b"select 1"),
            };
            let written = msg.encode(&mut buf).expect("encode");
            let (parsed, consumed) = parse_frontend(&buf[..written])
                .expect("parse")
                .expect("complete");
            assert_eq!(consumed, written);
            assert!(matches!(parsed, FrontendMessage::Query { .. }));
        } else {
            let (parsed, consumed) = parse_frontend(&bytes).expect("parse").expect("complete");
            assert_eq!(consumed, bytes.len());
            match tag[0] {
                b'H' => assert!(matches!(parsed, FrontendMessage::Flush)),
                b'S' => assert!(matches!(parsed, FrontendMessage::Sync)),
                b'X' => assert!(matches!(parsed, FrontendMessage::Terminate)),
                b'c' => assert!(matches!(parsed, FrontendMessage::CopyDone)),
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn parse_frontend_returns_none_on_incomplete() {
        let bytes = [b'Q', 0, 0, 0];
        let result = parse_frontend(&bytes).expect("no error");
        assert!(result.is_none());
    }

    #[test]
    fn parse_frontend_unknown_tag_returns_error() {
        let mut bytes = [0u8; 5];
        bytes[0] = b'?';
        bytes[1..5].copy_from_slice(&4i32.to_be_bytes());
        let result = parse_frontend(&bytes);
        assert!(matches!(result, Err(ParseError::UnknownTag { tag: b'?' })));
    }

    #[test]
    fn parse_frontend_bad_length_below_4_returns_error() {
        let mut bytes = [0u8; 5];
        bytes[0] = b'Q';
        bytes[1..5].copy_from_slice(&3i32.to_be_bytes());
        let result = parse_frontend(&bytes);
        assert!(matches!(
            result,
            Err(ParseError::BadLength { tag: b'Q', .. })
        ));
    }

    #[test]
    fn query_trailing_bytes_returns_error() {
        let sql = b"select 1";
        let body_len = sql.len() + 1 + 1;
        let total = 1 + 4 + body_len;
        let mut bytes = vec![0u8; total];
        bytes[0] = b'Q';
        bytes[1..5].copy_from_slice(&((body_len + 4) as i32).to_be_bytes());
        bytes[5..5 + sql.len()].copy_from_slice(sql);
        bytes[5 + sql.len()] = 0;
        bytes[5 + sql.len() + 1] = b'X';
        let result = parse_frontend(&bytes);
        assert!(matches!(
            result,
            Err(ParseError::TrailingBytes { tag: b'Q', .. })
        ));
    }

    #[test]
    fn execute_negative_max_rows_returns_error() {
        let portal = b"";
        let body_len = portal.len() + 1 + 4;
        let total = 1 + 4 + body_len;
        let mut bytes = vec![0u8; total];
        bytes[0] = b'E';
        bytes[1..5].copy_from_slice(&((body_len + 4) as i32).to_be_bytes());
        bytes[5] = 0;
        bytes[6..10].copy_from_slice(&(-1i32).to_be_bytes());
        let result = parse_frontend(&bytes);
        assert!(matches!(
            result,
            Err(ParseError::InvalidValue { tag: b'E', .. })
        ));
    }

    #[test]
    fn describe_invalid_target_byte_returns_error() {
        let name = b"stmt\0";
        let body_len = 1 + name.len();
        let total = 1 + 4 + body_len;
        let mut bytes = vec![0u8; total];
        bytes[0] = b'D';
        bytes[1..5].copy_from_slice(&((body_len + 4) as i32).to_be_bytes());
        bytes[5] = b'X';
        bytes[6..6 + name.len()].copy_from_slice(name);
        let result = parse_frontend(&bytes);
        assert!(matches!(
            result,
            Err(ParseError::InvalidValue { tag: b'D', .. })
        ));
    }

    #[test]
    fn truncated_parse_returns_none_until_frame_complete() {
        let msg = FrontendMessage::Query {
            sql: PgStr::new(b"select id, email from users where id = $1"),
        };
        let mut buf = vec![0u8; 256];
        let written = msg.encode(&mut buf).expect("encode must succeed");
        let full = buf[..written].to_vec();

        for prefix_len in 0..full.len() {
            let result = parse_frontend(&full[..prefix_len]).expect("no error at any split");
            assert!(
                result.is_none(),
                "prefix of length {prefix_len} must return None"
            );
        }

        let (parsed, consumed) = parse_frontend(&full)
            .expect("full frame parses")
            .expect("complete");
        assert_eq!(consumed, full.len());
        assert!(matches!(parsed, FrontendMessage::Query { .. }));
    }

    #[test]
    fn truncated_startup_returns_none_until_frame_complete() {
        let raw_params = b"user\0alice\0\0";
        let mut reader = super::super::cursor::Reader::new(raw_params, 0);
        let params = super::super::views::StartupParams::validate(&mut reader).expect("params validate");
        let msg = InitialMessage::Startup(Startup {
            version: ProtocolVersion::V3_0,
            parameters: params,
        });
        let full = encode_initial(&msg);

        for prefix_len in 0..full.len() {
            let result = parse_initial(&full[..prefix_len]).expect("no error at any split");
            assert!(
                result.is_none(),
                "prefix of length {prefix_len} must return None"
            );
        }

        let (_, consumed) = parse_initial(&full).expect("full parse").expect("complete");
        assert_eq!(consumed, full.len());
    }

    #[test]
    fn bind_message_with_null_param_round_trips() {
        let mut buf = vec![0u8; 256];
        let written = {
            let mut writer = BindWriter::begin(&mut buf, b"", b"get-user", &[FormatCode::Text])
                .expect("begin must succeed");
            writer.parameter(Some(b"42")).expect("param 1");
            writer.parameter(None).expect("null param");
            writer.parameter(Some(b"hello")).expect("param 3");
            writer.finish(&[FormatCode::Text]).expect("finish")
        };

        let (msg, consumed) = parse_frontend(&buf[..written])
            .expect("parse")
            .expect("complete");
        assert_eq!(consumed, written);
        let FrontendMessage::Bind(bind) = msg else {
            panic!("expected Bind");
        };
        assert_eq!(bind.parameters.len(), 3);
        let params: Vec<Option<&[u8]>> = bind.parameters.iter().collect();
        assert_eq!(params[0], Some(b"42".as_slice()));
        assert_eq!(params[1], None);
        assert_eq!(params[2], Some(b"hello".as_slice()));
    }

    #[test]
    fn bind_writer_buffer_too_small_returns_error() {
        let mut buf = [0u8; 4];
        let result = BindWriter::begin(&mut buf, b"", b"stmt", &[]);
        assert!(matches!(result, Err(EncodeError::BufferTooSmall { .. })));
    }

    #[test]
    fn copy_fail_tag_is_f_per_spec() {
        let msg = FrontendMessage::CopyFail {
            message: PgStr::new(b"copy aborted"),
        };
        let bytes = encode_frontend(&msg);
        assert_eq!(
            bytes[0], b'f',
            "CopyFail tag must be 'f' per PostgreSQL spec"
        );
    }

    #[test]
    fn auth_data_as_password_parses_cstring() {
        let mut buf = vec![0u8; 32];
        let msg = FrontendMessage::AuthData(AuthData { data: b"s3cr3t\0" });
        let written = msg.encode(&mut buf).expect("encode");
        let (parsed, _) = parse_frontend(&buf[..written])
            .expect("parse")
            .expect("complete");
        let FrontendMessage::AuthData(auth) = parsed else {
            panic!("expected AuthData");
        };
        let password = auth.as_password().expect("as_password");
        assert_eq!(password, "s3cr3t");
    }

    #[test]
    fn auth_data_as_sasl_initial_parses_mechanism_and_data() {
        let mechanism = b"SCRAM-SHA-256\0";
        let client_msg = b"client-first";
        let mut data = vec![];
        data.extend_from_slice(mechanism);
        data.extend_from_slice(&(client_msg.len() as i32).to_be_bytes());
        data.extend_from_slice(client_msg);
        let auth = AuthData { data: &data };
        let sasl = auth.as_sasl_initial().expect("as_sasl_initial");
        assert_eq!(sasl.mechanism, "SCRAM-SHA-256");
        assert_eq!(sasl.data, Some(client_msg.as_slice()));
    }

    #[test]
    fn auth_data_as_sasl_response_returns_raw_bytes() {
        let raw = b"client-final-message";
        let auth = AuthData { data: raw };
        assert_eq!(auth.as_sasl_response(), raw.as_slice());
    }

    #[test]
    fn function_call_invalid_result_format_returns_error() {
        let oid_val: u32 = 1753;
        let arg_data = b"data";
        let body_len: usize = 4 + 2 + 2 + 2 + 4 + arg_data.len() + 2;
        let total = 1 + 4 + body_len;
        let mut bytes = vec![0u8; total];
        bytes[0] = b'F';
        bytes[1..5].copy_from_slice(&((body_len + 4) as i32).to_be_bytes());
        let mut pos = 5;
        bytes[pos..pos + 4].copy_from_slice(&oid_val.to_be_bytes());
        pos += 4;
        bytes[pos..pos + 2].copy_from_slice(&1i16.to_be_bytes());
        pos += 2;
        bytes[pos..pos + 2].copy_from_slice(&0i16.to_be_bytes());
        pos += 2;
        bytes[pos..pos + 2].copy_from_slice(&1i16.to_be_bytes());
        pos += 2;
        bytes[pos..pos + 4].copy_from_slice(&(arg_data.len() as i32).to_be_bytes());
        pos += 4;
        bytes[pos..pos + arg_data.len()].copy_from_slice(arg_data);
        pos += arg_data.len();
        bytes[pos..pos + 2].copy_from_slice(&7i16.to_be_bytes());
        let result = parse_frontend(&bytes[..total]);
        assert!(matches!(
            result,
            Err(ParseError::InvalidValue { tag: b'F', .. })
        ));
    }
}
