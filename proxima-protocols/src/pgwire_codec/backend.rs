//! Backend (server → client) message decode and encode.
//!
//! Message shapes follow the PostgreSQL protocol 3.x message formats
//! reference (`https://www.postgresql.org/docs/current/protocol-message-formats.html`).
//! Encode is the server hot path: the streaming writers
//! ([`DataRowWriter`], [`RowDescriptionWriter`], [`ErrorResponseWriter`])
//! write straight into caller-owned buffers with zero allocation. Decode
//! exists for client-role use, proxies, and round-trip tests.

use super::cursor::Reader;
use super::error::{EncodeError, ParseError};
use super::frame::split_tagged;
use super::types::{CopyFormat, FormatCode, Oid, PgStr, TransactionStatus, error_field};
use super::views::{CStrList, CountedCStrs, ErrorFields, Fields, FormatCodes, Values};
use super::writer::MessageWriter;

/// Authentication request carried by a backend `R` message.
///
/// KerberosV5, ScmCredential, Gss, and Sspi are legacy / out-of-scope
/// methods represented as explicit typed variants so a caller can reject
/// them deliberately rather than fail to parse.
#[derive(Debug, Clone, Copy)]
pub enum AuthRequest<'a> {
    /// code 0 — authentication succeeded
    Ok,
    /// code 2 — Kerberos V5 (legacy)
    KerberosV5,
    /// code 3 — cleartext password
    CleartextPassword,
    /// code 5 — MD5 password with per-session salt
    Md5Password { salt: [u8; 4] },
    /// code 6 — SCM credentials (obsolete)
    ScmCredential,
    /// code 7 — GSSAPI
    Gss,
    /// code 8 — GSSAPI/SSPI continuation token
    GssContinue { data: &'a [u8] },
    /// code 9 — SSPI
    Sspi,
    /// code 10 — SASL with offered mechanism names
    Sasl { mechanisms: CStrList<'a> },
    /// code 11 — SASL challenge continuation
    SaslContinue { data: &'a [u8] },
    /// code 12 — SASL final server message
    SaslFinal { data: &'a [u8] },
}

/// Tagged backend message.
#[derive(Debug, Clone, Copy)]
pub enum BackendMessage<'a> {
    /// `R` — authentication request / completion
    Authentication(AuthRequest<'a>),
    /// `K` — cancellation key data (4-byte key in 3.0; up to 256 in 3.2)
    BackendKeyData {
        process_id: i32,
        secret_key: &'a [u8],
    },
    /// `2`
    BindComplete,
    /// `3`
    CloseComplete,
    /// `C` — command completion tag, e.g. `SELECT 1`
    CommandComplete { tag: PgStr<'a> },
    /// `d`
    CopyData { data: &'a [u8] },
    /// `c`
    CopyDone,
    /// `G`
    CopyInResponse {
        format: CopyFormat,
        column_formats: FormatCodes<'a>,
    },
    /// `H`
    CopyOutResponse {
        format: CopyFormat,
        column_formats: FormatCodes<'a>,
    },
    /// `W`
    CopyBothResponse {
        format: CopyFormat,
        column_formats: FormatCodes<'a>,
    },
    /// `D`
    DataRow { columns: Values<'a> },
    /// `I`
    EmptyQueryResponse,
    /// `E`
    ErrorResponse { fields: ErrorFields<'a> },
    /// `V` — fast-path function result (`None` = NULL result)
    FunctionCallResponse { value: Option<&'a [u8]> },
    /// `v` — sent when the client requested protocol 3.x above the
    /// server's newest minor, or unknown `_pq_.*` options
    NegotiateProtocolVersion {
        newest_minor: i32,
        unsupported_options: CountedCStrs<'a>,
    },
    /// `n`
    NoData,
    /// `N`
    NoticeResponse { fields: ErrorFields<'a> },
    /// `A`
    NotificationResponse {
        process_id: i32,
        channel: PgStr<'a>,
        payload: PgStr<'a>,
    },
    /// `t`
    ParameterDescription {
        parameter_types: super::views::Oids<'a>,
    },
    /// `S`
    ParameterStatus { name: PgStr<'a>, value: PgStr<'a> },
    /// `1`
    ParseComplete,
    /// `s`
    PortalSuspended,
    /// `Z`
    ReadyForQuery { status: TransactionStatus },
    /// `T`
    RowDescription { fields: Fields<'a> },
}

/// Single-byte response to SSLRequest, outside the tagged framing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SslResponse {
    /// `S` — proceed with TLS handshake
    Accept,
    /// `N` — continue in cleartext
    Refuse,
}

impl SslResponse {
    /// Encodes the one-byte response.
    ///
    /// # Errors
    /// [`EncodeError::BufferTooSmall`] when `out` is empty.
    pub fn encode(self, out: &mut [u8]) -> Result<usize, EncodeError> {
        if out.is_empty() {
            return Err(EncodeError::BufferTooSmall { needed: 1 });
        }
        out[0] = match self {
            Self::Accept => b'S',
            Self::Refuse => b'N',
        };
        Ok(1)
    }
}

/// Single-byte response to GSSENCRequest, outside the tagged framing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GssEncResponse {
    /// `G` — proceed with GSSAPI encryption
    Accept,
    /// `N` — continue in cleartext
    Refuse,
}

impl GssEncResponse {
    /// Encodes the one-byte response.
    ///
    /// # Errors
    /// [`EncodeError::BufferTooSmall`] when `out` is empty.
    pub fn encode(self, out: &mut [u8]) -> Result<usize, EncodeError> {
        if out.is_empty() {
            return Err(EncodeError::BufferTooSmall { needed: 1 });
        }
        out[0] = match self {
            Self::Accept => b'G',
            Self::Refuse => b'N',
        };
        Ok(1)
    }
}

/// Parses the single-byte SSLRequest response (client role).
///
/// # Errors
/// [`ParseError::InvalidValue`] on any byte other than `S` / `N`.
pub fn parse_ssl_response(input: &[u8]) -> Result<Option<(SslResponse, usize)>, ParseError> {
    let Some(&byte) = input.first() else {
        return Ok(None);
    };
    match byte {
        b'S' => Ok(Some((SslResponse::Accept, 1))),
        b'N' => Ok(Some((SslResponse::Refuse, 1))),
        _ => Err(ParseError::InvalidValue {
            tag: byte,
            field: "ssl response byte",
        }),
    }
}

/// Parses the single-byte GSSENCRequest response (client role).
///
/// # Errors
/// [`ParseError::InvalidValue`] on any byte other than `G` / `N`.
pub fn parse_gssenc_response(input: &[u8]) -> Result<Option<(GssEncResponse, usize)>, ParseError> {
    let Some(&byte) = input.first() else {
        return Ok(None);
    };
    match byte {
        b'G' => Ok(Some((GssEncResponse::Accept, 1))),
        b'N' => Ok(Some((GssEncResponse::Refuse, 1))),
        _ => Err(ParseError::InvalidValue {
            tag: byte,
            field: "gssenc response byte",
        }),
    }
}

/// Parses one tagged backend message.
///
/// Returns `Ok(None)` when `input` does not yet hold a complete frame.
///
/// # Errors
/// [`ParseError`] when the frame header, tag, or body is invalid.
pub fn parse_backend(input: &[u8]) -> Result<Option<(BackendMessage<'_>, usize)>, ParseError> {
    let Some((tag, body, consumed)) = split_tagged(input)? else {
        return Ok(None);
    };
    let mut reader = Reader::new(body, tag);
    let message = match tag {
        b'R' => BackendMessage::Authentication(parse_auth_request(&mut reader)?),
        b'K' => {
            let process_id = reader.take_i32()?;
            let secret_key = reader.take_rest();
            if secret_key.len() < 4 || secret_key.len() > 256 {
                return Err(ParseError::InvalidValue {
                    tag,
                    field: "cancellation key",
                });
            }
            BackendMessage::BackendKeyData {
                process_id,
                secret_key,
            }
        }
        b'2' => {
            reader.expect_end()?;
            BackendMessage::BindComplete
        }
        b'3' => {
            reader.expect_end()?;
            BackendMessage::CloseComplete
        }
        b'C' => {
            let command_tag = reader.take_cstr()?;
            reader.expect_end()?;
            BackendMessage::CommandComplete { tag: command_tag }
        }
        b'd' => BackendMessage::CopyData {
            data: reader.take_rest(),
        },
        b'c' => {
            reader.expect_end()?;
            BackendMessage::CopyDone
        }
        b'G' | b'H' | b'W' => {
            let raw_format = reader.take_u8()?;
            #[expect(clippy::cast_possible_wrap, reason = "wire Int8 read as raw byte")]
            let Some(format) = CopyFormat::from_i8(raw_format as i8) else {
                return Err(ParseError::InvalidValue {
                    tag,
                    field: "copy format",
                });
            };
            let column_formats = FormatCodes::validate(&mut reader, "copy column format count")?;
            reader.expect_end()?;
            match tag {
                b'G' => BackendMessage::CopyInResponse {
                    format,
                    column_formats,
                },
                b'H' => BackendMessage::CopyOutResponse {
                    format,
                    column_formats,
                },
                _ => BackendMessage::CopyBothResponse {
                    format,
                    column_formats,
                },
            }
        }
        b'D' => {
            let columns = Values::validate(&mut reader, "column count")?;
            reader.expect_end()?;
            BackendMessage::DataRow { columns }
        }
        b'I' => {
            reader.expect_end()?;
            BackendMessage::EmptyQueryResponse
        }
        b'E' => {
            let fields = ErrorFields::validate(&mut reader)?;
            reader.expect_end()?;
            BackendMessage::ErrorResponse { fields }
        }
        b'N' => {
            let fields = ErrorFields::validate(&mut reader)?;
            reader.expect_end()?;
            BackendMessage::NoticeResponse { fields }
        }
        b'V' => {
            let length = reader.take_i32()?;
            let value = match usize::try_from(length) {
                Ok(count) => Some(reader.take_bytes(count)?),
                Err(_) if length == -1 => None,
                Err(_) => {
                    return Err(ParseError::InvalidValue {
                        tag,
                        field: "result length",
                    });
                }
            };
            reader.expect_end()?;
            BackendMessage::FunctionCallResponse { value }
        }
        b'v' => {
            let newest_minor = reader.take_i32()?;
            let unsupported_options = CountedCStrs::validate(&mut reader, "option count")?;
            reader.expect_end()?;
            BackendMessage::NegotiateProtocolVersion {
                newest_minor,
                unsupported_options,
            }
        }
        b'n' => {
            reader.expect_end()?;
            BackendMessage::NoData
        }
        b'A' => {
            let process_id = reader.take_i32()?;
            let channel = reader.take_cstr()?;
            let payload = reader.take_cstr()?;
            reader.expect_end()?;
            BackendMessage::NotificationResponse {
                process_id,
                channel,
                payload,
            }
        }
        b't' => {
            let parameter_types =
                super::views::Oids::validate(&mut reader, "parameter type count")?;
            reader.expect_end()?;
            BackendMessage::ParameterDescription { parameter_types }
        }
        b'S' => {
            let name = reader.take_cstr()?;
            let value = reader.take_cstr()?;
            reader.expect_end()?;
            BackendMessage::ParameterStatus { name, value }
        }
        b'1' => {
            reader.expect_end()?;
            BackendMessage::ParseComplete
        }
        b's' => {
            reader.expect_end()?;
            BackendMessage::PortalSuspended
        }
        b'Z' => {
            let raw_status = reader.take_u8()?;
            let Some(status) = TransactionStatus::from_byte(raw_status) else {
                return Err(ParseError::InvalidValue {
                    tag,
                    field: "transaction status",
                });
            };
            reader.expect_end()?;
            BackendMessage::ReadyForQuery { status }
        }
        b'T' => {
            let fields = Fields::validate(&mut reader)?;
            reader.expect_end()?;
            BackendMessage::RowDescription { fields }
        }
        _ => return Err(ParseError::UnknownTag { tag }),
    };
    Ok(Some((message, consumed)))
}

fn parse_auth_request<'a>(reader: &mut Reader<'a>) -> Result<AuthRequest<'a>, ParseError> {
    let code = reader.take_i32()?;
    let request = match code {
        0 => {
            reader.expect_end()?;
            AuthRequest::Ok
        }
        2 => {
            reader.expect_end()?;
            AuthRequest::KerberosV5
        }
        3 => {
            reader.expect_end()?;
            AuthRequest::CleartextPassword
        }
        5 => {
            let salt = reader.take_bytes(4)?;
            reader.expect_end()?;
            AuthRequest::Md5Password {
                salt: [salt[0], salt[1], salt[2], salt[3]],
            }
        }
        6 => {
            reader.expect_end()?;
            AuthRequest::ScmCredential
        }
        7 => {
            reader.expect_end()?;
            AuthRequest::Gss
        }
        8 => AuthRequest::GssContinue {
            data: reader.take_rest(),
        },
        9 => {
            reader.expect_end()?;
            AuthRequest::Sspi
        }
        10 => {
            let mechanisms = CStrList::validate(reader)?;
            reader.expect_end()?;
            AuthRequest::Sasl { mechanisms }
        }
        11 => AuthRequest::SaslContinue {
            data: reader.take_rest(),
        },
        12 => AuthRequest::SaslFinal {
            data: reader.take_rest(),
        },
        _ => {
            return Err(ParseError::InvalidValue {
                tag: reader.tag(),
                field: "authentication code",
            });
        }
    };
    Ok(request)
}

impl BackendMessage<'_> {
    /// Wire tag byte of this message.
    #[must_use]
    pub const fn tag(&self) -> u8 {
        match self {
            Self::Authentication(_) => b'R',
            Self::BackendKeyData { .. } => b'K',
            Self::BindComplete => b'2',
            Self::CloseComplete => b'3',
            Self::CommandComplete { .. } => b'C',
            Self::CopyData { .. } => b'd',
            Self::CopyDone => b'c',
            Self::CopyInResponse { .. } => b'G',
            Self::CopyOutResponse { .. } => b'H',
            Self::CopyBothResponse { .. } => b'W',
            Self::DataRow { .. } => b'D',
            Self::EmptyQueryResponse => b'I',
            Self::ErrorResponse { .. } => b'E',
            Self::FunctionCallResponse { .. } => b'V',
            Self::NegotiateProtocolVersion { .. } => b'v',
            Self::NoData => b'n',
            Self::NoticeResponse { .. } => b'N',
            Self::NotificationResponse { .. } => b'A',
            Self::ParameterDescription { .. } => b't',
            Self::ParameterStatus { .. } => b'S',
            Self::ParseComplete => b'1',
            Self::PortalSuspended => b's',
            Self::ReadyForQuery { .. } => b'Z',
            Self::RowDescription { .. } => b'T',
        }
    }

    /// Encodes the message into `out`, returning the encoded size.
    ///
    /// # Errors
    /// [`EncodeError`] when `out` is too small or a field is invalid.
    pub fn encode(&self, out: &mut [u8]) -> Result<usize, EncodeError> {
        let mut writer = MessageWriter::tagged(out, self.tag())?;
        match self {
            Self::Authentication(request) => encode_auth_request(&mut writer, request)?,
            Self::BackendKeyData {
                process_id,
                secret_key,
            } => {
                writer.put_i32(*process_id)?;
                writer.put_bytes(secret_key)?;
            }
            Self::BindComplete
            | Self::CloseComplete
            | Self::CopyDone
            | Self::EmptyQueryResponse
            | Self::NoData
            | Self::ParseComplete
            | Self::PortalSuspended => {}
            Self::CommandComplete { tag } => {
                writer.put_cstr(tag.as_bytes())?;
            }
            Self::CopyData { data } => {
                writer.put_bytes(data)?;
            }
            Self::CopyInResponse {
                format,
                column_formats,
            }
            | Self::CopyOutResponse {
                format,
                column_formats,
            }
            | Self::CopyBothResponse {
                format,
                column_formats,
            } => {
                #[expect(clippy::cast_sign_loss, reason = "wire Int8 written as raw byte")]
                writer.put_u8(format.as_i8() as u8)?;
                put_count16(
                    &mut writer,
                    column_formats.len(),
                    "copy column format count",
                )?;
                for code in column_formats.iter() {
                    writer.put_i16(code.as_i16())?;
                }
            }
            Self::DataRow { columns } => {
                put_count16(&mut writer, columns.len(), "column count")?;
                for column in columns.iter() {
                    put_value(&mut writer, column)?;
                }
            }
            Self::ErrorResponse { fields } | Self::NoticeResponse { fields } => {
                for (field_type, value) in fields.iter() {
                    writer.put_u8(field_type)?;
                    writer.put_cstr(value.as_bytes())?;
                }
                writer.put_u8(0)?;
            }
            Self::FunctionCallResponse { value } => {
                put_value(&mut writer, *value)?;
            }
            Self::NegotiateProtocolVersion {
                newest_minor,
                unsupported_options,
            } => {
                writer.put_i32(*newest_minor)?;
                let Ok(count) = i32::try_from(unsupported_options.len()) else {
                    return Err(EncodeError::ValueTooLarge {
                        field: "option count",
                    });
                };
                writer.put_i32(count)?;
                for option in unsupported_options.iter() {
                    writer.put_cstr(option.as_bytes())?;
                }
            }
            Self::NotificationResponse {
                process_id,
                channel,
                payload,
            } => {
                writer.put_i32(*process_id)?;
                writer.put_cstr(channel.as_bytes())?;
                writer.put_cstr(payload.as_bytes())?;
            }
            Self::ParameterDescription { parameter_types } => {
                put_count16(&mut writer, parameter_types.len(), "parameter type count")?;
                for oid in parameter_types.iter() {
                    writer.put_u32(oid.0)?;
                }
            }
            Self::ParameterStatus { name, value } => {
                writer.put_cstr(name.as_bytes())?;
                writer.put_cstr(value.as_bytes())?;
            }
            Self::ReadyForQuery { status } => {
                writer.put_u8(status.as_byte())?;
            }
            Self::RowDescription { fields } => {
                put_count16(&mut writer, fields.len(), "field count")?;
                for field in fields.iter() {
                    put_field_description(&mut writer, &field)?;
                }
            }
        }
        writer.finish()
    }
}

fn encode_auth_request(
    writer: &mut MessageWriter<'_>,
    request: &AuthRequest<'_>,
) -> Result<(), EncodeError> {
    match request {
        AuthRequest::Ok => writer.put_i32(0).map(drop),
        AuthRequest::KerberosV5 => writer.put_i32(2).map(drop),
        AuthRequest::CleartextPassword => writer.put_i32(3).map(drop),
        AuthRequest::Md5Password { salt } => {
            writer.put_i32(5)?;
            writer.put_bytes(salt).map(drop)
        }
        AuthRequest::ScmCredential => writer.put_i32(6).map(drop),
        AuthRequest::Gss => writer.put_i32(7).map(drop),
        AuthRequest::GssContinue { data } => {
            writer.put_i32(8)?;
            writer.put_bytes(data).map(drop)
        }
        AuthRequest::Sspi => writer.put_i32(9).map(drop),
        AuthRequest::Sasl { mechanisms } => {
            writer.put_i32(10)?;
            for mechanism in mechanisms.iter() {
                writer.put_cstr(mechanism.as_bytes())?;
            }
            writer.put_u8(0).map(drop)
        }
        AuthRequest::SaslContinue { data } => {
            writer.put_i32(11)?;
            writer.put_bytes(data).map(drop)
        }
        AuthRequest::SaslFinal { data } => {
            writer.put_i32(12)?;
            writer.put_bytes(data).map(drop)
        }
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

fn put_value(writer: &mut MessageWriter<'_>, value: Option<&[u8]>) -> Result<(), EncodeError> {
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
    Ok(())
}

fn put_field_description(
    writer: &mut MessageWriter<'_>,
    field: &super::views::FieldDescription<'_>,
) -> Result<(), EncodeError> {
    writer.put_cstr(field.name.as_bytes())?;
    writer.put_u32(field.table_oid)?;
    writer.put_i16(field.column_attr)?;
    writer.put_u32(field.type_oid.0)?;
    writer.put_i16(field.type_size)?;
    writer.put_i32(field.type_modifier)?;
    writer.put_i16(field.format.as_i16())?;
    Ok(())
}

/// Encodes AuthenticationSASL from a mechanism slice (server role; the
/// borrowed [`AuthRequest::Sasl`] view only exists after a parse).
///
/// # Errors
/// [`EncodeError`] when `out` is too small or a mechanism embeds NUL.
pub fn encode_auth_sasl(out: &mut [u8], mechanisms: &[&[u8]]) -> Result<usize, EncodeError> {
    let mut writer = MessageWriter::tagged(out, b'R')?;
    writer.put_i32(10)?;
    for mechanism in mechanisms {
        writer.put_cstr(mechanism)?;
    }
    writer.put_u8(0)?;
    writer.finish()
}

/// Encodes ParameterDescription from an OID slice (server role).
///
/// # Errors
/// [`EncodeError`] when `out` is too small or the count exceeds Int16.
pub fn encode_parameter_description(out: &mut [u8], oids: &[Oid]) -> Result<usize, EncodeError> {
    let mut writer = MessageWriter::tagged(out, b't')?;
    put_count16(&mut writer, oids.len(), "parameter type count")?;
    for oid in oids {
        writer.put_u32(oid.0)?;
    }
    writer.finish()
}

/// Encodes one of the three COPY response messages from slices (server
/// role). `tag` must be `G`, `H`, or `W`.
fn encode_copy_response(
    out: &mut [u8],
    tag: u8,
    format: CopyFormat,
    column_formats: &[FormatCode],
) -> Result<usize, EncodeError> {
    let mut writer = MessageWriter::tagged(out, tag)?;
    #[expect(clippy::cast_sign_loss, reason = "wire Int8 written as raw byte")]
    writer.put_u8(format.as_i8() as u8)?;
    put_count16(
        &mut writer,
        column_formats.len(),
        "copy column format count",
    )?;
    for code in column_formats {
        writer.put_i16(code.as_i16())?;
    }
    writer.finish()
}

/// Encodes CopyInResponse (`G`).
///
/// # Errors
/// [`EncodeError`] when `out` is too small or the count exceeds Int16.
pub fn encode_copy_in_response(
    out: &mut [u8],
    format: CopyFormat,
    column_formats: &[FormatCode],
) -> Result<usize, EncodeError> {
    encode_copy_response(out, b'G', format, column_formats)
}

/// Encodes CopyOutResponse (`H`).
///
/// # Errors
/// [`EncodeError`] when `out` is too small or the count exceeds Int16.
pub fn encode_copy_out_response(
    out: &mut [u8],
    format: CopyFormat,
    column_formats: &[FormatCode],
) -> Result<usize, EncodeError> {
    encode_copy_response(out, b'H', format, column_formats)
}

/// Encodes CopyBothResponse (`W`).
///
/// # Errors
/// [`EncodeError`] when `out` is too small or the count exceeds Int16.
pub fn encode_copy_both_response(
    out: &mut [u8],
    format: CopyFormat,
    column_formats: &[FormatCode],
) -> Result<usize, EncodeError> {
    encode_copy_response(out, b'W', format, column_formats)
}

/// Encodes NegotiateProtocolVersion from slices (server role).
///
/// # Errors
/// [`EncodeError`] when `out` is too small or an option embeds NUL.
pub fn encode_negotiate_protocol_version(
    out: &mut [u8],
    newest_minor: i32,
    unsupported_options: &[&[u8]],
) -> Result<usize, EncodeError> {
    let mut writer = MessageWriter::tagged(out, b'v')?;
    writer.put_i32(newest_minor)?;
    let Ok(count) = i32::try_from(unsupported_options.len()) else {
        return Err(EncodeError::ValueTooLarge {
            field: "option count",
        });
    };
    writer.put_i32(count)?;
    for option in unsupported_options {
        writer.put_cstr(option)?;
    }
    writer.finish()
}

/// Streaming RowDescription encoder — the server-role construction path
/// (the borrowed [`Fields`] view only exists after a parse).
#[derive(Debug)]
pub struct RowDescriptionWriter<'a> {
    writer: MessageWriter<'a>,
    count_at: usize,
    count: usize,
}

impl<'a> RowDescriptionWriter<'a> {
    /// Starts a RowDescription message.
    ///
    /// # Errors
    /// [`EncodeError::BufferTooSmall`] when `out` cannot hold the header.
    pub fn begin(out: &'a mut [u8]) -> Result<Self, EncodeError> {
        let mut writer = MessageWriter::tagged(out, b'T')?;
        let count_at = writer.written();
        writer.put_i16(0)?;
        Ok(Self {
            writer,
            count_at,
            count: 0,
        })
    }

    /// Appends one field description.
    ///
    /// # Errors
    /// [`EncodeError`] when the buffer is too small, the name embeds NUL,
    /// or the field count exceeds Int16.
    pub fn field(
        &mut self,
        field: &super::views::FieldDescription<'_>,
    ) -> Result<&mut Self, EncodeError> {
        put_field_description(&mut self.writer, field)?;
        self.count += 1;
        if i16::try_from(self.count).is_err() {
            return Err(EncodeError::ValueTooLarge {
                field: "field count",
            });
        }
        Ok(self)
    }

    /// Patches the field count and message length.
    ///
    /// # Errors
    /// [`EncodeError::ValueTooLarge`] when the message exceeds `i32::MAX`.
    pub fn finish(mut self) -> Result<usize, EncodeError> {
        let Ok(count) = i16::try_from(self.count) else {
            return Err(EncodeError::ValueTooLarge {
                field: "field count",
            });
        };
        self.writer.patch_i16(self.count_at, count);
        self.writer.finish()
    }
}

/// Streaming DataRow encoder — the per-row server hot path. Zero
/// allocation: columns are written straight into the caller's buffer,
/// and [`DataRowWriter::reserve_column`] exposes the column slot for
/// in-place serialization.
#[derive(Debug)]
pub struct DataRowWriter<'a> {
    writer: MessageWriter<'a>,
    count_at: usize,
    count: usize,
}

impl<'a> DataRowWriter<'a> {
    /// Starts a DataRow message.
    ///
    /// # Errors
    /// [`EncodeError::BufferTooSmall`] when `out` cannot hold the header.
    pub fn begin(out: &'a mut [u8]) -> Result<Self, EncodeError> {
        let mut writer = MessageWriter::tagged(out, b'D')?;
        let count_at = writer.written();
        writer.put_i16(0)?;
        Ok(Self {
            writer,
            count_at,
            count: 0,
        })
    }

    /// Appends a NULL column.
    ///
    /// # Errors
    /// [`EncodeError`] when the buffer is too small or the column count
    /// exceeds Int16.
    pub fn null(&mut self) -> Result<&mut Self, EncodeError> {
        self.writer.put_i32(-1)?;
        self.bump_count()
    }

    /// Appends one column value.
    ///
    /// # Errors
    /// [`EncodeError`] when the buffer is too small, the value exceeds
    /// Int32, or the column count exceeds Int16.
    pub fn column(&mut self, value: &[u8]) -> Result<&mut Self, EncodeError> {
        put_value(&mut self.writer, Some(value))?;
        self.bump_count()
    }

    /// Reserves a `length`-byte column slot for in-place serialization
    /// and returns it.
    ///
    /// # Errors
    /// [`EncodeError`] when the buffer is too small, `length` exceeds
    /// Int32, or the column count exceeds Int16.
    pub fn reserve_column(&mut self, length: usize) -> Result<&mut [u8], EncodeError> {
        let Ok(wire_length) = i32::try_from(length) else {
            return Err(EncodeError::ValueTooLarge {
                field: "value length",
            });
        };
        self.writer.put_i32(wire_length)?;
        self.count += 1;
        if i16::try_from(self.count).is_err() {
            return Err(EncodeError::ValueTooLarge {
                field: "column count",
            });
        }
        self.writer.reserve(length)
    }

    fn bump_count(&mut self) -> Result<&mut Self, EncodeError> {
        self.count += 1;
        if i16::try_from(self.count).is_err() {
            return Err(EncodeError::ValueTooLarge {
                field: "column count",
            });
        }
        Ok(self)
    }

    /// Patches the column count and message length.
    ///
    /// # Errors
    /// [`EncodeError::ValueTooLarge`] when the message exceeds `i32::MAX`.
    pub fn finish(mut self) -> Result<usize, EncodeError> {
        let Ok(count) = i16::try_from(self.count) else {
            return Err(EncodeError::ValueTooLarge {
                field: "column count",
            });
        };
        self.writer.patch_i16(self.count_at, count);
        self.writer.finish()
    }
}

/// Streaming ErrorResponse / NoticeResponse encoder (server role).
#[derive(Debug)]
pub struct ErrorResponseWriter<'a> {
    writer: MessageWriter<'a>,
}

impl<'a> ErrorResponseWriter<'a> {
    /// Starts an ErrorResponse (`E`).
    ///
    /// # Errors
    /// [`EncodeError::BufferTooSmall`] when `out` cannot hold the header.
    pub fn error(out: &'a mut [u8]) -> Result<Self, EncodeError> {
        Ok(Self {
            writer: MessageWriter::tagged(out, b'E')?,
        })
    }

    /// Starts a NoticeResponse (`N`).
    ///
    /// # Errors
    /// [`EncodeError::BufferTooSmall`] when `out` cannot hold the header.
    pub fn notice(out: &'a mut [u8]) -> Result<Self, EncodeError> {
        Ok(Self {
            writer: MessageWriter::tagged(out, b'N')?,
        })
    }

    /// Appends one field pair (see [`error_field`] for type bytes).
    ///
    /// # Errors
    /// [`EncodeError`] when the buffer is too small or the value embeds
    /// NUL.
    pub fn field(&mut self, field_type: u8, value: &[u8]) -> Result<&mut Self, EncodeError> {
        self.writer.put_u8(field_type)?;
        self.writer.put_cstr(value)?;
        Ok(self)
    }

    /// Writes the field-list terminator and patches the length.
    ///
    /// # Errors
    /// [`EncodeError`] when the buffer is too small.
    pub fn finish(mut self) -> Result<usize, EncodeError> {
        self.writer.put_u8(0)?;
        self.writer.finish()
    }
}

/// Encodes a minimal spec-complete ErrorResponse: severity (localized +
/// non-localized), SQLSTATE code, and message — the three fields the
/// protocol requires in every ErrorResponse.
///
/// # Errors
/// [`EncodeError`] when `out` is too small or a field embeds NUL.
pub fn encode_error_response(
    out: &mut [u8],
    severity: &[u8],
    sqlstate: &[u8],
    message: &[u8],
) -> Result<usize, EncodeError> {
    let mut writer = ErrorResponseWriter::error(out)?;
    writer.field(error_field::SEVERITY, severity)?;
    writer.field(error_field::SEVERITY_NON_LOCALIZED, severity)?;
    writer.field(error_field::CODE, sqlstate)?;
    writer.field(error_field::MESSAGE, message)?;
    writer.finish()
}

// the test helpers build Vec<u8> frames; this crate carries no alloc
// dependency for its no_std tier, so the suite needs std, not just test
#[cfg(all(test, feature = "std"))]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use rstest::rstest;

    use super::*;
    use super::super::error::ParseError;
    use super::super::types::{CopyFormat, FormatCode, Oid, TransactionStatus};
    use super::super::views::FieldDescription;

    fn encode_backend(msg: &BackendMessage<'_>) -> Vec<u8> {
        let mut buf = vec![0u8; 1024];
        let written = msg.encode(&mut buf).expect("encode must succeed");
        buf[..written].to_vec()
    }

    #[rstest]
    #[case::idle(TransactionStatus::Idle, b'I')]
    #[case::in_transaction(TransactionStatus::InTransaction, b'T')]
    #[case::failed(TransactionStatus::Failed, b'E')]
    fn ready_for_query_all_statuses_round_trip(
        #[case] status: TransactionStatus,
        #[case] expected_byte: u8,
    ) {
        let msg = BackendMessage::ReadyForQuery { status };
        let bytes = encode_backend(&msg);
        assert_eq!(bytes[0], b'Z', "tag must be Z");
        assert_eq!(bytes[5], expected_byte, "status byte must match");
        let (parsed, _) = parse_backend(&bytes).expect("parse").expect("complete");
        let BackendMessage::ReadyForQuery {
            status: parsed_status,
        } = parsed
        else {
            panic!("expected ReadyForQuery");
        };
        assert_eq!(parsed_status, status);
    }

    #[test]
    fn parse_backend_returns_none_on_incomplete() {
        let bytes = [b'Z', 0, 0, 0];
        let result = parse_backend(&bytes).expect("no error on incomplete");
        assert!(result.is_none());
    }

    #[test]
    fn parse_backend_unknown_tag_returns_error() {
        let mut bytes = [0u8; 5];
        bytes[0] = b'?';
        bytes[1..5].copy_from_slice(&4i32.to_be_bytes());
        let result = parse_backend(&bytes);
        assert!(matches!(result, Err(ParseError::UnknownTag { tag: b'?' })));
    }

    #[test]
    fn parse_backend_bad_length_returns_error() {
        let mut bytes = [0u8; 5];
        bytes[0] = b'Z';
        bytes[1..5].copy_from_slice(&3i32.to_be_bytes());
        let result = parse_backend(&bytes);
        assert!(matches!(
            result,
            Err(ParseError::BadLength { tag: b'Z', .. })
        ));
    }

    #[test]
    fn invalid_transaction_status_byte_returns_error() {
        let mut bytes = vec![0u8; 6];
        bytes[0] = b'Z';
        bytes[1..5].copy_from_slice(&5i32.to_be_bytes());
        bytes[5] = b'X';
        let result = parse_backend(&bytes);
        assert!(matches!(
            result,
            Err(ParseError::InvalidValue { tag: b'Z', .. })
        ));
    }

    #[test]
    fn unknown_auth_code_99_returns_error() {
        let mut bytes = vec![0u8; 9];
        bytes[0] = b'R';
        bytes[1..5].copy_from_slice(&8i32.to_be_bytes());
        bytes[5..9].copy_from_slice(&99i32.to_be_bytes());
        let result = parse_backend(&bytes);
        assert!(matches!(
            result,
            Err(ParseError::InvalidValue { tag: b'R', .. })
        ));
    }

    #[test]
    fn authentication_ok_round_trips() {
        let msg = BackendMessage::Authentication(AuthRequest::Ok);
        let bytes = encode_backend(&msg);
        let (parsed, consumed) = parse_backend(&bytes).expect("parse").expect("complete");
        assert_eq!(consumed, bytes.len());
        assert!(matches!(
            parsed,
            BackendMessage::Authentication(AuthRequest::Ok)
        ));
    }

    #[test]
    fn authentication_md5_round_trips() {
        let msg = BackendMessage::Authentication(AuthRequest::Md5Password { salt: [1, 2, 3, 4] });
        let bytes = encode_backend(&msg);
        let (parsed, _) = parse_backend(&bytes).expect("parse").expect("complete");
        let BackendMessage::Authentication(AuthRequest::Md5Password { salt }) = parsed else {
            panic!("expected Md5Password");
        };
        assert_eq!(salt, [1, 2, 3, 4]);
    }

    #[test]
    fn backend_key_data_small_key_below_4_returns_error() {
        let mut bytes = vec![0u8; 10];
        bytes[0] = b'K';
        bytes[1..5].copy_from_slice(&9i32.to_be_bytes());
        bytes[5..9].copy_from_slice(&99i32.to_be_bytes());
        bytes[9] = 0;
        let result = parse_backend(&bytes);
        assert!(matches!(
            result,
            Err(ParseError::InvalidValue { tag: b'K', .. })
        ));
    }

    #[test]
    fn backend_key_data_round_trips() {
        let key: [u8; 4] = 67890i32.to_be_bytes();
        let msg = BackendMessage::BackendKeyData {
            process_id: 12345,
            secret_key: &key,
        };
        let bytes = encode_backend(&msg);
        let (parsed, _) = parse_backend(&bytes).expect("parse").expect("complete");
        let BackendMessage::BackendKeyData {
            process_id,
            secret_key,
        } = parsed
        else {
            panic!("expected BackendKeyData");
        };
        assert_eq!(process_id, 12345);
        assert_eq!(secret_key, key.as_slice());
    }

    #[rstest]
    #[case::bind_complete(BackendMessage::BindComplete, b'2')]
    #[case::close_complete(BackendMessage::CloseComplete, b'3')]
    #[case::parse_complete(BackendMessage::ParseComplete, b'1')]
    #[case::portal_suspended(BackendMessage::PortalSuspended, b's')]
    #[case::no_data(BackendMessage::NoData, b'n')]
    #[case::empty_query(BackendMessage::EmptyQueryResponse, b'I')]
    fn zero_body_backend_messages_round_trip(
        #[case] msg: BackendMessage<'static>,
        #[case] expected_tag: u8,
    ) {
        let bytes = encode_backend(&msg);
        assert_eq!(bytes[0], expected_tag);
        let (_, consumed) = parse_backend(&bytes).expect("parse").expect("complete");
        assert_eq!(consumed, bytes.len());
    }

    #[test]
    fn command_complete_round_trips_with_real_tag() {
        let msg = BackendMessage::CommandComplete {
            tag: PgStr::new(b"INSERT 0 1"),
        };
        let bytes = encode_backend(&msg);
        let (parsed, _) = parse_backend(&bytes).expect("parse").expect("complete");
        let BackendMessage::CommandComplete { tag } = parsed else {
            panic!("expected CommandComplete");
        };
        assert_eq!(tag, "INSERT 0 1");
    }

    #[test]
    fn error_response_multi_field_round_trips() {
        let mut buf = vec![0u8; 512];
        let written = encode_error_response(
            &mut buf,
            b"ERROR",
            b"42P01",
            b"relation \"users\" does not exist",
        )
        .expect("encode must succeed");
        let (parsed, consumed) = parse_backend(&buf[..written])
            .expect("parse")
            .expect("complete");
        assert_eq!(consumed, written);
        let BackendMessage::ErrorResponse { fields } = parsed else {
            panic!("expected ErrorResponse");
        };
        assert_eq!(
            fields.get(error_field::SEVERITY),
            Some(PgStr::new(b"ERROR"))
        );
        assert_eq!(fields.get(error_field::CODE), Some(PgStr::new(b"42P01")));
        assert_eq!(
            fields.get(error_field::MESSAGE),
            Some(PgStr::new(b"relation \"users\" does not exist"))
        );
    }

    #[test]
    fn error_response_writer_embedded_nul_returns_error() {
        let mut buf = vec![0u8; 128];
        let mut writer = ErrorResponseWriter::error(&mut buf).expect("begin");
        let result = writer.field(error_field::MESSAGE, b"bad\0nul");
        assert!(matches!(
            result,
            Err(super::super::error::EncodeError::InvalidValue { .. })
        ));
    }

    #[test]
    fn row_description_writer_patches_count_and_encodes_fields() {
        let field = FieldDescription {
            name: PgStr::new(b"user_id"),
            table_oid: 16384,
            column_attr: 1,
            type_oid: Oid(20),
            type_size: 8,
            type_modifier: -1,
            format: FormatCode::Binary,
        };
        let mut buf = vec![0u8; 256];
        let written = {
            let mut writer = RowDescriptionWriter::begin(&mut buf).expect("begin must succeed");
            writer.field(&field).expect("field must succeed");
            writer.finish().expect("finish must succeed")
        };

        let (parsed, _) = parse_backend(&buf[..written])
            .expect("parse")
            .expect("complete");
        let BackendMessage::RowDescription { fields } = parsed else {
            panic!("expected RowDescription");
        };
        assert_eq!(fields.len(), 1);
        let descs: Vec<FieldDescription<'_>> = fields.iter().collect();
        assert_eq!(descs[0].name, "user_id");
        assert_eq!(descs[0].type_oid, Oid(20));
        assert_eq!(descs[0].format, FormatCode::Binary);
    }

    #[test]
    fn data_row_writer_column_and_null_round_trip() {
        let mut buf = vec![0u8; 256];
        let written = {
            let mut writer = DataRowWriter::begin(&mut buf).expect("begin");
            writer.column(b"42").expect("column");
            writer.null().expect("null");
            writer.column(b"hello world").expect("column 3");
            writer.finish().expect("finish")
        };

        let (parsed, consumed) = parse_backend(&buf[..written])
            .expect("parse")
            .expect("complete");
        assert_eq!(consumed, written);
        let BackendMessage::DataRow { columns } = parsed else {
            panic!("expected DataRow");
        };
        assert_eq!(columns.len(), 3);
        let vals: Vec<Option<&[u8]>> = columns.iter().collect();
        assert_eq!(vals[0], Some(b"42".as_slice()));
        assert_eq!(vals[1], None);
        assert_eq!(vals[2], Some(b"hello world".as_slice()));
    }

    #[test]
    fn data_row_writer_reserve_column_in_place_fill() {
        let mut buf = vec![0u8; 256];
        let written = {
            let mut writer = DataRowWriter::begin(&mut buf).expect("begin");
            let slot = writer.reserve_column(4).expect("reserve");
            slot.copy_from_slice(&42i32.to_be_bytes());
            writer.finish().expect("finish")
        };

        let (parsed, _) = parse_backend(&buf[..written])
            .expect("parse")
            .expect("complete");
        let BackendMessage::DataRow { columns } = parsed else {
            panic!("expected DataRow");
        };
        let vals: Vec<Option<&[u8]>> = columns.iter().collect();
        assert_eq!(vals[0], Some(42i32.to_be_bytes().as_slice()));
    }

    #[test]
    fn data_row_writer_buffer_too_small_returns_error() {
        let mut buf = [0u8; 4];
        let result = DataRowWriter::begin(&mut buf);
        assert!(matches!(
            result,
            Err(super::super::error::EncodeError::BufferTooSmall { .. })
        ));
    }

    #[test]
    fn row_description_writer_buffer_too_small_returns_error() {
        let mut buf = [0u8; 4];
        let result = RowDescriptionWriter::begin(&mut buf);
        assert!(matches!(
            result,
            Err(super::super::error::EncodeError::BufferTooSmall { .. })
        ));
    }

    #[test]
    fn notification_response_round_trips() {
        let msg = BackendMessage::NotificationResponse {
            process_id: 54321,
            channel: PgStr::new(b"events"),
            payload: PgStr::new(b"user:created"),
        };
        let bytes = encode_backend(&msg);
        let (parsed, _) = parse_backend(&bytes).expect("parse").expect("complete");
        let BackendMessage::NotificationResponse {
            process_id,
            channel,
            payload,
        } = parsed
        else {
            panic!("expected NotificationResponse");
        };
        assert_eq!(process_id, 54321);
        assert_eq!(channel, "events");
        assert_eq!(payload, "user:created");
    }

    #[test]
    fn copy_in_response_round_trips() {
        let mut buf = vec![0u8; 64];
        let written = encode_copy_in_response(
            &mut buf,
            CopyFormat::Text,
            &[FormatCode::Text, FormatCode::Binary],
        )
        .expect("encode");
        let (parsed, _) = parse_backend(&buf[..written])
            .expect("parse")
            .expect("complete");
        let BackendMessage::CopyInResponse {
            format,
            column_formats,
        } = parsed
        else {
            panic!("expected CopyInResponse");
        };
        assert_eq!(format, CopyFormat::Text);
        assert_eq!(column_formats.len(), 2);
    }

    #[test]
    fn ssl_response_accept_encode_decode() {
        let mut buf = [0u8; 4];
        let written = SslResponse::Accept.encode(&mut buf).expect("encode");
        assert_eq!(written, 1);
        assert_eq!(buf[0], b'S');
        let (resp, _) = parse_ssl_response(&buf[..1])
            .expect("parse")
            .expect("complete");
        assert_eq!(resp, SslResponse::Accept);
    }

    #[test]
    fn ssl_response_refuse_encode_decode() {
        let mut buf = [0u8; 4];
        SslResponse::Refuse.encode(&mut buf).expect("encode");
        assert_eq!(buf[0], b'N');
        let (resp, _) = parse_ssl_response(&buf[..1])
            .expect("parse")
            .expect("complete");
        assert_eq!(resp, SslResponse::Refuse);
    }

    #[test]
    fn ssl_response_invalid_byte_returns_error() {
        let bytes = *b"X";
        let result = parse_ssl_response(&bytes);
        assert!(matches!(result, Err(ParseError::InvalidValue { .. })));
    }

    #[test]
    fn gssenc_response_accept_encode_decode() {
        let mut buf = [0u8; 4];
        GssEncResponse::Accept.encode(&mut buf).expect("encode");
        assert_eq!(buf[0], b'G');
        let (resp, _) = parse_gssenc_response(&buf[..1])
            .expect("parse")
            .expect("complete");
        assert_eq!(resp, GssEncResponse::Accept);
    }

    #[test]
    fn gssenc_response_invalid_byte_returns_error() {
        let bytes = *b"X";
        let result = parse_gssenc_response(&bytes);
        assert!(matches!(result, Err(ParseError::InvalidValue { .. })));
    }

    #[test]
    fn ssl_response_returns_none_on_empty_input() {
        let result = parse_ssl_response(&[]).expect("no error");
        assert!(result.is_none());
    }

    #[test]
    fn gssenc_response_returns_none_on_empty_input() {
        let result = parse_gssenc_response(&[]).expect("no error");
        assert!(result.is_none());
    }

    #[test]
    fn truncated_backend_parse_returns_none_until_complete() {
        let msg = BackendMessage::CommandComplete {
            tag: PgStr::new(b"SELECT 42"),
        };
        let full = encode_backend(&msg);
        for prefix_len in 0..full.len() {
            let result = parse_backend(&full[..prefix_len]).expect("no error");
            assert!(result.is_none(), "prefix {prefix_len} must return None");
        }
        let (_, consumed) = parse_backend(&full).expect("full parse").expect("complete");
        assert_eq!(consumed, full.len());
    }

    #[test]
    fn function_call_response_null_round_trips() {
        let msg = BackendMessage::FunctionCallResponse { value: None };
        let bytes = encode_backend(&msg);
        let (parsed, _) = parse_backend(&bytes).expect("parse").expect("complete");
        let BackendMessage::FunctionCallResponse { value } = parsed else {
            panic!("expected FunctionCallResponse");
        };
        assert!(value.is_none());
    }

    #[test]
    fn function_call_response_with_value_round_trips() {
        let payload = b"result_bytes";
        let msg = BackendMessage::FunctionCallResponse {
            value: Some(payload),
        };
        let bytes = encode_backend(&msg);
        let (parsed, _) = parse_backend(&bytes).expect("parse").expect("complete");
        let BackendMessage::FunctionCallResponse {
            value: Some(parsed_value),
        } = parsed
        else {
            panic!("expected FunctionCallResponse with value");
        };
        assert_eq!(parsed_value, payload);
    }

    #[test]
    fn negotiate_protocol_version_round_trips() {
        let mut buf = vec![0u8; 128];
        let written =
            encode_negotiate_protocol_version(&mut buf, 1, &[b"_pq_.trace", b"_pq_.debug"])
                .expect("encode");
        let (parsed, _) = parse_backend(&buf[..written])
            .expect("parse")
            .expect("complete");
        let BackendMessage::NegotiateProtocolVersion {
            newest_minor,
            unsupported_options,
        } = parsed
        else {
            panic!("expected NegotiateProtocolVersion");
        };
        assert_eq!(newest_minor, 1);
        assert_eq!(unsupported_options.len(), 2);
    }

    #[test]
    fn parameter_status_round_trips() {
        let msg = BackendMessage::ParameterStatus {
            name: PgStr::new(b"timezone"),
            value: PgStr::new(b"UTC"),
        };
        let bytes = encode_backend(&msg);
        let (parsed, _) = parse_backend(&bytes).expect("parse").expect("complete");
        let BackendMessage::ParameterStatus { name, value } = parsed else {
            panic!("expected ParameterStatus");
        };
        assert_eq!(name, "timezone");
        assert_eq!(value, "UTC");
    }

    #[test]
    fn parameter_description_round_trips() {
        let oids = [Oid(23), Oid(25), Oid(16), Oid(20)];
        let mut buf = vec![0u8; 64];
        let written = encode_parameter_description(&mut buf, &oids).expect("encode");
        let (parsed, _) = parse_backend(&buf[..written])
            .expect("parse")
            .expect("complete");
        let BackendMessage::ParameterDescription { parameter_types } = parsed else {
            panic!("expected ParameterDescription");
        };
        let collected: Vec<Oid> = parameter_types.iter().collect();
        assert_eq!(collected, oids.to_vec());
    }

    #[test]
    fn encode_auth_sasl_round_trips() {
        let mut buf = vec![0u8; 64];
        let written =
            encode_auth_sasl(&mut buf, &[b"SCRAM-SHA-256", b"SCRAM-SHA-256-PLUS"]).expect("encode");
        let (parsed, _) = parse_backend(&buf[..written])
            .expect("parse")
            .expect("complete");
        let BackendMessage::Authentication(AuthRequest::Sasl { mechanisms }) = parsed else {
            panic!("expected Sasl auth");
        };
        let mechs: Vec<PgStr<'_>> = mechanisms.iter().collect();
        assert_eq!(mechs.len(), 2);
        assert_eq!(mechs[0], "SCRAM-SHA-256");
    }
}
