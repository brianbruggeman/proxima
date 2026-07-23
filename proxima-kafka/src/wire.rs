//! Kafka Produce/Fetch/Metadata/ApiVersions v0 request+response body codec.
//!
//! `proxima_protocols::kafka` lifts the wire ENVELOPE only — the 4-byte
//! length prefix ([`proxima_protocols::kafka::parse_frame`]) and the
//! request header ([`proxima_protocols::kafka::parse_request_header`]:
//! api_key/api_version/correlation_id/client_id). It stops at the header's
//! body offset; nothing in proxima-protocols decodes what follows for a
//! given `api_key`. This module is that missing body layer — v0 layouts
//! only (the earliest, non-flexible, fixed-width-array wire shape for each
//! API), matching the facade scope this crate targets: a protocol-correct
//! broker facade, not a production Kafka broker speaking every negotiated
//! version. A real client's `ApiVersions` handshake sees v0-only ranges
//! advertised (see [`SUPPORTED_API_VERSIONS`]) and is expected to speak v0.
//!
//! Every non-flexible Kafka primitive follows one shape: `INT8/16/32/64` are
//! big-endian fixed width, `STRING` is a 2-byte length prefix (`-1` =
//! null), `BYTES` is a 4-byte length prefix (`-1` = null), `ARRAY` is a
//! 4-byte element count (`-1` = null, treated here as empty — a v0 client
//! never round-trips the null/empty distinction through this facade).

use bytes::Bytes;

/// Kafka API keys this facade recognizes on the wire. `Other` carries the
/// raw key so a caller can still render a well-formed `UNSUPPORTED_VERSION`
/// reply instead of dropping the connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiKey {
    Produce,
    Fetch,
    Metadata,
    ApiVersions,
    Other(i16),
}

impl ApiKey {
    #[must_use]
    pub const fn from_i16(value: i16) -> Self {
        match value {
            0 => Self::Produce,
            1 => Self::Fetch,
            3 => Self::Metadata,
            18 => Self::ApiVersions,
            other => Self::Other(other),
        }
    }

    #[must_use]
    pub const fn to_i16(self) -> i16 {
        match self {
            Self::Produce => 0,
            Self::Fetch => 1,
            Self::Metadata => 3,
            Self::ApiVersions => 18,
            Self::Other(value) => value,
        }
    }
}

/// The v0 API-version ranges this facade actually understands — the body
/// [`ApiKey::ApiVersions`] answers with, and the truth [`decode_request`]
/// enforces (any other advertised version is rejected as
/// [`WireError::UnsupportedVersion`]).
pub const SUPPORTED_API_VERSIONS: &[(i16, i16, i16)] = &[
    (ApiKey::Produce.to_i16(), 0, 0),
    (ApiKey::Fetch.to_i16(), 0, 0),
    (ApiKey::Metadata.to_i16(), 0, 0),
    (ApiKey::ApiVersions.to_i16(), 0, 0),
];

/// Real Kafka error codes this facade can emit — a small, honest subset,
/// not the full ~120-entry table.
pub mod error_code {
    pub const NONE: i16 = 0;
    pub const UNKNOWN_SERVER_ERROR: i16 = -1;
    pub const UNSUPPORTED_VERSION: i16 = 35;
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum WireError {
    #[error("buffer ended before a complete field")]
    Short,
    #[error("length {0} is invalid (must be -1 or >= 0)")]
    InvalidLength(i32),
    #[error("string was not valid utf-8")]
    Utf8,
    #[error("api_version {version} is not supported for api_key {api_key}")]
    UnsupportedVersion { api_key: i16, version: i16 },
    #[error("api_key {0} is not recognized by this facade")]
    UnknownApiKey(i16),
}

/// A cursor over a borrowed request body — the read half of the primitive
/// pair described in the module doc.
struct Reader<'a> {
    buffer: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn new(buffer: &'a [u8]) -> Self {
        Self {
            buffer,
            position: 0,
        }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], WireError> {
        let end = self.position + length;
        if end > self.buffer.len() {
            return Err(WireError::Short);
        }
        let slice = &self.buffer[self.position..end];
        self.position = end;
        Ok(slice)
    }

    fn read_i16(&mut self) -> Result<i16, WireError> {
        let bytes = self.take(2)?;
        Ok(i16::from_be_bytes([bytes[0], bytes[1]]))
    }

    fn read_i32(&mut self) -> Result<i32, WireError> {
        let bytes = self.take(4)?;
        Ok(i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_i64(&mut self) -> Result<i64, WireError> {
        let bytes = self.take(8)?;
        Ok(i64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    /// Nullable Kafka `STRING`: 2-byte length, `-1` means null. A null
    /// string decodes to empty — this facade's business types never carry
    /// the null/empty distinction (mirrors how `proxima_protocols::kafka`'s
    /// own `read_nullable_string` already treats the header's `client_id`).
    fn read_string(&mut self) -> Result<String, WireError> {
        let length = self.read_i16()?;
        if length == -1 {
            return Ok(String::new());
        }
        if length < -1 {
            return Err(WireError::InvalidLength(i32::from(length)));
        }
        let bytes = self.take(length as usize)?;
        core::str::from_utf8(bytes)
            .map(ToString::to_string)
            .map_err(|_error| WireError::Utf8)
    }

    /// Nullable Kafka `BYTES`: 4-byte length, `-1` means null (decoded as
    /// empty).
    fn read_bytes(&mut self) -> Result<Bytes, WireError> {
        let length = self.read_i32()?;
        if length == -1 {
            return Ok(Bytes::new());
        }
        if length < -1 {
            return Err(WireError::InvalidLength(length));
        }
        let bytes = self.take(length as usize)?;
        Ok(Bytes::copy_from_slice(bytes))
    }

    /// Nullable Kafka `ARRAY`: 4-byte element count, `-1` means null
    /// (decoded as empty, same convention as [`Self::read_string`]).
    fn read_array<T>(
        &mut self,
        mut item: impl FnMut(&mut Self) -> Result<T, WireError>,
    ) -> Result<Vec<T>, WireError> {
        let count = self.read_i32()?;
        if count == -1 {
            return Ok(Vec::new());
        }
        if count < -1 {
            return Err(WireError::InvalidLength(count));
        }
        let mut items = Vec::with_capacity(count as usize);
        for _ in 0..count {
            items.push(item(self)?);
        }
        Ok(items)
    }

    /// Kafka's request-side `Metadata` topics array is the one place a v0
    /// client distinguishes null (all topics) from present — every other
    /// array in this codec collapses that distinction, matching real
    /// clients' actual usage.
    fn read_nullable_array<T>(
        &mut self,
        mut item: impl FnMut(&mut Self) -> Result<T, WireError>,
    ) -> Result<Option<Vec<T>>, WireError> {
        let count = self.read_i32()?;
        if count == -1 {
            return Ok(None);
        }
        if count < -1 {
            return Err(WireError::InvalidLength(count));
        }
        let mut items = Vec::with_capacity(count as usize);
        for _ in 0..count {
            items.push(item(self)?);
        }
        Ok(Some(items))
    }
}

// `pub(crate)`: `crate::client::session` needs `write_i16`/`write_i32`/
// `write_string` to build the request HEADER (api_key/api_version/
// correlation_id/client_id) — the one piece of the wire this module
// itself never writes, since every `*Request`/`*Response` type here
// starts past the header.

pub(crate) fn write_i16(out: &mut Vec<u8>, value: i16) {
    out.extend_from_slice(&value.to_be_bytes());
}

pub(crate) fn write_i32(out: &mut Vec<u8>, value: i32) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn write_i64(out: &mut Vec<u8>, value: i64) {
    out.extend_from_slice(&value.to_be_bytes());
}

pub(crate) fn write_string(out: &mut Vec<u8>, value: &str) {
    let length = i16::try_from(value.len()).unwrap_or(i16::MAX);
    write_i16(out, length);
    out.extend_from_slice(&value.as_bytes()[..length as usize]);
}

fn write_bytes(out: &mut Vec<u8>, value: &[u8]) {
    let length = i32::try_from(value.len()).unwrap_or(i32::MAX);
    write_i32(out, length);
    out.extend_from_slice(&value[..length as usize]);
}

fn write_array<T>(out: &mut Vec<u8>, items: &[T], mut item: impl FnMut(&mut Vec<u8>, &T)) {
    write_i32(out, i32::try_from(items.len()).unwrap_or(i32::MAX));
    for entry in items {
        item(out, entry);
    }
}

// ---------------------------------------------------------------- Produce

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProducePartitionData {
    pub partition: i32,
    pub record_set: Bytes,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProduceTopicData {
    pub topic: String,
    pub partitions: Vec<ProducePartitionData>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProduceRequest {
    pub acks: i16,
    pub timeout_ms: i32,
    pub topics: Vec<ProduceTopicData>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProducePartitionResult {
    pub partition: i32,
    pub error_code: i16,
    pub base_offset: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProduceTopicResult {
    pub topic: String,
    pub partitions: Vec<ProducePartitionResult>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ProduceResponse {
    pub topics: Vec<ProduceTopicResult>,
}

fn decode_produce_request(reader: &mut Reader<'_>) -> Result<ProduceRequest, WireError> {
    let acks = reader.read_i16()?;
    let timeout_ms = reader.read_i32()?;
    let topics = reader.read_array(|reader| {
        let topic = reader.read_string()?;
        let partitions = reader.read_array(|reader| {
            let partition = reader.read_i32()?;
            let record_set = reader.read_bytes()?;
            Ok(ProducePartitionData {
                partition,
                record_set,
            })
        })?;
        Ok(ProduceTopicData { topic, partitions })
    })?;
    Ok(ProduceRequest {
        acks,
        timeout_ms,
        topics,
    })
}

fn encode_produce_response(out: &mut Vec<u8>, response: &ProduceResponse) {
    write_array(out, &response.topics, |out, topic| {
        write_string(out, &topic.topic);
        write_array(out, &topic.partitions, |out, partition| {
            write_i32(out, partition.partition);
            write_i16(out, partition.error_code);
            write_i64(out, partition.base_offset);
        });
    });
}

// ------------------------------------------------------------------ Fetch

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchPartitionData {
    pub partition: i32,
    pub fetch_offset: i64,
    pub max_bytes: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchTopicData {
    pub topic: String,
    pub partitions: Vec<FetchPartitionData>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchRequest {
    pub replica_id: i32,
    pub max_wait_ms: i32,
    pub min_bytes: i32,
    pub topics: Vec<FetchTopicData>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchPartitionResult {
    pub partition: i32,
    pub error_code: i16,
    pub high_watermark: i64,
    pub record_set: Bytes,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchTopicResult {
    pub topic: String,
    pub partitions: Vec<FetchPartitionResult>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FetchResponse {
    pub topics: Vec<FetchTopicResult>,
}

fn decode_fetch_request(reader: &mut Reader<'_>) -> Result<FetchRequest, WireError> {
    let replica_id = reader.read_i32()?;
    let max_wait_ms = reader.read_i32()?;
    let min_bytes = reader.read_i32()?;
    let topics = reader.read_array(|reader| {
        let topic = reader.read_string()?;
        let partitions = reader.read_array(|reader| {
            let partition = reader.read_i32()?;
            let fetch_offset = reader.read_i64()?;
            let max_bytes = reader.read_i32()?;
            Ok(FetchPartitionData {
                partition,
                fetch_offset,
                max_bytes,
            })
        })?;
        Ok(FetchTopicData { topic, partitions })
    })?;
    Ok(FetchRequest {
        replica_id,
        max_wait_ms,
        min_bytes,
        topics,
    })
}

fn encode_fetch_response(out: &mut Vec<u8>, response: &FetchResponse) {
    write_array(out, &response.topics, |out, topic| {
        write_string(out, &topic.topic);
        write_array(out, &topic.partitions, |out, partition| {
            write_i32(out, partition.partition);
            write_i16(out, partition.error_code);
            write_i64(out, partition.high_watermark);
            write_bytes(out, &partition.record_set);
        });
    });
}

// --------------------------------------------------------------- Metadata

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MetadataRequest {
    /// `None` means "every topic" (a null request array); `Some(topics)`
    /// (possibly empty) names exactly the requested topics.
    pub topics: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetadataBroker {
    pub node_id: i32,
    pub host: String,
    pub port: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetadataPartition {
    pub error_code: i16,
    pub partition_id: i32,
    pub leader: i32,
    pub replicas: Vec<i32>,
    pub isr: Vec<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetadataTopic {
    pub error_code: i16,
    pub topic: String,
    pub partitions: Vec<MetadataPartition>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MetadataResponse {
    pub brokers: Vec<MetadataBroker>,
    pub topics: Vec<MetadataTopic>,
}

fn decode_metadata_request(reader: &mut Reader<'_>) -> Result<MetadataRequest, WireError> {
    let topics = reader.read_nullable_array(Reader::read_string)?;
    Ok(MetadataRequest { topics })
}

fn encode_metadata_response(out: &mut Vec<u8>, response: &MetadataResponse) {
    write_array(out, &response.brokers, |out, broker| {
        write_i32(out, broker.node_id);
        write_string(out, &broker.host);
        write_i32(out, broker.port);
    });
    write_array(out, &response.topics, |out, topic| {
        write_i16(out, topic.error_code);
        write_string(out, &topic.topic);
        write_array(out, &topic.partitions, |out, partition| {
            write_i16(out, partition.error_code);
            write_i32(out, partition.partition_id);
            write_i32(out, partition.leader);
            write_array(out, &partition.replicas, |out, replica| {
                write_i32(out, *replica)
            });
            write_array(out, &partition.isr, |out, isr| write_i32(out, *isr));
        });
    });
}

// ------------------------------------------------------------ ApiVersions

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ApiVersionRange {
    pub api_key: i16,
    pub min_version: i16,
    pub max_version: i16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiVersionsResponse {
    pub error_code: i16,
    pub api_versions: Vec<ApiVersionRange>,
}

impl ApiVersionsResponse {
    /// The facade's own truth: exactly [`SUPPORTED_API_VERSIONS`], no error.
    #[must_use]
    pub fn supported() -> Self {
        Self {
            error_code: error_code::NONE,
            api_versions: SUPPORTED_API_VERSIONS
                .iter()
                .map(|&(api_key, min_version, max_version)| ApiVersionRange {
                    api_key,
                    min_version,
                    max_version,
                })
                .collect(),
        }
    }
}

fn encode_api_versions_response(out: &mut Vec<u8>, response: &ApiVersionsResponse) {
    write_i16(out, response.error_code);
    write_array(out, &response.api_versions, |out, range| {
        write_i16(out, range.api_key);
        write_i16(out, range.min_version);
        write_i16(out, range.max_version);
    });
}

// ---------------------------------------------------- client-side codec
//
// The broker facade above only ever needed request DECODE + response
// ENCODE. A real client needs exactly the inverse pair — request ENCODE +
// response DECODE — for the same four APIs. `KafkaClientUpstream`
// (`crate::client`) is this facade's only caller today, and scopes itself
// to Produce/Fetch (plus the ApiVersions handshake every real client
// opens with); Metadata's client-side encode/decode is not wired up yet —
// a caller that needs it can add `encode_metadata_request`/
// `decode_metadata_response` following the exact same shape as the pair
// below.

fn encode_produce_request(out: &mut Vec<u8>, request: &ProduceRequest) {
    write_i16(out, request.acks);
    write_i32(out, request.timeout_ms);
    write_array(out, &request.topics, |out, topic| {
        write_string(out, &topic.topic);
        write_array(out, &topic.partitions, |out, partition| {
            write_i32(out, partition.partition);
            write_bytes(out, &partition.record_set);
        });
    });
}

fn decode_produce_response(reader: &mut Reader<'_>) -> Result<ProduceResponse, WireError> {
    let topics = reader.read_array(|reader| {
        let topic = reader.read_string()?;
        let partitions = reader.read_array(|reader| {
            let partition = reader.read_i32()?;
            let error_code = reader.read_i16()?;
            let base_offset = reader.read_i64()?;
            Ok(ProducePartitionResult {
                partition,
                error_code,
                base_offset,
            })
        })?;
        Ok(ProduceTopicResult { topic, partitions })
    })?;
    Ok(ProduceResponse { topics })
}

fn encode_fetch_request(out: &mut Vec<u8>, request: &FetchRequest) {
    write_i32(out, request.replica_id);
    write_i32(out, request.max_wait_ms);
    write_i32(out, request.min_bytes);
    write_array(out, &request.topics, |out, topic| {
        write_string(out, &topic.topic);
        write_array(out, &topic.partitions, |out, partition| {
            write_i32(out, partition.partition);
            write_i64(out, partition.fetch_offset);
            write_i32(out, partition.max_bytes);
        });
    });
}

fn decode_fetch_response(reader: &mut Reader<'_>) -> Result<FetchResponse, WireError> {
    let topics = reader.read_array(|reader| {
        let topic = reader.read_string()?;
        let partitions = reader.read_array(|reader| {
            let partition = reader.read_i32()?;
            let error_code = reader.read_i16()?;
            let high_watermark = reader.read_i64()?;
            let record_set = reader.read_bytes()?;
            Ok(FetchPartitionResult {
                partition,
                error_code,
                high_watermark,
                record_set,
            })
        })?;
        Ok(FetchTopicResult { topic, partitions })
    })?;
    Ok(FetchResponse { topics })
}

fn decode_api_versions_response(reader: &mut Reader<'_>) -> Result<ApiVersionsResponse, WireError> {
    let error_code = reader.read_i16()?;
    let api_versions = reader.read_array(|reader| {
        let api_key = reader.read_i16()?;
        let min_version = reader.read_i16()?;
        let max_version = reader.read_i16()?;
        Ok(ApiVersionRange {
            api_key,
            min_version,
            max_version,
        })
    })?;
    Ok(ApiVersionsResponse {
        error_code,
        api_versions,
    })
}

// ------------------------------------------------------------- dispatch

/// One decoded request body, tagged by the API it came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestBody {
    Produce(ProduceRequest),
    Fetch(FetchRequest),
    Metadata(MetadataRequest),
    ApiVersions,
}

impl RequestBody {
    /// Client-side encode — the inverse of [`decode_request`]'s body half
    /// (framing + the header's `api_key`/`api_version`/`correlation_id`/
    /// `client_id` stay the caller's job, exactly like [`ResponseBody::encode`]
    /// leaves `correlation_id` to its own caller).
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match self {
            Self::Produce(request) => encode_produce_request(&mut out, request),
            Self::Fetch(request) => encode_fetch_request(&mut out, request),
            Self::Metadata(request) => encode_metadata_request(&mut out, request),
            Self::ApiVersions => {}
        }
        out
    }
}

fn encode_metadata_request(out: &mut Vec<u8>, request: &MetadataRequest) {
    match &request.topics {
        None => write_i32(out, -1),
        Some(topics) => write_array(out, topics, |out, topic| write_string(out, topic)),
    }
}

/// One encoded response body, tagged by the API it answers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResponseBody {
    Produce(ProduceResponse),
    Fetch(FetchResponse),
    Metadata(MetadataResponse),
    ApiVersions(ApiVersionsResponse),
}

impl ResponseBody {
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match self {
            Self::Produce(response) => encode_produce_response(&mut out, response),
            Self::Fetch(response) => encode_fetch_response(&mut out, response),
            Self::Metadata(response) => encode_metadata_response(&mut out, response),
            Self::ApiVersions(response) => encode_api_versions_response(&mut out, response),
        }
        out
    }
}

/// Client-side decode — the inverse of [`ResponseBody::encode`]. `api_key`
/// is the CLIENT's own record of what it asked for (a Kafka response
/// header carries only `correlation_id`, never `api_key` — the caller
/// must remember, exactly like [`crate::client::session::ClientSession`]
/// does via its `pending` field).
pub fn decode_response(api_key: i16, body: &[u8]) -> Result<ResponseBody, WireError> {
    let mut reader = Reader::new(body);
    match ApiKey::from_i16(api_key) {
        ApiKey::Produce => decode_produce_response(&mut reader).map(ResponseBody::Produce),
        ApiKey::Fetch => decode_fetch_response(&mut reader).map(ResponseBody::Fetch),
        ApiKey::Metadata => decode_metadata_response(&mut reader).map(ResponseBody::Metadata),
        ApiKey::ApiVersions => {
            decode_api_versions_response(&mut reader).map(ResponseBody::ApiVersions)
        }
        ApiKey::Other(other) => Err(WireError::UnknownApiKey(other)),
    }
}

fn decode_metadata_response(reader: &mut Reader<'_>) -> Result<MetadataResponse, WireError> {
    let brokers = reader.read_array(|reader| {
        let node_id = reader.read_i32()?;
        let host = reader.read_string()?;
        let port = reader.read_i32()?;
        Ok(MetadataBroker {
            node_id,
            host,
            port,
        })
    })?;
    let topics = reader.read_array(|reader| {
        let error_code = reader.read_i16()?;
        let topic = reader.read_string()?;
        let partitions = reader.read_array(|reader| {
            let error_code = reader.read_i16()?;
            let partition_id = reader.read_i32()?;
            let leader = reader.read_i32()?;
            let replicas = reader.read_array(Reader::read_i32)?;
            let isr = reader.read_array(Reader::read_i32)?;
            Ok(MetadataPartition {
                error_code,
                partition_id,
                leader,
                replicas,
                isr,
            })
        })?;
        Ok(MetadataTopic {
            error_code,
            topic,
            partitions,
        })
    })?;
    Ok(MetadataResponse { brokers, topics })
}

/// Decode `body` (the bytes past
/// [`proxima_protocols::kafka::parse_request_header`]'s body offset) per
/// `api_key`/`api_version`. Only [`SUPPORTED_API_VERSIONS`] decode; anything
/// else is [`WireError::UnsupportedVersion`] / [`WireError::UnknownApiKey`]
/// so the driver can render a real `UNSUPPORTED_VERSION` reply instead of
/// guessing at an unrecognized layout.
pub fn decode_request(
    api_key: i16,
    api_version: i16,
    body: &[u8],
) -> Result<RequestBody, WireError> {
    let key = ApiKey::from_i16(api_key);
    let supported = SUPPORTED_API_VERSIONS
        .iter()
        .find(|&&(candidate, _, _)| candidate == api_key);
    let Some(&(_, min_version, max_version)) = supported else {
        return Err(WireError::UnknownApiKey(api_key));
    };
    if api_version < min_version || api_version > max_version {
        return Err(WireError::UnsupportedVersion {
            api_key,
            version: api_version,
        });
    }

    let mut reader = Reader::new(body);
    match key {
        ApiKey::Produce => decode_produce_request(&mut reader).map(RequestBody::Produce),
        ApiKey::Fetch => decode_fetch_request(&mut reader).map(RequestBody::Fetch),
        ApiKey::Metadata => decode_metadata_request(&mut reader).map(RequestBody::Metadata),
        ApiKey::ApiVersions => Ok(RequestBody::ApiVersions),
        ApiKey::Other(_) => Err(WireError::UnknownApiKey(api_key)),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn produce_request_wire() -> Vec<u8> {
        // acks=1, timeout_ms=1500, one topic "orders" with one partition 0
        // carrying a 4-byte opaque record_set.
        let mut wire = Vec::new();
        write_i16(&mut wire, 1);
        write_i32(&mut wire, 1500);
        write_array(&mut wire, &[0_u8], |out, _| {
            write_string(out, "orders");
            write_array(out, &[0_u8], |out, _| {
                write_i32(out, 0);
                write_bytes(out, b"data");
            });
        });
        wire
    }

    #[test]
    fn decode_produce_request_v0_round_trips_a_real_shaped_request() {
        let wire = produce_request_wire();
        let decoded = decode_request(ApiKey::Produce.to_i16(), 0, &wire).expect("decode");
        let RequestBody::Produce(request) = decoded else {
            panic!("expected Produce");
        };
        assert_eq!(request.acks, 1);
        assert_eq!(request.timeout_ms, 1500);
        assert_eq!(request.topics.len(), 1);
        assert_eq!(request.topics[0].topic, "orders");
        assert_eq!(request.topics[0].partitions[0].partition, 0);
        assert_eq!(
            request.topics[0].partitions[0].record_set,
            Bytes::from_static(b"data")
        );
    }

    #[test]
    fn produce_response_encodes_the_documented_v0_layout() {
        let response = ProduceResponse {
            topics: vec![ProduceTopicResult {
                topic: "orders".to_string(),
                partitions: vec![ProducePartitionResult {
                    partition: 0,
                    error_code: error_code::NONE,
                    base_offset: 42,
                }],
            }],
        };
        let encoded = ResponseBody::Produce(response).encode();

        let mut reader = Reader::new(&encoded);
        let topics = reader
            .read_array(|reader| {
                let topic = reader.read_string()?;
                let partitions = reader.read_array(|reader| {
                    let partition = reader.read_i32()?;
                    let error_code = reader.read_i16()?;
                    let base_offset = reader.read_i64()?;
                    Ok((partition, error_code, base_offset))
                })?;
                Ok((topic, partitions))
            })
            .expect("re-decode with the raw reader");
        assert_eq!(topics, vec![("orders".to_string(), vec![(0, 0, 42)])]);
    }

    #[test]
    fn decode_fetch_request_v0_round_trips_a_real_shaped_request() {
        let mut wire = Vec::new();
        write_i32(&mut wire, -1); // replica_id: -1 = a real client (not a follower broker)
        write_i32(&mut wire, 100); // max_wait_ms
        write_i32(&mut wire, 1); // min_bytes
        write_array(&mut wire, &[0_u8], |out, _| {
            write_string(out, "orders");
            write_array(out, &[0_u8], |out, _| {
                write_i32(out, 0);
                write_i64(out, 0);
                write_i32(out, 1_048_576);
            });
        });

        let decoded = decode_request(ApiKey::Fetch.to_i16(), 0, &wire).expect("decode");
        let RequestBody::Fetch(request) = decoded else {
            panic!("expected Fetch");
        };
        assert_eq!(request.replica_id, -1);
        assert_eq!(request.max_wait_ms, 100);
        assert_eq!(request.topics[0].partitions[0].fetch_offset, 0);
    }

    #[test]
    fn decode_metadata_request_distinguishes_null_from_empty_topic_array() {
        let mut all_topics = Vec::new();
        write_i32(&mut all_topics, -1);
        let decoded = decode_request(ApiKey::Metadata.to_i16(), 0, &all_topics).expect("decode");
        assert_eq!(
            decoded,
            RequestBody::Metadata(MetadataRequest { topics: None })
        );

        let mut one_topic = Vec::new();
        write_array(&mut one_topic, &["orders".to_string()], |out, topic| {
            write_string(out, topic);
        });
        let decoded = decode_request(ApiKey::Metadata.to_i16(), 0, &one_topic).expect("decode");
        assert_eq!(
            decoded,
            RequestBody::Metadata(MetadataRequest {
                topics: Some(vec!["orders".to_string()])
            })
        );
    }

    #[test]
    fn api_versions_response_advertises_exactly_the_supported_table() {
        let response = ApiVersionsResponse::supported();
        assert_eq!(response.error_code, error_code::NONE);
        assert_eq!(response.api_versions.len(), SUPPORTED_API_VERSIONS.len());
        assert!(
            response
                .api_versions
                .iter()
                .all(|range| range.min_version == 0 && range.max_version == 0)
        );
    }

    #[test]
    fn unsupported_version_is_rejected_not_misparsed() {
        let error = decode_request(ApiKey::Produce.to_i16(), 9, b"").unwrap_err();
        assert_eq!(
            error,
            WireError::UnsupportedVersion {
                api_key: ApiKey::Produce.to_i16(),
                version: 9,
            }
        );
    }

    #[test]
    fn unknown_api_key_is_rejected() {
        let error = decode_request(255, 0, b"").unwrap_err();
        assert_eq!(error, WireError::UnknownApiKey(255));
    }

    #[test]
    fn truncated_produce_request_reports_short_not_a_panic() {
        let mut wire = produce_request_wire();
        wire.truncate(wire.len() - 2);
        let error = decode_request(ApiKey::Produce.to_i16(), 0, &wire).unwrap_err();
        assert_eq!(error, WireError::Short);
    }

    #[test]
    fn produce_request_encode_then_server_side_decode_round_trips() {
        let request = ProduceRequest {
            acks: 1,
            timeout_ms: 500,
            topics: vec![ProduceTopicData {
                topic: "orders".to_string(),
                partitions: vec![ProducePartitionData {
                    partition: 2,
                    record_set: Bytes::from_static(b"payload"),
                }],
            }],
        };
        let encoded = RequestBody::Produce(request.clone()).encode();
        let decoded = decode_request(ApiKey::Produce.to_i16(), 0, &encoded).expect("decode");
        assert_eq!(decoded, RequestBody::Produce(request));
    }

    #[test]
    fn fetch_request_encode_then_server_side_decode_round_trips() {
        let request = FetchRequest {
            replica_id: -1,
            max_wait_ms: 250,
            min_bytes: 1,
            topics: vec![FetchTopicData {
                topic: "orders".to_string(),
                partitions: vec![FetchPartitionData {
                    partition: 0,
                    fetch_offset: 4,
                    max_bytes: 65536,
                }],
            }],
        };
        let encoded = RequestBody::Fetch(request.clone()).encode();
        let decoded = decode_request(ApiKey::Fetch.to_i16(), 0, &encoded).expect("decode");
        assert_eq!(decoded, RequestBody::Fetch(request));
    }

    #[test]
    fn produce_response_server_side_encode_then_client_side_decode_round_trips() {
        let response = ProduceResponse {
            topics: vec![ProduceTopicResult {
                topic: "orders".to_string(),
                partitions: vec![ProducePartitionResult {
                    partition: 0,
                    error_code: error_code::NONE,
                    base_offset: 17,
                }],
            }],
        };
        let encoded = ResponseBody::Produce(response.clone()).encode();
        let decoded = decode_response(ApiKey::Produce.to_i16(), &encoded).expect("decode");
        assert_eq!(decoded, ResponseBody::Produce(response));
    }

    #[test]
    fn fetch_response_server_side_encode_then_client_side_decode_round_trips() {
        let response = FetchResponse {
            topics: vec![FetchTopicResult {
                topic: "orders".to_string(),
                partitions: vec![FetchPartitionResult {
                    partition: 0,
                    error_code: error_code::NONE,
                    high_watermark: 3,
                    record_set: Bytes::from_static(b"hello"),
                }],
            }],
        };
        let encoded = ResponseBody::Fetch(response.clone()).encode();
        let decoded = decode_response(ApiKey::Fetch.to_i16(), &encoded).expect("decode");
        assert_eq!(decoded, ResponseBody::Fetch(response));
    }

    #[test]
    fn api_versions_response_encode_then_client_side_decode_round_trips() {
        let response = ApiVersionsResponse::supported();
        let encoded = ResponseBody::ApiVersions(response.clone()).encode();
        let decoded = decode_response(ApiKey::ApiVersions.to_i16(), &encoded).expect("decode");
        assert_eq!(decoded, ResponseBody::ApiVersions(response));
    }

    #[test]
    fn metadata_response_server_side_encode_then_client_side_decode_round_trips() {
        let response = MetadataResponse {
            brokers: vec![MetadataBroker {
                node_id: 0,
                host: "localhost".to_string(),
                port: 9092,
            }],
            topics: vec![MetadataTopic {
                error_code: error_code::NONE,
                topic: "orders".to_string(),
                partitions: vec![MetadataPartition {
                    error_code: error_code::NONE,
                    partition_id: 0,
                    leader: 0,
                    replicas: vec![0],
                    isr: vec![0],
                }],
            }],
        };
        let encoded = ResponseBody::Metadata(response.clone()).encode();
        let decoded = decode_response(ApiKey::Metadata.to_i16(), &encoded).expect("decode");
        assert_eq!(decoded, ResponseBody::Metadata(response));
    }
}
