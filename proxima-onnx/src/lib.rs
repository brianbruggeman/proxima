//! A sans-IO reader for the ONNX model file format: the protobuf-encoded
//! `ModelProto` message, parsed into faithful Rust structs.
//!
//! # Scope: parsing only
//!
//! This crate reads the graph -- ops, in order, with their attributes,
//! which initializers they read, what outputs they produce -- and stops
//! there. It never translates a `NodeProto` into a `proxima_tensor::Op`;
//! that is a separate, already-surveyed job, and mixing it in here would
//! produce a half-done version of both.
//!
//! # Tier split
//!
//! The parser core ([`parser`], [`decode`], [`messages`], [`types`],
//! [`error`], [`pipe`], [`sized`]) is `no_std + alloc`: it operates on
//! `&[u8]` and never performs IO. It compiles under
//! `--no-default-features` (`alloc` is the floor; `std` only adds
//! `thiserror`'s `std::error::Error` impl, forwards
//! `proxima-protocols/std`, and adds [`config`]). [`sized`] holds the
//! build-time floor constant ([`sized::MAX_LEN_DELIMITED_FIELD`]) that
//! [`parser::OnnxParser::new`] always uses; [`config`]'s
//! `OnnxParserConfig` (std-only, conflaguration-backed) seeds its runtime
//! default from that same constant and can override it per-process via
//! [`parser::OnnxParser::with_config`].
//!
//! # Layout source
//!
//! Field numbers and enum discriminants for the nine messages this crate
//! covers (`ModelProto`, `GraphProto`, `NodeProto`, `TensorProto`,
//! `ValueInfoProto`, `TypeProto`, `TensorShapeProto`, `AttributeProto`,
//! `OperatorSetIdProto`) are sourced from
//! `https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx.proto3`,
//! fetched 2026-08-18 from the `main` branch. See [`messages`]'s module
//! doc for the exact scope boundary (which fields on those nine messages
//! are decoded vs. gracefully skipped as unknown) and [`types`] for the
//! enum source citations.
//!
//! # Protobuf primitive
//!
//! Wire-level varint/tag/wire-type decoding is
//! `proxima_protocols::protobuf_wire` (schema-agnostic, sans-IO, already
//! in this workspace) -- this crate is the schema-aware layer on top of
//! it, per that module's own doc ("schema-aware decode is the caller's
//! job"). No `prost`: it is workspace-optional-only for OTLP and framed as
//! a benchmark reference in this repo's own manifest comments, not the
//! shipped decode path for a hand-rollable dozen message types.
//!
//! # Zero-copy
//!
//! Every string/bytes field borrows from the input buffer --
//! [`messages::TensorProto::raw_data`] in particular is a borrowed byte
//! range with its declared dtype and dims, never copied into an owned
//! `Vec<u8>` and never reinterpreted based on that dtype. Weights stay
//! bytes.
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

#[cfg(feature = "std")]
pub mod config;
pub mod decode;
pub mod error;
pub mod messages;
pub mod parser;
pub mod pipe;
pub mod sized;
pub mod types;

pub use decode::{ModelField, decode_model_field, decode_model_proto};
pub use error::OnnxError;
pub use messages::{
    AttributeProto, Dimension, DimensionValue, GraphProto, ModelProto, NodeProto,
    OperatorSetIdProto, TensorProto, TensorShapeProto, TypeProto, TypeProtoMap, TypeProtoTensor,
    TypeValue, ValueInfoProto,
};
pub use parser::{OnnxParser, PollOutcome};
pub use pipe::parse_complete;
pub use types::{AttributeType, DataType};

#[cfg(test)]
mod tests;
