//! A checkpoint dialect's tensor names as data: [`Pattern`] (one layer slot),
//! [`Spellings`] (one role, one or several suffixes) and [`Dialect`] (the whole
//! spelling table). The decoder family TOML in `families/decoder_only.toml` is
//! the one copy: `proxima-model-interop` parses it at the std tier, and this
//! crate's `build.rs` bakes its `gguf` dialect into [`Dialect::gguf`] for the
//! alloc tier, which has no TOML parser.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// The decoder family TOML, embedded so every consumer reads the same text.
pub const DECODER_ONLY_TOML: &str = include_str!("../../families/decoder_only.toml");

const SLOT: &str = "{L}";

/// A tensor name with one layer-index slot, e.g. `a.layers.{L}.fc1.weight`.
///
/// The one-slot invariant is checked at [`Pattern::parse`], so a stack that
/// could mean two indices never reaches the inferrer.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(try_from = "String", into = "String")]
pub struct Pattern {
    head: String,
    tail: String,
}

/// Why a pattern string was refused.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PatternError {
    #[error("pattern {pattern:?} must hold exactly one {{L}} slot, found {found}")]
    SlotCount { pattern: String, found: usize },
}

impl Pattern {
    /// Splits `source` around its single `{L}`.
    ///
    /// # Errors
    ///
    /// [`PatternError::SlotCount`] when `source` holds zero or several slots.
    #[must_use = "a refused pattern must be handled"]
    pub fn parse(source: &str) -> Result<Self, PatternError> {
        let found = source.matches(SLOT).count();
        match source.split_once(SLOT) {
            Some((head, tail)) if found == 1 => Ok(Self {
                head: head.to_string(),
                tail: tail.to_string(),
            }),
            _ => Err(PatternError::SlotCount {
                pattern: source.to_string(),
                found,
            }),
        }
    }

    /// The layer index when `name` is exactly this pattern.
    #[must_use]
    pub fn index(&self, name: &str) -> Option<u64> {
        let (index, rest) = self.split_index(name)?;
        (rest == self.tail).then_some(index)
    }

    /// The layer index when `name` starts with this pattern (a stack prefix).
    #[must_use]
    pub fn claims(&self, name: &str) -> Option<u64> {
        let (index, rest) = self.split_index(name)?;
        rest.starts_with(self.tail.as_str()).then_some(index)
    }

    /// The concrete name for layer `index`.
    #[must_use]
    pub fn render(&self, index: u64) -> String {
        alloc::format!("{}{index}{}", self.head, self.tail)
    }

    fn split_index<'name>(&self, name: &'name str) -> Option<(u64, &'name str)> {
        let rest = name.strip_prefix(self.head.as_str())?;
        let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
        let index = rest.get(..digits)?.parse().ok()?;
        Some((index, rest.get(digits..)?))
    }
}

impl TryFrom<String> for Pattern {
    type Error = PatternError;

    fn try_from(source: String) -> Result<Self, Self::Error> {
        Self::parse(&source)
    }
}

impl From<Pattern> for String {
    fn from(pattern: Pattern) -> Self {
        pattern.to_string()
    }
}

impl fmt::Display for Pattern {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}{SLOT}{}", self.head, self.tail)
    }
}

/// A role's tensor suffix: one spelling, or several that name the same function.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Spellings {
    One(String),
    Many(Vec<String>),
}

impl Spellings {
    /// Every spelling, in file order.
    #[must_use]
    pub fn all(&self) -> &[String] {
        match self {
            Self::One(single) => core::slice::from_ref(single),
            Self::Many(several) => several,
        }
    }
}

/// One checkpoint format's spelling of the family: where the layer stack lives,
/// the non-layer tensors it carries, and its role and metadata names.
///
/// `leaves` maps a per-layer leaf key to its suffix after `layer`; `globals`
/// maps a whole-model leaf key (`embedding`, `output_norm`, `output`) to its
/// full tensor name. [`Self::gguf`] is the alloc-tier default.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct Dialect {
    pub layer: Pattern,
    #[serde(default)]
    pub claims: Vec<String>,
    #[serde(default)]
    pub claim_prefixes: Vec<String>,
    #[serde(default)]
    pub roles: BTreeMap<String, Spellings>,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
    #[serde(default)]
    pub leaves: BTreeMap<String, String>,
    #[serde(default)]
    pub globals: BTreeMap<String, String>,
}

impl Default for Dialect {
    fn default() -> Self {
        Self::gguf()
    }
}

impl Dialect {
    /// The `gguf` dialect of `families/decoder_only.toml`, generated by
    /// `build.rs` so the alloc tier needs no TOML parser. Parse the same file
    /// with the `toml` crate and the two are equal.
    #[must_use]
    pub fn gguf() -> Self {
        include!(concat!(env!("OUT_DIR"), "/proxima_tensor_dialect_gguf.rs"))
    }

    /// Whether this is the default [`Self::gguf`] dialect, which a serialised
    /// descriptor omits so existing configs and digests stay byte-identical.
    #[must_use]
    pub fn is_gguf(&self) -> bool {
        *self == Self::gguf()
    }
}
