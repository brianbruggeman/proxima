//! A tensor-name pattern with exactly one `{L}` layer slot.

use alloc::string::{String, ToString};
use core::fmt;

use serde::Deserialize;
use thiserror::Error;

const SLOT: &str = "{L}";

/// A tensor name with one layer-index slot, e.g. `a.layers.{L}.fc1.weight`.
///
/// The one-slot invariant is checked at [`Pattern::parse`], so a stack that
/// could mean two indices never reaches the inferrer.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(try_from = "String")]
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

impl fmt::Display for Pattern {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}{SLOT}{}", self.head, self.tail)
    }
}
