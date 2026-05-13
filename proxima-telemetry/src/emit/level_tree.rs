//! Named hierarchical levels — the "names not numbers" surface (C6).
//!
//! Operators declare a tree of *named* levels (`security`, `security.auth`,
//! `security.auth.token`); each node is auto-assigned an order-preserving
//! [`Coord`] under its parent, so the dotted integers stay an internal detail.
//! Names resolve to coords (for config + filters) and enumerate for discovery
//! (the typed, listable surface `RUST_LOG` lacks). Reuses [`Level::custom`] for
//! the name + flat-severity band, so a named level still collapses to a flat
//! `Level` and orders with the built-ins. Tier T1 (no_std + alloc).

use alloc::vec::Vec;

use crate::emit::Coord;
use crate::level::Level;

/// A named hierarchical level: the reused flat [`Level`] (name + severity band)
/// paired with its packed tree [`Coord`].
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct HierLevel {
    /// The flat level — `name` is the handle, `severity` is the band.
    pub level: Level,
    /// The packed tree coordinate.
    pub coord: Coord,
}

impl HierLevel {
    /// Pair a name with a coordinate; the flat band is the coordinate's band.
    #[must_use]
    pub const fn new(name: &'static str, coord: Coord) -> Self {
        Self {
            level: Level::custom(name, coord.band() as u8),
            coord,
        }
    }

    /// The level's name (its handle).
    #[must_use]
    pub const fn name(&self) -> &'static str {
        self.level.name()
    }
}

/// A resolved tree of named levels.
#[derive(Clone, Debug, Default)]
pub struct LevelTree {
    levels: Vec<HierLevel>,
}

impl LevelTree {
    /// An empty tree (only the built-in flat levels resolve).
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Start declaring a named tree.
    #[must_use]
    pub fn builder() -> LevelTreeBuilder {
        LevelTreeBuilder::default()
    }

    /// Resolve a name to its coordinate.
    #[must_use]
    pub fn resolve(&self, name: &str) -> Option<Coord> {
        self.levels
            .iter()
            .find(|level| level.name() == name)
            .map(|level| level.coord)
    }

    /// Enumerate every declared name (discovery).
    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.levels.iter().map(HierLevel::name)
    }

    /// The declared levels.
    #[must_use]
    pub fn levels(&self) -> &[HierLevel] {
        &self.levels
    }
}

/// Why a [`LevelTree`] failed to build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LevelTreeError {
    /// A child named a parent that was not declared before it.
    UnknownParent {
        /// The missing parent name.
        parent: &'static str,
        /// The child that referenced it.
        name: &'static str,
    },
    /// A node would exceed the maximum coordinate depth.
    TooDeep {
        /// The node that overflowed.
        name: &'static str,
    },
}

impl core::fmt::Display for LevelTreeError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnknownParent { parent, name } => {
                write!(
                    formatter,
                    "level '{name}': unknown parent '{parent}' (declare it first)"
                )
            }
            Self::TooDeep { name } => write!(formatter, "level '{name}': exceeds max tree depth"),
        }
    }
}

impl core::error::Error for LevelTreeError {}

enum Decl {
    Family {
        name: &'static str,
        band: Level,
    },
    Child {
        parent: &'static str,
        name: &'static str,
    },
}

/// Fluent builder for a [`LevelTree`]. Declarations are resolved in order at
/// [`build`](LevelTreeBuilder::build), so a parent must be declared before its
/// children.
#[derive(Default)]
pub struct LevelTreeBuilder {
    decls: Vec<Decl>,
}

impl LevelTreeBuilder {
    /// Declare a top-level named family in a severity band (e.g. `security`
    /// under `ERROR`). Auto-assigned the next ordinal within that band.
    #[must_use]
    pub fn family(mut self, name: &'static str, band: Level) -> Self {
        self.decls.push(Decl::Family { name, band });
        self
    }

    /// Declare a child under a previously-declared parent name. Auto-assigned the
    /// next ordinal under that parent.
    #[must_use]
    pub fn child(mut self, parent: &'static str, name: &'static str) -> Self {
        self.decls.push(Decl::Child { parent, name });
        self
    }

    /// Resolve every declaration, assigning coordinates. Errors (never silently
    /// drops) on an unknown parent or a depth overflow.
    pub fn build(self) -> Result<LevelTree, LevelTreeError> {
        let mut levels: Vec<HierLevel> = Vec::with_capacity(self.decls.len());
        let mut next: Vec<(Coord, u16)> = Vec::new();
        for decl in self.decls {
            let (name, parent_coord) = match decl {
                Decl::Family { name, band } => (name, Coord::from(band)),
                Decl::Child { parent, name } => {
                    let parent_coord = levels
                        .iter()
                        .find(|level| level.name() == parent)
                        .map(|level| level.coord)
                        .ok_or(LevelTreeError::UnknownParent { parent, name })?;
                    (name, parent_coord)
                }
            };
            let ordinal = next_ordinal(&mut next, parent_coord);
            let coord = parent_coord
                .child(ordinal)
                .ok_or(LevelTreeError::TooDeep { name })?;
            levels.push(HierLevel::new(name, coord));
        }
        Ok(LevelTree { levels })
    }
}

/// Next 1-based ordinal under `parent`, tracked per parent across the build.
fn next_ordinal(table: &mut Vec<(Coord, u16)>, parent: Coord) -> u16 {
    if let Some(entry) = table.iter_mut().find(|(coord, _)| *coord == parent) {
        entry.1 += 1;
        entry.1
    } else {
        table.push((parent, 1));
        1
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::field_reassign_with_default,
        clippy::type_complexity,
        clippy::useless_vec,
        clippy::needless_range_loop,
        clippy::default_constructed_unit_structs
    )]

    use super::{LevelTree, LevelTreeError};
    use crate::level::Level;

    fn tree() -> LevelTree {
        LevelTree::builder()
            .family("security", Level::ERROR)
            .child("security", "security.auth")
            .child("security.auth", "security.auth.token")
            .family("audit", Level::WARN)
            .build()
            .unwrap()
    }

    // names resolve to coordinates, and the auto-assigned coordinates NEST so a
    // filter on the parent name catches the whole named subtree.
    #[test]
    fn named_levels_nest_into_a_filterable_subtree() {
        let tree = tree();
        let security = tree.resolve("security").unwrap();
        let token = tree.resolve("security.auth.token").unwrap();
        let auth = tree.resolve("security.auth").unwrap();

        assert!(token.in_subtree_of(security)); // token is under security
        assert!(auth.in_subtree_of(security));
        assert!(!tree.resolve("audit").unwrap().in_subtree_of(security)); // different family

        // each level collapses to its declared band severity.
        assert_eq!(security.band(), Level::ERROR.severity() as u16);
        assert_eq!(
            tree.resolve("audit").unwrap().band(),
            Level::WARN.severity() as u16
        );
    }

    // every declared name enumerates for discovery (the RUST_LOG-killer surface).
    #[test]
    fn names_enumerate_for_discovery() {
        let names: alloc::vec::Vec<_> = tree().names().collect();
        for expected in ["security", "security.auth", "security.auth.token", "audit"] {
            assert!(
                names.contains(&expected),
                "{expected} missing from {names:?}"
            );
        }
    }

    // a child before its parent is an explicit error, not a silent drop.
    #[test]
    fn unknown_parent_is_an_error() {
        let err = LevelTree::builder()
            .child("missing", "orphan")
            .build()
            .unwrap_err();
        assert_eq!(
            err,
            LevelTreeError::UnknownParent {
                parent: "missing",
                name: "orphan"
            }
        );
    }
}
