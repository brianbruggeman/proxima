//! A counter-based normal generator: Philox4x32-10 feeding the Box-Muller sine branch.
//!
//! Element `i` of a draw is a pure function of `(seed, offset, i)`, so the generator is
//! stateless per element, parallel by construction and fixed-cost. It reproduces the noise a
//! CUDA-style `randn` yields, which is what lets a run be seeded by a number instead of by a
//! dumped file. [`RngKind`] is the typed value of the `rng` config key; [`Philox`] is the
//! generator it selects.
//!
//! Teaching pointer: [`Philox::fill_normal`] writes into a caller-owned slice and allocates
//! nothing; [`Philox::normal`] is the `alloc` convenience over it. The transcendental calls go
//! through `libm`, so the module builds at the no_std tier.

use core::fmt;
use core::str::FromStr;

use thiserror::Error;

const MULTIPLIER_LOW: u32 = 0xD251_1F53;
const MULTIPLIER_HIGH: u32 = 0xCD9E_8D57;
const KEY_BUMP_LOW: u32 = 0x9E37_79B9;
const KEY_BUMP_HIGH: u32 = 0xBB67_AE85;
const ROUNDS: u32 = 10;
const TWO_POW_NEG_32: f32 = 2.328_306_4e-10;
const TWO_POW_NEG_32_TAU: f32 = TWO_POW_NEG_32 * 6.283_185_5;

/// The `rng` config key refused a value.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RngError {
    /// A generator the workspace names but does not implement.
    #[error("rng `{name}` is named but not implemented; use `philox`")]
    Unimplemented { name: &'static str },
    /// A string that names no generator.
    #[error("unknown rng `{given}`; the only implemented generator is `philox`")]
    Unknown { given: alloc_string::Owned },
}

#[cfg(feature = "alloc")]
mod alloc_string {
    pub type Owned = alloc::string::String;
}
#[cfg(not(feature = "alloc"))]
mod alloc_string {
    pub type Owned = &'static str;
}

/// Which generator the `rng` config key selects.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RngKind {
    /// Philox4x32-10 with the Box-Muller sine branch; the default.
    #[default]
    Philox,
}

impl RngKind {
    /// The config spelling of this generator.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Philox => "philox",
        }
    }

    /// Build this generator for `seed`, offset 0.
    #[must_use]
    pub const fn generator(self, seed: u64) -> Philox {
        match self {
            Self::Philox => Philox::new(seed),
        }
    }
}

impl fmt::Display for RngKind {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        out.write_str(self.name())
    }
}

impl FromStr for RngKind {
    type Err = RngError;

    fn from_str(text: &str) -> Result<Self, RngError> {
        match text {
            "philox" | "philox4x32_10_box_muller" => Ok(Self::Philox),
            "std_default" => Err(RngError::Unimplemented { name: "std_default" }),
            "cpu" => Err(RngError::Unimplemented { name: "cpu" }),
            #[cfg(feature = "alloc")]
            other => Err(RngError::Unknown { given: other.into() }),
            #[cfg(not(feature = "alloc"))]
            _ => Err(RngError::Unknown { given: "" }),
        }
    }
}

/// Philox4x32-10 with a Box-Muller normal transform: the seed and a per-draw offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Philox {
    seed: u64,
    offset: u32,
}

impl Philox {
    /// A generator at `seed`, offset 0.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { seed, offset: 0 }
    }

    /// Reseed and rewind the offset to 0.
    pub const fn manual_seed(&mut self, seed: u64) {
        self.seed = seed;
        self.offset = 0;
    }

    /// The offset the next draw will use.
    #[must_use]
    pub const fn offset(&self) -> u32 {
        self.offset
    }

    /// Fill `out` with standard normals; element `i` uses counter `[offset, 0, i, 0]`.
    /// The offset then advances by one, whatever the length.
    pub fn fill_normal(&mut self, out: &mut [f32]) {
        let key = [self.seed as u32, (self.seed >> 32) as u32];
        for (index, slot) in out.iter_mut().enumerate() {
            let words = philox_block([self.offset, 0, index as u32, 0], key);
            *slot = box_muller(words[0] as f32, words[1] as f32);
        }
        self.offset = self.offset.wrapping_add(1);
    }

    /// `count` standard normals as a new vector.
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn normal(&mut self, count: usize) -> alloc::vec::Vec<f32> {
        let mut out = alloc::vec![0.0_f32; count];
        self.fill_normal(&mut out);
        out
    }
}

fn philox_round(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let product_low = u64::from(counter[0]) * u64::from(MULTIPLIER_LOW);
    let product_high = u64::from(counter[2]) * u64::from(MULTIPLIER_HIGH);
    [
        (product_high >> 32) as u32 ^ counter[1] ^ key[0],
        product_high as u32,
        (product_low >> 32) as u32 ^ counter[3] ^ key[1],
        product_low as u32,
    ]
}

fn philox_block(mut counter: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
    for _ in 1..ROUNDS {
        counter = philox_round(counter, key);
        key[0] = key[0].wrapping_add(KEY_BUMP_LOW);
        key[1] = key[1].wrapping_add(KEY_BUMP_HIGH);
    }
    philox_round(counter, key)
}

fn box_muller(first: f32, second: f32) -> f32 {
    let uniform = first * TWO_POW_NEG_32 + TWO_POW_NEG_32 / 2.0;
    let angle = second * TWO_POW_NEG_32_TAU + TWO_POW_NEG_32_TAU / 2.0;
    libm::sqrtf(-2.0 * libm::logf(uniform)) * libm::sinf(angle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn philox_is_the_default_and_both_spellings_select_it() {
        assert_eq!(RngKind::default(), RngKind::Philox);
        assert_eq!("philox".parse::<RngKind>(), Ok(RngKind::Philox));
        assert_eq!("philox4x32_10_box_muller".parse::<RngKind>(), Ok(RngKind::Philox));
    }

    #[test]
    fn unimplemented_generators_are_refused_by_name() {
        for name in ["std_default", "cpu"] {
            let message = name.parse::<RngKind>().expect_err("refused").to_string();
            assert!(message.contains(name), "{message}");
        }
    }

    #[test]
    fn an_unknown_generator_is_refused_with_what_was_given() {
        let message = "mersenne".parse::<RngKind>().expect_err("refused").to_string();
        assert!(message.contains("mersenne"), "{message}");
    }

    #[test]
    fn each_draw_advances_the_offset_and_changes_the_stream() {
        let mut generator = Philox::new(42);
        let first = generator.normal(8);
        let second = generator.normal(8);
        assert_eq!(generator.offset(), 2);
        assert_ne!(first, second);
        generator.manual_seed(42);
        assert_eq!(generator.normal(8), first);
    }

    #[test]
    fn element_value_does_not_depend_on_draw_length() {
        let short = Philox::new(7).normal(4);
        let long = Philox::new(7).normal(64);
        assert_eq!(short[..], long[..4]);
    }
}
