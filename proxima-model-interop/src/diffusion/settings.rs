//! `DiffusionSettings`: the conflaguration (env/TOML) face of the diffusion
//! graphs' lowering choices, the same split the serving settings use -- the
//! value lives where the graph builder reads it, this module owns the loader.

use conflaguration::Settings;
use serde::{Deserialize, Serialize};

use super::vae::ConvTaps;

/// The diffusion lowering's switches.
///
/// One field today: how a 3x3 convolution's nine kernel positions reach the
/// reduce. `three` is the default: one `[out_g, out_c, 9, in_g, in_c]` leaf,
/// one reduce over a tap-inner staged activation, parity with the nine-tap
/// incumbent proven on the reference decode (`vae::the_decoder_reproduces_the_
/// reference_image`) and on the end-to-end image.
///
/// `PROXIMA_DIFFUSION_CONV_TAPS=nine` restores the incumbent (nine `[out, in]`
/// leaves, nine reduces, eight adds) without a rebuild.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA_DIFFUSION")]
#[serde(default)]
pub struct DiffusionSettings {
    /// `"three"` (default) or `"nine"`; an unrecognised value keeps the
    /// default, matching conflaguration's lenient rule for every switch here.
    #[setting(default = "three")]
    pub conv_taps: String,
}

impl Default for DiffusionSettings {
    fn default() -> Self {
        Self {
            conv_taps: String::from("three"),
        }
    }
}

impl DiffusionSettings {
    /// [`Settings::from_env`] with the fallback every caller wants: an
    /// unreadable environment is the default, not a failed image run. Keeps the
    /// trait out of every call site that only wants the switch.
    #[must_use]
    pub fn resolved() -> Self {
        <Self as Settings>::from_env().unwrap_or_default()
    }

    /// The `conv_taps` field as the enum the graph builder takes.
    #[must_use]
    pub fn conv_taps(&self) -> ConvTaps {
        ConvTaps::from_name(&self.conv_taps).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The struct's `#[setting(default = ...)]` string and `ConvTaps::default`
    /// are two spellings of one default; this stops them drifting when the
    /// three-block form is promoted.
    #[test]
    fn the_declared_default_names_the_enum_default() {
        assert_eq!(
            DiffusionSettings::default().conv_taps(),
            ConvTaps::default()
        );
    }

    #[test]
    fn both_names_parse_and_anything_else_falls_back() {
        assert_eq!(ConvTaps::from_name("nine"), Some(ConvTaps::Nine));
        assert_eq!(ConvTaps::from_name("three"), Some(ConvTaps::Three));
        assert_eq!(ConvTaps::from_name("folded"), None);
    }
}
