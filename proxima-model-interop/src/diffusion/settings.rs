//! `DiffusionSettings`: the conflaguration (env/TOML) face of the diffusion
//! graphs' lowering choices, the same split the serving settings use -- the
//! value lives where the graph builder reads it, this module owns the loader.

use std::str::FromStr;

use conflaguration::{Error, Settings, Validate, ValidationMessage};
use proxima_numeric::{DType, NumericFormat, PackedRows};
use serde::{Deserialize, Serialize};

use super::png::is_supported_sample;
use super::vae::ConvTaps;

const FORMAT_DEFAULTS: &str = include_str!("defaults.toml");

/// The diffusion lowering's and pipeline's switches.
///
/// `conv_taps` is how a 3x3 convolution's nine kernel positions reach the
/// reduce; the other fields are the text mask and the op-profile knobs, each
/// `PROXIMA_DIFFUSION_<FIELD>`. `three` is the default: one `[out_g, out_c, 9, in_g, in_c]` leaf,
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
    /// Whether the text encoder masks the tokenizer's declared pad token's keys,
    /// as the reference does; `false` A/Bs the mask off in one binary.
    #[setting(default = true)]
    pub key_pad_mask: bool,
    /// How many of the slowest ops an op profile ranks.
    #[setting(default = 10)]
    pub profile_top: u32,
    /// Comma-separated graph node ids whose dispatches a profiled decode
    /// captures (grid, entry, operand codecs); empty captures none.
    #[setting(default = "")]
    pub capture_nodes: String,
    /// Capture the slowest op and the first projection op of a profiled decode.
    #[setting(default = false)]
    pub capture_auto: bool,
    /// How the backbone's activations are stored between ops; `defaults.toml`
    /// states the default.
    pub dit_activation: NumericFormat,
    /// How the backbone's checkpoint weights are packed for the backend.
    pub dit_weights: NumericFormat,
    /// How the decoder's activations are stored between ops.
    pub vae_activation: NumericFormat,
    /// How the decoder's checkpoint weights are packed for the backend.
    pub vae_weights: NumericFormat,
    /// The decoder's group-norm epsilon as decimal text; read through [`Self::group_eps`].
    pub group_eps: String,
    /// The backbone's norm epsilon as decimal text; read through [`Self::norm_eps`].
    pub norm_eps: String,
    /// The factor the timestep is scaled by before it is embedded, as decimal text; read through
    /// [`Self::time_factor`].
    pub time_factor: String,
    /// The embedding's longest period, as decimal text; read through [`Self::max_period`].
    pub max_period: String,
    /// The format one PNG channel sample is stored in.
    pub png_sample: NumericFormat,
    /// The format a gather index leaf is stored in.
    pub gather_index: NumericFormat,
    /// The element format of the external reference tool's dumps, raw little-endian.
    pub reference_format: NumericFormat,
}

/// The format data `defaults.toml` carries, parsed once per default.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FormatDefaults {
    dit_activation: NumericFormat,
    dit_weights: NumericFormat,
    vae_activation: NumericFormat,
    vae_weights: NumericFormat,
    group_eps: String,
    norm_eps: String,
    time_factor: String,
    max_period: String,
    png_sample: NumericFormat,
    gather_index: NumericFormat,
    reference_format: NumericFormat,
}

impl FormatDefaults {
    fn shipped() -> Result<Self, toml::de::Error> {
        toml::from_str(FORMAT_DEFAULTS)
    }
}

impl Default for DiffusionSettings {
    fn default() -> Self {
        let formats = FormatDefaults::shipped().unwrap_or_default();
        Self {
            conv_taps: String::from("three"),
            key_pad_mask: true,
            profile_top: 10,
            capture_nodes: String::new(),
            capture_auto: false,
            dit_activation: formats.dit_activation,
            dit_weights: formats.dit_weights,
            vae_activation: formats.vae_activation,
            vae_weights: formats.vae_weights,
            group_eps: formats.group_eps,
            norm_eps: formats.norm_eps,
            time_factor: formats.time_factor,
            max_period: formats.max_period,
            png_sample: formats.png_sample,
            gather_index: formats.gather_index,
            reference_format: formats.reference_format,
        }
    }
}

impl DiffusionSettings {
    /// [`Settings::from_env`] with the fallback every caller wants: an
    /// unreadable environment is the default, not a failed image run. Keeps the
    /// trait out of every call site that only wants the switch.
    ///
    /// Layered: the shipped `defaults.toml` formats first, then every
    /// `PROXIMA_DIFFUSION_<FIELD>` that is set.
    #[must_use]
    pub fn resolved() -> Self {
        let mut settings = Self::default();
        match settings.override_from_env() {
            Ok(()) => settings,
            Err(_) => Self::default(),
        }
    }

    /// The shipped formats with every set `PROXIMA_DIFFUSION_<FIELD>` over
    /// them, validated: a format a graph's element type cannot express fails
    /// here, at load, instead of being papered over where a graph is built.
    ///
    /// # Errors
    ///
    /// The environment layer's error, or the validation failure.
    pub fn try_resolved() -> conflaguration::Result<Self> {
        let mut settings = Self::default();
        settings.override_from_env()?;
        settings.validate()?;
        Ok(settings)
    }

    /// The backbone's activation format as the element type its graph is built in.
    ///
    /// # Errors
    ///
    /// A message when the configured format is a block format, which a graph's
    /// element type cannot express.
    pub fn dit_activation_dtype(&self) -> Result<DType, String> {
        self.dit_activation
            .to_dtype()
            .ok_or_else(|| String::from("dit_activation must be a scalar format"))
    }

    /// The decoder's activation format as the element type its graph is built in.
    ///
    /// # Errors
    ///
    /// As [`Self::dit_activation_dtype`].
    pub fn vae_activation_dtype(&self) -> Result<DType, String> {
        self.vae_activation
            .to_dtype()
            .ok_or_else(|| String::from("vae_activation must be a scalar format"))
    }

    /// The gather index format as the element type an index leaf is declared in.
    ///
    /// # Errors
    ///
    /// A message when the configured format is not a scalar format.
    pub fn gather_index_dtype(&self) -> Result<DType, String> {
        self.gather_index
            .to_dtype()
            .ok_or_else(|| String::from("gather_index must be a scalar format"))
    }

    /// The reference dump format as the element type its rows are read in.
    ///
    /// # Errors
    ///
    /// A message when the configured format is not a scalar format.
    pub fn reference_format_dtype(&self) -> Result<DType, String> {
        self.reference_format
            .to_dtype()
            .ok_or_else(|| String::from("reference_format must be a scalar format"))
    }

    /// The group-norm epsilon, parsed into whatever element type the caller needs.
    ///
    /// # Errors
    ///
    /// A message when the configured text is not a number.
    pub fn group_eps<T: FromStr>(&self) -> Result<T, String> {
        self.group_eps
            .parse()
            .map_err(|_| format!("group_eps {:?} is not a number", self.group_eps))
    }

    /// The backbone's norm epsilon, parsed as [`Self::group_eps`] is.
    ///
    /// # Errors
    ///
    /// A message when the configured text is not a number.
    pub fn norm_eps<T: FromStr>(&self) -> Result<T, String> {
        self.norm_eps
            .parse()
            .map_err(|_| format!("norm_eps {:?} is not a number", self.norm_eps))
    }

    /// The timestep scale, parsed as [`Self::group_eps`] is.
    ///
    /// # Errors
    ///
    /// A message when the configured text is not a number.
    pub fn time_factor<T: FromStr>(&self) -> Result<T, String> {
        self.time_factor
            .parse()
            .map_err(|_| format!("time_factor {:?} is not a number", self.time_factor))
    }

    /// The embedding's longest period, parsed as [`Self::group_eps`] is.
    ///
    /// # Errors
    ///
    /// A message when the configured text is not a number.
    pub fn max_period<T: FromStr>(&self) -> Result<T, String> {
        self.max_period
            .parse()
            .map_err(|_| format!("max_period {:?} is not a number", self.max_period))
    }

    /// The cosine/sine embedding of `timestep * time_factor`: `channels / 2` frequencies from
    /// `max_period^(-i / (channels / 2))`, cosine first and sine second (ggml's order), computed
    /// with the standard library's methods.
    ///
    /// # Errors
    ///
    /// A message when either setting is not a number, `timestep` is not one element, or a
    /// value has no codec in `format`.
    pub fn timestep_embedding(
        &self,
        timestep: &PackedRows,
        channels: u32,
        format: &NumericFormat,
    ) -> Result<PackedRows, String> {
        let host = NumericFormat::host_element();
        let encode = |target: &NumericFormat, values: &[_]| {
            PackedRows::from_values(target, values).map_err(|why| why.to_string())
        };
        let decode = |rows: &PackedRows| {
            let mut out = Vec::new();
            rows.decode_all(&mut out).map_err(|why| why.to_string())?;
            Ok::<_, String>(out)
        };
        let single = |value| decode(&encode(&host, &[value])?);
        let half = channels / 2;
        let positions: Vec<_> = (0..half).map(|index| index as _).collect();
        let indices = decode(&encode(&host, &positions)?)?;
        let stamps = decode(timestep)?;
        let factors = single(self.time_factor()?)?;
        let periods = single(self.max_period()?)?;
        let counts = single(half as _)?;
        let ([stamp], [factor], [period], [count]) = (
            stamps.as_slice(),
            factors.as_slice(),
            periods.as_slice(),
            counts.as_slice(),
        ) else {
            return Err(String::from(
                "timestep, time_factor and max_period must each be one element",
            ));
        };
        let angles: Vec<_> = indices
            .iter()
            .map(|index| stamp * factor * period.powf(-index / count))
            .collect();
        let mut values = Vec::with_capacity(channels as usize);
        values.extend(angles.iter().map(|angle| angle.cos()));
        values.extend(angles.iter().map(|angle| angle.sin()));
        encode(format, &values)
    }

    fn png_sample_supported(&self) -> Result<(), String> {
        if is_supported_sample(&self.png_sample) {
            Ok(())
        } else {
            Err(String::from(
                "png_sample must be a one-byte unsigned integer format",
            ))
        }
    }

    /// The `conv_taps` field as the enum the graph builder takes.
    #[must_use]
    pub fn conv_taps(&self) -> ConvTaps {
        ConvTaps::from_name(&self.conv_taps).unwrap_or_default()
    }
}

fn host_number<T: FromStr>(parsed: Result<T, String>, probe: &mut Vec<T>) -> Result<(), String> {
    probe.push(parsed?);
    Ok(())
}

impl Validate for DiffusionSettings {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut probe = Vec::new();
        let host = PackedRows::from_values(&NumericFormat::host_element(), &[Default::default()])
            .and_then(|rows| rows.decode_all(&mut probe))
            .map_err(|why| why.to_string());
        let errors: Vec<ValidationMessage> = [
            ("dit_activation", self.dit_activation_dtype().map(drop)),
            ("vae_activation", self.vae_activation_dtype().map(drop)),
            ("gather_index", self.gather_index_dtype().map(drop)),
            ("reference_format", self.reference_format_dtype().map(drop)),
            ("png_sample", self.png_sample_supported()),
            ("host_element", host),
            ("group_eps", host_number(self.group_eps(), &mut probe)),
            ("norm_eps", host_number(self.norm_eps(), &mut probe)),
            ("time_factor", host_number(self.time_factor(), &mut probe)),
            ("max_period", host_number(self.max_period(), &mut probe)),
        ]
        .into_iter()
        .filter_map(|(field, found)| {
            found
                .err()
                .map(|message| ValidationMessage::new(field, message))
        })
        .collect();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(Error::Validation { errors })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shipped_formats_validate() {
        assert!(DiffusionSettings::default().validate().is_ok());
    }

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
    fn the_pipeline_switches_default_to_the_reference_behaviour() {
        let settings = DiffusionSettings::default();

        assert!(settings.key_pad_mask);
        assert_eq!(settings.profile_top, 10);
        assert!(settings.capture_nodes.is_empty());
        assert!(!settings.capture_auto);
    }

    #[test]
    fn the_shipped_data_file_parses_and_sets_weights_apart_from_activations() {
        let formats = FormatDefaults::shipped().expect("defaults.toml parses as format data");

        assert_ne!(formats.dit_weights, formats.dit_activation);
        assert_ne!(formats.vae_weights, formats.vae_activation);
    }

    #[test]
    fn default_settings_carry_the_shipped_formats() {
        let formats = FormatDefaults::shipped().expect("defaults.toml parses as format data");
        let settings = DiffusionSettings::default();

        assert_eq!(settings.dit_weights, formats.dit_weights);
        assert_eq!(settings.vae_activation, formats.vae_activation);
        assert!(settings.dit_activation_dtype().is_ok());
    }

    #[test]
    fn validate_rejects_png_samples_that_are_wide_or_signed() {
        let signed = NumericFormat::from_name("int8").expect("int8 is a layout name");
        let unfit: Vec<NumericFormat> = NumericFormat::ALL
            .into_iter()
            .chain([signed])
            .filter(|format| {
                format.size_bytes() == 2 || format.is_signed_integer() || format.is_float()
            })
            .collect();
        assert!(unfit.iter().any(|format| format.size_bytes() == 2));
        assert!(unfit.iter().any(NumericFormat::is_signed_integer));

        for format in unfit {
            let settings = DiffusionSettings {
                png_sample: format,
                ..DiffusionSettings::default()
            };
            assert!(
                settings.validate().is_err(),
                "png_sample {format:?} must be rejected"
            );
        }
    }

    #[test]
    fn validate_rejects_a_malformed_norm_eps() {
        let settings = DiffusionSettings {
            norm_eps: String::from("one millionth"),
            ..DiffusionSettings::default()
        };

        assert!(settings.validate().is_err());
    }

    #[test]
    fn the_timestep_angles_at_the_defaults_match_the_direct_formula() {
        let settings = DiffusionSettings::default();
        let host = NumericFormat::host_element();
        let timestep =
            PackedRows::from_values(&host, &[0.37]).expect("one host element encodes");
        let channels = 8u32;
        let half = (channels / 2) as usize;

        let embedding = settings
            .timestep_embedding(&timestep, channels, &host)
            .expect("the default settings embed a timestep");
        let mut decoded = Vec::new();
        embedding
            .decode_all(&mut decoded)
            .expect("the embedding decodes at the host element");

        let mut inputs = Vec::new();
        timestep
            .decode_all(&mut inputs)
            .expect("the timestep decodes at the host element");
        inputs.push(settings.time_factor().expect("default time_factor parses"));
        inputs.push(settings.max_period().expect("default max_period parses"));
        inputs.push(half as _);
        for index in 0..half {
            inputs.push(index as _);
        }
        let (head, indices) = inputs.split_at(4);
        let [stamp, factor, period, count] = head else {
            panic!("four scalar inputs were pushed");
        };
        let angles: Vec<_> = indices
            .iter()
            .map(|index| stamp * factor * period.powf(-index / count))
            .collect();
        let expected: Vec<_> = angles
            .iter()
            .map(|angle| angle.cos())
            .chain(angles.iter().map(|angle| angle.sin()))
            .collect();
        assert_eq!(decoded.len(), expected.len());
        for (found, wanted) in decoded.iter().zip(&expected) {
            assert_eq!(found.to_bits(), wanted.to_bits());
        }
    }

    #[test]
    fn both_names_parse_and_anything_else_falls_back() {
        assert_eq!(ConvTaps::from_name("nine"), Some(ConvTaps::Nine));
        assert_eq!(ConvTaps::from_name("three"), Some(ConvTaps::Three));
        assert_eq!(ConvTaps::from_name("folded"), None);
    }
}
