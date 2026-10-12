use std::io;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Guidance {
    Baseline,
    FixedLogits { strength: f32 },
    AdaptiveLogits { max_strength: f32 },
    FixedHidden { strength: f32 },
}

#[derive(Debug, Clone, Copy)]
pub struct Adaptive {
    pub top_probability: f32,
    pub second_probability: f32,
    pub margin: f32,
    pub effective_strength: f32,
}

impl Guidance {
    pub fn parse(method: &str, strength: f32) -> io::Result<Self> {
        if !strength.is_finite() || strength < 0.0 {
            return Err(io::Error::other("strength must be finite and nonnegative"));
        }
        match method {
            "baseline" => Ok(Self::Baseline),
            "fixed-logits" => Ok(Self::FixedLogits { strength }),
            "adaptive-logits" => Ok(Self::AdaptiveLogits {
                max_strength: strength,
            }),
            "fixed-hidden" => Ok(Self::FixedHidden { strength }),
            _ => Err(io::Error::other("unknown guidance method")),
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::FixedLogits { .. } => "fixed-logits",
            Self::AdaptiveLogits { .. } => "adaptive-logits",
            Self::FixedHidden { .. } => "fixed-hidden",
        }
    }

    pub const fn strength(self) -> Option<f32> {
        match self {
            Self::Baseline => None,
            Self::FixedLogits { strength } | Self::FixedHidden { strength } => Some(strength),
            Self::AdaptiveLogits { max_strength } => Some(max_strength),
        }
    }

    pub const fn uses_logit_reference(self) -> bool {
        matches!(self, Self::FixedLogits { .. } | Self::AdaptiveLogits { .. })
    }
}

pub fn adaptive(terminal: &[f32], max_strength: f32) -> io::Result<Adaptive> {
    if !max_strength.is_finite() || max_strength < 0.0 {
        return Err(io::Error::other(
            "adaptive cap must be finite and nonnegative",
        ));
    }
    if terminal.len() < 2 || terminal.iter().any(|value| !value.is_finite()) {
        return Err(io::Error::other(
            "adaptive margin requires at least two finite logits",
        ));
    }
    let maximum = terminal.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut denominator = 0.0f32;
    let mut first = 0.0f32;
    let mut second = 0.0f32;
    for value in terminal {
        let exponential = (*value - maximum).exp();
        denominator += exponential;
        if exponential > first {
            second = first;
            first = exponential;
        } else if exponential > second {
            second = exponential;
        }
    }
    let top_probability = first / denominator;
    let second_probability = second / denominator;
    let margin = top_probability - second_probability;
    Ok(Adaptive {
        top_probability,
        second_probability,
        margin,
        effective_strength: max_strength * (1.0 - margin),
    })
}
