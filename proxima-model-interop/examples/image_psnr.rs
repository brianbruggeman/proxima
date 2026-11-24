//! PSNR between two PNGs: the sd.cpp reference image and ours (PROXNUM-588 AC7).
//!
//!   cargo run --release -p proxima-model-interop --example image_psnr -- ref.png omega.png
//!
//! Prints both images' dimensions, the RMS difference and the PSNR in dB against
//! the full `1.0` scale the spec's gates are stated on, and exits non-zero below
//! the 22.0 dB gate.

use proxima_model_interop::diffusion::{DiffusionSettings, PngImage, full_scale, read_png};
use proxima_numeric::NumericFormat;

fn main() {
    let mut arguments = std::env::args().skip(1);
    let (Some(reference), Some(actual)) = (arguments.next(), arguments.next()) else {
        eprintln!("usage: image_psnr <reference.png> <actual.png>");
        std::process::exit(64);
    };
    let sample = DiffusionSettings::resolved().png_sample;
    let reference = match read_png(reference.as_ref(), &sample) {
        Ok(image) => image,
        Err(why) => {
            eprintln!("{why}");
            std::process::exit(1);
        }
    };
    let actual = match read_png(actual.as_ref(), &sample) {
        Ok(image) => image,
        Err(why) => {
            eprintln!("{why}");
            std::process::exit(1);
        }
    };
    eprintln!(
        "reference {}x{}x{}, actual {}x{}x{}",
        reference.width, reference.height, reference.channels,
        actual.width, actual.height, actual.channels
    );
    match psnr(&actual, &reference, &sample) {
        Ok(true) => {}
        Ok(false) => {
            eprintln!("below the 22.0 dB gate");
            std::process::exit(2);
        }
        Err(why) => {
            eprintln!("{why}");
            std::process::exit(1);
        }
    }
}

fn psnr(actual: &PngImage, reference: &PngImage, sample: &NumericFormat) -> Result<bool, String> {
    if (actual.width, actual.height, actual.channels)
        != (reference.width, reference.height, reference.channels)
    {
        return Err(String::from("images differ in size or channel count"));
    }
    let mut scale = Vec::new();
    full_scale(sample)?
        .decode_all(&mut scale)
        .map_err(|why| why.to_string())?;
    let mut got = Vec::new();
    actual
        .pixels
        .decode_all(&mut got)
        .map_err(|why| why.to_string())?;
    let mut want = Vec::new();
    reference
        .pixels
        .decode_all(&mut want)
        .map_err(|why| why.to_string())?;
    let mut error = scale[0] * 0.0;
    let mut count = error;
    for (&left, &right) in got.iter().zip(&want) {
        let difference = left / scale[0] - right / scale[0];
        error += difference * difference;
        count += 1.0;
    }
    let rms = (error / count).sqrt();
    if rms == 0.0 {
        eprintln!("rms=0.000000 psnr=inf dB");
        return Ok(true);
    }
    let signal = 20.0 * rms.recip().log10();
    eprintln!("rms={rms:.6} psnr={signal:.2} dB");
    Ok(signal >= 22.0)
}
