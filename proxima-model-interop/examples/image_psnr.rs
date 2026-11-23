//! PSNR between two PNGs: the sd.cpp reference image and ours (PROXNUM-588 AC7).
//!
//!   cargo run --release -p proxima-model-interop --example image_psnr -- ref.png omega.png
//!
//! Prints both images' dimensions, the RMS difference and the PSNR in dB against
//! the full `1.0` scale the spec's gates are stated on, and exits non-zero below
//! the 22.0 dB gate.

use proxima_model_interop::diffusion::{PngImage, read_png};

fn main() {
    let mut arguments = std::env::args().skip(1);
    let (Some(reference), Some(actual)) = (arguments.next(), arguments.next()) else {
        eprintln!("usage: image_psnr <reference.png> <actual.png>");
        std::process::exit(64);
    };
    let reference = match read_png(reference.as_ref()) {
        Ok(image) => image,
        Err(why) => {
            eprintln!("{why}");
            std::process::exit(1);
        }
    };
    let actual = match read_png(actual.as_ref()) {
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
    match psnr(&actual, &reference) {
        Ok((signal, rms)) => {
            eprintln!("rms={rms:.6} psnr={signal:.2} dB");
            if signal < 22.0 {
                eprintln!("below the 22.0 dB gate");
                std::process::exit(2);
            }
        }
        Err(why) => {
            eprintln!("{why}");
            std::process::exit(1);
        }
    }
}

fn psnr(actual: &PngImage, reference: &PngImage) -> Result<(f64, f64), String> {
    if (actual.width, actual.height, actual.channels)
        != (reference.width, reference.height, reference.channels)
    {
        return Err(String::from("images differ in size or channel count"));
    }
    let mut error = 0.0f64;
    for (&got, &want) in actual.pixels.iter().zip(&reference.pixels) {
        let difference = f64::from(got) - f64::from(want);
        error += difference * difference;
    }
    let rms = (error / actual.pixels.len() as f64).sqrt();
    if rms == 0.0 {
        return Ok((f64::INFINITY, 0.0));
    }
    Ok((20.0 * (1.0 / rms).log10(), rms))
}
