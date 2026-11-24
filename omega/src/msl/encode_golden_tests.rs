use alloc::format;
use alloc::string::String;

use proxima_numeric::NumericFormat;

use super::encode_render::{encode_kernel_refusal, format_encode_kernel_msl};

const GOLDEN: &str = include_str!("encode_golden.txt");
const WRITE_VAR: &str = "PROXIMA_WRITE_ENCODE_GOLDEN";

fn render_all() -> String {
    let mut text = String::new();
    for format in NumericFormat::every() {
        text.push_str(&format!("=== {format:?}\n"));
        match format_encode_kernel_msl(&format, "golden_encode") {
            Ok(rendered) => text.push_str(&format!(
                "entry {} elements {} bytes {}\n{}",
                rendered.entry, rendered.elements_per_thread, rendered.bytes_per_thread, rendered.source
            )),
            Err(reason) => {
                assert_eq!(encode_kernel_refusal(&format), Err(reason));
                text.push_str(&format!("refused {reason}\n"));
            }
        }
    }
    text
}

#[test]
fn the_msl_encode_render_of_every_numeric_format_equals_the_recorded_text() {
    let rendered = render_all();
    if std::env::var_os(WRITE_VAR).is_some() {
        std::fs::write(concat!(env!("CARGO_MANIFEST_DIR"), "/src/msl/encode_golden.txt"), &rendered).expect("write golden");
        return;
    }
    let first = rendered.lines().zip(GOLDEN.lines()).position(|(left, right)| left != right);
    assert_eq!(first, None, "first differing line (0-based); rendered {} lines, golden {}", rendered.lines().count(), GOLDEN.lines().count());
    assert!(rendered == GOLDEN, "the texts differ past the shared lines");
    assert!(rendered.matches("=== ").count() >= NumericFormat::ALL.len(), "every format has an entry");
}
