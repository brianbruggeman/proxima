//! compares two sets of raw little-endian f32 embeddings: `emb_diff <dir_a> <prefix_a> <dir_b> <prefix_b>`
//! reads `<dir>/<prefix>_{0,1,2}.f32` from each side and prints max abs diff and cosine per sentence.

use std::env;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use ort_arm::{cosine, max_abs_diff};

const SENTENCE_COUNT: usize = 3;

fn read_f32(dir: &Path, prefix: &str, index: usize) -> Result<Vec<f32>> {
    let path = dir.join(format!("{prefix}_{index}.f32"));
    let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    if bytes.is_empty() || bytes.len() % 4 != 0 {
        bail!("{} is {} bytes, not a non-empty f32 array", path.display(), bytes.len());
    }
    Ok(bytes.as_chunks::<4>().0.iter().map(|chunk| f32::from_le_bytes(*chunk)).collect())
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let [dir_left, prefix_left, dir_right, prefix_right] = args.as_slice() else {
        bail!("usage: emb_diff <dir_a> <prefix_a> <dir_b> <prefix_b>");
    };
    for index in 0..SENTENCE_COUNT {
        let left = read_f32(Path::new(dir_left), prefix_left, index)?;
        let right = read_f32(Path::new(dir_right), prefix_right, index)?;
        if left.len() != right.len() {
            bail!("sentence {index}: lengths differ, {} vs {}", left.len(), right.len());
        }
        println!(
            "sentence {index}: dims={} max_abs_diff={:e} cosine={:.9}",
            left.len(),
            max_abs_diff(&left, &right),
            cosine(&left, &right)
        );
    }
    Ok(())
}
