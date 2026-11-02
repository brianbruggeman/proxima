//! Reports the no-copy-vs-copy Metal upload split. The counters are
//! process-global, so the number only means something inside one process:
//! this example runs one page-aligned input (zero-copy path) and one small
//! heap input (copying path) through real [`omega::execute`] calls and prints
//! the totals, exiting nonzero if no upload was observed at all.
//!
//! `cargo run -p proxima-omega --features metal --example metal_upload_totals`

#![allow(clippy::unwrap_used, clippy::expect_used)]

#[cfg(all(feature = "metal", target_os = "macos"))]
mod upload {
    use std::process::ExitCode;

    use proxima_tensor::{
        DType, Extent, IndexMap, NumericPolicy, Op, QuantizedBlock, ScalarOp, append, projection,
    };

    fn tanh_chain(elements: u32) -> Vec<Op> {
        let mut program = Vec::new();
        let input = append(
            &mut program,
            Op::Input {
                dtype: DType::FLOAT32,
                shape: vec![Extent::Static(elements)],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::FLOAT32,
                body: ScalarOp::Tanh,
                operands: vec![(input, IndexMap::Affine(projection(1, &[0])))],
                name: None,
            },
        );
        program
    }

    fn run_page_aligned() {
        let page_elements = omega::page_size() / size_of::<f32>();
        let program = tanh_chain(page_elements as u32);
        let mut aligned = proxima_tensor::AlignedBuffer::new(page_elements, omega::page_size())
            .expect("a one-page aligned request never fails");
        for (index, value) in aligned.iter_mut().enumerate() {
            *value = (index % 7) as f32 * 0.1;
        }
        omega::execute(
            &program,
            &[],
            &[QuantizedBlock::Float32(&aligned[..])],
            &[],
            NumericPolicy::default(),
        )
        .expect("metal executes the page-aligned block on a real device");
    }

    fn run_small_heap() {
        let data: Vec<f32> = (0..5).map(|index| index as f32 * 0.1).collect();
        omega::execute(
            &tanh_chain(data.len() as u32),
            &[],
            &[QuantizedBlock::Float32(&data[..])],
            &[],
            NumericPolicy::default(),
        )
        .expect("metal executes the small heap block on a real device");
    }

    pub fn report() -> ExitCode {
        run_page_aligned();
        run_small_heap();
        let nocopy = omega::metal::NOCOPY_BUFFER_UPLOADS.get();
        let copy = omega::metal::COPYING_BUFFER_UPLOADS.get();
        println!("metal upload path totals: nocopy={nocopy} copy={copy}");
        if nocopy + copy == 0 {
            eprintln!("no real upload_block call was observed by this process");
            return ExitCode::FAILURE;
        }
        ExitCode::SUCCESS
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() -> std::process::ExitCode {
    upload::report()
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    println!("metal_upload_totals requires --features metal on macOS; skipped");
}
