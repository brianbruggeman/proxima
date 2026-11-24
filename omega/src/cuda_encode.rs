//! The CUDA spelling of the neutral encode render ([`crate::encode_render`]): the headers, the
//! kernel signature and thread index, the integer storage types, and the layout store expression
//! [`layout_store_cuda`]. The scalar codes it can store are the ones [`crate::cuda`]'s type
//! mapping has a token for. A block format has no CUDA body yet and is refused by name.

use alloc::format;
use alloc::string::String;

use proxima_numeric::block_layout::BlockLayout;
use proxima_numeric::float_layout::FloatLayout;
use proxima_numeric::DType;
use proxima_tensor::NodeId;

use crate::cuda::target::Target;
use crate::cuda_layout::layout_store_cuda;
use crate::encode_render::EncodeDialect;

const BYTE: usize = core::mem::size_of::<u8>();
const HALF_WORD: usize = core::mem::size_of::<u16>();
const WORD: usize = core::mem::size_of::<u32>();

/// The encode dialect of one card-C [`Target`].
pub struct CudaEncode {
    pub target: Target,
}

impl EncodeDialect for CudaEncode {
    fn prelude(&self) -> String {
        format!("#include <{}>\n#include <{}>\n", self.target.fp16_header, self.target.bf16_header)
    }

    fn storage_token(&self, bytes: usize) -> Option<&'static str> {
        match bytes {
            BYTE => Some("unsigned char"),
            HALF_WORD => Some("unsigned short"),
            WORD => Some("unsigned int"),
            _ => None,
        }
    }

    fn scalar_store(&self, layout: FloatLayout, name: &str) -> String {
        layout_store_cuda(layout, name)
    }

    fn scalar_refusal(&self, layout: &FloatLayout) -> Option<&'static str> {
        crate::cuda::type_token(self.target, NodeId(0), DType::Layout(*layout))
            .err()
            .map(|_| "the CUDA type mapping has no token for the layout")
    }

    fn kernel_open(&self, name: &str, storage: &str) -> String {
        format!(
            "extern \"C\" __global__ void {name}_kernel(const float *input, {storage} *output) {{\n    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;\n"
        )
    }

    fn block_function(&self, _layout: &BlockLayout, _name: &str) -> Option<String> {
        None
    }

    fn block_refusal(&self) -> Option<&'static str> {
        Some("the CUDA dialect has no block encode body")
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use alloc::vec::Vec;

    use proxima_numeric::NumericFormat;

    use super::CudaEncode;
    use crate::cuda::target::Target;
    use crate::encode_render::{encode_kernel_refusal, format_encode_kernel};
    use crate::msl::encode_render::MslEncode;

    pub(crate) fn every_format() -> impl Iterator<Item = NumericFormat> {
        NumericFormat::ALL.into_iter().chain(NumericFormat::UNTAGGED.into_iter().map(|(format, _)| format))
    }

    #[test]
    fn cuda_refuses_exactly_the_block_formats_the_msl_dialect_renders() {
        let cuda = CudaEncode { target: Target::CUDA };
        let mut refused = Vec::new();
        let mut rendered = Vec::new();
        for format in every_format() {
            let metal_renders = encode_kernel_refusal(&MslEncode, &format).is_ok();
            let outcome = format_encode_kernel(&cuda, &format, "probe_encode");
            let expect_refusal = !metal_renders || matches!(format, NumericFormat::Block(_));
            assert_eq!(outcome.is_err(), expect_refusal, "{format:?}: {outcome:?}");
            match outcome {
                Ok(_) => rendered.push(format),
                Err(reason) if metal_renders => refused.push((format, reason)),
                Err(_) => {}
            }
        }
        eprintln!("cuda renders {rendered:?}\ncuda refuses {refused:?}");
        assert!(!rendered.is_empty(), "no format renders for CUDA");
        assert!(!refused.is_empty(), "the block refusal set is empty");
        assert!(refused.iter().all(|(format, reason)| matches!(format, NumericFormat::Block(_)) && reason.contains("CUDA")), "{refused:?}");
    }

    #[test]
    fn a_cuda_scalar_encode_names_no_metal_type() {
        let cuda = CudaEncode { target: Target::CUDA };
        for format in every_format().filter(|format| matches!(format, NumericFormat::Scalar(_))) {
            let Ok(rendered) = format_encode_kernel(&cuda, &format, "probe_encode") else { continue };
            for token in ["kernel void ", "thread_position_in_grid", "device ", " uchar", " ushort", " uint "] {
                assert!(!rendered.source.contains(token), "{format:?} carries {token}:\n{}", rendered.source);
            }
            assert!(rendered.source.contains("__global__"), "{format:?}");
        }
    }
}
