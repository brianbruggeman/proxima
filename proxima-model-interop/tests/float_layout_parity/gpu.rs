use core::ffi::c_void;
use core::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLanguageVersion, MTLLibrary, MTLMathMode, MTLResourceOptions, MTLSize,
};

pub struct Gpu {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

impl Gpu {
    pub fn open() -> Self {
        let device = MTLCreateSystemDefaultDevice().expect("this box has a Metal device");
        let queue = device
            .newCommandQueue()
            .expect("the device hands out a command queue");
        Self { device, queue }
    }

    fn library_function(
        &self,
        source: &str,
        entry: &str,
    ) -> Retained<ProtocolObject<dyn objc2_metal::MTLFunction>> {
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version3_1);
        options.setMathMode(MTLMathMode::Safe);
        let library = self
            .device
            .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
            .unwrap_or_else(|error| {
                panic!(
                    "metal rejected the probe source: {}\n{source}",
                    error.localizedDescription()
                )
            });
        library
            .newFunctionWithName(&NSString::from_str(entry))
            .unwrap_or_else(|| panic!("kernel entry `{entry}` is missing from its own library"))
    }

    fn shared_buffer(&self, bytes: &[u8]) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        // SAFETY: `bytes` is a live non-empty slice for the whole call and
        // `newBufferWithBytes_length_options` copies it once.
        let pointer = unsafe { NonNull::new_unchecked(bytes.as_ptr().cast_mut().cast::<c_void>()) };
        unsafe {
            self.device.newBufferWithBytes_length_options(
                pointer,
                bytes.len(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .expect("the device allocates a shared input buffer")
    }

    pub fn run(
        &self,
        source: &str,
        entry: &str,
        inputs: &[&[u8]],
        output_bytes: usize,
        threads: usize,
    ) -> Vec<u8> {
        let function = self.library_function(source, entry);
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .unwrap_or_else(|error| {
                panic!(
                    "no pipeline for `{entry}`: {}",
                    error.localizedDescription()
                )
            });
        let input_buffers: Vec<_> = inputs
            .iter()
            .map(|bytes| self.shared_buffer(bytes))
            .collect();
        let output = self
            .device
            .newBufferWithLength_options(output_bytes, MTLResourceOptions::StorageModeShared)
            .expect("the device allocates the output buffer");
        let command_buffer = self.queue.commandBuffer().expect("a command buffer");
        let encoder = command_buffer
            .computeCommandEncoder()
            .expect("a compute encoder");
        encoder.setComputePipelineState(&pipeline);
        for (index, buffer) in input_buffers.iter().enumerate() {
            // SAFETY: each buffer outlives the command buffer's completion below.
            unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), 0, index) };
        }
        unsafe { encoder.setBuffer_offset_atIndex(Some(&output), 0, input_buffers.len()) };
        let width = threads.clamp(1, pipeline.maxTotalThreadsPerThreadgroup().min(256));
        encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        let mut bytes = vec![0u8; output_bytes];
        // SAFETY: the shared output buffer holds `output_bytes` initialized bytes and the
        // command buffer has completed.
        unsafe {
            core::ptr::copy_nonoverlapping(
                output.contents().as_ptr().cast::<u8>(),
                bytes.as_mut_ptr(),
                output_bytes,
            );
        }
        bytes
    }
}

pub fn words(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

pub fn floats(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

pub fn read_words(bytes: &[u8]) -> Vec<u32> {
    let (chunks, _) = bytes.as_chunks::<4>();
    chunks.iter().map(|chunk| u32::from_le_bytes(*chunk)).collect()
}
