//! one BGE embedding session on the CPU execution provider, configured as the 2026-09 harness was:
//! `intra_op=1`, `inter_op=1`, sequential execution, default graph optimization, batch 1.
//!
//! composes the raw `OrtApi` function table from [`crate::OrtRuntime`]; every resource is released in `Drop`.

use std::ffi::{CStr, CString, c_char, c_void};
use std::ffi::OsStr;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::ptr;

use crate::bindings::{
    ExecutionMode_ORT_SEQUENTIAL, ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, OrtApi,
    OrtAllocatorType_OrtArenaAllocator, OrtEnv, OrtLoggingLevel_ORT_LOGGING_LEVEL_WARNING, OrtMemType_OrtMemTypeDefault,
    OrtMemoryInfo, OrtSession, OrtSessionOptions, OrtStatusPtr, OrtValue,
};
use crate::{LoadError, OrtRuntime, l2_normalize};

const HIDDEN_SIZE: usize = 384;
const EXPECTED_INPUTS: [&CStr; 3] = [c"input_ids", c"attention_mask", c"token_type_ids"];
const EXPECTED_OUTPUT: &CStr = c"last_hidden_state";

fn check(api: &OrtApi, call: &'static str, status: OrtStatusPtr) -> Result<(), LoadError> {
    if status.is_null() {
        return Ok(());
    }
    let message = api.GetErrorMessage.map_or_else(String::new, |get_message| {
        // safety: status is a live OrtStatus returned by the call just made
        let text = unsafe { get_message(status) };
        if text.is_null() {
            String::new()
        } else {
            // safety: non-null NUL-terminated string owned by the status
            unsafe { CStr::from_ptr(text) }.to_string_lossy().into_owned()
        }
    });
    if let Some(release) = api.ReleaseStatus {
        // safety: status is owned by us and released exactly once
        unsafe { release(status) };
    }
    Err(LoadError::Ort { call, message })
}

struct Tensor<'rt> {
    api: &'rt OrtApi,
    raw: *mut OrtValue,
}

impl Drop for Tensor<'_> {
    fn drop(&mut self) {
        if let Some(release) = self.api.ReleaseValue {
            // safety: raw is null or a value this wrapper owns; ReleaseValue accepts null
            unsafe { release(self.raw) };
        }
    }
}

pub struct EmbedSession<'rt> {
    runtime: &'rt OrtRuntime,
    env: *mut OrtEnv,
    options: *mut OrtSessionOptions,
    session: *mut OrtSession,
    memory_info: *mut OrtMemoryInfo,
}

impl<'rt> EmbedSession<'rt> {
    /// builds the single-thread sequential session over `model_path` and verifies the graph's
    /// input and output names match what [`embed`](Self::embed) feeds.
    ///
    /// # Errors
    /// any onnxruntime failure or a model whose signature is not the BGE one.
    pub fn new(runtime: &'rt OrtRuntime, model_path: &Path) -> Result<Self, LoadError> {
        Self::with_profile_prefix(runtime, model_path, None)
    }

    /// [`new`](Self::new) plus onnxruntime's built-in profiler: with `Some(prefix)` the session
    /// records one json event per node per `Run`, written by [`end_profiling`](Self::end_profiling).
    /// profiling adds per-node bookkeeping, so a profiled run is for attribution, never for timing.
    ///
    /// # Errors
    /// any onnxruntime failure, a prefix containing NUL, or a model whose signature is not the BGE one.
    pub fn with_profile_prefix(
        runtime: &'rt OrtRuntime,
        model_path: &Path,
        profile_prefix: Option<&Path>,
    ) -> Result<Self, LoadError> {
        let api = runtime.api();
        let prefix = profile_prefix
            .map(|prefix| CString::new(prefix.as_os_str().as_bytes()).map_err(|_| LoadError::BadPath(prefix.to_path_buf())))
            .transpose()?;
        let mut this = Self {
            runtime,
            env: ptr::null_mut(),
            options: ptr::null_mut(),
            session: ptr::null_mut(),
            memory_info: ptr::null_mut(),
        };
        let path = CString::new(model_path.as_os_str().as_bytes())
            .map_err(|_| LoadError::BadPath(model_path.to_path_buf()))?;
        let missing = LoadError::MissingFunction;
        let create_env = api.CreateEnv.ok_or(missing("CreateEnv"))?;
        let create_options = api.CreateSessionOptions.ok_or(missing("CreateSessionOptions"))?;
        let set_intra = api.SetIntraOpNumThreads.ok_or(missing("SetIntraOpNumThreads"))?;
        let set_inter = api.SetInterOpNumThreads.ok_or(missing("SetInterOpNumThreads"))?;
        let set_mode = api.SetSessionExecutionMode.ok_or(missing("SetSessionExecutionMode"))?;
        let create_session = api.CreateSession.ok_or(missing("CreateSession"))?;
        let create_memory = api.CreateCpuMemoryInfo.ok_or(missing("CreateCpuMemoryInfo"))?;
        // safety: each call receives live out-pointers into `this`; statuses are checked immediately
        unsafe {
            check(api, "CreateEnv", create_env(OrtLoggingLevel_ORT_LOGGING_LEVEL_WARNING, c"ort-arm".as_ptr(), &raw mut this.env))?;
            check(api, "CreateSessionOptions", create_options(&raw mut this.options))?;
            check(api, "SetIntraOpNumThreads", set_intra(this.options, 1))?;
            check(api, "SetInterOpNumThreads", set_inter(this.options, 1))?;
            check(api, "SetSessionExecutionMode", set_mode(this.options, ExecutionMode_ORT_SEQUENTIAL))?;
            if let Some(prefix) = &prefix {
                let enable = api.EnableProfiling.ok_or(missing("EnableProfiling"))?;
                check(api, "EnableProfiling", enable(this.options, prefix.as_ptr()))?;
            }
            check(api, "CreateSession", create_session(this.env, path.as_ptr(), this.options, &raw mut this.session))?;
            check(
                api,
                "CreateCpuMemoryInfo",
                create_memory(OrtAllocatorType_OrtArenaAllocator, OrtMemType_OrtMemTypeDefault, &raw mut this.memory_info),
            )?;
        }
        this.verify_signature()?;
        Ok(this)
    }

    /// stops the profiler and returns the path of the json file onnxruntime wrote.
    ///
    /// # Errors
    /// any onnxruntime failure, including a session built without a profile prefix.
    pub fn end_profiling(&self) -> Result<PathBuf, LoadError> {
        let api = self.runtime.api();
        let get_allocator = api.GetAllocatorWithDefaultOptions.ok_or(LoadError::MissingFunction("GetAllocatorWithDefaultOptions"))?;
        let end = api.SessionEndProfiling.ok_or(LoadError::MissingFunction("SessionEndProfiling"))?;
        let mut allocator = ptr::null_mut();
        let mut raw: *mut c_char = ptr::null_mut();
        // safety: allocator is the process-wide default; raw is allocated by it and freed after the copy
        unsafe {
            check(api, "GetAllocatorWithDefaultOptions", get_allocator(&raw mut allocator))?;
            check(api, "SessionEndProfiling", end(self.session, allocator, &raw mut raw))?;
            let owned = CStr::from_ptr(raw).to_owned();
            if let Some(free) = (*allocator).Free {
                free(allocator, raw.cast::<c_void>());
            }
            Ok(PathBuf::from(OsStr::from_bytes(owned.to_bytes())))
        }
    }

    fn verify_signature(&self) -> Result<(), LoadError> {
        let api = self.runtime.api();
        let input_count = api.SessionGetInputCount.ok_or(LoadError::MissingFunction("SessionGetInputCount"))?;
        let mut count = 0usize;
        // safety: session is live; count is a valid out-pointer
        check(api, "SessionGetInputCount", unsafe { input_count(self.session, &raw mut count) })?;
        if count != EXPECTED_INPUTS.len() {
            return Err(LoadError::Model(format!("{count} inputs, expected {}", EXPECTED_INPUTS.len())));
        }
        for (index, expected) in EXPECTED_INPUTS.iter().enumerate() {
            let name = self.io_name(true, index)?;
            if name.as_c_str() != *expected {
                return Err(LoadError::Model(format!("input {index} is {name:?}, expected {expected:?}")));
            }
        }
        let output = self.io_name(false, 0)?;
        if output.as_c_str() != EXPECTED_OUTPUT {
            return Err(LoadError::Model(format!("output 0 is {output:?}, expected {EXPECTED_OUTPUT:?}")));
        }
        Ok(())
    }

    fn io_name(&self, input: bool, index: usize) -> Result<CString, LoadError> {
        let api = self.runtime.api();
        let get_allocator = api.GetAllocatorWithDefaultOptions.ok_or(LoadError::MissingFunction("GetAllocatorWithDefaultOptions"))?;
        let (call, get_name) = if input {
            ("SessionGetInputName", api.SessionGetInputName)
        } else {
            ("SessionGetOutputName", api.SessionGetOutputName)
        };
        let get_name = get_name.ok_or(LoadError::MissingFunction(call))?;
        let mut allocator = ptr::null_mut();
        let mut raw: *mut c_char = ptr::null_mut();
        // safety: allocator is the process-wide default and outlives the session; raw is allocated by it
        unsafe {
            check(api, "GetAllocatorWithDefaultOptions", get_allocator(&raw mut allocator))?;
            check(api, call, get_name(self.session, index, allocator, &raw mut raw))?;
            let owned = CStr::from_ptr(raw).to_owned();
            if let Some(free) = (*allocator).Free {
                free(allocator, raw.cast::<c_void>());
            }
            Ok(owned)
        }
    }

    fn tensor(&self, data: &mut [i64], shape: &[i64; 2]) -> Result<Tensor<'rt>, LoadError> {
        let api = self.runtime.api();
        let create = api.CreateTensorWithDataAsOrtValue.ok_or(LoadError::MissingFunction("CreateTensorWithDataAsOrtValue"))?;
        let mut tensor = Tensor { api, raw: ptr::null_mut() };
        // safety: data outlives the tensor (both live inside one embed call); onnxruntime only reads int64 inputs
        let status = unsafe {
            create(
                self.memory_info,
                data.as_mut_ptr().cast::<c_void>(),
                std::mem::size_of_val(data),
                shape.as_ptr(),
                shape.len(),
                ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                &raw mut tensor.raw,
            )
        };
        check(api, "CreateTensorWithDataAsOrtValue", status)?;
        Ok(tensor)
    }

    /// one sentence: feeds ids, an all-ones mask and all-zero token types, takes the CLS row of
    /// `last_hidden_state`, l2-normalizes it. This whole call is what the timed window covers,
    /// the same span the python harness's `embed` closure covered.
    ///
    /// # Errors
    /// any onnxruntime failure, or an output whose element count is not `tokens * 384`.
    pub fn embed(&self, tokens: &[i64]) -> Result<Vec<f32>, LoadError> {
        let api = self.runtime.api();
        let length = i64::try_from(tokens.len()).map_err(|_| LoadError::Model("sequence too long".to_owned()))?;
        let shape = [1, length];
        let mut ids = tokens.to_vec();
        let mut mask = vec![1_i64; tokens.len()];
        let mut types = vec![0_i64; tokens.len()];
        let id_tensor = self.tensor(&mut ids, &shape)?;
        let mask_tensor = self.tensor(&mut mask, &shape)?;
        let type_tensor = self.tensor(&mut types, &shape)?;
        let input_names: [*const c_char; 3] = EXPECTED_INPUTS.map(CStr::as_ptr);
        let inputs: [*const OrtValue; 3] = [id_tensor.raw, mask_tensor.raw, type_tensor.raw];
        let output_names = [EXPECTED_OUTPUT.as_ptr()];
        let run = api.Run.ok_or(LoadError::MissingFunction("Run"))?;
        let mut output = Tensor { api, raw: ptr::null_mut() };
        // safety: arrays hold 3 live inputs and 1 name; output starts null so onnxruntime allocates it
        let status = unsafe {
            run(
                self.session,
                ptr::null(),
                input_names.as_ptr(),
                inputs.as_ptr(),
                inputs.len(),
                output_names.as_ptr(),
                output_names.len(),
                &raw mut output.raw,
            )
        };
        check(api, "Run", status)?;
        let hidden = self.read_output(&output, tokens.len())?;
        Ok(l2_normalize(&hidden[..HIDDEN_SIZE]))
    }

    fn read_output(&self, output: &Tensor<'_>, tokens: usize) -> Result<Vec<f32>, LoadError> {
        let api = self.runtime.api();
        let type_and_shape = api.GetTensorTypeAndShape.ok_or(LoadError::MissingFunction("GetTensorTypeAndShape"))?;
        let element_count = api.GetTensorShapeElementCount.ok_or(LoadError::MissingFunction("GetTensorShapeElementCount"))?;
        let release_info = api.ReleaseTensorTypeAndShapeInfo.ok_or(LoadError::MissingFunction("ReleaseTensorTypeAndShapeInfo"))?;
        let mutable_data = api.GetTensorMutableData.ok_or(LoadError::MissingFunction("GetTensorMutableData"))?;
        let mut info = ptr::null_mut();
        let mut count = 0usize;
        let mut data: *mut c_void = ptr::null_mut();
        // safety: output holds the live tensor Run produced; info is released right after the count is read
        unsafe {
            check(api, "GetTensorTypeAndShape", type_and_shape(output.raw, &raw mut info))?;
            let counted = element_count(info, &raw mut count);
            release_info(info);
            check(api, "GetTensorShapeElementCount", counted)?;
            if count != tokens * HIDDEN_SIZE {
                return Err(LoadError::Model(format!("output has {count} elements, expected {}", tokens * HIDDEN_SIZE)));
            }
            check(api, "GetTensorMutableData", mutable_data(output.raw, &raw mut data))?;
            // safety: data points at `count` contiguous f32 owned by the output tensor, copied before it drops
            Ok(std::slice::from_raw_parts(data.cast::<f32>().cast_const(), count).to_vec())
        }
    }
}

impl Drop for EmbedSession<'_> {
    fn drop(&mut self) {
        let api = self.runtime.api();
        // safety: each pointer is null or exclusively owned here; the Release functions accept null
        unsafe {
            if let Some(release) = api.ReleaseMemoryInfo {
                release(self.memory_info);
            }
            if let Some(release) = api.ReleaseSession {
                release(self.session);
            }
            if let Some(release) = api.ReleaseSessionOptions {
                release(self.options);
            }
            if let Some(release) = api.ReleaseEnv {
                release(self.env);
            }
        }
    }
}
