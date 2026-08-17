//! Hand-written extern declarations for exactly the ggml surface
//! `bench_vs_ggml` calls. No bindgen: this is a fixed, small slice of one
//! header, and the struct layouts below are opaque (`ggml_tensor` etc. are
//! read/written only through ggml's own accessor functions, never by
//! reaching into their fields from Rust).
#![allow(non_camel_case_types, dead_code)]

use std::ffi::c_void;
use std::os::raw::c_int;

#[repr(C)]
pub struct ggml_context {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ggml_tensor {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ggml_cgraph {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ggml_threadpool {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ggml_cplan {
    pub work_size: usize,
    pub work_data: *mut u8,
    pub n_threads: c_int,
    pub threadpool: *mut ggml_threadpool,
    pub abort_callback: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
    pub abort_callback_data: *mut c_void,
    pub use_ref: bool,
}

#[repr(C)]
pub struct ggml_init_params {
    pub mem_size: usize,
    pub mem_buffer: *mut c_void,
    pub no_alloc: bool,
}

pub const GGML_TYPE_F32: c_int = 0;
pub const GGML_TYPE_Q4_0: c_int = 2;
pub const GGML_TYPE_Q8_0: c_int = 8;
pub const GGML_TYPE_Q4_K: c_int = 12;
pub const GGML_TYPE_I32: c_int = 26;

unsafe extern "C" {
    pub fn ggml_init(params: ggml_init_params) -> *mut ggml_context;
    pub fn ggml_free(ctx: *mut ggml_context);
    pub fn ggml_used_mem(ctx: *const ggml_context) -> usize;

    pub fn ggml_new_tensor_1d(ctx: *mut ggml_context, type_: c_int, ne0: i64) -> *mut ggml_tensor;
    pub fn ggml_new_tensor_2d(
        ctx: *mut ggml_context,
        type_: c_int,
        ne0: i64,
        ne1: i64,
    ) -> *mut ggml_tensor;

    pub fn ggml_mul_mat(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
    pub fn ggml_rms_norm(ctx: *mut ggml_context, a: *mut ggml_tensor, eps: f32) -> *mut ggml_tensor;
    pub fn ggml_silu(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_add(ctx: *mut ggml_context, a: *mut ggml_tensor, b: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_mul(ctx: *mut ggml_context, a: *mut ggml_tensor, b: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_sub(ctx: *mut ggml_context, a: *mut ggml_tensor, b: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_div(ctx: *mut ggml_context, a: *mut ggml_tensor, b: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_sqr(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_sqrt(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_exp(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_neg(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_scale(ctx: *mut ggml_context, a: *mut ggml_tensor, s: f32) -> *mut ggml_tensor;
    pub fn ggml_get_rows(ctx: *mut ggml_context, a: *mut ggml_tensor, b: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_sum_rows(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_tanh(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;

    pub fn ggml_new_graph(ctx: *mut ggml_context) -> *mut ggml_cgraph;
    pub fn ggml_build_forward_expand(cgraph: *mut ggml_cgraph, tensor: *mut ggml_tensor);
    pub fn ggml_graph_compute_with_ctx(
        ctx: *mut ggml_context,
        cgraph: *mut ggml_cgraph,
        n_threads: c_int,
    ) -> c_int;
    pub fn ggml_graph_plan(
        cgraph: *const ggml_cgraph,
        n_threads: c_int,
        threadpool: *mut ggml_threadpool,
    ) -> ggml_cplan;
    pub fn ggml_graph_compute(cgraph: *mut ggml_cgraph, cplan: *mut ggml_cplan) -> c_int;

    pub fn ggml_get_data(tensor: *const ggml_tensor) -> *mut c_void;
    pub fn ggml_get_data_f32(tensor: *const ggml_tensor) -> *mut f32;
    pub fn ggml_nelements(tensor: *const ggml_tensor) -> i64;
    pub fn ggml_nbytes(tensor: *const ggml_tensor) -> usize;

    pub fn ggml_quantize_chunk(
        type_: c_int,
        src: *const f32,
        dst: *mut c_void,
        start: i64,
        nrows: i64,
        n_per_row: i64,
        imatrix: *const f32,
    ) -> usize;
}
