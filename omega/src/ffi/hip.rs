//! The HIP and hipRTC C ABI: the one place in omega that resolves `libamdhip64` and `libhiprtc`
//! and calls through their symbols.
//!
//! Everything here is `dlopen`-time: no link dependency, no headers, no build-script probing, so a
//! host with no ROCm installed gets a typed [`HipDriverError::RuntimeNotInstalled`] instead of a
//! build failure. [`crate::silicon::hip`] holds the device, buffers, modules and launch that sit on
//! the safe methods of [`HipApi`].

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::ffi::{c_char, c_int, c_void};
use core::mem::MaybeUninit;

use std::ffi::{CStr, CString};
use std::ptr;
use std::sync::Arc;

use crate::silicon::hip::error::HipDriverError;

/// The address a kernel launch reads one argument from: `KernelArgument::slot`'s return type.
pub type KernelSlot = *mut c_void;

/// The parameter array a launch hands to `hipModuleLaunchKernel`, as `KernelParams` builds it.
pub type LaunchParameters = *mut *mut c_void;

/// A device address returned by `hipMalloc`, or an interior address derived from one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct DevicePointer(*mut c_void);

impl DevicePointer {
    /// A pointer holding `address`, for tests and for arithmetic on a known base.
    #[must_use]
    pub fn from_address(address: usize) -> Self {
        Self(ptr::without_provenance_mut(address))
    }

    #[must_use]
    pub fn address(self) -> usize {
        self.0.addr()
    }

    #[must_use]
    pub fn is_null(self) -> bool {
        self.0.is_null()
    }

    /// `self` advanced `bytes`. The result is only dereferenced by the device, and only after the
    /// caller proved it stays inside the allocation.
    #[must_use]
    pub fn offset(self, bytes: usize) -> Self {
        Self(self.0.cast::<u8>().wrapping_add(bytes).cast::<c_void>())
    }

    /// The address of this pointer's own storage, which is what a launch reads the argument from.
    #[must_use]
    pub fn slot(&self) -> KernelSlot {
        (&raw const self.0).cast_mut().cast::<c_void>()
    }
}

/// A loaded code object, from `hipModuleLoadData`.
#[derive(Debug, Clone, Copy)]
pub struct ModuleHandle(*mut c_void);

/// An entry point inside a [`ModuleHandle`].
#[derive(Debug, Clone, Copy)]
pub struct FunctionHandle(*mut c_void);

/// Shared objects the runtime half probes, in order. The versioned names come
/// first so a host with several ROCm installs resolves the one on its linker
/// path rather than an arbitrary sibling.
#[cfg(not(windows))]
pub const RUNTIME_LIBRARIES: [&str; 4] = [
    "libamdhip64.so.6",
    "libamdhip64.so",
    "libamdhip64.so.5",
    "libamdhip64.dylib",
];

/// The HIP runtime DLL on Windows, newest first: `amdhip64_7.dll` (HIP 7),
/// `amdhip64_6.dll` (HIP 6, renamed from `amdhip64.dll` as of ROCm 6.1.2) and
/// `amdhip64.dll` (5.x). Source: the ROCm on Windows release notes,
/// <https://rocm.docs.amd.com/projects/install-on-windows/en/latest/about/releasenotes.html>
/// and <https://rocm.docs.amd.com/projects/install-on-windows/en/docs-6.2.4/>.
#[cfg(windows)]
pub const RUNTIME_LIBRARIES: [&str; 3] = ["amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"];

/// hipRTC — the runtime compiler the emitter's source needs. On Linux its
/// symbols are also exported by `libamdhip64.so`.
#[cfg(not(windows))]
pub const RTC_LIBRARIES: [&str; 3] = ["libhiprtc.so.6", "libhiprtc.so", "libhiprtc.dylib"];

/// Windows hipRTC: `amdhip64.dll` does not carry the HIPRTC symbols, so
/// `hiprtc.dll` (or the versioned `hiprtcXXYY.dll`, XX major and YY minor,
/// two digits each: HIP 5.3 is `hiprtc0503.dll`) must be loaded. The
/// versioned name is derived from the runtime version by
/// [`hiprtc_windows_candidates`]; this is the unversioned fallback. Source:
/// <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_rtc.html>.
#[cfg(windows)]
pub const RTC_LIBRARIES: [&str; 1] = ["hiprtc.dll"];

/// The Windows hipRTC names to probe for a runtime that reported `version`
/// from `hipRuntimeGetVersion` (HIP's `HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000
/// + HIP_VERSION_PATCH` encoding, per
/// <https://rocm.docs.amd.com/projects/HIP/en/docs-6.1.0/how-to/faq.html>): the
/// versioned DLL first, then plain `hiprtc.dll`.
#[must_use]
pub fn hiprtc_windows_candidates(version: i32) -> Vec<String> {
    let major = version / 10_000_000;
    let minor = (version / 100_000) % 100;
    let mut names = Vec::new();
    if major > 0 {
        names.push(format!("hiprtc{major:02}{minor:02}.dll"));
    }
    names.push(String::from("hiprtc.dll"));
    names
}

/// `hipMemcpyHostToDevice`.
const MEMCPY_HOST_TO_DEVICE: c_int = 1;
/// `hipMemcpyDeviceToHost`.
const MEMCPY_DEVICE_TO_HOST: c_int = 2;
/// `hipMemcpyDeviceToDevice`.
const MEMCPY_DEVICE_TO_DEVICE: c_int = 3;
const HIP_SUCCESS: c_int = 0;
const RTC_SUCCESS: c_int = 0;

/// The R0600 `hipDeviceProp_t` is about 1.5 KB; the margin keeps a later
/// runtime from writing past this buffer in `hipGetDevicePropertiesR0600`.
pub const HIP_DEVICE_PROPERTIES_BYTES: usize = 4096;
const HIP_DEVICE_PROPERTIES_WORDS: usize =
    HIP_DEVICE_PROPERTIES_BYTES / core::mem::size_of::<u64>();
const _: () = assert!(HIP_DEVICE_PROPERTIES_BYTES >= 2048);

type HipInit = unsafe extern "C" fn(u32) -> c_int;
type HipGetDeviceCount = unsafe extern "C" fn(*mut c_int) -> c_int;
type HipSetDevice = unsafe extern "C" fn(c_int) -> c_int;
type HipDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int;
type HipGetDeviceProperties = unsafe extern "C" fn(*mut c_void, c_int) -> c_int;
type HipDeviceSynchronize = unsafe extern "C" fn() -> c_int;
type HipMalloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int;
type HipFree = unsafe extern "C" fn(*mut c_void) -> c_int;
type HipMemcpy = unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int;
type HipMemset = unsafe extern "C" fn(*mut c_void, c_int, usize) -> c_int;
type HipMemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> c_int;
type HipGetErrorString = unsafe extern "C" fn(c_int) -> *const c_char;
type HipModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> c_int;
type HipModuleGetFunction =
    unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> c_int;
type HipModuleLaunchKernel = unsafe extern "C" fn(
    *mut c_void,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> c_int;
type HipModuleUnload = unsafe extern "C" fn(*mut c_void) -> c_int;

type RtcCreateProgram = unsafe extern "C" fn(
    *mut *mut c_void,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> c_int;
type RtcCompileProgram = unsafe extern "C" fn(*mut c_void, c_int, *const *const c_char) -> c_int;
type RtcSizeQuery = unsafe extern "C" fn(*mut c_void, *mut usize) -> c_int;
type RtcWriteText = unsafe extern "C" fn(*mut c_void, *mut c_char) -> c_int;
type RtcVersion = unsafe extern "C" fn(*mut c_int, *mut c_int) -> c_int;
type RtcDestroyProgram = unsafe extern "C" fn(*mut *mut c_void) -> c_int;
type RtcGetErrorString = unsafe extern "C" fn(c_int) -> *const c_char;
#[cfg(windows)]
type HipRuntimeGetVersion = unsafe extern "C" fn(*mut c_int) -> c_int;

/// One loaded shared object (`dlopen` on unix, `LoadLibrary` on Windows,
/// through `libloading`). Kept alive for as long as any symbol resolved from
/// it is callable, which is why every device, module and buffer holds an
/// `Arc` of the [`HipApi`] that owns it.
#[derive(Debug)]
struct Library(libloading::Library);

fn open_library<S: AsRef<str>>(
    candidates: &[S],
    absent: fn(String) -> HipDriverError,
) -> Result<Library, HipDriverError> {
    let mut tried = Vec::new();
    for name in candidates {
        let name = name.as_ref();
        // SAFETY: loading a library runs its initializers; the names are the
        // vendor runtime's own, and a failure is reported, never ignored.
        match unsafe { libloading::Library::new(name) } {
            Ok(library) => return Ok(Library(library)),
            Err(error) => tried.push(format!("{name} ({error})")),
        }
    }
    Err(absent(tried.join(", ")))
}

fn resolve<T: Copy>(
    library: &Library,
    library_name: &'static str,
    symbol: &'static str,
) -> Result<T, HipDriverError> {
    // SAFETY: the caller names `T` as the exported signature of `symbol`, and
    // the copied function pointer is only called while the `Library` is held.
    unsafe { library.0.get::<T>(symbol.as_bytes()) }
        .map(|address| *address)
        .map_err(|_| HipDriverError::MissingSymbol {
            library: library_name,
            symbol,
        })
}

#[cfg(windows)]
fn open_rtc(runtime: &Library) -> Result<Library, HipDriverError> {
    let version = resolve::<HipRuntimeGetVersion>(runtime, "amdhip64", "hipRuntimeGetVersion")
        .ok()
        .and_then(|query| {
            let mut version: c_int = 0;
            // SAFETY: `version` is a valid out-parameter.
            (unsafe { query(&mut version) } == HIP_SUCCESS).then_some(version)
        })
        .unwrap_or(0);
    open_library(&hiprtc_windows_candidates(version), |tried| {
        HipDriverError::RtcNotInstalled { tried }
    })
}

#[cfg(not(windows))]
fn open_rtc(_runtime: &Library) -> Result<Library, HipDriverError> {
    open_library(&RTC_LIBRARIES, |tried| HipDriverError::RtcNotInstalled {
        tried,
    })
}

/// The resolved ROCm entry points, shared by every device, module and buffer.
///
/// Holding the two library handles is what keeps the resolved pointers
/// callable for the lifetime of the `Arc`.
#[derive(Debug)]
pub struct HipApi {
    _runtime: Library,
    _rtc: Library,
    hip_init: HipInit,
    hip_get_device_count: HipGetDeviceCount,
    hip_set_device: HipSetDevice,
    hip_device_get_name: HipDeviceGetName,
    hip_get_device_properties: Option<HipGetDeviceProperties>,
    hip_device_synchronize: HipDeviceSynchronize,
    hip_malloc: HipMalloc,
    hip_free: HipFree,
    hip_memcpy: HipMemcpy,
    hip_memset: HipMemset,
    hip_mem_get_info: Option<HipMemGetInfo>,
    hip_get_error_string: HipGetErrorString,
    hip_module_load_data: HipModuleLoadData,
    hip_module_get_function: HipModuleGetFunction,
    hip_module_launch_kernel: HipModuleLaunchKernel,
    hip_module_unload: HipModuleUnload,
    rtc_create_program: RtcCreateProgram,
    rtc_compile_program: RtcCompileProgram,
    rtc_get_program_log_size: RtcSizeQuery,
    rtc_get_program_log: RtcWriteText,
    rtc_get_code_size: RtcSizeQuery,
    rtc_get_code: RtcWriteText,
    rtc_destroy_program: RtcDestroyProgram,
    rtc_get_error_string: RtcGetErrorString,
    rtc_version: Option<RtcVersion>,
}

impl HipApi {
    /// Resolves the HIP runtime and hipRTC. A host without ROCm installed
    /// returns [`HipDriverError::RuntimeNotInstalled`] naming every soname
    /// probed, never a panic and never a link failure.
    ///
    /// # Errors
    /// [`HipDriverError::RuntimeNotInstalled`] or
    /// [`HipDriverError::RtcNotInstalled`] when a library is absent, and
    /// [`HipDriverError::MissingSymbol`] when a library resolves but does not
    /// export a function this boundary needs.
    pub fn load() -> Result<Arc<Self>, HipDriverError> {
        let runtime = open_library(&RUNTIME_LIBRARIES, |tried| {
            HipDriverError::RuntimeNotInstalled { tried }
        })?;
        let rtc = open_rtc(&runtime)?;
        let runtime_name = "libamdhip64";
        let rtc_name = "libhiprtc";
        let hip_get_device_properties = resolve::<HipGetDeviceProperties>(
            &runtime,
            runtime_name,
            "hipGetDevicePropertiesR0600",
        )
        .or_else(|_| resolve(&runtime, runtime_name, "hipGetDeviceProperties"))
        .ok();
        Ok(Arc::new(Self {
            hip_init: resolve(&runtime, runtime_name, "hipInit")?,
            hip_get_device_count: resolve(&runtime, runtime_name, "hipGetDeviceCount")?,
            hip_set_device: resolve(&runtime, runtime_name, "hipSetDevice")?,
            hip_device_get_name: resolve(&runtime, runtime_name, "hipDeviceGetName")?,
            hip_get_device_properties,
            hip_device_synchronize: resolve(&runtime, runtime_name, "hipDeviceSynchronize")?,
            hip_malloc: resolve(&runtime, runtime_name, "hipMalloc")?,
            hip_free: resolve(&runtime, runtime_name, "hipFree")?,
            hip_memcpy: resolve(&runtime, runtime_name, "hipMemcpy")?,
            hip_memset: resolve(&runtime, runtime_name, "hipMemset")?,
            hip_mem_get_info: resolve(&runtime, runtime_name, "hipMemGetInfo").ok(),
            hip_get_error_string: resolve(&runtime, runtime_name, "hipGetErrorString")?,
            hip_module_load_data: resolve(&runtime, runtime_name, "hipModuleLoadData")?,
            hip_module_get_function: resolve(&runtime, runtime_name, "hipModuleGetFunction")?,
            hip_module_launch_kernel: resolve(&runtime, runtime_name, "hipModuleLaunchKernel")?,
            hip_module_unload: resolve(&runtime, runtime_name, "hipModuleUnload")?,
            rtc_create_program: resolve(&rtc, rtc_name, "hiprtcCreateProgram")?,
            rtc_compile_program: resolve(&rtc, rtc_name, "hiprtcCompileProgram")?,
            rtc_get_program_log_size: resolve(&rtc, rtc_name, "hiprtcGetProgramLogSize")?,
            rtc_get_program_log: resolve(&rtc, rtc_name, "hiprtcGetProgramLog")?,
            rtc_get_code_size: resolve(&rtc, rtc_name, "hiprtcGetCodeSize")?,
            rtc_get_code: resolve(&rtc, rtc_name, "hiprtcGetCode")?,
            rtc_destroy_program: resolve(&rtc, rtc_name, "hiprtcDestroyProgram")?,
            rtc_get_error_string: resolve(&rtc, rtc_name, "hiprtcGetErrorString")?,
            rtc_version: resolve(&rtc, rtc_name, "hiprtcVersion").ok(),
            _runtime: runtime,
            _rtc: rtc,
        }))
    }

    /// Whether the ROCm runtime resolves on this host. A caller choosing
    /// between drivers before any device work reads this.
    #[must_use]
    pub fn is_available() -> bool {
        Self::load().is_ok()
    }

    fn runtime_error(&self, call: &'static str, code: c_int) -> HipDriverError {
        // SAFETY: `hipGetErrorString` returns a NUL-terminated static string
        // for any code, including one it does not recognize.
        let message = unsafe { CStr::from_ptr((self.hip_get_error_string)(code)) }
            .to_string_lossy()
            .into_owned();
        HipDriverError::Runtime {
            call,
            code,
            message,
        }
    }

    fn rtc_error(&self, call: &'static str, code: c_int) -> HipDriverError {
        // SAFETY: same contract as `hipGetErrorString`.
        let message = unsafe { CStr::from_ptr((self.rtc_get_error_string)(code)) }
            .to_string_lossy()
            .into_owned();
        HipDriverError::Rtc {
            call,
            code,
            message,
        }
    }

    fn checked(&self, call: &'static str, status: c_int) -> Result<(), HipDriverError> {
        if status != HIP_SUCCESS {
            return Err(self.runtime_error(call, status));
        }
        Ok(())
    }

    /// `hipInit`.
    ///
    /// # Errors
    /// [`HipDriverError::Runtime`] when the runtime refuses to initialize.
    pub fn initialize(&self) -> Result<(), HipDriverError> {
        // SAFETY: `hipInit` takes a flags word that must be 0.
        let status = unsafe { (self.hip_init)(0) };
        self.checked("hipInit", status)
    }

    /// `hipGetDeviceCount`.
    ///
    /// # Errors
    /// [`HipDriverError::Runtime`] when the count cannot be read.
    pub fn device_count(&self) -> Result<i32, HipDriverError> {
        let mut count: c_int = 0;
        // SAFETY: `count` is a valid out-parameter.
        let status = unsafe { (self.hip_get_device_count)(&mut count) };
        self.checked("hipGetDeviceCount", status)?;
        Ok(count)
    }

    /// `hipDeviceGetName`.
    ///
    /// # Errors
    /// [`HipDriverError::Runtime`] when the name cannot be read.
    pub fn device_name(&self, ordinal: i32) -> Result<String, HipDriverError> {
        let mut buffer = [0 as c_char; 256];
        // SAFETY: `buffer` is 256 bytes, which is the length passed.
        let status = unsafe { (self.hip_device_get_name)(buffer.as_mut_ptr(), 256, ordinal) };
        self.checked("hipDeviceGetName", status)?;
        // SAFETY: `hipDeviceGetName` NUL-terminates within the buffer.
        Ok(unsafe { CStr::from_ptr(buffer.as_ptr()) }
            .to_string_lossy()
            .into_owned())
    }

    /// The device's `gcnArchName` when the runtime exposes
    /// `hipGetDevicePropertiesR0600`.
    #[must_use]
    pub fn device_arch(&self, ordinal: i32) -> Option<String> {
        let query = self.hip_get_device_properties?;
        let mut properties = [0_u64; HIP_DEVICE_PROPERTIES_WORDS];
        // SAFETY: the buffer is HIP_DEVICE_PROPERTIES_BYTES and the runtime
        // writes at most one R0600 hipDeviceProp_t (~1.5 KB) into it.
        let status = unsafe { query(properties.as_mut_ptr().cast::<c_void>(), ordinal) };
        if status != HIP_SUCCESS {
            return None;
        }
        // SAFETY: the call filled the initialized buffer; the scan reads only bytes.
        let bytes = unsafe {
            core::slice::from_raw_parts(
                properties.as_ptr().cast::<u8>(),
                HIP_DEVICE_PROPERTIES_BYTES,
            )
        };
        gcn_arch_name(bytes).map(str::to_owned)
    }

    /// `hipSetDevice`.
    ///
    /// # Errors
    /// [`HipDriverError::Runtime`] when the device cannot be selected.
    pub fn set_device(&self, ordinal: i32) -> Result<(), HipDriverError> {
        // SAFETY: callers pass an ordinal bounds-checked against the device count at open.
        let status = unsafe { (self.hip_set_device)(ordinal) };
        self.checked("hipSetDevice", status)
    }

    /// `hipDeviceSynchronize` on the calling thread's current device.
    ///
    /// # Errors
    /// [`HipDriverError::Runtime`] when synchronization fails.
    pub fn synchronize(&self) -> Result<(), HipDriverError> {
        // SAFETY: no arguments beyond the implicit current device.
        let status = unsafe { (self.hip_device_synchronize)() };
        self.checked("hipDeviceSynchronize", status)
    }

    /// `hipMemGetInfo`: `(free, total)` device bytes.
    ///
    /// # Errors
    /// [`HipDriverError::MemoryInfoUnavailable`] when the runtime does not export it, and
    /// [`HipDriverError::Runtime`] when the call fails.
    pub fn memory_info(&self) -> Result<(usize, usize), HipDriverError> {
        let query = self
            .hip_mem_get_info
            .ok_or(HipDriverError::MemoryInfoUnavailable)?;
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: both out-parameters are valid for the call.
        let status = unsafe { query(&mut free, &mut total) };
        self.checked("hipMemGetInfo", status)?;
        Ok((free, total))
    }

    /// `hipMalloc` of `bytes` on the current device.
    ///
    /// # Errors
    /// [`HipDriverError::Runtime`] when the allocation fails.
    pub fn allocate(&self, bytes: usize) -> Result<DevicePointer, HipDriverError> {
        let mut pointer = ptr::null_mut();
        // SAFETY: `pointer` is a valid out-parameter.
        let status = unsafe { (self.hip_malloc)(&mut pointer, bytes) };
        self.checked("hipMalloc", status)?;
        Ok(DevicePointer(pointer))
    }

    /// `hipFree`, ignoring the status: a `Drop` cannot report.
    ///
    /// # Safety
    /// `pointer` came from [`Self::allocate`] on the current device and is freed once.
    pub unsafe fn free(&self, pointer: DevicePointer) {
        // SAFETY: forwarded to the caller's contract.
        unsafe { (self.hip_free)(pointer.0) };
    }

    /// `hipMemset` to zero over `bytes`.
    ///
    /// # Safety
    /// `pointer` owns at least `bytes` bytes of device memory.
    ///
    /// # Errors
    /// [`HipDriverError::Runtime`] when the memset fails.
    pub unsafe fn zero(&self, pointer: DevicePointer, bytes: usize) -> Result<(), HipDriverError> {
        // SAFETY: forwarded to the caller's contract.
        let status = unsafe { (self.hip_memset)(pointer.0, 0, bytes) };
        self.checked("hipMemset", status)
    }

    fn copy(
        &self,
        destination: *mut c_void,
        source: *const c_void,
        bytes: usize,
        direction: c_int,
    ) -> Result<(), HipDriverError> {
        if bytes == 0 {
            return Ok(());
        }
        // SAFETY: every caller bounds-checked both sides against live allocations or host slices of `bytes`.
        let status = unsafe { (self.hip_memcpy)(destination, source, bytes, direction) };
        self.checked("hipMemcpy", status)
    }

    /// Copies `source` to `destination` on the device.
    ///
    /// # Safety
    /// `destination` owns at least `source.len()` bytes of device memory.
    ///
    /// # Errors
    /// [`HipDriverError::Runtime`] when the copy fails.
    pub unsafe fn copy_to_device(
        &self,
        destination: DevicePointer,
        source: &[u8],
    ) -> Result<(), HipDriverError> {
        self.copy(
            destination.0,
            source.as_ptr().cast::<c_void>(),
            source.len(),
            MEMCPY_HOST_TO_DEVICE,
        )
    }

    /// Fills `destination` from the device bytes at `source`.
    ///
    /// # Safety
    /// `source` owns at least `destination.len()` bytes of device memory.
    ///
    /// # Errors
    /// [`HipDriverError::Runtime`] when the copy fails.
    pub unsafe fn copy_from_device(
        &self,
        destination: &mut [MaybeUninit<u8>],
        source: DevicePointer,
    ) -> Result<(), HipDriverError> {
        self.copy(
            destination.as_mut_ptr().cast::<c_void>(),
            source.0.cast_const(),
            destination.len(),
            MEMCPY_DEVICE_TO_HOST,
        )
    }

    /// Copies `bytes` between two device addresses.
    ///
    /// # Safety
    /// Both sides own at least `bytes` bytes of live device memory.
    ///
    /// # Errors
    /// [`HipDriverError::Runtime`] when the copy fails.
    pub unsafe fn copy_within_device(
        &self,
        destination: DevicePointer,
        source: DevicePointer,
        bytes: usize,
    ) -> Result<(), HipDriverError> {
        self.copy(
            destination.0,
            source.0.cast_const(),
            bytes,
            MEMCPY_DEVICE_TO_DEVICE,
        )
    }

    /// `hipModuleLoadData` over a complete code object.
    ///
    /// # Errors
    /// [`HipDriverError::Runtime`] when the object cannot be loaded.
    pub fn load_module(&self, code: &[u8]) -> Result<ModuleHandle, HipDriverError> {
        let mut handle = ptr::null_mut();
        // SAFETY: `code` holds a complete code object; HIP copies it.
        let status =
            unsafe { (self.hip_module_load_data)(&mut handle, code.as_ptr().cast::<c_void>()) };
        self.checked("hipModuleLoadData", status)?;
        Ok(ModuleHandle(handle))
    }

    /// The entry point `name` inside `module`.
    ///
    /// # Safety
    /// `module` is live.
    ///
    /// # Errors
    /// [`HipDriverError::BindingContract`] for a name with an interior NUL and
    /// [`HipDriverError::Runtime`] when the module has no such symbol.
    pub unsafe fn module_function(
        &self,
        module: ModuleHandle,
        name: &str,
    ) -> Result<FunctionHandle, HipDriverError> {
        let Ok(name_c) = CString::new(name) else {
            return Err(HipDriverError::BindingContract);
        };
        let mut function = ptr::null_mut();
        // SAFETY: `module` is live per the caller; `name_c` is NUL-terminated.
        let status =
            unsafe { (self.hip_module_get_function)(&mut function, module.0, name_c.as_ptr()) };
        self.checked("hipModuleGetFunction", status)?;
        Ok(FunctionHandle(function))
    }

    /// `hipModuleUnload`, ignoring the status: a `Drop` cannot report.
    ///
    /// # Safety
    /// `module` came from [`Self::load_module`] and is unloaded once.
    pub unsafe fn unload_module(&self, module: ModuleHandle) {
        // SAFETY: forwarded to the caller's contract.
        unsafe { (self.hip_module_unload)(module.0) };
    }

    /// `hipModuleLaunchKernel` on the default stream.
    ///
    /// # Safety
    /// `parameters` matches the kernel's parameter list and every argument it names stays borrowed
    /// until this call returns.
    ///
    /// # Errors
    /// [`HipDriverError::Runtime`] when the launch is refused.
    pub unsafe fn launch(
        &self,
        function: FunctionHandle,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_bytes: u32,
        parameters: LaunchParameters,
    ) -> Result<(), HipDriverError> {
        // SAFETY: forwarded to the caller's contract.
        let status = unsafe {
            (self.hip_module_launch_kernel)(
                function.0,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                shared_bytes,
                ptr::null_mut(),
                parameters,
                ptr::null_mut(),
            )
        };
        self.checked("hipModuleLaunchKernel", status)
    }

    /// The `(major, minor)` of the hipRTC library, or `None` when it does not export
    /// `hiprtcVersion` or the call fails.
    #[must_use]
    pub fn compiler_version(&self) -> Option<(i32, i32)> {
        let query = self.rtc_version?;
        let (mut major, mut minor) = (0, 0);
        // SAFETY: both out-parameters are valid for the call.
        let status = unsafe { query(&mut major, &mut minor) };
        (status == RTC_SUCCESS).then_some((major, minor))
    }

    /// Compiles HIP C++ to code-object bytes with hipRTC under the resolved `options`.
    ///
    /// # Errors
    /// [`HipDriverError::CompileLog`] with the compiler's own log when the source is rejected, and
    /// [`HipDriverError::Rtc`] when hipRTC itself fails.
    pub fn compile_code_object(
        &self,
        source: &str,
        options: &[String],
    ) -> Result<Vec<u8>, HipDriverError> {
        let Ok(source_c) = CString::new(source) else {
            return Err(HipDriverError::BindingContract);
        };
        let Ok(name_c) = CString::new("proxima_hip") else {
            return Err(HipDriverError::BindingContract);
        };
        let mut program = ptr::null_mut();
        // SAFETY: all pointers are valid for the duration of the call; there
        // are no header sources, so those arrays are NULL/0.
        let status = unsafe {
            (self.rtc_create_program)(
                &mut program,
                source_c.as_ptr(),
                name_c.as_ptr(),
                0,
                ptr::null(),
                ptr::null(),
            )
        };
        if status != RTC_SUCCESS {
            return Err(self.rtc_error("hiprtcCreateProgram", status));
        }

        let options_c = options
            .iter()
            .filter_map(|option| CString::new(option.as_str()).ok())
            .collect::<Vec<_>>();
        let option_pointers = options_c
            .iter()
            .map(|option| option.as_ptr())
            .collect::<Vec<_>>();
        let option_count = c_int::try_from(option_pointers.len()).unwrap_or(c_int::MAX);
        // SAFETY: `program` is live; every option pointer is NUL-terminated
        // and outlives the call.
        let status =
            unsafe { (self.rtc_compile_program)(program, option_count, option_pointers.as_ptr()) };
        if status != RTC_SUCCESS {
            let log = self.program_log(program);
            // SAFETY: `program` is live and is destroyed exactly once.
            unsafe { (self.rtc_destroy_program)(&mut program) };
            return Err(HipDriverError::CompileLog { log });
        }

        let mut code_size: usize = 0;
        // SAFETY: `program` is live and `code_size` is a valid out-parameter.
        let status = unsafe { (self.rtc_get_code_size)(program, &mut code_size) };
        if status != RTC_SUCCESS {
            // SAFETY: `program` is live and is destroyed exactly once.
            unsafe { (self.rtc_destroy_program)(&mut program) };
            return Err(self.rtc_error("hiprtcGetCodeSize", status));
        }
        let mut code = vec![0u8; code_size];
        // SAFETY: the buffer is exactly `code_size` bytes, as the API expects.
        let status = unsafe { (self.rtc_get_code)(program, code.as_mut_ptr().cast::<c_char>()) };
        // SAFETY: `program` is live and is destroyed exactly once.
        unsafe { (self.rtc_destroy_program)(&mut program) };
        if status != RTC_SUCCESS {
            return Err(self.rtc_error("hiprtcGetCode", status));
        }
        Ok(code)
    }

    fn program_log(&self, program: *mut c_void) -> String {
        let mut size: usize = 0;
        // SAFETY: `program` is live and `size` is a valid out-parameter.
        let status = unsafe { (self.rtc_get_program_log_size)(program, &mut size) };
        if status != RTC_SUCCESS || size == 0 {
            return String::from("hipRTC produced no log");
        }
        let mut log = vec![0 as c_char; size];
        // SAFETY: the buffer is exactly `size` bytes, which is what the API writes.
        let status = unsafe { (self.rtc_get_program_log)(program, log.as_mut_ptr()) };
        if status != RTC_SUCCESS {
            return String::from("hipRTC produced an unreadable log");
        }
        // SAFETY: the log is NUL-terminated within `size` bytes.
        unsafe { CStr::from_ptr(log.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }
}

/// The `gcnArchName` token inside a `hipDeviceProp_tR0600` image: the first
/// NUL-terminated `gfx...` run of architecture-name bytes. Scanning rather
/// than declaring the ~1.5 KB struct keeps this independent of the header's
/// versioned layout.
pub fn gcn_arch_name(properties: &[u8]) -> Option<&str> {
    properties
        .windows(3)
        .position(|window| window == b"gfx")
        .and_then(|start| {
            let rest = &properties[start..];
            let end = rest.iter().position(|byte| *byte == 0)?;
            let token = core::str::from_utf8(&rest[..end]).ok()?;
            let suffix = token.get(3..)?;
            let shaped = !suffix.is_empty()
                && suffix.chars().all(|value| {
                    value.is_ascii_alphanumeric() || matches!(value, ':' | '-' | '+' | '_')
                });
            shaped.then_some(token)
        })
}
