use super::error::DpdkError;
use super::ffi;
use std::ffi::CString;
use std::os::raw::{c_char, c_int};

/// Owns the dpdk Environment Abstraction Layer for the process lifetime.
///
/// `rte_eal_init` is process-global and must run exactly once; this guard makes
/// that a move-once resource and calls `rte_eal_cleanup` on drop. The `CString`
/// argv backing store is kept alive because dpdk may retain pointers into it.
pub struct Eal {
    _argv_storage: Vec<CString>,
    _argv_ptrs: Vec<*mut c_char>,
}

impl Eal {
    /// Initialise the EAL from a command line. `args[0]` is the program name;
    /// the rest are dpdk options (`-l`, `--vdev`, `--no-pci`, `-d`, ...).
    ///
    /// # Errors
    /// Returns [`DpdkError::ArgNul`] if any argument has an interior nul,
    /// [`DpdkError::TooManyArgs`] if the count exceeds `c_int`, or
    /// [`DpdkError::EalInit`] with dpdk's negative code if init fails.
    pub fn init(args: &[&str]) -> Result<Self, DpdkError> {
        let storage: Vec<CString> = args
            .iter()
            .map(|arg| CString::new(*arg).map_err(|_| DpdkError::ArgNul))
            .collect::<Result<_, _>>()?;

        // dpdk takes argv as *mut*mut c_char and may permute it in place.
        let mut ptrs: Vec<*mut c_char> = storage.iter().map(|s| s.as_ptr().cast_mut()).collect();
        let arg_count = c_int::try_from(ptrs.len()).map_err(|_| DpdkError::TooManyArgs)?;

        let consumed = unsafe { ffi::eal_init(arg_count, ptrs.as_mut_ptr()) };
        if consumed < 0 {
            return Err(DpdkError::EalInit(consumed));
        }
        // version gate: warn if the runtime .so differs from the headers we
        // generated offsets against (the abi self-check at pool create is the
        // hard guard; this names the likely cause early).
        if !ffi::version_matches_build() {
            eprintln!(
                "warn: runtime {} differs from the dpdk {} this was built against; abi self-check will verify layout",
                ffi::version(),
                ffi::version_built(),
            );
        }
        Ok(Self {
            _argv_storage: storage,
            _argv_ptrs: ptrs,
        })
    }
}

impl Drop for Eal {
    fn drop(&mut self) {
        unsafe { ffi::eal_cleanup() }
    }
}
