//! The kernel-parameter array every card-C runtime launches with.
//!
//! A driver-API launch (`hipModuleLaunchKernel`, `cuLaunchKernel`) takes a
//! `void**`: one entry per kernel argument, each entry the ADDRESS of the
//! storage holding that argument's value. The driver reads through those
//! addresses at launch, so each one must still point at the argument, unmoved
//! and alive, when the call runs.
//!
//! [`KernelParams`] makes that a borrow rule instead of a convention. Every
//! argument is pushed by shared reference, and the params carry the borrow
//! for as long as they live, so moving or dropping an argument before the
//! launch returns is a compile error. Build the params in the function that
//! performs the launch, after every argument sits where it will stay.
//!
//! ```compile_fail,E0505
//! use omega::kernel_params::{KernelArgument, KernelParams};
//!
//! struct Buffer(*mut core::ffi::c_void);
//! // SAFETY: the slot is the address of the field holding the pointer, valid while `self` lives.
//! unsafe impl KernelArgument for Buffer {
//!     fn slot(&self) -> *mut core::ffi::c_void {
//!         (&raw const self.0).cast_mut().cast()
//!     }
//! }
//!
//! let buffer = Buffer(core::ptr::null_mut());
//! let mut params = KernelParams::new();
//! params.push(&buffer);
//! let moved = buffer; // error[E0505]: cannot move out of `buffer` while borrowed
//! let _ = params.as_mut_ptr();
//! drop(moved);
//! ```
//!
//! Teaching pointer: this is the one place a launch's parameter pointers are
//! made, for every runtime. A runtime implements [`KernelArgument`] for its
//! device-buffer type and calls [`KernelParams::as_mut_ptr`] inside its own
//! `unsafe` FFI call; it never builds a `*mut c_void` array by hand.

use alloc::vec::Vec;
use core::ffi::c_void;
use core::marker::PhantomData;

/// A value a kernel receives as one parameter.
///
/// # Safety
/// [`slot`](Self::slot) must return the address of storage inside `self` that holds the argument's
/// value exactly as the kernel reads it, and that address must stay valid and unchanged for as long
/// as `self` is borrowed. The driver dereferences it at launch; an address of a temporary or of
/// anything else makes the launch read freed or foreign memory.
pub unsafe trait KernelArgument {
    /// The address of the storage that holds this argument's value as the
    /// kernel reads it (for a device buffer, the address of its stored device
    /// pointer, not the device pointer itself). It borrows `self`, so it is
    /// valid exactly while `self` is neither moved nor dropped.
    fn slot(&self) -> *mut c_void;
}

/// The `void**` parameter array of one launch, holding a borrow on every
/// argument pushed into it.
#[derive(Debug)]
pub struct KernelParams<'arguments> {
    slots: Vec<*mut c_void>,
    borrowed: PhantomData<&'arguments ()>,
}

impl<'arguments> KernelParams<'arguments> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            borrowed: PhantomData,
        }
    }

    #[must_use]
    pub fn with_capacity(arguments: usize) -> Self {
        Self {
            slots: Vec::with_capacity(arguments),
            borrowed: PhantomData,
        }
    }

    /// Appends the next kernel parameter, borrowing `argument` until these
    /// params are dropped.
    pub fn push<Argument: KernelArgument>(&mut self, argument: &'arguments Argument) -> &mut Self {
        self.slots.push(argument.slot());
        self
    }

    /// Appends a scalar kernel parameter: its slot is its own address, so the
    /// scalar is borrowed until these params are dropped.
    pub fn push_value<Value: Copy>(&mut self, value: &'arguments Value) -> &mut Self {
        self.slots.push(core::ptr::from_ref(value).cast_mut().cast());
        self
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// The parameter slots in launch order.
    #[must_use]
    pub fn slots(&self) -> &[*mut c_void] {
        &self.slots
    }

    /// The `void**` to hand the launch call. Valid while `self` is alive; the
    /// launch must return before `self` or any pushed argument is dropped.
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut *mut c_void {
        self.slots.as_mut_ptr()
    }
}

impl Default for KernelParams<'_> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use alloc::vec;

    use super::*;

    struct Argument {
        pointer: *mut c_void,
        bytes: usize,
    }

    impl Argument {
        fn of(address: usize, bytes: usize) -> Self {
            Self {
                pointer: address as *mut c_void,
                bytes,
            }
        }
    }

    // SAFETY: the slot is the address of the `pointer` field, which lives as long as `self`.
    unsafe impl KernelArgument for Argument {
        fn slot(&self) -> *mut c_void {
            (&raw const self.pointer).cast_mut().cast()
        }
    }

    fn arguments() -> Vec<Argument> {
        // two inputs of different sizes, the output, the uniforms and a u32 fault array
        vec![
            Argument::of(0x1000, 4096 * 4),
            Argument::of(0x9000, 64 * 4),
            Argument::of(0x20000, 4096 * 4),
            Argument::of(0x30000, 48),
            Argument::of(0x40000, 7 * 4),
        ]
    }

    #[test]
    fn kernel_params_point_at_borrowed_arguments() {
        let held = arguments();
        let mut params = KernelParams::with_capacity(held.len());
        for argument in &held {
            params.push(argument);
        }

        assert_eq!(params.len(), held.len());
        for (slot, argument) in params.slots().iter().zip(&held) {
            assert_eq!(*slot, (&raw const argument.pointer).cast_mut().cast::<c_void>());
        }
        for (argument, bytes) in held.iter().zip([16384, 256, 16384, 48, 28]) {
            assert_eq!(argument.bytes, bytes, "the arguments differ in size, as a real launch's do");
        }
    }

    #[test]
    fn kernel_params_still_point_at_their_arguments_after_the_params_move() {
        let held = arguments();
        let mut params = KernelParams::new();
        for argument in &held {
            params.push(argument);
        }

        let mut moved = vec![params];
        let launch = moved.remove(0);

        for (slot, argument) in launch.slots().iter().zip(&held) {
            assert_eq!(*slot, (&raw const argument.pointer).cast_mut().cast::<c_void>());
        }
    }

    #[test]
    fn the_launch_pointer_reads_back_each_device_pointer_in_order() {
        let held = arguments();
        let mut params = KernelParams::new();
        for argument in &held {
            params.push(argument);
        }

        let array = params.as_mut_ptr();

        for (index, argument) in held.iter().enumerate() {
            // SAFETY: `array` points at `held.len()` live slots and `index < held.len()`.
            let slot = unsafe { *array.add(index) };
            // SAFETY: each slot is the address of a live `*mut c_void` field of `held`.
            let device_pointer = unsafe { *slot.cast::<*mut c_void>() };
            assert_eq!(device_pointer, argument.pointer);
        }
    }

    #[test]
    fn kernel_params_read_back_a_scalar_after_a_buffer_in_launch_order() {
        let buffer = Argument::of(0x7000, 64);
        let rows: u32 = 48;
        let mut params = KernelParams::new();
        params.push(&buffer).push_value(&rows);

        let array = params.as_mut_ptr();

        // SAFETY: `array` holds two live slots: the buffer's pointer field, then `rows`.
        let (pointer, scalar) = unsafe { (*(*array).cast::<*mut c_void>(), *(*array.add(1)).cast::<u32>()) };
        assert_eq!(pointer, buffer.pointer);
        assert_eq!(scalar, rows);
    }

    #[test]
    fn an_empty_launch_has_no_parameters() {
        let params = KernelParams::default();

        assert!(params.is_empty());
        assert_eq!(params.slots().len(), 0);
    }

    #[test]
    fn the_slot_is_the_stored_pointer_not_the_pointer_it_holds() {
        let argument = Argument::of(0x5000, 8);

        let slot = argument.slot();

        assert_ne!(slot, argument.pointer, "the driver needs the address of the pointer");
        assert_eq!(slot as usize, (&raw const argument.pointer) as usize);
    }
}
