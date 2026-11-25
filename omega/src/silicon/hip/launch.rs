use crate::gpu_runtime::{DeviceElement, LaunchArguments};
use crate::kernel_params::KernelParams;

use super::module::HipFunction;
use super::runtime::HipRuntime;
use super::window::{HipSlice, HipWindow, HipWindowMut, WindowRefusal};

/// The borrowed parameter array of one HIP launch: every argument is borrowed by the
/// [`KernelParams`] until the launch returns.
pub struct HipLaunch<'launch> {
    pub(super) function: &'launch HipFunction,
    pub(super) parameters: KernelParams<'launch>,
    pub(super) refusal: Option<WindowRefusal>,
}

impl<'launch> LaunchArguments<'launch, HipRuntime> for HipLaunch<'launch> {
    fn push<Element: DeviceElement>(&mut self, buffer: &'launch HipSlice<Element>) -> &mut Self {
        self.parameters.push(&*buffer.buffer);
        self
    }

    fn push_mut<Element: DeviceElement>(
        &mut self,
        buffer: &'launch mut HipSlice<Element>,
    ) -> &mut Self {
        self.parameters.push(&*buffer.buffer);
        self
    }

    fn push_view<Element: DeviceElement>(
        &mut self,
        view: &'launch HipWindow<'_, Element>,
    ) -> &mut Self {
        self.refusal = self.refusal.or(view.refusal);
        self.parameters.push(view);
        self
    }

    fn push_view_mut<Element: DeviceElement>(
        &mut self,
        view: &'launch mut HipWindowMut<'_, Element>,
    ) -> &mut Self {
        self.refusal = self.refusal.or(view.refusal);
        self.parameters.push(&*view);
        self
    }

    fn push_value<Element: DeviceElement>(&mut self, value: &'launch Element) -> &mut Self {
        self.parameters.push_value(value);
        self
    }
}
