use std::ffi::CStr;
use std::io;
use std::os::windows::io::{AsHandle, OwnedHandle};
use std::process::{
    Child as NativeChild, ChildStderr, ChildStdin, ChildStdout, Command as NativeCommand,
    Stdio as NativeStdio,
};

use proxima_primitives::pipe::ProximaError;

use super::command::Output;
use super::descriptor::{CommandDescriptor, Stdio};

#[derive(Debug)]
pub struct Child {
    inner: NativeChild,
    pub stdin: Option<ChildStdin>,
    pub stdout: Option<ChildStdout>,
    pub stderr: Option<ChildStderr>,
}

impl Child {
    pub(crate) fn from_command(command: &mut NativeCommand) -> Result<Self, ProximaError> {
        let mut inner = command.spawn().map_err(process_error)?;
        Ok(Self {
            stdin: inner.stdin.take(),
            stdout: inner.stdout.take(),
            stderr: inner.stderr.take(),
            inner,
        })
    }

    #[must_use]
    pub fn pid(&self) -> u32 {
        self.inner.id()
    }

    pub fn wait(&mut self) -> Result<i32, ProximaError> {
        self.stdin.take();
        self.inner
            .wait()
            .map(|status| status.code().unwrap_or(-1))
            .map_err(process_error)
    }

    pub fn try_wait(&mut self) -> Result<Option<i32>, ProximaError> {
        self.inner
            .try_wait()
            .map(|status| status.map(|status| status.code().unwrap_or(-1)))
            .map_err(process_error)
    }

    pub fn kill(&mut self) -> Result<(), ProximaError> {
        if self.try_wait()?.is_some() {
            return Ok(());
        }
        self.inner.kill().map_err(process_error)
    }

    pub(crate) fn clone_process_handle(&mut self) -> Result<OwnedHandle, ProximaError> {
        match self.inner.as_handle().try_clone_to_owned() {
            Ok(handle) => Ok(handle),
            Err(error) => {
                let _ = self.kill();
                Err(process_error(error))
            }
        }
    }

    pub(crate) fn collect(mut self) -> Result<Output, ProximaError> {
        self.stdin.take();
        self.inner.stdout = self.stdout.take();
        self.inner.stderr = self.stderr.take();
        let output = self.inner.wait_with_output().map_err(process_error)?;
        Ok(Output {
            status: output.status.code().unwrap_or(-1),
            stdout: output.stdout,
            stderr: output.stderr,
        })
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SpawnOptions {
    pub dispatch_fd: Option<i32>,
    pub controlling_tty: bool,
    pub umask: Option<u32>,
}

pub fn spawn(descriptor: &CommandDescriptor, options: SpawnOptions) -> Result<Child, ProximaError> {
    if options.dispatch_fd.is_some() || options.controlling_tty || options.umask.is_some() {
        return Err(process_error(io::Error::new(
            io::ErrorKind::Unsupported,
            "POSIX dispatch, terminal and umask options require Unix",
        )));
    }
    let mut command = NativeCommand::new(descriptor_text(&descriptor.program)?);
    command.env_clear();
    for argument in &descriptor.args {
        command.arg(descriptor_text(argument)?);
    }
    if let Some(directory) = &descriptor.current_dir {
        command.current_dir(descriptor_text(directory)?);
    }
    for entry in &descriptor.env {
        command.env(descriptor_text(&entry.key)?, descriptor_text(&entry.value)?);
    }
    command
        .stdin(native_stdio(descriptor.stdin)?)
        .stdout(native_stdio(descriptor.stdout)?)
        .stderr(native_stdio(descriptor.stderr)?);
    Child::from_command(&mut command)
}

pub(crate) fn native_stdio(stdio: Stdio) -> Result<NativeStdio, ProximaError> {
    match stdio {
        Stdio::Inherit => Ok(NativeStdio::inherit()),
        Stdio::Null => Ok(NativeStdio::null()),
        Stdio::Piped => Ok(NativeStdio::piped()),
        Stdio::Fd(_) => Err(process_error(io::Error::new(
            io::ErrorKind::Unsupported,
            "POSIX file descriptors require Unix",
        ))),
    }
}

fn descriptor_text(value: &CStr) -> Result<&str, ProximaError> {
    value.to_str().map_err(|error| {
        ProximaError::Body(format!(
            "Windows descriptors require UTF-8; use Command for native OS strings: {error}"
        ))
    })
}

pub(crate) fn process_error(error: io::Error) -> ProximaError {
    ProximaError::Io(error)
}
