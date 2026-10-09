use std::os::unix::process::CommandExt;

pub fn unix_only_command_extension() {
    let mut command = std::process::Command::new("cargo-val-negative-control");
    unsafe {
        command.pre_exec(|| Ok(()));
    }
}
