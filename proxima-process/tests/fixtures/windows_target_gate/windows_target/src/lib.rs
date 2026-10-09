#[cfg(not(target_os = "windows"))]
compile_error!("this fixture must be checked for a Windows target");

#[cfg(target_os = "windows")]
pub fn windows_command_target_gate() {
    use std::os::windows::process::CommandExt;

    let mut command = std::process::Command::new("cmd.exe");
    command.creation_flags(0);
}
