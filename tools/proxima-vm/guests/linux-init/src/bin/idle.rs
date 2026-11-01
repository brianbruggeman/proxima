fn main() {
    linux_guest::mark_init_started();
    loop {
        unsafe { libc::pause() };
    }
}
