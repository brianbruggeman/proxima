fn main() {
    linux_guest::mark_init_started();
    loop {
        std::hint::spin_loop();
    }
}
