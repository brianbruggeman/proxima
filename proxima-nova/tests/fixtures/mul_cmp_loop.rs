fn main() {
    let (mut i, mut s) = (0i32, 0i32);
    while i < 10 {
        s += i * i;
        i += 1;
    }
    std::process::exit(s - 200)
}
