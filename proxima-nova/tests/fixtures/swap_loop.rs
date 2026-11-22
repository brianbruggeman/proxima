fn main() {
    let (mut x, mut y, mut acc, mut n) = (3i32, 5i32, 0i32, 7i32);
    while n != 0 {
        let n1 = n - 1;
        let acc1 = acc + y;
        (x, y, acc, n) = (y, x, acc1, n1);
    }
    std::process::exit(acc + x - y + 100)
}
