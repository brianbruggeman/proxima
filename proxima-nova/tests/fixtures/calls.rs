#[inline(never)]
fn smash() {}

#[inline(never)]
fn square(x: i32) -> i32 {
    x * x
}

#[inline(never)]
fn add3(a: i32, b: i32, c: i32) -> i32 {
    a + b + c
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn sum8(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32, g: i32, h: i32) -> i32 {
    a + b + c + d + e + f + g + h
}

fn main() {
    let argc = std::env::args().count() as i32;
    let a = argc + 4;
    let b = square(a);
    smash();
    let c = add3(a, b, 12);
    let d = sum8(1, 2, 3, 4, 5, 6, 7, 8);
    std::process::exit(c + a + (argc - 9).abs() + d - 36 + argc - 1)
}
