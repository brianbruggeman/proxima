//! Bringing up a new target is not a runtime switch — it is a `build.rs`
//! reading a TOML file named by `PROXIMA_PROFILE` and baking the answer
//! into `pub const`s. There is no `if` branching on a config value at
//! runtime here: whichever profile built this binary is the only profile
//! this binary knows about.
//!
//! Run it twice, once per profile, and diff the output:
//!
//!   PROXIMA_PROFILE=linux-daemon cargo run --example-path examples/new-platform
//!   PROXIMA_PROFILE=bare-metal   cargo run --example-path examples/new-platform
//!
//! (see the README beside this file for the exact `cargo run -p` invocation)

mod profile {
    include!(concat!(env!("OUT_DIR"), "/proxima_profile.rs"));
}

fn main() {
    println!("{}", profile_source());
    println!();
    println!("schema        = {}", profile::SCHEMA);
    println!("alloc         = {}", profile::ALLOC);
    println!("std           = {}", profile::STD);
    println!("executor      = {}", profile::EXECUTOR);
    println!("reactor       = {}", profile::REACTOR);
    println!("tls           = {}", profile::TLS);
    println!("timer         = {}", profile::TIMER);
    println!("quic_enabled  = {}", profile::QUIC_ENABLED);
    println!("quic_impl     = {}", profile::QUIC_IMPL);
    println!("h3_impl       = {}", profile::H3_IMPL);
}

/// The first line of the generated module is a comment recording which
/// profile file produced these constants; surface it so the two runs are
/// visibly different, not just the numbers.
fn profile_source() -> &'static str {
    include_str!(concat!(env!("OUT_DIR"), "/proxima_profile.rs"))
        .lines()
        .find(|line| line.starts_with("// source:"))
        .unwrap_or("// source: <PROXIMA_PROFILE unset — Profile::default()>")
}
