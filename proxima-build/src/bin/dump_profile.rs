//! Verification binary. Resolves the active profile and dumps it to stdout.
//!
//! Use: `PROXIMA_PROFILE=linux-daemon cargo run -p proxima-build --bin dump-profile`

fn main() {
    let resolved = match proxima_build::resolve_profile() {
        Ok(resolved) => resolved,
        Err(err) => {
            eprintln!("resolve failed: {err}");
            std::process::exit(1);
        }
    };

    println!("source: {}", resolved.profile_file.display());
    println!();
    let profile = &resolved.profile;
    println!("schema        = {}", profile.schema);
    println!("alloc         = {}", profile.alloc);
    println!("std           = {}", profile.std);
    println!("executor      = {}", profile.executor);
    println!("reactor       = {}", profile.reactor);
    println!("tls           = {}", profile.tls);
    println!("timer         = {}", profile.timer);
    println!("quic_enabled  = {}", profile.quic_enabled);
    println!("quic_impl     = {}", profile.quic_impl);
    println!("h3_impl       = {}", profile.h3_impl);
}
