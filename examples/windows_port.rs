use std::io;

#[cfg(all(feature = "serve-prime", feature = "http-prime-deps"))]
#[path = "support/windows_port.rs"]
mod support;

#[cfg(all(feature = "serve-prime", feature = "http-prime-deps"))]
fn main() -> io::Result<()> {
    let corrupt = std::env::args().any(|argument| argument == "--corrupt-payload");
    support::incumbent_exchange(corrupt)?;
    println!("incumbent_payload_matches=2");
    support::tcp_exchange()?;
    println!("tcp_payload_matches=1");
    support::udp_exchange()?;
    println!("udp_payload_matches=1");
    #[cfg(feature = "http1-native")]
    {
        support::multiworker_listener_exchange()?;
        println!("multiworker_listener_matches=1");
    }
    Ok(())
}

#[cfg(not(all(feature = "serve-prime", feature = "http-prime-deps")))]
fn main() -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "windows_port requires serve-prime and http-prime-deps",
    ))
}
