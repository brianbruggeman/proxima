use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

const LISTEN: &str = "0.0.0.0:80";
const BODY: &str = "serve-once-init\n";

// the host dials and hangs up once to learn the guest is listening; only a connection that sends bytes is the request
fn accept_request(listener: &TcpListener) -> std::io::Result<TcpStream> {
    loop {
        let (mut stream, _) = listener.accept()?;
        let mut head = [0u8; 1024];
        if stream.read(&mut head)? > 0 {
            return Ok(stream);
        }
    }
}

fn serve()-> std::io::Result<()> {
    let _network = linux_guest::network::bring_up("serve-once-init")?;
    let listener = TcpListener::bind(LISTEN)?;
    let mut stream = accept_request(&listener)?;
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{BODY}",
        BODY.len()
    );
    stream.write_all(response.as_bytes())
}

fn main() {
    linux_guest::mark_init_started();
    if let Err(error) = serve() {
        eprintln!("serve-once-init: {error}");
    }
    linux_guest::power_off()
}
