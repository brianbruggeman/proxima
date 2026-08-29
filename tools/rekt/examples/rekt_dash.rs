// live observability dashboard for the closed-loop drives. renders a scrolling
// hits/errors graph + a human-readable rps readout while the run is in flight.
//
//   cargo run --release --features tui --example rekt_dash -- \
//     <target> <connections_per_core> <duration_secs> <cores> [flags]
//
// protocol (default h1):
//   --adaptive            h1, hillclimb the per-core concurrency from the seed
//   --h2                  multiplexed http/2 (target is http://host:port/)
//   --h3                  multiplexed native-quic http/3 (target is host:port)
//   --grpc                grpc over multiplexed h2 (target is http://host:port/)
// tuning:
//   --streams N           streams per connection for h2/h3/grpc (default 64/100/64)
//   --server-name NAME    tls server name for h3 (default localhost)
//   --path PATH           grpc method path (default /helloworld.Greeter/SayHello)
//
//   rekt_dash http://127.0.0.1:8080/ 16 20 4
//   rekt_dash http://127.0.0.1:8090/ 4 20 4 --h2 --streams 64
//   rekt_dash 127.0.0.1:8094 8 20 4 --h3 --streams 100 --server-name localhost
//   rekt_dash http://127.0.0.1:8100/ 4 20 4 --grpc --path /helloworld.Greeter/SayHello
//
// q / esc / ctrl-c quits (dismisses the frozen final frame).

use std::time::Duration;

use rekt::tui::{DashboardConfig, Mode, run};

enum Protocol {
    H1FlatOut,
    H1Adaptive,
    H2,
    H3,
    Grpc,
}

fn main() {
    let mut positional = Vec::new();
    let mut protocol = Protocol::H1FlatOut;
    let mut streams: Option<usize> = None;
    let mut server_name = "localhost".to_string();
    let mut path = "/helloworld.Greeter/SayHello".to_string();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--adaptive" => protocol = Protocol::H1Adaptive,
            "--h2" => protocol = Protocol::H2,
            "--h3" => protocol = Protocol::H3,
            "--grpc" => protocol = Protocol::Grpc,
            "--streams" => streams = args.next().and_then(|value| value.parse().ok()),
            "--server-name" => server_name = args.next().unwrap_or(server_name),
            "--path" => path = args.next().unwrap_or(path),
            _ => positional.push(arg),
        }
    }

    let target = positional.first().cloned().unwrap_or_else(|| "http://127.0.0.1:8080/".to_string());
    let connections: usize = positional.get(1).and_then(|value| value.parse().ok()).unwrap_or(8);
    let seconds: u64 = positional.get(2).and_then(|value| value.parse().ok()).unwrap_or(10);
    let cores: usize = positional.get(3).and_then(|value| value.parse().ok()).unwrap_or(1);

    let mode = match protocol {
        Protocol::H1FlatOut => Mode::H1FlatOut { connections_per_core: connections },
        Protocol::H1Adaptive => Mode::H1Adaptive { seed: connections },
        Protocol::H2 => Mode::H2 { connections_per_core: connections, streams_per_conn: streams.unwrap_or(64) },
        Protocol::H3 => Mode::H3 {
            connections_per_core: connections,
            streams_per_conn: streams.unwrap_or(100),
            server_name,
        },
        Protocol::Grpc => Mode::Grpc {
            connections_per_core: connections,
            streams_per_conn: streams.unwrap_or(64),
            path,
        },
    };
    let config = DashboardConfig { target, cores, duration: Duration::from_secs(seconds), mode };

    match run(config) {
        Ok(throughput) => {
            println!(
                "rekt: {} completed, {} errors over {}s ({:.0} req/s)",
                throughput.completed,
                throughput.errors,
                seconds,
                throughput.per_sec(),
            );
        }
        Err(error) => {
            eprintln!("rekt_dash: {error}");
            std::process::exit(1);
        }
    }
}
