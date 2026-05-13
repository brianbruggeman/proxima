use std::path::PathBuf;

use futures::StreamExt;
use proxima_recording::BinSource;
use proxima_recording::event::{HttpEvent, ProtocolEvent};
use proxima_recording::source::RecordingSource;

fn main() -> Result<(), proxima_core::ProximaError> {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/tmp".into()))
                .join(".proxima")
                .join("intercept.bin")
        });

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| proxima_core::ProximaError::Config(format!("runtime: {err}")))?;
    let runtime = proxima::offline_runtime()?;
    rt.block_on(async move {
        let source = BinSource::new(&path, runtime);
        let mut stream = source.events();
        let mut count: usize = 0;
        let mut chunk_count: usize = 0;
        let mut total_chunk_bytes: usize = 0;
        while let Some(item) = stream.next().await {
            match item {
                Ok(event) => {
                    count += 1;
                    match &event.event {
                        ProtocolEvent::Http(HttpEvent::Started {
                            ts: _,
                            pipe,
                            request,
                            meta: _,
                        }) => {
                            println!(
                                "[{count}] HttpEvent::Started pipe={pipe} {} {}",
                                request.method, request.path
                            );
                            for (name, value) in &request.headers {
                                println!("    {name}: {value}");
                            }
                        }
                        ProtocolEvent::Http(HttpEvent::RequestChunk { data, metadata: _ }) => {
                            chunk_count += 1;
                            total_chunk_bytes += data.len();
                            let preview = std::str::from_utf8(data)
                                .map(|text| text.chars().take(60).collect::<String>())
                                .unwrap_or_else(|_| format!("<{} non-utf8 bytes>", data.len()));
                            println!(
                                "[{count}] HttpEvent::RequestChunk {} bytes: {preview}",
                                data.len()
                            );
                        }
                        ProtocolEvent::Http(HttpEvent::RequestEnded) => {
                            println!("[{count}] HttpEvent::RequestEnded");
                        }
                        ProtocolEvent::Http(HttpEvent::ResponseStarted { status, headers }) => {
                            println!(
                                "[{count}] HttpEvent::ResponseStarted status={status} headers={}",
                                headers.len()
                            );
                            for (name, value) in headers {
                                println!("    {name}: {value}");
                            }
                        }
                        ProtocolEvent::Http(HttpEvent::ResponseChunk { data, metadata: _ }) => {
                            chunk_count += 1;
                            total_chunk_bytes += data.len();
                            let preview = std::str::from_utf8(data)
                                .map(|text| text.chars().take(60).collect::<String>())
                                .unwrap_or_else(|_| format!("<{} non-utf8 bytes>", data.len()));
                            println!(
                                "[{count}] HttpEvent::ResponseChunk {} bytes: {preview}",
                                data.len()
                            );
                        }
                        ProtocolEvent::Http(HttpEvent::Ended {
                            latency_ms,
                            meta: _,
                        }) => {
                            println!("[{count}] HttpEvent::Ended latency_ms={latency_ms}");
                        }
                        other => println!("[{count}] {other:?}"),
                    }
                }
                Err(err) => {
                    eprintln!("read error: {err}");
                    break;
                }
            }
        }
        println!("---");
        println!("total events: {count}, chunks: {chunk_count}, chunk bytes: {total_chunk_bytes}");
        Ok::<(), proxima_core::ProximaError>(())
    })
}
