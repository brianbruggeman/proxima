use std::path::PathBuf;
use std::sync::Arc;

use bytes::Bytes;
use futures::StreamExt;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::request::Request;
use proxima_recording::BinSource;
use proxima_recording::replay::ReplayUpstream;
use proxima_recording::source::DynRecordingSource;

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
        let source: DynRecordingSource = Arc::new(BinSource::new(&path, runtime));
        let replay = ReplayUpstream::from_source(source, "intercept-replay").await?;

        println!("replay loaded from {}", path.display());
        let mut keys = replay.known_keys();
        keys.sort();
        println!("known match keys ({}):", keys.len());
        for key in &keys {
            println!("  {key}");
        }

        let key_to_replay = match keys.first() {
            Some(key) => key.clone(),
            None => {
                println!("no recorded interactions to replay");
                return Ok::<(), proxima_core::ProximaError>(());
            }
        };

        let (method, rest) = key_to_replay
            .split_once(' ')
            .ok_or_else(|| proxima_core::ProximaError::Config("malformed match key".into()))?;
        let (path_only, query_str) = match rest.split_once('?') {
            Some((path, query)) => (path, query),
            None => (rest, ""),
        };

        // the match key fingerprints method+path+query, so the replay request
        // must carry the query params or it can never match a recorded request
        // that had them (every real api call does).
        let mut builder = Request::builder()
            .method(method.as_bytes())
            .path(path_only.as_bytes().to_vec());
        for pair in query_str.split('&').filter(|part| !part.is_empty()) {
            let (name, value) = pair.split_once('=').unwrap_or((pair, ""));
            builder = builder.query_param(name.as_bytes().to_vec(), value.as_bytes().to_vec());
        }
        let request = builder.build()?;
        println!("\nreplaying {method} {path_only}?{query_str}...");

        let response = replay.call(request).await?;
        println!("response status: {}", response.status);
        println!("response headers ({}):", response.metadata.iter().count());
        for (name, value) in response.metadata.iter() {
            let name_str = std::str::from_utf8(name).unwrap_or("<binary>");
            let value_str = std::str::from_utf8(value).unwrap_or("<binary>");
            println!("  {name_str}: {value_str}");
        }

        let mut body_stream = response.into_chunk_stream();
        let mut total: usize = 0;
        let mut first_preview: Option<String> = None;
        while let Some(item) = body_stream.next().await {
            let chunk: Bytes = item?;
            total += chunk.len();
            if first_preview.is_none() {
                let preview: String = std::str::from_utf8(&chunk)
                    .unwrap_or("<binary>")
                    .chars()
                    .take(120)
                    .collect();
                first_preview = Some(preview);
            }
        }
        println!("response body: {total} bytes total");
        if let Some(preview) = first_preview {
            println!("preview: {preview}");
        }
        Ok(())
    })
}
