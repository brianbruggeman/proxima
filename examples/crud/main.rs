//! proxima *is* the origin: a small REST service.
//!
//! Every earlier example puts proxima in front of something else (a
//! transform, a filter, an upstream). Here proxima answers directly: four
//! `SendPipe` handlers mounted by method + path, sharing one in-memory
//! store. Routing is `App::mount_with_methods` (the same method+path
//! admit-or-not decision `filter` taught you, wired into the router instead
//! of a standalone gate); each handler is the same `Pipe` shape `transform`
//! taught you, `Request<Bytes> -> Response<Bytes>`.
//!
//! Run: `cargo run --example crud`

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::fmt;
use std::future::Future;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use bytes::Bytes;
use proxima::mount::MethodFilter;
use proxima::shutdown::ShutdownBarrier;
use proxima::{
    App, ListenerSpec, PipeHandle, ProximaError, Request, Response, SendPipe,
    into_handle,
};

const BIND: &str = "127.0.0.1:8080";

/// Shared resource store. Every handler holds an `Arc` clone, so cloning a
/// pipe (which the router does once per mount) shares state instead of
/// forking it.
#[derive(Clone)]
struct Store {
    items: Arc<Mutex<BTreeMap<u64, Bytes>>>,
    next_id: Arc<AtomicU64>,
}

impl Store {
    fn new() -> Self {
        Self {
            items: Arc::new(Mutex::new(BTreeMap::new())),
            next_id: Arc::new(AtomicU64::new(1)),
        }
    }

    // a poisoned lock still holds a valid map; recovering it keeps one
    // failed request from taking every later request down with it.
    fn lock_items(&self) -> MutexGuard<'_, BTreeMap<u64, Bytes>> {
        self.items.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

fn item_id(request: &Request<Bytes>) -> Option<u64> {
    request
        .context
        .path_params
        .get("id")
        .and_then(|raw| raw.parse().ok())
}

/// CREATE: `POST /items`. Assigns the next id, stores the request body
/// verbatim, and hands the id back as a `Location` header.
struct CreateItem {
    store: Store,
}

impl SendPipe for CreateItem {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let store = self.store.clone();
        async move {
            let (_, body) = request.body_bytes().await?;
            let new_id = store.next_id.fetch_add(1, Ordering::Relaxed);
            store.lock_items().insert(new_id, body.clone());
            Ok(Response::new(201)
                .with_header("location", format!("/items/{new_id}"))
                .with_body(body))
        }
    }
}


/// READ: `GET /items/{id}`. `{id}` is a `PathPattern` param, extracted by
/// the router before this pipe ever runs.
struct ReadItem {
    store: Store,
}

impl SendPipe for ReadItem {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let store = self.store.clone();
        async move {
            let Some(resource_id) = item_id(&request) else {
                return Ok(Response::not_found());
            };
            match store.lock_items().get(&resource_id) {
                Some(value) => Ok(Response::ok(value.clone())),
                None => Ok(Response::not_found()),
            }
        }
    }
}


/// UPDATE: `PUT /items/{id}`. Replaces an existing item's value; a missing
/// id is a 404, not a silent create — `PUT` here updates, it doesn't upsert.
struct UpdateItem {
    store: Store,
}

impl SendPipe for UpdateItem {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let store = self.store.clone();
        async move {
            let Some(resource_id) = item_id(&request) else {
                return Ok(Response::not_found());
            };
            let (_, body) = request.body_bytes().await?;
            let mut items = store.lock_items();
            if !items.contains_key(&resource_id) {
                return Ok(Response::not_found());
            }
            items.insert(resource_id, body.clone());
            Ok(Response::ok(body))
        }
    }
}


/// DELETE: `DELETE /items/{id}`. Removes an existing item; a missing id is
/// a 404, not a silent no-op.
struct DeleteItem {
    store: Store,
}

impl SendPipe for DeleteItem {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let store = self.store.clone();
        async move {
            let Some(resource_id) = item_id(&request) else {
                return Ok(Response::not_found());
            };
            match store.lock_items().remove(&resource_id) {
                Some(_) => Ok(Response::no_data()),
                None => Ok(Response::not_found()),
            }
        }
    }
}


// the route table: one mount per (path pattern, method), each pointing at
// its own handler. `App` dispatches by matching both; unmatched requests
// get a 404 from the router itself, no fallback pipe required.
fn mount_routes(app: &App, store: Store) -> Result<(), ProximaError> {
    let create_item: PipeHandle = into_handle(CreateItem {
        store: store.clone(),
    });
    app.mount_with_methods(
        "/items",
        create_item,
        MethodFilter::only(["POST".to_string()]),
    )?;

    let read_item: PipeHandle = into_handle(ReadItem {
        store: store.clone(),
    });
    app.mount_with_methods(
        "/items/{id}",
        read_item,
        MethodFilter::only(["GET".to_string()]),
    )?;

    let update_item: PipeHandle = into_handle(UpdateItem {
        store: store.clone(),
    });
    app.mount_with_methods(
        "/items/{id}",
        update_item,
        MethodFilter::only(["PUT".to_string()]),
    )?;

    let delete_item: PipeHandle = into_handle(DeleteItem { store });
    app.mount_with_methods(
        "/items/{id}",
        delete_item,
        MethodFilter::only(["DELETE".to_string()]),
    )?;

    Ok(())
}

// one core is enough for one listener answering a handful of requests —
// `#[proxima::main(cores = 1)]` boots it; `App::builder()` adopts
// it ambiently (see `default_runtime` in `src/app.rs`).
#[proxima::main(cores = 1)]
async fn main() -> Result<(), ProximaError> {
    let bind: SocketAddr = BIND.parse().expect("valid socket addr");
    let app = App::builder().with_defaults()?.build()?;

    mount_routes(&app, Store::new())?;

    // blocks until every accept lane has acked ready — no polling, no
    // sleeping, no discovering ECONNREFUSED the hard way.
    let listener = app.build_listener(ListenerSpec::http(bind))?;
    let cores = app.runtime().expect("builder installs a runtime").num_cores();
    println!("listening on {bind} (prime runtime, {cores} core)");

    run_crud_flow(bind);

    listener.shutdown();
    let runtime = app
        .runtime()
        .ok_or_else(|| ProximaError::Config("app has no runtime installed".into()))?;
    let report = ShutdownBarrier::new(runtime).broadcast_drop().await;
    println!(
        "drained: cores_acked={} hooks_drained={}",
        report.cores_acked, report.hooks_drained
    );

    Ok(())
}

// drives the whole CRUD lifecycle over loopback HTTP/1.1, then the two sad
// paths (update/delete on an already-deleted item) — proof that the router
// and the handlers, not just the happy path, behave as documented.
fn run_crud_flow(bind: SocketAddr) {
    let created = blocking_request(bind, "POST", "/items", "banana bread");
    println!("POST /items ->\n{created}\n");
    assert_eq!(created.status, 201, "create returns 201");
    let location = created
        .header("location")
        .expect("create response carries a Location header")
        .to_string();

    let fetched = blocking_request(bind, "GET", &location, "");
    println!("GET {location} ->\n{fetched}\n");
    assert_eq!(
        fetched.status, 200,
        "read returns 200 for an item that exists"
    );
    assert_eq!(
        fetched.body, "banana bread",
        "read returns exactly what was created"
    );

    let updated = blocking_request(bind, "PUT", &location, "banana bread, toasted");
    println!("PUT {location} ->\n{updated}\n");
    assert_eq!(
        updated.status, 200,
        "update returns 200 for an item that exists"
    );
    assert_eq!(
        updated.body, "banana bread, toasted",
        "update returns the new value"
    );

    let refetched = blocking_request(bind, "GET", &location, "");
    assert_eq!(
        refetched.body, "banana bread, toasted",
        "the store now holds the updated value, not the original"
    );

    let deleted = blocking_request(bind, "DELETE", &location, "");
    println!("DELETE {location} ->\n{deleted}\n");
    assert_eq!(
        deleted.status, 204,
        "delete returns 204 for an item that existed"
    );

    let gone = blocking_request(bind, "GET", &location, "");
    println!("GET {location} (after delete) ->\n{gone}\n");
    assert_eq!(gone.status, 404, "the deleted item is gone");

    let missing_update = blocking_request(bind, "PUT", &location, "resurrected");
    assert_eq!(
        missing_update.status, 404,
        "update on a missing item is 404, not a silent upsert"
    );

    let missing_delete = blocking_request(bind, "DELETE", &location, "");
    assert_eq!(
        missing_delete.status, 404,
        "delete on a missing item is 404, not a silent no-op"
    );
}

struct HttpResponse {
    status: u16,
    headers: Vec<(String, String)>,
    body: String,
}

impl HttpResponse {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(header_name, _)| header_name.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }
}

impl fmt::Display for HttpResponse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} {}", self.status, self.body)
    }
}

/// One-shot request over a plain blocking `TcpStream`, same as `hello`'s
/// client — proof that the CRUD service is a real HTTP/1 server, not an
/// in-process call. `Connection: close` lets us read to EOF for the
/// response instead of framing the body ourselves.
fn blocking_request(addr: SocketAddr, method: &str, path: &str, body: &str) -> HttpResponse {
    let mut stream = TcpStream::connect(addr).expect("connect");
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(request.as_bytes()).expect("write request");
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).expect("read response");
    parse_response(&raw)
}

fn parse_response(raw: &[u8]) -> HttpResponse {
    let text = String::from_utf8_lossy(raw);
    let mut sections = text.splitn(2, "\r\n\r\n");
    let head = sections.next().unwrap_or_default();
    let body = sections.next().unwrap_or_default().to_string();

    let mut lines = head.split("\r\n");
    let status_line = lines.next().unwrap_or_default();
    let status = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse().ok())
        .unwrap_or(0);

    let headers = lines
        .filter_map(|line| {
            line.split_once(": ")
                .map(|(name, value)| (name.to_string(), value.to_string()))
        })
        .collect();

    HttpResponse {
        status,
        headers,
        body,
    }
}
