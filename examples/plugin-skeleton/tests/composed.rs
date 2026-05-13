#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use proxima::{App, Request, SendPipe};
use serde_json::json;

#[proxima::test]
async fn plugin_register_composes_into_app_builder() {
    let app = proxima_plugin_skeleton::register(App::builder().with_defaults().expect("defaults"))
        .expect("register plugin")
        .build()
        .expect("build");
    assert!(app.load_context().registry.get("stamp_header").is_ok());
}

#[proxima::test]
async fn plugin_middleware_actually_wraps_a_pipe() {
    let app = proxima_plugin_skeleton::register(App::builder().with_defaults().expect("defaults"))
        .expect("register plugin")
        .build()
        .expect("build");
    let synth_factory = app
        .load_context()
        .registry
        .get("synth")
        .expect("synth registered");
    let inner = synth_factory
        .build(&json!({"status": 200, "body": "hello"}), None)
        .await
        .expect("build synth");
    let stamp_factory = app
        .load_context()
        .registry
        .get("stamp_header")
        .expect("stamp registered");
    let stacked = stamp_factory
        .build(
            &json!({"name": "x-from-plugin", "value": "yes"}),
            Some(inner),
        )
        .await
        .expect("build stamp");
    let request = Request::builder()
        .method("GET")
        .path("/")
        .build()
        .expect("request");
    let response = SendPipe::call(&stacked, request).await.expect("call");
    assert_eq!(response.status, 200);
    assert_eq!(response.metadata.get_str("x-from-plugin"), Some("yes"));
}
