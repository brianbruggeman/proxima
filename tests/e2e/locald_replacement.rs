#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use std::sync::Arc;

use proxima::{App, ControlPlane, DaemonControlPlane, PipeConfig, PipeState};
use serde_json::json;

fn shell() -> &'static str {
    if cfg!(windows) { "cmd" } else { "sh" }
}

fn shell_arg() -> &'static str {
    if cfg!(windows) { "/c" } else { "-c" }
}

fn cart_api_config() -> PipeConfig {
    PipeConfig {
        name: "cart_api".into(),
        spec: json!({
            "process": {
                "command": shell(),
                "args": [shell_arg(), "echo cart_api ready; sleep 30"],
                "ready_probe": {
                    "type": "stdout_line",
                    "pattern": "cart_api ready",
                    "timeout_ms": 5000,
                },
                "restart": "never",
            },
        }),
        requires: Vec::new(),
    }
}

fn cart_www_config() -> PipeConfig {
    PipeConfig {
        name: "cart_www".into(),
        spec: json!({
            "process": {
                "command": shell(),
                "args": [shell_arg(), "echo cart_www ready; sleep 30"],
                "ready_probe": {
                    "type": "stdout_line",
                    "pattern": "cart_www ready",
                    "timeout_ms": 5000,
                },
                "restart": "never",
            },
        }),
        requires: vec!["cart_api".into()],
    }
}

#[proxima::test]
async fn locald_style_dep_graph_starts_supervised_processes_in_topological_order() {
    let app = App::new().expect("app");
    let plane: Arc<DaemonControlPlane> = Arc::new(DaemonControlPlane::new(
        app,
        vec![cart_api_config(), cart_www_config()],
    ));

    // start cart_www; the dep graph should bring cart_api up first.
    let status = plane.start("cart_www").await.expect("start cart_www");
    assert_eq!(status.state, PipeState::Running);

    // every configured pipe must now be running.
    let listed = plane.list_pipes().await.expect("list");
    for pipe in &listed {
        assert_eq!(
            pipe.state,
            PipeState::Running,
            "pipe `{}` must be running after cascading start",
            pipe.name,
        );
    }

    // logs for each pipe should contain at least the readiness line.
    let api_logs = plane.logs("cart_api", None).await.expect("api logs");
    assert!(api_logs.iter().any(|line| line.contains("cart_api ready")));
    let www_logs = plane.logs("cart_www", None).await.expect("www logs");
    assert!(www_logs.iter().any(|line| line.contains("cart_www ready")));

    // restart cart_api; cart_www stays running independently.
    let restarted = plane.restart("cart_api").await.expect("restart");
    assert_eq!(restarted.state, PipeState::Running);
    assert_eq!(restarted.restart_count, 1);
    let www = plane.status("cart_www").await.expect("www status");
    assert_eq!(www.state, PipeState::Running);

    // stop cart_api; cart_www isn't auto-stopped — caller manages dep order on the way down.
    plane.stop("cart_api").await.expect("stop");
    let api = plane.status("cart_api").await.expect("api status");
    assert_eq!(api.state, PipeState::Stopped);
    let www = plane.status("cart_www").await.expect("www status");
    assert_eq!(www.state, PipeState::Running);
}
