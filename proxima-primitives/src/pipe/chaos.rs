use serde::{Deserialize, Serialize};

#[cfg(feature = "std")]
use crate::pipe::delay::{Delay, Dist};
#[cfg(feature = "std")]
use crate::pipe::filter::{FilterConfig, Predicate, RejectMode};
#[cfg(feature = "std")]
use crate::pipe::mutate::{MutateOp, Mutation};
#[cfg(feature = "std")]
use crate::pipe::handler::{PipeHandle, into_handle};
#[cfg(feature = "std")]
use crate::pipe::transform::{ResponseOp, Transform};
#[cfg(feature = "std")]
use crate::pipe::when::When;

#[cfg(feature = "std")]
const ERROR_SALT: u64 = 0x1111_1111_1111_1111;
#[cfg(feature = "std")]
const DROP_SALT: u64 = 0x2222_2222_2222_2222;
#[cfg(feature = "std")]
const LATENCY_SALT: u64 = 0x3333_3333_3333_3333;
#[cfg(feature = "std")]
const CORRUPT_SALT: u64 = 0x4444_4444_4444_4444;

/// A latency fault: with probability `prob`, sleep `ms` milliseconds before the
/// call reaches the inner pipe.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LatencyFault {
    pub prob: f64,
    pub ms: u64,
}

impl Default for LatencyFault {
    fn default() -> Self {
        Self { prob: 0.0, ms: 0 }
    }
}

/// A chaos preset: a bundle of independently-tunable fault probabilities plus a
/// base seed. Every part is optional/default-off.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ChaosConfig {
    #[serde(default)]
    pub error: f64,
    #[serde(default)]
    pub drop: f64,
    #[serde(default)]
    pub latency: LatencyFault,
    #[serde(default)]
    pub corrupt: f64,
    #[serde(default)]
    pub seed: u64,
}

impl Default for ChaosConfig {
    fn default() -> Self {
        Self {
            error: 0.0,
            drop: 0.0,
            latency: LatencyFault::default(),
            corrupt: 0.0,
            seed: 0,
        }
    }
}

impl ChaosConfig {
    #[cfg(feature = "std")]
    #[must_use]
    pub fn into_pipe(self, inner: PipeHandle) -> PipeHandle {
        let corrupt_seed = self.seed.wrapping_add(CORRUPT_SALT);
        let corrupted = if self.corrupt > 0.0 {
            let gate = When::prob(self.corrupt).seed(corrupt_seed);
            let op = MutateOp::new(Mutation::BitFlip { bits: 8 }, corrupt_seed).with_when(gate);
            into_handle(Transform::new(inner).with_response_op(ResponseOp::Mutate(op)))
        } else {
            inner
        };

        let latency_seed = self.seed.wrapping_add(LATENCY_SALT);
        let delayed = if self.latency.prob > 0.0 && self.latency.ms > 0 {
            let gate = When::prob(self.latency.prob).seed(latency_seed);
            into_handle(
                Delay::new(
                    corrupted,
                    Dist::Const {
                        ms: self.latency.ms,
                    },
                )
                .with_when(gate),
            )
        } else {
            corrupted
        };

        let drop_seed = self.seed.wrapping_add(DROP_SALT);
        let dropped = if self.drop > 0.0 {
            let gate = When::prob(self.drop).seed(drop_seed);
            FilterConfig {
                predicate: Predicate::unless(gate),
                on_reject: RejectMode::Drop,
            }
            .into_filter(delayed)
        } else {
            delayed
        };

        let error_seed = self.seed.wrapping_add(ERROR_SALT);
        if self.error > 0.0 {
            let gate = When::prob(self.error).seed(error_seed);
            FilterConfig {
                predicate: Predicate::unless(gate),
                on_reject: RejectMode::Error,
            }
            .into_filter(dropped)
        } else {
            dropped
        }
    }

    #[must_use]
    pub fn into_builder(self) -> ChaosBuilder {
        ChaosBuilder { config: self }
    }
}

/// Start a fluent chaos preset.
#[must_use]
pub fn chaos() -> ChaosBuilder {
    ChaosBuilder {
        config: ChaosConfig::default(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChaosBuilder {
    config: ChaosConfig,
}

impl ChaosBuilder {
    #[must_use]
    pub fn error(mut self, prob: f64) -> Self {
        self.config.error = prob.clamp(0.0, 1.0);
        self
    }

    #[must_use]
    pub fn drop(mut self, prob: f64) -> Self {
        self.config.drop = prob.clamp(0.0, 1.0);
        self
    }

    #[must_use]
    pub fn latency(mut self, prob: f64, ms: u64) -> Self {
        self.config.latency = LatencyFault {
            prob: prob.clamp(0.0, 1.0),
            ms,
        };
        self
    }

    #[must_use]
    pub fn corrupt(mut self, prob: f64) -> Self {
        self.config.corrupt = prob.clamp(0.0, 1.0);
        self
    }

    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
        self
    }

    #[must_use]
    pub fn to_config(&self) -> ChaosConfig {
        self.config
    }

    #[cfg(feature = "std")]
    #[must_use]
    pub fn into_pipe(self, inner: PipeHandle) -> PipeHandle {
        self.config.into_pipe(inner)
    }
}

// `#[proxima::test]` pulls in the `proxima` dev-dependency, which the
// loom build keeps out of the graph (see
// `[target.'cfg(not(loom))'.dev-dependencies]` in Cargo.toml); these
// tests are unrelated to the Notify/watch loom protocol.
#[cfg(all(test, not(loom)))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::future::Future;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    use bytes::Bytes;
    use proxima_core::ProximaError;
    use crate::pipe::SendPipe;

    use crate::pipe::handler::into_handle;
    use crate::pipe::request::{Request, Response};

    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Outcome {
        Errored,
        Dropped,
        Corrupted,
        Clean,
    }

    #[derive(Clone)]
    struct Echo {
        reached: Arc<AtomicU64>,
    }

    impl SendPipe for Echo {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            let reached = self.reached.clone();
            async move {
                reached.fetch_add(1, Ordering::Relaxed);
                let (_, body) = request.body_bytes().await?;
                Ok(Response::new(200).with_body(body))
            }
        }
    }


    const PAYLOAD: &[u8] = b"the quick brown fox jumps over the lazy dog";

    fn build_request() -> Request<Bytes> {
        Request::builder()
            .method("POST")
            .path("/")
            .body(Bytes::from_static(PAYLOAD))
            .build()
            .expect("builder")
    }

    async fn drive_once(pipe: &PipeHandle) -> Outcome {
        // a reject now lives entirely in the `Err` channel (no more sentinel
        // `Ok(403)`): `RejectMode::Drop` builds `ProximaError::Forbidden`,
        // `RejectMode::Error` builds `ProximaError::Config` — same two
        // observable outcomes, distinguished by which error it is.
        match SendPipe::call(pipe, build_request()).await {
            Err(ProximaError::Forbidden(_)) => Outcome::Dropped,
            Err(_) => Outcome::Errored,
            Ok(response) => {
                let body = response.collect_body().await.expect("body");
                if &body[..] == PAYLOAD {
                    Outcome::Clean
                } else {
                    Outcome::Corrupted
                }
            }
        }
    }

    async fn drive_sequence(config: ChaosConfig, count: usize) -> Vec<Outcome> {
        let pipe = config.into_pipe(into_handle(Echo {
            reached: Arc::new(AtomicU64::new(0)),
        }));
        let mut outcomes = Vec::with_capacity(count);
        for _ in 0..count {
            outcomes.push(drive_once(&pipe).await);
        }
        outcomes
    }

    #[proxima::test]
    async fn same_seed_reproduces_the_identical_fault_sequence() {
        let config = ChaosConfig {
            error: 0.2,
            drop: 0.2,
            latency: LatencyFault { prob: 0.3, ms: 5 },
            corrupt: 0.3,
            seed: 0xC0FFEE,
        };

        let first = drive_sequence(config, 200).await;
        let second = drive_sequence(config, 200).await;

        assert_eq!(
            first, second,
            "same seed must replay the identical fault sequence bit-for-bit"
        );
    }

    #[proxima::test]
    async fn a_different_seed_yields_a_different_sequence() {
        let base = ChaosConfig {
            error: 0.2,
            drop: 0.2,
            latency: LatencyFault { prob: 0.3, ms: 5 },
            corrupt: 0.3,
            seed: 1,
        };
        let other = ChaosConfig { seed: 2, ..base };

        let left = drive_sequence(base, 200).await;
        let right = drive_sequence(other, 200).await;

        assert_ne!(
            left, right,
            "a different seed must diverge somewhere in the sequence"
        );
    }

    #[proxima::test]
    async fn every_fault_type_is_reachable() {
        let config = ChaosConfig {
            error: 0.2,
            drop: 0.2,
            latency: LatencyFault { prob: 0.3, ms: 5 },
            corrupt: 0.3,
            seed: 0xABCD,
        };
        let outcomes = drive_sequence(config, 500).await;

        assert!(
            outcomes.contains(&Outcome::Errored),
            "error fault must be reachable"
        );
        assert!(
            outcomes.contains(&Outcome::Dropped),
            "drop fault must be reachable"
        );
        assert!(
            outcomes.contains(&Outcome::Corrupted),
            "corrupt fault must be reachable"
        );
        assert!(
            outcomes.contains(&Outcome::Clean),
            "a clean pass-through must be reachable"
        );
    }

    #[proxima::test]
    async fn an_all_default_preset_is_a_transparent_pass_through() {
        let reached = Arc::new(AtomicU64::new(0));
        let pipe = ChaosConfig::default().into_pipe(into_handle(Echo {
            reached: reached.clone(),
        }));

        for _ in 0..32 {
            assert_eq!(
                drive_once(&pipe).await,
                Outcome::Clean,
                "default chaos never injects a fault"
            );
        }
        assert_eq!(
            reached.load(Ordering::Relaxed),
            32,
            "every call reaches the inner pipe untouched"
        );
    }

    #[proxima::test]
    async fn error_only_preset_short_circuits_before_the_inner_pipe() {
        let reached = Arc::new(AtomicU64::new(0));
        let config = ChaosConfig {
            error: 1.0,
            seed: 7,
            ..ChaosConfig::default()
        };
        let pipe = config.into_pipe(into_handle(Echo {
            reached: reached.clone(),
        }));

        for _ in 0..16 {
            assert_eq!(
                drive_once(&pipe).await,
                Outcome::Errored,
                "prob 1.0 error fires every call"
            );
        }
        assert_eq!(
            reached.load(Ordering::Relaxed),
            0,
            "an injected error never reaches the inner pipe"
        );
    }

    #[test]
    fn fluent_builder_matches_the_config() {
        let built = chaos()
            .error(0.1)
            .drop(0.2)
            .latency(0.3, 25)
            .corrupt(0.4)
            .seed(0xC0FFEE)
            .to_config();

        let expected = ChaosConfig {
            error: 0.1,
            drop: 0.2,
            latency: LatencyFault { prob: 0.3, ms: 25 },
            corrupt: 0.4,
            seed: 0xC0FFEE,
        };
        assert_eq!(
            built, expected,
            "the fluent builder produces the same config"
        );
    }

    #[test]
    fn config_builder_round_trip_parity() {
        let config = ChaosConfig {
            error: 0.1,
            drop: 0.2,
            latency: LatencyFault { prob: 0.3, ms: 25 },
            corrupt: 0.4,
            seed: 0xC0FFEE,
        };

        let back = config.into_builder().to_config();
        let json = serde_json::to_value(config).expect("serialize");
        let parsed: ChaosConfig = serde_json::from_value(json).expect("deserialize");

        assert_eq!(
            back, config,
            "builder projects back to the originating config"
        );
        assert_eq!(parsed, config, "serde round-trip is lossless");
    }

    #[test]
    fn omitted_fields_default_off() {
        let value = serde_json::json!({ "seed": 99 });
        let config: ChaosConfig = serde_json::from_value(value).expect("deserialize");
        assert_eq!(
            config,
            ChaosConfig {
                seed: 99,
                ..ChaosConfig::default()
            },
            "absent faults default off"
        );
    }

    #[proxima::test]
    async fn fluent_and_config_expand_to_the_same_sequence() {
        let config = ChaosConfig {
            error: 0.2,
            drop: 0.2,
            latency: LatencyFault { prob: 0.3, ms: 5 },
            corrupt: 0.3,
            seed: 0x5EED,
        };
        let from_config = drive_sequence(config, 128).await;
        let from_fluent = drive_sequence(config.into_builder().to_config(), 128).await;
        assert_eq!(
            from_config, from_fluent,
            "both surfaces expand to the identical fault sequence"
        );
    }
}
