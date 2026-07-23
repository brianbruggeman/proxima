//! Protocol axis for [`ClientBuilder`] — TYPE-SPECIFIC (no blanket impl over
//! every `SpecBuilder`, unlike the retired
//! `proxima_config::sugar::ProtocolSugar`). Names the app protocol + upstream
//! dial target. `.http()`/`.https()`/`.grpc()` lower straight to spec keys
//! (mirroring the retired blanket trait exactly); every DSN-carrying method
//! (`.kafka()`/`.mqtt()`/`.amqp()`/`.dns()`/`.memcached()`/`.redis()`/
//! `.valkey()`/`.pgwire()`) delegates to [`ClientBuilder::protocol`] with a
//! thin per-crate `XxxClientProtocol` — ONE mechanism for every terminal,
//! including `redis`/`pgwire`, migrated off their old bespoke inherent
//! methods (Section E of the builder-sugar design).

use crate::client::handle::ClientBuilder;
use proxima_config::sugar::SpecBuilder;

/// `.http()`/`.https()`/`.grpc()` + every DSN-carrying protocol terminal.
/// Bring into scope with `use proxima::ClientProtocolExt;` (or
/// `proxima::prelude::*`).
pub trait ClientProtocolExt: Sized {
    /// Point at an HTTP upstream base url (the `http` key).
    #[must_use]
    fn http(self, url: impl Into<String>) -> Self;

    /// HTTPS base url — the `http` key with an `https` scheme; TLS falls out
    /// of the scheme, so this is `.http()` with an https url.
    #[must_use]
    fn https(self, url: impl Into<String>) -> Self;

    /// Point at a gRPC upstream base url (the `grpc` key — gRPC over h2).
    #[must_use]
    fn grpc(self, url: impl Into<String>) -> Self;

    /// Point at a Kafka broker by DSN (`kafka://broker[:port]`).
    #[cfg(all(feature = "kafka-client", any(target_os = "linux", target_os = "macos")))]
    #[must_use]
    fn kafka(self, dsn: impl Into<String>) -> Self;

    /// Point at an MQTT broker by DSN (`mqtt://[user:pass@]broker[:port]`).
    #[cfg(all(feature = "mqtt-client", any(target_os = "linux", target_os = "macos")))]
    #[must_use]
    fn mqtt(self, dsn: impl Into<String>) -> Self;

    /// Point at an AMQP broker by DSN (`amqp://[user:pass@]broker[:port][/vhost]`).
    #[cfg(all(feature = "amqp-client", any(target_os = "linux", target_os = "macos")))]
    #[must_use]
    fn amqp(self, dsn: impl Into<String>) -> Self;

    /// Point at a DNS resolver by DSN (`dns://resolver_ip[:port]`).
    #[cfg(all(feature = "dns-client", any(target_os = "linux", target_os = "macos")))]
    #[must_use]
    fn dns(self, dsn: impl Into<String>) -> Self;

    /// Point at a memcached server by DSN (`memcached://host[:port]`).
    #[cfg(all(
        feature = "memcached-client",
        any(target_os = "linux", target_os = "macos")
    ))]
    #[must_use]
    fn memcached(self, dsn: impl Into<String>) -> Self;

    /// Point at a Redis server by DSN (`redis://[user:pass@]host[:port][/db]`).
    #[cfg(all(feature = "redis-client", any(target_os = "linux", target_os = "macos")))]
    #[must_use]
    fn redis(self, dsn: impl Into<String>) -> Self;

    /// Point at a Valkey server by DSN — Valkey speaks the same RESP wire
    /// protocol as Redis, so this aliases [`Self::redis`] onto the one
    /// `redis` factory.
    #[cfg(all(feature = "redis-client", any(target_os = "linux", target_os = "macos")))]
    #[must_use]
    fn valkey(self, dsn: impl Into<String>) -> Self;

    /// Point at a PostgreSQL server by DSN (`postgres://user:pw@host:port/db`).
    #[cfg(all(
        feature = "pgwire-client",
        any(target_os = "linux", target_os = "macos")
    ))]
    #[must_use]
    fn pgwire(self, dsn: impl Into<String>) -> Self;
}

impl ClientProtocolExt for ClientBuilder {
    fn http(self, url: impl Into<String>) -> Self {
        self.set("http", url.into())
    }

    fn https(self, url: impl Into<String>) -> Self {
        self.set("http", url.into())
    }

    fn grpc(self, url: impl Into<String>) -> Self {
        self.set("grpc", url.into())
    }

    #[cfg(all(feature = "kafka-client", any(target_os = "linux", target_os = "macos")))]
    fn kafka(self, dsn: impl Into<String>) -> Self {
        self.protocol(crate::upstreams::kafka::KafkaClientProtocol::dsn(dsn))
    }

    #[cfg(all(feature = "mqtt-client", any(target_os = "linux", target_os = "macos")))]
    fn mqtt(self, dsn: impl Into<String>) -> Self {
        self.protocol(crate::upstreams::mqtt::MqttClientProtocol::dsn(dsn))
    }

    #[cfg(all(feature = "amqp-client", any(target_os = "linux", target_os = "macos")))]
    fn amqp(self, dsn: impl Into<String>) -> Self {
        self.protocol(crate::upstreams::amqp::AmqpClientProtocol::dsn(dsn))
    }

    #[cfg(all(feature = "dns-client", any(target_os = "linux", target_os = "macos")))]
    fn dns(self, dsn: impl Into<String>) -> Self {
        self.protocol(crate::upstreams::dns::DnsClientProtocol::dsn(dsn))
    }

    #[cfg(all(
        feature = "memcached-client",
        any(target_os = "linux", target_os = "macos")
    ))]
    fn memcached(self, dsn: impl Into<String>) -> Self {
        self.protocol(crate::upstreams::memcached::MemcachedClientProtocol::dsn(dsn))
    }

    #[cfg(all(feature = "redis-client", any(target_os = "linux", target_os = "macos")))]
    fn redis(self, dsn: impl Into<String>) -> Self {
        self.protocol(crate::upstreams::redis::RedisClientProtocol::dsn(dsn))
    }

    #[cfg(all(feature = "redis-client", any(target_os = "linux", target_os = "macos")))]
    fn valkey(self, dsn: impl Into<String>) -> Self {
        self.redis(dsn)
    }

    #[cfg(all(
        feature = "pgwire-client",
        any(target_os = "linux", target_os = "macos")
    ))]
    fn pgwire(self, dsn: impl Into<String>) -> Self {
        self.protocol(crate::upstreams::pgwire::PgwireClientProtocol::dsn(dsn))
    }
}
