use alloc::string::String;
use core::time::Duration;

use bytes::Bytes;
use proxima_primitives::pipe::request::Response;

#[derive(Debug, Clone)]
pub struct WriteBackRule {
    pub from: String,
    pub to: String,
    pub ttl: Option<Duration>,
    pub conditions: WriteBackConditions,
}

#[derive(Debug, Clone)]
pub struct WriteBackConditions {
    pub only_on_success: bool,
    pub min_status: u16,
    pub max_status: u16,
}

impl Default for WriteBackConditions {
    fn default() -> Self {
        Self {
            only_on_success: true,
            min_status: 200,
            max_status: 299,
        }
    }
}

impl WriteBackConditions {
    #[must_use]
    pub fn applies_to(&self, response: &Response<Bytes>) -> bool {
        if self.only_on_success {
            return (self.min_status..=self.max_status).contains(&response.status);
        }
        true
    }
}

impl WriteBackRule {
    #[must_use]
    pub fn new(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            ttl: None,
            conditions: WriteBackConditions::default(),
        }
    }

    #[must_use]
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn default_conditions_only_apply_to_2xx() {
        let conditions = WriteBackConditions::default();
        assert!(conditions.applies_to(&Response::new(200)));
        assert!(conditions.applies_to(&Response::new(204)));
        assert!(conditions.applies_to(&Response::new(299)));
        assert!(!conditions.applies_to(&Response::new(404)));
        assert!(!conditions.applies_to(&Response::new(500)));
    }

    #[test]
    fn rule_builder_sets_fields() {
        let rule =
            WriteBackRule::new("origin", "cache").with_ttl(std::time::Duration::from_secs(60));
        assert_eq!(rule.from, "origin");
        assert_eq!(rule.to, "cache");
        assert_eq!(rule.ttl, Some(std::time::Duration::from_secs(60)));
    }
}
