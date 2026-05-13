use std::time::Duration;

use crate::outcome::Outcome;
use crate::scenario::Thresholds;

#[derive(Default)]
struct StageStats {
    latencies_ms: Vec<f64>,
    errors: u64,
    timeouts: u64,
    total: u64,
}

#[derive(Default)]
pub struct Recorder {
    stages: Vec<StageStats>,
}

impl Recorder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, stage: usize, o: Outcome) {
        while self.stages.len() <= stage {
            self.stages.push(StageStats::default());
        }
        if let Some(s) = self.stages.get_mut(stage) {
            s.total += 1;
            if !o.ok {
                s.errors += 1;
            }
            if o.timed_out {
                s.timeouts += 1;
            }
            s.latencies_ms
                .push(o.latency.as_secs_f64() * 1000.0);
        }
    }

    pub fn report(&self, th: &Thresholds) -> Report {
        let mut stages = Vec::with_capacity(self.stages.len());
        let mut all = Vec::new();
        let mut total = 0u64;
        let mut total_errors = 0u64;
        let mut total_timeouts = 0u64;
        for s in &self.stages {
            let mut lat = s.latencies_ms.clone();
            lat.sort_by(|a, b| {
                a.partial_cmp(b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            all.extend_from_slice(&lat);
            total += s.total;
            total_errors += s.errors;
            total_timeouts += s.timeouts;
            stages.push(StageReport {
                count: s.total,
                errors: s.errors,
                timeouts: s.timeouts,
                error_rate: ratio(s.errors, s.total),
                p50: ms(percentile(&lat, 50.0)),
                p90: ms(percentile(&lat, 90.0)),
                p99: ms(percentile(&lat, 99.0)),
                p999: ms(percentile(&lat, 99.9)),
            });
        }
        all.sort_by(|a, b| {
            a.partial_cmp(b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let overall_p99 = ms(percentile(&all, 99.0));
        let overall_error_rate = ratio(total_errors, total);

        let mut passed = true;
        if let Some(limit) = th.p99
            && overall_p99 > limit
        {
            passed = false;
        }
        if let Some(limit) = th.error_rate
            && overall_error_rate > limit
        {
            passed = false;
        }

        Report {
            stages,
            overall_p99,
            overall_error_rate,
            overall_timeouts: total_timeouts,
            passed,
        }
    }
}

pub struct Report {
    pub stages: Vec<StageReport>,
    pub overall_p99: Duration,
    pub overall_error_rate: f64,
    pub overall_timeouts: u64,
    pub passed: bool,
}

pub struct StageReport {
    pub count: u64,
    pub errors: u64,
    pub timeouts: u64,
    pub error_rate: f64,
    pub p50: Duration,
    pub p90: Duration,
    pub p99: Duration,
    pub p999: Duration,
}

impl Report {
    pub fn render(&self) -> String {
        let mut out = String::new();
        for (i, s) in self.stages.iter().enumerate() {
            out.push_str(&format!(
                "stage {i}: {n} reqs, {e} err ({er:.2}%), {t} timeout  p50 {p50}  p90 {p90}  p99 {p99}  p999 {p999}\n",
                n = s.count,
                e = s.errors,
                t = s.timeouts,
                er = s.error_rate * 100.0,
                p50 = fmt_ms(s.p50),
                p90 = fmt_ms(s.p90),
                p99 = fmt_ms(s.p99),
                p999 = fmt_ms(s.p999),
            ));
        }
        out.push_str(&format!(
            "overall: p99 {p99}  errors {er:.2}%  timeouts {t}  -> {verdict}\n",
            p99 = fmt_ms(self.overall_p99),
            er = self.overall_error_rate * 100.0,
            t = self.overall_timeouts,
            verdict = if self.passed { "pass" } else { "fail" },
        ));
        out
    }
}

fn ratio(num: u64, den: u64) -> f64 {
    if den == 0 { 0.0 } else { num as f64 / den as f64 }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (p / 100.0) * (sorted.len() as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    match (sorted.get(lo), sorted.get(hi)) {
        (Some(&a), Some(&b)) => {
            let frac = rank - lo as f64;
            a * (1.0 - frac) + b * frac
        }
        _ => 0.0,
    }
}

fn ms(value: f64) -> Duration {
    Duration::from_secs_f64((value / 1000.0).max(0.0))
}

fn fmt_ms(d: Duration) -> String {
    format!("{:.1}ms", d.as_secs_f64() * 1000.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome(ms: f64, ok: bool) -> Outcome {
        Outcome {
            latency: Duration::from_secs_f64(ms / 1000.0),
            ok,
            timed_out: false,
        }
    }

    #[test]
    fn p99_threshold_fails_the_run() {
        let mut rec = Recorder::new();
        for _ in 0..98 {
            rec.record(0, outcome(10.0, true));
        }
        rec.record(0, outcome(500.0, true));
        rec.record(0, outcome(500.0, true));
        let th = Thresholds {
            p99: Some(Duration::from_millis(250)),
            error_rate: None,
        };
        assert!(!rec.report(&th).passed);
    }

    #[test]
    fn timeouts_tally_apart_from_errors() {
        let mut rec = Recorder::new();
        for _ in 0..90 {
            rec.record(0, outcome(10.0, true));
        }
        rec.record(0, outcome(20.0, false)); // a real server error
        for _ in 0..9 {
            rec.record(
                0,
                Outcome {
                    latency: Duration::from_secs(30),
                    ok: false,
                    timed_out: true,
                },
            );
        }
        let report = rec.report(&Thresholds { p99: None, error_rate: None });
        assert_eq!(report.stages[0].count, 100);
        assert_eq!(report.stages[0].errors, 10); // error + 9 timeouts are all not-ok
        assert_eq!(report.stages[0].timeouts, 9); // but timeouts are counted apart
        assert_eq!(report.overall_timeouts, 9);
    }

    #[test]
    fn clean_run_passes() {
        let mut rec = Recorder::new();
        for _ in 0..100 {
            rec.record(0, outcome(10.0, true));
        }
        let th = Thresholds {
            p99: Some(Duration::from_millis(250)),
            error_rate: Some(0.01),
        };
        assert!(rec.report(&th).passed);
    }
}
