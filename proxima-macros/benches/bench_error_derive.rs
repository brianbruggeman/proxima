//! Home-turf bench for the `proxima_macros::Error` derive (Pass 0 of
//! the cliff-extension plan, disciplined-component gate point 13).
//!
//! Named incumbent: `thiserror` v2 with `default-features = false`.
//! Incumbent design point: drop-in derive that just works for arbitrary
//! user enums and stays no_std-friendly when told to.
//!
//! Arms below all use enums thiserror v2 was designed for (unit
//! variants, tuple variants with format-arg interpolation, source
//! chains via `#[from]`). Marker: `design-favors: thiserror` — they
//! get to engage their full machinery on shapes they own.

#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

mod proxima_arm {
    use proxima_macros::Error;

    #[derive(Debug, Error)]
    pub enum TenVariants {
        #[error("a")]
        A,
        #[error("b")]
        B,
        #[error("c")]
        C,
        #[error("d")]
        D,
        #[error("e")]
        E,
        #[error("f")]
        F,
        #[error("g")]
        G,
        #[error("h")]
        H,
        #[error("i")]
        I,
        #[error("j")]
        J,
    }

    #[derive(Debug, Error)]
    pub enum WithFormat {
        #[error("a {0}")]
        A(u32),
        #[error("b {0} {1}")]
        B(u32, u32),
        #[error("c {value}")]
        C { value: u32 },
        #[error("d")]
        D,
    }

    #[derive(Debug)]
    pub struct Inner(pub u32);
    impl core::fmt::Display for Inner {
        fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            write!(formatter, "inner({})", self.0)
        }
    }
    impl core::error::Error for Inner {}

    #[derive(Debug, Error)]
    pub enum WithSource {
        #[error("wrapped")]
        Wrapped(#[from] Inner),
    }
}

mod thiserror_arm {
    use thiserror::Error;

    #[derive(Debug, Error)]
    pub enum TenVariants {
        #[error("a")]
        A,
        #[error("b")]
        B,
        #[error("c")]
        C,
        #[error("d")]
        D,
        #[error("e")]
        E,
        #[error("f")]
        F,
        #[error("g")]
        G,
        #[error("h")]
        H,
        #[error("i")]
        I,
        #[error("j")]
        J,
    }

    #[derive(Debug, Error)]
    pub enum WithFormat {
        #[error("a {0}")]
        A(u32),
        #[error("b {0} {1}")]
        B(u32, u32),
        #[error("c {value}")]
        C { value: u32 },
        #[error("d")]
        D,
    }

    #[derive(Debug)]
    pub struct Inner(pub u32);
    impl core::fmt::Display for Inner {
        fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            write!(formatter, "inner({})", self.0)
        }
    }
    impl core::error::Error for Inner {}

    #[derive(Debug, Error)]
    pub enum WithSource {
        #[error("wrapped")]
        Wrapped(#[from] Inner),
    }
}

fn bench_unit_display(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("unit_display_10_variants");
    let variants_proxima = [
        proxima_arm::TenVariants::A,
        proxima_arm::TenVariants::B,
        proxima_arm::TenVariants::C,
        proxima_arm::TenVariants::D,
        proxima_arm::TenVariants::E,
        proxima_arm::TenVariants::F,
        proxima_arm::TenVariants::G,
        proxima_arm::TenVariants::H,
        proxima_arm::TenVariants::I,
        proxima_arm::TenVariants::J,
    ];
    let variants_thiserror = [
        thiserror_arm::TenVariants::A,
        thiserror_arm::TenVariants::B,
        thiserror_arm::TenVariants::C,
        thiserror_arm::TenVariants::D,
        thiserror_arm::TenVariants::E,
        thiserror_arm::TenVariants::F,
        thiserror_arm::TenVariants::G,
        thiserror_arm::TenVariants::H,
        thiserror_arm::TenVariants::I,
        thiserror_arm::TenVariants::J,
    ];
    group.bench_function("proxima_error", |bencher| {
        bencher.iter(|| {
            let mut total = 0usize;
            for variant in &variants_proxima {
                let formatted = format!("{}", black_box(variant));
                total += formatted.len();
            }
            black_box(total)
        });
    });
    group.bench_function("thiserror", |bencher| {
        bencher.iter(|| {
            let mut total = 0usize;
            for variant in &variants_thiserror {
                let formatted = format!("{}", black_box(variant));
                total += formatted.len();
            }
            black_box(total)
        });
    });
    group.finish();
}

fn bench_format_args_display(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("format_args_display");
    let cases_proxima = [
        proxima_arm::WithFormat::A(42),
        proxima_arm::WithFormat::B(100, 200),
        proxima_arm::WithFormat::C { value: 7 },
        proxima_arm::WithFormat::D,
    ];
    let cases_thiserror = [
        thiserror_arm::WithFormat::A(42),
        thiserror_arm::WithFormat::B(100, 200),
        thiserror_arm::WithFormat::C { value: 7 },
        thiserror_arm::WithFormat::D,
    ];
    group.bench_function("proxima_error", |bencher| {
        bencher.iter(|| {
            let mut total = 0usize;
            for case in &cases_proxima {
                let formatted = format!("{}", black_box(case));
                total += formatted.len();
            }
            black_box(total)
        });
    });
    group.bench_function("thiserror", |bencher| {
        bencher.iter(|| {
            let mut total = 0usize;
            for case in &cases_thiserror {
                let formatted = format!("{}", black_box(case));
                total += formatted.len();
            }
            black_box(total)
        });
    });
    group.finish();
}

fn bench_source_chain(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("source_chain_walk");
    let err_proxima = proxima_arm::WithSource::Wrapped(proxima_arm::Inner(42));
    let err_thiserror = thiserror_arm::WithSource::Wrapped(thiserror_arm::Inner(42));
    group.bench_function("proxima_error", |bencher| {
        bencher.iter(|| {
            let direct = core::error::Error::source(black_box(&err_proxima));
            black_box(direct.is_some())
        });
    });
    group.bench_function("thiserror", |bencher| {
        bencher.iter(|| {
            let direct = core::error::Error::source(black_box(&err_thiserror));
            black_box(direct.is_some())
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_unit_display,
    bench_format_args_display,
    bench_source_chain
);
criterion_main!(benches);
