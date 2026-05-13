#![allow(clippy::expect_used)] // bench setup failures are bugs, not recoverable errors
use std::hint::black_box;
use std::time::Duration;

use bytes::{BufMut, BytesMut};
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use pgwire::messages::Message;
use pgwire::messages::data::{DataRow, FORMAT_CODE_TEXT, RowDescription};
use pgwire::messages::extendedquery::{
    Bind, Describe, Execute, Parse, Sync, TARGET_TYPE_BYTE_PORTAL,
};
use pgwire::messages::response::{ReadyForQuery, TransactionStatus as PgTransactionStatus};
use pgwire::messages::startup::{Authentication, BackendKeyData, ParameterStatus};
use proxima_protocols::pgwire_codec::AuthRequest;
use proxima_protocols::pgwire_codec::backend::{BackendMessage, DataRowWriter, RowDescriptionWriter};
use proxima_protocols::pgwire_codec::frontend::{parse_frontend, parse_initial};
use proxima_protocols::pgwire_codec::types::{FormatCode, Oid, PgStr, ProtocolVersion, TransactionStatus};
use proxima_protocols::pgwire_codec::views::FieldDescription as ProxFieldDescription;

const MEASUREMENT_SECS: u64 = 2;

fn build_pipeline_wire() -> Vec<u8> {
    let mut out = Vec::with_capacity(256);

    {
        let mut frame = Vec::new();
        let body: Vec<u8> = {
            let mut body = Vec::new();
            body.extend_from_slice(b"find-user\0");
            body.extend_from_slice(b"select id, email from users where id = $1\0");
            body.extend_from_slice(&1u16.to_be_bytes());
            body.extend_from_slice(&23u32.to_be_bytes());
            body
        };
        frame.push(b'P');
        frame.extend_from_slice(&((body.len() + 4) as i32).to_be_bytes());
        frame.extend_from_slice(&body);
        out.extend_from_slice(&frame);
    }

    {
        let mut frame = Vec::new();
        let body: Vec<u8> = {
            let mut body = Vec::new();
            body.extend_from_slice(b"\0");
            body.extend_from_slice(b"find-user\0");
            body.extend_from_slice(&1u16.to_be_bytes());
            body.extend_from_slice(&1i16.to_be_bytes());
            body.extend_from_slice(&1u16.to_be_bytes());
            body.extend_from_slice(&8i32.to_be_bytes());
            body.extend_from_slice(&1234i64.to_be_bytes());
            body.extend_from_slice(&1u16.to_be_bytes());
            body.extend_from_slice(&0i16.to_be_bytes());
            body
        };
        frame.push(b'B');
        frame.extend_from_slice(&((body.len() + 4) as i32).to_be_bytes());
        frame.extend_from_slice(&body);
        out.extend_from_slice(&frame);
    }

    {
        let body: Vec<u8> = vec![b'P', 0u8];
        let mut frame = vec![b'D'];
        frame.extend_from_slice(&((body.len() + 4) as i32).to_be_bytes());
        frame.extend_from_slice(&body);
        out.extend_from_slice(&frame);
    }

    {
        let body: Vec<u8> = {
            let mut body = Vec::new();
            body.push(0u8);
            body.extend_from_slice(&0i32.to_be_bytes());
            body
        };
        let mut frame = vec![b'E'];
        frame.extend_from_slice(&((body.len() + 4) as i32).to_be_bytes());
        frame.extend_from_slice(&body);
        out.extend_from_slice(&frame);
    }

    {
        let mut frame = vec![b'S'];
        frame.extend_from_slice(&4i32.to_be_bytes());
        out.extend_from_slice(&frame);
    }

    out
}

fn build_extended_pipeline_pgwire() -> BytesMut {
    let mut buf = BytesMut::with_capacity(256);
    Parse::new(
        Some("find-user".to_owned()),
        "select id, email from users where id = $1".to_owned(),
        vec![23],
    )
    .encode(&mut buf)
    .expect("parse");
    Bind::new(
        Some("".to_owned()),
        Some("find-user".to_owned()),
        vec![1],
        vec![Some(bytes::Bytes::copy_from_slice(&1234i64.to_be_bytes()))],
        vec![0],
    )
    .encode(&mut buf)
    .expect("bind");
    Describe::new(TARGET_TYPE_BYTE_PORTAL, Some("".to_owned()))
        .encode(&mut buf)
        .expect("describe");
    Execute::new(Some("".to_owned()), 0)
        .encode(&mut buf)
        .expect("execute");
    Sync::new().encode(&mut buf).expect("sync");
    buf
}

fn build_simple_query_bytes() -> Vec<u8> {
    let sql = b"select id, email, created_at from users limit 10";
    let mut body = Vec::new();
    body.extend_from_slice(sql);
    body.push(0u8);
    let mut frame = vec![b'Q'];
    frame.extend_from_slice(&((body.len() + 4) as i32).to_be_bytes());
    frame.extend_from_slice(&body);
    frame
}

fn build_simple_query_pgwire() -> BytesMut {
    let mut buf = BytesMut::with_capacity(80);
    pgwire::messages::simplequery::Query::new(
        "select id, email, created_at from users limit 10".to_owned(),
    )
    .encode(&mut buf)
    .expect("query encode");
    buf
}

fn build_startup_bytes() -> Vec<u8> {
    let pairs: &[u8] =
        b"user\0brian\0database\0mydb\0application_name\0psql\0client_encoding\0UTF8\0\0";
    let version_be = ProtocolVersion::V3_0.as_code().to_be_bytes();
    let body_len = 4 + pairs.len();
    let total = (body_len + 4) as i32;
    let mut out = Vec::with_capacity(4 + body_len);
    out.extend_from_slice(&total.to_be_bytes());
    out.extend_from_slice(&version_be);
    out.extend_from_slice(pairs);
    out
}

fn build_startup_pgwire() -> BytesMut {
    let mut buf = BytesMut::with_capacity(128);
    let mut msg = pgwire::messages::startup::Startup::default();
    msg.parameters.insert("user".to_owned(), "brian".to_owned());
    msg.parameters
        .insert("database".to_owned(), "mydb".to_owned());
    msg.parameters
        .insert("application_name".to_owned(), "psql".to_owned());
    msg.parameters
        .insert("client_encoding".to_owned(), "UTF8".to_owned());
    msg.encode(&mut buf).expect("startup encode");
    buf
}

fn proxima_rd_fields() -> Vec<ProxFieldDescription<'static>> {
    vec![
        ProxFieldDescription {
            name: PgStr::new(b"id"),
            table_oid: 16384,
            column_attr: 1,
            type_oid: Oid(23),
            type_size: 4,
            type_modifier: -1,
            format: FormatCode::Text,
        },
        ProxFieldDescription {
            name: PgStr::new(b"email"),
            table_oid: 16384,
            column_attr: 2,
            type_oid: Oid(25),
            type_size: -1,
            type_modifier: -1,
            format: FormatCode::Text,
        },
        ProxFieldDescription {
            name: PgStr::new(b"created_at"),
            table_oid: 16384,
            column_attr: 3,
            type_oid: Oid(1184),
            type_size: 8,
            type_modifier: -1,
            format: FormatCode::Text,
        },
        ProxFieldDescription {
            name: PgStr::new(b"active"),
            table_oid: 16384,
            column_attr: 4,
            type_oid: Oid(16),
            type_size: 1,
            type_modifier: -1,
            format: FormatCode::Text,
        },
        ProxFieldDescription {
            name: PgStr::new(b"score"),
            table_oid: 16384,
            column_attr: 5,
            type_oid: Oid(701),
            type_size: 8,
            type_modifier: -1,
            format: FormatCode::Text,
        },
        ProxFieldDescription {
            name: PgStr::new(b"payload"),
            table_oid: 16384,
            column_attr: 6,
            type_oid: Oid(17),
            type_size: -1,
            type_modifier: -1,
            format: FormatCode::Text,
        },
    ]
}

fn pgwire_rd_fields() -> RowDescription {
    let mut rd = RowDescription::default();
    let specs: &[(&str, i32, i16, u32, i16)] = &[
        ("id", 16384, 1, 23, 4),
        ("email", 16384, 2, 25, -1),
        ("created_at", 16384, 3, 1184, 8),
        ("active", 16384, 4, 16, 1),
        ("score", 16384, 5, 701, 8),
        ("payload", 16384, 6, 17, -1),
    ];
    for (name, table_id, column_id, type_id, type_size) in specs {
        let mut field = pgwire::messages::data::FieldDescription::default();
        field.name = (*name).to_owned();
        field.table_id = *table_id;
        field.column_id = *column_id;
        field.type_id = *type_id;
        field.type_size = *type_size;
        field.type_modifier = -1;
        field.format_code = FORMAT_CODE_TEXT;
        rd.fields.push(field);
    }
    rd
}

fn encoded_rd_len() -> u64 {
    let rd = pgwire_rd_fields();
    let mut buf = BytesMut::new();
    rd.encode(&mut buf).expect("encode");
    buf.len() as u64
}

fn bench_decode_extended_pipeline(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("decode_extended_pipeline");
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));

    let proxima_bytes = build_pipeline_wire();
    let pgwire_bytes = build_extended_pipeline_pgwire();
    group.throughput(Throughput::Bytes(proxima_bytes.len() as u64));

    group.bench_function(BenchmarkId::new("proxima", "5msg"), |bencher| {
        bencher.iter(|| {
            let buf = black_box(&proxima_bytes[..]);
            let mut offset = 0usize;
            while offset < buf.len() {
                let (msg, consumed) = parse_frontend(&buf[offset..])
                    .expect("valid")
                    .expect("complete");
                black_box(msg);
                offset += consumed;
            }
        });
    });

    group.bench_function(BenchmarkId::new("pgwire028", "5msg"), |bencher| {
        bencher.iter_batched(
            || pgwire_bytes.clone(),
            |mut buf| {
                while buf.len() > 1 {
                    let msg = pgwire::messages::PgWireFrontendMessage::decode(black_box(&mut buf))
                        .expect("valid")
                        .expect("complete");
                    black_box(msg);
                }
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_decode_query_simple(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("decode_query_simple");
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));

    let proxima_bytes = build_simple_query_bytes();
    let pgwire_bytes = build_simple_query_pgwire();
    group.throughput(Throughput::Bytes(proxima_bytes.len() as u64));

    group.bench_function(BenchmarkId::new("proxima", "query"), |bencher| {
        bencher.iter(|| {
            let (msg, _) = parse_frontend(black_box(&proxima_bytes))
                .expect("valid")
                .expect("complete");
            black_box(msg);
        });
    });

    group.bench_function(BenchmarkId::new("pgwire028", "query"), |bencher| {
        bencher.iter_batched(
            || pgwire_bytes.clone(),
            |mut buf| {
                let msg = pgwire::messages::PgWireFrontendMessage::decode(black_box(&mut buf))
                    .expect("valid")
                    .expect("complete");
                black_box(msg);
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn encode_datarow_proxima(cols: &[Option<&[u8]>], out: &mut [u8]) -> usize {
    let mut writer = DataRowWriter::begin(out).expect("begin");
    for col in cols {
        match col {
            Some(val) => {
                writer.column(val).expect("column");
            }
            None => {
                writer.null().expect("null");
            }
        }
    }
    writer.finish().expect("finish")
}

fn encode_datarow_pgwire(cols: &[Option<&[u8]>]) -> BytesMut {
    let field_count = cols.len() as i16;
    let mut data = BytesMut::new();
    for col in cols {
        match col {
            Some(val) => {
                data.put_i32(val.len() as i32);
                data.put_slice(val);
            }
            None => data.put_i32(-1),
        }
    }
    let row = DataRow::new(data, field_count);
    let mut out = BytesMut::new();
    row.encode(&mut out).expect("encode");
    out
}

fn bench_encode_datarow(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("encode_datarow");
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));

    let narrow: Vec<Option<&[u8]>> = vec![
        Some(b"1234"),
        Some(b"alice@example.com"),
        Some(b"some text value here!!"),
        None,
    ];
    let wide: Vec<Option<&[u8]>> = (0..16)
        .map(|_| Some(b"abcdefghijklmnopqrstuvwxyz01234567890123456789012345678901234" as &[u8]))
        .collect();
    let large_data = vec![b'x'; 1024];
    let large: Vec<Option<&[u8]>> = vec![Some(large_data.as_slice())];

    for (label, cols) in [
        ("narrow_4col", narrow.as_slice()),
        ("wide_16col", wide.as_slice()),
        ("large_1col_1kb", large.as_slice()),
    ] {
        let encoded_len = encode_datarow_pgwire(cols).len() as u64;
        group.throughput(Throughput::Bytes(encoded_len));

        group.bench_function(BenchmarkId::new("proxima", label), |bencher| {
            let mut out = vec![0u8; 8192];
            bencher.iter(|| {
                let written = encode_datarow_proxima(black_box(cols), &mut out);
                black_box(written);
            });
        });

        group.bench_function(BenchmarkId::new("pgwire028", label), |bencher| {
            bencher.iter(|| {
                let out = encode_datarow_pgwire(black_box(cols));
                black_box(out);
            });
        });
    }

    group.finish();
}

fn bench_encode_rowdescription(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("encode_rowdescription");
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));

    let prox_fields = proxima_rd_fields();
    group.throughput(Throughput::Bytes(encoded_rd_len()));

    group.bench_function(BenchmarkId::new("proxima", "6fields"), |bencher| {
        let mut out = [0u8; 512];
        bencher.iter(|| {
            let mut writer = RowDescriptionWriter::begin(&mut out).expect("begin");
            for field in black_box(&prox_fields) {
                writer.field(field).expect("field");
            }
            let written = writer.finish().expect("finish");
            black_box(written);
        });
    });

    group.bench_function(BenchmarkId::new("pgwire028", "6fields"), |bencher| {
        bencher.iter(|| {
            let rd = black_box(pgwire_rd_fields());
            let mut buf = BytesMut::new();
            rd.encode(&mut buf).expect("encode");
            black_box(buf);
        });
    });

    group.finish();
}

fn bench_parse_startup(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("parse_startup");
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));

    let proxima_bytes = build_startup_bytes();
    let pgwire_bytes = build_startup_pgwire();
    group.throughput(Throughput::Bytes(proxima_bytes.len() as u64));

    group.bench_function(BenchmarkId::new("proxima", "startup"), |bencher| {
        bencher.iter(|| {
            let (msg, _) = parse_initial(black_box(&proxima_bytes))
                .expect("valid")
                .expect("complete");
            black_box(msg);
        });
    });

    group.bench_function(BenchmarkId::new("pgwire028", "startup"), |bencher| {
        bencher.iter_batched(
            || pgwire_bytes.clone(),
            |mut buf| {
                let msg = pgwire::messages::startup::Startup::decode(black_box(&mut buf))
                    .expect("valid")
                    .expect("complete");
                black_box(msg);
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_encode_auth_ok_ready_sequence(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("encode_auth_ok_ready_sequence");
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));

    let encoded_len = {
        let mut buf = BytesMut::new();
        Authentication::Ok.encode(&mut buf).expect("auth");
        ParameterStatus::new("server_version".to_owned(), "15.0".to_owned())
            .encode(&mut buf)
            .expect("ps1");
        ParameterStatus::new("client_encoding".to_owned(), "UTF8".to_owned())
            .encode(&mut buf)
            .expect("ps2");
        BackendKeyData::new(42, 99999)
            .encode(&mut buf)
            .expect("bkd");
        ReadyForQuery::new(PgTransactionStatus::Idle)
            .encode(&mut buf)
            .expect("rfq");
        buf.len() as u64
    };
    group.throughput(Throughput::Bytes(encoded_len));

    group.bench_function(BenchmarkId::new("proxima", "5msgs"), |bencher| {
        let mut out = [0u8; 512];
        bencher.iter(|| {
            let mut pos = 0usize;
            pos += BackendMessage::Authentication(AuthRequest::Ok)
                .encode(black_box(&mut out[pos..]))
                .expect("auth");
            pos += BackendMessage::ParameterStatus {
                name: PgStr::new(b"server_version"),
                value: PgStr::new(b"15.0"),
            }
            .encode(&mut out[pos..])
            .expect("ps1");
            pos += BackendMessage::ParameterStatus {
                name: PgStr::new(b"client_encoding"),
                value: PgStr::new(b"UTF8"),
            }
            .encode(&mut out[pos..])
            .expect("ps2");
            pos += BackendMessage::BackendKeyData {
                process_id: 42,
                secret_key: &[0x00, 0x01, 0x86, 0x9f],
            }
            .encode(&mut out[pos..])
            .expect("bkd");
            pos += BackendMessage::ReadyForQuery {
                status: TransactionStatus::Idle,
            }
            .encode(&mut out[pos..])
            .expect("rfq");
            black_box(pos);
        });
    });

    group.bench_function(BenchmarkId::new("pgwire028", "5msgs"), |bencher| {
        bencher.iter(|| {
            let mut buf = BytesMut::with_capacity(512);
            Authentication::Ok
                .encode(black_box(&mut buf))
                .expect("auth");
            ParameterStatus::new("server_version".to_owned(), "15.0".to_owned())
                .encode(&mut buf)
                .expect("ps1");
            ParameterStatus::new("client_encoding".to_owned(), "UTF8".to_owned())
                .encode(&mut buf)
                .expect("ps2");
            BackendKeyData::new(42, 99999)
                .encode(&mut buf)
                .expect("bkd");
            ReadyForQuery::new(PgTransactionStatus::Idle)
                .encode(&mut buf)
                .expect("rfq");
            black_box(buf);
        });
    });

    group.finish();
}

fn bench_reject_malformed(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("reject_malformed");
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));

    let unknown_tag: Vec<u8> = {
        let mut v = vec![0x99u8];
        v.extend_from_slice(&8i32.to_be_bytes());
        v.extend_from_slice(&[0u8; 3]);
        v
    };

    let truncated_bind: Vec<u8> = {
        let mut v = vec![b'B'];
        v.extend_from_slice(&200i32.to_be_bytes());
        v.extend_from_slice(&[0u8; 50]);
        v
    };

    let invalid_format_bind: Vec<u8> = {
        let body: Vec<u8> = {
            let mut body = Vec::new();
            body.push(0u8);
            body.push(0u8);
            body.extend_from_slice(&0i16.to_be_bytes());
            body.extend_from_slice(&1i16.to_be_bytes());
            body.extend_from_slice(&7i16.to_be_bytes());
            body.extend_from_slice(&0i16.to_be_bytes());
            body.extend_from_slice(&1i16.to_be_bytes());
            body.extend_from_slice(&7i16.to_be_bytes());
            body
        };
        let mut v = vec![b'B'];
        v.extend_from_slice(&((body.len() + 4) as i32).to_be_bytes());
        v.extend_from_slice(&body);
        v
    };

    group.bench_function("unknown_tag_0x99", |bencher| {
        bencher.iter(|| {
            let result = parse_frontend(black_box(&unknown_tag));
            let _ = black_box(result);
        });
    });

    group.bench_function("truncated_bind", |bencher| {
        bencher.iter(|| {
            let result = parse_frontend(black_box(&truncated_bind));
            let _ = black_box(result);
        });
    });

    group.bench_function("bind_invalid_format_code", |bencher| {
        bencher.iter(|| {
            let result = parse_frontend(black_box(&invalid_format_bind));
            let _ = black_box(result);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_decode_extended_pipeline,
    bench_decode_query_simple,
    bench_encode_datarow,
    bench_encode_rowdescription,
    bench_parse_startup,
    bench_encode_auth_ok_ready_sequence,
    bench_reject_malformed,
);
criterion_main!(benches);
