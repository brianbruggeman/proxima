#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use proxima_protocols::pgwire_codec::BackendMessage;
use proxima_protocols::pgwire_codec::backend::{DataRowWriter, RowDescriptionWriter, parse_backend};
use proxima_protocols::pgwire_codec::frontend::{parse_frontend, parse_initial};
use proxima_protocols::pgwire_codec::types::{FormatCode, Oid, TransactionStatus};
use proxima_protocols::pgwire_codec::views::FieldDescription;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn build_startup_bytes() -> Vec<u8> {
    let params: &[(&[u8], &[u8])] = &[
        (b"user", b"alice"),
        (b"database", b"appdb"),
        (b"application_name", b"testapp"),
        (b"client_encoding", b"UTF8"),
    ];
    let pairs_len: usize = params
        .iter()
        .map(|(key, val)| key.len() + 1 + val.len() + 1)
        .sum();
    let total = 4 + 4 + pairs_len + 1;
    let mut buf = vec![0u8; total];
    buf[..4].copy_from_slice(&(total as i32).to_be_bytes());
    let version: i32 = 3 << 16;
    buf[4..8].copy_from_slice(&version.to_be_bytes());
    let mut pos = 8;
    for (key, val) in params {
        buf[pos..pos + key.len()].copy_from_slice(key);
        pos += key.len();
        buf[pos] = 0;
        pos += 1;
        buf[pos..pos + val.len()].copy_from_slice(val);
        pos += val.len();
        buf[pos] = 0;
        pos += 1;
    }
    buf[pos] = 0;
    buf
}

fn build_parse_bytes() -> Vec<u8> {
    let sql = b"select id, email from users where id = $1";
    let stmt = b"get-user";
    let oid: u32 = 23;
    let body_len = stmt.len() + 1 + sql.len() + 1 + 2 + 4;
    let total = 1 + 4 + body_len;
    let mut buf = vec![0u8; total];
    buf[0] = b'P';
    buf[1..5].copy_from_slice(&((body_len + 4) as i32).to_be_bytes());
    let mut pos = 5;
    buf[pos..pos + stmt.len()].copy_from_slice(stmt);
    pos += stmt.len();
    buf[pos] = 0;
    pos += 1;
    buf[pos..pos + sql.len()].copy_from_slice(sql);
    pos += sql.len();
    buf[pos] = 0;
    pos += 1;
    buf[pos..pos + 2].copy_from_slice(&1i16.to_be_bytes());
    pos += 2;
    buf[pos..pos + 4].copy_from_slice(&oid.to_be_bytes());
    buf
}

fn build_bind_bytes() -> Vec<u8> {
    let portal = b"";
    let stmt = b"get-user";
    let param_val = b"42";
    let fmt_count = 0i16;
    let param_count = 1i16;
    let result_fmt_count = 0i16;
    let body_len = portal.len() + 1 + stmt.len() + 1 + 2 + 2 + 4 + param_val.len() + 2;
    let total = 1 + 4 + body_len;
    let mut buf = vec![0u8; total];
    buf[0] = b'B';
    buf[1..5].copy_from_slice(&((body_len + 4) as i32).to_be_bytes());
    let mut pos = 5;
    buf[pos] = 0;
    pos += 1;
    buf[pos..pos + stmt.len()].copy_from_slice(stmt);
    pos += stmt.len();
    buf[pos] = 0;
    pos += 1;
    buf[pos..pos + 2].copy_from_slice(&fmt_count.to_be_bytes());
    pos += 2;
    buf[pos..pos + 2].copy_from_slice(&param_count.to_be_bytes());
    pos += 2;
    buf[pos..pos + 4].copy_from_slice(&(param_val.len() as i32).to_be_bytes());
    pos += 4;
    buf[pos..pos + param_val.len()].copy_from_slice(param_val);
    pos += param_val.len();
    buf[pos..pos + 2].copy_from_slice(&result_fmt_count.to_be_bytes());
    buf
}

fn build_describe_bytes() -> Vec<u8> {
    let name = b"get-user";
    let body_len = 1 + name.len() + 1;
    let total = 1 + 4 + body_len;
    let mut buf = vec![0u8; total];
    buf[0] = b'D';
    buf[1..5].copy_from_slice(&((body_len + 4) as i32).to_be_bytes());
    buf[5] = b'S';
    buf[6..6 + name.len()].copy_from_slice(name);
    buf[6 + name.len()] = 0;
    buf
}

fn build_execute_bytes() -> Vec<u8> {
    let name = b"";
    let body_len = name.len() + 1 + 4;
    let total = 1 + 4 + body_len;
    let mut buf = vec![0u8; total];
    buf[0] = b'E';
    buf[1..5].copy_from_slice(&((body_len + 4) as i32).to_be_bytes());
    buf[5] = 0;
    buf[6..10].copy_from_slice(&0i32.to_be_bytes());
    buf
}

fn build_sync_bytes() -> Vec<u8> {
    let mut buf = vec![0u8; 5];
    buf[0] = b'S';
    buf[1..5].copy_from_slice(&4i32.to_be_bytes());
    buf
}

fn build_field_description<'a>() -> FieldDescription<'a> {
    FieldDescription {
        name: proxima_protocols::pgwire_codec::PgStr::new(b"id"),
        table_oid: 0,
        column_attr: 1,
        type_oid: Oid(23),
        type_size: 4,
        type_modifier: -1,
        format: FormatCode::Text,
    }
}

struct CodecFrames<'a> {
    startup: &'a [u8],
    parse: &'a [u8],
    bind: &'a [u8],
    describe: &'a [u8],
    execute: &'a [u8],
    sync: &'a [u8],
    field: FieldDescription<'a>,
}

fn run_cycle(frames: &CodecFrames<'_>, encode_buf: &mut [u8], row_buf: &mut [u8]) {
    let _ = parse_initial(frames.startup)
        .expect("startup parse must succeed")
        .expect("startup must be complete");

    let _ = parse_frontend(frames.parse)
        .expect("parse parse must succeed")
        .expect("must be complete");

    let _ = parse_frontend(frames.bind)
        .expect("bind parse must succeed")
        .expect("must be complete");

    let _ = parse_frontend(frames.describe)
        .expect("describe parse must succeed")
        .expect("must be complete");

    let _ = parse_frontend(frames.execute)
        .expect("execute parse must succeed")
        .expect("must be complete");

    let _ = parse_frontend(frames.sync)
        .expect("sync parse must succeed")
        .expect("must be complete");

    let row_desc_written = {
        let mut writer =
            RowDescriptionWriter::begin(encode_buf).expect("row desc begin must succeed");
        writer.field(&frames.field).expect("field must succeed");
        writer.finish().expect("row desc finish must succeed")
    };

    let _ = parse_backend(&encode_buf[..row_desc_written])
        .expect("row desc parse must succeed")
        .expect("must be complete");

    for col_idx in 0..1000i32 {
        let col_val = col_idx.to_be_bytes();
        let written = {
            let mut writer = DataRowWriter::begin(row_buf).expect("data row begin must succeed");
            writer.column(&col_val).expect("column must succeed");
            writer.finish().expect("data row finish must succeed")
        };
        let _ = parse_backend(&row_buf[..written])
            .expect("data row parse must succeed")
            .expect("must be complete");
    }

    let cmd_complete = BackendMessage::CommandComplete {
        tag: proxima_protocols::pgwire_codec::PgStr::new(b"SELECT 1000"),
    };
    let written = cmd_complete
        .encode(encode_buf)
        .expect("cmd complete encode must succeed");
    let _ = parse_backend(&encode_buf[..written])
        .expect("cmd complete parse must succeed")
        .expect("must be complete");

    let rfq = BackendMessage::ReadyForQuery {
        status: TransactionStatus::Idle,
    };
    let written = rfq.encode(encode_buf).expect("rfq encode must succeed");
    let _ = parse_backend(&encode_buf[..written])
        .expect("rfq parse must succeed")
        .expect("must be complete");
}

#[test]
fn zero_allocations_over_codec_hot_path() {
    let startup_bytes = build_startup_bytes();
    let parse_bytes = build_parse_bytes();
    let bind_bytes = build_bind_bytes();
    let describe_bytes = build_describe_bytes();
    let execute_bytes = build_execute_bytes();
    let sync_bytes = build_sync_bytes();

    let frames = CodecFrames {
        startup: &startup_bytes,
        parse: &parse_bytes,
        bind: &bind_bytes,
        describe: &describe_bytes,
        execute: &execute_bytes,
        sync: &sync_bytes,
        field: build_field_description(),
    };

    let mut encode_buf = vec![0u8; 16 * 1024];
    let mut row_buf = vec![0u8; 4096];

    run_cycle(&frames, &mut encode_buf, &mut row_buf);

    let count_before = ALLOC_COUNT.load(Ordering::Relaxed);

    for _ in 0..1000 {
        run_cycle(&frames, &mut encode_buf, &mut row_buf);
    }

    let count_after = ALLOC_COUNT.load(Ordering::Relaxed);
    assert_eq!(
        count_after,
        count_before,
        "codec hot path must not allocate: {} allocation(s) observed over 1000 iterations",
        count_after - count_before
    );
}
