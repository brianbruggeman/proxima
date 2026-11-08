//! Read a GGUF header's key/value block and print the two flint per-layer
//! boolean arrays, so a fixture capture does not go through the reader those
//! arrays are used to hold to account.
//!
//! Deliberately standalone: no `proxima-gguf`, no llama.cpp bindings. The key
//! list is fixed to the two arrays this repo captures; the format is the wire
//! layout in `gguf.h` (magic, u32 version, u64 tensor count, u64 kv count, then
//! typed key/value pairs). The version is printed, not validated.

use std::env;
use std::fs::File;
use std::io::{BufReader, Read};
use std::process::ExitCode;

const KEYS: [&str; 2] = [
    "flint.attention.sliding_window_pattern",
    "flint.attention.rope_pattern",
];

const TYPE_UINT32: u32 = 4;
const TYPE_BOOL: u32 = 7;
const TYPE_STRING: u32 = 8;
const TYPE_ARRAY: u32 = 9;
const TYPE_UINT64: u32 = 10;

struct Cursor<R: Read> {
    reader: R,
}

impl<R: Read> Cursor<R> {
    fn take(&mut self, count: usize) -> Result<Vec<u8>, String> {
        let mut buffer = vec![0u8; count];
        self.reader
            .read_exact(&mut buffer)
            .map_err(|error| format!("short read of {count} bytes: {error}"))?;
        Ok(buffer)
    }

    fn u32(&mut self) -> Result<u32, String> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn u64(&mut self) -> Result<u64, String> {
        let bytes = self.take(8)?;
        let mut eight = [0u8; 8];
        eight.copy_from_slice(&bytes);
        Ok(u64::from_le_bytes(eight))
    }

    fn string(&mut self) -> Result<String, String> {
        let length = self.u64()?;
        let length = usize::try_from(length).map_err(|error| format!("string length: {error}"))?;
        let bytes = self.take(length)?;
        String::from_utf8(bytes).map_err(|error| format!("string is not utf8: {error}"))
    }
}

fn skip_scalar<R: Read>(cursor: &mut Cursor<R>, value_type: u32) -> Result<(), String> {
    let width = match value_type {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        TYPE_UINT32 | 5 | 6 => 4,
        TYPE_UINT64 | 11 | 12 => 8,
        TYPE_STRING => {
            cursor.string()?;
            return Ok(());
        }
        other => return Err(format!("unknown value type {other}")),
    };
    cursor.take(width)?;
    Ok(())
}

fn main() -> ExitCode {
    let Some(path) = env::args().nth(1) else {
        eprintln!("usage: gguf-kv-extract <checkpoint.gguf>");
        return ExitCode::from(2);
    };
    match run(&path) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("gguf-kv-extract: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(path: &str) -> Result<(), String> {
    let file = File::open(path).map_err(|error| format!("open {path}: {error}"))?;
    let mut cursor = Cursor {
        reader: BufReader::new(file),
    };

    let magic = cursor.u32()?;
    if magic != 0x4655_4747 {
        return Err(format!("not a gguf file: magic {magic:#x}"));
    }
    let version = cursor.u32()?;
    let tensor_count = cursor.u64()?;
    let kv_count = cursor.u64()?;
    println!("# version={version} tensors={tensor_count} kv_count={kv_count}");

    for _ in 0..kv_count {
        let key = cursor.string()?;
        let value_type = cursor.u32()?;
        if value_type != TYPE_ARRAY {
            skip_scalar(&mut cursor, value_type)?;
            continue;
        }
        let element_type = cursor.u32()?;
        let count = cursor.u64()?;
        if !KEYS.contains(&key.as_str()) {
            for _ in 0..count {
                skip_scalar(&mut cursor, element_type)?;
            }
            continue;
        }
        if element_type != TYPE_BOOL {
            return Err(format!("{key} is not a bool array"));
        }
        let mut entries = Vec::with_capacity(usize::try_from(count).unwrap_or(0));
        for _ in 0..count {
            entries.push(cursor.take(1)?[0] != 0);
        }
        let rendered: Vec<&str> = entries
            .iter()
            .map(|value| if *value { "true" } else { "false" })
            .collect();
        println!("{key} = [{}]", rendered.join(", "));
    }
    Ok(())
}
