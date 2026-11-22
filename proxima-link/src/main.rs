//! `proxima-link` joins one proxima-nova arm64 Mach-O object into a running executable.
//!
//! L1 scope (SPEC R32): a single object, one `__text` section, no data, no external
//! symbols. The output is a static `MH_EXECUTE` with `__PAGEZERO`, `__TEXT`,
//! `__LINKEDIT`, an `LC_MAIN` entry at `_main`, and an ad-hoc code signature (arm64
//! macOS refuses to exec an unsigned binary). No platform linker is spawned.

use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::process::ExitCode;

const MH_MAGIC_64: u32 = 0xfeed_facf;
const MH_EXECUTE: u32 = 2;
const CPU_TYPE_ARM64: u32 = 0x0100_000c;

const LC_SEGMENT_64: u32 = 0x19;
const LC_SYMTAB: u32 = 0x2;
const LC_DYSYMTAB: u32 = 0xb;
const LC_CODE_SIGNATURE: u32 = 0x1d;
const LC_LOAD_DYLINKER: u32 = 0xe;
const LC_BUILD_VERSION: u32 = 0x32;
const LC_MAIN: u32 = 0x8000_0028;
const LC_DYLD_INFO_ONLY: u32 = 0x8000_0022;

const PAGE: u64 = 0x4000;
const SIGN_PAGE: u64 = 0x1000;
const VM_BASE: u64 = 0x1_0000_0000;
const PROT_RX: i32 = 5;

/// A section read from the input object.
struct Section {
    name: String,
    segname: String,
    addr: u64,
    size: u64,
    offset: u64,
    align: u32,
}

/// A defined symbol read from the input object.
struct Symbol {
    name: String,
    value: u64,
    n_type: u8,
    n_sect: u8,
    n_desc: u16,
}

struct Object {
    text: Section,
    symbols: Vec<Symbol>,
    platform: u32,
    minos: u32,
}

fn u32_at(b: &[u8], at: usize) -> Result<u32, String> {
    b.get(at..at + 4)
        .map(|s| u32::from_le_bytes(s.try_into().unwrap()))
        .ok_or_else(|| "truncated object".to_string())
}

fn u64_at(b: &[u8], at: usize) -> Result<u64, String> {
    b.get(at..at + 8)
        .map(|s| u64::from_le_bytes(s.try_into().unwrap()))
        .ok_or_else(|| "truncated object".to_string())
}

fn cstr16(b: &[u8], at: usize) -> String {
    let raw = &b[at..at + 16];
    let end = raw.iter().position(|&c| c == 0).unwrap_or(16);
    String::from_utf8_lossy(&raw[..end]).into_owned()
}

fn parse(bytes: &[u8]) -> Result<Object, String> {
    if u32_at(bytes, 0)? != MH_MAGIC_64 {
        return Err("not a 64-bit Mach-O object".into());
    }
    if u32_at(bytes, 12)? != 1 {
        return Err("not an MH_OBJECT".into());
    }
    let ncmds = u32_at(bytes, 16)? as usize;
    let sizeofcmds = u32_at(bytes, 20)? as usize;
    if 32 + sizeofcmds > bytes.len() {
        return Err("load commands run past the file".into());
    }

    let mut text: Option<Section> = None;
    let mut symbols_at: Option<(usize, usize, usize, usize)> = None;
    let mut platform = 1;
    let mut minos = 0x000b_0000;
    let mut at = 32;
    for _ in 0..ncmds {
        let cmd = u32_at(bytes, at)?;
        let cmdsize = u32_at(bytes, at + 4)? as usize;
        if cmdsize < 8 || at + cmdsize > bytes.len() {
            return Err("bad load command size".into());
        }
        match cmd {
            LC_SEGMENT_64 => {
                let nsects = u32_at(bytes, at + 64)? as usize;
                let mut sect = at + 72;
                for _ in 0..nsects {
                    let s = Section {
                        name: cstr16(bytes, sect),
                        segname: cstr16(bytes, sect + 16),
                        addr: u64_at(bytes, sect + 32)?,
                        size: u64_at(bytes, sect + 40)?,
                        offset: u32_at(bytes, sect + 48)? as u64,
                        align: u32_at(bytes, sect + 52)?,
                    };
                    if s.name == "__text" && s.segname == "__TEXT" {
                        if text.is_some() {
                            return Err("more than one __TEXT,__text section".into());
                        }
                        text = Some(s);
                    } else if s.size != 0 && s.name != "__text" {
                        return Err(format!(
                            "data section {} is later than L1 (no data yet)",
                            s.name
                        ));
                    }
                    sect += 80;
                }
            }
            LC_SYMTAB => {
                symbols_at = Some((
                    u32_at(bytes, at + 8)? as usize,
                    u32_at(bytes, at + 12)? as usize,
                    u32_at(bytes, at + 16)? as usize,
                    u32_at(bytes, at + 20)? as usize,
                ));
            }
            LC_BUILD_VERSION => {
                platform = u32_at(bytes, at + 8)?;
                minos = u32_at(bytes, at + 12)?;
            }
            _ => {}
        }
        at += cmdsize;
    }

    let text = text.ok_or_else(|| "no __TEXT,__text section".to_string())?;
    let (symoff, nsyms, stroff, strsize) =
        symbols_at.ok_or_else(|| "no LC_SYMTAB".to_string())?;
    if stroff + strsize > bytes.len() || symoff + nsyms * 16 > bytes.len() {
        return Err("symbol table runs past the file".into());
    }

    let mut symbols = Vec::with_capacity(nsyms);
    for i in 0..nsyms {
        let at = symoff + i * 16;
        let n_strx = u32_at(bytes, at)? as usize;
        let n_type = bytes[at + 4];
        let n_sect = bytes[at + 5];
        let n_desc = u16::from_le_bytes([bytes[at + 6], bytes[at + 7]]);
        let value = u64_at(bytes, at + 8)?;
        let name = if n_strx < strsize {
            let raw = &bytes[stroff + n_strx..stroff + strsize];
            let end = raw.iter().position(|&c| c == 0).unwrap_or(raw.len());
            String::from_utf8_lossy(&raw[..end]).into_owned()
        } else {
            String::new()
        };
        symbols.push(Symbol {
            name,
            value,
            n_type,
            n_sect,
            n_desc,
        });
    }

    Ok(Object {
        text,
        symbols,
        platform,
        minos,
    })
}

fn align_up(value: u64, align: u64) -> u64 {
    value.div_ceil(align) * align
}

fn be32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_be_bytes());
}

fn be64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_be_bytes());
}

fn le32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn le64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn segment(name: &str, vmaddr: u64, vmsize: u64, fileoff: u64, filesize: u64, prot: i32) -> Vec<u8> {
    let mut out = Vec::with_capacity(72);
    le32(&mut out, LC_SEGMENT_64);
    le32(&mut out, 72);
    let mut seg = [0u8; 16];
    seg[..name.len()].copy_from_slice(name.as_bytes());
    out.extend_from_slice(&seg);
    le64(&mut out, vmaddr);
    le64(&mut out, vmsize);
    le64(&mut out, fileoff);
    le64(&mut out, filesize);
    le32(&mut out, prot as u32);
    le32(&mut out, prot as u32);
    le32(&mut out, 0);
    le32(&mut out, 0);
    out
}

fn text_section(section: &Section) -> Vec<u8> {
    let mut out = Vec::with_capacity(80);
    let mut sect = [0u8; 16];
    sect[..section.name.len()].copy_from_slice(section.name.as_bytes());
    out.extend_from_slice(&sect);
    let mut seg = [0u8; 16];
    seg[..section.segname.len()].copy_from_slice(section.segname.as_bytes());
    out.extend_from_slice(&seg);
    le64(&mut out, section.addr);
    le64(&mut out, section.size);
    le32(&mut out, section.offset as u32);
    le32(&mut out, section.align);
    le32(&mut out, 0);
    le32(&mut out, 0);
    le32(&mut out, 0x8000_0400);
    le32(&mut out, 0);
    le32(&mut out, 0);
    le32(&mut out, 0);
    out
}

fn nlist(name: &str, strtab: &mut Vec<u8>, value: u64, n_type: u8, n_sect: u8, n_desc: u16) -> Vec<u8> {
    let n_strx = strtab.len() as u32;
    strtab.extend_from_slice(name.as_bytes());
    strtab.push(0);
    let mut out = Vec::with_capacity(16);
    le32(&mut out, n_strx);
    out.push(n_type);
    out.push(n_sect);
    out.extend_from_slice(&n_desc.to_le_bytes());
    le64(&mut out, value);
    out
}

fn code_directory(
    ident: &str,
    text_size: u64,
    code_limit: u64,
    hashes: &[u8],
    n_code: u32,
) -> Vec<u8> {
    let hash_off = 88 + ident.len() as u32 + 1;
    let length = hash_off + 32 * n_code;
    let mut out = Vec::with_capacity(length as usize);
    be32(&mut out, 0xfade_0c02);
    be32(&mut out, length);
    be32(&mut out, 0x0002_0400);
    be32(&mut out, 0x0002_0002);
    be32(&mut out, hash_off);
    be32(&mut out, 88);
    be32(&mut out, 0);
    be32(&mut out, n_code);
    be32(&mut out, code_limit as u32);
    out.push(32);
    out.push(2);
    out.push(0);
    out.push(12);
    be32(&mut out, 0);
    be32(&mut out, 0);
    be32(&mut out, 0);
    be32(&mut out, 0);
    be64(&mut out, 0);
    be64(&mut out, 0);
    be64(&mut out, text_size);
    be64(&mut out, 1);
    out.extend_from_slice(ident.as_bytes());
    out.push(0);
    assert_eq!(out.len(), hash_off as usize);
    out.extend_from_slice(hashes);
    out
}

fn signature(ident: &str, text_size: u64, bytes: &[u8]) -> Vec<u8> {
    let code_limit = bytes.len() as u64;
    let n_code = code_limit.div_ceil(SIGN_PAGE) as u32;
    let mut hashes = Vec::with_capacity(n_code as usize * 32);
    for page in 0..n_code as u64 {
        let start = page * SIGN_PAGE;
        let end = ((page + 1) * SIGN_PAGE).min(code_limit);
        let digest = Sha256::digest(&bytes[start as usize..end as usize]);
        hashes.extend_from_slice(&digest);
    }
    let cd = code_directory(ident, text_size, code_limit, &hashes, n_code);
    let mut out = Vec::with_capacity(20 + cd.len());
    be32(&mut out, 0xfade_0cc0);
    be32(&mut out, 20 + cd.len() as u32);
    be32(&mut out, 1);
    be32(&mut out, 0);
    be32(&mut out, 20);
    out.extend_from_slice(&cd);
    out
}

fn link(object: &[u8], ident: &str) -> Result<Vec<u8>, String> {
    let obj = parse(object)?;
    let main = obj
        .symbols
        .iter()
        .find(|s| s.name == "_main" && s.n_type & 0x0e == 0x0e && s.n_sect == 1)
        .ok_or_else(|| "_main is undefined".to_string())?;

    // layout: header + commands, then __text at a 64-byte file offset
    let text_cmd = 72 + 80;
    // __PAGEZERO, __TEXT, __LINKEDIT, dylinker, build, symtab, dysymtab, main, dyld info, sig
    let cmds_size = text_cmd + 72 + 72 + 32 + 24 + 24 + 80 + 24 + 48 + 16;
    let text_start = align_up(32 + cmds_size as u64, 64);
    let text_size = obj.text.size;
    let text_page = align_up(text_start + text_size, PAGE);
    let symoff = text_page;
    let nsyms = obj.symbols.len() as u64;
    let stroff = symoff + nsyms * 16;
    let mut strtab_len = 1u64; // leading NUL
    for s in &obj.symbols {
        strtab_len += s.name.len() as u64 + 1;
    }
    let sig_off = align_up(stroff + strtab_len, 16);
    let linkedit_off = PAGE;

    let entryoff = text_start + (main.value - obj.text.addr);
    let text_vmaddr = VM_BASE + text_start;
    let mut commands = Vec::new();

    commands.extend(segment("__PAGEZERO", 0, VM_BASE, 0, 0, 0));
    let mut text_seg = segment("__TEXT", VM_BASE, text_page, 0, text_page, PROT_RX);
    text_seg[4..8].copy_from_slice(&152u32.to_le_bytes()); // cmdsize includes one section
    text_seg[64..68].copy_from_slice(&1u32.to_le_bytes()); // nsects
    text_seg.extend_from_slice(&text_section(&Section {
        name: "__text".into(),
        segname: "__TEXT".into(),
        addr: text_vmaddr,
        size: text_size,
        offset: text_start,
        align: 6,
    }));
    commands.extend(text_seg);

    let mut linkedit = segment("__LINKEDIT", VM_BASE + linkedit_off, 0, linkedit_off, 0, 1);
    linkedit.truncate(72);
    commands.extend(linkedit);

    // LC_LOAD_DYLINKER: macOS runs user-space executables through dyld even with no dylibs
    let dylinker = b"/usr/lib/dyld";
    let dylinker_cmdsize = (12 + dylinker.len() + 1 + 7) & !7;
    le32(&mut commands, LC_LOAD_DYLINKER);
    le32(&mut commands, dylinker_cmdsize as u32);
    le32(&mut commands, 12);
    commands.extend_from_slice(dylinker);
    commands.push(0);
    commands.resize(commands.len().next_multiple_of(8), 0);

    // LC_BUILD_VERSION
    le32(&mut commands, LC_BUILD_VERSION);
    le32(&mut commands, 24);
    le32(&mut commands, obj.platform);
    le32(&mut commands, obj.minos);
    le32(&mut commands, 0);
    le32(&mut commands, 0);

    // LC_SYMTAB
    le32(&mut commands, LC_SYMTAB);
    le32(&mut commands, 24);
    le32(&mut commands, symoff as u32);
    le32(&mut commands, nsyms as u32);
    le32(&mut commands, stroff as u32);
    le32(&mut commands, strtab_len as u32);

    // LC_DYSYMTAB
    le32(&mut commands, LC_DYSYMTAB);
    le32(&mut commands, 80);
    le32(&mut commands, 0);
    le32(&mut commands, 0);
    le32(&mut commands, 0);
    le32(&mut commands, nsyms as u32);
    le32(&mut commands, nsyms as u32);
    le32(&mut commands, 0);
    for _ in 0..12 {
        le32(&mut commands, 0);
    }

    // LC_MAIN
    le32(&mut commands, LC_MAIN);
    le32(&mut commands, 24);
    le64(&mut commands, entryoff);
    le64(&mut commands, 0);

    // LC_DYLD_INFO_ONLY with no fixups: L1 has no rebases or binds
    le32(&mut commands, LC_DYLD_INFO_ONLY);
    le32(&mut commands, 48);
    for _ in 0..10 {
        le32(&mut commands, 0);
    }

    // LC_CODE_SIGNATURE, dataoff patched once the signature offset is known
    le32(&mut commands, LC_CODE_SIGNATURE);
    le32(&mut commands, 16);
    le32(&mut commands, sig_off as u32);
    le32(&mut commands, 0);

    if commands.len() != cmds_size {
        return Err(format!(
            "command size {} != planned {}",
            commands.len(),
            cmds_size
        ));
    }

    // header
    let mut out = Vec::new();
    le32(&mut out, MH_MAGIC_64);
    le32(&mut out, CPU_TYPE_ARM64);
    le32(&mut out, 0);
    le32(&mut out, MH_EXECUTE);
    le32(&mut out, 10);
    le32(&mut out, cmds_size as u32);
    le32(&mut out, 0x0020_0005);
    le32(&mut out, 0);
    out.extend_from_slice(&commands);
    out.resize(text_start as usize, 0);
    out.extend_from_slice(&object[obj.text.offset as usize..(obj.text.offset + obj.text.size) as usize]);
    out.resize(symoff as usize, 0);

    let mut strtab = vec![0u8];
    for s in &obj.symbols {
        let value = if s.n_sect == 1 && s.n_type & 0x0e == 0x0e {
            text_vmaddr + (s.value - obj.text.addr)
        } else {
            s.value
        };
        out.extend_from_slice(&nlist(
            &s.name, &mut strtab, value, s.n_type, s.n_sect, s.n_desc,
        ));
    }
    out.extend_from_slice(&strtab);
    out.resize(sig_off as usize, 0);

    // the signature length is fixed by codeLimit and the identifier, so patch every field
    // that lives inside the signed region BEFORE the page hashes are taken
    let n_code = sig_off.div_ceil(SIGN_PAGE) as u32;
    let hash_off = 88 + ident.len() as u64 + 1;
    let sig_len = (20 + hash_off + 32 * n_code as u64) as u32;
    let cmds_start = 32;
    let code_sig = cmds_start + (cmds_size - 16);
    out[code_sig + 12..code_sig + 16].copy_from_slice(&sig_len.to_le_bytes());

    let linkedit_filesize = (sig_off + sig_len as u64) - linkedit_off;
    let linkedit_vmsize = align_up(linkedit_filesize, PAGE);
    let linkedit_at = cmds_start + text_cmd + 72;
    out[linkedit_at + 32..linkedit_at + 40].copy_from_slice(&linkedit_vmsize.to_le_bytes());
    out[linkedit_at + 48..linkedit_at + 56].copy_from_slice(&linkedit_filesize.to_le_bytes());

    let sig = signature(ident, text_size, &out);
    assert_eq!(sig.len() as u32, sig_len);
    out.extend_from_slice(&sig);
    Ok(out)
}

/// `--verify <binary>`: recompute the CodeDirectory's page hashes over the file and count
/// mismatches. This is the structural check for a linker-signed blob, which `codesign -v`
/// trusts without re-reading the pages on this host.
fn verify(path: &str) -> Result<(), String> {
    let bytes = fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let ncmds = u32_at(&bytes, 16)? as usize;
    let mut at = 32;
    let mut sig = None;
    for _ in 0..ncmds {
        let cmd = u32_at(&bytes, at)?;
        let sz = u32_at(&bytes, at + 4)? as usize;
        if cmd == LC_CODE_SIGNATURE {
            sig = Some((u32_at(&bytes, at + 8)? as usize, u32_at(&bytes, at + 12)? as usize));
        }
        if sz < 8 {
            return Err("bad load command size".into());
        }
        at += sz;
    }
    let (dataoff, datasize) = sig.ok_or_else(|| "not signed: no LC_CODE_SIGNATURE".to_string())?;
    if dataoff + datasize > bytes.len() {
        return Err("LC_CODE_SIGNATURE runs past the file".into());
    }
    let blob = &bytes[dataoff..dataoff + datasize];
    if u32::from_be_bytes(blob[0..4].try_into().unwrap()) != 0xfade_0cc0 {
        return Err("not a SuperBlob".into());
    }
    let cd_off = u32::from_be_bytes(blob[16..20].try_into().unwrap()) as usize;
    let cd = &blob[cd_off..];
    if u32::from_be_bytes(cd[0..4].try_into().unwrap()) != 0xfade_0c02 {
        return Err("not a CodeDirectory".into());
    }
    let hash_off = u32::from_be_bytes(cd[16..20].try_into().unwrap()) as usize;
    let n_code = u32::from_be_bytes(cd[28..32].try_into().unwrap());
    let code_limit = u32::from_be_bytes(cd[32..36].try_into().unwrap()) as usize;
    let hash_size = cd[36] as usize;
    let page_size = 1usize << cd[39];
    if code_limit > bytes.len() || hash_off + hash_size * n_code as usize > cd.len() {
        return Err("CodeDirectory bounds are invalid".into());
    }
    let mut mismatches = 0;
    for page in 0..n_code as usize {
        let start = page * page_size;
        let end = ((page + 1) * page_size).min(code_limit);
        let digest = Sha256::digest(&bytes[start..end]);
        let stored = &cd[hash_off + page * hash_size..hash_off + (page + 1) * hash_size];
        if digest.as_slice() != stored {
            mismatches += 1;
        }
    }
    println!("hashes={n_code} mismatches={mismatches} code_limit={code_limit}");
    if mismatches == 0 {
        Ok(())
    } else {
        Err(format!("{mismatches} page hashes do not match"))
    }
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(first) = args.next() else {
        eprintln!("usage: proxima-link <object> -o <output> | --verify <binary>");
        return ExitCode::FAILURE;
    };
    if first == "--verify" {
        let Some(path) = args.next() else {
            eprintln!("usage: proxima-link --verify <binary>");
            return ExitCode::FAILURE;
        };
        return match verify(&path) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("proxima-link: {e}");
                ExitCode::FAILURE
            }
        };
    }
    let object = first;
    let mut output = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-o" => output = args.next(),
            other => {
                eprintln!("proxima-link: unexpected argument {other}");
                return ExitCode::FAILURE;
            }
        }
    }
    let Some(output) = output else {
        eprintln!("usage: proxima-link <object> -o <output>");
        return ExitCode::FAILURE;
    };
    let input = match fs::read(&object) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("proxima-link: cannot read {object}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let ident = std::path::Path::new(&output)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "a.out".into());
    match link(&input, &ident) {
        Ok(bytes) => {
            if let Err(e) = fs::write(&output, &bytes) {
                eprintln!("proxima-link: cannot write {output}: {e}");
                return ExitCode::FAILURE;
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = fs::set_permissions(&output, fs::Permissions::from_mode(0o755));
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("proxima-link: {e}");
            ExitCode::FAILURE
        }
    }
}
