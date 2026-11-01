use std::io::Write;

use proxima_core::ProximaError;

use crate::kernel_modules::KernelModule;
use crate::linux::MODULES_DIR;
use crate::oci::error::OciError;
use crate::oci::rootfs::{Entry, Node, Rootfs};
use crate::oci::runspec::{RUN_SPEC_PATH, RunSpec};

const TYPE_DIRECTORY: u32 = 0o040_000;
const TYPE_FILE: u32 = 0o100_000;
const TYPE_SYMLINK: u32 = 0o120_000;
const TYPE_CHAR: u32 = 0o020_000;
const TYPE_BLOCK: u32 = 0o060_000;
const TYPE_FIFO: u32 = 0o010_000;
const TTY_MAJOR: u32 = 4;
const COM2_MINOR: u32 = 65;
const CONSOLE_MAJOR: u32 = 5;
const CONSOLE_MINOR: u32 = 1;
const DEVICE_MODE: u32 = 0o600;
const INIT_MODE: u32 = 0o755;
const RUN_SPEC_MODE: u32 = 0o444;
const MODULE_MODE: u32 = 0o444;

fn put(rootfs: &mut Rootfs, path: &str, mode: u32, node: Node) {
    rootfs.insert(path.to_string(), Entry { mode, uid: 0, gid: 0, node });
}

/// Adds what the linux flavor needs around an image: `/init`, the run spec, and the console and COM2 nodes.
pub fn add_guest_files(rootfs: &mut Rootfs, init: &[u8], run: &RunSpec) -> Result<(), OciError> {
    put(rootfs, "dev", 0o755, Node::Directory);
    put(rootfs, "dev/console", DEVICE_MODE, Node::CharDevice { major: CONSOLE_MAJOR, minor: CONSOLE_MINOR });
    put(rootfs, "dev/ttyS1", DEVICE_MODE, Node::CharDevice { major: TTY_MAJOR, minor: COM2_MINOR });
    put(rootfs, "init", INIT_MODE, Node::File(init.to_vec()));
    put(rootfs, ".proxima", 0o755, Node::Directory);
    put(rootfs, RUN_SPEC_PATH, RUN_SPEC_MODE, Node::File(run.to_json()?));
    Ok(())
}

/// Adds `modules` under `.proxima/modules`, each prefixed with its load position so the guest loads them in order.
pub fn add_modules(rootfs: &mut Rootfs, modules: &[KernelModule]) {
    put(rootfs, ".proxima", 0o755, Node::Directory);
    put(rootfs, MODULES_DIR, 0o755, Node::Directory);
    for (index, module) in modules.iter().enumerate() {
        let path = format!("{MODULES_DIR}/{index:02}-{}", module.file_name);
        put(rootfs, &path, MODULE_MODE, Node::File(module.bytes.clone()));
    }
}

pub fn to_initramfs(rootfs: &Rootfs) -> Result<Vec<u8>, ProximaError> {
    let mut archive = Vec::new();
    for (index, (path, entry)) in rootfs.iter().enumerate() {
        let inode = u32::try_from(index + 1).map_err(|_| ProximaError::Config("rootfs holds more than 2^32 entries".into()))?;
        archive = write_entry(archive, path, entry, inode)?;
    }
    Ok(cpio::newc::trailer(archive)?)
}

fn write_entry(archive: Vec<u8>, path: &str, entry: &Entry, inode: u32) -> Result<Vec<u8>, ProximaError> {
    let (kind, links, contents, (major, minor)): (u32, u32, &[u8], (u32, u32)) = match &entry.node {
        Node::Directory => (TYPE_DIRECTORY, 2, &[], (0, 0)),
        Node::File(data) => (TYPE_FILE, 1, data, (0, 0)),
        Node::Symlink(target) => (TYPE_SYMLINK, 1, target.as_bytes(), (0, 0)),
        Node::CharDevice { major, minor } => (TYPE_CHAR, 1, &[], (*major, *minor)),
        Node::BlockDevice { major, minor } => (TYPE_BLOCK, 1, &[], (*major, *minor)),
        Node::Fifo => (TYPE_FIFO, 1, &[], (0, 0)),
    };
    let size = u32::try_from(contents.len())
        .map_err(|_| ProximaError::Config(format!("initramfs entry `{path}` of {} bytes exceeds newc's 4 GiB", contents.len())))?;
    let builder = cpio::newc::Builder::new(path)
        .ino(inode)
        .mode(kind | entry.mode)
        .uid(entry.uid)
        .gid(entry.gid)
        .nlink(links)
        .rdev_major(major)
        .rdev_minor(minor);
    let mut writer = builder.write(archive, size);
    writer.write_all(contents)?;
    Ok(writer.finish()?)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use std::collections::BTreeMap;
    use std::io::Read;

    use super::*;

    fn read_back(archive: &[u8]) -> BTreeMap<String, (u32, Vec<u8>)> {
        let mut cursor = std::io::Cursor::new(archive);
        let mut found = BTreeMap::new();
        loop {
            let mut reader = cpio::newc::Reader::new(&mut cursor).expect("newc header");
            if reader.entry().is_trailer() {
                return found;
            }
            let (name, mode) = (reader.entry().name().to_string(), reader.entry().mode());
            let mut data = Vec::new();
            reader.read_to_end(&mut data).expect("entry data");
            found.insert(name, (mode, data));
            reader.finish().expect("entry end");
        }
    }

    fn spec() -> RunSpec {
        RunSpec { argv: vec!["/whoami".into()], env: vec!["PATH=/bin".into()], cwd: "/".into() }
    }

    #[test]
    fn files_directories_symlinks_and_devices_survive_the_cpio_round_trip() {
        let mut rootfs = Rootfs::new();
        put(&mut rootfs, "bin", 0o755, Node::Directory);
        put(&mut rootfs, "bin/tool", 0o755, Node::File(b"\x7fELF".to_vec()));
        put(&mut rootfs, "bin/sh", 0o777, Node::Symlink("tool".into()));
        add_guest_files(&mut rootfs, b"init-binary", &spec()).expect("guest files");

        let archive = to_initramfs(&rootfs).expect("archive");

        let found = read_back(&archive);
        assert_eq!(&archive[..6], b"070701");
        assert_eq!(found["bin/tool"], (TYPE_FILE | 0o755, b"\x7fELF".to_vec()));
        assert_eq!(found["bin/sh"], (TYPE_SYMLINK | 0o777, b"tool".to_vec()));
        assert_eq!(found["init"], (TYPE_FILE | INIT_MODE, b"init-binary".to_vec()));
        assert_eq!(found["dev/console"].0, TYPE_CHAR | DEVICE_MODE);
        assert_eq!(found["dev/ttyS1"].0, TYPE_CHAR | DEVICE_MODE);
        assert_eq!(found[RUN_SPEC_PATH].1, spec().to_json().expect("json"));
    }

    #[test]
    fn parents_precede_children_in_the_archive() {
        let mut rootfs = Rootfs::new();
        put(&mut rootfs, "a", 0o755, Node::Directory);
        put(&mut rootfs, "a-b", 0o644, Node::File(Vec::new()));
        put(&mut rootfs, "a/c", 0o644, Node::File(Vec::new()));
        let archive = to_initramfs(&rootfs).expect("archive");
        let mut cursor = std::io::Cursor::new(archive);
        let mut names = Vec::new();
        loop {
            let reader = cpio::newc::Reader::new(&mut cursor).expect("header");
            if reader.entry().is_trailer() {
                break;
            }
            names.push(reader.entry().name().to_string());
            reader.finish().expect("end");
        }

        let parent = names.iter().position(|name| name == "a").expect("a");
        let child = names.iter().position(|name| name == "a/c").expect("a/c");

        assert!(parent < child, "{names:?}");
    }
}
