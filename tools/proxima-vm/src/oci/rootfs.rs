use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::Component;

use flate2::read::GzDecoder;
use tar::{Archive, EntryType};

use crate::oci::error::OciError;
use crate::oci::registry::Layer;

const WHITEOUT_PREFIX: &str = ".wh.";
const OPAQUE_MARKER: &str = ".wh..wh..opq";
const DEFAULT_DIRECTORY_MODE: u32 = 0o755;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Node {
    Directory,
    File(Vec<u8>),
    Symlink(String),
    CharDevice { major: u32, minor: u32 },
    BlockDevice { major: u32, minor: u32 },
    Fifo,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub mode: u32,
    pub uid: u32,
    pub gid: u32,
    pub node: Node,
}

/// Slash-separated paths without a leading slash, sorted so a parent always precedes its children.
pub type Rootfs = BTreeMap<String, Entry>;

pub fn flatten(layers: &[Layer]) -> Result<Rootfs, OciError> {
    let mut rootfs = Rootfs::new();
    for layer in layers {
        apply_layer(&mut rootfs, layer)?;
    }
    Ok(rootfs)
}

pub fn apply_layer(rootfs: &mut Rootfs, layer: &Layer) -> Result<(), OciError> {
    let media_type = layer.media_type.as_str();
    if media_type.contains("zstd") {
        return Err(OciError::LayerMediaType { digest: layer.digest.to_string(), media_type: layer.media_type.clone() });
    }
    if media_type.contains("gzip") {
        unpack(rootfs, layer, GzDecoder::new(layer.bytes.as_slice()))
    } else {
        unpack(rootfs, layer, layer.bytes.as_slice())
    }
}

fn unpack<Source: Read>(rootfs: &mut Rootfs, layer: &Layer, source: Source) -> Result<(), OciError> {
    let failure = |source| OciError::Layer { digest: layer.digest.to_string(), source };
    let mut added = BTreeSet::new();
    for entry in Archive::new(source).entries().map_err(failure)? {
        let mut entry = entry.map_err(failure)?;
        let raw_path = entry.path().map_err(failure)?.into_owned();
        let Some(path) = normalize(&raw_path.to_string_lossy())? else { continue };
        let name = path.rsplit('/').next().unwrap_or(&path).to_string();
        let parent = path.rsplit_once('/').map_or("", |(parent, _)| parent).to_string();
        if name == OPAQUE_MARKER {
            remove_below(rootfs, &added, &parent);
            continue;
        }
        if let Some(target) = name.strip_prefix(WHITEOUT_PREFIX) {
            let victim = join(&parent, target);
            remove_below(rootfs, &added, &victim);
            if !added.contains(&victim) {
                rootfs.remove(&victim);
            }
            continue;
        }
        let header = entry.header();
        let mode = header.mode().map_err(failure)? & 0o7777;
        let uid = narrow(header.uid().map_err(failure)?, &path)?;
        let gid = narrow(header.gid().map_err(failure)?, &path)?;
        let node = match header.entry_type() {
            EntryType::Directory => Node::Directory,
            EntryType::Regular | EntryType::Continuous | EntryType::GNUSparse => {
                let mut data = Vec::with_capacity(usize::try_from(header.size().map_err(failure)?).unwrap_or(0));
                entry.read_to_end(&mut data).map_err(failure)?;
                Node::File(data)
            }
            EntryType::Symlink => Node::Symlink(link_target(&entry, &path)?),
            EntryType::Link => hard_link(rootfs, &normalize(&link_target(&entry, &path)?)?.unwrap_or_default(), &path)?,
            EntryType::Char => Node::CharDevice { major: major(&entry, &path)?, minor: minor(&entry, &path)? },
            EntryType::Block => Node::BlockDevice { major: major(&entry, &path)?, minor: minor(&entry, &path)? },
            EntryType::Fifo => Node::Fifo,
            _ => continue,
        };
        place(rootfs, &added, &path, Entry { mode, uid, gid, node });
        added.insert(path);
    }
    Ok(())
}

fn narrow(value: u64, path: &str) -> Result<u32, OciError> {
    u32::try_from(value).map_err(|_| OciError::Entry { path: path.to_string(), reason: format!("owner id {value} does not fit newc's 32 bits") })
}

fn join(parent: &str, name: &str) -> String {
    if parent.is_empty() { name.to_string() } else { format!("{parent}/{name}") }
}

fn normalize(raw: &str) -> Result<Option<String>, OciError> {
    let mut parts = Vec::new();
    for component in std::path::Path::new(raw).components() {
        match component {
            Component::Normal(part) => parts.push(part.to_string_lossy().into_owned()),
            Component::CurDir | Component::RootDir => {}
            Component::ParentDir | Component::Prefix(_) => {
                return Err(OciError::Entry { path: raw.to_string(), reason: "a `..` component would escape the rootfs".into() });
            }
        }
    }
    Ok((!parts.is_empty()).then(|| parts.join("/")))
}

fn link_target<R: Read>(entry: &tar::Entry<'_, R>, path: &str) -> Result<String, OciError> {
    entry
        .link_name()
        .ok()
        .flatten()
        .map(|target| target.to_string_lossy().into_owned())
        .ok_or_else(|| OciError::Entry { path: path.to_string(), reason: "link entry names no target".into() })
}

fn major<R: Read>(entry: &tar::Entry<'_, R>, path: &str) -> Result<u32, OciError> {
    entry.header().device_major().ok().flatten().ok_or_else(|| OciError::Entry { path: path.to_string(), reason: "device entry has no major number".into() })
}

fn minor<R: Read>(entry: &tar::Entry<'_, R>, path: &str) -> Result<u32, OciError> {
    entry.header().device_minor().ok().flatten().ok_or_else(|| OciError::Entry { path: path.to_string(), reason: "device entry has no minor number".into() })
}

fn hard_link(rootfs: &Rootfs, target: &str, path: &str) -> Result<Node, OciError> {
    match rootfs.get(target) {
        Some(Entry { node, .. }) => Ok(node.clone()),
        None => Err(OciError::Entry { path: path.to_string(), reason: format!("hard link target `{target}` is not in the rootfs") }),
    }
}

fn remove_below(rootfs: &mut Rootfs, added: &BTreeSet<String>, directory: &str) {
    let prefix = if directory.is_empty() { String::new() } else { format!("{directory}/") };
    let doomed: Vec<String> = rootfs
        .range(prefix.clone()..)
        .take_while(|(path, _)| path.starts_with(&prefix))
        .map(|(path, _)| path.clone())
        .filter(|path| !added.contains(path))
        .collect();
    for path in doomed {
        rootfs.remove(&path);
    }
}

fn place(rootfs: &mut Rootfs, added: &BTreeSet<String>, path: &str, entry: Entry) {
    ensure_parents(rootfs, path);
    let replaces_directory = rootfs.get(path).is_some_and(|existing| existing.node == Node::Directory) && entry.node != Node::Directory;
    if replaces_directory {
        remove_below(rootfs, added, path);
    }
    rootfs.insert(path.to_string(), entry);
}

fn ensure_parents(rootfs: &mut Rootfs, path: &str) {
    let mut parent = String::new();
    for component in path.split('/').take(path.matches('/').count()) {
        parent = join(&parent, component);
        rootfs
            .entry(parent.clone())
            .or_insert(Entry { mode: DEFAULT_DIRECTORY_MODE, uid: 0, gid: 0, node: Node::Directory });
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use std::io::Write;

    use flate2::Compression;
    use flate2::write::GzEncoder;
    use tar::{Builder, Header};

    use super::*;
    use crate::oci::digest::Digest;

    const GZIP: &str = "application/vnd.oci.image.layer.v1.tar+gzip";

    struct LayerBuilder(Builder<Vec<u8>>);

    impl LayerBuilder {
        fn new() -> Self {
            Self(Builder::new(Vec::new()))
        }

        fn header(kind: EntryType, mode: u32, size: u64) -> Header {
            let mut header = Header::new_gnu();
            header.set_entry_type(kind);
            header.set_mode(mode);
            header.set_size(size);
            header.set_uid(0);
            header.set_gid(0);
            header.set_mtime(0);
            header
        }

        fn directory(mut self, path: &str) -> Self {
            self.0.append_data(&mut Self::header(EntryType::Directory, 0o755, 0), path, std::io::empty()).expect("directory");
            self
        }

        fn file(mut self, path: &str, contents: &[u8]) -> Self {
            self.0.append_data(&mut Self::header(EntryType::Regular, 0o644, contents.len() as u64), path, contents).expect("file");
            self
        }

        fn symlink(mut self, path: &str, target: &str) -> Self {
            self.0.append_link(&mut Self::header(EntryType::Symlink, 0o777, 0), path, target).expect("symlink");
            self
        }

        fn hard_link(mut self, path: &str, target: &str) -> Self {
            self.0.append_link(&mut Self::header(EntryType::Link, 0o644, 0), path, target).expect("hard link");
            self
        }

        fn whiteout(self, directory: &str, name: &str) -> Self {
            self.file(&format!("{directory}.wh.{name}"), b"")
        }

        fn layer(self) -> Layer {
            let archive = self.0.into_inner().expect("tar finish");
            let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
            encoder.write_all(&archive).expect("gzip");
            let bytes = encoder.finish().expect("gzip finish");
            Layer { digest: Digest::of(&bytes), media_type: GZIP.into(), bytes }
        }
    }

    fn contents(rootfs: &Rootfs, path: &str) -> Option<Vec<u8>> {
        match rootfs.get(path) {
            Some(Entry { node: Node::File(data), .. }) => Some(data.clone()),
            _ => None,
        }
    }

    #[test]
    fn a_later_layer_overrides_a_file_and_parents_are_created() {
        let lower = LayerBuilder::new().file("etc/app/conf", b"old").layer();
        let upper = LayerBuilder::new().file("./etc/app/conf", b"new").layer();

        let rootfs = flatten(&[lower, upper]).expect("flatten");

        assert_eq!(contents(&rootfs, "etc/app/conf").expect("file"), b"new");
        assert_eq!(rootfs["etc"].node, Node::Directory);
        assert_eq!(rootfs["etc/app"].node, Node::Directory);
    }

    #[test]
    fn a_whiteout_deletes_the_lower_file_and_leaves_its_siblings() {
        let lower = LayerBuilder::new().file("var/cache/a", b"a").file("var/cache/b", b"b").layer();
        let upper = LayerBuilder::new().whiteout("var/cache/", "a").layer();

        let rootfs = flatten(&[lower, upper]).expect("flatten");

        assert!(!rootfs.contains_key("var/cache/a"), "whiteout removes the file");
        assert!(!rootfs.contains_key("var/cache/.wh.a"), "the marker itself is never materialized");
        assert_eq!(contents(&rootfs, "var/cache/b").expect("sibling"), b"b");
    }

    #[test]
    fn a_whiteout_of_a_directory_removes_its_whole_subtree() {
        let lower = LayerBuilder::new().directory("opt/tool/").file("opt/tool/bin/run", b"x").file("opt/keep", b"k").layer();
        let upper = LayerBuilder::new().whiteout("opt/", "tool").layer();

        let rootfs = flatten(&[lower, upper]).expect("flatten");

        assert!(rootfs.keys().all(|path| !path.starts_with("opt/tool")), "{:?}", rootfs.keys().collect::<Vec<_>>());
        assert!(rootfs.contains_key("opt/keep"));
    }

    #[test]
    fn an_opaque_marker_hides_lower_children_but_keeps_this_layers_own() {
        let lower = LayerBuilder::new().file("srv/data/old1", b"1").file("srv/data/old2", b"2").layer();
        let upper = LayerBuilder::new().file("srv/data/fresh", b"f").file("srv/data/.wh..wh..opq", b"").layer();

        let rootfs = flatten(&[lower, upper]).expect("flatten");

        assert!(!rootfs.contains_key("srv/data/old1") && !rootfs.contains_key("srv/data/old2"));
        assert_eq!(contents(&rootfs, "srv/data/fresh").expect("own file"), b"f");
        assert!(!rootfs.contains_key("srv/data/.wh..wh..opq"));
    }

    #[test]
    fn an_opaque_marker_listed_before_the_siblings_still_spares_them() {
        let lower = LayerBuilder::new().file("srv/data/old", b"o").layer();
        let upper = LayerBuilder::new().file("srv/data/.wh..wh..opq", b"").file("srv/data/fresh", b"f").layer();

        let rootfs = flatten(&[lower, upper]).expect("flatten");

        assert!(!rootfs.contains_key("srv/data/old"));
        assert_eq!(contents(&rootfs, "srv/data/fresh").expect("own file"), b"f");
    }

    #[test]
    fn a_file_replacing_a_directory_drops_the_directory_children() {
        let lower = LayerBuilder::new().directory("thing/").file("thing/inner", b"i").layer();
        let upper = LayerBuilder::new().file("thing", b"now a file").layer();

        let rootfs = flatten(&[lower, upper]).expect("flatten");

        assert_eq!(contents(&rootfs, "thing").expect("file"), b"now a file");
        assert!(!rootfs.contains_key("thing/inner"));
    }

    #[test]
    fn symlinks_and_hard_links_resolve_within_the_flattened_tree() {
        let lower = LayerBuilder::new().file("bin/busybox", b"ELF").symlink("bin/sh", "busybox").layer();
        let upper = LayerBuilder::new().hard_link("bin/ash", "bin/busybox").layer();

        let rootfs = flatten(&[lower, upper]).expect("flatten");

        assert_eq!(rootfs["bin/sh"].node, Node::Symlink("busybox".into()));
        assert_eq!(contents(&rootfs, "bin/ash").expect("hard link copies the data"), b"ELF");
    }

    #[test]
    fn a_hard_link_to_nothing_is_an_error_naming_the_path() {
        let layer = LayerBuilder::new().hard_link("bin/ash", "bin/missing").layer();

        let error = flatten(&[layer]).expect_err("dangling hard link");

        assert!(error.to_string().contains("bin/ash") && error.to_string().contains("bin/missing"), "{error}");
    }

    #[test]
    fn a_zstd_layer_is_refused_by_media_type() {
        let mut layer = LayerBuilder::new().file("a", b"a").layer();
        layer.media_type = "application/vnd.oci.image.layer.v1.tar+zstd".into();

        let error = flatten(&[layer]).expect_err("zstd unsupported");

        assert!(matches!(error, OciError::LayerMediaType { .. }), "{error}");
    }

    #[test]
    fn bytes_that_are_not_gzip_are_reported_with_the_layer_digest() {
        let bytes = b"definitely not gzip".to_vec();
        let layer = Layer { digest: Digest::of(&bytes), media_type: GZIP.into(), bytes };

        let error = flatten(std::slice::from_ref(&layer)).expect_err("not gzip");

        assert!(error.to_string().contains(layer.digest.as_str()), "{error}");
    }
}
