//! Build-time constants of the lane, one per key of `proxima-mil.toml`.
//!
//! `build.rs` reads the toml through `proxima_build::sizing::SizingSource`,
//! applies the `PROXIMA_MIL_<SECTION>_<KEY>` environment overrides and emits
//! the generated module included below. No other source file holds a copy of
//! an axis; they read these constants.

mod generated {
    include!(concat!(env!("OUT_DIR"), "/proxima_mil_sized.rs"));
}

/// Largest single graph, in megabytes, the Neural Engine is expected to
/// accept; the initial value is a community figure until the graphs bench
/// replaces it with a measurement.
pub const GRAPH_CEILING_MB: usize = generated::GRAPH_CEILING_MB;

/// Sequence lengths are padded up to a multiple of this.
pub const PAD_MULTIPLE: usize = generated::PAD_MULTIPLE;

/// Longest padded sequence a program is emitted for.
pub const MAX_SEQUENCE: usize = generated::MAX_SEQUENCE;

/// Core ML specification version written into `Model.proto`.
pub const SPECIFICATION_VERSION: usize = generated::SPECIFICATION_VERSION;

/// MIL opset named by the emitted program.
pub const OPSET: &str = generated::OPSET;

/// Compute units used when a door's config names none.
pub const DEFAULT_COMPUTE_UNITS: &str = generated::DEFAULT_COMPUTE_UNITS;

#[cfg(test)]
mod tests {
    use std::format;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::string::{String, ToString};
    use std::vec::Vec;

    use proxima_build::sizing::SizingSource;
    use toml::Value;

    use super::{
        DEFAULT_COMPUTE_UNITS, GRAPH_CEILING_MB, MAX_SEQUENCE, OPSET, PAD_MULTIPLE,
        SPECIFICATION_VERSION,
    };

    const TOML_TEXT: &str = include_str!("../proxima-mil.toml");
    const GENERATED_TEXT: &str = include_str!(concat!(env!("OUT_DIR"), "/proxima_mil_sized.rs"));
    const BUILD_SCRIPT: &str = include_str!("../build.rs");
    // sized.rs holds the axes; wire.rs and weights.rs hold protobuf and blob byte layouts, whose numbers are format facts that happen to equal an axis value
    const FORMAT_LAYOUT_SOURCES: [&str; 3] = ["sized.rs", "wire.rs", "weights.rs"];
    const AXES: [(&str, &str); 6] = [
        ("ane", "graph_ceiling_mb"),
        ("shape", "pad_multiple"),
        ("shape", "max_sequence"),
        ("package", "specification_version"),
        ("package", "opset"),
        ("units", "default_compute_units"),
    ];

    fn parsed_toml() -> toml::Table {
        match TOML_TEXT.parse::<toml::Table>() {
            Ok(table) => table,
            Err(source) => panic!("proxima-mil.toml must parse: {source}"),
        }
    }

    fn count_token(haystack: &str, needle: &str) -> usize {
        let is_word = |character: char| character.is_alphanumeric() || character == '_';
        haystack
            .match_indices(needle)
            .filter(|(start, _)| !haystack[..*start].chars().next_back().is_some_and(is_word))
            .filter(|(start, _)| {
                !haystack[start + needle.len()..]
                    .chars()
                    .next()
                    .is_some_and(is_word)
            })
            .count()
    }

    fn sibling_sources() -> Vec<(PathBuf, String)> {
        let source_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let entries = match fs::read_dir(&source_dir) {
            Ok(entries) => entries,
            Err(source) => panic!("read {}: {source}", source_dir.display()),
        };
        entries
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|extension| extension == "rs"))
            .filter(|path| {
                path.file_name().is_some_and(|name| {
                    !FORMAT_LAYOUT_SOURCES.contains(&name.to_string_lossy().as_ref())
                })
            })
            .filter_map(|path| fs::read_to_string(&path).ok().map(|text| (path, text)))
            .collect()
    }

    #[test]
    fn mil_sized_toml_holds_6_keys_in_4_sections_each_with_a_doc_line() {
        let table = parsed_toml();
        let lines: Vec<&str> = TOML_TEXT.lines().collect();
        let key_lines: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter(|(_, line)| line.contains(" = ") && !line.starts_with('#'))
            .map(|(index, _)| index)
            .collect();

        let key_count: usize = table
            .values()
            .filter_map(Value::as_table)
            .map(|section| section.len())
            .sum();

        assert_eq!(
            table.len(),
            4,
            "sections: {:?}",
            table.keys().collect::<Vec<_>>()
        );
        assert_eq!(key_count, 6);
        assert_eq!(key_lines.len(), 6);
        assert!(
            key_lines
                .iter()
                .all(|index| lines[index - 1].starts_with('#'))
        );
        assert!(
            AXES.iter()
                .all(|(section, key)| table[*section][*key].is_integer()
                    || table[*section][*key].is_str())
        );
    }

    #[test]
    fn mil_sized_generated_module_holds_6_consts_equal_to_the_toml() {
        let table = parsed_toml();
        let integer = |section: &str, key: &str| {
            table[section][key]
                .as_integer()
                .and_then(|value| usize::try_from(value).ok())
        };
        let text = |section: &str, key: &str| table[section][key].as_str().map(ToString::to_string);

        let generated_consts = GENERATED_TEXT
            .lines()
            .filter(|line| line.starts_with("pub const "))
            .count();

        assert_eq!(generated_consts, 6);
        assert_eq!(Some(GRAPH_CEILING_MB), integer("ane", "graph_ceiling_mb"));
        assert_eq!(Some(PAD_MULTIPLE), integer("shape", "pad_multiple"));
        assert_eq!(Some(MAX_SEQUENCE), integer("shape", "max_sequence"));
        assert_eq!(
            Some(SPECIFICATION_VERSION),
            integer("package", "specification_version")
        );
        assert_eq!(Some(OPSET.to_string()), text("package", "opset"));
        assert_eq!(
            Some(DEFAULT_COMPUTE_UNITS.to_string()),
            text("units", "default_compute_units")
        );
    }

    #[test]
    fn mil_sized_env_override_names_follow_proxima_mil_section_key_for_all_6() {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let source = match SizingSource::load(manifest_dir, "proxima-mil.toml", "PROXIMA_MIL") {
            Ok(source) => source,
            Err(error) => panic!("sizing source must load: {error}"),
        };

        for (section, key) in AXES {
            let name = format!(
                "PROXIMA_MIL_{}_{}",
                section.to_uppercase(),
                key.to_uppercase()
            );
            let resolved = temp_env::with_var(&name, Some("4096"), || {
                source.resolve_str(section, key).ok()
            });
            assert_eq!(
                resolved.as_deref(),
                Some("4096"),
                "{name} must override {section}.{key}"
            );
            assert!(
                BUILD_SCRIPT.contains(&format!("\"{section}\", \"{key}\"")),
                "build.rs must resolve {section}.{key}"
            );
        }
        assert!(BUILD_SCRIPT.contains("\"PROXIMA_MIL\""));
    }

    #[test]
    fn mil_sized_source_holds_no_literal_copy_of_an_axis() {
        let axis_literals = [
            GRAPH_CEILING_MB.to_string(),
            PAD_MULTIPLE.to_string(),
            MAX_SEQUENCE.to_string(),
            format!("\"{OPSET}\""),
        ];

        let copies: usize = sibling_sources()
            .iter()
            .map(|(_, text)| {
                axis_literals
                    .iter()
                    .map(|literal| count_token(text, literal))
                    .sum::<usize>()
            })
            .sum();

        assert_eq!(copies, 0);
    }

    #[test]
    fn mil_sized_literal_counter_control_counts_a_planted_opset_literal() {
        let planted = "let opset = \"CoreML6\";";

        assert_eq!(count_token(planted, "\"CoreML6\""), 1);
        assert_eq!(count_token("let opset = \"CoreML60\";", "\"CoreML6\""), 0);
    }
}
