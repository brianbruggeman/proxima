//! `fnmatch`-compatible glob matching.
//!
//! The area map was written against Python's `fnmatch.fnmatch`, whose `*`
//! crosses `/` (it compiles to `.*`, not a path-segment match) and whose `?`
//! and `[seq]` forms follow the shell. `globset`/`glob` are path-segment aware
//! by default, so using them would silently change which areas a change trips.
//! This is the reference semantics, kept deliberately small.

/// Whether `name` matches `pattern` with shell glob rules: `*` any run of
/// characters including `/`, `?` exactly one character, `[seq]` one character
/// from the set (ranges allowed, leading `!` or `^` negates).
pub(crate) fn matches(pattern: &str, name: &str) -> bool {
    let pattern: Vec<char> = pattern.chars().collect();
    let name: Vec<char> = name.chars().collect();
    match_from(&pattern, &name)
}

fn match_from(pattern: &[char], name: &[char]) -> bool {
    let Some(&head) = pattern.first() else {
        return name.is_empty();
    };
    match head {
        '*' => {
            // collapse a run of stars, then try every split point
            let rest = pattern.iter().skip_while(|&&c| c == '*').count();
            let rest = &pattern[pattern.len() - rest..];
            if rest.is_empty() {
                return true;
            }
            (0..=name.len()).any(|consumed| match_from(rest, &name[consumed..]))
        }
        '?' => !name.is_empty() && match_from(&pattern[1..], &name[1..]),
        '[' => match class(pattern, name) {
            Some((matched, consumed)) => matched && match_from(&pattern[consumed..], &name[1..]),
            None => !name.is_empty() && head == name[0] && match_from(&pattern[1..], &name[1..]),
        },
        literal => !name.is_empty() && literal == name[0] && match_from(&pattern[1..], &name[1..]),
    }
}

/// Parse `[...]` at the start of `pattern`. Returns whether `name`'s first
/// character is in the class and how many pattern characters the class spans,
/// or `None` when the bracket never closes (a literal `[` under shell rules).
fn class(pattern: &[char], name: &[char]) -> Option<(bool, usize)> {
    let first = *name.first()?;
    let mut index = 1;
    let negated = matches!(pattern.get(index), Some('!') | Some('^'));
    if negated {
        index += 1;
    }
    let mut matched = false;
    let mut saw_close = false;
    while index < pattern.len() {
        match pattern[index] {
            ']' => {
                saw_close = true;
                index += 1;
                break;
            }
            low => {
                let mut high = low;
                index += 1;
                if pattern.get(index) == Some(&'-')
                    && let Some(&next) = pattern.get(index + 1)
                    && next != ']'
                {
                    high = next;
                    index += 2;
                }
                if low <= first && first <= high {
                    matched = true;
                }
            }
        }
    }
    saw_close.then_some((matched != negated, index))
}

#[cfg(test)]
mod tests {
    use super::matches;

    #[test]
    fn star_crosses_directory_separators_like_fnmatch() {
        assert!(matches("docs/tutorials/**", "docs/tutorials/deep/page.md"));
        assert!(matches("proxima-dns/**", "proxima-dns/src/lib.rs"));
        assert!(matches("*.rs", "a/b/c.rs"));
    }

    #[test]
    fn exact_paths_and_prefixes() {
        assert!(matches("Cargo.lock", "Cargo.lock"));
        assert!(!matches("Cargo.lock", "Cargo.lock.bak"));
        assert!(matches("rust-toolchain*", "rust-toolchain.toml"));
    }

    #[test]
    fn question_mark_and_classes() {
        assert!(matches("a?c", "abc"));
        assert!(!matches("a?c", "ac"));
        assert!(matches("file[0-9].rs", "file7.rs"));
        assert!(!matches("file[0-9].rs", "filex.rs"));
        assert!(matches("file[!0-9].rs", "filex.rs"));
    }

    #[test]
    fn unterminated_bracket_is_literal() {
        assert!(matches("a[bc", "a[bc"));
        assert!(!matches("a[bc", "abc"));
    }
}
