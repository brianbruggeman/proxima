pub(crate) fn native_roots_default(target_os: &str) -> bool {
    matches!(target_os, "windows")
}

#[cfg(test)]
mod tests {
    use super::native_roots_default;

    #[test]
    fn windows_selects_native_roots() {
        assert!(native_roots_default("windows"));
    }

    #[test]
    fn linux_selects_mozilla_roots() {
        assert!(!native_roots_default("linux"));
    }

    #[test]
    fn macos_selects_mozilla_roots() {
        assert!(!native_roots_default("macos"));
    }
}
