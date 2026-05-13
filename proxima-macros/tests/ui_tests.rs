// trybuild tests: verify that known misuses produce the expected compiler errors.

#[test]
fn compile_fail_cases() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/ui/*.rs");
}
