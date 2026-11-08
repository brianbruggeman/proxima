# proxima-numeric

The one place this workspace defines, describes or converts a number.

`FloatLayout` is a scalar (`llvm::fltSemantics`: exponent width, mantissa width, special-value
policy, bias, sign). `BlockLayout` is a run of codes under a shared scale with its exact planes.
`LevelFormat` is a run of codes whose value is an explicit level set. `NumericFormat` unites the
three, `decode` turns any of them into numbers with zero allocation, and `convert` narrows a value
into a format with a rounding mode and an overflow policy the caller names.

Nothing outside this crate declares a numeric type. A crate that needs one depends on this one.
