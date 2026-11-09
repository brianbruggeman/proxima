# proxima-numeric

The one place this workspace defines, describes or converts a number.

`FloatLayout` is a scalar (`llvm::fltSemantics`: exponent width, mantissa width, special-value
policy, bias, sign). `BlockLayout` is a run of codes under a shared scale with its exact planes.
`LevelFormat` is a run of codes whose value is an explicit level set. `NumericFormat` unites the
three, `decode` turns any of them into numbers with zero allocation, and `encode` is the inverse.

`Numeric` is the value: a format together with the wide numbers it holds, so one type spans a
scalar, a 32-to-256-element block and a level-set codebook. `Numeric::decode` builds it from
packed bytes, `Numeric::encode` writes it back in its own format, and `Numeric::to` writes it into
any other format under a rounding mode and an overflow policy the caller names. Both directions,
every pair of formats, no allocation.

Nothing outside this crate declares a numeric type. A crate that needs one depends on this one.
