# Proxima BF8 scalar contract

Proxima BF8 is a scalar E5M2 encoding. Each value is one byte with sign `s` in
bit 7, exponent `e` in bits 6..2, and mantissa `m` in bits 1..0. The exponent
bias is 15.

For `1 <= e <= 30`, decode as `(-1)^s * 2^(e - 15) * (1 + m / 4)`. For `e=0`
and nonzero `m`, decode as `(-1)^s * 2^-14 * (m / 4)`. `e=0,m=0` is signed
zero. `e=31,m=0` is signed infinity; `e=31,m!=0` is NaN.

Encoding from F32 uses round-to-nearest, ties-to-even. Values below the normal
range round in units of `2^-16`; normal values round the three significant
binary digits to even. A rounded carry advances the exponent. Overflow rounds
to infinity, including the exact midpoint above the largest finite value.
Zero's sign is preserved. NaN encodes to quiet NaN mantissa `2`, retaining the
input sign; decoding accepts every NaN mantissa payload.

This is a Proxima format decision. “BF8” here names neither a block format nor
an external standard. There is no shared exponent or per-block scale.
