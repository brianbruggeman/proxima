#!/usr/bin/env python3
"""Check the BF8 golden vectors and prove the checker rejects a bad vector."""

import csv
import io
import pathlib
import re
import sys


VECTOR_PATH = pathlib.Path(__file__).with_name("bf8_vectors.csv")
SPEC_PATH = pathlib.Path(__file__).with_name("SPEC.md")
SPEC_VECTOR = re.compile(
    r"^\| ([a-z_]+) \| `0x([0-9a-fA-F]{8})` \| `0x([0-9a-fA-F]{2})` \|$"
)


def round_shift_to_even(value: int, shift: int) -> int:
    if shift <= 0:
        return value << -shift
    quotient = value >> shift
    remainder = value - (quotient << shift)
    halfway = 1 << (shift - 1)
    if remainder > halfway or (remainder == halfway and quotient & 1):
        quotient += 1
    return quotient


def encode_f32_bits(source: int) -> int:
    sign = (source >> 24) & 0x80
    exponent = (source >> 23) & 0xff
    mantissa = source & 0x7fffff

    if exponent == 0xff:
        if mantissa == 0:
            return sign | 0x7c
        return sign | 0x7e
    if exponent == 0 and mantissa == 0:
        return sign

    if exponent == 0:
        significand = mantissa
        binary_power = -149
        leading_exponent = significand.bit_length() - 1 + binary_power
    else:
        significand = (1 << 23) | mantissa
        binary_power = exponent - 150
        leading_exponent = exponent - 127

    if leading_exponent < -14:
        # E5M2 subnormals are integer multiples of 2^-16.
        units = round_shift_to_even(significand, -(binary_power + 16))
        return sign | min(units, 4)

    rounded_significand = round_shift_to_even(significand, 21)
    if rounded_significand == 8:
        leading_exponent += 1
        rounded_significand = 4
    if leading_exponent > 15:
        return sign | 0x7c
    return sign | ((leading_exponent + 15) << 2) | (rounded_significand - 4)


def read_rows(source: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(source)))


def spec_rows(source: str) -> list[tuple[str, str, str]]:
    return [
        (match.group(1), f"0x{match.group(2).lower()}", f"0x{match.group(3).lower()}")
        for line in source.splitlines()
        if (match := SPEC_VECTOR.fullmatch(line)) is not None
    ]


def validate_rows(rows: list[dict[str, str]], expected: list[tuple[str, str, str]]) -> bool:
    if len(rows) != 16 or len(expected) != 16:
        return False
    try:
        actual = [
            (row["case"], row["f32_bits"].lower(), row["bf8_bits"].lower())
            for row in rows
        ]
        if actual != expected or len({row["case"] for row in rows}) != 16:
            return False
        return all(encode_f32_bits(int(source, 16)) == int(encoded, 16) for _, source, encoded in actual)
    except (KeyError, ValueError):
        return False


def main() -> int:
    fixture = VECTOR_PATH.read_text(encoding="utf-8")
    specification = SPEC_PATH.read_text(encoding="utf-8")
    rows = read_rows(fixture)
    expected = spec_rows(specification)
    if not validate_rows(rows, expected):
        print("check 1 failed: fixture differs from specified vectors or integer E5M2 reference", file=sys.stderr)
        return 1

    corrupted_rows = [dict(row) for row in rows]
    corrupted_rows[0]["bf8_bits"] = "0x01"
    if validate_rows(corrupted_rows, expected):
        print("check 2 failed: corrupted vector was accepted", file=sys.stderr)
        return 1

    print(f"checks=2 vectors={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
