#!/usr/bin/env python3
"""Check captured Granite attention replay records and reject corrupted copies."""

import argparse
import copy
import json
import math
import pathlib
import re
import sys


DEFAULT_SELECTED_ARM = "shared_k"
ROUND_COUNT = 20
CHECKPOINT_SHA256 = "cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80"
CHECKPOINT_BYTES = 1422239776
SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")
RESOURCE_FIELDS = (
    "wall_ms",
    "cpu_pct",
    "rss_peak_mb",
    "footprint_mb",
    "gpu_alloc_mb",
    "load_before",
    "load_after",
)


class ReportError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportError(message)


def object_field(record: dict, name: str, where: str) -> dict:
    value = record.get(name)
    require(isinstance(value, dict), f"{where}.{name} must be an object")
    return value


def array_field(record: dict, name: str, where: str) -> list:
    value = record.get(name)
    require(isinstance(value, list), f"{where}.{name} must be an array")
    return value


def text_field(record: dict, name: str, where: str) -> str:
    value = record.get(name)
    require(isinstance(value, str) and bool(value), f"{where}.{name} must be nonempty text")
    return value


def integer_field(record: dict, name: str, where: str, minimum: int = 0) -> int:
    value = record.get(name)
    require(type(value) is int and value >= minimum, f"{where}.{name} must be >= {minimum}")
    return value


def finite_field(record: dict, name: str, where: str, positive: bool = False) -> float:
    value = record.get(name)
    require(type(value) in (int, float), f"{where}.{name} must be numeric")
    number = float(value)
    require(math.isfinite(number), f"{where}.{name} must be finite")
    require(not positive or number > 0, f"{where}.{name} must be positive")
    return number


def sha256_field(record: dict, name: str, where: str) -> str:
    value = text_field(record, name, where)
    require(SHA256_HEX.fullmatch(value) is not None, f"{where}.{name} must be lowercase SHA256")
    return value


def near(actual: float, expected: float, tolerance: float, where: str) -> None:
    require(abs(actual - expected) <= tolerance, f"{where}: stored {actual} differs from raw {expected}")


def validate_provenance(report: dict) -> None:
    require(type(report.get("version")) is int and report["version"] == 1, "version must be 1")
    require(text_field(report, "model", "report") == "Granite 3.1 1B A400M Instruct", "wrong model")
    checkpoint = object_field(report, "checkpoint", "report")
    checksum = sha256_field(checkpoint, "sha256", "checkpoint")
    require(checksum == CHECKPOINT_SHA256, "wrong checkpoint SHA256")
    checkpoint_path = text_field(checkpoint, "path", "checkpoint")
    require(checkpoint_path.endswith(f"sha256-{checksum}"), "checkpoint path and SHA disagree")
    require(integer_field(checkpoint, "bytes", "checkpoint", 1) == CHECKPOINT_BYTES, "wrong checkpoint byte length")
    require(text_field(checkpoint, "architecture", "checkpoint") == "granitemoe", "wrong GGUF architecture")
    require(text_field(checkpoint, "model_name", "checkpoint") == "Granite 3.1 1b A400M Instruct", "wrong GGUF model name")
    require(integer_field(checkpoint, "gguf_file_type", "checkpoint") == 7, "wrong GGUF file type")
    require(text_field(checkpoint, "weight_quant", "checkpoint") == "Q8_0", "wrong weight quantization")

    host = object_field(report, "host", "report")
    text_field(host, "arch", "host")
    text_field(host, "os", "host")
    require("hostname" in host, "host.hostname is missing")
    require(host["hostname"] is None or (isinstance(host["hostname"], str) and bool(host["hostname"])), "host.hostname must be nonempty text or null")
    require("device_description" in report, "device_description is missing")
    device_description = report["device_description"]
    require(
        device_description is None or (isinstance(device_description, str) and bool(device_description)),
        "device_description must be nonempty text or null",
    )
    serving = object_field(report, "serving", "report")
    require(serving.get("kv_cache_key") == "F32", "serving key cache must be F32")
    require(serving.get("kv_cache_value") == "F32", "serving value cache must be F32")
    require(serving.get("prompt_cache") == "off", "serving prompt cache must be off")
    require(serving.get("flash_attention") is False, "serving flash attention must be off")
    require(integer_field(serving, "generated_tokens", "serving") == 1, "serving token count must be 1")
    require(integer_field(serving, "ubatch_size", "serving") == 0, "serving ubatch must be 0")
    require(integer_field(serving, "reasoning_budget", "serving") == 0, "serving reasoning budget must be 0")
    require(type(serving.get("gpu_layers")) is int and serving["gpu_layers"] == -1, "serving GPU layers must be all")


def validate_grid(arm: dict, where: str) -> None:
    grid = object_field(arm, "grid", where)
    integer_field(grid, "threads", f"{where}.grid", 1)
    integer_field(grid, "depth", f"{where}.grid", 1)
    width = grid.get("threadgroup_width")
    require(width is None or (type(width) is int and width > 0), f"{where}.grid width must be positive or null")
    grid2d = grid.get("grid2d")
    require(grid2d is None or isinstance(grid2d, dict), f"{where}.grid2d must be an object or null")
    if grid2d is not None:
        require(grid2d.get("form") in ("tile_coordinates", "flat_threadgroup_index"), f"{where}.grid2d form")
        for name in (
            "threadgroups_x",
            "threadgroups_y",
            "threads_per_threadgroup_x",
            "threads_per_threadgroup_y",
        ):
            integer_field(grid2d, name, f"{where}.grid2d", 1)
        integer_field(grid2d, "threadgroup_bytes", f"{where}.grid2d")


def validate_summary(arm: dict, samples: list[float], where: str) -> None:
    summary = object_field(arm, "summary", where)
    require(integer_field(summary, "count", f"{where}.summary") == ROUND_COUNT, f"{where} summary count")
    sorted_samples = sorted(samples)
    mean = sum(samples) / ROUND_COUNT
    variance = sum((sample - mean) ** 2 for sample in samples) / ROUND_COUNT
    expected = {
        "min_gpu_ns": sorted_samples[0],
        "max_gpu_ns": sorted_samples[-1],
        "mean_gpu_ns": mean,
        "p50_gpu_ns": sorted_samples[9],
        "p90_gpu_ns": sorted_samples[17],
        "p99_gpu_ns": sorted_samples[19],
    }
    for name, value in expected.items():
        near(finite_field(summary, name, f"{where}.summary"), value, 1e-6, f"{where}.summary.{name}")
    near(
        finite_field(summary, "cov_percent", f"{where}.summary"),
        100 * math.sqrt(variance) / mean,
        1e-8,
        f"{where}.summary.cov_percent",
    )


def validate_arm(arm: dict, name: str, selected_name: str, shape_index: int, actual_prompt_tokens: int, rounds: list[dict]) -> list[float]:
    where = f"shapes[{shape_index}].arms[{name}]"
    require(text_field(arm, "arm", where) == name, f"{where} label mismatch")
    integer_field(arm, "node", where)
    extents = array_field(arm, "extents", where)
    require(len(extents) == 4 and all(type(value) is int and value > 0 for value in extents), f"{where} extents")
    require(extents[0] > 1 and extents[1:] == [8, 2, 64], f"{where} is not multi-row Granite attention")
    require(extents[0] == actual_prompt_tokens, f"{where} query extent differs from actual prompt tokens")
    entry = text_field(arm, "entry", where)
    sha256_field(arm, "source_sha256", where)
    validate_grid(arm, where)
    if selected_name.startswith("simdgroups"):
        selected_groups = int(selected_name.removeprefix("simdgroups"))
        expected_width = selected_groups * 32 if name == selected_name else 64
        require(arm["grid"].get("threadgroup_width") == expected_width, f"{where} threadgroup width must be {expected_width}")
        if name == selected_name:
            require(entry.endswith(f"_{selected_name}"), f"{where} entry lacks _{selected_name} suffix")
        else:
            require(not entry.endswith(f"_{selected_name}"), f"{where} legacy entry has _{selected_name} suffix")
    pipeline = object_field(arm, "pipeline_resources", where)
    integer_field(pipeline, "tg_static_bytes", f"{where}.pipeline_resources")
    integer_field(pipeline, "max_threads", f"{where}.pipeline_resources", 1)
    integer_field(pipeline, "exec_width", f"{where}.pipeline_resources", 1)
    integer_field(arm, "bound_bytes", where, 1)
    require(arm.get("fault_binding_present") is False, f"{where} contains a fault binding")
    output_bytes = integer_field(arm, "output_bytes", where, 1)
    require(output_bytes == math.prod(extents) * 4, f"{where} output span is incomplete")
    sha256_field(arm, "output_sha256", where)
    generated_ids = array_field(arm, "generated_ids", where)
    require(generated_ids and all(type(value) is int and 0 <= value <= 0xFFFFFFFF for value in generated_ids), f"{where} generated IDs")
    require(integer_field(arm, "timing_attempts", where) == ROUND_COUNT, f"{where} timing attempts")
    require(integer_field(arm, "resource_replay_attempts", where) == 5, f"{where} resource attempts")
    require(integer_field(arm, "replay_errors", where) == 0, f"{where} replay errors")
    resource = text_field(arm, "resource", where)
    resource_tokens = resource.split()
    require(len(resource_tokens) >= 2 and resource_tokens[:2] == ["cell", f"label={name}"], f"{where} resource label mismatch")
    required_tokens = []
    for token in resource_tokens[2:]:
        field, separator, value = token.partition("=")
        if field in RESOURCE_FIELDS:
            require(separator == "=" and bool(value), f"{where} empty resource {field}")
            required_tokens.append(field)
    require(tuple(required_tokens) == RESOURCE_FIELDS, f"{where} resource fields must occur once in order")

    samples = array_field(arm, "samples", where)
    require(len(samples) == ROUND_COUNT, f"{where} must have {ROUND_COUNT} samples")
    values = []
    for round_number, sample in enumerate(samples):
        require(isinstance(sample, dict), f"{where}.samples[{round_number}] must be an object")
        sample_where = f"{where}.samples[{round_number}]"
        require(integer_field(sample, "round", sample_where) == round_number, f"{sample_where} round order")
        expected_position = rounds[round_number]["arm_order"].index(name)
        require(integer_field(sample, "position", sample_where) == expected_position, f"{sample_where} position")
        values.append(finite_field(sample, "gpu_ns", sample_where, positive=True))
    validate_summary(arm, values, where)
    return values


def validate_shape(shape: dict, shape_index: int, selected_name: str) -> tuple[int, int, int]:
    where = f"shapes[{shape_index}]"
    nominal = integer_field(shape, "nominal_prompt_tokens", where, 1)
    actual = integer_field(shape, "actual_prompt_tokens", where, nominal)
    sha256_field(shape, "prompt_ids_sha256", where)
    require(shape.get("output_equal") is True, f"{where} output equality is false")
    require(shape.get("ids_equal") is True, f"{where} ID equality is false")
    rounds = array_field(shape, "rounds", where)
    require(len(rounds) == ROUND_COUNT, f"{where} must have {ROUND_COUNT} rounds")
    for round_number, round_record in enumerate(rounds):
        require(isinstance(round_record, dict), f"{where}.rounds[{round_number}] must be an object")
        round_where = f"{where}.rounds[{round_number}]"
        require(integer_field(round_record, "round", round_where) == round_number, f"{round_where} number")
        arm_names = ["legacy", selected_name]
        expected_order = arm_names if round_number % 2 == 0 else list(reversed(arm_names))
        require(round_record.get("arm_order") == expected_order, f"{round_where} arm order")

    arms = array_field(shape, "arms", where)
    require(len(arms) == 2 and all(isinstance(arm, dict) for arm in arms), f"{where} must have two arm objects")
    by_name = {arm.get("arm"): arm for arm in arms}
    require(set(by_name) == {"legacy", selected_name}, f"{where} must have one legacy and one {selected_name} arm")
    legacy = by_name["legacy"]
    selected = by_name[selected_name]
    legacy_values = validate_arm(legacy, "legacy", selected_name, shape_index, actual, rounds)
    selected_values = validate_arm(selected, selected_name, selected_name, shape_index, actual, rounds)
    require(legacy["node"] == selected["node"] and legacy["extents"] == selected["extents"], f"{where} dispatch identity mismatch")
    require(legacy["entry"] != selected["entry"], f"{where} source entry was not selected")
    require(legacy["source_sha256"] != selected["source_sha256"], f"{where} source SHA was not selected")
    require(legacy["output_bytes"] == selected["output_bytes"], f"{where} output byte count mismatch")
    require(legacy["output_sha256"] == selected["output_sha256"], f"{where} output hash mismatch")
    require(legacy["generated_ids"] == selected["generated_ids"], f"{where} generated IDs differ")

    positive = negative = 0
    for round_number, round_record in enumerate(rounds):
        delta = selected_values[round_number] - legacy_values[round_number]
        near(
            finite_field(round_record, "selected_minus_legacy_ns", f"{where}.rounds[{round_number}]"),
            delta,
            1e-6,
            f"{where}.rounds[{round_number}].selected_minus_legacy_ns",
        )
        positive += delta > 0
        negative += delta < 0
    return actual, positive, negative


def validate_report(report: dict, expected_shapes: int, selected_name: str = DEFAULT_SELECTED_ARM) -> tuple[int, int]:
    require(isinstance(report, dict), "report must be an object")
    validate_provenance(report)
    if selected_name.startswith("simdgroups"):
        require(report.get("selected_arm") == selected_name, "selected simdgroup count metadata mismatch")
    shapes = array_field(report, "shapes", "report")
    require(len(shapes) == expected_shapes, f"report must have {expected_shapes} shapes")
    nominal_targets = {971} if expected_shapes == 1 else {256, 971}
    require(all(isinstance(shape, dict) for shape in shapes), "each shape must be an object")
    require({shape.get("nominal_prompt_tokens") for shape in shapes} == nominal_targets, "nominal prompt targets differ")
    observed = [validate_shape(shape, index, selected_name) for index, shape in enumerate(shapes)]
    require(len({value[0] for value in observed}) == expected_shapes, "actual tokenizer counts are not distinct")
    if expected_shapes == 2:
        by_nominal = {shape["nominal_prompt_tokens"]: shape["actual_prompt_tokens"] for shape in shapes}
        require(by_nominal[256] < by_nominal[971], "short tokenizer count is not below long count")
    return sum(value[1] for value in observed), sum(value[2] for value in observed)


def reorder_resource_fields(record: dict) -> None:
    arm = record["shapes"][0]["arms"][0]
    tokens = arm["resource"].split()
    positions = {token.partition("=")[0]: index for index, token in enumerate(tokens)}
    first, second = positions["wall_ms"], positions["cpu_pct"]
    tokens[first], tokens[second] = tokens[second], tokens[first]
    arm["resource"] = " ".join(tokens)


def reject_mutations(report: dict, expected_shapes: int, selected_name: str) -> int:
    mutations = [
        lambda record: record["shapes"][0]["arms"].pop(),
        lambda record: record["shapes"][0]["arms"][0]["samples"].pop(),
        lambda record: record["shapes"][0]["arms"][1].update(
            source_sha256=record["shapes"][0]["arms"][0]["source_sha256"]
        ),
        lambda record: record["shapes"][0].update(output_equal=False),
        lambda record: record["shapes"][0]["arms"][0].update(replay_errors=1),
        lambda record: record["shapes"][0]["rounds"][0].update(arm_order=[selected_name, "legacy"]),
        lambda record: record["checkpoint"].update(
            sha256="0" * 64,
            path=record["checkpoint"]["path"].rsplit("sha256-", 1)[0] + "sha256-" + "0" * 64,
        ),
        lambda record: record["shapes"][0].update(
            actual_prompt_tokens=record["shapes"][0]["nominal_prompt_tokens"] - 1
        ),
        lambda record: record["shapes"][0]["arms"][0].update(
            resource=record["shapes"][0]["arms"][0]["resource"].replace(
                "label=legacy", "label=legacy_suffix", 1
            )
        ),
        lambda record: record["shapes"][0].update(
            actual_prompt_tokens=record["shapes"][0]["actual_prompt_tokens"] + 1
        ),
        lambda record: record.pop("device_description"),
        reorder_resource_fields,
    ]
    if selected_name.startswith("simdgroups"):
        mutations.append(
            lambda record: record["shapes"][0]["arms"][1].update(
                entry=record["shapes"][0]["arms"][1]["entry"].removesuffix(f"_{selected_name}")
            )
        )
    rejected = 0
    for mutation in mutations:
        changed = copy.deepcopy(report)
        mutation(changed)
        try:
            validate_report(changed, expected_shapes)
        except ReportError:
            rejected += 1
    return rejected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-shapes", type=int, choices=(1, 2), required=True)
    parser.add_argument("--negative-controls", action="store_true")
    parser.add_argument(
        "--selected-arm",
        choices=("shared_k", "simdgroups2", "simdgroups4", "simdgroups8"),
        default=DEFAULT_SELECTED_ARM,
    )
    parser.add_argument("report", type=pathlib.Path)
    arguments = parser.parse_args()
    try:
        original = arguments.report.read_bytes()
        report = json.loads(original)
        positive, negative = validate_report(report, arguments.expected_shapes, arguments.selected_arm)
        if arguments.negative_controls:
            expected_controls = 13 if arguments.selected_arm.startswith("simdgroups") else 12
            rejected = reject_mutations(report, arguments.expected_shapes, arguments.selected_arm)
            require(rejected == expected_controls, f"only {rejected} of {expected_controls} negative controls were rejected")
            require(arguments.report.read_bytes() == original, "negative controls changed the original report")
            print(f"negative_controls={expected_controls} rejected={rejected}")
        else:
            arms = arguments.expected_shapes * 2
            print(
                f"prompt_shapes={arguments.expected_shapes} arms={arms} "
                f"samples_per_arm={ROUND_COUNT} resource_cells={arms} "
                f"matching_pairs={arguments.expected_shapes} errors=0"
            )
            print(f"round_differences_positive={positive} negative={negative} zero={ROUND_COUNT * arguments.expected_shapes - positive - negative}")
    except (OSError, ValueError, TypeError) as error:
        print(f"report rejected: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
