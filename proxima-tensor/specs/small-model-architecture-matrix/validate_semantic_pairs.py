#!/usr/bin/env python3
"""Check Card 7's pinned operator witnesses and wrong substitutions."""

import gzip
import hashlib
import json
import math
from pathlib import Path


SPEC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SPEC_DIR.parents[2]
FIXTURE = SPEC_DIR / "fixtures" / "semantic-pairs.json"
CONFIG_DIR = SPEC_DIR / "fixtures" / "semantic-pair-configs"
SOURCE_DIR = SPEC_DIR / "fixtures" / "upstream-source"
PAIR_IDS = {
    "lfm_text_vl_conv",
    "minicpm_granite_gqa",
    "lfm_conv_granite_mamba",
    "sala_lightning_nemotron_mamba",
    "sala_sparse_minicpm_dense",
    "nanbeige_loop_llama_single",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected):
    return math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)


def config_value(config, path):
    value = config
    for component in path.split("."):
        value = value[int(component)] if isinstance(value, list) else value[component]
    return value


def upstream_bytes(source_hash, snapshots):
    if source_hash not in snapshots:
        compressed = (SOURCE_DIR / f"{source_hash}.py.gz").read_bytes()
        snapshots[source_hash] = gzip.decompress(compressed)
    return snapshots[source_hash]


def validate_sources(sources):
    inventory = {
        item["case"]: item
        for item in map(json.loads, (SPEC_DIR / "inventory.jsonl").read_text().splitlines())
    }
    snapshots = {}
    for source_id, source in sources.items():
        checkpoint = inventory[source["case"]]
        require(source["config_revision"] == checkpoint["revision"], f"{source_id}: revision")
        require(source["config_url"] == checkpoint["config_url"], f"{source_id}: config URL")
        config_path = CONFIG_DIR / source["config_file"]
        payload = config_path.read_bytes()
        require(hashlib.sha256(payload).hexdigest() == source["config_sha256"], f"{source_id}: config hash")
        config = json.loads(payload)
        for key, expected in source["config_checks"].items():
            require(config_value(config, key) == expected, f"{source_id}: config {key}")

        operator = source["operator"]
        require(operator["url"].endswith(f'#L{operator["line"]}'), f"{source_id}: operator line")
        require(len(operator["sha256"]) == 64 and operator["line_text"], f"{source_id}: operator source")
        source_bytes = upstream_bytes(operator["sha256"], snapshots)
        require(hashlib.sha256(source_bytes).hexdigest() == operator["sha256"], f"{source_id}: operator hash")
        source_lines = source_bytes.decode("utf-8").splitlines()
        require(source_lines[operator["line"] - 1].strip() == operator["line_text"], f"{source_id}: operator line bytes")
        require(operator["supporting_lines"], f"{source_id}: operator context")
        require(all(0 < line <= len(source_lines) for line in operator["supporting_lines"]), f"{source_id}: operator range")

        if "huggingface.co" in operator["url"]:
            require(source["config_revision"] in operator["url"], f"{source_id}: operator revision")
        else:
            version = config.get("transformers_version")
            require(version and f"/v{version}/" in operator["url"], f"{source_id}: operator version")

        if dependency := source.get("dependency_operator"):
            dependency_bytes = upstream_bytes(dependency["sha256"], snapshots)
            require(hashlib.sha256(dependency_bytes).hexdigest() == dependency["sha256"], f"{source_id}: dependency hash")
            dependency_lines = dependency_bytes.decode("utf-8").splitlines()
            require("exp(b_g_gamma)" in dependency_lines[110], f"{source_id}: decay equation")
            require("b_h +=" in dependency_lines[124], f"{source_id}: state equation")

        local = source["local"]
        path = REPO_ROOT / local["path"]
        lines = path.read_text().splitlines()
        require(local["line"] > 0 and local["line"] <= len(lines), f"{source_id}: local line")
        require(local["needle"] in lines[local["line"] - 1], f"{source_id}: local contract")
        require(local["role"] in {"matching_core", "rejected_substitute"}, f"{source_id}: local role")

    require(len(sources) == 10, "source count")


def evaluate(vector):
    operation = vector["op"]
    model_input = vector["input"]
    state = vector["state_in"]

    if operation == "short_conv":
        require(state["kind"] == "conv_taps", "state_kind_mismatch")
        old = state["values"]
        taps = model_input["taps"]
        require(len(old) == len(taps), "conv_state_shape_mismatch")
        updated = old[1:] + [model_input["b"] * model_input["x"]]
        output = model_input["c"] * sum(weight * sample for weight, sample in zip(taps, updated))
        output *= model_input["out_proj"]
        return {"kind": "conv_taps", "values": updated}, [output]

    if operation in {"gqa_softmax_core", "selected_softmax", "dense_softmax"}:
        require(state["kind"] in {"kv", "kv_selected"}, "state_kind_mismatch")
        scores = model_input["qk_dot"]
        values = model_input["values"]
        require(len(scores) == len(values) == state["key_rows"], "kv_shape_mismatch")
        if operation == "selected_softmax":
            require(model_input["context_tokens"] >= 8192, "sparse_branch_not_selected")
            selected = model_input["selected_indices"]
            require(selected and len(set(selected)) == len(selected), "invalid_selection")
            require(all(0 <= index < len(scores) for index in selected), "invalid_selection")
        else:
            selected = range(len(scores))
        logits = [scores[index] * model_input["scale"] for index in selected]
        maximum = max(logits)
        weights = [math.exp(logit - maximum) for logit in logits]
        output = sum(weight * values[index] for weight, index in zip(weights, selected)) / sum(weights)
        return state, [output]

    if operation == "mamba2_step":
        require(state["kind"] == "selective_ssm", "state_kind_mismatch")
        require(len(state["values"]) == 1, "ssm_state_shape_mismatch")
        old = state["values"][0]
        updated = model_input["dA"] * old + model_input["dt"] * model_input["B"] * model_input["x"]
        output = model_input["C"] * updated + model_input["D"] * model_input["x"]
        return {"kind": "selective_ssm", "values": [updated]}, [output]

    if operation == "simple_gla_step":
        require(state["kind"] == "gla_matrix", "state_kind_mismatch")
        require(len(state["values"]) == 1, "gla_state_shape_mismatch")
        require(close(math.exp(model_input["g_gamma"]), model_input["decay_factor"]), "gla_decay_mismatch")
        updated = state["values"][0] * model_input["decay_factor"] + model_input["k"] * model_input["v"]
        output = model_input["q"] * model_input["scale"] * updated
        return {"kind": "gla_matrix", "values": [updated]}, [output]

    if operation == "loop_stack":
        require(state["kind"] in {"loop_cache", "single_cache"}, "state_kind_mismatch")
        require(model_input["norm"] == "identity", "unsupported_toy_norm")
        expected_slots = [f"{index}:0" for index in range(model_input["loops"])]
        require(not state["slots"] or state["slots"] == expected_slots, "cache_identity_mismatch")
        value = model_input["x"]
        slots = []
        for loop_index in range(model_input["loops"]):
            value = model_input["layer_multiplier"] * value + model_input["layer_bias"]
            slots.append(f"{loop_index}:0")
        return {"kind": state["kind"], "slots": slots}, [value]

    raise ValueError(f"unknown operator {operation}")


def validate_vector(vector, pair_id, side):
    state_out, output = evaluate(vector)
    require(state_out == vector["state_out"], f"{pair_id}/{side}: state output")
    require(len(output) == len(vector["output"]), f"{pair_id}/{side}: output shape")
    require(all(close(actual, expected) for actual, expected in zip(output, vector["output"])), f"{pair_id}/{side}: output")
    return output


def reject_control(pair):
    control = pair["control"]
    left = pair["left"]
    right = pair["right"]
    kind = control["kind"]

    if kind in {"wrong_state", "feed_left_state_to_right"}:
        candidate = json.loads(json.dumps(left if kind == "wrong_state" else right))
        candidate["state_in"] = {"kind": control["replace_kind"] if kind == "wrong_state" else left["state_in"]["kind"], "values": [2]}
        try:
            evaluate(candidate)
        except ValueError as error:
            require(str(error) == control["expected"], f"{pair['id']}: wrong rejection")
            return
        raise ValueError(f"{pair['id']}: wrong state accepted")

    if kind == "omit_scale_and_head_dim":
        omitted_scale = json.loads(json.dumps(right))
        omitted_scale["input"]["scale"] = 1
        omitted_head = json.loads(json.dumps(left))
        omitted_head["input"]["scale"] = (1536 / 16) ** -0.5
        wrong_scale = evaluate(omitted_scale)[1][0]
        wrong_head = evaluate(omitted_head)[1][0]
        require(not close(wrong_scale, right["output"][0]), f"{pair['id']}: scale omission accepted")
        require(not close(wrong_head, left["output"][0]), f"{pair['id']}: head dimension omission accepted")
        return

    if kind == "exchange_recurrent_states":
        for original, exchanged in ((left, right), (right, left)):
            candidate = json.loads(json.dumps(original))
            candidate["state_in"] = exchanged["state_in"]
            try:
                evaluate(candidate)
            except ValueError as error:
                require(str(error) == control["expected"], f"{pair['id']}: wrong rejection")
                continue
            raise ValueError(f"{pair['id']}: exchanged state accepted")
        return

    if kind == "drop_block_selection":
        candidate = json.loads(json.dumps(left))
        candidate["op"] = "dense_softmax"
        require(not close(evaluate(candidate)[1][0], left["output"][0]), f"{pair['id']}: dropped selection accepted")
        return

    if kind == "single_pass_with_two_loop_cache":
        candidate = json.loads(json.dumps(right))
        candidate["state_in"] = left["state_out"]
        try:
            evaluate(candidate)
        except ValueError as error:
            require(str(error) == control["expected"], f"{pair['id']}: wrong rejection")
            return
        raise ValueError(f"{pair['id']}: two-loop cache accepted by one pass")

    raise ValueError(f"{pair['id']}: unknown control {kind}")


def main():
    data = json.loads(FIXTURE.read_text())
    require(data["version"] == 1 and "not full checkpoint execution" in data["scope"], "fixture scope")
    validate_sources(data["sources"])
    pairs = data["pairs"]
    require(len(pairs) == 6 and {pair["id"] for pair in pairs} == PAIR_IDS, "candidate pair set")
    reuse_pairs = 0
    distinct_pairs = 0
    wrong_substitutions_rejected = 0
    for pair in pairs:
        require(pair.get("status") == "candidate", f"{pair['id']}: status")
        require(pair["left_source"] in data["sources"] and pair["right_source"] in data["sources"], f"{pair['id']}: source")
        left_output = validate_vector(pair["left"], pair["id"], "left")
        right_output = validate_vector(pair["right"], pair["id"], "right")
        if pair["relationship"] == "reuse":
            require(pair.get("shared_contract") and not pair.get("separate_contract"), f"{pair['id']}: reuse contract")
            require(close(left_output[0], right_output[0]), f"{pair['id']}: common core differs")
            reuse_pairs += 1
        elif pair["relationship"] == "distinct":
            require(pair.get("separate_contract") and not pair.get("shared_contract"), f"{pair['id']}: distinct contract")
            distinct_pairs += 1
        else:
            raise ValueError(f"{pair['id']}: relationship")
        reject_control(pair)
        if pair["relationship"] == "distinct":
            wrong_substitutions_rejected += 1
    require(reuse_pairs == 2 and distinct_pairs == 4 and wrong_substitutions_rejected == 4, "pair counts")
    print(f"candidate_pairs={len(pairs)} reuse_pairs={reuse_pairs} distinct_pairs={distinct_pairs} wrong_substitutions_rejected={wrong_substitutions_rejected} missing_evidence=0")


if __name__ == "__main__":
    main()
