#!/bin/bash
set -euo pipefail

verify_artifact() {
    local format_name="$1"
    local output_dir="$2"
    local manifest_path="$output_dir/manifest.json"
    local tensor_count
    local file_count
    local output_bytes

    tensor_count=$(jq -r '.tensors | length' "$manifest_path")
    file_count=$(find "$output_dir" -maxdepth 1 -type f -name 'tensor-*.pxw' | wc -l | tr -d ' ')
    output_bytes=$(jq -r '[.tensors[].output_bytes_written] | add' "$manifest_path")
    if [[ "$tensor_count" != "$file_count" ]]; then
        printf 'format=%s tensor_count=%s file_count=%s status=count-mismatch\n' \
            "$format_name" "$tensor_count" "$file_count"
        return 1
    fi

    jq -r '.tensors[] | [.output_file, .output_sha256, .output_bytes_written] | @tsv' \
        "$manifest_path" |
        while IFS=$'\t' read -r output_file expected_hash expected_bytes; do
            local output_path="$output_dir/$output_file"
            local actual_bytes
            local actual_hash

            actual_bytes=$(wc -c < "$output_path" | tr -d ' ')
            actual_hash=$(shasum -a 256 "$output_path")
            actual_hash=${actual_hash%% *}
            if [[ "$expected_hash" != "$actual_hash" || "$expected_bytes" != "$actual_bytes" ]]; then
                printf 'format=%s file=%s expected_bytes=%s actual_bytes=%s expected_sha256=%s actual_sha256=%s status=mismatch\n' \
                    "$format_name" "$output_file" "$expected_bytes" "$actual_bytes" \
                    "$expected_hash" "$actual_hash"
                exit 1
            fi
        done

    printf 'format=%s tensors=%s output_bytes=%s status=all-file-hashes-and-lengths-match\n' \
        "$format_name" "$tensor_count" "$output_bytes"
}

verify_artifact e5m2 /private/tmp/granite4-cold-e5m2-release
verify_artifact e3m2 /private/tmp/granite4-cold-e3m2-release
verify_artifact e2m2 /private/tmp/granite4-cold-e2m2-release
verify_artifact e5m2-tensor-scaled /private/tmp/granite4-cold-e5m2-tensor-scaled
verify_artifact e3m2-tensor-scaled /private/tmp/granite4-cold-e3m2-tensor-scaled
verify_artifact e2m2-tensor-scaled /private/tmp/granite4-cold-e2m2-tensor-scaled

source_root=/private/tmp/proxima-granite4-micro-bf16-hf
source_manifest=/private/tmp/granite4-cold-e5m2-release/manifest.json
source_file_count=$(jq -r '.source_artifacts | length' "$source_manifest")
jq -r '.source_artifacts | to_entries[] | [.key, .value] | @tsv' "$source_manifest" |
    while IFS=$'\t' read -r source_file expected_hash; do
        actual_hash=$(shasum -a 256 "$source_root/$source_file")
        actual_hash=${actual_hash%% *}
        if [[ "$expected_hash" != "$actual_hash" ]]; then
            printf 'source_file=%s expected_sha256=%s actual_sha256=%s status=mismatch\n' \
                "$source_file" "$expected_hash" "$actual_hash"
            exit 1
        fi
    done
printf 'source_artifacts=%s status=all-file-hashes-match\n' "$source_file_count"
