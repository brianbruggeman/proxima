# Evidence kept out of git

Three result files are not committed because the local content gate refuses
them: each record embeds a host process listing that names a private tool's
binary. They exist in the source tree at
`/Users/brianbruggeman/repos/slot-0/proxima` under the same relative path:

- `proxima-tensor/specs/decode-performance-lab/results/runs.jsonl`
- `proxima-tensor/specs/decode-performance-lab/results-ngram-1x3/runs.jsonl`
- `proxima-tensor/specs/decode-performance-lab/results-ngram-3x3/runs.jsonl`

The `pairs.jsonl` and `summary.json` files beside them are committed.
