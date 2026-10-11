# bge-small fixtures

Real-model fixtures of the ONNX door (`proxima-tensor/specs/onnx-door/SPEC.md`, requirement R6).

## the model

- repository: `BAAI/bge-small-en-v1.5` on Hugging Face
- revision: `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`
- file: `onnx/model.onnx`, 133093490 bytes, producer pytorch 2.0.1
- sha256: `828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35` (`model.sha256`)
- fetch, once, into the Hugging Face cache layout the tests read (`BGE_MODEL_PATH` overrides the path):

```sh
curl -fL --create-dirs -o "$HOME/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/onnx/model.onnx" \
  https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/onnx/model.onnx
```

The repository vendors only the first 4096 bytes (`model_header.bin`) and the hash.

## the corpus

`corpus.jsonl` holds 64 rows, one JSON object per line: `index`, `source`, `text`, `ids` (WordPiece ids including 101 and 102).

- rows 0 to 2 are the three sentences hard-coded in `proxima-onnx/examples/bge_eval.rs`; their ids are checked against the tokenizer when the file is built
- rows 3 to 63 are 61 sentences taken at an even stride from the prose of, in this order: `SHAPE.md`, `FEATURES.md`, `SECURITY.md`, then `book/src/algebra/*.md` in name order, then the remaining `book/src/**/*.md` in path order (the root documents and the algebra pages held 50 usable sentences, 11 short of the 61 needed)
- a sentence is kept when it starts with an uppercase letter, ends with `.`, `?` or `!`, has 6 to 30 words, contains only ASCII letters, digits, spaces and `,;:'"-.?!`, has no interior `.`, passes the repository content gate, and tokenizes to at most 64 ids
- the ids come from `tokenizer.json` of the revision above, `encode(text, true)`

The builder is `examples/corpus_build.rs` of the indexer repo (it owns the tokenizer dependency). It takes the proxima checkout root, the revision's `tokenizer.json` and the output path.
