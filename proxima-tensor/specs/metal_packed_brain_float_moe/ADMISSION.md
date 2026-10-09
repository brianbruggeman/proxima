VERDICT: ADMIT

The spec defines bounded Metal forward execution for the existing scalar BF8
E5M2 and BF4 E2M1 packed codecs. It binds the flattened BF4 nibble layout,
uses an odd-width row boundary, requires direct packed-byte reads with no
expanded FP32 weight staging, checks the selected-expert result against CPU,
and preserves typed rejection for an unsupported packed codec. GPU autograd,
optimizer updates, broader training feasibility, and performance claims are
outside this admitted work.
