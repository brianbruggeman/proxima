# fixture scoreboard for the census test

Whisper tiny on Metal, proxima against whisper.cpp.

| bar | proxima | incumbent | measured |
|---|---|---|---|
| encode p50, cited | 870.0 ms | 792.0 ms | runs=../whisper_metal.jsonl@393606f389733269e8431c0129350165cd3e5b6cc9173c4f0111e99fd76ca7c8 |
| decode, hand typed | 15.4 tok/s | 17.2 tok/s | n=20 |
| peak RAM, wrong sha | 612.0 MB | 541.0 MB | runs=../whisper_metal.jsonl@0000000000000000000000000000000000000000000000000000000000000000 |

| bar | proxima | incumbent |
|---|---|---|
| output parity | yes | yes | extra cell |

TTFT cold is 1,261.2 ms on proxima against 1,145.2 ms on whisper.cpp, typed by hand.
The ratio is 1.101x at p50, typed by hand.
Encode is 870.0 ms at p50 runs=../whisper_metal.jsonl@393606f389733269e8431c0129350165cd3e5b6cc9173c4f0111e99fd76ca7c8
Block 10 ran on the quiet box with no number and unit here.

```
fenced numbers are skipped: 4,964 ms and 3.5 s
```
