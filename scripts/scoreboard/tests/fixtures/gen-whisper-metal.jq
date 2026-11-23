def noise($i; $salt): ((($i * 7919) + ($salt * 104729)) % 1000) / 1000;
def jitter($base; $spread; $i; $salt): ($base + (noise($i; $salt) - 0.5) * $spread) | floor;
def stamp($sec; $frac): (1795442400 + $sec | strftime("%Y-%m-%dT%H:%M:%S")) + "." + (("000000" + ($frac | tostring))[-6:]) + "Z";

def inputs_: [
  {id: "jfk-11s", duration_us: 11000000, language: "en", tokens: 28},
  {id: "libri-6s", duration_us: 6200000, language: "en", tokens: 17},
  {id: "podcast-25s", duration_us: 24500000, language: "en", tokens: 55}
];

def arm_spec($arm): if $arm == "proxima" then
    {arm: "proxima", version: "0.1.0", commit: "fa5923952", load: 310000, mel: 21000, encode: 870000, step: 65000,
     footprint: 612000000, rss: 548000000, gpu: 410000000, first_gap: 3000}
  else
    {arm: "whisper.cpp", version: "1.7.5", commit: "a2b4c6e", load: 282000, mel: 18000, encode: 792000, step: 58000,
     footprint: 541000000, rss: 497000000, gpu: null, first_gap: 4000}
  end;

def record($idx; $round; $pos; $proc; $spec; $in; $repeat; $warmup; $t):
  (jitter($spec.step; 9000; $idx; 3)) as $s0
  | ([range(0; $in.tokens)] | map(jitter($spec.step; 9000; $idx * 100 + .; 5))) as $steps
  | (if $repeat == 0 then jitter($spec.load; 20000; $idx; 1) else 0 end) as $load
  | (jitter($spec.mel; 3000; $idx; 2)) as $mel
  | (jitter($spec.encode; 40000; $idx; 4)) as $enc
  | ($steps | add) as $dec
  | ($load + $mel + $enc + $steps[0]) as $ttft
  | ($load + $mel + $enc + $dec) as $ttlt
  | ($ttlt + (if $repeat == 0 then 90000 else 4000 end)) as $wall
  | {
      record: "run",
      model: "whisper-tiny", modality: "speech", backend: "metal",
      arm: $spec.arm, arm_version: $spec.version, commit: $spec.commit,
      session_id: "s-20261123T140000Z-a1",
      index: $idx, round: $round, position: $pos, repeat: $repeat,
      started_at: stamp($t; ($idx * 13) % 1000000),
      ended_at: stamp($t + ($wall / 1000000 | ceil); ($idx * 13) % 1000000),
      exit_status: 0, admitted: true, discarded_warmup: $warmup,
      host: {load1: ((1 + noise($idx; 6)) * 100 | floor / 100), hw_model: "MacBookPro18,2", chip: "Apple M1 Max",
             memory_bytes: 34359738368, os_build: "24G84", power_source: "AC"},
      input: {id: (if $repeat == 0 then $in.id else $in.id + "-b" end), duration_us: $in.duration_us, language: $in.language},
      prompt_tokens: 4, generated_tokens: $in.tokens,
      timings: ({ttft_us: $ttft, ttlt_us: $ttlt, load_us: $load, mel_us: $mel, encode_us: $enc, decode_us: $dec,
                 step_us: (if $spec.arm == "proxima" then $steps else {unreported: "incumbent-does-not-report"} end)}),
      resources: {footprint_peak_bytes: jitter($spec.footprint; 24000000; $idx; 7),
                  steady_rss_bytes: jitter($spec.rss; 16000000; $idx; 8),
                  gpu_allocated_peak_bytes: (if $spec.gpu == null then {unreported: "incumbent-does-not-report"} else jitter($spec.gpu; 8000000; $idx; 9) end),
                  cpu_user_us: ($wall * 0.46 | floor), cpu_sys_us: ($wall * 0.07 | floor), wall_us: $wall},
      quality: {match: true, match_incumbent: true}
    };

def processes: (["proxima", "whisper.cpp"] | map(arm_spec(.))) as $specs
  | [ ($specs[] | {warm: true, spec: .}), (range(0; 20) as $r | $specs[] | {warm: false, spec: ., round: $r}) ];

[ processes | to_entries[] | .key as $p | .value as $v
  | (inputs_[$p % 3]) as $in
  | (($p * 12) + 5) as $t
  | record($p * 2; ($v.round // -1); ($p % 2); $p; $v.spec; $in; 0; $v.warm; $t),
    record($p * 2 + 1; ($v.round // -1); ($p % 2); $p; $v.spec; $in; 1; $v.warm; $t + 6)
]
| .[],
  {record: "not-run", backend: "cpu", reason: "unexercised", detail: "fixture: only the Metal arm is exercised"},
  {record: "not-run", backend: "cuda", reason: "driver-mismatch", detail: "versailles NVML driver and library versions differ"},
  {record: "not-run", backend: "hip", reason: "no-host", detail: "no AMD host"},
  {record: "not-run", backend: "wgpu", reason: "unexercised", detail: "wgpu path not exercised"}
