| generation | new tokens | pipeline misses (compile ms) | prepare ms | pre_encode ms | encode ms | commit ms | commit end to scheduled ms | scheduled to gpu start ms | gpu ms | commit end to gpu end ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 36 | 33 (23.6) | 31.18 | 32.58 | 3.59 | 0.026 | 44.006 | 0.021 | 42.90 | 86.93 |
| 1 | 1000 | 8 (4.0) | 30.85 | 10.42 | 3.10 | 0.009 | 5.123 | 0.081 | 239.45 | 244.65 |
| 2 | 1002 | 1 (0.9) | 30.13 | 7.82 | 3.35 | 0.010 | 4.790 | 0.226 | 230.22 | 235.24 |
