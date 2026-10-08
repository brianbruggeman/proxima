| generation | new tokens | pipeline misses (compile ms) | prepare ms | pre_encode ms | encode ms | commit ms | commit end to scheduled ms | scheduled to gpu start ms | gpu ms | commit end to gpu end ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 36 | 33 (24.7) | 30.77 | 33.49 | 3.55 | 0.026 | 48.532 | 0.039 | 43.62 | 92.19 |
| 1 | 1000 | 8 (4.0) | 30.21 | 10.48 | 3.12 | 0.010 | 8.879 | 0.059 | 244.13 | 253.07 |
| 2 | 1002 | 1 (0.9) | 30.18 | 7.36 | 3.25 | 0.009 | 8.350 | 0.068 | 230.97 | 239.39 |
