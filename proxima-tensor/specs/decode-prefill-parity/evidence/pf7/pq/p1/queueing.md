| generation | new tokens | pipeline misses (compile ms) | prepare ms | pre_encode ms | encode ms | commit ms | commit end to scheduled ms | scheduled to gpu start ms | gpu ms | commit end to gpu end ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1000 | 20 (24.4) | 32.75 | 35.80 | 3.41 | 0.030 | 52.654 | 0.046 | 209.55 | 262.25 |
| 1 | 1002 | 4 (181.4) | 31.61 | 289.26 | 2.73 | 0.012 | 4.921 | 0.067 | 208.98 | 213.97 |
| 2 | 1002 | 0 (0.0) | 33.68 | 238.88 | 3.21 | 0.010 | 5.506 | 0.051 | 207.81 | 213.37 |
