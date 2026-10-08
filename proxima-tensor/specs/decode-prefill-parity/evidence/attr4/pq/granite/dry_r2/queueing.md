| generation | new tokens | pipeline misses (compile ms) | prepare ms | pre_encode ms | encode ms | commit ms | commit end to scheduled ms | scheduled to gpu start ms | gpu ms | commit end to gpu end ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1000 | 20 (10.8) | 34.19 | 19.43 | 4.10 | 0.012 | 55.437 | 0.289 | 236.74 | 292.46 |
| 1 | 1002 | 1 (0.9) | 29.93 | 7.35 | 3.15 | 0.010 | 8.464 | 0.390 | 232.25 | 241.10 |
