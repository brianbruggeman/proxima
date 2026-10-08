| generation | new tokens | pipeline misses (compile ms) | prepare ms | pre_encode ms | encode ms | commit ms | commit end to scheduled ms | scheduled to gpu start ms | gpu ms | commit end to gpu end ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 36 | 33 (26.0) | 32.82 | 34.92 | 3.74 | 0.030 | 44.429 | 0.038 | 45.96 | 90.42 |
| 1 | 1000 | 8 (4.1) | 30.51 | 10.62 | 3.29 | 0.009 | 5.083 | 0.081 | 245.96 | 251.13 |
| 2 | 1002 | 1 (0.9) | 29.87 | 7.36 | 3.33 | 0.009 | 4.349 | 0.084 | 230.08 | 234.51 |
