| generation | new tokens | pipeline misses (compile ms) | prepare ms | pre_encode ms | encode ms | commit ms | commit end to scheduled ms | scheduled to gpu start ms | gpu ms | commit end to gpu end ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1000 | 20 (10.8) | 33.50 | 18.96 | 4.38 | 0.013 | 50.625 | 0.051 | 238.33 | 289.00 |
| 1 | 1002 | 1 (0.9) | 31.18 | 7.29 | 3.31 | 0.010 | 5.453 | 0.076 | 230.30 | 235.83 |
