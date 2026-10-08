| generation | new tokens | pipeline misses (compile ms) | prepare ms | pre_encode ms | encode ms | commit ms | commit end to scheduled ms | scheduled to gpu start ms | gpu ms | commit end to gpu end ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1000 | 20 (11.3) | 30.49 | 19.68 | 4.69 | 0.010 | 49.496 | 0.313 | 238.31 | 288.12 |
| 1 | 1002 | 1 (1.1) | 31.06 | 7.46 | 3.76 | 0.011 | 5.327 | 0.474 | 231.86 | 237.66 |
