| generation | new tokens | pipeline misses (compile ms) | prepare ms | pre_encode ms | encode ms | commit ms | commit end to scheduled ms | scheduled to gpu start ms | gpu ms | commit end to gpu end ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1000 | 20 (18.7) | 30.69 | 26.90 | 3.73 | 0.027 | 48.724 | 0.211 | 236.24 | 285.17 |
| 1 | 1002 | 1 (0.9) | 30.19 | 7.31 | 3.11 | 0.010 | 4.679 | 0.079 | 229.88 | 234.63 |
