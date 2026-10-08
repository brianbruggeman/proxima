| generation | new tokens | pipeline misses (compile ms) | prepare ms | pre_encode ms | encode ms | commit ms | commit end to scheduled ms | scheduled to gpu start ms | gpu ms | commit end to gpu end ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1000 | 20 (19.1) | 30.82 | 27.38 | 3.89 | 0.028 | 53.650 | 0.035 | 237.25 | 290.94 |
| 1 | 1002 | 1 (0.8) | 29.99 | 7.23 | 3.17 | 0.011 | 4.533 | 0.059 | 229.89 | 234.49 |
