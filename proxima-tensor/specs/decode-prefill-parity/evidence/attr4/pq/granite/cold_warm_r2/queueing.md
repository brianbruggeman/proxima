| generation | new tokens | pipeline misses (compile ms) | prepare ms | pre_encode ms | encode ms | commit ms | commit end to scheduled ms | scheduled to gpu start ms | gpu ms | commit end to gpu end ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1000 | 20 (19.2) | 30.52 | 27.59 | 3.71 | 0.027 | 51.561 | 0.280 | 237.18 | 289.02 |
| 1 | 1002 | 1 (0.9) | 30.57 | 7.47 | 3.13 | 0.014 | 8.682 | 0.075 | 230.80 | 239.55 |
