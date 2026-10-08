| generation | new tokens | pipeline misses (compile ms) | prepare ms | pre_encode ms | encode ms | commit ms | commit end to scheduled ms | scheduled to gpu start ms | gpu ms | commit end to gpu end ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1000 | 20 (19.5) | 30.66 | 27.84 | 3.76 | 0.025 | 52.041 | 0.308 | 237.49 | 289.84 |
| 1 | 1002 | 1 (0.9) | 31.42 | 7.47 | 3.13 | 0.011 | 8.767 | 0.379 | 231.32 | 240.47 |
