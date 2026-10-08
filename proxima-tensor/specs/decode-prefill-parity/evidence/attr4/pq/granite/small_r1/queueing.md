| generation | new tokens | pipeline misses (compile ms) | prepare ms | pre_encode ms | encode ms | commit ms | commit end to scheduled ms | scheduled to gpu start ms | gpu ms | commit end to gpu end ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 36 | 33 (162.6) | 30.35 | 183.88 | 3.59 | 0.025 | 43.498 | 0.037 | 45.73 | 89.26 |
| 1 | 1000 | 8 (4.0) | 30.31 | 10.43 | 3.05 | 0.009 | 8.699 | 0.288 | 242.32 | 251.31 |
| 2 | 1002 | 1 (0.8) | 30.62 | 7.21 | 3.15 | 0.009 | 5.952 | 0.069 | 230.34 | 236.36 |
