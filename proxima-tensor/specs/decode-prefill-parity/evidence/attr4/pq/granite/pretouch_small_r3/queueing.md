| generation | new tokens | pipeline misses (compile ms) | prepare ms | pre_encode ms | encode ms | commit ms | commit end to scheduled ms | scheduled to gpu start ms | gpu ms | commit end to gpu end ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 36 | 33 (24.0) | 31.08 | 32.82 | 3.55 | 0.027 | 47.763 | 0.017 | 43.57 | 91.35 |
| 1 | 1000 | 8 (4.1) | 30.24 | 10.77 | 3.24 | 0.010 | 9.471 | 0.058 | 244.20 | 253.73 |
| 2 | 1002 | 1 (0.8) | 30.67 | 7.28 | 3.33 | 0.011 | 8.592 | 0.059 | 230.99 | 239.64 |
