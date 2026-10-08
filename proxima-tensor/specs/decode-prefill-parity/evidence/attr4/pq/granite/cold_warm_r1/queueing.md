| generation | new tokens | pipeline misses (compile ms) | prepare ms | pre_encode ms | encode ms | commit ms | commit end to scheduled ms | scheduled to gpu start ms | gpu ms | commit end to gpu end ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1000 | 20 (34.3) | 30.74 | 43.67 | 3.81 | 0.029 | 50.047 | 0.011 | 236.35 | 286.41 |
| 1 | 1002 | 1 (0.8) | 29.94 | 7.31 | 3.11 | 0.010 | 4.576 | 0.069 | 230.05 | 234.69 |
