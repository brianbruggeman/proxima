| generation | new tokens | pipeline misses (compile ms) | prepare ms | pre_encode ms | encode ms | commit ms | commit end to scheduled ms | scheduled to gpu start ms | gpu ms | commit end to gpu end ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 36 | 33 (26.8) | 31.50 | 35.87 | 3.74 | 0.038 | 45.564 | 0.033 | 43.24 | 88.84 |
| 1 | 1000 | 8 (4.2) | 30.78 | 10.62 | 3.25 | 0.012 | 5.291 | 0.067 | 238.84 | 244.20 |
| 2 | 1002 | 1 (1.0) | 31.98 | 8.02 | 3.45 | 0.014 | 5.019 | 0.056 | 230.63 | 235.71 |
