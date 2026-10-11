| generation | new tokens | pipeline misses (compile ms) | prepare ms | pre_encode ms | encode ms | commit ms | commit end to scheduled ms | scheduled to gpu start ms | gpu ms | commit end to gpu end ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1000 | 20 (25.8) | 33.57 | 35.76 | 3.67 | 0.036 | 50.885 | 0.294 | 208.52 | 259.70 |
| 1 | 1002 | 4 (3.4) | 33.36 | 13.62 | 2.88 | 0.015 | 4.937 | 0.056 | 202.09 | 207.08 |
| 2 | 1002 | 0 (0.0) | 32.47 | 12.76 | 2.67 | 0.011 | 6.191 | 0.326 | 203.46 | 209.98 |
