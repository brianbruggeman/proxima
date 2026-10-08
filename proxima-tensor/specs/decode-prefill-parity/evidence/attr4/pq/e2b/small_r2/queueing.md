| generation | new tokens | pipeline misses (compile ms) | prepare ms | pre_encode ms | encode ms | commit ms | commit end to scheduled ms | scheduled to gpu start ms | gpu ms | commit end to gpu end ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 27 | 48 (30.0) | 16.74 | 38.42 | 5.28 | 0.046 | 82.009 | 0.220 | 88.63 | 170.86 |
| 1 | 971 | 11 (6.2) | 16.72 | 14.88 | 4.79 | 0.010 | 12.658 | 0.397 | 571.95 | 585.00 |
| 2 | 972 | 2 (1.4) | 16.80 | 10.04 | 4.77 | 0.010 | 6.990 | 0.299 | 576.70 | 583.99 |
