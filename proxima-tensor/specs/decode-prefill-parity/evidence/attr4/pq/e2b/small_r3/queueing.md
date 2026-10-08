| generation | new tokens | pipeline misses (compile ms) | prepare ms | pre_encode ms | encode ms | commit ms | commit end to scheduled ms | scheduled to gpu start ms | gpu ms | commit end to gpu end ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 27 | 48 (31.8) | 16.77 | 40.28 | 5.35 | 0.026 | 86.092 | 0.192 | 89.56 | 175.85 |
| 1 | 971 | 11 (5.8) | 16.72 | 14.44 | 4.83 | 0.009 | 13.644 | 0.432 | 571.78 | 585.86 |
| 2 | 972 | 2 (1.5) | 16.48 | 10.12 | 4.82 | 0.009 | 7.237 | 0.388 | 571.37 | 578.99 |
