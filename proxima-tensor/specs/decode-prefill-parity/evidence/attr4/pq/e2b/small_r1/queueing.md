| generation | new tokens | pipeline misses (compile ms) | prepare ms | pre_encode ms | encode ms | commit ms | commit end to scheduled ms | scheduled to gpu start ms | gpu ms | commit end to gpu end ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 27 | 48 (199.0) | 16.48 | 207.63 | 5.35 | 0.030 | 85.922 | 0.201 | 89.62 | 175.74 |
| 1 | 971 | 11 (5.7) | 16.52 | 14.23 | 4.74 | 0.009 | 13.337 | 0.351 | 580.24 | 593.92 |
| 2 | 972 | 2 (1.5) | 16.49 | 10.09 | 4.81 | 0.010 | 7.452 | 0.387 | 571.48 | 579.32 |
