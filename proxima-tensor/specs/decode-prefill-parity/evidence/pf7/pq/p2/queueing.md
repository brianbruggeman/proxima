| generation | new tokens | pipeline misses (compile ms) | prepare ms | pre_encode ms | encode ms | commit ms | commit end to scheduled ms | scheduled to gpu start ms | gpu ms | commit end to gpu end ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1000 | 20 (25.1) | 32.85 | 35.23 | 3.40 | 0.035 | 51.082 | 0.286 | 208.16 | 259.53 |
| 1 | 1002 | 4 (3.6) | 33.57 | 13.68 | 2.68 | 0.019 | 4.742 | 0.049 | 201.60 | 206.40 |
| 2 | 1002 | 0 (0.0) | 32.34 | 13.65 | 2.86 | 0.017 | 5.083 | 0.062 | 201.10 | 206.25 |
