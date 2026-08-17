# Topological and Weighted Network Features of Datasets

This table summarizes the topological properties, edge weight characteristics, clustering coefficients, and path metrics for all weighted datasets across training and testing splits.

| Dataset | Split | Category | Nodes (N) | Edges (M) | Avg Deg <k> | Max Deg | Avg Wt <w> | Avg Str <s> | Density ρ | C_unw | C_w | # CC | LCC Nodes (%) | APL (LCC) | Diameter |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Budapest.txt | TRAIN | Real-World | 480 | 989 | 4.12 | 94 | 4.947 | 20.39 | 0.0086 | 0.3004 | 0.0109 | 5 | 467 (97.3%) | 4.924 | 20 |
| C_elegans.txt | TRAIN | Real-World | 297 | 2,148 | 14.46 | 134 | 3.947 | 57.10 | 0.0489 | 0.2924 | 0.0183 | 1 | 297 (100.0%) | 2.455 | 5 |
| E.coli.edge | TRAIN | Real-World | 1,100 | 3,637 | 6.61 | 152 | 1.364 | 9.02 | 0.0060 | 0.4571 | 0.0201 | 1 | 1,100 (100.0%) | 3.833 | 11 |
| US_airports.txt | TRAIN | Real-World | 500 | 2,980 | 11.92 | 145 | 152320.190 | 1815656.66 | 0.0239 | 0.6175 | 0.0217 | 1 | 500 (100.0%) | 2.991 | 7 |
| carrib.txt | TRAIN | Real-World | 249 | 3,492 | 28.05 | 248 | 0.067 | 1.88 | 0.1131 | 0.4002 | 0.0012 | 1 | 249 (100.0%) | 1.887 | 2 |
| open_flights.txt | TRAIN | Real-World | 2,939 | 15,677 | 10.67 | 242 | 1.462 | 15.60 | 0.0036 | 0.4526 | 0.0595 | 11 | 2,905 (98.8%) | 4.116 | 14 |
| out.advogato | TRAIN | Real-World | 5,155 | 39,285 | 15.24 | 803 | 0.853 | 13.00 | 0.0030 | 0.2477 | 0.2081 | 57 | 5,042 (97.8%) | 3.294 | 8 |
| out.foldoc | TRAIN | Real-World | 13,356 | 91,471 | 13.70 | 728 | 1.052 | 14.41 | 0.0010 | 0.3379 | 0.0233 | 1 | 13,356 (100.0%) | 3.880 | 7 |
| synthetic_sf_100.txt | TRAIN | Synthetic (BBV) | 100 | 384 | 7.68 | 23 | 0.838 | 6.43 | 0.0776 | 0.1559 | 0.0057 | 1 | 100 (100.0%) | 2.399 | 4 |
| synthetic_sf_1000.txt | TRAIN | Synthetic (BBV) | 1,000 | 3,984 | 7.97 | 117 | 1.078 | 8.59 | 0.0080 | 0.0295 | 0.0007 | 1 | 1,000 (100.0%) | 3.202 | 5 |
| synthetic_sf_1500.txt | TRAIN | Synthetic (BBV) | 1,500 | 5,984 | 7.98 | 132 | 1.084 | 8.65 | 0.0053 | 0.0241 | 0.0007 | 1 | 1,500 (100.0%) | 3.321 | 5 |
| synthetic_sf_2000.txt | TRAIN | Synthetic (BBV) | 2,000 | 7,984 | 7.98 | 156 | 1.149 | 9.18 | 0.0040 | 0.0214 | 0.0005 | 1 | 2,000 (100.0%) | 3.425 | 5 |
| synthetic_sf_250.txt | TRAIN | Synthetic (BBV) | 250 | 984 | 7.87 | 65 | 1.152 | 9.07 | 0.0316 | 0.1150 | 0.0041 | 1 | 250 (100.0%) | 2.681 | 4 |
| synthetic_sf_2500.txt | TRAIN | Synthetic (BBV) | 2,500 | 9,984 | 7.99 | 127 | 1.137 | 9.08 | 0.0032 | 0.0172 | 0.0004 | 1 | 2,500 (100.0%) | 3.488 | 6 |
| synthetic_sf_3000.txt | TRAIN | Synthetic (BBV) | 3,000 | 11,984 | 7.99 | 214 | 1.146 | 9.16 | 0.0027 | 0.0152 | 0.0003 | 1 | 3,000 (100.0%) | 3.558 | 6 |
| synthetic_sf_4000.txt | TRAIN | Synthetic (BBV) | 4,000 | 15,984 | 7.99 | 220 | 1.167 | 9.33 | 0.0020 | 0.0138 | 0.0003 | 1 | 4,000 (100.0%) | 3.626 | 6 |
| synthetic_sf_500.txt | TRAIN | Synthetic (BBV) | 500 | 1,984 | 7.94 | 61 | 1.027 | 8.15 | 0.0159 | 0.0490 | 0.0014 | 1 | 500 (100.0%) | 2.963 | 5 |
| synthetic_sf_850.txt | TRAIN | Synthetic (BBV) | 850 | 3,384 | 7.96 | 106 | 1.088 | 8.66 | 0.0094 | 0.0438 | 0.0007 | 1 | 850 (100.0%) | 3.123 | 5 |
| cargoshipsBB.txt | TEST | Real-World | 834 | 4,349 | 10.43 | 173 | 97.709 | 1019.04 | 0.0125 | 0.4170 | 0.0037 | 7 | 821 (98.4%) | 3.339 | 9 |
| facebook_combined.txt | TEST | Real-World | 4,039 | 88,234 | 43.69 | 1045 | 1.000 | 43.69 | 0.0108 | 0.6055 | 0.6055 | 1 | 4,039 (100.0%) | 3.706 | 8 |
| karate.txt | TEST | Real-World | 34 | 78 | 4.59 | 17 | 1.000 | 4.59 | 0.1390 | 0.5706 | 0.5706 | 1 | 34 (100.0%) | 2.408 | 5 |
| synthetic_test_realworld.txt | TEST | Synthetic (BBV) | 800 | 3,184 | 7.96 | 111 | 1.000 | 7.96 | 0.0100 | 0.0388 | 0.0388 | 1 | 800 (100.0%) | 3.114 | 5 |
| synthetic_test_sf_1000.txt | TEST | Synthetic (BBV) | 1,000 | 3,984 | 7.97 | 112 | 1.079 | 8.60 | 0.0080 | 0.0366 | 0.0008 | 1 | 1,000 (100.0%) | 3.193 | 5 |
| synthetic_test_sf_2000.txt | TEST | Synthetic (BBV) | 2,000 | 7,984 | 7.98 | 168 | 1.085 | 8.66 | 0.0040 | 0.0200 | 0.0004 | 1 | 2,000 (100.0%) | 3.398 | 5 |
| synthetic_test_sf_250.txt | TEST | Synthetic (BBV) | 250 | 984 | 7.87 | 50 | 0.944 | 7.43 | 0.0316 | 0.1052 | 0.0028 | 1 | 250 (100.0%) | 2.711 | 4 |
| synthetic_test_sf_500.txt | TEST | Synthetic (BBV) | 500 | 1,984 | 7.94 | 72 | 1.052 | 8.35 | 0.0159 | 0.0586 | 0.0017 | 1 | 500 (100.0%) | 2.936 | 5 |
| synthetic_test_sf_5000.txt | TEST | Synthetic (BBV) | 5,000 | 19,984 | 7.99 | 224 | 1.170 | 9.36 | 0.0016 | 0.0106 | 0.0002 | 1 | 5,000 (100.0%) | 3.693 | 6 |
