# Block 2: Stability Tests

A ranking is computed from one view of a population (A) and checked against the SIR spreading on another view (B), over the people present in both. **SIR on A (ref.)** uses the true spreading ranking of view A: it shows how much the spreading itself changes between the two views. Calibrated SIR regime (10% mean outbreak) in every view. Best competing method in **bold**.

## (a) Face-to-face vs co-location (same people, different contact definition)

### Kendall tau inside the true top 20% of view B

| site | direction | common | Ours | Deep6 | W-Eig | Strength | W-Clos | W-Betw | SIR on A (ref.) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LH10_Hospital | face-to-face -> co-location | 67 | 0.231 | 0.256 | 0.179 | 0.282 | 0.154 | **0.374** | 0.256 |
| LH10_Hospital | co-location -> face-to-face | 67 | **0.487** | 0.462 | **0.487** | 0.462 | 0.462 | 0.168 | 0.513 |
| InVS13_Workplace | face-to-face -> co-location | 92 | -0.359 | -0.451 | **-0.190** | -0.307 | -0.373 | -0.287 | -0.359 |
| InVS13_Workplace | co-location -> face-to-face | 92 | -0.085 | 0.033 | -0.098 | -0.020 | 0.046 | **0.339** | -0.020 |
| InVS15_Workplace | face-to-face -> co-location | 217 | -0.118 | -0.076 | -0.114 | -0.068 | -0.023 | **0.004** | -0.127 |
| InVS15_Workplace | co-location -> face-to-face | 217 | 0.200 | 0.205 | **0.274** | 0.229 | 0.110 | 0.103 | 0.200 |
| SFHH_Conference | face-to-face -> co-location | 403 | 0.221 | 0.178 | 0.216 | 0.231 | 0.179 | **0.231** | 0.236 |
| SFHH_Conference | co-location -> face-to-face | 403 | 0.223 | 0.192 | 0.199 | **0.228** | 0.204 | 0.208 | 0.239 |
| LyonSchool_Primary | face-to-face -> co-location | 242 | **0.406** | 0.239 | 0.358 | 0.397 | 0.335 | 0.279 | 0.429 |
| LyonSchool_Primary | co-location -> face-to-face | 242 | 0.278 | -0.261 | **0.294** | 0.227 | 0.083 | -0.024 | 0.280 |
| Thiers13_HighSchool | face-to-face -> co-location | 327 | -0.042 | -0.035 | -0.069 | **0.084** | -0.046 | -0.047 | -0.041 |
| Thiers13_HighSchool | co-location -> face-to-face | 327 | 0.171 | **0.186** | 0.108 | 0.179 | 0.081 | -0.169 | 0.095 |
| **Mean** | | | 0.134 | 0.077 | 0.137 | **0.160** | 0.101 | 0.098 | 0.142 |

### Top-20% overlap with view B

| site | direction | common | Ours | Deep6 | W-Eig | Strength | W-Clos | W-Betw | SIR on A (ref.) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LH10_Hospital | face-to-face -> co-location | 67 | 0.385 | 0.385 | 0.308 | 0.385 | 0.385 | **0.462** | 0.308 |
| LH10_Hospital | co-location -> face-to-face | 67 | 0.231 | **0.308** | 0.000 | **0.308** | 0.231 | 0.154 | 0.308 |
| InVS13_Workplace | face-to-face -> co-location | 92 | 0.056 | 0.056 | 0.056 | 0.000 | **0.111** | **0.111** | 0.111 |
| InVS13_Workplace | co-location -> face-to-face | 92 | 0.111 | **0.167** | 0.111 | 0.056 | 0.111 | 0.111 | 0.111 |
| InVS15_Workplace | face-to-face -> co-location | 217 | 0.349 | **0.372** | 0.302 | 0.326 | 0.349 | 0.302 | 0.349 |
| InVS15_Workplace | co-location -> face-to-face | 217 | 0.349 | **0.442** | 0.279 | 0.302 | 0.279 | 0.349 | 0.349 |
| SFHH_Conference | face-to-face -> co-location | 403 | **0.637** | 0.562 | 0.487 | 0.537 | 0.562 | 0.588 | 0.625 |
| SFHH_Conference | co-location -> face-to-face | 403 | **0.637** | 0.600 | 0.613 | **0.637** | 0.588 | 0.425 | 0.625 |
| LyonSchool_Primary | face-to-face -> co-location | 242 | **0.604** | 0.542 | 0.521 | 0.542 | 0.542 | 0.458 | 0.625 |
| LyonSchool_Primary | co-location -> face-to-face | 242 | 0.583 | 0.229 | **0.688** | 0.583 | 0.604 | 0.375 | 0.625 |
| Thiers13_HighSchool | face-to-face -> co-location | 327 | **0.538** | 0.523 | 0.385 | 0.431 | 0.431 | 0.262 | 0.523 |
| Thiers13_HighSchool | co-location -> face-to-face | 327 | 0.600 | 0.231 | **0.615** | 0.585 | 0.246 | 0.246 | 0.523 |
| **Mean** | | | **0.423** | 0.368 | 0.364 | 0.391 | 0.370 | 0.320 | 0.423 |

### Agreement: Kendall tau between each method's own scores on A and on B

| site | direction | common | Ours | Deep6 | W-Eig | Strength | W-Clos | W-Betw | SIR on A (ref.) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LH10_Hospital | face-to-face -> co-location | 67 | 0.347 | **0.360** | 0.128 | 0.330 | 0.342 | 0.275 | 0.408 |
| LH10_Hospital | co-location -> face-to-face | 67 | 0.347 | **0.360** | 0.128 | 0.330 | 0.342 | 0.275 | 0.408 |
| InVS13_Workplace | face-to-face -> co-location | 92 | 0.027 | 0.041 | 0.078 | 0.050 | 0.024 | **0.090** | 0.016 |
| InVS13_Workplace | co-location -> face-to-face | 92 | 0.027 | 0.041 | 0.078 | 0.050 | 0.024 | **0.090** | 0.016 |
| InVS15_Workplace | face-to-face -> co-location | 217 | 0.314 | 0.319 | **0.364** | 0.307 | 0.252 | 0.276 | 0.310 |
| InVS15_Workplace | co-location -> face-to-face | 217 | 0.314 | 0.319 | **0.364** | 0.307 | 0.252 | 0.276 | 0.310 |
| SFHH_Conference | face-to-face -> co-location | 403 | **0.435** | 0.368 | 0.317 | 0.425 | 0.343 | 0.357 | 0.445 |
| SFHH_Conference | co-location -> face-to-face | 403 | **0.435** | 0.368 | 0.317 | 0.425 | 0.343 | 0.357 | 0.445 |
| LyonSchool_Primary | face-to-face -> co-location | 242 | 0.556 | 0.127 | **0.581** | 0.536 | 0.436 | 0.301 | 0.527 |
| LyonSchool_Primary | co-location -> face-to-face | 242 | 0.556 | 0.127 | **0.581** | 0.536 | 0.436 | 0.301 | 0.527 |
| Thiers13_HighSchool | face-to-face -> co-location | 327 | 0.391 | 0.206 | **0.473** | 0.311 | 0.028 | 0.113 | 0.363 |
| Thiers13_HighSchool | co-location -> face-to-face | 327 | 0.391 | 0.206 | **0.473** | 0.311 | 0.028 | 0.113 | 0.363 |
| **Mean** | | | **0.345** | 0.237 | 0.323 | 0.327 | 0.238 | 0.236 | 0.345 |

## (b) Utah schools: contacts within 2 m vs within 1 m

### Kendall tau inside the true top 20% of view B

| site | direction | common | Ours | Deep6 | W-Eig | Strength | W-Clos | W-Betw | SIR on A (ref.) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Utah_MiddleSchool | 2 m -> 1 m | 590 | **0.569** | 0.195 | 0.520 | 0.443 | 0.308 | 0.206 | 0.520 |
| Utah_MiddleSchool | 1 m -> 2 m | 590 | **0.350** | 0.188 | 0.298 | **0.350** | 0.155 | 0.173 | 0.387 |
| Utah_ElementarySchool | 2 m -> 1 m | 339 | **0.650** | 0.393 | 0.454 | 0.487 | -0.215 | -0.078 | 0.619 |
| Utah_ElementarySchool | 1 m -> 2 m | 339 | **0.510** | 0.367 | 0.318 | 0.432 | 0.070 | 0.005 | 0.660 |
| **Mean** | | | **0.520** | 0.286 | 0.397 | 0.428 | 0.079 | 0.077 | 0.547 |

### Top-20% overlap with view B

| site | direction | common | Ours | Deep6 | W-Eig | Strength | W-Clos | W-Betw | SIR on A (ref.) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Utah_MiddleSchool | 2 m -> 1 m | 590 | **0.754** | 0.500 | 0.695 | 0.695 | 0.636 | 0.551 | 0.712 |
| Utah_MiddleSchool | 1 m -> 2 m | 590 | **0.695** | 0.483 | 0.653 | 0.610 | 0.483 | 0.458 | 0.712 |
| Utah_ElementarySchool | 2 m -> 1 m | 339 | **0.806** | 0.657 | 0.642 | 0.701 | 0.328 | 0.299 | 0.821 |
| Utah_ElementarySchool | 1 m -> 2 m | 339 | **0.791** | 0.567 | 0.582 | 0.746 | 0.493 | 0.313 | 0.821 |
| **Mean** | | | **0.762** | 0.552 | 0.643 | 0.688 | 0.485 | 0.405 | 0.766 |

### Agreement: Kendall tau between each method's own scores on A and on B

| site | direction | common | Ours | Deep6 | W-Eig | Strength | W-Clos | W-Betw | SIR on A (ref.) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Utah_MiddleSchool | 2 m -> 1 m | 590 | **0.718** | 0.355 | 0.717 | 0.696 | 0.647 | 0.626 | 0.692 |
| Utah_MiddleSchool | 1 m -> 2 m | 590 | **0.718** | 0.355 | 0.717 | 0.696 | 0.647 | 0.626 | 0.692 |
| Utah_ElementarySchool | 2 m -> 1 m | 339 | **0.871** | 0.359 | 0.747 | 0.822 | 0.667 | 0.512 | 0.869 |
| Utah_ElementarySchool | 1 m -> 2 m | 339 | **0.871** | 0.359 | 0.747 | 0.822 | 0.667 | 0.512 | 0.869 |
| **Mean** | | | **0.794** | 0.357 | 0.732 | 0.759 | 0.657 | 0.569 | 0.780 |

## (c) Day 1 -> day 2: rank on day 1, score against spreading on day 2

### Kendall tau inside the true top 20% of view B

| site | direction | common | Ours | Deep6 | W-Eig | Strength | W-Clos | W-Betw | SIR on A (ref.) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LH10_Hospital | day 1 -> day 2 | 41 | **0.357** | 0.214 | **0.357** | **0.357** | 0.286 | 0.327 | 0.500 |
| LyonSchool_Primary | day 1 -> day 2 | 232 | 0.149 | 0.174 | **0.193** | 0.092 | 0.077 | 0.093 | 0.155 |
| Thiers11_HighSchool | day 1 -> day 2 | 111 | -0.035 | -0.061 | **0.156** | -0.113 | -0.078 | 0.026 | -0.087 |
| Thiers12_HighSchool | day 1 -> day 2 | 140 | 0.109 | 0.024 | **0.193** | 0.125 | -0.093 | 0.005 | 0.098 |
| Thiers13_HighSchool | day 1 -> day 2 | 296 | 0.340 | 0.276 | 0.072 | 0.372 | **0.403** | 0.141 | 0.516 |
| InVS13_Workplace | day 1 -> day 2 | 61 | 0.333 | 0.273 | 0.212 | 0.290 | 0.303 | **0.572** | 0.273 |
| InVS15_Workplace | day 1 -> day 2 | 153 | -0.076 | **0.062** | -0.094 | -0.221 | -0.237 | -0.234 | -0.172 |
| SFHH_Conference | day 1 -> day 2 | 360 | 0.302 | 0.278 | 0.282 | **0.414** | 0.224 | 0.378 | 0.300 |
| HT09_Conference | day 1 -> day 2 | 91 | 0.346 | 0.268 | 0.268 | 0.111 | 0.359 | **0.412** | 0.294 |
| Malawi_Village | day 1 -> day 2 | 53 | 0.568 | 0.432 | **0.841** | 0.023 | 0.432 | -0.047 | 0.579 |
| Utah_MiddleSchool | day 1 -> day 2 | 591 | 0.163 | 0.154 | **0.179** | 0.137 | 0.114 | 0.060 | 0.152 |
| Utah_ElementarySchool | day 1 -> day 2 | 339 | 0.161 | **0.210** | 0.088 | 0.166 | 0.030 | 0.094 | 0.215 |
| **Mean** | | | 0.227 | 0.192 | **0.229** | 0.146 | 0.152 | 0.152 | 0.235 |

### Top-20% overlap with view B

| site | direction | common | Ours | Deep6 | W-Eig | Strength | W-Clos | W-Betw | SIR on A (ref.) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LH10_Hospital | day 1 -> day 2 | 41 | 0.375 | **0.500** | **0.500** | 0.375 | 0.375 | 0.375 | 0.375 |
| LyonSchool_Primary | day 1 -> day 2 | 232 | **0.500** | 0.457 | 0.457 | 0.435 | 0.478 | 0.370 | 0.522 |
| Thiers11_HighSchool | day 1 -> day 2 | 111 | 0.273 | **0.364** | 0.273 | 0.227 | 0.273 | 0.273 | 0.227 |
| Thiers12_HighSchool | day 1 -> day 2 | 140 | 0.464 | **0.500** | **0.500** | 0.357 | 0.250 | 0.321 | 0.464 |
| Thiers13_HighSchool | day 1 -> day 2 | 296 | **0.644** | 0.542 | 0.017 | 0.458 | 0.525 | 0.288 | 0.712 |
| InVS13_Workplace | day 1 -> day 2 | 61 | **0.500** | 0.417 | 0.333 | **0.500** | **0.500** | 0.417 | 0.500 |
| InVS15_Workplace | day 1 -> day 2 | 153 | 0.467 | 0.400 | 0.067 | 0.400 | **0.533** | 0.433 | 0.600 |
| SFHH_Conference | day 1 -> day 2 | 360 | 0.389 | **0.458** | 0.403 | **0.458** | 0.389 | 0.403 | 0.431 |
| HT09_Conference | day 1 -> day 2 | 91 | 0.444 | **0.556** | 0.444 | **0.556** | 0.389 | 0.500 | 0.444 |
| Malawi_Village | day 1 -> day 2 | 53 | 0.700 | **0.800** | 0.700 | 0.700 | 0.200 | 0.300 | 0.700 |
| Utah_MiddleSchool | day 1 -> day 2 | 591 | 0.441 | **0.475** | **0.475** | 0.381 | 0.364 | 0.305 | 0.415 |
| Utah_ElementarySchool | day 1 -> day 2 | 339 | 0.433 | 0.313 | 0.418 | 0.463 | **0.522** | 0.239 | 0.328 |
| **Mean** | | | 0.469 | **0.482** | 0.382 | 0.442 | 0.400 | 0.352 | 0.477 |

### Agreement: Kendall tau between each method's own scores on A and on B

| site | direction | common | Ours | Deep6 | W-Eig | Strength | W-Clos | W-Betw | SIR on A (ref.) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LH10_Hospital | day 1 -> day 2 | 41 | 0.410 | 0.354 | 0.251 | **0.453** | 0.324 | 0.365 | 0.463 |
| LyonSchool_Primary | day 1 -> day 2 | 232 | **0.367** | 0.327 | 0.326 | 0.358 | 0.278 | 0.253 | 0.350 |
| Thiers11_HighSchool | day 1 -> day 2 | 111 | 0.013 | 0.099 | **0.227** | 0.123 | 0.158 | 0.202 | -0.010 |
| Thiers12_HighSchool | day 1 -> day 2 | 140 | 0.286 | 0.225 | **0.489** | 0.215 | 0.168 | 0.223 | 0.280 |
| Thiers13_HighSchool | day 1 -> day 2 | 296 | **0.509** | 0.471 | -0.036 | 0.419 | 0.421 | 0.249 | 0.503 |
| InVS13_Workplace | day 1 -> day 2 | 61 | 0.213 | 0.097 | -0.236 | 0.133 | 0.126 | **0.242** | 0.186 |
| InVS15_Workplace | day 1 -> day 2 | 153 | 0.190 | 0.189 | 0.233 | **0.259** | 0.199 | 0.195 | 0.217 |
| SFHH_Conference | day 1 -> day 2 | 360 | 0.234 | **0.253** | 0.212 | 0.237 | 0.202 | 0.250 | 0.238 |
| HT09_Conference | day 1 -> day 2 | 91 | 0.367 | 0.347 | 0.162 | **0.450** | 0.375 | 0.420 | 0.406 |
| Malawi_Village | day 1 -> day 2 | 53 | **0.433** | 0.386 | -0.443 | 0.322 | -0.093 | 0.205 | 0.364 |
| Utah_MiddleSchool | day 1 -> day 2 | 591 | 0.344 | 0.316 | **0.432** | 0.286 | 0.259 | 0.210 | 0.345 |
| Utah_ElementarySchool | day 1 -> day 2 | 339 | **0.542** | 0.386 | 0.435 | 0.475 | 0.393 | 0.182 | 0.471 |
| **Mean** | | | **0.326** | 0.287 | 0.171 | 0.311 | 0.234 | 0.250 | 0.318 |

## (c) Day-2 outbreak reduction when the day-1 top 5% are vaccinated

*SIR day 2 (oracle)* knows the day-2 contacts in advance, so it is an upper reference only.

| Site | Ours | Deep6 | W-Eig | Strength | W-Clos | W-Betw | Random | SIR day 1 (ref.) | SIR day 2 (oracle) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LH10_Hospital | **0.114** | **0.114** | **0.114** | **0.114** | **0.114** | **0.114** | 0.139 | 0.114 | 0.506 |
| LyonSchool_Primary | **0.290** | **0.290** | **0.290** | **0.290** | **0.290** | **0.290** | 0.229 | 0.290 | 0.589 |
| Thiers11_HighSchool | **0.380** | **0.380** | **0.380** | **0.380** | **0.380** | **0.380** | 0.206 | 0.380 | 0.607 |
| Thiers12_HighSchool | **0.218** | **0.218** | **0.218** | **0.218** | **0.218** | **0.218** | 0.199 | 0.218 | 0.486 |
| Thiers13_HighSchool | **0.298** | **0.298** | **0.298** | **0.298** | **0.298** | **0.298** | 0.242 | 0.298 | 0.463 |
| InVS13_Workplace | **0.353** | **0.353** | **0.353** | **0.353** | **0.353** | **0.353** | 0.148 | 0.353 | 0.431 |
| InVS15_Workplace | **0.113** | **0.113** | **0.113** | **0.113** | **0.113** | **0.113** | 0.207 | 0.113 | 0.473 |
| SFHH_Conference | **0.532** | **0.532** | **0.532** | **0.532** | **0.532** | **0.532** | 0.214 | 0.532 | 0.820 |
| HT09_Conference | **0.146** | **0.146** | **0.146** | **0.146** | **0.146** | **0.146** | 0.166 | 0.146 | 0.614 |
| Malawi_Village | 0.223 | 0.165 | 0.165 | **0.223** | 0.125 | 0.125 | 0.110 | 0.243 | 0.189 |
| Utah_MiddleSchool | **0.553** | 0.437 | 0.480 | 0.510 | 0.454 | 0.450 | 0.323 | 0.506 | 0.697 |
| Utah_ElementarySchool | 0.220 | 0.265 | 0.185 | 0.403 | **0.413** | 0.406 | 0.291 | 0.234 | 0.435 |
| **Mean** | 0.287 | 0.276 | 0.273 | **0.298** | 0.286 | 0.285 | 0.206 | 0.286 | 0.526 |

## (c) Day-2 outbreak reduction when the day-1 top 10% are vaccinated

*SIR day 2 (oracle)* knows the day-2 contacts in advance, so it is an upper reference only.

| Site | Ours | Deep6 | W-Eig | Strength | W-Clos | W-Betw | Random | SIR day 1 (ref.) | SIR day 2 (oracle) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LH10_Hospital | **0.121** | **0.121** | **0.121** | **0.121** | **0.121** | **0.121** | 0.220 | 0.121 | 0.585 |
| LyonSchool_Primary | **0.471** | **0.471** | **0.471** | **0.471** | **0.471** | **0.471** | 0.409 | 0.471 | 0.713 |
| Thiers11_HighSchool | **0.477** | **0.477** | **0.477** | **0.477** | **0.477** | **0.477** | 0.321 | 0.477 | 0.652 |
| Thiers12_HighSchool | **0.448** | **0.448** | **0.448** | **0.448** | **0.448** | **0.448** | 0.343 | 0.448 | 0.616 |
| Thiers13_HighSchool | **0.317** | **0.317** | **0.317** | **0.317** | **0.317** | **0.317** | 0.414 | 0.317 | 0.616 |
| InVS13_Workplace | **0.444** | **0.444** | **0.444** | **0.444** | **0.444** | **0.444** | 0.244 | 0.444 | 0.456 |
| InVS15_Workplace | **0.213** | **0.213** | **0.213** | **0.213** | **0.213** | **0.213** | 0.390 | 0.213 | 0.554 |
| SFHH_Conference | **0.666** | **0.666** | **0.666** | **0.666** | **0.666** | **0.666** | 0.396 | 0.666 | 0.904 |
| HT09_Conference | **0.362** | **0.362** | **0.362** | **0.362** | **0.362** | **0.362** | 0.321 | 0.362 | 0.722 |
| Malawi_Village | 0.272 | 0.219 | 0.207 | **0.277** | 0.107 | 0.245 | 0.174 | 0.219 | 0.236 |
| Utah_MiddleSchool | 0.705 | 0.627 | 0.682 | **0.719** | 0.665 | 0.670 | 0.573 | 0.694 | 0.858 |
| Utah_ElementarySchool | 0.369 | 0.398 | 0.263 | **0.631** | 0.580 | 0.580 | 0.493 | 0.352 | 0.639 |
| **Mean** | 0.405 | 0.397 | 0.389 | **0.429** | 0.406 | 0.418 | 0.358 | 0.399 | 0.629 |

Figure: `fig_stability.png`.
