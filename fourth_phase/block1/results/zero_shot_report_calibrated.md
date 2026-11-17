# Block 1: Zero-Shot Results on Human Contact Networks (calibrated SIR regime)

The Third-Phase Hybrid Global-Local WNLGCN (3-seed ensemble) was applied to the contact networks **without any retraining**. Ground truth = expected SIR outbreak size from each node (calibrated: p = 1 - exp(-tau * w_norm) with tau chosen per network so that the mean outbreak is 10% of the population, 500 runs; the model's beta-dependent global descriptors use the same tau).

## Kendall tau inside the true top 20%

How well each method orders the real top spreaders. Best value per row in **bold**.

| Dataset | Group | n | Ours | Deep6 | W-Eig | Strength | W-Clos | W-Betw |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LH10_Hospital | Face-to-face | 75 | **0.848** | 0.543 | 0.581 | 0.771 | 0.543 | 0.574 |
| LyonSchool_Primary | Face-to-face | 242 | **0.832** | 0.303 | 0.596 | 0.520 | 0.574 | 0.333 |
| Thiers11_HighSchool | Face-to-face | 126 | **0.813** | 0.407 | 0.380 | 0.595 | 0.320 | 0.253 |
| Thiers12_HighSchool | Face-to-face | 180 | **0.686** | 0.425 | 0.308 | 0.378 | 0.257 | 0.057 |
| Thiers13_HighSchool | Face-to-face | 327 | **0.380** | 0.278 | 0.123 | 0.287 | 0.237 | 0.104 |
| InVS13_Workplace | Face-to-face | 92 | 0.529 | 0.216 | 0.085 | 0.294 | **0.569** | 0.480 |
| InVS15_Workplace | Face-to-face | 217 | **0.794** | 0.464 | 0.329 | 0.534 | 0.621 | 0.422 |
| SFHH_Conference | Face-to-face | 403 | **0.740** | 0.532 | 0.490 | 0.644 | 0.571 | 0.411 |
| HT09_Conference | Face-to-face | 113 | **0.861** | 0.117 | 0.558 | 0.602 | 0.489 | 0.576 |
| Infectious_Museum | Face-to-face | 410 | **0.760** | 0.488 | 0.021 | 0.343 | 0.126 | -0.059 |
| Malawi_Village | Face-to-face | 84 | **0.573** | 0.412 | 0.501 | 0.304 | -0.251 | -0.271 |
| Kenya_Households | Face-to-face | 47 | **0.833** | 0.667 | 0.667 | **0.833** | 0.611 | 0.141 |
| Utah_MiddleSchool | Proximity | 591 | **0.636** | 0.189 | 0.481 | 0.591 | 0.263 | 0.248 |
| Utah_ElementarySchool | Proximity | 339 | **0.607** | 0.398 | 0.357 | 0.504 | -0.130 | -0.029 |
| Utah_MiddleSchool_1m | Proximity | 590 | **0.716** | 0.159 | 0.563 | 0.438 | 0.167 | 0.171 |
| Utah_ElementarySchool_1m | Proximity | 339 | **0.584** | 0.466 | 0.397 | 0.494 | 0.049 | -0.020 |
| Copenhagen_BT | Proximity | 675 | **0.669** | 0.284 | 0.314 | 0.527 | 0.228 | 0.230 |
| RealityMining_BT | Proximity | 96 | **0.883** | 0.497 | 0.836 | 0.798 | 0.579 | 0.253 |
| Haggle_Infocom05 | Proximity | 274 | 0.536 | 0.548 | 0.548 | **0.601** | 0.485 | 0.496 |
| LH10_Hospital_Coloc | Co-location | 73 | 0.495 | -0.143 | 0.121 | 0.451 | **0.560** | 0.451 |
| InVS13_Workplace_Coloc | Co-location | 95 | **0.813** | 0.427 | 0.509 | 0.719 | 0.006 | 0.000 |
| InVS15_Workplace_Coloc | Co-location | 219 | **0.821** | 0.555 | 0.573 | 0.750 | 0.229 | 0.282 |
| SFHH_Conference_Coloc | Co-location | 403 | **0.813** | 0.692 | 0.738 | 0.782 | 0.563 | 0.246 |
| LyonSchool_Primary_Coloc | Co-location | 242 | **0.697** | -0.582 | 0.645 | 0.683 | 0.413 | 0.152 |
| Thiers13_HighSchool_Coloc | Co-location | 328 | 0.221 | -0.048 | 0.062 | **0.277** | 0.029 | -0.047 |
| Thiers13_Diaries | Survey | 120 | **0.833** | 0.623 | 0.623 | 0.620 | 0.254 | 0.283 |
| **Mean** | | | **0.691** | 0.343 | 0.439 | 0.551 | 0.322 | 0.221 |

## Top-20% overlap

Share of the real top-20% spreaders that each method also puts in its top 20%. Best value per row in **bold**.

| Dataset | Group | n | Ours | Deep6 | W-Eig | Strength | W-Clos | W-Betw |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LH10_Hospital | Face-to-face | 75 | **0.867** | **0.867** | 0.733 | 0.800 | 0.733 | 0.733 |
| LyonSchool_Primary | Face-to-face | 242 | **0.917** | 0.729 | 0.771 | 0.771 | 0.729 | 0.583 |
| Thiers11_HighSchool | Face-to-face | 126 | **0.960** | 0.760 | 0.800 | 0.640 | 0.760 | 0.560 |
| Thiers12_HighSchool | Face-to-face | 180 | **0.889** | 0.806 | 0.417 | **0.889** | 0.639 | 0.639 |
| Thiers13_HighSchool | Face-to-face | 327 | **0.800** | 0.646 | 0.569 | 0.646 | 0.585 | 0.369 |
| InVS13_Workplace | Face-to-face | 92 | **0.833** | **0.833** | 0.444 | 0.722 | 0.667 | 0.556 |
| InVS15_Workplace | Face-to-face | 217 | **0.930** | 0.744 | 0.581 | 0.744 | 0.744 | 0.558 |
| SFHH_Conference | Face-to-face | 403 | **0.900** | 0.738 | 0.588 | 0.812 | 0.700 | 0.675 |
| HT09_Conference | Face-to-face | 113 | **1.000** | 0.864 | 0.364 | 0.773 | 0.727 | 0.636 |
| Infectious_Museum | Face-to-face | 410 | **0.817** | 0.756 | 0.659 | 0.524 | 0.317 | 0.317 |
| Malawi_Village | Face-to-face | 84 | **0.688** | **0.688** | 0.625 | 0.500 | 0.375 | 0.500 |
| Kenya_Households | Face-to-face | 47 | **0.889** | **0.889** | 0.667 | **0.889** | 0.778 | 0.333 |
| Utah_MiddleSchool | Proximity | 591 | **0.881** | 0.508 | 0.805 | 0.780 | 0.644 | 0.542 |
| Utah_ElementarySchool | Proximity | 339 | **0.881** | 0.642 | 0.597 | 0.776 | 0.358 | 0.343 |
| Utah_MiddleSchool_1m | Proximity | 590 | **0.873** | 0.508 | 0.797 | 0.771 | 0.593 | 0.585 |
| Utah_ElementarySchool_1m | Proximity | 339 | **0.806** | 0.597 | 0.627 | 0.746 | 0.478 | 0.343 |
| Copenhagen_BT | Proximity | 675 | **0.926** | 0.607 | 0.585 | 0.756 | 0.719 | 0.578 |
| RealityMining_BT | Proximity | 96 | **0.947** | 0.842 | 0.842 | 0.842 | 0.842 | 0.579 |
| Haggle_Infocom05 | Proximity | 274 | 0.963 | 0.944 | **0.981** | 0.944 | 0.907 | 0.759 |
| LH10_Hospital_Coloc | Co-location | 73 | 0.786 | 0.786 | 0.286 | 0.714 | **0.929** | 0.571 |
| InVS13_Workplace_Coloc | Co-location | 95 | **0.947** | **0.947** | 0.789 | 0.789 | 0.684 | 0.526 |
| InVS15_Workplace_Coloc | Co-location | 219 | **0.860** | 0.767 | 0.651 | 0.814 | 0.651 | 0.628 |
| SFHH_Conference_Coloc | Co-location | 403 | **0.950** | 0.925 | 0.938 | 0.938 | 0.762 | 0.588 |
| LyonSchool_Primary_Coloc | Co-location | 242 | **0.896** | 0.250 | **0.896** | 0.854 | 0.792 | 0.438 |
| Thiers13_HighSchool_Coloc | Co-location | 328 | **0.877** | 0.400 | 0.431 | 0.846 | 0.169 | 0.200 |
| Thiers13_Diaries | Survey | 120 | **0.875** | 0.667 | **0.875** | 0.667 | 0.042 | 0.167 |
| **Mean** | | | **0.883** | 0.720 | 0.666 | 0.767 | 0.628 | 0.512 |

## Kendall tau over all nodes

Agreement with the SIR ranking over the whole network. Best value per row in **bold**.

| Dataset | Group | n | Ours | Deep6 | W-Eig | Strength | W-Clos | W-Betw |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LH10_Hospital | Face-to-face | 75 | **0.941** | 0.776 | 0.858 | 0.869 | 0.810 | 0.530 |
| LyonSchool_Primary | Face-to-face | 242 | **0.890** | 0.641 | 0.787 | 0.744 | 0.688 | 0.450 |
| Thiers11_HighSchool | Face-to-face | 126 | **0.888** | 0.693 | 0.745 | 0.636 | 0.773 | 0.461 |
| Thiers12_HighSchool | Face-to-face | 180 | **0.903** | 0.712 | 0.481 | 0.749 | 0.735 | 0.483 |
| Thiers13_HighSchool | Face-to-face | 327 | **0.855** | 0.675 | 0.445 | 0.673 | 0.679 | 0.340 |
| InVS13_Workplace | Face-to-face | 92 | **0.901** | 0.759 | 0.447 | 0.778 | 0.716 | 0.425 |
| InVS15_Workplace | Face-to-face | 217 | **0.907** | 0.621 | 0.630 | 0.759 | 0.746 | 0.540 |
| SFHH_Conference | Face-to-face | 403 | **0.892** | 0.657 | 0.654 | 0.773 | 0.714 | 0.522 |
| HT09_Conference | Face-to-face | 113 | **0.921** | 0.652 | 0.429 | 0.775 | 0.731 | 0.580 |
| Infectious_Museum | Face-to-face | 410 | **0.840** | 0.743 | 0.558 | 0.547 | 0.614 | 0.244 |
| Malawi_Village | Face-to-face | 84 | **0.758** | 0.666 | 0.518 | 0.439 | 0.519 | 0.212 |
| Kenya_Households | Face-to-face | 47 | **0.917** | 0.841 | 0.769 | 0.809 | 0.804 | 0.474 |
| Utah_MiddleSchool | Proximity | 591 | **0.862** | 0.469 | 0.814 | 0.778 | 0.635 | 0.487 |
| Utah_ElementarySchool | Proximity | 339 | **0.879** | 0.613 | 0.475 | 0.763 | 0.444 | 0.219 |
| Utah_MiddleSchool_1m | Proximity | 590 | **0.882** | 0.532 | 0.832 | 0.773 | 0.651 | 0.560 |
| Utah_ElementarySchool_1m | Proximity | 339 | **0.830** | 0.436 | 0.471 | 0.711 | 0.535 | 0.188 |
| Copenhagen_BT | Proximity | 675 | **0.903** | 0.651 | 0.620 | 0.767 | 0.688 | 0.436 |
| RealityMining_BT | Proximity | 96 | **0.925** | 0.666 | 0.725 | 0.760 | 0.831 | 0.483 |
| Haggle_Infocom05 | Proximity | 274 | 0.900 | 0.657 | 0.883 | **0.931** | 0.892 | 0.496 |
| LH10_Hospital_Coloc | Co-location | 73 | **0.895** | 0.689 | 0.632 | 0.793 | 0.813 | 0.545 |
| InVS13_Workplace_Coloc | Co-location | 95 | **0.935** | 0.849 | 0.764 | 0.853 | 0.676 | 0.488 |
| InVS15_Workplace_Coloc | Co-location | 219 | **0.912** | 0.692 | 0.665 | 0.867 | 0.629 | 0.501 |
| SFHH_Conference_Coloc | Co-location | 403 | **0.894** | 0.815 | 0.872 | 0.878 | 0.678 | 0.412 |
| LyonSchool_Primary_Coloc | Co-location | 242 | 0.879 | 0.409 | **0.885** | 0.845 | 0.797 | 0.435 |
| Thiers13_HighSchool_Coloc | Co-location | 328 | **0.859** | 0.424 | 0.666 | 0.799 | 0.186 | 0.107 |
| Thiers13_Diaries | Survey | 120 | **0.896** | 0.807 | 0.423 | 0.701 | -0.009 | 0.179 |
| **Mean** | | | **0.887** | 0.660 | 0.656 | 0.760 | 0.653 | 0.415 |

## Summary by group (mean Kendall tau, top 20%)

| Group | Networks | Ours | Deep6 | W-Eig | Strength | W-Clos | W-Betw |
|---|---:|---:|---:|---:|---:|---:|---:|
| Face-to-face | 12 | 0.721 | 0.404 | 0.387 | 0.509 | 0.389 | 0.252 |
| Proximity | 7 | 0.662 | 0.363 | 0.499 | 0.565 | 0.234 | 0.193 |
| Co-location | 6 | 0.643 | 0.150 | 0.441 | 0.610 | 0.300 | 0.181 |
| Survey | 1 | 0.833 | 0.623 | 0.623 | 0.620 | 0.254 | 0.283 |

## Head-to-head (Kendall tau, top 20%)

- Networks where ours is best or tied: **22 / 26**
- Ours beats Prev. Deep 6-Ch: 25 / 26 networks
- Ours beats W-Eigenvector: 25 / 26 networks
- Ours beats Strength (W-Deg): 23 / 26 networks
- Ours beats W-Closeness: 24 / 26 networks
- Ours beats W-Betweenness: 26 / 26 networks

## Where the model is weak (ours more than 0.02 below the best other method)

| Dataset | Ours | Best other | Method | Mean outbreak (fraction) | beta |
|---|---:|---:|---|---:|---:|
| InVS13_Workplace | 0.529 | 0.569 | W-Closeness | 0.10 | 0.992 |
| Haggle_Infocom05 | 0.536 | 0.601 | Strength (W-Deg) | 0.10 | 1.000 |
| LH10_Hospital_Coloc | 0.495 | 0.560 | W-Closeness | 0.10 | 0.682 |
| Thiers13_HighSchool_Coloc | 0.221 | 0.277 | Strength (W-Deg) | 0.10 | 0.055 |

## SIR ground-truth settings per network

| Dataset | LCC nodes / total | LCC edges | lambda_max | beta | Mean outbreak (fraction) | Max outbreak (fraction) |
|---|---:|---:|---:|---:|---:|---:|
| LH10_Hospital | 75 / 75 | 1139 | 2.02 | 0.669 | 0.11 | 0.27 |
| LyonSchool_Primary | 242 / 242 | 8317 | 2.09 | 0.592 | 0.10 | 0.23 |
| Thiers11_HighSchool | 126 / 126 | 1709 | 1.06 | 0.982 | 0.10 | 0.23 |
| Thiers12_HighSchool | 180 / 180 | 2220 | 1.59 | 0.986 | 0.11 | 0.24 |
| Thiers13_HighSchool | 327 / 327 | 5818 | 1.07 | 0.993 | 0.11 | 0.22 |
| InVS13_Workplace | 92 / 92 | 755 | 1.08 | 0.992 | 0.10 | 0.22 |
| InVS15_Workplace | 217 / 217 | 4274 | 1.35 | 0.900 | 0.10 | 0.23 |
| SFHH_Conference | 403 / 403 | 9565 | 1.04 | 0.985 | 0.10 | 0.31 |
| HT09_Conference | 113 / 113 | 2196 | 1.01 | 0.981 | 0.10 | 0.26 |
| Infectious_Museum | 410 / 10972 | 2765 | 1.10 | 0.992 | 0.10 | 0.27 |
| Malawi_Village | 84 / 86 | 346 | 1.28 | 1.000 | 0.09 | 0.17 |
| Kenya_Households | 47 / 47 | 504 | 1.54 | 0.784 | 0.10 | 0.23 |
| Utah_MiddleSchool | 591 / 591 | 56867 | 3.90 | 0.291 | 0.09 | 0.16 |
| Utah_ElementarySchool | 339 / 339 | 16546 | 5.37 | 0.579 | 0.10 | 0.20 |
| Utah_MiddleSchool_1m | 590 / 590 | 24432 | 1.72 | 0.588 | 0.10 | 0.21 |
| Utah_ElementarySchool_1m | 339 / 339 | 7835 | 5.99 | 0.896 | 0.11 | 0.22 |
| Copenhagen_BT | 675 / 675 | 34094 | 1.00 | 0.926 | 0.10 | 0.28 |
| RealityMining_BT | 96 / 96 | 2539 | 2.58 | 0.696 | 0.10 | 0.27 |
| Haggle_Infocom05 | 274 / 274 | 2124 | 1.38 | 1.000 | 0.10 | 0.31 |
| LH10_Hospital_Coloc | 73 / 73 | 1381 | 2.27 | 0.682 | 0.10 | 0.21 |
| InVS13_Workplace_Coloc | 95 / 95 | 3915 | 2.97 | 0.466 | 0.10 | 0.21 |
| InVS15_Workplace_Coloc | 219 / 219 | 16725 | 2.29 | 0.566 | 0.09 | 0.21 |
| SFHH_Conference_Coloc | 403 / 403 | 73557 | 5.53 | 0.242 | 0.09 | 0.27 |
| LyonSchool_Primary_Coloc | 242 / 242 | 26594 | 25.09 | 0.049 | 0.09 | 0.16 |
| Thiers13_HighSchool_Coloc | 328 / 328 | 43496 | 28.31 | 0.055 | 0.10 | 0.16 |
| Thiers13_Diaries | 120 / 120 | 348 | 6.48 | 0.335 | 0.10 | 0.17 |

Figures: `fig_tau20_by_network_calibrated.png`, `fig_mean_by_group_calibrated.png`.
