# Block 1: Zero-Shot Results on Human Contact Networks (original SIR regime)

The Third-Phase Hybrid Global-Local WNLGCN (3-seed ensemble) was applied to the contact networks **without any retraining**. Ground truth = expected SIR outbreak size from each node (Third-Phase recipe: beta = min(1.5 / lambda_max, 0.9), p = 1 - (1 - beta)^w_norm, 500 runs).

## Kendall tau inside the true top 20%

How well each method orders the real top spreaders. Best value per row in **bold**.

| Dataset | Group | n | Ours | Deep6 | W-Eig | Strength | W-Clos | W-Betw |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LH10_Hospital | Face-to-face | 75 | **0.810** | 0.486 | 0.619 | **0.810** | 0.524 | 0.517 |
| LyonSchool_Primary | Face-to-face | 242 | 0.670 | 0.285 | 0.441 | **0.690** | 0.420 | 0.420 |
| Thiers11_HighSchool | Face-to-face | 126 | **0.807** | 0.260 | 0.433 | 0.595 | 0.173 | 0.147 |
| Thiers12_HighSchool | Face-to-face | 180 | **0.743** | 0.505 | 0.330 | 0.438 | 0.244 | 0.111 |
| Thiers13_HighSchool | Face-to-face | 327 | **0.645** | 0.213 | 0.159 | 0.323 | 0.225 | 0.060 |
| InVS13_Workplace | Face-to-face | 92 | 0.529 | 0.477 | 0.216 | **0.542** | 0.333 | 0.296 |
| InVS15_Workplace | Face-to-face | 217 | **0.813** | 0.479 | 0.359 | 0.575 | 0.618 | 0.465 |
| SFHH_Conference | Face-to-face | 403 | **0.847** | 0.571 | 0.560 | 0.519 | 0.578 | 0.430 |
| HT09_Conference | Face-to-face | 113 | **0.790** | 0.226 | 0.434 | 0.607 | 0.547 | 0.599 |
| Infectious_Museum | Face-to-face | 410 | **0.746** | 0.435 | 0.142 | 0.304 | 0.095 | -0.055 |
| Malawi_Village | Face-to-face | 84 | 0.900 | 0.733 | **0.917** | 0.367 | 0.083 | -0.193 |
| Kenya_Households | Face-to-face | 47 | 0.667 | 0.722 | 0.722 | **0.778** | 0.556 | 0.085 |
| Utah_MiddleSchool | Proximity | 591 | **0.686** | 0.259 | 0.512 | 0.640 | 0.295 | 0.232 |
| Utah_ElementarySchool | Proximity | 339 | **0.759** | 0.519 | 0.492 | 0.397 | -0.178 | -0.059 |
| Utah_MiddleSchool_1m | Proximity | 590 | 0.439 | 0.073 | 0.337 | **0.640** | 0.250 | 0.366 |
| Utah_ElementarySchool_1m | Proximity | 339 | **0.593** | 0.337 | 0.319 | 0.530 | 0.034 | 0.037 |
| Copenhagen_BT | Proximity | 675 | **0.621** | 0.271 | 0.280 | 0.448 | 0.239 | 0.198 |
| RealityMining_BT | Proximity | 96 | **0.895** | 0.556 | 0.801 | 0.754 | 0.579 | 0.373 |
| Haggle_Infocom05 | Proximity | 274 | **0.913** | 0.816 | 0.782 | 0.890 | 0.674 | 0.510 |
| LH10_Hospital_Coloc | Co-location | 73 | 0.319 | -0.011 | -0.011 | 0.319 | **0.495** | **0.495** |
| InVS13_Workplace_Coloc | Co-location | 95 | **0.848** | 0.427 | 0.567 | 0.544 | -0.041 | -0.141 |
| InVS15_Workplace_Coloc | Co-location | 219 | **0.798** | 0.599 | 0.546 | 0.739 | 0.251 | 0.300 |
| SFHH_Conference_Coloc | Co-location | 403 | **0.835** | 0.713 | 0.760 | 0.787 | 0.556 | 0.223 |
| LyonSchool_Primary_Coloc | Co-location | 242 | 0.668 | -0.472 | 0.638 | **0.715** | 0.397 | 0.104 |
| Thiers13_HighSchool_Coloc | Co-location | 328 | 0.288 | -0.082 | 0.193 | **0.394** | 0.043 | 0.011 |
| Thiers13_Diaries | Survey | 120 | **0.826** | 0.565 | 0.667 | 0.612 | 0.210 | 0.210 |
| **Mean** | | | **0.710** | 0.383 | 0.470 | 0.575 | 0.315 | 0.221 |

## Top-20% overlap

Share of the real top-20% spreaders that each method also puts in its top 20%. Best value per row in **bold**.

| Dataset | Group | n | Ours | Deep6 | W-Eig | Strength | W-Clos | W-Betw |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LH10_Hospital | Face-to-face | 75 | 0.867 | **0.933** | 0.667 | 0.867 | 0.667 | 0.800 |
| LyonSchool_Primary | Face-to-face | 242 | **0.875** | 0.708 | 0.708 | 0.792 | 0.688 | 0.562 |
| Thiers11_HighSchool | Face-to-face | 126 | **1.000** | 0.760 | 0.800 | 0.640 | 0.760 | 0.560 |
| Thiers12_HighSchool | Face-to-face | 180 | **0.972** | 0.806 | 0.417 | 0.861 | 0.611 | 0.611 |
| Thiers13_HighSchool | Face-to-face | 327 | **0.938** | 0.646 | 0.462 | 0.754 | 0.615 | 0.415 |
| InVS13_Workplace | Face-to-face | 92 | **0.944** | 0.889 | 0.556 | 0.833 | 0.556 | 0.556 |
| InVS15_Workplace | Face-to-face | 217 | **0.907** | 0.721 | 0.581 | 0.721 | 0.744 | 0.535 |
| SFHH_Conference | Face-to-face | 403 | **0.912** | 0.700 | 0.650 | 0.787 | 0.750 | 0.650 |
| HT09_Conference | Face-to-face | 113 | **0.955** | 0.864 | 0.364 | 0.773 | 0.727 | 0.636 |
| Infectious_Museum | Face-to-face | 410 | **0.890** | 0.829 | 0.610 | 0.561 | 0.354 | 0.305 |
| Malawi_Village | Face-to-face | 84 | **0.938** | 0.750 | 0.688 | 0.625 | 0.188 | 0.375 |
| Kenya_Households | Face-to-face | 47 | **1.000** | 0.889 | 0.667 | 0.889 | 0.778 | 0.333 |
| Utah_MiddleSchool | Proximity | 591 | **0.864** | 0.432 | 0.754 | 0.856 | 0.678 | 0.576 |
| Utah_ElementarySchool | Proximity | 339 | **0.836** | 0.597 | 0.612 | 0.806 | 0.224 | 0.299 |
| Utah_MiddleSchool_1m | Proximity | 590 | 0.805 | 0.458 | 0.712 | **0.864** | 0.593 | 0.610 |
| Utah_ElementarySchool_1m | Proximity | 339 | **0.866** | 0.612 | 0.612 | 0.851 | 0.403 | 0.299 |
| Copenhagen_BT | Proximity | 675 | **0.874** | 0.622 | 0.593 | 0.748 | 0.719 | 0.578 |
| RealityMining_BT | Proximity | 96 | **0.947** | 0.842 | 0.895 | 0.789 | 0.842 | 0.526 |
| Haggle_Infocom05 | Proximity | 274 | **0.981** | 0.944 | 0.963 | 0.944 | 0.889 | 0.778 |
| LH10_Hospital_Coloc | Co-location | 73 | **0.857** | 0.714 | 0.357 | 0.786 | **0.857** | 0.571 |
| InVS13_Workplace_Coloc | Co-location | 95 | **1.000** | **1.000** | 0.789 | 0.842 | 0.684 | 0.579 |
| InVS15_Workplace_Coloc | Co-location | 219 | **0.884** | 0.744 | 0.605 | 0.860 | 0.651 | 0.651 |
| SFHH_Conference_Coloc | Co-location | 403 | 0.938 | 0.925 | **0.950** | 0.925 | 0.775 | 0.600 |
| LyonSchool_Primary_Coloc | Co-location | 242 | **0.896** | 0.188 | **0.896** | **0.896** | 0.854 | 0.479 |
| Thiers13_HighSchool_Coloc | Co-location | 328 | **0.862** | 0.400 | 0.415 | 0.846 | 0.185 | 0.200 |
| Thiers13_Diaries | Survey | 120 | **0.917** | 0.667 | 0.875 | 0.667 | 0.042 | 0.167 |
| **Mean** | | | **0.912** | 0.717 | 0.661 | 0.799 | 0.609 | 0.510 |

## Kendall tau over all nodes

Agreement with the SIR ranking over the whole network. Best value per row in **bold**.

| Dataset | Group | n | Ours | Deep6 | W-Eig | Strength | W-Clos | W-Betw |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LH10_Hospital | Face-to-face | 75 | **0.929** | 0.774 | 0.846 | 0.882 | 0.805 | 0.537 |
| LyonSchool_Primary | Face-to-face | 242 | **0.903** | 0.633 | 0.745 | 0.812 | 0.655 | 0.483 |
| Thiers11_HighSchool | Face-to-face | 126 | **0.927** | 0.710 | 0.672 | 0.722 | 0.736 | 0.524 |
| Thiers12_HighSchool | Face-to-face | 180 | **0.918** | 0.725 | 0.479 | 0.775 | 0.719 | 0.503 |
| Thiers13_HighSchool | Face-to-face | 327 | **0.907** | 0.673 | 0.422 | 0.745 | 0.675 | 0.376 |
| InVS13_Workplace | Face-to-face | 92 | **0.881** | 0.794 | 0.434 | 0.837 | 0.696 | 0.453 |
| InVS15_Workplace | Face-to-face | 217 | **0.907** | 0.622 | 0.629 | 0.760 | 0.745 | 0.540 |
| SFHH_Conference | Face-to-face | 403 | **0.883** | 0.658 | 0.687 | 0.733 | 0.716 | 0.506 |
| HT09_Conference | Face-to-face | 113 | **0.903** | 0.654 | 0.446 | 0.788 | 0.713 | 0.577 |
| Infectious_Museum | Face-to-face | 410 | **0.889** | 0.788 | 0.513 | 0.660 | 0.535 | 0.282 |
| Malawi_Village | Face-to-face | 84 | **0.855** | 0.772 | 0.349 | 0.729 | 0.275 | 0.166 |
| Kenya_Households | Face-to-face | 47 | **0.907** | 0.859 | 0.769 | 0.821 | 0.826 | 0.484 |
| Utah_MiddleSchool | Proximity | 591 | **0.885** | 0.448 | 0.777 | 0.846 | 0.642 | 0.527 |
| Utah_ElementarySchool | Proximity | 339 | **0.898** | 0.607 | 0.498 | 0.812 | 0.366 | 0.238 |
| Utah_MiddleSchool_1m | Proximity | 590 | 0.836 | 0.484 | 0.748 | **0.886** | 0.636 | 0.625 |
| Utah_ElementarySchool_1m | Proximity | 339 | **0.849** | 0.435 | 0.462 | 0.830 | 0.433 | 0.250 |
| Copenhagen_BT | Proximity | 675 | **0.883** | 0.653 | 0.616 | 0.750 | 0.687 | 0.432 |
| RealityMining_BT | Proximity | 96 | **0.920** | 0.665 | 0.730 | 0.747 | 0.816 | 0.475 |
| Haggle_Infocom05 | Proximity | 274 | 0.841 | 0.692 | 0.824 | **0.850** | 0.818 | 0.536 |
| LH10_Hospital_Coloc | Co-location | 73 | **0.901** | 0.691 | 0.636 | 0.798 | 0.805 | 0.545 |
| InVS13_Workplace_Coloc | Co-location | 95 | **0.947** | 0.852 | 0.754 | 0.867 | 0.665 | 0.492 |
| InVS15_Workplace_Coloc | Co-location | 219 | **0.918** | 0.683 | 0.642 | 0.890 | 0.632 | 0.519 |
| SFHH_Conference_Coloc | Co-location | 403 | **0.905** | 0.814 | 0.877 | 0.893 | 0.676 | 0.412 |
| LyonSchool_Primary_Coloc | Co-location | 242 | **0.898** | 0.424 | 0.890 | 0.873 | 0.818 | 0.445 |
| Thiers13_HighSchool_Coloc | Co-location | 328 | **0.860** | 0.421 | 0.674 | 0.804 | 0.191 | 0.105 |
| Thiers13_Diaries | Survey | 120 | **0.921** | 0.818 | 0.423 | 0.709 | -0.028 | 0.180 |
| **Mean** | | | **0.895** | 0.667 | 0.636 | 0.801 | 0.625 | 0.431 |

## Summary by group (mean Kendall tau, top 20%)

| Group | Networks | Ours | Deep6 | W-Eig | Strength | W-Clos | W-Betw |
|---|---:|---:|---:|---:|---:|---:|---:|
| Face-to-face | 12 | 0.747 | 0.449 | 0.444 | 0.546 | 0.366 | 0.240 |
| Proximity | 7 | 0.701 | 0.404 | 0.503 | 0.614 | 0.270 | 0.237 |
| Co-location | 6 | 0.626 | 0.196 | 0.449 | 0.583 | 0.284 | 0.165 |
| Survey | 1 | 0.826 | 0.565 | 0.667 | 0.612 | 0.210 | 0.210 |

## Head-to-head (Kendall tau, top 20%)

- Networks where ours is best or tied: **18 / 26**
- Ours beats Prev. Deep 6-Ch: 25 / 26 networks
- Ours beats W-Eigenvector: 24 / 26 networks
- Ours beats Strength (W-Deg): 18 / 26 networks
- Ours beats W-Closeness: 25 / 26 networks
- Ours beats W-Betweenness: 25 / 26 networks

## Where the model is weak (ours more than 0.02 below the best other method)

| Dataset | Ours | Best other | Method | Mean outbreak (fraction) | beta |
|---|---:|---:|---|---:|---:|
| LyonSchool_Primary | 0.670 | 0.690 | Strength (W-Deg) | 0.34 | 0.716 |
| Kenya_Households | 0.667 | 0.778 | Strength (W-Deg) | 0.20 | 0.900 |
| Utah_MiddleSchool_1m | 0.439 | 0.640 | Strength (W-Deg) | 0.77 | 0.873 |
| LH10_Hospital_Coloc | 0.319 | 0.495 | W-Closeness | 0.09 | 0.661 |
| LyonSchool_Primary_Coloc | 0.668 | 0.715 | Strength (W-Deg) | 0.23 | 0.060 |
| Thiers13_HighSchool_Coloc | 0.288 | 0.394 | Strength (W-Deg) | 0.08 | 0.053 |

## SIR ground-truth settings per network

| Dataset | LCC nodes / total | LCC edges | lambda_max | beta | Mean outbreak (fraction) | Max outbreak (fraction) |
|---|---:|---:|---:|---:|---:|---:|
| LH10_Hospital | 75 / 75 | 1139 | 2.02 | 0.742 | 0.16 | 0.36 |
| LyonSchool_Primary | 242 / 242 | 8317 | 2.09 | 0.716 | 0.34 | 0.55 |
| Thiers11_HighSchool | 126 / 126 | 1709 | 1.06 | 0.900 | 0.03 | 0.06 |
| Thiers12_HighSchool | 180 / 180 | 2220 | 1.59 | 0.900 | 0.02 | 0.07 |
| Thiers13_HighSchool | 327 / 327 | 5818 | 1.07 | 0.900 | 0.01 | 0.03 |
| InVS13_Workplace | 92 / 92 | 755 | 1.08 | 0.900 | 0.03 | 0.06 |
| InVS15_Workplace | 217 / 217 | 4274 | 1.35 | 0.900 | 0.10 | 0.23 |
| SFHH_Conference | 403 / 403 | 9565 | 1.04 | 0.900 | 0.01 | 0.07 |
| HT09_Conference | 113 / 113 | 2196 | 1.01 | 0.900 | 0.03 | 0.08 |
| Infectious_Museum | 410 / 10972 | 2765 | 1.10 | 0.900 | 0.01 | 0.07 |
| Malawi_Village | 84 / 86 | 346 | 1.28 | 0.900 | 0.03 | 0.07 |
| Kenya_Households | 47 / 47 | 504 | 1.54 | 0.900 | 0.20 | 0.40 |
| Utah_MiddleSchool | 591 / 591 | 56867 | 3.90 | 0.384 | 0.41 | 0.59 |
| Utah_ElementarySchool | 339 / 339 | 16546 | 5.37 | 0.280 | 0.01 | 0.02 |
| Utah_MiddleSchool_1m | 590 / 590 | 24432 | 1.72 | 0.873 | 0.77 | 0.88 |
| Utah_ElementarySchool_1m | 339 / 339 | 7835 | 5.99 | 0.250 | 0.00 | 0.02 |
| Copenhagen_BT | 675 / 675 | 34094 | 1.00 | 0.900 | 0.05 | 0.17 |
| RealityMining_BT | 96 / 96 | 2539 | 2.58 | 0.581 | 0.05 | 0.17 |
| Haggle_Infocom05 | 274 / 274 | 2124 | 1.38 | 0.900 | 0.02 | 0.11 |
| LH10_Hospital_Coloc | 73 / 73 | 1381 | 2.27 | 0.661 | 0.09 | 0.18 |
| InVS13_Workplace_Coloc | 95 / 95 | 3915 | 2.97 | 0.505 | 0.14 | 0.29 |
| InVS15_Workplace_Coloc | 219 / 219 | 16725 | 2.29 | 0.656 | 0.23 | 0.43 |
| SFHH_Conference_Coloc | 403 / 403 | 73557 | 5.53 | 0.271 | 0.16 | 0.39 |
| LyonSchool_Primary_Coloc | 242 / 242 | 26594 | 25.09 | 0.060 | 0.23 | 0.36 |
| Thiers13_HighSchool_Coloc | 328 / 328 | 43496 | 28.31 | 0.053 | 0.08 | 0.14 |
| Thiers13_Diaries | 120 / 120 | 348 | 6.48 | 0.231 | 0.04 | 0.09 |

Figures: `fig_tau20_by_network_original.png`, `fig_mean_by_group_original.png`.
