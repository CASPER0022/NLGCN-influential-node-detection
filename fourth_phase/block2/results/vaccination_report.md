# Block 2: Targeted Vaccination on Human Contact Networks

Each method vaccinates its top-ranked k% of people (they can neither catch nor pass on the infection). SIR is then run from a random unvaccinated index case (200 live-edge samples, calibrated regime: 10% mean outbreak without vaccination). The same random samples are used for every method.

**Reduction** = 1 - (outbreak with vaccination) / (outbreak without vaccination). Higher is better. *Random* and *True SIR ranking* (vaccinating the real top spreaders) are references, not competitors. Best competing method per row in **bold**.

## Outbreak reduction with 10% vaccinated

| Dataset | Group | Ours | Deep6 | W-Eig | Strength | W-Clos | W-Betw | Random | SIR (ref.) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LH10_Hospital | Face-to-face | 0.763 | 0.749 | 0.704 | **0.776** | 0.713 | 0.756 | 0.280 | 0.763 |
| LyonSchool_Primary | Face-to-face | 0.698 | 0.692 | 0.476 | **0.759** | 0.626 | 0.689 | 0.435 | 0.673 |
| Thiers11_HighSchool | Face-to-face | 0.667 | **0.711** | 0.550 | 0.502 | 0.632 | 0.618 | 0.378 | 0.662 |
| Thiers12_HighSchool | Face-to-face | 0.764 | 0.748 | 0.405 | **0.766** | 0.499 | 0.729 | 0.371 | 0.730 |
| Thiers13_HighSchool | Face-to-face | 0.715 | 0.696 | 0.427 | **0.769** | 0.640 | 0.679 | 0.440 | 0.617 |
| InVS13_Workplace | Face-to-face | 0.656 | **0.668** | 0.244 | 0.651 | 0.625 | 0.650 | 0.280 | 0.626 |
| InVS15_Workplace | Face-to-face | 0.714 | 0.729 | 0.499 | **0.778** | 0.690 | 0.740 | 0.429 | 0.718 |
| SFHH_Conference | Face-to-face | 0.914 | 0.919 | 0.849 | **0.929** | 0.900 | 0.927 | 0.370 | 0.923 |
| HT09_Conference | Face-to-face | 0.725 | 0.715 | 0.283 | 0.725 | 0.664 | **0.747** | 0.330 | 0.738 |
| Infectious_Museum | Face-to-face | 0.595 | **0.786** | 0.612 | 0.783 | 0.490 | 0.736 | 0.401 | 0.599 |
| Malawi_Village | Face-to-face | 0.283 | 0.333 | 0.223 | **0.378** | 0.213 | 0.358 | 0.213 | 0.279 |
| Kenya_Households | Face-to-face | 0.553 | 0.553 | 0.453 | **0.563** | 0.553 | 0.375 | 0.217 | 0.563 |
| Utah_MiddleSchool | Proximity | 0.802 | 0.692 | 0.741 | **0.838** | 0.802 | 0.803 | 0.575 | 0.802 |
| Utah_ElementarySchool | Proximity | 0.512 | 0.548 | 0.350 | **0.726** | 0.472 | 0.621 | 0.501 | 0.610 |
| Utah_MiddleSchool_1m | Proximity | 0.833 | 0.730 | 0.750 | **0.871** | 0.785 | 0.820 | 0.578 | 0.820 |
| Utah_ElementarySchool_1m | Proximity | 0.511 | 0.421 | 0.424 | **0.708** | 0.564 | 0.595 | 0.498 | 0.520 |
| Copenhagen_BT | Proximity | 0.932 | 0.902 | 0.766 | **0.934** | 0.821 | 0.905 | 0.507 | 0.927 |
| RealityMining_BT | Proximity | 0.667 | **0.698** | 0.667 | 0.697 | 0.659 | 0.639 | 0.275 | 0.667 |
| Haggle_Infocom05 | Proximity | 0.831 | 0.802 | 0.805 | **0.834** | 0.800 | 0.818 | 0.150 | 0.768 |
| LH10_Hospital_Coloc | Co-location | 0.502 | **0.565** | 0.174 | 0.473 | 0.537 | 0.564 | 0.279 | 0.572 |
| InVS13_Workplace_Coloc | Co-location | 0.656 | 0.557 | 0.425 | **0.702** | 0.543 | 0.668 | 0.313 | 0.642 |
| InVS15_Workplace_Coloc | Co-location | 0.768 | 0.767 | 0.640 | **0.782** | 0.666 | 0.750 | 0.424 | 0.763 |
| SFHH_Conference_Coloc | Co-location | 0.904 | 0.899 | 0.897 | **0.907** | 0.876 | 0.831 | 0.449 | 0.905 |
| LyonSchool_Primary_Coloc | Co-location | 0.719 | 0.514 | 0.711 | **0.725** | 0.721 | 0.680 | 0.444 | 0.726 |
| Thiers13_HighSchool_Coloc | Co-location | 0.504 | 0.554 | 0.347 | **0.671** | 0.538 | 0.545 | 0.477 | 0.587 |
| Thiers13_Diaries | Survey | 0.224 | 0.437 | 0.222 | **0.569** | 0.233 | 0.342 | 0.271 | 0.348 |
| **Mean** | | 0.670 | 0.669 | 0.525 | **0.724** | 0.626 | 0.676 | 0.380 | 0.675 |

## Outbreak reduction with 5% vaccinated

| Dataset | Group | Ours | Deep6 | W-Eig | Strength | W-Clos | W-Betw | Random | SIR (ref.) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LH10_Hospital | Face-to-face | 0.602 | **0.615** | 0.602 | 0.602 | 0.565 | 0.578 | 0.169 | 0.602 |
| LyonSchool_Primary | Face-to-face | 0.504 | 0.542 | 0.447 | **0.590** | 0.517 | 0.554 | 0.228 | 0.509 |
| Thiers11_HighSchool | Face-to-face | 0.523 | **0.524** | 0.381 | 0.373 | 0.453 | 0.404 | 0.194 | 0.497 |
| Thiers12_HighSchool | Face-to-face | 0.518 | 0.530 | 0.344 | **0.591** | 0.323 | 0.505 | 0.198 | 0.536 |
| Thiers13_HighSchool | Face-to-face | 0.492 | **0.535** | 0.326 | 0.496 | 0.500 | 0.506 | 0.261 | 0.491 |
| InVS13_Workplace | Face-to-face | **0.529** | 0.461 | 0.217 | 0.502 | 0.494 | 0.513 | 0.148 | 0.440 |
| InVS15_Workplace | Face-to-face | 0.617 | 0.549 | 0.329 | **0.670** | 0.472 | 0.571 | 0.248 | 0.589 |
| SFHH_Conference | Face-to-face | 0.831 | 0.845 | 0.644 | 0.838 | 0.809 | **0.867** | 0.199 | 0.850 |
| HT09_Conference | Face-to-face | **0.633** | 0.538 | 0.184 | 0.563 | 0.521 | 0.613 | 0.196 | 0.623 |
| Infectious_Museum | Face-to-face | 0.554 | **0.643** | 0.468 | 0.583 | 0.390 | 0.610 | 0.222 | 0.556 |
| Malawi_Village | Face-to-face | 0.225 | 0.277 | 0.224 | **0.279** | 0.228 | 0.228 | 0.121 | 0.225 |
| Kenya_Households | Face-to-face | **0.338** | 0.312 | 0.275 | **0.338** | **0.338** | 0.273 | 0.103 | 0.322 |
| Utah_MiddleSchool | Proximity | 0.624 | 0.497 | 0.515 | **0.671** | 0.618 | 0.614 | 0.346 | 0.636 |
| Utah_ElementarySchool | Proximity | 0.314 | 0.349 | 0.209 | **0.491** | 0.340 | 0.434 | 0.309 | 0.414 |
| Utah_MiddleSchool_1m | Proximity | 0.618 | 0.522 | 0.459 | **0.728** | 0.580 | 0.633 | 0.351 | 0.623 |
| Utah_ElementarySchool_1m | Proximity | 0.335 | 0.261 | 0.230 | **0.469** | 0.392 | 0.445 | 0.280 | 0.329 |
| Copenhagen_BT | Proximity | 0.799 | 0.716 | 0.531 | **0.832** | 0.683 | 0.757 | 0.282 | 0.806 |
| RealityMining_BT | Proximity | 0.553 | 0.546 | 0.553 | **0.601** | 0.560 | 0.522 | 0.146 | 0.553 |
| Haggle_Infocom05 | Proximity | 0.473 | 0.398 | 0.407 | 0.429 | 0.479 | **0.511** | 0.082 | 0.425 |
| LH10_Hospital_Coloc | Co-location | 0.343 | 0.409 | 0.186 | 0.343 | 0.408 | **0.426** | 0.165 | 0.444 |
| InVS13_Workplace_Coloc | Co-location | 0.500 | 0.331 | 0.344 | **0.535** | 0.425 | 0.442 | 0.182 | 0.535 |
| InVS15_Workplace_Coloc | Co-location | 0.571 | 0.576 | 0.355 | **0.599** | 0.411 | 0.537 | 0.255 | 0.549 |
| SFHH_Conference_Coloc | Co-location | 0.821 | 0.802 | 0.821 | **0.823** | 0.763 | 0.738 | 0.238 | 0.825 |
| LyonSchool_Primary_Coloc | Co-location | 0.513 | 0.253 | 0.507 | **0.532** | 0.515 | 0.460 | 0.256 | 0.528 |
| Thiers13_HighSchool_Coloc | Co-location | 0.315 | 0.346 | 0.315 | **0.411** | 0.374 | 0.398 | 0.283 | 0.428 |
| Thiers13_Diaries | Survey | 0.205 | 0.294 | 0.205 | **0.351** | 0.192 | 0.146 | 0.157 | 0.205 |
| **Mean** | | 0.513 | 0.487 | 0.388 | **0.548** | 0.475 | 0.511 | 0.216 | 0.521 |

## Peak reduction with 10% vaccinated

| Dataset | Group | Ours | Deep6 | W-Eig | Strength | W-Clos | W-Betw | Random | SIR (ref.) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LH10_Hospital | Face-to-face | 0.563 | 0.555 | 0.508 | **0.577** | 0.516 | 0.557 | 0.226 | 0.563 |
| LyonSchool_Primary | Face-to-face | 0.488 | 0.486 | 0.316 | **0.552** | 0.440 | 0.492 | 0.266 | 0.469 |
| Thiers11_HighSchool | Face-to-face | 0.482 | **0.516** | 0.390 | 0.341 | 0.445 | 0.420 | 0.252 | 0.474 |
| Thiers12_HighSchool | Face-to-face | 0.569 | 0.545 | 0.263 | **0.580** | 0.331 | 0.545 | 0.212 | 0.524 |
| Thiers13_HighSchool | Face-to-face | 0.548 | 0.511 | 0.275 | **0.598** | 0.479 | 0.504 | 0.334 | 0.446 |
| InVS13_Workplace | Face-to-face | 0.452 | **0.458** | 0.148 | 0.442 | 0.407 | 0.426 | 0.160 | 0.407 |
| InVS15_Workplace | Face-to-face | 0.529 | 0.549 | 0.344 | **0.590** | 0.512 | 0.556 | 0.322 | 0.536 |
| SFHH_Conference | Face-to-face | 0.802 | 0.806 | 0.745 | 0.818 | 0.789 | **0.820** | 0.344 | 0.810 |
| HT09_Conference | Face-to-face | 0.537 | 0.525 | 0.195 | 0.539 | 0.488 | **0.555** | 0.253 | 0.548 |
| Infectious_Museum | Face-to-face | 0.468 | **0.622** | 0.456 | 0.622 | 0.313 | 0.573 | 0.298 | 0.471 |
| Malawi_Village | Face-to-face | 0.208 | 0.239 | 0.154 | **0.288** | 0.118 | 0.220 | 0.146 | 0.200 |
| Kenya_Households | Face-to-face | 0.324 | 0.324 | 0.264 | **0.331** | 0.324 | 0.189 | 0.113 | 0.331 |
| Utah_MiddleSchool | Proximity | 0.628 | 0.540 | 0.573 | **0.659** | 0.623 | 0.639 | 0.460 | 0.629 |
| Utah_ElementarySchool | Proximity | 0.353 | 0.375 | 0.243 | **0.525** | 0.317 | 0.457 | 0.334 | 0.428 |
| Utah_MiddleSchool_1m | Proximity | 0.672 | 0.570 | 0.597 | **0.710** | 0.624 | 0.662 | 0.469 | 0.657 |
| Utah_ElementarySchool_1m | Proximity | 0.395 | 0.338 | 0.331 | **0.557** | 0.402 | 0.400 | 0.368 | 0.415 |
| Copenhagen_BT | Proximity | 0.811 | 0.766 | 0.656 | **0.813** | 0.696 | 0.776 | 0.398 | 0.798 |
| RealityMining_BT | Proximity | 0.503 | 0.523 | 0.503 | **0.528** | 0.496 | 0.467 | 0.189 | 0.503 |
| Haggle_Infocom05 | Proximity | **0.847** | 0.825 | 0.823 | 0.844 | 0.823 | 0.829 | 0.129 | 0.774 |
| LH10_Hospital_Coloc | Co-location | 0.332 | **0.361** | 0.102 | 0.311 | 0.339 | 0.359 | 0.181 | 0.354 |
| InVS13_Workplace_Coloc | Co-location | 0.416 | 0.336 | 0.251 | **0.453** | 0.333 | 0.431 | 0.169 | 0.407 |
| InVS15_Workplace_Coloc | Co-location | 0.584 | 0.578 | 0.471 | **0.599** | 0.490 | 0.567 | 0.328 | 0.574 |
| SFHH_Conference_Coloc | Co-location | 0.766 | 0.761 | 0.761 | **0.768** | 0.740 | 0.699 | 0.333 | 0.768 |
| LyonSchool_Primary_Coloc | Co-location | 0.515 | 0.351 | 0.505 | 0.521 | **0.524** | 0.492 | 0.249 | 0.517 |
| Thiers13_HighSchool_Coloc | Co-location | 0.386 | 0.398 | 0.262 | **0.511** | 0.350 | 0.372 | 0.325 | 0.432 |
| Thiers13_Diaries | Survey | 0.198 | 0.331 | 0.190 | **0.437** | 0.142 | 0.232 | 0.166 | 0.282 |
| **Mean** | | 0.514 | 0.507 | 0.397 | **0.558** | 0.464 | 0.509 | 0.270 | 0.512 |

## Mean outbreak reduction vs. share vaccinated (mean over networks)

| % vaccinated | Ours | Deep6 | W-Eig | Strength | W-Clos | W-Betw | Random | SIR (ref.) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.5% | 0.337 | 0.310 | 0.261 | 0.334 | 0.326 | 0.333 | 0.111 | 0.350 |
| 5.0% | 0.513 | 0.487 | 0.388 | 0.548 | 0.475 | 0.511 | 0.216 | 0.521 |
| 10.0% | 0.670 | 0.669 | 0.525 | 0.724 | 0.626 | 0.676 | 0.380 | 0.675 |
| 15.0% | 0.770 | 0.758 | 0.625 | 0.800 | 0.712 | 0.769 | 0.501 | 0.756 |
| 20.0% | 0.817 | 0.812 | 0.675 | 0.840 | 0.759 | 0.815 | 0.595 | 0.799 |

## By group (outbreak reduction, 10% vaccinated)

| Group | Networks | Ours | Deep6 | W-Eig | Strength | W-Clos | W-Betw | Random | SIR (ref.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Proximity | 7 | 0.727 | 0.685 | 0.643 | 0.801 | 0.700 | 0.743 | 0.441 | 0.731 |
| Face-to-face | 12 | 0.671 | 0.692 | 0.477 | 0.698 | 0.604 | 0.667 | 0.345 | 0.658 |
| Co-location | 6 | 0.675 | 0.643 | 0.532 | 0.710 | 0.647 | 0.673 | 0.397 | 0.699 |
| Survey | 1 | 0.224 | 0.437 | 0.222 | 0.569 | 0.233 | 0.342 | 0.271 | 0.348 |

## Head-to-head (outbreak reduction, 10% vaccinated)

- Ours is the best (or tied) method on **0 / 26** networks
- Ours beats Prev. Deep 6-Ch: 14 / 26
- Ours beats W-Eigenvector: 24 / 26
- Ours beats Strength (W-Deg): 4 / 26
- Ours beats W-Closeness: 20 / 26
- Ours beats W-Betweenness: 14 / 26
- Ours beats random vaccination: 25 / 26
- Ours reaches 99% of the reduction obtained by vaccinating the true top spreaders

Figures: `fig_vaccination_curves.png`, `fig_vaccination_by_group.png`, `fig_reduction10_by_network.png`.
