# Block 1 Summary: SIR Ground Truth and Zero-Shot Baseline (Nov 14 - Nov 17)

## What was done
1. **Networks:** 26 of the 27 contact networks were used. `SouthAfrica_Households` was left out, because it is
   88 separate households (its largest connected part is 8 people). Each network was reduced to its largest
   connected component (LCC); for `Infectious_Museum` that is 410 of 10,972 visitors. All weights were treated
   as `positive` (more contact = stronger link).
2. **SIR ground truth:** the same process as Third_Phase. Edge weights are divided by the maximum weight, each
   edge transmits with p = 1 - (1 - beta)^w_norm, and a node's score is its expected final outbreak size over
   500 runs.
3. **Features:** local 6-channel patches and global descriptors, both computed with the unchanged Third_Phase code.
4. **Zero-shot evaluation:** the Third_Phase Hybrid G-L ensemble (seeds 42-44), with **no retraining**, compared
   against the previous Deep 6-Ch model, W-Eigenvector, Strength, W-Closeness and W-Betweenness.

## Faster SIR, same process
The Third_Phase SIR loop is pure Python and too slow on these dense networks. Block 1 instead uses the
equivalent **live-edge** method (Kempe et al., 2003): in each run, each directed edge is kept with
probability p, and the outbreak from a seed is everything reachable from it. This estimates the same quantity.
`validate_sir.py` checks it against the original code:

| Check | Kendall tau | Pearson r |
|---|---:|---:|
| karate: original code vs original code (two random seeds) = noise floor | 0.815 | 0.981 |
| karate: fast vs original | 0.790 | 0.982 |
| LH10_Hospital: fast vs original | 0.951 | 0.995 |

The fast estimator differs from the original code no more than the original code differs from itself.
The mean outbreak size is identical (0.578 on karate).

## Main finding: the Third_Phase beta rule does not transfer to contact networks
The Third_Phase rule is beta = min(1.5 / lambda_max, 0.9), with weights divided by the maximum weight. Contact
times are very heavy-tailed: the median edge is about 60 s, while the maximum is up to 310,000 s. So:
- **13 of 26 networks hit the 0.9 cap**, and most edges end up with a transmission probability of about 0.002.
- **The outbreak size swings wildly:** from 0.4% of people (Utah_ElementarySchool_1m) to 77%
  (Utah_MiddleSchool_1m). At both extremes most nodes have near-identical SIR scores, so rankings are mostly
  scored against noise.

**Fix (calibrated regime):** keep the same transmission formula, written as p = 1 - exp(-tau * w_norm) with
tau = -ln(1 - beta) and no cap. tau is chosen per network (bisection) so that the **mean outbreak is 10% of the
population**. The model's beta-dependent global descriptors (transmission eigenvector, dynamics-sensitive
centrality, mean-field outbreak estimate) are recomputed with the same tau, exactly as during training. With this, every network
ends up at an average outbreak of 9-11%.

## Results (26 networks, no retraining)

| Mean over 26 networks | Ours | Strength | W-Eig | Prev. Deep 6-Ch | W-Clos | W-Betw |
|---|---:|---:|---:|---:|---:|---:|
| tau, top 20%: original regime | **0.710** | 0.575 | 0.470 | 0.383 | 0.315 | 0.221 |
| tau, top 20%: calibrated regime | **0.691** | 0.551 | 0.439 | 0.343 | 0.322 | 0.221 |
| Top-20% overlap: calibrated | **0.883** | 0.767 | 0.666 | 0.720 | 0.628 | 0.512 |
| tau, all nodes: calibrated | **0.887** | 0.760 | 0.656 | 0.660 | 0.653 | 0.415 |

Head-to-head, calibrated regime (tau, top 20%):
- Ours is the best method on **22 of 26** networks (18 of 26 in the original regime).
- Ours beats Strength on 23 of 26, W-Eigenvector on 25 of 26, W-Closeness on 24 of 26 and W-Betweenness on
  26 of 26 networks.

By group, calibrated regime (tau, top 20%):

| Group | Networks | Ours | Strength | W-Eigenvector |
|---|---:|---:|---:|---:|
| Face-to-face | 12 | **0.721** | 0.509 | 0.387 |
| Proximity | 7 | **0.662** | 0.565 | 0.499 |
| Co-location | 6 | **0.643** | 0.610 | 0.441 |
| Survey | 1 | **0.833** | 0.620 | 0.623 |

## What this means
- **The model transfers well without retraining.** It is the best method on average for every metric and in
  every group, and it is far ahead on face-to-face networks, the main data source for the application
  (0.72 vs 0.51 for Strength).
- **The previous Deep 6-Ch model does not transfer** (0.34-0.38), which confirms the value of the hybrid global
  branch.
- **Calibrated labels are the more reliable test.** In the original regime, near-dead or near-total outbreaks make
  many nodes tie, so the scores there depend partly on noise.
- **Weak spots for Blocks 3-4.** These networks are 0.02 or more behind the best other method (calibrated):

| Network | Ours | Best other method |
|---|---:|---|
| Thiers13_HighSchool_Coloc | 0.221 | Strength 0.277 |
| LH10_Hospital_Coloc | 0.495 | W-Closeness 0.560 |
| Haggle_Infocom05 | 0.536 | Strength 0.601 |
| InVS13_Workplace | 0.529 | W-Closeness 0.569 |

  The two co-location networks are very **dense**: people have hundreds of contacts each. The 41x41 local patch
  sees only a small part of them. Retraining and contact-specific features should target these.

## Recommendation
Use the **calibrated regime** (10% mean outbreak) as the ground truth from Block 2 onward. Report the
original-regime numbers as well, to show the issue. If the guide prefers, a second target (for example 20%) can
be added as a sensitivity check.

## Files
- `contact_common.py`: network list, fast SIR, calibration, features, baselines
- `validate_sir.py`: checks the fast SIR against the original Third_Phase code
- `run_block1.py`: full pipeline (`python run_block1.py --regime calibrated` or `--regime original`)
- `results/zero_shot_report_{original,calibrated}.md`: per-network tables (3 metrics), group means, weak spots,
  SIR settings
- `results/zero_shot_results_*.csv`, `results/network_sir_stats_*.csv`: raw numbers
- `results/fig_tau20_by_network_*.png`, `results/fig_mean_by_group_*.png`: figures
- `cache/`: labels, features and predictions (`.npy`, not committed; rebuilt by `run_block1.py`)
