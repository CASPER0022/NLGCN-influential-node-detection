# Block 2 Summary: Application Evaluation (Nov 19 - Nov 22)

Block 1 measured ranking accuracy. Block 2 asks the public-health questions: **does the ranking reduce an
outbreak when those people are vaccinated**, and **does it still hold when the contact data changes**
(another sensor definition, a stricter distance, the next day)?

All experiments use the calibrated SIR regime from Block 1 (10% mean outbreak without intervention), the
same zero-shot Hybrid G-L ensemble (no retraining), and the same baselines. MF-Reach is a model input and is
not used as a baseline.

## 1. Targeted vaccination (26 networks)
Each method vaccinates its top-k% of people (they can neither catch nor pass on the infection). SIR is then
run from a random unvaccinated index case (200 live-edge samples). Every method is tested on the **same**
random samples, so the differences between methods are not caused by sampling noise. We measure the final
outbreak size and the peak size (the largest number infected in one generation). Two references are included:
**Random** vaccination, and **True SIR ranking**, which vaccinates the real top spreaders (a perfect
super-spreader detector).

Mean outbreak reduction over the 26 networks (higher is better):

| % vaccinated | Ours | Strength | W-Betw | Deep6 | W-Clos | W-Eig | Random | True SIR ranking |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.5% | 0.337 | 0.334 | 0.333 | 0.310 | 0.326 | 0.261 | 0.111 | 0.350 |
| 5% | 0.513 | **0.548** | 0.511 | 0.487 | 0.475 | 0.388 | 0.216 | 0.521 |
| 10% | 0.670 | **0.724** | 0.676 | 0.669 | 0.626 | 0.525 | 0.380 | 0.675 |
| 20% | 0.817 | **0.840** | 0.815 | 0.812 | 0.759 | 0.675 | 0.595 | 0.799 |

**Findings**
- Vaccinating our top 10% stops two thirds of the outbreak (0.670), compared with 0.380 for random
  vaccination. Ours beats random vaccination on 25 of 26 networks and W-Eigenvector on 24 of 26.
- **Ours matches vaccinating the true top spreaders**: 99% of its reduction at 10%, and slightly better
  at 15-20%. In other words, as a *super-spreader detector* the model is already close to perfect here.
- **Strength is the best vaccination strategy** (0.724 at 10%), and it beats even the true top-spreader list.
  This is not a ranking error, because the perfect detector loses too. The reason is **redundancy**: the true
  top spreaders (and our top 10%) spend **54%** of their contact time with each other, compared with **44%**
  for Strength's top 10%. They form one dense cluster, so vaccinating all of them removes overlapping
  infection routes. Strength picks people from more parts of the network.
- **Takeaway:** "Who spreads the most" (the training target) is not the same as "whom to vaccinate". This
  is a known effect in network epidemiology, and it gives a clear direction for Block 4: a
  vaccination-aware target or a redundancy-aware selection step. Our model gives the spreader scores
  such a step would use.
- Weakest network: `Thiers13_Diaries`. There, even the true top spreaders give only 0.35 at 10%, compared
  with 0.57 for Strength. The diary network is small and clustered, which is an extreme case of the
  same redundancy effect.

## 2. Stability tests
A ranking is computed on view A of a population and evaluated against SIR spreading on view B, using only
the people present in both. The reference **SIR on A** ranks people by their real spreading in view A. It shows how
much the spreading itself changes between views, so it is close to the best any view-A ranking can do.

| Test | Pairs | Ours | Strength | W-Eig | Deep6 | W-Clos | SIR on A (ref.) |
|---|---:|---:|---:|---:|---:|---:|---:|
| (a) face-to-face <-> co-location, top-20% overlap | 12 | **0.423** | 0.391 | 0.364 | 0.368 | 0.370 | 0.423 |
| (b) 2 m <-> 1 m (Utah), top-20% overlap | 4 | **0.762** | 0.688 | 0.643 | 0.552 | 0.485 | 0.766 |
| (b) 2 m <-> 1 m (Utah), tau top 20% | 4 | **0.520** | 0.428 | 0.397 | 0.286 | 0.079 | 0.547 |
| (c) day 1 -> day 2, top-20% overlap | 12 | 0.469 | 0.442 | 0.382 | **0.482** | 0.400 | 0.477 |
| (c) day 1 -> day 2, tau top 20% | 12 | 0.227 | 0.146 | **0.229** | 0.192 | 0.152 | 0.235 |
| Score agreement A vs B, mean of (a), (b), (c) | | **0.488** | 0.466 | 0.409 | 0.294 | 0.376 | 0.481 |

**Findings**
- In every test **ours is at the reference level** (0.423 = 0.423, 0.762 vs 0.766, 0.469 vs 0.477). The
  model's day-1 ranking predicts day-2 spreading as well as knowing the real day-1 spreading would.
- **The limit is the data, not the model.** Spreading roles change a lot between views. Only about 42% of the
  top-20% spreaders are the same in face-to-face and co-location data, and about 48% from one day to the
  next. The 2 m vs 1 m change matters much less (77% are the same).
- **Day 1 -> day 2 vaccination** (vaccinate the day-1 top 10% before a day-2 outbreak, 12 sites): Ours
  0.405, Strength 0.429, W-Betw 0.418, true day-1 spreaders 0.399, random 0.358. A detector that already
  knew the day-2 contacts would reach 0.629. The gap from about 0.40 to 0.63 is caused by contacts
  changing from day to day. **Block 3 should target this gap** with temporal features (how many days a
  person is active, how regular their contacts are), so that the ranking favours people who are central on
  every day, not only on the observed day.

## Recommendation for Block 3 and Block 4
1. Temporal features (Block 3): measure consistency across days, aiming at the day-1 -> day-2 gap.
2. Vaccination target (Block 4): fine-tune on, or post-process with, a redundancy-aware objective, so the
   chosen set covers different parts of the network. Present this honestly: the spreader ranking is already
   at the reference level, and choosing whom to vaccinate is a separate task built on top of it.

## Files
- `common2.py`: method scores for any network, and the vaccination simulator (same random samples for every method)
- `run_vaccination.py`: vaccination experiment on the 26 networks
- `make_day_networks.py`: splits the raw records into day-1 and day-2 networks (`data/`)
- `run_stability.py`: tests (a), (b), (c) and the day-2 vaccination
- `results/vaccination_report.md`, `results/stability_report.md`: full per-network tables
- `results/vaccination_results.csv`, `stability_results.csv`, `day2_vaccination_results.csv`: raw numbers
- `results/fig_vaccination_curves.png`, `fig_vaccination_by_group.png`, `fig_reduction10_by_network.png`,
  `fig_stability.png`: figures
- `cache/`: scores and labels (`.npz`, rebuilt by the scripts)
