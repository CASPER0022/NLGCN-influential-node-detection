# Fourth Phase Work Plan: Super-Spreader Identification in Human Contact Networks

**Goal:** Apply and adapt the Hybrid Global-Local WNLGCN to weighted human contact networks, to identify
super-spreaders for targeted vaccination, testing and isolation.

**Deadline:** December 10, 2026

**Structure:** 4-day work blocks with one buffer day after each block. Use the buffer day to catch up on
anything unfinished, or to rest if the block is done.

---

## Done: Nov 13 (Domain and datasets)
- [x] Chose the application domain: epidemic control in human contact networks
- [x] Downloaded 27 weighted contact networks into `datasets/`
- [x] Processed them into `u v weight` edge lists, with node metadata (`datasets/processed/`)
- [x] Wrote the dataset summaries (`Dataset_Summary.pdf`, `Dataset_Summary.docx`)
- [x] Downloaded 4 reference papers (`references/`)

---

## Block 1: Nov 14 - Nov 17 (Ground truth and zero-shot baseline)
**Deliverable:** SIR labels for all usable networks, and a first results table for the current model, unchanged.

- **Nov 14:** Read Stehle 2011 and Smieszek & Salathe 2013. Decide which networks to use: leave out
  `SouthAfrica_Households` as a training graph, and use the largest component (LCC) of `Infectious_Museum`.
  Register all networks as `positive` in the loader.
  - [ ] Read the two papers
  - [x] Network selection: 26 networks, LCC, `positive` weights
- [x] **Nov 15:** Generate SIR ground-truth labels for every network, reusing the Third_Phase pipeline
  (beta = 1.5 / lambda_max, same settings as before).
  - Added: a fast live-edge SIR, validated against the original code.
  - Added: a calibrated regime (10% mean outbreak), because the original beta rule gives near-dead or
    near-total outbreaks on contact networks.
- [x] **Nov 16:** Compute local and global features for all networks. Run the trained hybrid model zero-shot
  (no retraining).
- [x] **Nov 17:** Compare against the baselines (W-Eigenvector, Strength, W-Closeness, W-Betweenness):
  Kendall tau and top-20% overlap. Save the results table, and note where the model is weak.
  See `block1/BLOCK1_SUMMARY.md`.

**Nov 18: Buffer**

---

## Block 2: Nov 19 - Nov 22 (Application evaluation framework)
**Deliverable:** Evaluation scripts that measure real public-health impact, not only ranking accuracy.

- **Nov 19:** Read Pung 2024 and Dong 2024. Write down how their evaluation differs from ours.
  - [ ] Read the two papers
- [x] **Nov 20:** Implement the targeted-vaccination experiment: remove the top-k% nodes chosen by each method,
  rerun SIR, and measure the drop in final outbreak size and in peak size.
- [x] **Nov 21:** Implement the stability tests:
  - (a) face-to-face vs co-location networks of the same people
  - (b) the 2 m vs 1 m Utah networks
  - (c) day 1 vs day 2: rank on day 1, then check the ranking against spreading on day 2
- [x] **Nov 22:** Run all three test types on the zero-shot model and the baselines. Make the first plots
  (outbreak size vs % vaccinated). See `block2/BLOCK2_SUMMARY.md`.
  - Finding: ours matches vaccinating the true top spreaders, but Strength vaccinates better (top spreaders
    are clustered, so vaccinating all of them is redundant). Day-to-day contact changes cap the day-2 result.

**Nov 23: Buffer** (Block 2 finished early, so use it for vaccination step 1)
- [ ] **Vaccination step 1: smarter selection, no retraining.** Pick people one at a time instead of taking the
  top k% at once. After vaccinating v, lower each contact's score: s_j <- s_j * (1 - c_jv / C_j), where c_jv is the
  contact time between j and v and C_j is j's total contact time. Also try adaptive re-ranking: vaccinate the top
  1-2%, recompute the scores on the remaining network, repeat. Rerun `block2/run_vaccination.py` and compare with
  Strength (Block 2 showed plain top-k is redundant: top spreaders sit in one dense cluster).

---

## Block 3: Nov 24 - Nov 27 (Domain-specific features)
**Deliverable:** Contact-network feature set, with an ablation showing which features help.

- **Nov 24:** Add the temporal features from the raw records: number of distinct contact days, contact frequency
  (number of separate meetings), average contact length, and burstiness.
- **Nov 25:** Add the metadata features where available: role (patient / nurse / doctor), and cross-group
  bridging (the share of a person's contact time spent with other classes or departments).
- **Nov 26:** Feature ablation. Remove the features that add nothing here, for example component size, since
  most of these networks are fully connected. Keep the features that help.
- **Nov 27:** Freeze the final feature set. Document each new feature: formula, and why it matters for
  disease spread.

**Nov 28: Buffer**

---

## Block 4: Nov 29 - Dec 2 (Fine-tuning and retraining)
**Deliverable:** Contact-network version of the model, better than the zero-shot model and the baselines.

- **Nov 29:** Decide the train / validation / test split. Split by setting (for example, train on schools and
  workplaces, test on hospital, households and conferences) so that no test network is seen during training.
  **Vaccination step 2 (labels):** compute a vaccination-impact label for every node: how much the mean outbreak
  shrinks when that person alone is removed (live-edge SIR, same random samples for every node).
- **Nov 30:** Fine-tune the hybrid model from the Third_Phase checkpoints, with the new features. Also train
  from scratch for comparison.
  **Vaccination step 2 (model):** add a second output head trained on the vaccination-impact labels.
- **Dec 1:** Run 3 seeds (42, 43, 44). Pick the best variant on the validation networks only.
- **Dec 2:** Evaluate on the test networks: Kendall tau, top-20% overlap, and the vaccination experiment.
  Compare three vaccination plans: plain top-k, top-k with the step 1 selection rule, and the step 2 impact head.

**Dec 3: Buffer**

---

## Block 5: Dec 4 - Dec 7 (Final experiments and results)
**Deliverable:** All final tables and figures for the report.

- **Dec 4:** Rerun all evaluations with the final model: ranking metrics, vaccination, and the stability tests.
  **Vaccination step 3 (upper bound):** on the small networks, run a simulation-based greedy search (each step
  vaccinates whoever reduces the simulated outbreak the most) and report how close our plan gets to it.
- **Dec 5:** Validate on `SouthAfrica_Households`: check whether high-ranked people have more infected
  household members (it is the only dataset with real SARS-CoV-2 status). Do a hospital case study on
  `LH10_Hospital`: which roles (nurses, doctors, patients) the model ranks highest.
- **Dec 6:** Make the final figures: vaccination curves, a stability heatmap, and the ranked nodes on one
  example network. Make the LaTeX tables.
- **Dec 7:** Write a results summary (`fourth_phase/results/`), and commit all code and results.

**Dec 8: Buffer**

---

## Final: Dec 9 - Dec 10 (Report and presentation)
**Deliverable:** Updated report chapter and final presentation.

- **Dec 9:** Write the application chapter for the report: motivation, datasets (from `Dataset_Summary.docx`),
  method changes, results, and limitations.
- **Dec 10:** Prepare the presentation slides, do a final check of all files, then commit and push.

---

## Weekly check
- [x] Nov 17: zero-shot results table ready
- [x] Nov 22: vaccination and stability experiments working
- [ ] Nov 27: final feature set frozen
- [ ] Dec 2: fine-tuned model evaluated on the test networks
- [ ] Dec 7: all results and figures final
- [ ] Dec 10: report and presentation submitted
