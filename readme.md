# N-LGCN Influential Node Detection

Influential node detection project based on **N‑LGCN** (a GCN-style model) for ranking the most important/influential nodes in a graph/network.

This repository contains the code and experiments for my honors project: **[your school/program]** — **[your name]**.

---

## What this project does

Given a graph (e.g., social network, citation network, communication network), the pipeline:

1. Loads and preprocesses graph data
2. Trains an **N‑LGCN** model on the graph
3. Produces an **influence score per node**
4. Outputs a **ranked list of influential nodes** + evaluation results (if labels/ground truth are available)

---

## Method (high level)

- **Model:** N‑LGCN (Graph Convolutional Network variant)
- **Task:** influential node detection / node ranking
- **Output:** node influence scores and top‑K influential node set

> Replace this section with your exact details: objective/loss, how influence is computed (e.g., learned score head, centrality supervision, diffusion proxy labels, etc.).

---

## Repository structure

Update to match your repo:

- `data/` — datasets (often excluded from git; see below)
- `src/` — model + training code
- `notebooks/` — experiments/plots (optional)
- `scripts/` — training/evaluation runners
- `results/` — saved outputs (metrics, rankings, figures)
- `requirements.txt` / `environment.yml` — dependencies

---

## Setup

### 1) Create environment

**Option A: pip**
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

**Option B: conda**
```bash
conda env create -f environment.yml
conda activate [env-name]
```

### 2) Add data

Place your datasets in `data/`.

If datasets are large or restricted, add a note here explaining where they come from and how to download them.

---

## How to run

Replace the commands below with your real entry points.

### Train
```bash
python scripts/train.py --dataset [name] --epochs 200 --lr 0.01
```

### Evaluate / Rank nodes
```bash
python scripts/eval.py --dataset [name] --topk 50
```

### Output
Typical outputs include:
- `results/metrics.json`
- `results/node_rankings.csv` (node_id, influence_score, rank)
- `results/plots/` (optional)

---

## Reproducibility

Recommended settings:
- Fix random seeds (Python/NumPy/PyTorch)
- Log hyperparameters and dataset splits
- Save model checkpoints

If you use GPU:
- CUDA version: **[your CUDA version]**
- PyTorch version: **[your version]**

---

## Results

Add your main results here:
- Dataset(s): **[dataset names]**
- Metric(s): **[AUC/F1/Precision@K/NDCG/etc.]**
- Best configuration: **[summary]**

---

## Citation / Reference

If this work is based on a paper or your thesis document, link it here:

- Paper/Thesis: **[link or title]**
- Implementation notes: **[optional]**

---

## License

Choose one:
- MIT (common for code)
- Apache-2.0
- GPL-3.0

Add a `LICENSE` file to the repo.
