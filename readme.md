# N-LGCN Influential Node Detection

Influential node detection project based on **NLGCN** (a GCN-style model) for ranking the most important/influential nodes in a graph/network.


---

## What this project does

Given a graph (e.g., social network, citation network, communication network), the pipeline:

1. Loads and preprocesses graph data
2. Trains an **NLGCN** model on the graph
3. Produces an **influence score per node**
4. Outputs a **ranked list of influential nodes** + evaluation results (if labels/ground truth are available)

---

## Method (high level)

- **Model:** N‑LGCN (Graph Convolutional Network variant)
- **Task:** influential node detection / node ranking
- **Output:** node influence scores and top‑K influential node set

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

