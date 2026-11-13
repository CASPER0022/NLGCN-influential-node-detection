"""
Convert the raw (temporal) human contact datasets in fourth_phase/datasets/raw into
static, undirected, weighted edge lists for the Hybrid Global-Local WNLGCN.

Output (fourth_phase/datasets/processed):
    <Name>.txt          "u v w" per line, w = aggregated contact weight (see WEIGHT_DESC)
    <Name>_nodes.csv    node metadata (role/class/department/...), when available
    dataset_stats.csv   summary statistics used for the PDF report

Aggregation rule: every observed contact record between i and j adds its duration
(in seconds) to edge (i, j). Self-loops are dropped. Semantics = "positive"
(larger weight = more contact time = higher transmission probability).
"""
import os
import sys
import numpy as np
import pandas as pd
import networkx as nx

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets"))
RAW = os.path.join(BASE, "raw")
OUT = os.path.join(BASE, "processed")
os.makedirs(OUT, exist_ok=True)

COPENHAGEN_RSSI_MIN = -80   # keep only close-range Bluetooth detections
UTAH_UNIT = 19.53125        # seconds per time unit in the Toth et al. files


def read_nz(folder):
    """Read a Netzschleuder CSV export (header line starts with '# ')."""
    def _read(f):
        p = os.path.join(RAW, folder, f)
        with open(p, encoding="utf-8") as fh:
            cols = [c.strip() for c in fh.readline().lstrip("#").split(",")]
        return pd.read_csv(p, comment=None, skiprows=1, names=cols, header=None,
                           skipinitialspace=True, low_memory=False)
    return _read("edges.csv"), _read("nodes.csv")


def window_records(edges, window=20):
    """SocioPatterns style: one row per active 20-s window -> each row = 20 s of contact."""
    return edges[["source", "target"]].assign(sec=float(window))


def aggregate(rec):
    rec = rec[rec["source"] != rec["target"]].copy()
    u = np.minimum(rec["source"].values, rec["target"].values)
    v = np.maximum(rec["source"].values, rec["target"].values)
    rec = pd.DataFrame({"u": u, "v": v, "w": rec["sec"].values})
    return rec.groupby(["u", "v"], as_index=False)["w"].sum()


def span_days(t):
    t = pd.to_numeric(t, errors="coerce").dropna()
    return (t.max() - t.min()) / 86400.0 if len(t) else np.nan


def load_all():
    """Yield (name, aggregated edges df, n_raw_records, node metadata df or None, span_days)."""
    nz_window = [
        ("LH10_Hospital", "sp_hospital__sp_hospital"),
        ("LyonSchool_Primary", "sp_primary_school__contacts"),
        ("Thiers11_HighSchool", "sp_high_school_new__2011"),
        ("Thiers12_HighSchool", "sp_high_school_new__2012"),
        ("Thiers13_HighSchool", "sp_high_school__proximity"),
        ("InVS13_Workplace", "sp_office__sp_office"),
        ("InVS15_Workplace", "sp_office_2015__sp_office_2015"),
        ("SFHH_Conference", "sp_sfhh_conference__sp_sfhh_conference"),
        ("HT09_Conference", "sp_hypertext__contacts"),
        ("Infectious_Museum", "sp_infectious__contacts"),
        ("Malawi_Village", "sp_malawi_village__sp_malawi_village"),
    ]
    for name, folder in nz_window:
        e, n = read_nz(folder)
        sp = span_days(e["time"])
        yield name, aggregate(window_records(e)), len(e), n, sp

    # Records that carry their own duration (seconds)
    e, n = read_nz("sp_kenyan_households__sp_kenyan_households")
    yield ("Kenya_Households", aggregate(e[["source", "target"]].assign(sec=e["duration"].astype(float))),
           len(e), n, float(e["day"].nunique()))

    e, n = read_nz("sp_south_african_households__sp_south_african_households")
    yield ("SouthAfrica_Households", aggregate(e[["source", "target"]].assign(sec=e["duration"].astype(float))),
           len(e), n, span_days(e.loc[e["timestamped"] == 1, "time"]))

    # Haggle: ~half the records are single sightings with duration 0 -> count them as 1 s
    e, n = read_nz("contact__contact")
    yield ("Haggle_Infocom05", aggregate(e[["source", "target"]].assign(sec=e["duration"].clip(lower=1).astype(float))),
           len(e), n, span_days(e["time"]))

    # Copenhagen Bluetooth: one row per 5-min scan; weight = number of close-range scans
    e, n = read_nz("copenhagen__bt")
    e = e[(e["rssi"] >= COPENHAGEN_RSSI_MIN) & (e["rssi"] < 0)]
    yield ("Copenhagen_BT", aggregate(e[["source", "target"]].assign(sec=1.0)),
           len(e), n, span_days(e["timestamp"]))

    # Utah schools (Toth et al. 2015): D1-D2 = Mid1 day 1/2, D3-D4 = Elem1 day 1/2
    utah = os.path.join(RAW, "utah_schools_toth2015")
    for name, files, meta in [("Utah_MiddleSchool", ["rsif20150279supp2.txt", "rsif20150279supp3.txt"], "rsif20150279supp6.txt"),
                              ("Utah_ElementarySchool", ["rsif20150279supp4.txt", "rsif20150279supp5.txt"], "rsif20150279supp7.txt")]:
        e = pd.concat([pd.read_csv(os.path.join(utah, f), sep=r"\s+") for f in files], ignore_index=True)
        rec = pd.DataFrame({"source": e["id1"], "target": e["id2"], "sec": e["duration"] * UTAH_UNIT})
        n = pd.read_csv(os.path.join(utah, meta), sep=r"\s+")
        yield name, aggregate(rec), len(e), n, 2.0

    # ---------------- Additional networks (second batch) ----------------
    # Utah stricter < 1 m contact definition: D7-D8 = Mid1 day 1/2, D9-D10 = Elem1 day 1/2 (no header row)
    for name, files, meta in [("Utah_MiddleSchool_1m", ["rsif20150279supp8.txt", "rsif20150279supp9.txt"], "rsif20150279supp6.txt"),
                              ("Utah_ElementarySchool_1m", ["rsif20150279supp10.txt", "rsif20150279supp11.txt"], "rsif20150279supp7.txt")]:
        e = pd.concat([pd.read_csv(os.path.join(utah, f), sep=r"\s+", header=None,
                                   names=["id1", "id2", "startTime", "duration"]) for f in files], ignore_index=True)
        rec = pd.DataFrame({"source": e["id1"], "target": e["id2"], "sec": e["duration"] * UTAH_UNIT})
        n = pd.read_csv(os.path.join(utah, meta), sep=r"\s+")
        yield name, aggregate(rec), len(e), n, 2.0

    # MIT Reality Mining: one row per Bluetooth detection; weight = number of detections
    e, n = read_nz("reality_mining__reality_mining")
    yield ("RealityMining_BT", aggregate(e[["source", "target"]].assign(sec=1.0)),
           len(e), n, span_days(e["time"]))

    # Thiers13 contact diaries: self-reported, directed, ordinal weight 1-4; keep the max of the two reports
    e, n = read_nz("sp_high_school__diaries")
    e = e[e["source"] != e["target"]]
    u = np.minimum(e["source"].values, e["target"].values)
    v = np.maximum(e["source"].values, e["target"].values)
    agg = (pd.DataFrame({"u": u, "v": v, "w": e["weight"].astype(float).values})
           .groupby(["u", "v"], as_index=False)["w"].max())
    yield "Thiers13_Diaries", agg, len(e), n, np.nan

    # SocioPatterns co-location (same set of RFID readers in a 20-s window); weight = co-presence time (s)
    for name, s in [("LH10_Hospital_Coloc", "LH10"), ("InVS13_Workplace_Coloc", "InVS13"),
                    ("InVS15_Workplace_Coloc", "InVS15"), ("SFHH_Conference_Coloc", "SFHH"),
                    ("LyonSchool_Primary_Coloc", "LyonSchool"), ("Thiers13_HighSchool_Coloc", "Thiers13")]:
        e, n = read_nz(f"sp_colocation__{s}")
        yield name, aggregate(window_records(e)), len(e), n, span_days(e["time"])


def main():
    rows = []
    for name, agg, n_raw, nodes, sp in load_all():
        agg["w"] = agg["w"].round(2)
        agg.to_csv(os.path.join(OUT, f"{name}.txt"), sep=" ", header=False, index=False)
        if nodes is not None:
            nodes.drop(columns=[c for c in nodes.columns if c == "_pos"]).to_csv(
                os.path.join(OUT, f"{name}_nodes.csv"), index=False)

        G = nx.Graph()
        G.add_weighted_edges_from(agg[["u", "v", "w"]].itertuples(index=False, name=None))
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        L = G.subgraph(comps[0])
        w = agg["w"].values
        rows.append(dict(
            name=name, nodes=G.number_of_nodes(), edges=G.number_of_edges(), raw_records=n_raw,
            components=len(comps), lcc_nodes=L.number_of_nodes(), lcc_edges=L.number_of_edges(),
            avg_degree=2 * G.number_of_edges() / G.number_of_nodes(), density=nx.density(G),
            clustering=nx.average_clustering(G),
            w_min=w.min(), w_median=float(np.median(w)), w_max=w.max(), span_days=sp,
        ))
        print(f"{name:24s} N={G.number_of_nodes():6d} E={G.number_of_edges():7d} "
              f"LCC={L.number_of_nodes():6d} comps={len(comps):4d} w=[{w.min():.0f},{np.median(w):.0f},{w.max():.0f}]")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "dataset_stats.csv"), index=False)


if __name__ == "__main__":
    sys.exit(main())
