"""
Block 2 / Nov 21: split the temporal records into single-day networks (for the day 1 -> day 2 test).

Day boundaries: a gap of more than 4 hours with no contact records (night). If the recording runs through
the night (hospital, village), consecutive 24-hour windows from the first record are used instead.
Day 1 / day 2 = the first two days that hold at least 10% as many records as the busiest day.
Weights are built exactly as in scripts/process_contact_networks.py, and node labels are the same as in
datasets/processed, so day networks and full networks can be matched node by node.

Output: block2/data/<Name>_day1.txt, <Name>_day2.txt ("u v w") and day_networks.csv.
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import process_contact_networks as pcn     # noqa: E402

DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

NZ = {"LH10_Hospital": "sp_hospital__sp_hospital",
      "LyonSchool_Primary": "sp_primary_school__contacts",
      "Thiers11_HighSchool": "sp_high_school_new__2011",
      "Thiers12_HighSchool": "sp_high_school_new__2012",
      "Thiers13_HighSchool": "sp_high_school__proximity",
      "InVS13_Workplace": "sp_office__sp_office",
      "InVS15_Workplace": "sp_office_2015__sp_office_2015",
      "SFHH_Conference": "sp_sfhh_conference__sp_sfhh_conference",
      "HT09_Conference": "sp_hypertext__contacts",
      "Malawi_Village": "sp_malawi_village__sp_malawi_village"}
UTAH = {"Utah_MiddleSchool": ("rsif20150279supp2.txt", "rsif20150279supp3.txt"),
        "Utah_ElementarySchool": ("rsif20150279supp4.txt", "rsif20150279supp5.txt")}


def day_index(t, gap=4 * 3600, min_share=0.10):
    t = np.asarray(t, dtype=float)
    ut = np.unique(t)
    cuts = ut[1:][np.diff(ut) > gap]
    day = np.searchsorted(cuts, t, side="right")
    lengths = [np.ptp(t[day == d]) for d in np.unique(day)]
    if max(lengths) > 26 * 3600:                         # recorded through the night
        day = ((t - t.min()) // 86400).astype(int)
    counts = np.bincount(day)
    good = [d for d in range(len(counts)) if counts[d] >= min_share * counts.max()]
    return day, good[:2]


def write(name, d, rec):
    agg = pcn.aggregate(rec)
    agg["w"] = agg["w"].round(2)
    path = os.path.join(DATA, f"{name}_day{d}.txt")
    agg.to_csv(path, sep=" ", header=False, index=False)
    return len(set(agg.u) | set(agg.v)), len(agg), len(rec)


def main():
    rows = []
    for name, folder in NZ.items():
        e, _ = pcn.read_nz(folder)
        day, (d1, d2) = day_index(e["time"])
        for k, d in ((1, d1), (2, d2)):
            n, m, r = write(name, k, pcn.window_records(e[day == d]))
            rows.append(dict(dataset=name, day=k, nodes=n, edges=m, records=r))
    for name, files in UTAH.items():
        for k, f in enumerate(files, start=1):
            e = pd.read_csv(os.path.join(pcn.RAW, "utah_schools_toth2015", f), sep=r"\s+")
            rec = pd.DataFrame({"source": e["id1"], "target": e["id2"], "sec": e["duration"] * pcn.UTAH_UNIT})
            n, m, r = write(name, k, rec)
            rows.append(dict(dataset=name, day=k, nodes=n, edges=m, records=r))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "day_networks.csv"), index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
