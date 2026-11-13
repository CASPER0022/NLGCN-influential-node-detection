"""Build fourth_phase/datasets/Dataset_Summary.pdf from processed/dataset_stats.csv."""
import os
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets"))
stats = pd.read_csv(os.path.join(BASE, "processed", "dataset_stats.csv")).set_index("name")

SP = "Wearable RFID sensors (SocioPatterns), face-to-face proximity (~1.5 m), 20-s resolution"
DUR = "Total face-to-face contact time between the two persons (seconds)"
NZ = "SocioPatterns via Netzschleuder"

# name -> (setting & year, who the nodes are, what an edge means, edge weight, node metadata, source)
INFO = {
    "LH10_Hospital": ("Hospital ward, Lyon, France (Dec 2010)", "29 patients, 27 nurses, 11 doctors, 8 admin staff",
                      SP, DUR, "role (PAT/NUR/MED/ADM)", f"{NZ} (sp_hospital)"),
    "LyonSchool_Primary": ("Primary school, Lyon, France (Oct 2009)", "232 children, 10 teachers",
                           SP, DUR, "class, gender", f"{NZ} (sp_primary_school)"),
    "Thiers11_HighSchool": ("High school, Marseille, France (2011)", "118 students (3 classes), 8 teachers",
                            SP, DUR, "class, gender", f"{NZ} (sp_high_school_new/2011)"),
    "Thiers12_HighSchool": ("High school, Marseille, France (2012)", "Students (5 classes)",
                            SP, DUR, "class, gender", f"{NZ} (sp_high_school_new/2012)"),
    "Thiers13_HighSchool": ("High school, Marseille, France (Dec 2013)", "Students (9 classes)",
                            SP, DUR, "class, gender", f"{NZ} (sp_high_school/proximity)"),
    "InVS13_Workplace": ("Office building, France (2013)", "Employees",
                         SP, DUR, "department", f"{NZ} (sp_office)"),
    "InVS15_Workplace": ("Office building, France (2015)", "Employees",
                         SP, DUR, "department", f"{NZ} (sp_office_2015)"),
    "SFHH_Conference": ("Scientific conference (SFHH), Nice, France (2009)", "Conference participants",
                        SP, DUR, "none", f"{NZ} (sp_sfhh_conference)"),
    "HT09_Conference": ("ACM Hypertext conference, Turin, Italy (2009)", "Conference participants",
                        SP, DUR, "none", f"{NZ} (sp_hypertext/contacts)"),
    "Infectious_Museum": ("Science museum exhibition, Dublin, Ireland (Apr-Jul 2009)", "Museum visitors",
                          SP, DUR, "none", f"{NZ} (sp_infectious/contacts)"),
    "Malawi_Village": ("Rural village, Malawi (Dec 2019 - Jan 2020)", "Village residents",
                       SP, DUR, "none", f"{NZ} (sp_malawi_village)"),
    "Kenya_Households": ("5 rural households, Kenya (2012)", "Household members",
                         "Wearable sensors (SocioPatterns), close-range proximity",
                         DUR, "household, age, sex", f"{NZ} (sp_kenyan_households)"),
    "SouthAfrica_Households": ("Households of COVID-19 cases, Soweto & Klerksdorp, South Africa (2020-21)",
                               "Household members (241 SARS-CoV-2 positive, 99 negative)",
                               "Wearable proximity sensors, close-range contact",
                               DUR, "household, age group, sex, role, SARS-CoV-2 status, BMI, ...",
                               f"{NZ} (sp_south_african_households)"),
    "Haggle_Infocom05": ("IEEE Infocom 2005 conference", "41 participants carrying iMotes + 233 external Bluetooth devices",
                         "Bluetooth sighting between devices",
                         "Total contact duration (seconds; single sightings counted as 1 s)", "is_imote",
                         "Haggle project via Netzschleuder (contact)"),
    "Copenhagen_BT": ("Technical University of Denmark students, Copenhagen (2013, 4 weeks)", "University students",
                      "Smartphones detected each other in a Bluetooth scan (every 5 min)",
                      "Number of 5-min scans with RSSI >= -80 dBm (close range)", "gender",
                      "Copenhagen Networks Study via Netzschleuder (copenhagen/bt)"),
    "Utah_MiddleSchool": ("Urban middle school, Utah, USA (Nov 2012, 2 days)", "Grade 7-8 students",
                          "Wireless sensors (WREN), within ~2 m, 19.5-s resolution", DUR + ", summed over 2 days",
                          "grade, gender", "Toth et al. 2015, J R Soc Interface (suppl. D1, D2, D5)"),
    "Utah_ElementarySchool": ("Suburban elementary school, Utah, USA (Jan-Feb 2013, 2 days)", "Grade K-6 students",
                              "Wireless sensors (WREN), within ~2 m, 19.5-s resolution", DUR + ", summed over 2 days",
                              "grade, class, gender", "Toth et al. 2015, J R Soc Interface (suppl. D3, D4, D6)"),
    # ---- second batch ----
    "Utah_MiddleSchool_1m": ("Urban middle school, Utah, USA (Nov 2012, 2 days)", "Grade 7-8 students",
                             "Same sensors as above, stricter definition: within ~1 m", DUR + ", summed over 2 days",
                             "grade, gender", "Toth et al. 2015, J R Soc Interface (suppl. D7, D8, D5)"),
    "Utah_ElementarySchool_1m": ("Suburban elementary school, Utah, USA (Jan-Feb 2013, 2 days)", "Grade K-6 students",
                                 "Same sensors as above, stricter definition: within ~1 m", DUR + ", summed over 2 days",
                                 "grade, class, gender", "Toth et al. 2015, J R Soc Interface (suppl. D9, D10, D6)"),
    "RealityMining_BT": ("MIT Media Lab & Sloan students, Cambridge, USA (2004-05)", "Students and staff",
                         "Mobile phones detected each other via Bluetooth",
                         "Number of Bluetooth detections between the two phones", "none",
                         "MIT Reality Mining via Netzschleuder (reality_mining)"),
    "Thiers13_Diaries": ("High school, Marseille, France (Dec 2013)", "Students who filled in a contact diary",
                         "Student reported a contact with the other student in a diary",
                         "Reported total contact time, ordinal: 1 = <5 min, 2 = 5-15 min, 3 = 15-60 min, 4 = >1 h "
                         "(max of the two reports)", "class, gender", f"{NZ} (sp_high_school/diaries)"),
}

COLOC_EDGE = ("Co-presence: both persons' RFID tags heard by exactly the same set of room readers in a 20-s window "
              "(same area, not necessarily face-to-face)")
COLOC_W = "Total co-presence time between the two persons (seconds)"
for _name, _setting, _who, _meta, _sub in [
        ("LH10_Hospital_Coloc", "Hospital ward, Lyon, France (Dec 2010)", "Patients and health-care workers", "role", "LH10"),
        ("InVS13_Workplace_Coloc", "Office building, France (2013)", "Employees", "department", "InVS13"),
        ("InVS15_Workplace_Coloc", "Office building, France (2015)", "Employees", "department", "InVS15"),
        ("SFHH_Conference_Coloc", "Scientific conference (SFHH), Nice, France (2009)", "Conference participants", "none", "SFHH"),
        ("LyonSchool_Primary_Coloc", "Primary school, Lyon, France (Oct 2009)", "Children and teachers", "class", "LyonSchool"),
        ("Thiers13_HighSchool_Coloc", "High school, Marseille, France (Dec 2013)", "Students", "class", "Thiers13")]:
    INFO[_name] = (_setting, _who, COLOC_EDGE, COLOC_W, _meta, f"{NZ} (sp_colocation/{_sub})")

NOTES = [
    ("Infectious_Museum", " aggregates 69 days of visitors who mostly never met, so it splits into 304 components; "
                          "use its largest component (410 nodes) or the per-day networks."),
    ("SouthAfrica_Households", " is 88 separate households studied at different times; it is not one connected network, "
                               "but it is the only set with real SARS-CoV-2 infection status, which makes it useful for validation."),
    ("Copenhagen_BT", ": Bluetooth records with RSSI below -80 dBm (far apart) were dropped."),
    ("Haggle_Infocom05", ": only 41 nodes are study participants; the rest are other Bluetooth devices seen by them."),
    ("*_Coloc", " networks are the same populations as the face-to-face sets, but an edge means sharing the same "
                "area (relevant for airborne spread), so they are much denser."),
    ("*_1m", " Utah networks use a stricter < 1 m proximity rule than the main Utah networks (< 2 m)."),
    ("Thiers13_Diaries", " is self-reported (survey), not sensor data; its weight is an ordinal category, not seconds."),
]
NOTES_INTRO = ("Nodes, edges and degree are for the full aggregated graph. Span = time between the first and last "
               "contact record (it includes nights and weekends; '-' = no timestamps). ")


def fmt_span(v):
    return "-" if pd.isna(v) else f"{v:.1f}"


def fmt_w(v):
    return f"{v:,.0f}" if v >= 10 else f"{v:g}"


def main():
    ss = getSampleStyleSheet()
    cell = ParagraphStyle("cell", parent=ss["Normal"], fontSize=7.2, leading=8.6)
    head = ParagraphStyle("head", parent=cell, fontName="Helvetica-Bold")
    P = lambda t, s=cell: Paragraph(str(t), s)

    cols = ["#", "Dataset (file name)", "Setting (place, year)", "What a node is", "What an edge means",
            "What the edge weight means", "Nodes", "Edges", "Largest conn. comp. (nodes)", "Avg. degree",
            "Density", "Weight median / max", "Span (days)", "Node metadata", "Source"]
    data = [[P(c, head) for c in cols]]
    for i, (name, (setting, who, edge, weight, meta, src)) in enumerate(INFO.items(), 1):
        r = stats.loc[name]
        data.append([P(i), P(f"<b>{name}</b>"), P(setting), P(who), P(edge), P(weight),
                     P(f"{int(r.nodes):,}"), P(f"{int(r.edges):,}"),
                     P(f"{int(r.lcc_nodes):,}" + ("" if r.components == 1 else f" ({int(r.components)} comps)")),
                     P(f"{r.avg_degree:.1f}"), P(f"{r.density:.3f}"),
                     P(f"{fmt_w(r.w_median)} / {fmt_w(r.w_max)}"), P(fmt_span(r.span_days)), P(meta), P(src)])

    widths = [9, 43, 40, 37, 42, 42, 15, 15, 21, 14, 14, 22, 13, 30, 38]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6E6E6")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))

    note = ParagraphStyle("note", parent=ss["Normal"], fontSize=8, leading=10)
    story = [
        Paragraph("Human Contact Network Datasets for Epidemic Super-Spreader Identification", ss["Title"]),
        Paragraph("Application: ranking influential nodes (potential super-spreaders) in weighted human contact "
                  "networks, for targeted vaccination, testing and isolation. All networks are undirected and "
                  "weighted, built by summing the time-stamped contact records of each pair of people into one "
                  "weighted edge (self-loops removed). A larger weight means more contact, so the weights are used "
                  "with positive semantics. Files: <i>fourth_phase/datasets/processed/&lt;Dataset&gt;.txt</i> "
                  "(format \"u v weight\") and <i>&lt;Dataset&gt;_nodes.csv</i> (node metadata). Raw downloads are "
                  "in <i>fourth_phase/datasets/raw/</i>.", note),
        Spacer(1, 4 * mm), t, Spacer(1, 4 * mm),
        Paragraph("<b>Notes.</b> " + NOTES_INTRO + " ".join(
            f"({i}) <b>{k}</b>{t}" for i, (k, t) in enumerate(NOTES, 1)), note),
    ]
    out = os.path.join(BASE, "Dataset_Summary.pdf")
    SimpleDocTemplate(out, pagesize=landscape(A3), leftMargin=10 * mm, rightMargin=10 * mm,
                      topMargin=10 * mm, bottomMargin=10 * mm, title="Contact Network Dataset Summary").build(story)
    print(out)


if __name__ == "__main__":
    main()
