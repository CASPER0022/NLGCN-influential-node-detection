"""Build fourth_phase/datasets/Dataset_Summary.docx - a simplified, easy-to-read version of the
dataset summary (the detailed version is Dataset_Summary.pdf).

Usage: python make_dataset_summary_docx.py [output_path]
"""
import os
import sys
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_dataset_summary_pdf import BASE, stats  # noqa: E402

FONT = "Calibri"
COLS = ["#", "Dataset", "What it is", "What an edge means", "Nodes", "Edges", "Source"]
WIDTHS_MM = [8, 48, 68, 71, 16, 18, 38]          # sums to 267 mm (A4 landscape, 15 mm margins)

FTF = "Two people were face-to-face (within ~1.5 m). Weight = total contact time (seconds)."
COLOC = "Two people were in the same area of the building at the same time. Weight = total time together (seconds)."
SP = "SocioPatterns [1]"

# (group title, group explanation, [(dataset, what it is, edge meaning, source), ...])
GROUPS = [
    ("A. Face-to-face contact networks (wearable RFID sensors)",
     "People wore small sensors that record when two people face each other at close range.",
     [("LH10_Hospital", "Hospital ward, Lyon, France (2010): patients, nurses, doctors and staff", FTF, SP),
      ("LyonSchool_Primary", "Primary school, Lyon, France (2009): children and teachers", FTF, SP),
      ("Thiers11_HighSchool", "High school, Marseille, France (2011): students and teachers", FTF, SP),
      ("Thiers12_HighSchool", "High school, Marseille, France (2012): students", FTF, SP),
      ("Thiers13_HighSchool", "High school, Marseille, France (2013): students", FTF, SP),
      ("InVS13_Workplace", "Office building, France (2013): employees", FTF, SP),
      ("InVS15_Workplace", "Office building, France (2015): employees", FTF, SP),
      ("SFHH_Conference", "Medical conference, Nice, France (2009): attendees", FTF, SP),
      ("HT09_Conference", "Computer science conference, Turin, Italy (2009): attendees", FTF, SP),
      ("Infectious_Museum", "Science museum exhibition, Dublin, Ireland (2009): visitors", FTF, SP),
      ("Malawi_Village", "Rural village, Malawi (2019-20): residents", FTF, SP),
      ("Kenya_Households", "Five rural households, Kenya (2012): family members", FTF, SP),
      ("SouthAfrica_Households", "Households of COVID-19 patients, South Africa (2020-21): family members, "
                                 "with each person's infection status", FTF, SP)]),
    ("B. Proximity networks (wireless sensors and mobile phones)",
     "Sensors or phones recorded when two people were physically near each other.",
     [("Utah_MiddleSchool", "Middle school, Utah, USA (2012): students",
       "Two students were within ~2 m. Weight = total contact time (seconds).", "Toth et al. 2015 [2]"),
      ("Utah_ElementarySchool", "Elementary school, Utah, USA (2013): students",
       "Two students were within ~2 m. Weight = total contact time (seconds).", "Toth et al. 2015 [2]"),
      ("Utah_MiddleSchool_1m", "Same middle school, stricter contact rule",
       "Two students were within ~1 m. Weight = total contact time (seconds).", "Toth et al. 2015 [2]"),
      ("Utah_ElementarySchool_1m", "Same elementary school, stricter contact rule",
       "Two students were within ~1 m. Weight = total contact time (seconds).", "Toth et al. 2015 [2]"),
      ("Copenhagen_BT", "University students, Copenhagen, Denmark (2013)",
       "Two phones detected each other by Bluetooth at close range. Weight = number of 5-minute scans.",
       "Copenhagen Networks Study [3]"),
      ("RealityMining_BT", "MIT students and staff, USA (2004-05)",
       "Two phones detected each other by Bluetooth. Weight = number of detections.", "MIT Reality Mining [4]"),
      ("Haggle_Infocom05", "IEEE Infocom conference (2005): attendees and nearby devices",
       "Two Bluetooth devices were near each other. Weight = total contact time (seconds).", "Haggle project [5]")]),
    ("C. Co-location networks (same people as group A, looser contact rule)",
     "Same sensor data as group A, but an edge only needs both people to be in the same area. "
     "This matters for airborne diseases.",
     [("LH10_Hospital_Coloc", "Hospital ward, Lyon (same people as LH10_Hospital)", COLOC, SP),
      ("InVS13_Workplace_Coloc", "Office building, France 2013 (same people as InVS13)", COLOC, SP),
      ("InVS15_Workplace_Coloc", "Office building, France 2015 (same people as InVS15)", COLOC, SP),
      ("SFHH_Conference_Coloc", "Medical conference, Nice (same people as SFHH)", COLOC, SP),
      ("LyonSchool_Primary_Coloc", "Primary school, Lyon (same people as LyonSchool)", COLOC, SP),
      ("Thiers13_HighSchool_Coloc", "High school, Marseille 2013 (same people as Thiers13)", COLOC, SP)]),
    ("D. Self-reported contact network (survey)",
     "Students wrote down whom they met, instead of using sensors.",
     [("Thiers13_Diaries", "High school, Marseille (2013): students who filled in a contact diary",
       "A student reported meeting the other. Weight = 1 to 4 (reported contact time: "
       "under 5 min, 5-15 min, 15-60 min, over 1 h).", SP)]),
]

SOURCES = [
    "[1] SocioPatterns collaboration, http://www.sociopatterns.org. Downloaded from the Netzschleuder network "
    "repository, https://networks.skewed.de (networks sp_hospital, sp_primary_school, sp_high_school, "
    "sp_high_school_new, sp_office, sp_office_2015, sp_sfhh_conference, sp_hypertext, sp_infectious, "
    "sp_malawi_village, sp_kenyan_households, sp_south_african_households, sp_colocation).",
    "[2] Toth D.J.A. et al. (2015). The role of heterogeneity in contact timing and duration in network models of "
    "influenza spread in schools. Journal of the Royal Society Interface 12: 20150279. "
    "https://doi.org/10.1098/rsif.2015.0279 (supplementary data D1-D10).",
    "[3] Sapiezynski P. et al. (2019). Interaction data from the Copenhagen Networks Study. Scientific Data 6: 315. "
    "https://doi.org/10.1038/s41597-019-0325-x. Downloaded from Netzschleuder (copenhagen).",
    "[4] Eagle N. and Pentland A. (2006). Reality mining: sensing complex social systems. Personal and Ubiquitous "
    "Computing 10(4). Downloaded from Netzschleuder (reality_mining).",
    "[5] Haggle project, Infocom 2005 Bluetooth contact traces (Chaintreau A. et al., 2007, IEEE Transactions on "
    "Mobile Computing). Downloaded from Netzschleuder (contact).",
]

NOTES = [
    "Infectious_Museum covers 69 days of visitors who mostly never met, so only its largest connected part "
    "(410 people) is used.",
    "SouthAfrica_Households is 88 separate small households, not one connected network. It is kept because it is "
    "the only dataset with real COVID-19 infection data, which is useful for checking results.",
    "Groups A and C (and the 2 m and 1 m Utah networks) record the same people with different contact rules, so "
    "we can test whether the model's ranking stays stable when the definition of contact changes.",
]


def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def add_text(p, text, size, bold=False, italic=False):
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    r.font.size = Pt(size)
    r.font.name = FONT
    return r


def cell_text(cell, text, size=9.5, bold=False, italic=False, first=True):
    p = cell.paragraphs[0] if first else cell.add_paragraph()
    p.paragraph_format.space_after = Pt(0)
    add_text(p, text, size, bold, italic)


def doc_par(doc, text, size=10.5, bold=False, align=None, before=0, after=4, style=None):
    p = doc.add_paragraph(style=style)
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    if align is not None:
        p.alignment = align
    add_text(p, text, size, bold)
    if bold:
        p.paragraph_format.keep_with_next = True   # headings stay with the text below
    return p


def row_props(row, header=False):
    trPr = row._tr.get_or_add_trPr()
    cs = OxmlElement("w:cantSplit")          # keep each row on one page
    cs.set(qn("w:val"), "true")
    trPr.append(cs)
    if header:                                # repeat header row on each page
        th = OxmlElement("w:tblHeader")
        th.set(qn("w:val"), "true")
        trPr.append(th)


def main():
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = FONT
    st.font.size = Pt(10)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)

    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width, sec.page_height = Mm(297), Mm(210)          # A4 landscape
    for m in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, m, Mm(15))

    doc_par(doc, "Datasets Used: Human Contact Networks", size=18, bold=True,
            align=WD_ALIGN_PARAGRAPH.CENTER, after=6)
    doc_par(doc, "Application: finding the most influential people (potential super-spreaders) in a contact network, "
                 "so that vaccination, testing or isolation can be targeted at them.")
    doc_par(doc, "In every network, a node is a person and an edge joins two people who were in contact. The edge "
                 "weight says how much contact they had (usually total contact time), so a larger weight means a "
                 "higher chance of passing on an infection. All networks are undirected and weighted.", after=8)

    table = doc.add_table(rows=1, cols=len(COLS))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    hdr = table.rows[0]
    row_props(hdr, header=True)
    for c, name in zip(hdr.cells, COLS):
        cell_text(c, name, size=10, bold=True)
        shade(c, "D9D9D9")

    idx = 0
    for title, expl, items in GROUPS:
        grow = table.add_row()
        row_props(grow)
        merged = grow.cells[0].merge(grow.cells[-1])
        cell_text(merged, title, size=10, bold=True)
        cell_text(merged, expl, size=9.5, italic=True, first=False)
        shade(merged, "F2F2F2")
        for name, what, edge, src in items:
            idx += 1
            r = stats.loc[name]
            row = table.add_row()
            row_props(row)
            vals = [str(idx), name, what, edge, f"{int(r.nodes):,}", f"{int(r.edges):,}", src]
            for j, (c, v) in enumerate(zip(row.cells, vals)):
                cell_text(c, v, bold=(j == 1))

    for row in table.rows:
        for c, w in zip(row.cells, WIDTHS_MM):
            c.width = Mm(w)

    doc_par(doc, "Things to keep in mind", size=11, bold=True, before=10, after=2)
    for t in NOTES:
        doc_par(doc, t, size=10, after=2, style="List Bullet")

    doc_par(doc, "Sources", size=11, bold=True, before=10, after=2)
    for s in SOURCES:
        doc_par(doc, s, size=9, after=3)

    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "Dataset_Summary.docx")
    doc.save(out)
    print(out)


if __name__ == "__main__":
    main()
