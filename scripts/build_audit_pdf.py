"""Build the DermaScan Serving Audit as a print-ready A4 PDF.

    python scripts/build_audit_pdf.py

Every figure in the document is measured, not asserted: the per-class table and
the confusion matrix are read from docs/evaluation_results.json, produced by
scripts/evaluate_model.py over the held-out test split. The "before" column is
the same checkpoint measured under the previous serving configuration
(224 + centre crop), retained here so the comparison stays reproducible.

Requires reportlab and the Windows system fonts registered below; substitute any
serif/sans/mono trio if building elsewhere.
"""

import json
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                                NextPageTemplate, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "DermaScan_Serving_Audit.pdf")
FONTS = "C:/Windows/Fonts"

# ── palette (matches the published artifact) ────────────────────────────
INK = colors.HexColor("#17181c")
INK_SOFT = colors.HexColor("#41454d")
INK_FAINT = colors.HexColor("#7c818b")
RULE = colors.HexColor("#d3d2cd")
RULE_FIRM = colors.HexColor("#a9a8a2")
PANEL = colors.HexColor("#f2f1ee")
PANEL_2 = colors.HexColor("#e6e5e1")
SLATE = colors.HexColor("#2f4858")
GAIN = colors.HexColor("#1a6e53")
GAIN_BG = colors.HexColor("#e3efea")
LOSS = colors.HexColor("#9c3f27")
LOSS_BG = colors.HexColor("#f6e7e1")
AMBER = colors.HexColor("#8a5c0c")
AMBER_BG = colors.HexColor("#f7eeda")
WHITE = colors.white

for name, fn in [("Body", "georgia.ttf"), ("Body-Bold", "georgiab.ttf"),
                 ("Body-Italic", "georgiai.ttf"),
                 ("Display", "arialbd.ttf"), ("Display-Black", "ARIALNB.TTF"),
                 ("Mono", "consola.ttf"), ("Mono-Bold", "consolab.ttf")]:
    pdfmetrics.registerFont(TTFont(name, os.path.join(FONTS, fn)))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic")

PAGE_W, PAGE_H = A4
M_L, M_R, M_T, M_B = 24 * mm, 20 * mm, 22 * mm, 20 * mm
CONTENT_W = PAGE_W - M_L - M_R

S = {
    "body": ParagraphStyle("body", fontName="Body", fontSize=9.4, leading=14.6,
                           textColor=INK, alignment=TA_JUSTIFY, spaceAfter=7),
    "lede": ParagraphStyle("lede", fontName="Body", fontSize=10.6, leading=16.4,
                           textColor=INK_SOFT, spaceAfter=9),
    "h1": ParagraphStyle("h1", fontName="Display", fontSize=21, leading=23.5,
                         textColor=INK, spaceAfter=4),
    "h2": ParagraphStyle("h2", fontName="Display", fontSize=13.4, leading=16.5,
                         textColor=INK, spaceBefore=2, spaceAfter=6),
    "h3": ParagraphStyle("h3", fontName="Display", fontSize=10.2, leading=13,
                         textColor=SLATE, spaceBefore=10, spaceAfter=4),
    "eyebrow": ParagraphStyle("eyebrow", fontName="Mono-Bold", fontSize=7.4, leading=10,
                              textColor=SLATE, spaceAfter=3),
    "caption": ParagraphStyle("caption", fontName="Mono", fontSize=7.4, leading=10.5,
                              textColor=INK_FAINT, spaceBefore=3, spaceAfter=8),
    "callout": ParagraphStyle("callout", fontName="Body", fontSize=9.2, leading=14,
                              textColor=INK, spaceAfter=5),
    "callout_tag": ParagraphStyle("callout_tag", fontName="Mono-Bold", fontSize=7.2,
                                  leading=9.5, textColor=INK_FAINT, spaceAfter=4),
    "bullet": ParagraphStyle("bullet", fontName="Body", fontSize=9.4, leading=14.2,
                             textColor=INK, spaceAfter=5, leftIndent=11,
                             bulletIndent=1, bulletFontName="Body"),
    "cover_title": ParagraphStyle("ct", fontName="Display-Black", fontSize=38, leading=39,
                                  textColor=INK, spaceAfter=10),
    "cover_sub": ParagraphStyle("cs", fontName="Body", fontSize=11.6, leading=17.4,
                                textColor=INK_SOFT, spaceAfter=6),
    "cover_meta": ParagraphStyle("cm", fontName="Mono", fontSize=8.2, leading=13.5,
                                 textColor=INK_FAINT),
    "th": ParagraphStyle("th", fontName="Mono-Bold", fontSize=6.9, leading=8.6,
                         textColor=INK_FAINT),
    "td": ParagraphStyle("td", fontName="Body", fontSize=8.4, leading=11.6, textColor=INK),
}


def para(t, s="body"):
    return Paragraph(t, S[s])


def rule(color=RULE_FIRM, thickness=0.6, space_before=0, space_after=0):
    t = Table([[""]], colWidths=[CONTENT_W], rowHeights=[0.1])
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), thickness, color),
        ("TOPPADDING", (0, 0), (-1, -1), space_before),
        ("BOTTOMPADDING", (0, 0), (-1, -1), space_after),
    ]))
    return t


def section(num, title):
    """Numbered section head. Numbers encode the audit's actual sequence."""
    t = Table([[para(num, "eyebrow"), para(title, "h2")]],
              colWidths=[30 * mm, CONTENT_W - 30 * mm])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEABOVE", (0, 0), (-1, 0), 1.0, INK),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    return KeepTogether([Spacer(1, 12), t, Spacer(1, 5)])


def callout(tag, paras, accent=SLATE, bg=PANEL):
    inner = [para(tag, "callout_tag")] + [para(p, "callout") for p in paras]
    t = Table([[inner]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LINEBEFORE", (0, 0), (0, -1), 2.4, accent),
        ("BOX", (0, 0), (-1, -1), 0.4, RULE_FIRM),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return KeepTogether([Spacer(1, 4), t, Spacer(1, 9)])


def data_table(header, rows, widths, highlight=None, align_left=(0,)):
    """Mono numeric table. `highlight` marks the shipped/current row."""
    data = [[Paragraph(h, S["th"]) for h in header]] + rows
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 1), (-1, -1), "Mono"),
        ("FONTSIZE", (0, 1), (-1, -1), 7.9),
        ("LEADING", (0, 1), (-1, -1), 10.5),
        ("TEXTCOLOR", (0, 1), (-1, -1), INK),
        ("BACKGROUND", (0, 0), (-1, 0), PANEL_2),
        ("ALIGN", (0, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, RULE_FIRM),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, RULE),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE_FIRM),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    for c in align_left:
        style.append(("ALIGN", (c, 0), (c, -1), "LEFT"))
    if highlight is not None:
        r = highlight + 1
        style += [("BACKGROUND", (0, r), (-1, r), GAIN_BG),
                  ("FONTNAME", (0, r), (-1, r), "Mono-Bold")]
    t.setStyle(TableStyle(style))
    return KeepTogether([Spacer(1, 4), t, Spacer(1, 8)])


def delta(v, unit="pp", good_up=True):
    """Signed delta, coloured by whether the movement is an improvement."""
    good = (v > 0) if good_up else (v < 0)
    col = "#1a6e53" if good else "#9c3f27"
    return ('<font name="Mono-Bold" size="7.9" color="%s">%+0.2f %s</font>'
            % (col, v, unit))


def page_furniture(canvas, doc):
    canvas.saveState()
    canvas.setFont("Mono", 7)
    canvas.setFillColor(INK_FAINT)
    canvas.drawString(M_L, M_B - 8 * mm, "DermaScan Serving Audit")
    canvas.drawRightString(PAGE_W - M_R, M_B - 8 * mm, "%d" % doc.page)
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(M_L, M_B - 5.5 * mm, PAGE_W - M_R, M_B - 5.5 * mm)
    canvas.restoreState()


def cover_furniture(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(SLATE)
    canvas.rect(0, PAGE_H - 9 * mm, PAGE_W, 9 * mm, stroke=0, fill=1)
    canvas.restoreState()


def build():
    with open(os.path.join(ROOT, "docs", "evaluation_results.json")) as f:
        ev = json.load(f)
    pc = ev["per_class"]

    before = {
        "akiec": (0.6286, 0.6197, 0.6241), "bcc": (0.7027, 0.7123, 0.7075),
        "bkl": (0.7343, 0.6213, 0.6731), "df": (0.6154, 0.6667, 0.6400),
        "mel": (0.4821, 0.6798, 0.5641), "nv": (0.9253, 0.8809, 0.9026),
        "vasc": (0.8696, 0.8696, 0.8696),
    }

    doc = BaseDocTemplate(OUT, pagesize=A4,
                          leftMargin=M_L, rightMargin=M_R,
                          topMargin=M_T, bottomMargin=M_B,
                          title="DermaScan Serving Audit",
                          author="Engineering audit",
                          subject="Skin lesion classifier: measured before/after review")
    frame = Frame(M_L, M_B, CONTENT_W, PAGE_H - M_T - M_B, id="f",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([
        PageTemplate(id="cover", frames=[frame], onPage=cover_furniture),
        PageTemplate(id="main", frames=[frame], onPage=page_furniture),
    ])

    st = []

    # ── COVER ───────────────────────────────────────────────────────────
    st.append(Spacer(1, 26 * mm))
    st.append(para("ENGINEERING AUDIT &nbsp;/&nbsp; SKIN LESION CLASSIFIER", "eyebrow"))
    st.append(Spacer(1, 5))
    st.append(Paragraph("DermaScan<br/>Serving Audit", S["cover_title"]))
    st.append(rule(INK, 1.6, 4, 10))
    st.append(Paragraph(
        "A review of the inference path and the evidence behind it. The network's weights "
        "were never retrained &mdash; what changed is how the model is fed, how its output "
        "is read, and whether the reported numbers were ever measured.", S["cover_sub"]))
    st.append(Spacer(1, 14 * mm))

    hdr = ["", "BEFORE", "AFTER", "CHANGE"]
    rows = [
        ["Accuracy", "0.8066", "0.8505", Paragraph(delta(4.39), S["td"])],
        ["Macro F1", "0.7116", "0.7450", Paragraph(delta(3.34), S["td"])],
        ["Melanoma F1", "0.5641", "0.6361", Paragraph(delta(7.20), S["td"])],
        ["Melanoma recall", "0.6798", "0.6236", Paragraph(delta(-5.62), S["td"])],
    ]
    w = [CONTENT_W - 3 * 30 * mm, 30 * mm, 30 * mm, 30 * mm]
    t = Table([hdr] + rows, colWidths=w, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Mono-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 7.2),
        ("TEXTCOLOR", (0, 0), (-1, 0), INK_FAINT),
        ("FONTNAME", (0, 1), (0, -1), "Body-Bold"),
        ("FONTSIZE", (0, 1), (0, -1), 9.6),
        ("FONTNAME", (1, 1), (2, -1), "Mono"),
        ("FONTSIZE", (1, 1), (2, -1), 11),
        ("TEXTCOLOR", (1, 1), (1, -1), INK_FAINT),
        ("FONTNAME", (2, 1), (2, -1), "Mono-Bold"),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEABOVE", (0, 0), (-1, 0), 1.0, INK),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, RULE_FIRM),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, RULE),
        ("LINEBELOW", (0, -1), (-1, -1), 1.0, INK),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    st.append(t)
    st.append(Spacer(1, 6))
    st.append(para("Measured on the held-out HAM10000 test split, n=1525. Identical "
                   "checkpoint in both columns.", "caption"))

    st.append(Spacer(1, 20 * mm))
    st.append(Paragraph(
        "Checkpoint &nbsp;models/latest.pt<br/>"
        "Dataset &nbsp;&nbsp;&nbsp;HAM10000 held-out test split, n=1525<br/>"
        "Date &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2 September 2026", S["cover_meta"]))

    st.append(NextPageTemplate("main"))
    st.append(PageBreak())

    # ── 01 PREMISE ──────────────────────────────────────────────────────
    st.append(section("01 / PREMISE", "The weights did not change. Everything around them did."))
    st.append(para(
        "No retraining took place. The served checkpoint is byte-identical to the one "
        "already deployed. Every gain in this report comes from correcting how that "
        "checkpoint is used, and from replacing asserted numbers with measured ones.", "lede"))
    st.append(callout("WHAT ACTUALLY MOVED", [
        "<b>Input resolution.</b> The model was trained and evaluated at 300&times;300 with "
        "no crop. Production resized to 256 and centre-cropped to 224 &mdash; feeding it a "
        "resolution it had never seen and discarding the border of every image. Correcting "
        "this accounts for essentially all of the accuracy gain.",
        "<b>Evidence.</b> The previously published performance figures were not "
        "measurements. They are now."]))
    st.append(para(
        "A second finding inverted a decision midway through the work: the checkpoint in the "
        "training repository (<font name='Mono' size='8.6'>best.pt</font>) is <i>not</i> the "
        "model behind the published metrics. Measured on the same test split it scores "
        "accuracy 0.6570 and macro-F1 0.4464 &mdash; an earlier, weaker run. The deployed "
        "<font name='Mono' size='8.6'>latest.pt</font> is the better model. Swapping to "
        "<font name='Mono' size='8.6'>best.pt</font> would have cost roughly 20 points of "
        "accuracy."))

    # ── 02 MEASURED ─────────────────────────────────────────────────────
    st.append(section("02 / MEASURED", "Per-class performance, before and after"))
    st.append(para(
        "Both columns were produced by running the same checkpoint over all 1525 held-out "
        "test images. <b>Before</b> is 224 with a centre crop; <b>after</b> is 300&times;300, "
        "matching training."))

    header = ["Class", "n", "Prec\nbefore", "Prec\nafter", "Recall\nbefore",
              "Recall\nafter", "F1\nbefore", "F1\nafter", "&Delta; F1"]
    rows = []
    for c in ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]:
        bp, br, bf = before[c]
        a = pc[c]
        rows.append([
            c, str(a["support"]),
            Paragraph('<font name="Mono" size="7.9" color="#7c818b">%.4f</font>' % bp, S["td"]),
            "%.4f" % a["precision"],
            Paragraph('<font name="Mono" size="7.9" color="#7c818b">%.4f</font>' % br, S["td"]),
            "%.4f" % a["recall"],
            Paragraph('<font name="Mono" size="7.9" color="#7c818b">%.4f</font>' % bf, S["td"]),
            "%.4f" % a["f1"],
            Paragraph(delta((a["f1"] - bf) * 100, ""), S["td"]),
        ])
    rows.append(["macro", "1525", "", "", "", "",
                 Paragraph('<font name="Mono-Bold" size="7.9" color="#7c818b">0.7116</font>', S["td"]),
                 Paragraph('<font name="Mono-Bold" size="7.9">%.4f</font>' % ev["macro_f1"], S["td"]),
                 Paragraph(delta(3.34, ""), S["td"])])
    cw = [17 * mm, 10 * mm] + [17.4 * mm] * 6 + [16 * mm]
    t = Table([[Paragraph(h.replace("\n", "<br/>"), S["th"]) for h in header]] + rows,
              colWidths=cw, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 1), (-1, -1), "Mono"),
        ("FONTSIZE", (0, 1), (-1, -1), 7.9),
        ("BACKGROUND", (0, 0), (-1, 0), PANEL_2),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, RULE_FIRM),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, RULE),
        ("LINEABOVE", (0, -1), (-1, -1), 0.7, RULE_FIRM),
        ("BACKGROUND", (0, -1), (-1, -1), PANEL),
        ("FONTNAME", (0, -1), (-1, -1), "Mono-Bold"),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE_FIRM),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    st.append(Spacer(1, 4))
    st.append(t)
    st.append(Spacer(1, 7))
    st.append(para(
        "Six of seven classes improve. <font name='Mono' size='8.6'>df</font> regresses, but "
        "with only 12 test cases a single reclassification moves its F1 by roughly 8 points "
        "&mdash; that row is noise, not signal. The substantive movements are "
        "<font name='Mono' size='8.6'>akiec</font> (+12.0) and melanoma precision "
        "(0.4821 <font name='Mono'>&rarr;</font> 0.6491)."))

    cm_path = os.path.join(ROOT, "docs", "confusion_matrix_measured.png")
    if os.path.exists(cm_path):
        st.append(Spacer(1, 4))
        img = Image(cm_path)
        img._restrictSize(CONTENT_W * 0.76, 104 * mm)
        img.hAlign = "LEFT"
        st.append(KeepTogether([img, para(
            "Measured confusion matrix, n=1525. Counts in parentheses, row-normalised "
            "shading. Melanoma error concentrates in the nv column.", "caption")]))

    # ── 03 TRADEOFF ─────────────────────────────────────────────────────
    st.append(section("03 / TRADEOFF", "Melanoma recall was bought down, deliberately"))
    st.append(callout("THE ONE METRIC THAT GOT WORSE", [
        "Melanoma recall fell from <b>0.6798 to 0.6236</b>. In absolute terms the system now "
        "misses about <b>38% of melanomas</b> in the test set, most of them misread as benign "
        "nevi. This is the dominant clinical risk in the product, and no interface work "
        "changes it."], AMBER, AMBER_BG))
    st.append(para(
        "Melanoma F1 nonetheless improved by 7.2 points, because precision rose far more than "
        "recall fell &mdash; the previous configuration bought its recall by over-flagging. "
        "Threshold re-optimisation was run to test whether the recall could be recovered for "
        "free. It cannot. Fitted on the calibration split (n=997) and reported on test, no "
        "configuration dominates:"))
    st.append(data_table(
        ["Thresholds", "Accuracy", "Macro F1", "Mel recall", "Mel F1"],
        [["Current (shipped)", "0.8505", "0.7450", "0.6236", "0.6361"],
         ["Refit for macro F1", "0.8557", "0.7551", "0.5000", "0.6075"],
         ["Refit for mel recall", "0.8308", "0.7367", "0.6798", "0.6111"]],
        [CONTENT_W - 4 * 25 * mm] + [25 * mm] * 4, highlight=0))
    st.append(para(
        "The shipped thresholds already sit on the efficient frontier and hold the best "
        "melanoma F1, so they were left unchanged. Sweeping the melanoma threshold alone "
        "isolates the exchange rate &mdash; all other thresholds held at the macro-F1 fit, so "
        "these rows are not the shipped configuration:"))
    st.append(data_table(
        ["mel threshold", "Accuracy", "Mel recall", "Mel F1"],
        [["0.20", "0.8256", "0.7416", "0.5986"],
         ["0.25", "0.8348", "0.7135", "0.6135"],
         ["0.30", "0.8393", "0.6798", "0.6189"],
         ["0.35", "0.8446", "0.6404", "0.6230"],
         ["0.40", "0.8531", "0.6180", "0.6377"],
         ["0.45", "0.8584", "0.5843", "0.6480"],
         ["0.50", "0.8557", "0.5000", "0.6075"]],
        [CONTENT_W - 3 * 28 * mm] + [28 * mm] * 3))
    st.append(para(
        "Roughly <b>one point of accuracy per three to four points of melanoma recall</b>, "
        "monotone. Choosing a different operating point is a one-line change to the "
        "<font name='Mono' size='8.6'>mel</font> entry in "
        "<font name='Mono' size='8.6'>class_thresholds.json</font>. It is a clinical "
        "judgement about the cost of a missed melanoma versus a false alarm &mdash; not a "
        "tuning detail, and not one that should be made silently."))

    # ── 04 INTEGRITY ────────────────────────────────────────────────────
    st.append(PageBreak())
    st.append(section("04 / INTEGRITY", "Fabricated outputs, removed"))
    st.append(para(
        "Five separate places generated convincing artefacts that were not derived from "
        "anything. In a diagnostic tool this is the most serious class of defect present, "
        "because each one is indistinguishable from the real thing at a glance.", "lede"))

    pairs = [
        ("Evaluation figures",
         "evaluate_model.py never imported torch, loaded the checkpoint, or read a dataset. "
         "Confusion-matrix off-diagonals were np.random.dirichlet noise; loss and F1 curves "
         "were hardcoded exponentials tuned to land near 0.73.",
         "Rewritten to run the served model over a labelled hold-out set and report what it "
         "measures. Requires --data-dir; produces nothing without data. The confusion matrix "
         "in these docs is now from 1525 real images."),
        ("Grad-CAM attribution",
         "On any failure, both server and client drew a radial gradient centred on the image "
         "and presented it as model attribution &mdash; a picture of \"the model looked at the "
         "middle\", regardless of what the model did.",
         "Failure returns null, the canvas stays empty, and the UI reports attribution as "
         "unavailable. A lock was added around hook state that was shared across concurrent "
         "requests and could swap heatmaps between images."),
        ("Clinical sample gallery",
         "Five Unsplash stock photographs labelled as melanoma, BCC, AKIEC and others. Real "
         "HAM10000 images sat unused in the repository and were never served.",
         "Real dermoscopic images served from /samples, with labels sourced from the training "
         "split manifests. ISIC_0024307 is confirmed ground-truth nv."),
        ("Submitted image",
         "If a sample image failed to fetch &mdash; likely, since the stock photos were "
         "cross-origin &mdash; the client hand-drew a brown ellipse on a skin-coloured canvas "
         "and submitted it. The user received a diagnosis of a drawing, stored as a clinical "
         "record.",
         "Removed entirely. A failed fetch raises an error. Nothing is analysed except the "
         "image the user actually supplied."),
        ("Model loading",
         "A failed weight load printed the error and continued, serving confident predictions "
         "from a randomly initialised network. Architecture was built inline, so a mismatched "
         "checkpoint could pass unnoticed.",
         "Load failure is fatal. Architecture is selected explicitly via MODEL_ARCH and loaded "
         "strictly &mdash; this is what surfaced the best.pt mix-up rather than silently "
         "serving the wrong network."),
    ]
    half = (CONTENT_W - 4) / 2
    for title, b, a in pairs:
        head = Table([[Paragraph(title, S["h3"])]], colWidths=[CONTENT_W])
        head.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), PANEL_2),
            ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("BOX", (0, 0), (-1, -1), 0.5, RULE_FIRM),
        ]))
        cells = Table([[
            [Paragraph("BEFORE", ParagraphStyle("kb", parent=S["callout_tag"], textColor=LOSS)),
             Paragraph(b, S["callout"])],
            [Paragraph("AFTER", ParagraphStyle("ka", parent=S["callout_tag"], textColor=GAIN)),
             Paragraph(a, S["callout"])],
        ]], colWidths=[half, half])
        cells.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (0, -1), LOSS_BG),
            ("BACKGROUND", (1, 0), (1, -1), GAIN_BG),
            ("BOX", (0, 0), (-1, -1), 0.5, RULE_FIRM),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, RULE_FIRM),
            ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        st.append(KeepTogether([Spacer(1, 6), head, cells]))

    # ── 05 FAIRNESS ─────────────────────────────────────────────────────
    st.append(PageBreak())
    st.append(section("05 / FAIRNESS", "The OOD gate rejected dark images, not non-skin images"))
    st.append(para(
        "The gate keyed on absolute channel standard deviation, which <b>scales with "
        "brightness</b>. Identical lesions were accepted or rejected purely on how dark the "
        "image was. Measured on the repository's own sample: ISIC_0024307 darkened to 35% "
        "brightness was rejected as <font name='Mono' size='8.6'>too_uniform</font> while the "
        "same image at full brightness passed. That is a direct bias against darker skin "
        "tones, underexposure, and poor lighting."))
    st.append(para(
        "Two further rules were actively wrong: grayscale input was rejected outright, "
        "excluding legitimate grayscale dermoscopy; and an upper bound on standard deviation "
        "rejected <i>high-contrast</i> images &mdash; a dark lesion on pale skin, the "
        "presentation of most concern."))
    st.append(data_table(
        ["Metric", "Property", "Real dermoscopy", "Noise / flat"],
        [["rel_contrast", "std / mean, illumination-invariant", "0.12 - 0.44", "0.00"],
         ["hf_ratio", "high-frequency residual share", "0.04 - 0.16", "0.92+"],
         ["blue_green", "hue, chromatic pixels only", "0.00", "1.00"]],
        [26 * mm, CONTENT_W - 26 * mm - 2 * 27 * mm, 27 * mm, 27 * mm],
        align_left=(0, 1)))
    st.append(para(
        "Every replacement metric is a ratio, so brightness cancels: "
        "<font name='Mono' size='8.6'>rel_contrast</font> holds at 0.119 <font name='Mono'>&rarr;</font> 0.120 across "
        "the full darkening range that previously triggered rejection. On a 21-case suite of "
        "skin images (darkened, grayscale, contrast-boosted) plus non-skin controls, "
        "<b>the old gate produced 6 incorrect verdicts; the new gate produces 1</b>."))
    st.append(callout("FINDING WORTH PRESERVING", [
        "Model confidence is unusable as an OOD signal on this checkpoint. A blank white field "
        "scores max-softmax <b>0.994</b> and a <i>more</i> in-distribution energy score "
        "(&minus;3.87) than any real lesion image (&minus;2.23 to &minus;3.20). The textbook "
        "fix would have made this worse; measurement caught it. A feature-space Mahalanobis "
        "stage was added instead &mdash; it rejects that same white field by orders of "
        "magnitude &mdash; but it stays inactive until fitted on real data."]))

    # ── 06 EXPOSURE ─────────────────────────────────────────────────────
    st.append(section("06 / EXPOSURE", "Security and robustness"))
    sec_rows = [
        ["Bulk delete", "Unauthenticated DELETE /history/all wiped every record and image",
         "Requires X-Admin-Token; disabled (403) when unset"],
        ["CORS", "allow_origins=[\"*\"] with credentials", "Env-driven allowlist, localhost default"],
        ["Uploads", "Client extension used as a path component; no size cap; content_type trusted",
         "Extension allowlist, 10 MB cap, validated by decoding"],
        ["Memory", "Two predictor instances - model loaded twice", "Single lazily-built instance"],
        ["Concurrency", "Grad-CAM hook state shared across threadpool requests",
         "Serialised behind a lock"],
    ]
    body_w = (CONTENT_W - 24 * mm) / 2
    t = Table([[Paragraph("Area", S["th"]), Paragraph("Before", S["th"]),
                Paragraph("After", S["th"])]] +
              [[Paragraph('<font name="Mono" size="7.8">%s</font>' % r[0], S["td"]),
                Paragraph('<font color="#7c818b">%s</font>' % r[1], S["td"]),
                Paragraph(r[2], S["td"])] for r in sec_rows],
              colWidths=[24 * mm, body_w, body_w], repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PANEL_2),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, RULE_FIRM),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, RULE),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE_FIRM),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    st.append(Spacer(1, 4))
    st.append(t)

    # ── 07 OPEN ─────────────────────────────────────────────────────────
    st.append(section("07 / OPEN", "What this audit did not fix"))
    for b in [
        "<b>Melanoma recall is the product's ceiling.</b> At 0.624, roughly two in five "
        "melanomas are missed. No configuration on the measured frontier escapes this; it "
        "needs a better model, not better serving.",
        "<b>Training curves are unrecoverable.</b> Checkpoints store a single epoch's metrics "
        "and no history log survives. They cannot be honestly reconstructed &mdash; "
        "reconstructing them is precisely what the deleted script did.",
        "<b>The feature-space OOD stage is unfitted.</b> Until calibrate_ood.py is run on real "
        "data, the system accepts some non-clinical photographs. Colour statistics alone "
        "cannot reject a desaturated animal photo.",
        "<b>Thresholds remain off-protocol.</b> They were fitted upstream for a softmax "
        "readout; the shipped pairing is sigmoid. It measures best on melanoma F1, but the "
        "pairing is empirical rather than principled.",
        "<b>The classifier is brightness-sensitive.</b> The gate no longer discriminates by "
        "brightness, but the model's own answer still shifts under darkening. Evaluating "
        "across Fitzpatrick skin types (e.g. the DDI dataset) is the real test and has not "
        "been done.",
    ]:
        st.append(Paragraph(b, S["bullet"], bulletText="\u2014"))

    st.append(callout("INCIDENT DURING THIS WORK", [
        "An end-to-end verification run was pointed at the live database rather than a "
        "temporary one, deleting 21 stored diagnostic sessions and their uploaded images. Both "
        "were gitignored and are unrecoverable. A pytest suite with a temporary-database "
        "fixture has since been added so this cannot recur."], AMBER, AMBER_BG))

    st.append(Spacer(1, 8))
    st.append(rule(INK, 1.2, 0, 6))
    st.append(Paragraph(
        "12 files modified &nbsp;/&nbsp; 8 added &nbsp;/&nbsp; 5 fabricated figures removed "
        "&nbsp;/&nbsp; 31 tests<br/>"
        "New tooling: evaluate_model.py, calibrate_ood.py, optimize_thresholds.py<br/>"
        "All figures measured on the HAM10000 test split, n=1525.", S["cover_meta"]))

    doc.build(st)
    print("Wrote", OUT, "(%.1f KB)" % (os.path.getsize(OUT) / 1024))


if __name__ == "__main__":
    build()
