"""
Detailed layman-friendly pipeline diagram for the Zero-Shot AMR prediction project.
Generates a high-resolution PDF + PNG. No emoji — fully portable fonts.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

fig = plt.figure(figsize=(26, 36))
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 26)
ax.set_ylim(0, 36)
ax.axis("off")
fig.patch.set_facecolor("#F0F4F8")

# ── Palette ───────────────────────────────────────────────────────────────────
C = {
    "hdr":    "#1B2A4A",  "hdr_fg": "#FFFFFF",  "hdr_sub": "#90AFC5",
    "s1_bg":  "#DCEEFB",  "s1_bdr": "#1565C0",
    "s2_bg":  "#D4EDDA",  "s2_bdr": "#2E7D32",
    "s3_bg":  "#FFF3CD",  "s3_bdr": "#E65100",
    "s4_bg":  "#EDE0FF",  "s4_bdr": "#6A1B9A",
    "s5_bg":  "#FFD6E0",  "s5_bdr": "#AD1457",
    "s6_bg":  "#C8F7F3",  "s6_bdr": "#00695C",
    "s7_bg":  "#FFF9C4",  "s7_bdr": "#F57F17",
    "s8_bg":  "#E8DDD5",  "s8_bdr": "#4E342E",
    "panel":  "#FFFFFF",  "arrow":  "#37474F",
    "foot":   "#243354",  "foot_fg":"#CFD8DC",
    "random": "#F9A825",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def rbox(x, y, w, h, fc, ec, lw=2.5, r=0.35, zorder=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder))

def panel(x, y, w, h, fc=None, ec=None, lw=1.4):
    fc = fc or C["panel"]; ec = ec or "#BBBBBB"
    rbox(x, y, w, h, fc, ec, lw=lw, r=0.18, zorder=4)

def txt(x, y, s, fs=10, color="#1A1A1A", bold=False, ha="center", va="center",
        zorder=5, style="normal", alpha=1.0):
    ax.text(x, y, s, fontsize=fs, color=color,
            fontweight="bold" if bold else "normal",
            ha=ha, va=va, zorder=zorder, style=style, alpha=alpha,
            multialignment="center")

def badge(cx, cy, r_=0.38, color="#333", label_text="1"):
    ax.add_patch(plt.Circle((cx, cy), r_, color=color, zorder=6))
    txt(cx, cy, label_text, fs=13, color="white", bold=True, zorder=7)

def bullet_list(bx, by, items, fs=9.3, gap=0.37, color="#333333"):
    for i, item in enumerate(items):
        txt(bx, by - i*gap, "  *  " + item, fs=fs, color=color,
            ha="left", va="center", zorder=5)

def down_arrow(x, y1, y2, lw=3.2):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="->,head_width=0.32,head_length=0.20",
                                color=C["arrow"], lw=lw),
                zorder=9)

def step_frame(x, y, w, h, num, title, subtitle, bg, bdr):
    rbox(x, y, w, h, bg, bdr, lw=3, r=0.4)
    badge(x+0.55, y+h-0.55, color=bdr, label_text=str(num))
    txt(x+1.15, y+h-0.56, title, fs=14.5, color=bdr, bold=True, ha="left")
    txt(x+1.15, y+h-1.08, subtitle, fs=9.5, color="#555555", ha="left", style="italic")

def tag(x, y, text, color):
    """Small coloured pill label."""
    rbox(x-0.05, y-0.15, len(text)*0.115+0.1, 0.3, color, color, lw=0, r=0.1, zorder=6)
    txt(x + len(text)*0.057, y, text, fs=8, color="white", bold=True, zorder=7)

# ═══════════════════════════════════════════════════════════════════════════════
# TITLE BANNER
# ═══════════════════════════════════════════════════════════════════════════════
rbox(0.5, 34.15, 25, 1.6, C["hdr"], C["hdr"], lw=0, r=0.5, zorder=2)
txt(13, 35.1, "Zero-Shot Antimicrobial Resistance (AMR) Prediction Pipeline",
    fs=19.5, color="white", bold=True)
txt(13, 34.52,
    "How our AI learns from known antibiotics and predicts resistance to brand-new ones  --  step by step",
    fs=11.5, color=C["hdr_sub"], style="italic")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1  —  RAW DATA
# ═══════════════════════════════════════════════════════════════════════════════
step_frame(0.5, 31.5, 25, 2.4, 1,
           "Raw Data  —  CARD Database (v4.0.1)",
           "Where we start: a massive, peer-reviewed catalogue of antibiotic resistance facts",
           C["s1_bg"], C["s1_bdr"])

panel(1.0, 31.75, 7.0, 1.6, "#EBF5FB", C["s1_bdr"])
txt(4.5, 33.06, "[GENE]  6,397 Resistance Genes", fs=11, color=C["s1_bdr"], bold=True)
bullet_list(1.3, 32.7,
    ["Genes that make bacteria resistant to antibiotics",
     "e.g.  blaZ, mecA, tetM, ermB, vanA ...",
     "Each gene has a name + mechanism description"],
    fs=9.3, gap=0.34)

panel(8.4, 31.75, 7.0, 1.6, "#EBF5FB", C["s1_bdr"])
txt(11.9, 33.06, "[DRUG]  46 Drug Classes", fs=11, color=C["s1_bdr"], bold=True)
bullet_list(8.7, 32.7,
    ["Groups of antibiotics  (penicillins, tetracyclines ...)",
     "Each class has a biological target",
     "Encoded as a 97-dimensional feature vector"],
    fs=9.3, gap=0.34)

panel(15.8, 31.75, 9.1, 1.6, "#EBF5FB", C["s1_bdr"])
txt(20.35, 33.06, "[LINK]  12,593 Resistance Edges", fs=11, color=C["s1_bdr"], bold=True)
bullet_list(16.1, 32.7,
    ["Each edge = 'Gene X makes bacteria resistant to Drug Class Y'",
     "These are the ground-truth labels for training",
     "Verified by expert curators -- no fabrication"],
    fs=9.3, gap=0.34)

down_arrow(13, 31.5, 31.05)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2  —  BUILD THE KNOWLEDGE GRAPH
# ═══════════════════════════════════════════════════════════════════════════════
step_frame(0.5, 28.1, 25, 2.65, 2,
           "Build the Knowledge Graph",
           "Connect genes, drugs and resistance mechanisms into one big network",
           C["s2_bg"], C["s2_bdr"])

panel(1.0, 28.35, 7.8, 2.1, "#F0FBF4", C["s2_bdr"])
txt(4.9, 30.15, "City-Map Analogy", fs=11, color=C["s2_bdr"], bold=True)
bullet_list(1.3, 29.8,
    ["Genes     =  Buildings  (6,397 buildings)",
     "Drug Classes  =  Districts  (46 districts)",
     "Mechanisms  =  Road types  (8 road types)",
     "Resistance link  =  Building connected to District",
     "Graph lets AI learn neighbourhood patterns"],
    fs=9.3, gap=0.35)

panel(9.2, 28.35, 7.4, 2.1, "#F0FBF4", C["s2_bdr"])
txt(12.9, 30.15, "Node Feature Vectors", fs=11, color=C["s2_bdr"], bold=True)
bullet_list(9.5, 29.8,
    ["Gene node  -->  154-dim feature vector",
     "   (drug-class memberships + mechanism flags + TF-IDF)",
     "Drug class node  -->  97-dim feature vector",
     "   (target type + 1-hot class + TF-IDF description)",
     "Mechanism node  -->  8 resistance mechanism types"],
    fs=9.3, gap=0.35)

panel(17.0, 28.35, 7.5, 2.1, "#F0FBF4", C["s2_bdr"])
txt(20.75, 30.15, "Why Feature Vectors?", fs=11, color=C["s2_bdr"], bold=True)
bullet_list(17.3, 29.8,
    ["Numbers describe each drug's properties",
     "A NEW drug can also be described this way",
     "   -- even if it was never in the training data!",
     "This is the KEY TRICK that enables prediction",
     "   for brand-new antibiotics (zero-shot)"],
    fs=9.3, gap=0.35)

down_arrow(13, 28.1, 27.65)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3  —  ZERO-SHOT SPLIT
# ═══════════════════════════════════════════════════════════════════════════════
step_frame(0.5, 24.85, 25, 2.5, 3,
           "Zero-Shot Split  --  Hide 5 Drug Classes from Training",
           "Simulate a real scenario: 5 drug classes are treated as 'new' and completely hidden",
           C["s3_bg"], C["s3_bdr"])

panel(1.0, 25.1, 10.5, 2.0, "#FFF8E1", C["s3_bdr"])
txt(6.25, 26.8, "The Simulation", fs=11, color=C["s3_bdr"], bold=True)
bullet_list(1.3, 26.45,
    ["We pretend 5 drug classes do not exist during training",
     "The model must predict resistance to them from scratch",
     "Selected to span different levels of biological similarity",
     "This mimics a real new antibiotic entering clinical use"],
    fs=9.3, gap=0.35)

panel(11.9, 25.1, 12.6, 2.0, "#FFF8E1", C["s3_bdr"])
txt(18.2, 26.8, "The 5 Hidden Drug Classes  (gene overlap with a related training class)",
    fs=10, color=C["s3_bdr"], bold=True)

rows3 = [
    ("Glycylcycline",      "Tetracycline",    "100%", "#4CAF50"),
    ("Streptogramin A",    "Streptogramin",   "100%", "#4CAF50"),
    ("Lincosamide",        "Macrolide",        "56%", "#FF9800"),
    ("Rifamycin",          "Fluoroquinolone",  "47%", "#FF9800"),
    ("Glycopeptide",       "beta-lactam",       "1%", "#F44336"),
]
cx3 = [12.2, 16.2, 20.8]
ax.text(cx3[0], 26.45, "Hidden Drug Class", fontsize=8.8, fontweight="bold", color="#333", zorder=5, va="center")
ax.text(cx3[1], 26.45, "Related Training Class", fontsize=8.8, fontweight="bold", color="#333", zorder=5, va="center")
ax.text(cx3[2], 26.45, "Gene Overlap", fontsize=8.8, fontweight="bold", color="#333", zorder=5, va="center")
for i, (dc, rel, ov, col) in enumerate(rows3):
    ry = 26.1 - i*0.27
    ax.text(cx3[0], ry, dc, fontsize=8.8, color="#222", zorder=5, va="center")
    ax.text(cx3[1], ry, rel, fontsize=8.8, color="#555", zorder=5, va="center")
    rbox(cx3[2]-0.1, ry-0.12, 1.35, 0.25, col, col, lw=0, r=0.07, zorder=4)
    ax.text(cx3[2]+0.57, ry, ov, fontsize=8.8, color="white", fontweight="bold",
            ha="center", va="center", zorder=5)

down_arrow(13, 24.85, 24.4)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4  —  TRAIN THE MODELS
# ═══════════════════════════════════════════════════════════════════════════════
step_frame(0.5, 20.7, 25, 3.4, 4,
           "Train the Models   (5 models  x  5 random seeds  =  25 training runs)",
           "Teach each AI on the 41 known drug classes, then test on the 5 hidden ones",
           C["s4_bg"], C["s4_bdr"])

models4 = [
    ("ZS-HetGAT\n(Ours)", C["s4_bdr"], "#F3E5F5",
     "Graph Attention Network\nwith drug feature encoder\nZero-shot capable"),
    ("R-GCN\n(Best ZS!)", "#1565C0", "#E3F2FD",
     "Graph Conv. Network\nwith linear drug projection\nAlso zero-shot capable"),
    ("Feature-MLP\n(No Graph)", "#2E7D32", "#E8F5E9",
     "Pure neural network on\nfeature numbers only\nNo graph structure"),
    ("DistMult\n(Transductive)", "#C62828", "#FFEBEE",
     "Classic KG embedding\nLearns lookup tables\nFails on new drugs!"),
    ("TransE\n(Transductive)", "#4527A0", "#EDE7F6",
     "Translational KG embed.\nLearns lookup tables\nFails on new drugs!"),
]

bw4, bx4 = 4.56, 0.82
for i, (name, bdr, bgc, desc) in enumerate(models4):
    bxi = bx4 + i*(bw4+0.3)
    panel(bxi, 20.95, bw4, 2.8, bgc, bdr, lw=2.2)
    txt(bxi+bw4/2, 23.45, name, fs=10.5, color=bdr, bold=True)
    txt(bxi+bw4/2, 22.55, desc, fs=8.8, color="#333")
    # seed dots
    for s in range(5):
        ax.add_patch(plt.Circle((bxi+0.5+s*0.78, 21.25), 0.16, color=bdr, alpha=0.7, zorder=5))
    txt(bxi+bw4/2, 21.25, "                 x5 seeds", fs=8, color="#666")

# Training config strip
rbox(0.82, 20.72, 24.36, 0.22, "#E9D5FF", C["s4_bdr"], lw=1, r=0.08, zorder=4)
txt(13, 20.83,
    "AdamW optimiser  |  Cosine-annealing LR  |  Early stopping (patience 30 epochs)  |  "
    "Contrastive drug-class regularisation  |  Binary cross-entropy loss",
    fs=8, color="#4A148C")

down_arrow(13, 20.7, 20.25)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5  —  EVALUATE
# ═══════════════════════════════════════════════════════════════════════════════
step_frame(0.5, 16.6, 25, 3.35, 5,
           "Evaluate  --  How Well Does Each Model Predict?",
           "Two tests: (a) drug classes seen during training   and   (b) the 5 hidden zero-shot classes",
           C["s5_bg"], C["s5_bdr"])

panel(1.0, 16.85, 7.5, 2.8, "#FFF0F3", C["s5_bdr"])
txt(4.75, 19.35, "What We Measure", fs=11, color=C["s5_bdr"], bold=True)
bullet_list(1.3, 18.95,
    ["MRR  =  Mean Reciprocal Rank",
     "   If true gene is ranked #1  ->  score = 1.0",
     "   If ranked #2  ->  0.5,  ranked #5  ->  0.2",
     "   Random MRR for 5-class zero-shot test  =  0.477",
     "Hits@10  =  Is the true gene in top 10?",
     "AUC  =  Area under ROC curve (1.0 = perfect)"],
    fs=9.3, gap=0.38)

panel(9.0, 16.85, 15.5, 2.8, "#FFF0F3", C["s5_bdr"])
txt(16.75, 19.35, "Zero-Shot MRR Results   (5 seeds, mean +/- std)",
    fs=11, color=C["s5_bdr"], bold=True)

res5 = [
    ("R-GCN",           "0.775 +/- 0.013", "BEST -- beats random by +62%",  "#2E7D32"),
    ("ZS-HetGAT (ours)","0.543 +/- 0.054", "Above random",                  "#1565C0"),
    ("Feature-MLP",     "0.476 +/- ---",   "~= random (no graph help)",      "#795548"),
    ("DistMult",        "0.428 +/- 0.039", "BELOW random -- fails!",         "#C62828"),
    ("TransE",          "0.325 +/- 0.064", "Well below random -- fails!",    "#4527A0"),
    ("Random baseline", "0.477",           "-- chance level --",             "#777777"),
]
cx5 = [9.3, 13.7, 17.6]
ax.text(cx5[0], 18.95, "Model", fontsize=9, fontweight="bold", color="#333", zorder=5, va="center")
ax.text(cx5[1], 18.95, "ZS-MRR", fontsize=9, fontweight="bold", color="#333", zorder=5, va="center")
ax.text(cx5[2], 18.95, "Verdict", fontsize=9, fontweight="bold", color="#333", zorder=5, va="center")
for i, (m, mrr, verdict, col) in enumerate(res5):
    ry = 18.55 - i*0.36
    ax.text(cx5[0], ry, m, fontsize=8.8, color="#222", zorder=5, va="center")
    ax.text(cx5[1], ry, mrr, fontsize=8.8, color=col, fontweight="bold", zorder=5, va="center")
    ax.text(cx5[2], ry, verdict, fontsize=8.5, color=col, zorder=5, va="center")

# Random baseline dashed line
ax.plot([9.2, 24.3], [17.03, 17.03], color=C["random"], lw=1.4, linestyle="--", zorder=5, alpha=0.8)
ax.text(24.35, 17.03, "random", fontsize=7.8, color=C["random"], va="center", zorder=5)

down_arrow(13, 16.6, 16.15)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6  —  ABLATION STUDY
# ═══════════════════════════════════════════════════════════════════════════════
step_frame(0.5, 12.55, 25, 3.3, 6,
           "Ablation Study  --  What Matters Most?",
           "Switch off one component at a time to discover which parts are essential",
           C["s6_bg"], C["s6_bdr"])

ablations6 = [
    ("A.  Full Model\n(baseline)",
     "ZS-MRR: 0.568", "#00695C", True,
     "Complete model with\nall components enabled."),
    ("B.  Remove\nGraph Structure",
     "ZS-MRR: 0.499", "#E65100", False,
     "No message-passing.\nDrops near random."),
    ("C.  Lookup Drug\nEmbeddings",
     "ZS-MRR: 0.000 !!!", "#C62828", False,
     "Learned tables instead\nof features. Zero-shot\ncompletely breaks!"),
    ("D.  Single GAT\nLayer  (winner)",
     "ZS-MRR: 0.634", "#1565C0", True,
     "Simpler model is\nBETTER at zero-shot.\nLess overfitting!"),
    ("E.  No Mechanism\nNodes",
     "ZS-MRR: 0.616", "#6A1B9A", True,
     "Still above baseline;\nmechanism nodes help\nbut not critical."),
]

bw6, bx6 = 4.56, 0.82
for i, (name, result, col, good, insight) in enumerate(ablations6):
    bxi = bx6 + i*(bw6+0.3)
    bgcol = "#E0F7F4" if good else "#FFEBEE"
    panel(bxi, 12.8, bw6, 2.55, bgcol, col, lw=2.2)
    txt(bxi+bw6/2, 15.03, name, fs=10, color=col, bold=True)
    txt(bxi+bw6/2, 14.27, result, fs=10, color=col, bold=True)
    txt(bxi+bw6/2, 13.38, insight, fs=8.5, color="#333")

# Key insight strip
rbox(0.82, 12.57, 24.36, 0.22, "#B2EBE6", C["s6_bdr"], lw=1, r=0.08, zorder=4)
txt(13, 12.68,
    "KEY FINDING: Graph structure (message-passing) is essential. "
    "Lookup tables completely break zero-shot. Simpler depth = better zero-shot generalisation.",
    fs=8.5, color="#004D40", bold=True)

down_arrow(13, 12.55, 12.1)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7  —  CASE STUDY (LINCOSAMIDE)
# ═══════════════════════════════════════════════════════════════════════════════
step_frame(0.5, 8.65, 25, 3.15, 7,
           "Case Study  --  Lincosamide Antibiotic (Never Seen in Training)",
           "A real example: can the model identify the correct resistance genes for a completely new drug?",
           C["s7_bg"], C["s7_bdr"])

panel(1.0, 8.9, 8.0, 2.6, "#FFFDE7", C["s7_bdr"])
txt(5.0, 11.17, "The Scenario", fs=11, color=C["s7_bdr"], bold=True)
bullet_list(1.3, 10.8,
    ["Lincosamide antibiotics were 100% HIDDEN from training",
     "Model has zero direct knowledge of lincosamide resistance",
     "Task: rank all 6,397 genes by likely resistance",
     "Model leverages: drug feature vector + macrolide",
     "   neighbour information (56% gene overlap in CARD)",
     "Mechanism nodes propagate cross-resistance signal"],
    fs=9.3, gap=0.36)

panel(9.4, 8.9, 7.1, 2.6, "#FFFDE7", C["s7_bdr"])
txt(12.95, 11.17, "Result: Top-20 Predictions", fs=11, color=C["s7_bdr"], bold=True)
txt(12.95, 10.55, "P@20  =  90%", fs=24, color="#E65100", bold=True)
txt(12.95, 9.95, "18 out of 20 predicted genes\nare REAL resistance genes!", fs=10, color="#333")
txt(12.95, 9.25, "erm family, vmlR, lnuA, lnuB ...\n(MLSB cross-resistance genes)",
    fs=8.8, color="#555", style="italic")

panel(16.9, 8.9, 7.6, 2.6, "#FFFDE7", C["s7_bdr"])
txt(20.7, 11.17, "Why Does This Work?", fs=11, color=C["s7_bdr"], bold=True)
bullet_list(17.2, 10.8,
    ["Biology: lincosamide and macrolides share 56%",
     "   of resistance genes (MLSB phenotype)",
     "Graph message-passing propagates signal",
     "   through shared mechanism nodes",
     "Drug feature vectors encode target similarity",
     "   --> model transfers macrolide knowledge"],
    fs=9.3, gap=0.36)

down_arrow(13, 8.65, 8.2)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8  —  OUTPUTS & PAPER
# ═══════════════════════════════════════════════════════════════════════════════
step_frame(0.5, 4.95, 25, 2.95, 8,
           "Output  --  Publishable Results & IEEE Conference Paper",
           "Everything generated automatically; paper written in IEEEtran LaTeX format",
           C["s8_bg"], C["s8_bdr"])

outputs8 = [
    ("8 Figures\n(PDF + PNG)",
     "Bar charts, radar,\nheatmaps, training curves,\nablation charts"),
    ("4 Result Tables\n(LaTeX + CSV)",
     "Standard test / zero-shot\nper-class breakdown\nablation table"),
    ("25 Model\nCheckpoints (.pt)",
     "5 models x 5 seeds\nFully reproducible\nAll weights saved"),
    ("Gene Analysis\nJSON",
     "Top predicted genes\nfor each hidden class\nwith precision scores"),
    ("Full IEEE\nConference Paper",
     "8-section IEEEtran paper\nAll real results included\nReady for Overleaf / KDD"),
]

bw8, bx8 = 4.56, 0.82
for i, (name, desc) in enumerate(outputs8):
    bxi = bx8 + i*(bw8+0.3)
    panel(bxi, 5.2, bw8, 2.25, C["s8_bg"], C["s8_bdr"], lw=1.8)
    txt(bxi+bw8/2, 7.1, name, fs=10.5, color=C["s8_bdr"], bold=True)
    txt(bxi+bw8/2, 6.15, desc, fs=9, color="#444")

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER  --  BIG PICTURE
# ═══════════════════════════════════════════════════════════════════════════════
rbox(0.5, 0.4, 25, 4.2, C["foot"], C["foot"], lw=0, r=0.4, zorder=2)
txt(13, 4.3, "The Big Picture  --  Why Does This Matter?",
    fs=15, color="white", bold=True, zorder=6)

foot_panels = [
    ("The Problem",
     "10 million people will die from\nantibiotic resistance by 2050.\n"
     "Resistance is detected only AFTER\nwidespread use -- too late."),
    ("What We Built",
     "An AI that predicts which genes\nconfer resistance to NEW antibiotics\n"
     "before they enter clinical use --\nusing only the drug description."),
    ("What We Proved",
     "Graph neural networks beat random\nguessing on new drug classes.\n"
     "Lookup tables completely fail.\n"
     "Graph structure is the key ingredient."),
    ("Real-World Impact",
     "Pharma companies can screen new\nantibiotics for resistance risk early.\n"
     "Hospitals can prepare before resistant\nstrains spread globally."),
]

sw9, sx9 = 5.8, 0.8
for i, (title, body) in enumerate(foot_panels):
    sxi = sx9 + i*(sw9+0.47)
    rbox(sxi, 0.55, sw9, 3.45, C["foot"], C["foot"], lw=0, r=0.22, zorder=4)
    # Coloured top accent line
    accent_colors = ["#FFD54F", "#81D4FA", "#A5D6A7", "#CE93D8"]
    rbox(sxi, 3.6, sw9, 0.28, accent_colors[i], accent_colors[i], lw=0, r=0.12, zorder=5)
    txt(sxi+sw9/2, 3.74, title, fs=11, color=accent_colors[i], bold=True, zorder=6)
    txt(sxi+sw9/2, 2.55, body, fs=9.2, color=C["foot_fg"], zorder=5)

# ═══════════════════════════════════════════════════════════════════════════════
# DOWN ARROWS between every step
# ═══════════════════════════════════════════════════════════════════════════════
for y1, y2 in [
    (34.15, 33.9),   # title to step 1
    (31.5,  31.05),  # 1 -> 2
    (28.1,  27.65),  # 2 -> 3
    (24.85, 24.4),   # 3 -> 4
    (20.7,  20.25),  # 4 -> 5
    (16.6,  16.15),  # 5 -> 6
    (12.55, 12.1),   # 6 -> 7
    (8.65,  8.2),    # 7 -> 8
    (4.95,  4.6),    # 8 -> footer
]:
    down_arrow(13, y1, y2)

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════════
out = "/Users/anishgupta/Desktop/SIN_Project/results/paper"
plt.savefig(f"{out}/pipeline_diagram.pdf", dpi=300, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.savefig(f"{out}/pipeline_diagram.png", dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"Saved  pipeline_diagram.pdf  and  pipeline_diagram.png  ->  {out}")
plt.close()
