"""
Step 10: Generate publication-quality figures for the paper.

Figures:
  F1_leakage_diagram.pdf    — conceptual diagram: leaky vs clean features
  F2_main_results.pdf       — ZS-all MRR bar chart (primary result)
  F3_perclass_heatmap.pdf   — per-class ZS-all MRR heatmap (reordered)
  F4_tanimoto_scatter.pdf   — Tanimoto vs ZS-all MRR with r annotation
  F5_ablation.pdf           — ablation bar chart (generated after ablations)
"""

import json
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr
import torch

ROOT      = Path(__file__).resolve().parents[1]
EVAL_JSON = ROOT / "results/biomolamr/eval_summary.json"
SPLITS    = ROOT / "data/processed/extended_splits.pkl"
TAN_FILE  = ROOT / "data/processed/drug_tanimoto.pt"
GRAPH     = ROOT / "data/processed/biomolamr_graph.pkl"
FIGS_DIR  = ROOT / "results/biomolamr/figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

MODEL_ORDER = ["feature_mlp", "distmult", "rgcn_bio", "biomolamr", "transe"]
LABELS = {
    "feature_mlp": "Feature-MLP",
    "rgcn_bio":    "R-GCN (bio)",
    "distmult":    "DistMult$^\\dagger$",
    "transe":      "TransE$^\\dagger$",
    "biomolamr":   "BioMolAMR",
}
COLORS = {
    "feature_mlp": "#2166ac",
    "distmult":    "#762a83",
    "rgcn_bio":    "#d6604d",
    "biomolamr":   "#e63946",
    "transe":      "#9467bd",
}


# ── Figure 1: Leakage Diagram ─────────────────────────────────────────────────
def fig1_leakage_diagram():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, title, leaky in zip(axes, ["(a) Prior Work: Leaky Features",
                                        "(b) BioMolAMR: Clean Features"], [True, False]):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

        # Gene node
        gene_patch = mpatches.FancyBboxPatch(
            (0.3, 5), 2.8, 1.8, boxstyle="round,pad=0.1",
            facecolor="#a8d8f0", edgecolor="#2166ac", linewidth=2)
        ax.add_patch(gene_patch)
        ax.text(1.7, 5.9, "Gene\n(ARO:xxx)", ha="center", va="center", fontsize=9)

        # Feature vector
        feat_patch = mpatches.FancyBboxPatch(
            (0.3, 2.5), 2.8, 2.0, boxstyle="round,pad=0.1",
            facecolor="#ffd700" if leaky else "#90EE90",
            edgecolor="#b8860b" if leaky else "#228B22", linewidth=2)
        ax.add_patch(feat_patch)
        if leaky:
            ax.text(1.7, 3.5, "154-dim features\n[drug class ✓/✗ …]",
                    ha="center", va="center", fontsize=8, color="darkred",
                    fontweight="bold")
        else:
            ax.text(1.7, 3.5, "480-dim\nESM-2 embedding\n(sequence only)",
                    ha="center", va="center", fontsize=8)

        # Arrow
        ax.annotate("", xy=(1.7, 2.5), xytext=(1.7, 5.0),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))

        # Model box
        model_patch = mpatches.FancyBboxPatch(
            (4, 3.5), 2.0, 1.5, boxstyle="round,pad=0.1",
            facecolor="#e8e8e8", edgecolor="#555", linewidth=1.5)
        ax.add_patch(model_patch)
        ax.text(5.0, 4.25, "Model", ha="center", va="center", fontsize=10)

        ax.annotate("", xy=(4.0, 4.25), xytext=(3.1, 4.25),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))

        # Drug node (ZS)
        drug_patch = mpatches.FancyBboxPatch(
            (7.2, 5), 2.3, 1.5, boxstyle="round,pad=0.1",
            facecolor="#ffb3b3", edgecolor="#c00", linewidth=2, linestyle="--")
        ax.add_patch(drug_patch)
        ax.text(8.35, 5.75, "Novel Drug\n(ZS class)", ha="center", va="center",
                fontsize=9)

        ax.annotate("", xy=(8.35, 5.0), xytext=(8.35, 4.5),
                    arrowprops=dict(arrowstyle="-", color="#c00", lw=1.5))
        ax.annotate("", xy=(6.0, 4.25), xytext=(7.2, 4.25),
                    arrowprops=dict(arrowstyle="<-", color="#333", lw=1.5))

        # Drug features
        if leaky:
            ax.text(8.35, 2.5,
                    "97-dim\n[one-hot ID]\n(identity ≠ structure)",
                    ha="center", va="center", fontsize=8,
                    color="darkred",
                    bbox=dict(fc="#ffd700", ec="#b8860b", boxstyle="round"))
        else:
            ax.text(8.35, 2.5,
                    "3245-dim\nMol fingerprint\n(Morgan+MACCS+TT)",
                    ha="center", va="center", fontsize=8,
                    bbox=dict(fc="#90EE90", ec="#228B22", boxstyle="round"))

        ax.annotate("", xy=(8.35, 5.0), xytext=(8.35, 3.0),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))

        # Verdict
        if leaky:
            ax.text(5.0, 1.3,
                    "⚠ Gene features encode ZS drug class\n→ ZS MRR inflated (0.775)",
                    ha="center", va="center", fontsize=9,
                    color="#900",
                    bbox=dict(fc="#fff0f0", ec="#900", boxstyle="round"))
        else:
            ax.text(5.0, 1.3,
                    "✓ No drug-class info in gene features\n→ Honest ZS MRR (0.069)",
                    ha="center", va="center", fontsize=9,
                    color="#004d00",
                    bbox=dict(fc="#f0fff0", ec="#228B22", boxstyle="round"))

    plt.tight_layout(pad=2.0)
    plt.savefig(FIGS_DIR / "F1_leakage_diagram.pdf")
    plt.savefig(FIGS_DIR / "F1_leakage_diagram.png")
    plt.close()
    print("  Saved F1 leakage diagram")


# ── Figure 2: Main results bar chart ─────────────────────────────────────────
def fig2_main_results(summary):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    random_within = 1 / 12
    random_all    = 1 / 46

    for ax, metric_key, random_val, ylabel, panel_title, letter in zip(
        axes,
        ["zs_all", "zs_within"],
        [random_all, random_within],
        ["ZS-All MRR (Primary Metric)", "ZS-Within MRR (12-way)"],
        ["All-Class Ranking", "Within-ZS Ranking"],
        ["(a)", "(b)"],
    ):
        models = [m for m in MODEL_ORDER if m in summary]
        mrrs   = [summary[m][metric_key]["mrr"]["mean"] for m in models]
        stds   = [summary[m][metric_key]["mrr"]["std"]  for m in models]
        xs     = np.arange(len(models))

        bars = ax.bar(xs, mrrs, yerr=stds, capsize=4,
                      color=[COLORS[m] for m in models],
                      alpha=0.88, edgecolor="white", linewidth=0.8)
        ax.axhline(random_val, ls="--", color="gray", linewidth=1.5, alpha=0.7,
                   label=f"Random ({random_val:.3f})")
        ax.set_xticks(xs)
        ax.set_xticklabels([LABELS[m] for m in models], rotation=18, ha="right",
                            fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{letter} {panel_title}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

        for bar, mrr, std in zip(bars, mrrs, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + std + 0.002,
                    f"{mrr:.3f}", ha="center", va="bottom", fontsize=8.5)

    plt.suptitle("Zero-Shot AMR Prediction: Main Results", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "F2_main_results.pdf")
    plt.savefig(FIGS_DIR / "F2_main_results.png")
    plt.close()
    print("  Saved F2 main results")


# ── Figure 3: Per-class heatmap (reordered by Feature-MLP) ───────────────────
def fig3_perclass_heatmap(summary):
    models_present = [m for m in MODEL_ORDER if m in summary and
                      "per_dc_zs_all" in summary[m]]
    if not models_present:
        print("  Skipping F3 (no per-class data)")
        return

    fmlp_pdc = summary["feature_mlp"].get("per_dc_zs_all", {})
    classes  = sorted(fmlp_pdc.keys(),
                      key=lambda c: fmlp_pdc[c]["mrr"], reverse=True)
    if not classes:
        print("  Skipping F3 (empty per-class)")
        return

    data_mat = np.zeros((len(models_present), len(classes)))
    for i, m in enumerate(models_present):
        pdc = summary[m].get("per_dc_zs_all", {})
        for j, dc in enumerate(classes):
            data_mat[i, j] = pdc.get(dc, {}).get("mrr", 0)

    fig, ax = plt.subplots(figsize=(14, 4.5))
    im = ax.imshow(data_mat, cmap="RdYlGn", vmin=0, vmax=0.9, aspect="auto")

    short = lambda c: c.replace(" antibiotic", "").replace(" acid", " ac.")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([short(c) for c in classes], rotation=38, ha="right",
                        fontsize=9.5)
    ax.set_yticks(range(len(models_present)))
    ax.set_yticklabels([LABELS[m] for m in models_present], fontsize=10)
    ax.set_title("Per-Class Zero-Shot MRR (All-Class Ranking, sorted by Feature-MLP)",
                 fontsize=12, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("ZS-All MRR", fontsize=10)
    # Mark random baseline on colorbar
    cbar.ax.axhline(1/46, color="gray", ls="--", linewidth=1.5)
    cbar.ax.text(1.3, 1/46, "random", va="center", fontsize=8, color="gray")

    plt.tight_layout()
    plt.savefig(FIGS_DIR / "F3_perclass_heatmap.pdf")
    plt.savefig(FIGS_DIR / "F3_perclass_heatmap.png")
    plt.close()
    print("  Saved F3 per-class heatmap")


# ── Figure 4: Tanimoto scatter with Pearson r ────────────────────────────────
def fig4_tanimoto_scatter(summary):
    if not TAN_FILE.exists() or not SPLITS.exists() or not GRAPH.exists():
        print("  Skipping F4 (missing files)")
        return

    tanimoto = torch.load(TAN_FILE, map_location="cpu").numpy()
    with open(SPLITS, "rb") as f:
        splits = pickle.load(f)
    with open(GRAPH, "rb") as f:
        g_obj = pickle.load(f)

    dc2i     = g_obj["dc2i"]
    meta     = splits["meta"]
    zs_names = meta["zs_drug_class_names"]
    train_idx = set(meta["train_drug_class_indices"])

    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name in MODEL_ORDER:
        if model_name not in summary or "per_dc_zs_all" not in summary[model_name]:
            continue
        pdc = summary[model_name]["per_dc_zs_all"]
        xs, ys = [], []
        for dc_name in zs_names:
            if dc_name not in dc2i or dc_name not in pdc:
                continue
            zs_idx   = dc2i[dc_name]
            max_tan  = max((tanimoto[zs_idx, ti] for ti in train_idx), default=0)
            xs.append(max_tan)
            ys.append(pdc[dc_name]["mrr"])

        if len(xs) < 2:
            continue
        r, pval = pearsonr(xs, ys)
        label = f"{LABELS[model_name]} (r={r:.2f})"
        ax.scatter(xs, ys, color=COLORS[model_name], s=90, alpha=0.75,
                   label=label, zorder=3)
        # Trend line
        z  = np.polyfit(xs, ys, 1)
        px = np.linspace(min(xs) - 0.02, max(xs) + 0.02, 100)
        ax.plot(px, np.polyval(z, px), "--", color=COLORS[model_name],
                alpha=0.45, linewidth=1.4)

    ax.axhline(1/46, ls=":", color="gray", lw=1.5, label="Random (1/46)")
    ax.set_xlabel("Max Tanimoto Similarity to Nearest Training Drug Class",
                  fontsize=12)
    ax.set_ylabel("ZS-All MRR (Primary Metric)", fontsize=12)
    ax.set_title("Chemical Similarity vs. Zero-Shot Transfer Performance",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "F4_tanimoto_scatter.pdf")
    plt.savefig(FIGS_DIR / "F4_tanimoto_scatter.png")
    plt.close()
    print("  Saved F4 Tanimoto scatter")


# ── Figure 5: Ablation bar chart ─────────────────────────────────────────────
def fig5_ablation(summary):
    ablation_names = ["fmlp_gene_zero", "fmlp_drug_zero", "fmlp_leaky"]
    ablation_labels = {
        "feature_mlp":    "Feature-MLP\n(full)",
        "fmlp_gene_zero": "FMLP\ngene=0\n(drug FP only)",
        "fmlp_drug_zero": "FMLP\ndrug=0\n(ESM-2 only)",
        "fmlp_leaky":     "FMLP\nleaky features\n(prior work)",
    }
    ablation_colors = {
        "feature_mlp":    "#2166ac",
        "fmlp_gene_zero": "#4dac26",
        "fmlp_drug_zero": "#d01c8b",
        "fmlp_leaky":     "#b35806",
    }

    plot_order = ["feature_mlp"] + ablation_names
    available  = [m for m in plot_order if m in summary]
    if len(available) < 2:
        print("  Skipping F5 (ablation results not yet available)")
        return

    mrrs_w = [summary[m]["zs_within"]["mrr"]["mean"] for m in available]
    stds_w = [summary[m]["zs_within"]["mrr"]["std"]  for m in available]
    mrrs_a = [summary[m]["zs_all"]["mrr"]["mean"]    for m in available]
    stds_a = [summary[m]["zs_all"]["mrr"]["std"]     for m in available]

    x = np.arange(len(available))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, mrrs, stds, random_v, ylabel, title in zip(
        axes,
        [mrrs_a, mrrs_w],
        [stds_a, stds_w],
        [1/46, 1/12],
        ["ZS-All MRR (Primary)", "ZS-Within MRR (12-way)"],
        ["(a) All-Class Ranking", "(b) Within-ZS Ranking"],
    ):
        bars = ax.bar(x, mrrs, yerr=stds, capsize=4,
                      color=[ablation_colors[m] for m in available],
                      alpha=0.88, edgecolor="white", linewidth=0.8)
        ax.axhline(random_v, ls="--", color="gray", lw=1.5, alpha=0.7,
                   label=f"Random ({random_v:.3f})")
        ax.set_xticks(x)
        ax.set_xticklabels([ablation_labels[m] for m in available],
                            rotation=0, ha="center", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        for bar, mrr, std in zip(bars, mrrs, stds):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + std + 0.001,
                    f"{mrr:.3f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Ablation Study: Feature Contributions to Zero-Shot Transfer",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "F5_ablation.pdf")
    plt.savefig(FIGS_DIR / "F5_ablation.png")
    plt.close()
    print("  Saved F5 ablation figure")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Step 10: Generating paper figures")
    print("=" * 60)

    # Load eval summary (includes both main models and any ablation checkpoints)
    if not EVAL_JSON.exists():
        print(f"ERROR: {EVAL_JSON} not found. Run pipeline/08_extended_evaluation.py first.")
        return

    with open(EVAL_JSON) as f:
        summary = json.load(f)

    print(f"Models in summary: {list(summary.keys())}")

    fig1_leakage_diagram()
    fig2_main_results(summary)
    fig3_perclass_heatmap(summary)
    fig4_tanimoto_scatter(summary)
    fig5_ablation(summary)

    print(f"\nAll figures saved to {FIGS_DIR}")


if __name__ == "__main__":
    main()
