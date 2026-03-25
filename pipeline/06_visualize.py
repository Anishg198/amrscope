"""
Step 6: Generate publication-quality figures.

Figures produced:
  F1 – Bar chart: MRR comparison (Test vs Zero-shot) across all models
  F2 – Radar chart: Multi-metric comparison of all models on zero-shot split
  F3 – Heatmap: Per-drug-class zero-shot performance (ZS-HetGAT)
  F4 – Line: Training curves (loss + val MRR) for ZS-HetGAT
  F5 – Scatter: Generalisation gap vs model capacity
  F6 – Bar: Ablation study on zero-shot MRR
"""

import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

ROOT     = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EVAL_JSON  = ROOT / "results/zeroshot/eval_summary.json"
ABL_JSON   = ROOT / "results/zeroshot/ablation_results.json"
CKPT_DIR   = ROOT / "results/zeroshot/models"
FIG_DIR    = ROOT / "results/zeroshot/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Publication style
plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

MODEL_LABELS = {
    "zs_hetgat":   "ZS-HetGAT (ours)",
    "rgcn":        "R-GCN",
    "feature_mlp": "Feature-MLP",
    "distmult":    "DistMult",
    "transe":      "TransE",
}
COLORS = {
    "zs_hetgat":   "#1f77b4",
    "rgcn":        "#ff7f0e",
    "feature_mlp": "#2ca02c",
    "distmult":    "#d62728",
    "transe":      "#9467bd",
}


def load_eval():
    if not EVAL_JSON.exists():
        return None
    with open(EVAL_JSON) as f:
        return json.load(f)


def load_ablation():
    if not ABL_JSON.exists():
        return None
    with open(ABL_JSON) as f:
        return json.load(f)


def load_history():
    """Load training history for ZS-HetGAT seed 42."""
    import torch
    ck_path = CKPT_DIR / "zs_hetgat_seed42.pt"
    if not ck_path.exists():
        return None
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    return ck.get("history", [])


# ─── Figure 1: MRR bar chart ──────────────────────────────────────────────────

def fig1_mrr_bar(eval_data):
    models = [m for m in MODEL_LABELS if m in eval_data]
    if not models:
        return

    test_mrr  = [eval_data[m]["test"]["mrr"]["mean"]     for m in models]
    test_std  = [eval_data[m]["test"]["mrr"]["std"]      for m in models]
    zs_mrr    = [eval_data[m]["zeroshot"]["mrr"]["mean"] for m in models]
    zs_std    = [eval_data[m]["zeroshot"]["mrr"]["std"]  for m in models]

    x     = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width/2, test_mrr, width, yerr=test_std,
                label="Test (seen classes)", capsize=4,
                color=[COLORS[m] for m in models], alpha=0.85)
    b2 = ax.bar(x + width/2, zs_mrr, width, yerr=zs_std,
                label="Zero-shot (unseen classes)", capsize=4,
                color=[COLORS[m] for m in models], alpha=0.45,
                edgecolor=[COLORS[m] for m in models], linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=15, ha="right")
    ax.set_ylabel("Mean Reciprocal Rank (MRR)")
    ax.set_title("AMR Resistance Prediction: Standard vs Zero-Shot Evaluation")
    ax.legend(loc="upper right")
    ax.set_ylim(0, min(1.0, max(test_mrr + zs_mrr) * 1.35))

    # Annotate bars with values
    for rect, val in zip(b1.patches, test_mrr):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    for rect, val in zip(b2.patches, zs_mrr):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "F1_mrr_comparison.pdf")
    plt.savefig(FIG_DIR / "F1_mrr_comparison.png")
    plt.close()
    print("  Saved F1_mrr_comparison")


# ─── Figure 2: Radar chart ────────────────────────────────────────────────────

def fig2_radar(eval_data):
    metrics = ["mrr", "hits@1", "hits@3", "hits@10"]
    labels  = ["MRR", "Hits@1", "Hits@3", "Hits@10"]
    N       = len(metrics)
    angles  = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                              subplot_kw={"projection": "polar"})
    titles = ["Test Split (seen classes)", "Zero-shot Split (unseen classes)"]
    keys   = ["test", "zeroshot"]

    for ax, title, key in zip(axes, titles, keys):
        ax.set_title(title, pad=15, fontsize=12)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)

        for mname in MODEL_LABELS:
            if mname not in eval_data:
                continue
            vals = [eval_data[mname][key][m]["mean"] for m in metrics]
            vals += vals[:1]
            ax.plot(angles, vals, "o-", linewidth=2,
                    label=MODEL_LABELS[mname], color=COLORS[mname])
            ax.fill(angles, vals, alpha=0.08, color=COLORS[mname])

        if key == "test":
            ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)

    plt.suptitle("Multi-metric Comparison of AMR Prediction Models", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "F2_radar_comparison.pdf")
    plt.savefig(FIG_DIR / "F2_radar_comparison.png")
    plt.close()
    print("  Saved F2_radar_comparison")


# ─── Figure 3: Per-drug-class heatmap ────────────────────────────────────────

def fig3_per_class_heatmap(eval_data):
    if "zs_hetgat" not in eval_data:
        return
    per_dc = eval_data["zs_hetgat"].get("per_dc_zeroshot", {})
    if not per_dc:
        return

    classes = list(per_dc.keys())
    mrr_vals  = [per_dc[c]["mrr"]    for c in classes]
    h10_vals  = [per_dc[c]["hits@10"] for c in classes]
    n_edges   = [per_dc[c]["n_edges"] for c in classes]

    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, len(classes) * 0.5 + 1)))
    for ax, vals, metric in zip(axes, [mrr_vals, h10_vals], ["MRR", "Hits@10"]):
        y = np.arange(len(classes))
        bars = ax.barh(y, vals, color="#1f77b4", alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(classes, fontsize=10)
        ax.set_xlabel(metric)
        ax.set_title(f"Zero-shot {metric} per Drug Class (ZS-HetGAT)")
        ax.set_xlim(0, 1)
        for bar, ne in zip(bars, n_edges):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f"n={ne}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "F3_per_class_zeroshot.pdf")
    plt.savefig(FIG_DIR / "F3_per_class_zeroshot.png")
    plt.close()
    print("  Saved F3_per_class_zeroshot")


# ─── Figure 4: Training curves ────────────────────────────────────────────────

def fig4_training_curves(history):
    if not history:
        return
    epochs  = [h["epoch"]   for h in history]
    losses  = [h["loss"]    for h in history]
    val_mrr = [h["mrr"]     for h in history]
    val_h10 = [h["hits@10"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, losses, "-o", markersize=3, color="#1f77b4", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.set_title("Training Loss (ZS-HetGAT)")

    ax2.plot(epochs, val_mrr, "-o", markersize=3, color="#1f77b4",
             linewidth=1.5, label="Val MRR")
    ax2.plot(epochs, val_h10, "-s", markersize=3, color="#ff7f0e",
             linewidth=1.5, label="Val Hits@10")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric")
    ax2.set_title("Validation Metrics (ZS-HetGAT)")
    ax2.legend()

    plt.suptitle("ZS-HetGAT Training Dynamics (seed=42)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "F4_training_curves.pdf")
    plt.savefig(FIG_DIR / "F4_training_curves.png")
    plt.close()
    print("  Saved F4_training_curves")


# ─── Figure 5: Generalisation gap scatter ────────────────────────────────────

def fig5_gen_gap(eval_data):
    models = [m for m in MODEL_LABELS if m in eval_data]
    if not models:
        return

    test_mrr = [eval_data[m]["test"]["mrr"]["mean"]     for m in models]
    zs_mrr   = [eval_data[m]["zeroshot"]["mrr"]["mean"] for m in models]
    gaps     = [t - z for t, z in zip(test_mrr, zs_mrr)]

    fig, ax = plt.subplots(figsize=(7, 5))
    for m, t, z, g in zip(models, test_mrr, zs_mrr, gaps):
        ax.scatter(t, g, s=120, color=COLORS[m], zorder=5,
                   label=MODEL_LABELS[m])
        ax.annotate(MODEL_LABELS[m], (t, g), textcoords="offset points",
                    xytext=(6, 3), fontsize=8)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Test MRR (seen drug classes)")
    ax.set_ylabel("Generalisation Gap (Test MRR − Zero-shot MRR)")
    ax.set_title("Generalisation Ability: Standard vs Zero-shot")
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "F5_gen_gap.pdf")
    plt.savefig(FIG_DIR / "F5_gen_gap.png")
    plt.close()
    print("  Saved F5_gen_gap")


# ─── Figure 6: Ablation bar chart ────────────────────────────────────────────

def fig6_ablation(abl_data):
    if not abl_data:
        return

    variants = list(abl_data.keys())
    descs    = [abl_data[v]["description"] for v in variants]
    test_mrr = [abl_data[v]["test"]["mrr"]["mean"]     for v in variants]
    test_std = [abl_data[v]["test"]["mrr"]["std"]      for v in variants]
    zs_mrr   = [abl_data[v]["zeroshot"]["mrr"]["mean"] for v in variants]
    zs_std   = [abl_data[v]["zeroshot"]["mrr"]["std"]  for v in variants]

    x     = np.arange(len(variants))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, test_mrr, width, yerr=test_std, capsize=4,
           label="Test MRR", color="#1f77b4", alpha=0.85)
    ax.bar(x + width/2, zs_mrr,   width, yerr=zs_std,   capsize=4,
           label="Zero-shot MRR", color="#1f77b4", alpha=0.4,
           edgecolor="#1f77b4", linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(descs, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("MRR")
    ax.set_title("Ablation Study: ZS-HetGAT Component Analysis")
    ax.legend()
    ax.set_ylim(0, max(test_mrr + zs_mrr) * 1.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "F6_ablation.pdf")
    plt.savefig(FIG_DIR / "F6_ablation.png")
    plt.close()
    print("  Saved F6_ablation")


def main():
    print("=" * 60)
    print("Step 6: Generating publication figures")
    print("=" * 60)

    eval_data = load_eval()
    abl_data  = load_ablation()
    history   = load_history()

    if eval_data:
        fig1_mrr_bar(eval_data)
        fig2_radar(eval_data)
        fig3_per_class_heatmap(eval_data)
        fig5_gen_gap(eval_data)
    else:
        print("  No evaluation data found (run 04_evaluate.py first)")

    if history:
        fig4_training_curves(history)
    else:
        print("  No training history found")

    if abl_data:
        fig6_ablation(abl_data)
    else:
        print("  No ablation data found (run 05_ablations.py first)")

    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
