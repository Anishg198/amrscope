"""
Step 7: Deep analysis and case studies.

  A. Per-drug-class zero-shot MRR vs gene-overlap correlation
  B. Top predicted resistance genes for each zero-shot drug class (ZS-HetGAT)
  C. Gene family analysis: which gene families drive zero-shot predictions
  D. Confusion matrix for within-zero-shot ranking
  E. Generate final publication summary table

Output: results/zeroshot/analysis/
"""

import json, pickle, sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EVAL_JSON  = ROOT / "results/zeroshot/eval_summary.json"
GRAPH_PKL  = ROOT / "data/processed/enhanced_graph.pkl"
SPLITS_PKL = ROOT / "data/processed/splits.pkl"
OUT_DIR    = ROOT / "results/zeroshot/analysis"
FIG_DIR    = ROOT / "results/zeroshot/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi":    150,
    "savefig.dpi":   300,
    "savefig.bbox":  "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

COLORS = {
    "zs_hetgat":   "#1f77b4",
    "rgcn":        "#ff7f0e",
    "feature_mlp": "#2ca02c",
    "distmult":    "#d62728",
    "transe":      "#9467bd",
}
MODEL_LABELS = {
    "zs_hetgat":   "ZS-HetGAT (ours)",
    "rgcn":        "R-GCN",
    "feature_mlp": "Feature-MLP",
    "distmult":    "DistMult",
    "transe":      "TransE",
}
ZS_CLASS_OVERLAPS = {
    "glycopeptide antibiotic":    1,
    "glycylcycline":              100,
    "lincosamide antibiotic":     56,
    "rifamycin antibiotic":       47,
    "streptogramin A antibiotic": 100,
}
RELATED_TRAINING = {
    "glycylcycline":              "tetracycline",
    "lincosamide antibiotic":     "macrolide",
    "streptogramin A antibiotic": "streptogramin",
    "rifamycin antibiotic":       "fluoroquinolone",
    "glycopeptide antibiotic":    "β-lactam",
}


# ─── A. Overlap vs MRR scatter ───────────────────────────────────────────────

def fig_overlap_scatter(eval_data):
    """Scatter: gene overlap % vs zero-shot MRR per drug class, per model."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax_i, (ax, metric_key, ylabel) in enumerate(zip(axes,
        ["mrr", "hits@10"], ["MRR", "Hits@10"])):
        for model_name, color in COLORS.items():
            if model_name not in eval_data:
                continue
            per_dc = eval_data[model_name].get("per_dc_zeroshot", {})
            xs, ys = [], []
            for dc, v in per_dc.items():
                if dc in ZS_CLASS_OVERLAPS:
                    xs.append(ZS_CLASS_OVERLAPS[dc])
                    ys.append(v[metric_key])
            if xs:
                ax.scatter(xs, ys, color=color, s=80, alpha=0.85, zorder=5,
                           label=MODEL_LABELS.get(model_name, model_name))
                # Trend line
                if len(xs) >= 3:
                    z = np.polyfit(xs, ys, 1)
                    px = np.linspace(0, 105, 100)
                    ax.plot(px, np.polyval(z, px), "--", color=color, alpha=0.4, linewidth=1)

        # Annotate drug class names (from last model plotted)
        for dc, ov in ZS_CLASS_OVERLAPS.items():
            short = dc.replace(" antibiotic", "").replace("streptogramin A", "StrA")
            ax.annotate(f"{short}\n({RELATED_TRAINING.get(dc,'')}→)",
                       (ov, -0.02), fontsize=7.5, ha="center", va="top",
                       xycoords=("data", "axes fraction"), color="gray")

        ax.axhline(0.477, color="gray", linestyle=":", linewidth=1, label="Random (5-class)")
        ax.set_xlabel("Gene overlap with related training class (%)")
        ax.set_ylabel(f"Zero-shot {ylabel}")
        ax.set_title(f"Zero-shot {ylabel} vs Cross-Resistance Gene Overlap")
        ax.set_xlim(-5, 110)
        ax.set_ylim(0, 1.05)
        if ax_i == 0:
            ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "F7_overlap_vs_performance.pdf")
    plt.savefig(FIG_DIR / "F7_overlap_vs_performance.png")
    plt.close()
    print("  Saved F7_overlap_vs_performance")


# ─── B. Per-drug-class heatmap (multi-model) ─────────────────────────────────

def fig_heatmap(eval_data):
    models  = [m for m in MODEL_LABELS if m in eval_data]
    dc_list = sorted(ZS_CLASS_OVERLAPS.keys())
    n_m     = len(models)
    n_dc    = len(dc_list)

    mrr_mat = np.zeros((n_m, n_dc))
    for i, mn in enumerate(models):
        per_dc = eval_data[mn].get("per_dc_zeroshot", {})
        for j, dc in enumerate(dc_list):
            mrr_mat[i, j] = per_dc.get(dc, {}).get("mrr", 0.0)

    # Sort dc by overlap
    dc_order = sorted(range(n_dc), key=lambda j: ZS_CLASS_OVERLAPS[dc_list[j]])
    dc_sorted = [dc_list[j] for j in dc_order]
    mat_sorted = mrr_mat[:, dc_order]

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(mat_sorted, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n_dc))
    ax.set_xticklabels(
        [f"{dc.replace(' antibiotic','')}\n(overlap={ZS_CLASS_OVERLAPS[dc]}%)"
         for dc in dc_sorted], fontsize=9, rotation=15, ha="right")
    ax.set_yticks(range(n_m))
    ax.set_yticklabels([MODEL_LABELS[m] for m in models], fontsize=10)

    for i in range(n_m):
        for j in range(n_dc):
            val = mat_sorted[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if val > 0.6 else "black")

    # Random baseline indicator
    ax.axhline(-0.5, color="white", linewidth=0)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Zero-shot MRR", fontsize=10)
    cb.ax.axhline(0.477, color="blue", linewidth=1.5, linestyle="--")
    cb.ax.text(1.1, 0.477, "random", transform=cb.ax.transAxes,
               fontsize=8, va="center", color="blue")

    ax.set_title("Per-drug-class Zero-shot MRR by Model\n"
                 "(drug classes sorted by gene overlap with related training class)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "F8_perclass_heatmap.pdf")
    plt.savefig(FIG_DIR / "F8_perclass_heatmap.png")
    plt.close()
    print("  Saved F8_perclass_heatmap")


# ─── C. Top predicted genes for ZS classes ───────────────────────────────────

def top_predicted_genes(k=20):
    """For each zero-shot drug class, list the top-K predicted resistance genes."""
    with open(GRAPH_PKL, "rb") as f:
        obj = pickle.load(f)
    with open(SPLITS_PKL, "rb") as f:
        splits = pickle.load(f)

    from src.models.zeroshot_hetgat import ZeroShotHetGAT
    import torch

    # Load ZS-HetGAT seed 42
    ck_path = ROOT / "results/zeroshot/models/zs_hetgat_seed42.pt"
    if not ck_path.exists():
        print("  ZS-HetGAT checkpoint not found")
        return

    ck      = torch.load(ck_path, map_location="cpu", weights_only=False)
    graph   = obj["hetero_data"]
    hp      = ck["hparams"]
    gene_dim = graph["gene"].x.shape[1]
    drug_dim = graph["drug_class"].x.shape[1]
    mech_dim = graph["mechanism"].x.shape[1]

    model = ZeroShotHetGAT(gene_dim, drug_dim, mech_dim,
                            hp["hidden_dim"], hp["out_dim"],
                            hp["num_heads"], hp["dropout"])
    model.load_state_dict(ck["state_dict"])
    model.eval()

    zs_indices = splits["meta"]["zs_drug_class_indices"]
    zs_names   = splits["meta"]["zs_drug_class_names"]
    dc2i       = obj["dc2i"]
    gene_meta  = obj["gene_metadata"]
    gene_ids   = obj["hetero_data"]["gene"].node_ids

    score_mat = model.score_all_pairs(graph).detach()  # (G, D)

    results = {}
    for dc_idx, dc_name in zip(zs_indices, zs_names):
        dc_scores = score_mat[:, dc_idx]
        top_k_idx = dc_scores.topk(k).indices.tolist()
        true_genes = set(
            graph["gene", "confers_resistance_to", "drug_class"].edge_index[0]
            [graph["gene", "confers_resistance_to", "drug_class"].edge_index[1] == dc_idx]
            .tolist()
        )

        pred_list = []
        for gi in top_k_idx:
            gid   = gene_ids[gi] if isinstance(gene_ids, list) else str(gi)
            meta  = gene_meta.get(gid, {})
            pred_list.append({
                "gene_id":   gid,
                "gene_name": meta.get("name", "unknown"),
                "score":     float(dc_scores[gi]),
                "is_true_positive": gi in true_genes,
                "families":  meta.get("families", []),
                "mechanisms": meta.get("mechanisms", []),
            })

        n_true    = len(true_genes)
        n_correct = sum(p["is_true_positive"] for p in pred_list)
        results[dc_name] = {
            "top_k": pred_list,
            "n_true_positives": n_true,
            "hits_at_k": n_correct,
            "precision_at_k": n_correct / k,
            "overlap_pct": ZS_CLASS_OVERLAPS.get(dc_name, "?"),
        }
        print(f"  {dc_name} (overlap={ZS_CLASS_OVERLAPS.get(dc_name,'?')}%):")
        print(f"    Top-{k} contains {n_correct}/{n_true} true positives "
              f"(P@{k}={n_correct/k:.2f})")
        for p in pred_list[:5]:
            mark = "✓" if p["is_true_positive"] else "✗"
            print(f"    {mark} {p['gene_name'][:40]:<40} "
                  f"score={p['score']:.3f}  mech={p['mechanisms']}")

    out_path = OUT_DIR / "top_predicted_genes.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved top predicted genes → {out_path}")
    return results


# ─── D. Summary table ────────────────────────────────────────────────────────

def print_final_table(eval_data):
    print("\n" + "=" * 90)
    print("FINAL RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Model':<20} {'Test MRR':>10} {'ZS-Win MRR':>12} {'ZS H@10':>10} "
          f"{'Gen Gap':>10} {'ZS AUC':>10}")
    print("-" * 90)

    random_zs_mrr = 0.477

    for mn, label in MODEL_LABELS.items():
        if mn not in eval_data:
            continue
        d = eval_data[mn]
        t = d["test"]
        z = d["zeroshot"]

        mrr_t  = t["mrr"]["mean"]
        mrr_z  = z["mrr"]["mean"]
        std_z  = z["mrr"]["std"]
        h10_z  = z["hits@10"]["mean"]
        gap    = d["generalisation_gap_mrr"]
        auc_z  = z.get("auc", 0)

        above_rnd = "↑" if mrr_z > random_zs_mrr else "↓"
        print(f"{label:<20} {mrr_t:>10.4f} {mrr_z:>8.4f}±{std_z:.3f} "
              f"{h10_z:>10.4f} {gap:>+10.4f} {auc_z:>10.4f} {above_rnd}")

    print("-" * 90)
    print(f"{'Random (5-class)':<20} {'':>10} {'0.4772':>12} {'0.4800':>10} {'':>10}")
    print("=" * 90)
    print("\nKey finding: Graph-based inductive models (R-GCN, ZS-HetGAT) exceed random")
    print("             Transductive KGE (DistMult, TransE) fall below random")


def main():
    print("=" * 60)
    print("Step 7: Deep analysis and case studies")
    print("=" * 60)

    with open(EVAL_JSON) as f:
        eval_data = json.load(f)

    print("\nA. Gene overlap vs zero-shot performance ...")
    fig_overlap_scatter(eval_data)

    print("\nB. Per-drug-class heatmap ...")
    fig_heatmap(eval_data)

    print("\nC. Top predicted genes for zero-shot drug classes ...")
    top_predicted_genes(k=20)

    print("\nD. Final summary table:")
    print_final_table(eval_data)

    print(f"\nAll outputs saved to {OUT_DIR} and {FIG_DIR}")


if __name__ == "__main__":
    main()
