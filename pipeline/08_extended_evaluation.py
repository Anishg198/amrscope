"""
Step 08: Extended evaluation of BioMolAMR vs all baselines.

Primary metric: All-class filtered MRR (ZS-all)
  — ranks each ZS drug class among ALL 46 drug classes (realistic scenario)
  — random baseline MRR ≈ 1/46 ≈ 0.022 (much harder than 5-way ranking)

Secondary metric: Within-ZS MRR (ZS-within)
  — ranks among the 12 ZS classes (for comparison with previous 5-class paper)

Additional analyses:
  A. Per-drug-class breakdown with Tanimoto similarity to nearest training drug
  B. Correlation: Tanimoto(ZS_drug, training_drug) vs ZS-all MRR
  C. P@20 for top drug classes (case studies)
  D. Mechanism attribution analysis (BioMolAMR only)

Output: results/biomolamr/eval_summary.json
        results/biomolamr/tables/
        results/biomolamr/figures/
"""

import json, pickle, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.biomolamr import BioMolAMR
from src.models.baselines  import DistMultBaseline, TransEBaseline, FeatureMLPBaseline, RGCNBaseline
from src.models.crossamr   import CrossContrastAMR

MODELS_DIR = ROOT / "results/biomolamr/models"
TABLES_DIR = ROOT / "results/biomolamr/tables"
FIGS_DIR   = ROOT / "results/biomolamr/figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

GRAPH_NEW  = ROOT / "data/processed/biomolamr_graph.pkl"
GRAPH_ORIG = ROOT / "data/processed/enhanced_graph.pkl"
SPLITS_EXT = ROOT / "data/processed/extended_splits.pkl"
SPLITS_STD = ROOT / "data/processed/splits.pkl"
TAN_FILE   = ROOT / "data/processed/drug_tanimoto.pt"

MODEL_NAMES = ["crossamr", "feature_mlp", "biomolamr", "rgcn_bio", "distmult", "transe",
               "fmlp_gene_zero", "fmlp_drug_zero", "fmlp_leaky"]
METRICS     = ["mrr", "hits@1", "hits@3", "hits@10"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})
COLORS = {
    "crossamr":    "#6200ea",
    "biomolamr":   "#e63946",
    "rgcn_bio":    "#ff7f0e",
    "feature_mlp": "#2ca02c",
    "distmult":    "#d62728",
    "transe":      "#9467bd",
}
LABELS = {
    "crossamr":    "CrossContrastAMR (ours)",
    "biomolamr":   "AMRScope (GAT)",
    "rgcn_bio":    "R-GCN (bio graph)",
    "feature_mlp": "Feature-MLP",
    "distmult":    "DistMult†",
    "transe":      "TransE†",
}


def load_checkpoints():
    ckpts = defaultdict(list)
    for path in sorted(MODELS_DIR.glob("*.pt")):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        ckpts[ck["model_name"]].append(ck)
    return ckpts


def agg_metrics(runs, split_key):
    out = {}
    for m in METRICS:
        vals = [r[split_key][m] for r in runs if split_key in r and m in r[split_key]]
        if vals:
            out[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


def build_model_from_ck(model_name, ck, graph):
    hp      = ck["hparams"]
    gene_dim = graph["gene"].x.shape[1]
    drug_dim = graph["drug_class"].x.shape[1]
    mech_dim = graph["mechanism"].x.shape[1]
    n_genes  = graph["gene"].x.shape[0]
    n_dc     = graph["drug_class"].x.shape[0]

    if model_name == "crossamr":
        m = CrossContrastAMR(gene_dim, drug_dim,
                             hp["hidden_dim"], hp["n_heads"], hp["dropout"])
    elif model_name == "biomolamr":
        m = BioMolAMR(gene_dim, drug_dim, mech_dim,
                      hp["hidden_dim"], hp["out_dim"],
                      hp["num_heads"], hp["num_gat_layers"], hp["dropout"])
    elif model_name == "rgcn_bio":
        m = RGCNBaseline(gene_dim, drug_dim, mech_dim,
                         hp["hidden_dim"], hp["out_dim"], hp["dropout"])
    elif model_name == "distmult":
        m = DistMultBaseline(n_genes, n_dc, hp["emb_dim"], hp["dropout"])
    elif model_name == "transe":
        m = TransEBaseline(n_genes, n_dc, hp["emb_dim"], hp["margin"], hp["dropout"])
    elif model_name == "feature_mlp":
        m = FeatureMLPBaseline(gene_dim, drug_dim, hp["hidden_dim"], hp["dropout"])
    elif model_name in ("fmlp_gene_zero", "fmlp_drug_zero", "fmlp_leaky"):
        # Ablation variants: use dims stored in checkpoint (may differ from graph)
        g_dim = ck.get("gene_feat_dim", gene_dim)
        d_dim = ck.get("drug_feat_dim", drug_dim)
        m = FeatureMLPBaseline(g_dim, d_dim, hp["hidden_dim"], hp["dropout"])
    m.load_state_dict(ck["state_dict"])
    return m


def score_all_pairs(model_name, model, graph, gene_idx, drug_idx, device,
                    gene_x_override=None, drug_x_override=None):
    model.eval()
    with torch.no_grad():
        if model_name in ("biomolamr", "rgcn_bio"):
            return model(graph, gene_idx, drug_idx)
        elif model_name in ("distmult", "transe"):
            return model(gene_idx, drug_idx)
        elif model_name == "crossamr":
            gx = graph["gene"].x
            dx = graph["drug_class"].x
            return model(gx, dx, gene_idx, drug_idx)
        else:
            # feature_mlp and all fmlp_* ablations
            gx = gene_x_override if gene_x_override is not None else graph["gene"].x
            dx = drug_x_override if drug_x_override is not None else graph["drug_class"].x
            return model(gx, dx, gene_idx, drug_idx)


def compute_mrr_hits(pos_scores, neg_mat):
    ps    = pos_scores.unsqueeze(1)
    ranks = (neg_mat >= ps).sum(dim=1).float() + 1.0
    return {
        "mrr":     (1.0 / ranks).mean().item(),
        "hits@1":  (ranks <= 1).float().mean().item(),
        "hits@3":  (ranks <= 3).float().mean().item(),
        "hits@10": (ranks <= 10).float().mean().item(),
        "mean_rank": ranks.mean().item(),
    }


def per_class_zeroshot_all(model_name, model, graph, splits, device,
                           gene_x_override=None, drug_x_override=None):
    """Compute all-class ZS MRR per drug class (RANKING AGAINST ALL 46 CLASSES)."""
    meta = splits["meta"]
    zs_idx_list  = meta["zs_drug_class_indices"]
    zs_name_list = meta["zs_drug_class_names"]
    zs_split     = splits["zeroshot"]
    dc2i         = {v: k for k, v in enumerate(range(graph["drug_class"].x.shape[0]))}

    pos_src = np.array(zs_split["pos_src"])
    pos_dst = np.array(zs_split["pos_dst"])
    neg_src = np.array(zs_split["neg_src_all"])
    neg_dst = np.array(zs_split["neg_dst_all"])

    results = {}
    for dc_idx, dc_name in zip(zs_idx_list, zs_name_list):
        p_mask = (pos_dst == dc_idx)
        n_mask = (neg_dst == dc_idx)
        if p_mask.sum() == 0 or n_mask.sum() == 0:
            continue

        ps_t = torch.tensor(pos_src[p_mask], dtype=torch.long, device=device)
        pd_t = torch.tensor(pos_dst[p_mask], dtype=torch.long, device=device)
        ns_t = torch.tensor(neg_src[n_mask], dtype=torch.long, device=device)
        nd_t = torch.tensor(neg_dst[n_mask], dtype=torch.long, device=device)

        pss = score_all_pairs(model_name, model, graph, ps_t, pd_t, device,
                              gene_x_override, drug_x_override)
        nss = score_all_pairs(model_name, model, graph, ns_t, nd_t, device,
                              gene_x_override, drug_x_override)

        n_p   = len(pss)
        n_npp = max(1, len(nss) // n_p)
        nm    = nss[:n_p * n_npp].view(n_p, n_npp)
        m     = compute_mrr_hits(pss, nm)
        m["n_edges"] = int(p_mask.sum())
        results[dc_name] = m

    return results


def compute_p_at_k_case_study(model_name, model, graph, splits, device, k=20,
                              gene_x_override=None, drug_x_override=None):
    """
    For each ZS drug class, compute P@k by ranking ALL genes for that drug class
    and checking how many top-k are true positives.
    """
    meta      = splits["meta"]
    gene_meta = None  # Will load separately if needed
    zs_idx_list  = meta["zs_drug_class_indices"]
    zs_name_list = meta["zs_drug_class_names"]
    pos_pairs    = meta["pos_pairs"]

    # Get all gene embeddings at once
    n_genes = graph["gene"].x.shape[0]
    n_dc    = graph["drug_class"].x.shape[0]

    results = {}
    model.eval()
    with torch.no_grad():
        if model_name == "biomolamr":
            gene_emb = model.encode_genes(graph.to(device))
            drug_emb = model.encode_drug_classes(graph.to(device))
        elif model_name == "rgcn_bio":
            gene_emb, drug_emb = model.encode(graph.to(device))
        elif model_name in ("distmult", "transe"):
            gene_emb = model.gene_emb.weight
            drug_emb = model.drug_emb.weight
        else:
            return {}   # Feature-MLP doesn't have explicit embeddings

    for dc_idx, dc_name in zip(zs_idx_list, zs_name_list):
        # Score all genes for this drug class
        if model_name == "biomolamr":
            d_emb = drug_emb[dc_idx : dc_idx+1]   # (1, D)
            scores = model.decoder.score_all(gene_emb, d_emb).squeeze(1)  # (G,)
        elif model_name == "rgcn_bio":
            d_emb = drug_emb[dc_idx : dc_idx+1]   # (1, D)
            scores = (gene_emb * model.W * d_emb).sum(dim=1)              # (G,)
        elif model_name in ("distmult", "transe"):
            d_emb = drug_emb[dc_idx : dc_idx+1]
            if model_name == "distmult":
                scores = (gene_emb * model.W * d_emb).sum(dim=1)
            else:
                r = model.relation
                scores = -torch.norm(gene_emb + r - d_emb, dim=1)
        else:
            continue

        top_k_idx = scores.topk(k).indices.cpu().numpy()
        true_set  = set(g for (g, d) in pos_pairs if d == dc_idx)
        n_correct = sum(1 for g in top_k_idx if g in true_set)
        results[dc_name] = {
            "P@20":       n_correct / k,
            "n_correct":  n_correct,
            "n_true":     len(true_set),
            "top_genes":  top_k_idx.tolist(),
        }

    return results


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_zs_all_comparison(summary):
    """Bar chart: ZS-all MRR comparison (all-class, primary metric)."""
    models = [m for m in MODEL_NAMES if m in summary]
    mrrs   = [summary[m]["zs_all"]["mrr"]["mean"] for m in models]
    stds   = [summary[m]["zs_all"]["mrr"]["std"]  for m in models]
    random_baseline = 1 / 46

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(models))
    bars = ax.bar(x, mrrs, yerr=stds, capsize=5,
                  color=[COLORS.get(m, "#888") for m in models],
                  alpha=0.85, edgecolor="white", linewidth=0.8)
    ax.axhline(random_baseline, ls="--", color="gray", linewidth=1.2,
               label=f"Random baseline ({random_baseline:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(m, m) for m in models], rotation=15, ha="right")
    ax.set_ylabel("MRR (All-Class Filtered)", fontsize=12)
    ax.set_title("Zero-Shot AMR Prediction: All-Class Ranking (Primary Metric)", fontsize=13)
    ax.legend()
    for bar, mrr, std in zip(bars, mrrs, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.003,
                f"{mrr:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "F_zs_all_comparison.png")
    plt.savefig(FIGS_DIR / "F_zs_all_comparison.pdf")
    plt.close()
    print("  Saved ZS-all comparison figure")


def plot_per_class_heatmap(summary):
    """Heatmap of per-class ZS-all MRR across all models."""
    models   = [m for m in MODEL_NAMES if m in summary and "per_dc_zs_all" in summary[m]]
    if not models:
        return
    classes  = sorted(summary[models[0]]["per_dc_zs_all"].keys())

    data_mat = np.zeros((len(models), len(classes)))
    for i, m in enumerate(models):
        pdc = summary[m].get("per_dc_zs_all", {})
        for j, dc in enumerate(classes):
            data_mat[i, j] = pdc.get(dc, {}).get("mrr", 0)

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(data_mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([c.replace(" antibiotic", "") for c in classes],
                        rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([LABELS.get(m, m) for m in models])
    ax.set_title("Per-Class Zero-Shot MRR (All-Class Ranking)", fontsize=13)
    plt.colorbar(im, ax=ax, label="ZS-All MRR")
    # Add random baseline line
    for i in range(len(classes)):
        ax.axhline(len(models) - 0.5, color="black", linewidth=0.3)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "F_perclass_heatmap_all.png")
    plt.savefig(FIGS_DIR / "F_perclass_heatmap_all.pdf")
    plt.close()
    print("  Saved per-class heatmap")


def plot_tanimoto_correlation(summary, tanimoto, splits, dc2i):
    """Scatter: Tanimoto(ZS drug, nearest training drug) vs ZS-all MRR."""
    meta = splits["meta"]
    i2dc = {v: k for k, v in dc2i.items()}
    zs_names   = meta["zs_drug_class_names"]
    train_idxs = set(meta["train_drug_class_indices"])
    tan_np     = tanimoto.numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name in MODEL_NAMES:
        if model_name not in summary or "per_dc_zs_all" not in summary[model_name]:
            continue
        pdc = summary[model_name]["per_dc_zs_all"]
        xs, ys = [], []
        for dc_name in zs_names:
            if dc_name not in dc2i or dc_name not in pdc:
                continue
            zs_idx = dc2i[dc_name]
            # Nearest training drug by Tanimoto
            train_sims = [tan_np[zs_idx, ti] for ti in train_idxs]
            max_tan = max(train_sims) if train_sims else 0
            xs.append(max_tan)
            ys.append(pdc[dc_name]["mrr"])
        if xs:
            ax.scatter(xs, ys, color=COLORS.get(model_name, "#888"),
                      s=80, alpha=0.75, label=LABELS.get(model_name, model_name))
            if len(xs) >= 3:
                z = np.polyfit(xs, ys, 1)
                px = np.linspace(0, 1, 100)
                ax.plot(px, np.polyval(z, px), "--",
                        color=COLORS.get(model_name, "#888"), alpha=0.4, lw=1.2)

    ax.axhline(1/46, ls=":", color="gray", lw=1.2, label="Random (1/46)")
    ax.set_xlabel("Max Tanimoto Similarity to Nearest Training Drug Class", fontsize=12)
    ax.set_ylabel("Zero-Shot All-Class MRR", fontsize=12)
    ax.set_title("Chemical Similarity ↔ Zero-Shot Transfer Performance", fontsize=13)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "F_tanimoto_vs_zs_mrr.png")
    plt.savefig(FIGS_DIR / "F_tanimoto_vs_zs_mrr.pdf")
    plt.close()
    print("  Saved Tanimoto correlation figure")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Step 08: Extended evaluation (all-class ZS ranking)")
    print("=" * 60)

    ckpts = load_checkpoints()
    if not ckpts:
        print("No checkpoints found. Run pipeline/03b_train_biomolamr.py first.")
        return
    print(f"Models found: {sorted(ckpts.keys())}")

    graph_path  = GRAPH_NEW  if GRAPH_NEW.exists()  else GRAPH_ORIG
    splits_path = SPLITS_EXT if SPLITS_EXT.exists() else SPLITS_STD
    with open(graph_path,  "rb") as f:  obj    = pickle.load(f)
    with open(splits_path, "rb") as f:  splits = pickle.load(f)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    graph  = obj["hetero_data"].to(device)
    dc2i   = obj["dc2i"]

    tanimoto = None
    if TAN_FILE.exists():
        tanimoto = torch.load(TAN_FILE, map_location="cpu")

    # Load leaky graph for fmlp_leaky ablation
    leaky_graph = None
    if GRAPH_ORIG.exists():
        with open(GRAPH_ORIG, "rb") as f:
            leaky_obj = pickle.load(f)
        leaky_graph = leaky_obj["hetero_data"]

    def get_feature_overrides(model_name, graph_dev, leaky_g):
        """Return (gene_x, drug_x) overrides for ablation models."""
        if model_name == "fmlp_gene_zero":
            return torch.zeros_like(graph_dev["gene"].x), graph_dev["drug_class"].x
        elif model_name == "fmlp_drug_zero":
            return graph_dev["gene"].x, torch.zeros_like(graph_dev["drug_class"].x)
        elif model_name == "fmlp_leaky" and leaky_g is not None:
            return leaky_g["gene"].x.to(graph_dev["gene"].x.device), graph_dev["drug_class"].x
        return None, None

    summary = {}

    for model_name in MODEL_NAMES:
        if model_name not in ckpts:
            continue

        seed_list = ckpts[model_name]
        test_agg  = agg_metrics(seed_list, "test_metrics")
        zw_key    = "zs_within_metrics" if "zs_within_metrics" in seed_list[0] else "zs_metrics"
        zw_agg    = agg_metrics(seed_list, zw_key)
        za_agg    = agg_metrics(seed_list, "zs_all_metrics")

        # Full analysis on best seed (42)
        ck0 = next((c for c in seed_list if c["seed"] == 42), seed_list[0])
        model = build_model_from_ck(model_name, ck0, obj["hetero_data"]).to(device)
        model.eval()

        gx_ov, dx_ov = get_feature_overrides(model_name, graph, leaky_graph)

        per_dc = per_class_zeroshot_all(model_name, model, graph, splits, device,
                                        gx_ov, dx_ov)
        p_at_k = compute_p_at_k_case_study(model_name, model, graph, splits, device,
                                            gene_x_override=gx_ov,
                                            drug_x_override=dx_ov)

        gen_gap_within = (test_agg["mrr"]["mean"] - zw_agg["mrr"]["mean"]
                          if "mrr" in zw_agg and "mrr" in test_agg else None)
        gen_gap_all    = (test_agg["mrr"]["mean"] - za_agg["mrr"]["mean"]
                          if "mrr" in za_agg and "mrr" in test_agg else None)

        summary[model_name] = {
            "n_seeds":         len(seed_list),
            "test":            test_agg,
            "zs_within":       zw_agg,
            "zs_all":          za_agg,
            "gen_gap_within":  gen_gap_within,
            "gen_gap_all":     gen_gap_all,
            "per_dc_zs_all":   per_dc,
            "p_at_20":         p_at_k,
        }

        n_zs = splits["meta"]["n_zs_classes"]
        print(f"\n  {model_name.upper()} ({len(seed_list)} seeds)")
        if "mrr" in test_agg:
            print(f"    Test     MRR={test_agg['mrr']['mean']:.4f}±{test_agg['mrr']['std']:.4f}  "
                  f"H@10={test_agg['hits@10']['mean']:.4f}")
        if "mrr" in zw_agg:
            print(f"    ZS-win   MRR={zw_agg['mrr']['mean']:.4f}±{zw_agg['mrr']['std']:.4f}  "
                  f"[{n_zs}-way]")
        if "mrr" in za_agg:
            print(f"    ZS-all   MRR={za_agg['mrr']['mean']:.4f}±{za_agg['mrr']['std']:.4f}  "
                  f"H@10={za_agg['hits@10']['mean']:.4f}  [PRIMARY]")
        if per_dc:
            print(f"    Per-class ZS-all MRR:")
            for dc, v in sorted(per_dc.items()):
                print(f"      {dc[:35]:<35}: MRR={v['mrr']:.4f}  n={v['n_edges']}")
        if p_at_k:
            print(f"    P@20 case studies:")
            for dc, v in p_at_k.items():
                print(f"      {dc[:35]:<35}: P@20={v['P@20']:.2f}  ({v['n_correct']}/{20})")

    # Save JSON
    out_json = ROOT / "results/biomolamr/eval_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved → {out_json}")

    # Generate figures
    print("\nGenerating figures …")
    plot_zs_all_comparison(summary)
    plot_per_class_heatmap(summary)
    if tanimoto is not None:
        plot_tanimoto_correlation(summary, tanimoto, splits, dc2i)

    # LaTeX summary table
    lines = []
    lines.append(r"\begin{table}[t]\centering")
    lines.append(r"\caption{BioMolAMR Extended Evaluation (12 ZS classes, 5 seeds)}")
    lines.append(r"\label{tab:biomolamr}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Test MRR} & \textbf{ZS-Within MRR} "
                 r"& \textbf{ZS-All MRR$\dagger$} & \textbf{Gen. Gap} \\")
    lines.append(r"\midrule")
    for mn, md in summary.items():
        t = md["test"].get("mrr", {})
        w = md["zs_within"].get("mrr", {})
        a = md["zs_all"].get("mrr", {})
        gg = md.get("gen_gap_all") or 0
        lines.append(
            f"{LABELS.get(mn,mn)} & "
            f"{t.get('mean',0):.3f}$\\pm${t.get('std',0):.3f} & "
            f"{w.get('mean',0):.3f}$\\pm${w.get('std',0):.3f} & "
            f"{a.get('mean',0):.3f}$\\pm${a.get('std',0):.3f} & "
            f"{gg:+.3f} \\\\"
        )
    rand_zs_all = 1/46
    rand_zs_win = 1/splits["meta"]["n_zs_classes"]
    lines.append(rf"\textit{{Random}} & — & {rand_zs_win:.3f} & {rand_zs_all:.3f} & — \\")
    lines.append(r"\bottomrule")
    lines.append(r"\multicolumn{5}{l}{\small $\dagger$ Primary metric: rank among all 46 drug classes.}")
    lines.append(r"\end{tabular}\end{table}")

    latex_path = TABLES_DIR / "biomolamr_main_results.tex"
    latex_path.write_text("\n".join(lines))
    print(f"LaTeX table → {latex_path}")


if __name__ == "__main__":
    main()
