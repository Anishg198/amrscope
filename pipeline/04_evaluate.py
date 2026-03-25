"""
Step 4: Comprehensive evaluation across all models and seeds.

Computes:
  • MRR, Hits@1, Hits@3, Hits@10 with mean ± std across seeds
  • Generalisation gap (test → zero-shot degradation)
  • Per-drug-class zero-shot performance
  • AUC-ROC and Average Precision
  • Saves tables as CSV + LaTeX

Output: results/zeroshot/eval_summary.json
         results/zeroshot/tables/
"""

import json, pickle, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "results/zeroshot/models"
TABLES_DIR = ROOT / "results/zeroshot/tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_PKL  = ROOT / "data/processed/enhanced_graph.pkl"
SPLITS_PKL = ROOT / "data/processed/splits.pkl"

MODEL_NAMES = ["zs_hetgat", "rgcn", "feature_mlp", "distmult", "transe"]
METRICS     = ["mrr", "hits@1", "hits@3", "hits@10"]

RANDOM_BASELINE = {
    "zs_within_mrr":  0.4772,  # H(5)/5 for 5-class ranking
    "zs_within_h10":  0.4800,
}

ZS_CLASS_OVERLAPS = {
    "glycylcycline":             100,
    "lincosamide antibiotic":     56,
    "streptogramin A antibiotic": 100,
    "rifamycin antibiotic":        47,
    "glycopeptide antibiotic":      1,
}


def load_checkpoints():
    ckpts = defaultdict(list)
    for path in sorted(MODELS_DIR.glob("*.pt")):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        ckpts[ck["model_name"]].append(ck)
    return ckpts


def agg(runs, key):
    vals = [r[key][m] for r in runs for m in METRICS if key in r]
    # Actually restructure:
    out = {}
    for m in METRICS:
        vs = [r[key][m] for r in runs if key in r and m in r[key]]
        if vs:
            out[m] = {"mean": float(np.mean(vs)), "std": float(np.std(vs))}
    return out


def compute_auc(model_name, model, graph, splits, split_key, device):
    from src.models.zeroshot_hetgat import ZeroShotHetGAT
    from src.models.baselines import (DistMultBaseline, TransEBaseline,
                                       FeatureMLPBaseline, RGCNBaseline)
    split   = splits[split_key]
    neg_src_k = "neg_src_all" if split_key == "zeroshot" else "neg_src"
    neg_dst_k = "neg_dst_all" if split_key == "zeroshot" else "neg_dst"

    ps = torch.tensor(split["pos_src"], dtype=torch.long, device=device)
    pd = torch.tensor(split["pos_dst"], dtype=torch.long, device=device)
    n  = len(ps)
    ns = torch.tensor(split[neg_src_k], dtype=torch.long, device=device)[:n]
    nd = torch.tensor(split[neg_dst_k], dtype=torch.long, device=device)[:n]

    model.eval()
    with torch.no_grad():
        if model_name in ("zs_hetgat", "rgcn"):
            s_pos = model(graph, ps, pd)
            s_neg = model(graph, ns, nd)
        elif model_name in ("distmult", "transe"):
            s_pos = model(ps, pd)
            s_neg = model(ns, nd)
        elif model_name == "feature_mlp":
            s_pos = model(graph["gene"].x, graph["drug_class"].x, ps, pd)
            s_neg = model(graph["gene"].x, graph["drug_class"].x, ns, nd)

    scores = torch.cat([s_pos, s_neg]).sigmoid().cpu().numpy()
    labels = np.concatenate([np.ones(n), np.zeros(n)])
    auc = roc_auc_score(labels, scores)
    ap  = average_precision_score(labels, scores)
    return auc, ap


def per_drug_class_zs(model_name, model, graph, splits, device):
    meta = splits["meta"]
    zs_idx_list  = meta["zs_drug_class_indices"]
    zs_name_list = meta["zs_drug_class_names"]
    zs_split     = splits["zeroshot"]

    pos_src = np.array(zs_split["pos_src"])
    pos_dst = np.array(zs_split["pos_dst"])
    neg_src = np.array(zs_split["neg_src"])
    neg_dst = np.array(zs_split["neg_dst"])

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

        model.eval()
        with torch.no_grad():
            if model_name in ("zs_hetgat", "rgcn"):
                pss = model(graph, ps_t, pd_t)
                nss = model(graph, ns_t, nd_t)
            elif model_name in ("distmult", "transe"):
                pss = model(ps_t, pd_t)
                nss = model(ns_t, nd_t)
            elif model_name == "feature_mlp":
                pss = model(graph["gene"].x, graph["drug_class"].x, ps_t, pd_t)
                nss = model(graph["gene"].x, graph["drug_class"].x, ns_t, nd_t)

        n_p   = len(pss)
        n_npp = max(1, len(nss) // n_p)
        nm    = nss[:n_p * n_npp].view(n_p, n_npp)
        ranks = (nm >= pss.unsqueeze(1)).sum(dim=1).float() + 1.0
        mrr   = (1.0 / ranks).mean().item()
        h1    = (ranks <= 1).float().mean().item()
        h10   = (ranks <= 10).float().mean().item()
        overlap = ZS_CLASS_OVERLAPS.get(dc_name, "?")
        results[dc_name] = {
            "mrr": mrr, "hits@1": h1, "hits@10": h10,
            "n_edges": int(p_mask.sum()),
            "gene_overlap_pct": overlap,
        }
    return results


def build_model_from_ck(model_name, ck, graph):
    from src.models.zeroshot_hetgat import ZeroShotHetGAT
    from src.models.baselines import (DistMultBaseline, TransEBaseline,
                                       FeatureMLPBaseline, RGCNBaseline)
    hp       = ck["hparams"]
    n_genes  = graph["gene"].x.shape[0]
    n_dc     = graph["drug_class"].x.shape[0]
    gene_dim = graph["gene"].x.shape[1]
    drug_dim = graph["drug_class"].x.shape[1]
    mech_dim = graph["mechanism"].x.shape[1]

    if model_name == "zs_hetgat":
        m = ZeroShotHetGAT(gene_dim, drug_dim, mech_dim,
                            hp["hidden_dim"], hp["out_dim"],
                            hp["num_heads"], hp["dropout"])
    elif model_name == "distmult":
        m = DistMultBaseline(n_genes, n_dc, hp["emb_dim"], hp["dropout"])
    elif model_name == "transe":
        m = TransEBaseline(n_genes, n_dc, hp["emb_dim"], hp["margin"], hp["dropout"])
    elif model_name == "rgcn":
        m = RGCNBaseline(gene_dim, drug_dim, mech_dim,
                         hp["hidden_dim"], hp["out_dim"], hp["dropout"])
    elif model_name == "feature_mlp":
        m = FeatureMLPBaseline(gene_dim, drug_dim, hp["hidden_dim"], hp["dropout"])
    m.load_state_dict(ck["state_dict"])
    return m


def to_latex(rows, caption, label, header_cols):
    col_spec = "l" + "c" * (len(header_cols) - 1)
    header   = " & ".join(f"\\textbf{{{c}}}" for c in header_cols) + " \\\\"
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        header,
        "\\midrule",
    ] + [r + " \\\\" for r in rows] + ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def fmt(mean, std=None):
    if std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f}$\\pm${std:.4f}"


def main():
    print("=" * 60)
    print("Step 4: Comprehensive evaluation")
    print("=" * 60)

    ckpts = load_checkpoints()
    if not ckpts:
        print("No checkpoints. Run pipeline/03_train.py first.")
        return
    print(f"Models found: {sorted(ckpts.keys())}")

    with open(GRAPH_PKL, "rb") as f:
        obj = pickle.load(f)
    with open(SPLITS_PKL, "rb") as f:
        splits = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph  = obj["hetero_data"].to(device)

    summary     = {}
    test_rows   = []
    zs_rows     = []

    for model_name in MODEL_NAMES:
        if model_name not in ckpts:
            continue

        seed_list = ckpts[model_name]
        test_agg  = agg(seed_list, "test_metrics")
        zs_key    = "zs_within_metrics" if "zs_within_metrics" in seed_list[0] else "zs_metrics"
        zs_agg    = agg(seed_list, zs_key)

        # AUC from best seed (seed 42)
        ck0   = next((c for c in seed_list if c["seed"] == 42), seed_list[0])
        model = build_model_from_ck(model_name, ck0, graph).to(device)
        test_auc, test_ap = compute_auc(model_name, model, graph, splits, "test",     device)
        zs_auc,   zs_ap   = compute_auc(model_name, model, graph, splits, "zeroshot", device)

        # Per-drug-class zero-shot
        per_dc = per_drug_class_zs(model_name, model, graph, splits, device)

        gen_gap = test_agg["mrr"]["mean"] - zs_agg["mrr"]["mean"]
        summary[model_name] = {
            "n_seeds":  len(seed_list),
            "test":     {**test_agg, "auc": test_auc, "ap": test_ap},
            "zeroshot": {**zs_agg,   "auc": zs_auc,   "ap": zs_ap},
            "generalisation_gap_mrr": gen_gap,
            "per_dc_zeroshot": per_dc,
        }

        label = model_name.replace("_", r"\_")
        def row_str(d):
            return " & ".join([
                label,
                fmt(d["mrr"]["mean"], d["mrr"]["std"]),
                fmt(d["hits@1"]["mean"], d["hits@1"]["std"]),
                fmt(d["hits@3"]["mean"], d["hits@3"]["std"]),
                fmt(d["hits@10"]["mean"], d["hits@10"]["std"]),
                fmt(test_auc if d is test_agg else zs_auc),
            ])

        test_rows.append(row_str(test_agg))
        zs_rows.append(row_str(zs_agg))

        print(f"\n  {model_name.upper()} ({len(seed_list)} seeds)")
        print(f"    Test     MRR={test_agg['mrr']['mean']:.4f}±{test_agg['mrr']['std']:.4f}  "
              f"H@10={test_agg['hits@10']['mean']:.4f}  AUC={test_auc:.4f}")
        print(f"    ZS-Win   MRR={zs_agg['mrr']['mean']:.4f}±{zs_agg['mrr']['std']:.4f}  "
              f"H@10={zs_agg['hits@10']['mean']:.4f}  AUC={zs_auc:.4f}")
        print(f"    Gen.Gap  ΔMRR={gen_gap:+.4f}")
        if per_dc:
            print(f"    Per-class ZS MRR:")
            for dc, v in per_dc.items():
                print(f"      {dc} (overlap={v['gene_overlap_pct']}%): "
                      f"MRR={v['mrr']:.4f}  H@10={v['hits@10']:.4f}  n={v['n_edges']}")

    # Random baseline row
    test_rows.append(r"Random & -- & -- & -- & -- & 0.5000 \\")
    zs_rows.append(r"Random & 0.4772 & -- & -- & -- & 0.5000 \\")

    # Save JSON
    out_json = ROOT / "results/zeroshot/eval_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved eval summary → {out_json}")

    # LaTeX tables
    cols = ["Model", "MRR", "Hits@1", "Hits@3", "Hits@10", "AUC"]
    (TABLES_DIR / "test_results.tex").write_text(
        to_latex(test_rows,
                 "Standard link prediction on seen drug classes",
                 "tab:standard", cols))
    (TABLES_DIR / "zeroshot_results.tex").write_text(
        to_latex(zs_rows,
                 r"Zero-shot link prediction (within held-out drug classes, n$_{\rm zs}$=5)",
                 "tab:zeroshot", cols))
    print(f"LaTeX tables → {TABLES_DIR}")

    # CSV
    import csv
    for sname, rows_src, auc_key in [
        ("test",     "test",     "test"),
        ("zeroshot", "zeroshot", "zeroshot"),
    ]:
        with open(TABLES_DIR / f"{sname}_results.csv", "w", newline="") as cf:
            w = csv.writer(cf)
            w.writerow(["model", "n_seeds", "mrr_mean", "mrr_std",
                        "h1_mean", "h1_std", "h3_mean", "h3_std",
                        "h10_mean", "h10_std", "auc", "ap", "gen_gap"])
            for mn, md in summary.items():
                sd  = md[auc_key if auc_key == "zeroshot" else "test"]
                auc = sd.get("auc", 0)
                ap  = sd.get("ap", 0)
                def g(k): return sd.get(k, {}).get("mean", 0), sd.get(k, {}).get("std", 0)
                mrr = g("mrr"); h1 = g("hits@1"); h3 = g("hits@3"); h10 = g("hits@10")
                w.writerow([mn, md["n_seeds"],
                             f"{mrr[0]:.4f}", f"{mrr[1]:.4f}",
                             f"{h1[0]:.4f}",  f"{h1[1]:.4f}",
                             f"{h3[0]:.4f}",  f"{h3[1]:.4f}",
                             f"{h10[0]:.4f}", f"{h10[1]:.4f}",
                             f"{auc:.4f}", f"{ap:.4f}",
                             f"{md['generalisation_gap_mrr']:+.4f}"])
    print(f"CSV files → {TABLES_DIR}")


if __name__ == "__main__":
    main()
