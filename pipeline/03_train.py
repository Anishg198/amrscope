"""
Step 3: Train ZeroShotHetGAT and baselines.

Evaluation protocols:
  • test         – standard split on seen drug classes
  • zeroshot     – within-zero-shot ranking (rank among held-out drug classes)
  • zeroshot_all – all-class ranking with seen class negatives

Usage:
    python pipeline/03_train.py --model zs_hetgat --seed 42
    python pipeline/03_train.py --model distmult  --seed 42
    python pipeline/03_train.py --all_models
"""

import argparse, json, pickle, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.zeroshot_hetgat import ZeroShotHetGAT
from src.models.baselines import DistMultBaseline, TransEBaseline, FeatureMLPBaseline, RGCNBaseline

GRAPH_PKL  = ROOT / "data/processed/enhanced_graph.pkl"
SPLITS_PKL = ROOT / "data/processed/splits.pkl"
MODELS_DIR = ROOT / "results/zeroshot/models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

HPARAMS = {
    "zs_hetgat": {
        "hidden_dim": 256, "out_dim": 128, "num_heads": 4,
        "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-4,
        "epochs": 300, "patience": 3, "batch_size": 2048,
        "contrastive_weight": 0.1,
    },
    "distmult": {
        "emb_dim": 128, "dropout": 0.3,
        "lr": 5e-3, "weight_decay": 1e-5,
        "epochs": 500, "patience": 4, "batch_size": 2048,
    },
    "transe": {
        "emb_dim": 128, "margin": 1.0, "dropout": 0.3,
        "lr": 1e-3, "weight_decay": 1e-5,
        "epochs": 500, "patience": 4, "batch_size": 2048,
    },
    "rgcn": {
        "hidden_dim": 256, "out_dim": 128, "dropout": 0.3,
        "lr": 1e-3, "weight_decay": 1e-4,
        "epochs": 300, "patience": 3, "batch_size": 2048,
    },
    "feature_mlp": {
        "hidden_dim": 256, "dropout": 0.3,
        "lr": 1e-3, "weight_decay": 1e-4,
        "epochs": 300, "patience": 3, "batch_size": 2048,
    },
}


def compute_mrr_hits(pos_scores: torch.Tensor, neg_mat: torch.Tensor):
    """
    pos_scores : (N,)
    neg_mat    : (N, K) – K negatives per positive
    """
    ps = pos_scores.unsqueeze(1)   # (N, 1)
    ranks = (neg_mat >= ps).sum(dim=1).float() + 1.0
    mrr    = (1.0 / ranks).mean().item()
    hits1  = (ranks <= 1).float().mean().item()
    hits3  = (ranks <= 3).float().mean().item()
    hits10 = (ranks <= 10).float().mean().item()
    return mrr, hits1, hits3, hits10


def bce_loss(ps, ns):
    return (
        F.binary_cross_entropy_with_logits(ps, torch.ones_like(ps)) +
        F.binary_cross_entropy_with_logits(ns, torch.zeros_like(ns))
    )


def score_pairs(model_name, model, graph, src_t, dst_t):
    if model_name in ("zs_hetgat", "rgcn"):
        return model(graph, src_t, dst_t)
    elif model_name in ("distmult", "transe"):
        return model(src_t, dst_t)
    elif model_name == "feature_mlp":
        return model(graph["gene"].x, graph["drug_class"].x, src_t, dst_t)
    raise ValueError(f"Unknown model: {model_name}")


def evaluate_split(model_name, model, graph, splits, split_name, device,
                   use_within_zs=False):
    """
    Evaluate on a split.
    For zero-shot split: use within-ZS negatives (use_within_zs=True)
                         or all-class negatives (use_within_zs=False).
    """
    split = splits[split_name]
    pos_src = torch.tensor(split["pos_src"], dtype=torch.long, device=device)
    pos_dst = torch.tensor(split["pos_dst"], dtype=torch.long, device=device)

    if split_name == "zeroshot" and not use_within_zs:
        neg_src_key, neg_dst_key = "neg_src_all", "neg_dst_all"
    else:
        neg_src_key, neg_dst_key = "neg_src", "neg_dst"

    neg_src = torch.tensor(split[neg_src_key], dtype=torch.long, device=device)
    neg_dst = torch.tensor(split[neg_dst_key], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        ps = score_pairs(model_name, model, graph, pos_src, pos_dst)

        n_pos    = len(pos_src)
        n_neg    = len(neg_src)
        n_neg_pp = max(1, n_neg // n_pos)
        ns = score_pairs(model_name, model, graph,
                         neg_src[:n_pos * n_neg_pp],
                         neg_dst[:n_pos * n_neg_pp])
        neg_mat  = ns.view(n_pos, n_neg_pp)
        mrr, h1, h3, h10 = compute_mrr_hits(ps, neg_mat)

    return {"mrr": mrr, "hits@1": h1, "hits@3": h3, "hits@10": h10}


def compute_drug_sim_matrix(graph):
    """Pre-compute cosine similarity matrix for drug class features."""
    drug_x = graph["drug_class"].x
    drug_n = F.normalize(drug_x.float(), dim=-1)
    return (drug_n @ drug_n.t()).clamp(-1, 1)


def build_model(model_name, graph, hp, device):
    n_genes  = graph["gene"].x.shape[0]
    n_dc     = graph["drug_class"].x.shape[0]
    gene_dim = graph["gene"].x.shape[1]
    drug_dim = graph["drug_class"].x.shape[1]
    mech_dim = graph["mechanism"].x.shape[1]

    if model_name == "zs_hetgat":
        model = ZeroShotHetGAT(
            gene_feat_dim=gene_dim, drug_feat_dim=drug_dim, mech_feat_dim=mech_dim,
            hidden_dim=hp["hidden_dim"], out_dim=hp["out_dim"],
            num_heads=hp["num_heads"], dropout=hp["dropout"])
    elif model_name == "distmult":
        model = DistMultBaseline(n_genes, n_dc, hp["emb_dim"], hp["dropout"])
    elif model_name == "transe":
        model = TransEBaseline(n_genes, n_dc, hp["emb_dim"], hp["margin"], hp["dropout"])
    elif model_name == "rgcn":
        model = RGCNBaseline(gene_dim, drug_dim, mech_dim,
                             hp["hidden_dim"], hp["out_dim"], hp["dropout"])
    elif model_name == "feature_mlp":
        model = FeatureMLPBaseline(gene_dim, drug_dim, hp["hidden_dim"], hp["dropout"])
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)


def train_model(model_name: str, seed: int, graph, splits):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[{model_name}] seed={seed}  device={device}")

    graph = graph.to(device)
    hp    = HPARAMS[model_name]
    model = build_model(model_name, graph, hp, device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Drug similarity matrix for contrastive loss
    drug_sim = compute_drug_sim_matrix(graph).to(device) \
               if model_name == "zs_hetgat" else None

    optimizer = optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp["epochs"])

    split = splits["train"]
    pos_src, pos_dst = split["pos_src"], split["pos_dst"]
    neg_src, neg_dst = split["neg_src"], split["neg_dst"]
    n_pos     = len(pos_src)
    n_neg_pp  = max(1, len(neg_src) // n_pos)

    best_val_mrr = -1.0
    best_state   = None
    patience_ctr = 0
    history      = []
    t0           = time.time()

    for epoch in range(1, hp["epochs"] + 1):
        model.train()
        perm     = np.random.permutation(n_pos)
        epoch_loss = 0.0; n_batches = 0

        for start in range(0, n_pos, hp["batch_size"]):
            bi   = perm[start:start + hp["batch_size"]]
            ps_t = torch.tensor(pos_src[bi], dtype=torch.long, device=device)
            pd_t = torch.tensor(pos_dst[bi], dtype=torch.long, device=device)
            ns   = start * n_neg_pp
            ne   = min(ns + len(bi) * n_neg_pp, len(neg_src))
            ns_t = torch.tensor(neg_src[ns:ne], dtype=torch.long, device=device)
            nd_t = torch.tensor(neg_dst[ns:ne], dtype=torch.long, device=device)

            optimizer.zero_grad()
            ps_s = score_pairs(model_name, model, graph, ps_t, pd_t)
            ns_s = score_pairs(model_name, model, graph, ns_t, nd_t)
            loss = bce_loss(ps_s, ns_s)

            # Contrastive drug-class regularisation (ZS-HetGAT only)
            if drug_sim is not None and hp.get("contrastive_weight", 0) > 0:
                loss = loss + hp["contrastive_weight"] * \
                       model.drug_class_contrastive_loss(graph, drug_sim)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item(); n_batches += 1

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            val_m = evaluate_split(model_name, model, graph, splits, "val", device)
            val_mrr = val_m["mrr"]
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:4d}/{hp['epochs']}  loss={epoch_loss/max(n_batches,1):.4f}  "
                  f"val_MRR={val_mrr:.4f}  H@1={val_m['hits@1']:.4f}  "
                  f"H@10={val_m['hits@10']:.4f}  ({elapsed:.0f}s)")
            history.append({"epoch": epoch, "loss": epoch_loss/max(n_batches,1), **val_m})

            if val_mrr > best_val_mrr + 1e-4:
                best_val_mrr = val_mrr
                best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= hp["patience"]:
                    print(f"  Early stopping at epoch {epoch}")
                    break

    if best_state:
        model.load_state_dict(best_state); model.to(device)

    test_m    = evaluate_split(model_name, model, graph, splits, "test",     device)
    zs_within = evaluate_split(model_name, model, graph, splits, "zeroshot", device,
                                use_within_zs=True)
    zs_all    = evaluate_split(model_name, model, graph, splits, "zeroshot", device,
                                use_within_zs=False)

    print(f"\n  [RESULTS] {model_name} seed={seed}")
    print(f"    Test        MRR={test_m['mrr']:.4f}  H@1={test_m['hits@1']:.4f}  "
          f"H@3={test_m['hits@3']:.4f}  H@10={test_m['hits@10']:.4f}")
    print(f"    ZS-within   MRR={zs_within['mrr']:.4f}  H@1={zs_within['hits@1']:.4f}  "
          f"H@3={zs_within['hits@3']:.4f}  H@10={zs_within['hits@10']:.4f}  "
          f"[ranked among {len(splits['meta']['zs_drug_class_names'])} ZS classes]")
    print(f"    ZS-all      MRR={zs_all['mrr']:.4f}  H@1={zs_all['hits@1']:.4f}  "
          f"H@3={zs_all['hits@3']:.4f}  H@10={zs_all['hits@10']:.4f}  "
          f"[ranked among all drug classes]")

    ckpt_path = MODELS_DIR / f"{model_name}_seed{seed}.pt"
    torch.save({
        "model_name": model_name, "seed": seed, "state_dict": best_state,
        "test_metrics": test_m,
        "zs_within_metrics": zs_within,
        "zs_all_metrics": zs_all,
        "history": history, "hparams": hp,
    }, ckpt_path)
    print(f"  Saved → {ckpt_path}")

    return test_m, zs_within, zs_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="zs_hetgat",
                        choices=list(HPARAMS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--all_models", action="store_true")
    args = parser.parse_args()

    print("Loading graph and splits …")
    with open(GRAPH_PKL, "rb") as f:
        obj = pickle.load(f)
    with open(SPLITS_PKL, "rb") as f:
        splits = pickle.load(f)
    graph = obj["hetero_data"]

    if args.all_models:
        seeds   = [42, 1, 2, 3, 4]
        models  = list(HPARAMS.keys())
        all_res = {}
        for mname in models:
            all_res[mname] = {"test": [], "zs_within": [], "zs_all": []}
            for s in seeds:
                tm, zw, za = train_model(mname, s, graph, splits)
                all_res[mname]["test"].append(tm)
                all_res[mname]["zs_within"].append(zw)
                all_res[mname]["zs_all"].append(za)
        out = ROOT / "results/zeroshot/all_results.json"
        with open(out, "w") as f:
            json.dump(all_res, f, indent=2)
        print(f"\nAll results → {out}")
    else:
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
        for s in seeds:
            train_model(args.model, s, graph, splits)


if __name__ == "__main__":
    main()
