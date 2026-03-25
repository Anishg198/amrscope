"""
Step 03b: Train BioMolAMR and all baselines on the BioMolAMR graph.

Models trained:
  biomolamr   — full proposed model (ESM-2 + mol FP + mechanism-weighted decoder)
  rgcn_bio    — R-GCN baseline on new graph (same architecture as current best)
  distmult    — transductive KGE baseline (cannot zero-shot)
  transe      — transductive KGE baseline (cannot zero-shot)
  feature_mlp — feature-only baseline (no graph)

Evaluation:
  test        — standard split (seen drug classes), all-class ranked
  zs_within   — within-ZS ranking (rank among 12 ZS classes)  ← narrow task
  zs_all      — all-class ranking (rank among all 46 classes)  ← PRIMARY METRIC

Usage:
  python pipeline/03b_train_biomolamr.py --model biomolamr --seed 42
  python pipeline/03b_train_biomolamr.py --all_models
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

from src.models.biomolamr import BioMolAMR
from src.models.baselines  import DistMultBaseline, TransEBaseline, FeatureMLPBaseline, RGCNBaseline
from src.training.losses   import ListNetLoss

# ── Paths ─────────────────────────────────────────────────────────────────────
GRAPH_NEW  = ROOT / "data/processed/biomolamr_graph.pkl"
GRAPH_ORIG = ROOT / "data/processed/enhanced_graph.pkl"
SPLITS_EXT = ROOT / "data/processed/extended_splits.pkl"
SPLITS_STD = ROOT / "data/processed/splits.pkl"
MODELS_DIR = ROOT / "results/biomolamr/models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
HPARAMS = {
    "biomolamr": {
        "hidden_dim": 256, "out_dim": 128, "num_heads": 4,
        "num_gat_layers": 2, "dropout": 0.3,
        "lr": 1e-3, "weight_decay": 1e-4,
        "epochs": 300, "patience": 5, "batch_size": 100000,  # full-batch: 1 fwd/epoch
        "lambda_struct_drug": 0.5, "lambda_struct_gene": 0.3,
        "lambda_mech": 0.2, "lambda_proto": 0.1,
        "listnet_tau": 0.1, "warmup_epochs": 20,
        "n_neg_train": 5,    # K=5 negatives per positive (matches splits sampling)
    },
    "rgcn_bio": {
        "hidden_dim": 256, "out_dim": 128, "dropout": 0.3,
        "lr": 1e-3, "weight_decay": 1e-4,
        "epochs": 300, "patience": 5, "batch_size": 100000,  # full-batch
        "n_neg_train": 5,
    },
    "distmult": {
        "emb_dim": 128, "dropout": 0.3,
        "lr": 5e-3, "weight_decay": 1e-5,
        "epochs": 500, "patience": 5, "batch_size": 100000,  # full-batch
        "n_neg_train": 5,
    },
    "transe": {
        "emb_dim": 128, "margin": 1.0, "dropout": 0.3,
        "lr": 1e-3, "weight_decay": 1e-5,
        "epochs": 500, "patience": 5, "batch_size": 100000,  # full-batch
        "n_neg_train": 5,
    },
    "feature_mlp": {
        "hidden_dim": 256, "dropout": 0.3,
        "lr": 1e-3, "weight_decay": 1e-4,
        "epochs": 300, "patience": 5, "batch_size": 100000,  # full-batch
        "n_neg_train": 5,
    },
}


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_mrr_hits(pos_scores: torch.Tensor, neg_mat: torch.Tensor):
    """pos_scores (N,), neg_mat (N, K) → MRR, H@1, H@3, H@10"""
    ps    = pos_scores.unsqueeze(1)
    ranks = (neg_mat >= ps).sum(dim=1).float() + 1.0
    mrr   = (1.0 / ranks).mean().item()
    h1    = (ranks <= 1).float().mean().item()
    h3    = (ranks <= 3).float().mean().item()
    h10   = (ranks <= 10).float().mean().item()
    return mrr, h1, h3, h10


def bce_loss(ps, ns):
    return (F.binary_cross_entropy_with_logits(ps, torch.ones_like(ps)) +
            F.binary_cross_entropy_with_logits(ns, torch.zeros_like(ns)))


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(model_name, graph, hp, device):
    n_genes  = graph["gene"].x.shape[0]
    n_dc     = graph["drug_class"].x.shape[0]
    gene_dim = graph["gene"].x.shape[1]
    drug_dim = graph["drug_class"].x.shape[1]
    mech_dim = graph["mechanism"].x.shape[1]

    if model_name == "biomolamr":
        return BioMolAMR(
            gene_feat_dim=gene_dim, drug_feat_dim=drug_dim, mech_feat_dim=mech_dim,
            hidden_dim=hp["hidden_dim"], out_dim=hp["out_dim"],
            num_heads=hp["num_heads"], num_gat_layers=hp["num_gat_layers"],
            dropout=hp["dropout"]
        ).to(device)
    elif model_name == "rgcn_bio":
        return RGCNBaseline(gene_dim, drug_dim, mech_dim,
                            hp["hidden_dim"], hp["out_dim"], hp["dropout"]).to(device)
    elif model_name == "distmult":
        return DistMultBaseline(n_genes, n_dc, hp["emb_dim"], hp["dropout"]).to(device)
    elif model_name == "transe":
        return TransEBaseline(n_genes, n_dc, hp["emb_dim"], hp["margin"],
                              hp["dropout"]).to(device)
    elif model_name == "feature_mlp":
        return FeatureMLPBaseline(gene_dim, drug_dim, hp["hidden_dim"], hp["dropout"]).to(device)
    raise ValueError(f"Unknown model: {model_name}")


def score_pairs(model_name, model, graph, src_t, dst_t):
    if model_name in ("biomolamr", "rgcn_bio"):
        return model(graph, src_t, dst_t)
    elif model_name in ("distmult", "transe"):
        return model(src_t, dst_t)
    elif model_name == "feature_mlp":
        return model(graph["gene"].x, graph["drug_class"].x, src_t, dst_t)
    raise ValueError(model_name)


def evaluate_split(model_name, model, graph, splits, split_name, device,
                   use_within_zs=False):
    split = splits[split_name]
    pos_src = torch.tensor(split["pos_src"], dtype=torch.long, device=device)
    pos_dst = torch.tensor(split["pos_dst"], dtype=torch.long, device=device)

    if split_name == "zeroshot" and not use_within_zs:
        neg_src_k, neg_dst_k = "neg_src_all", "neg_dst_all"
    else:
        neg_src_k, neg_dst_k = "neg_src", "neg_dst"

    neg_src = torch.tensor(split[neg_src_k], dtype=torch.long, device=device)
    neg_dst = torch.tensor(split[neg_dst_k], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        ps     = score_pairs(model_name, model, graph, pos_src, pos_dst)
        n_pos  = len(pos_src)
        n_npp  = max(1, len(neg_src) // n_pos)
        ns     = score_pairs(model_name, model, graph,
                             neg_src[:n_pos * n_npp], neg_dst[:n_pos * n_npp])
        neg_mat = ns.view(n_pos, n_npp)
        mrr, h1, h3, h10 = compute_mrr_hits(ps, neg_mat)

    return {"mrr": mrr, "hits@1": h1, "hits@3": h3, "hits@10": h10}


# ── Training loop ─────────────────────────────────────────────────────────────

def train_model(model_name: str, seed: int, graph, splits):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n[{model_name}] seed={seed}  device={device}")

    graph = graph.to(device)
    hp    = HPARAMS[model_name]
    model = build_model(model_name, graph, hp, device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Pre-compute Tanimoto matrix for structural alignment loss
    tanimoto = None
    esm2_emb_all = None
    if model_name == "biomolamr":
        tan_path = ROOT / "data/processed/drug_tanimoto.pt"
        if tan_path.exists():
            tanimoto = torch.load(tan_path, map_location=device)

    optimizer = optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp["epochs"])

    listnet_loss = ListNetLoss(
        temperature=hp.get("listnet_tau", 0.1), margin=1.0
    ) if model_name in ("biomolamr", "rgcn_bio") else None

    split  = splits["train"]
    pos_src, pos_dst = split["pos_src"], split["pos_dst"]
    neg_src, neg_dst = split["neg_src"], split["neg_dst"]
    n_pos    = len(pos_src)
    n_neg_pp = hp.get("n_neg_train", 5)

    best_val_mrr = -1.0
    best_state   = None
    patience_ctr = 0
    history      = []
    t0           = time.time()

    warmup = hp.get("warmup_epochs", 0)

    for epoch in range(1, hp["epochs"] + 1):
        model.train()
        perm       = np.random.permutation(n_pos)
        epoch_loss = 0.0; n_batches = 0

        for start in range(0, n_pos, hp["batch_size"]):
            bi   = perm[start : start + hp["batch_size"]]
            ps_t = torch.tensor(pos_src[bi], dtype=torch.long, device=device)
            pd_t = torch.tensor(pos_dst[bi], dtype=torch.long, device=device)
            # Wrap negatives cyclically to handle last batch overflow
            ns_start = (start * n_neg_pp) % len(neg_src)
            ns_end   = ns_start + len(bi) * n_neg_pp
            if ns_end <= len(neg_src):
                ns_idx = np.arange(ns_start, ns_end)
            else:
                ns_idx = np.concatenate([
                    np.arange(ns_start, len(neg_src)),
                    np.arange(0, ns_end - len(neg_src))
                ])
            ns_t = torch.tensor(neg_src[ns_idx], dtype=torch.long, device=device)
            nd_t = torch.tensor(neg_dst[ns_idx], dtype=torch.long, device=device)

            optimizer.zero_grad()
            ps_s = score_pairs(model_name, model, graph, ps_t, pd_t)

            if listnet_loss is not None and model_name in ("biomolamr", "rgcn_bio"):
                # Use ListNet for proposed and strong baseline
                ns_s  = score_pairs(model_name, model, graph, ns_t, nd_t)
                n_per = max(1, len(ns_t) // len(ps_t))
                neg_m = ns_s[:len(ps_t) * n_per].view(len(ps_t), n_per)
                loss  = listnet_loss(ps_s, neg_m)
            else:
                ns_s = score_pairs(model_name, model, graph, ns_t, nd_t)
                loss = bce_loss(ps_s, ns_s)

            # BioMolAMR extra losses
            if model_name == "biomolamr" and tanimoto is not None:
                scale = min(1.0, (epoch - warmup) / 50) if epoch > warmup else 0.0
                if scale > 0:
                    loss += scale * hp["lambda_struct_drug"] * \
                            model.drug_class_contrastive_loss(graph, tanimoto)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item(); n_batches += 1

        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            val_m   = evaluate_split(model_name, model, graph, splits, "val", device)
            val_mrr = val_m["mrr"]
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:4d}/{hp['epochs']}  "
                  f"loss={epoch_loss/max(n_batches,1):.4f}  "
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
        model.load_state_dict(best_state)
        model.to(device)

    test_m    = evaluate_split(model_name, model, graph, splits, "test",     device)
    zs_within = evaluate_split(model_name, model, graph, splits, "zeroshot", device,
                                use_within_zs=True)
    zs_all    = evaluate_split(model_name, model, graph, splits, "zeroshot", device,
                                use_within_zs=False)

    n_zs = splits["meta"]["n_zs_classes"]
    print(f"\n  [RESULTS] {model_name} seed={seed}")
    print(f"    Test        MRR={test_m['mrr']:.4f}  H@1={test_m['hits@1']:.4f}  "
          f"H@3={test_m['hits@3']:.4f}  H@10={test_m['hits@10']:.4f}")
    print(f"    ZS-within   MRR={zs_within['mrr']:.4f}  H@10={zs_within['hits@10']:.4f}  "
          f"[{n_zs}-way ranking]")
    print(f"    ZS-all      MRR={zs_all['mrr']:.4f}    H@10={zs_all['hits@10']:.4f}  "
          f"[all-class, PRIMARY METRIC]")

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
    parser.add_argument("--model",  type=str, default="biomolamr",
                        choices=list(HPARAMS.keys()))
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--seeds",  type=str, default=None)
    parser.add_argument("--all_models", action="store_true")
    args = parser.parse_args()

    print("Loading graph and splits …")
    graph_path  = GRAPH_NEW  if GRAPH_NEW.exists()  else GRAPH_ORIG
    splits_path = SPLITS_EXT if SPLITS_EXT.exists() else SPLITS_STD
    print(f"  Graph:  {graph_path.name}")
    print(f"  Splits: {splits_path.name}")

    with open(graph_path,  "rb") as f:  obj    = pickle.load(f)
    with open(splits_path, "rb") as f:  splits = pickle.load(f)
    graph = obj["hetero_data"]

    if args.all_models:
        seeds  = [42, 1, 2, 3, 4]
        models = list(HPARAMS.keys())
        all_res = {}
        for mname in models:
            all_res[mname] = {"test": [], "zs_within": [], "zs_all": []}
            for s in seeds:
                tm, zw, za = train_model(mname, s, graph, splits)
                all_res[mname]["test"].append(tm)
                all_res[mname]["zs_within"].append(zw)
                all_res[mname]["zs_all"].append(za)
        out = ROOT / "results/biomolamr/all_results.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(all_res, f, indent=2)
        print(f"\nAll results → {out}")
    else:
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
        for s in seeds:
            train_model(args.model, s, graph, splits)


if __name__ == "__main__":
    main()
