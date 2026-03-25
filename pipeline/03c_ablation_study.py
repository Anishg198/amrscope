"""
Step 03c: Ablation study for BioMolAMR paper.

Three ablations of FeatureMLPBaseline to isolate the source of zero-shot
predictive power:

  fmlp_gene_zero  — zero out ESM-2 gene features; drug fingerprints only
  fmlp_drug_zero  — zero out drug fingerprints; ESM-2 gene features only
  fmlp_leaky      — use original 154-dim leaky gene features (drug-class
                    membership encoded in gene vector → inflates ZS metrics,
                    reproducing the prior-work leakage)

5 seeds each.  Uses the same evaluation logic as 03b (ZS-within + ZS-all).
Output: results/biomolamr/models/fmlp_{variant}_seed{seed}.pt
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.baselines import FeatureMLPBaseline

# ── Paths ─────────────────────────────────────────────────────────────────────
GRAPH_NEW  = ROOT / "data/processed/biomolamr_graph.pkl"
GRAPH_ORIG = ROOT / "data/processed/enhanced_graph.pkl"
SPLITS     = ROOT / "data/processed/extended_splits.pkl"
MODELS_DIR = ROOT / "results/biomolamr/models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 1, 2, 3, 4]

ABLATIONS = {
    "fmlp_gene_zero": {
        "graph": GRAPH_NEW,
        "zero_gene": True, "zero_drug": False,
        "desc": "Drug fingerprints only (ESM-2 zeroed)",
    },
    "fmlp_drug_zero": {
        "graph": GRAPH_NEW,
        "zero_gene": False, "zero_drug": True,
        "desc": "ESM-2 gene features only (drug FP zeroed)",
    },
    "fmlp_leaky": {
        "graph": GRAPH_ORIG,   # 154-dim leaky gene features
        "zero_gene": False, "zero_drug": False,
        "desc": "Original leaky gene features (drug-class membership in gene vector)",
    },
}

HP = {
    "hidden_dim": 256, "dropout": 0.3,
    "lr": 1e-3, "weight_decay": 1e-4,
    "epochs": 300, "patience": 5, "batch_size": 100000,
    "n_neg_train": 5,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_mrr_hits(pos_scores, neg_mat):
    ps    = pos_scores.unsqueeze(1)
    ranks = (neg_mat >= ps).sum(dim=1).float() + 1.0
    mrr   = (1.0 / ranks).mean().item()
    h1    = (ranks <= 1).float().mean().item()
    h3    = (ranks <= 3).float().mean().item()
    h10   = (ranks <= 10).float().mean().item()
    return {"mrr": mrr, "hits@1": h1, "hits@3": h3, "hits@10": h10}


def evaluate_split(model, gene_x, drug_x, split, device, n_neg=5):
    """Evaluate a split dict {pos_src, pos_dst, neg_src, neg_dst}."""
    model.eval()
    pos_src = torch.tensor(split["pos_src"], device=device)
    pos_dst = torch.tensor(split["pos_dst"], device=device)
    neg_src = torch.tensor(split["neg_src"], device=device)
    neg_dst = torch.tensor(split["neg_dst"], device=device)

    with torch.no_grad():
        pos_s = model(gene_x, drug_x, pos_src, pos_dst)
        neg_s = model(gene_x, drug_x, neg_src, neg_dst)

    n_pos   = len(pos_src)
    n_neg_s = len(neg_src) // n_pos
    neg_mat = neg_s[:n_pos * n_neg_s].view(n_pos, n_neg_s)
    return compute_mrr_hits(pos_s, neg_mat)


def evaluate_zs_all(model, gene_x, drug_x, splits, device):
    """All-class ZS evaluation (primary metric)."""
    zs_split = splits["zeroshot"]
    pos_src  = torch.tensor(zs_split["pos_src"], device=device)
    pos_dst  = torch.tensor(zs_split["pos_dst"], device=device)
    neg_src  = torch.tensor(zs_split["neg_src_all"], device=device)
    neg_dst  = torch.tensor(zs_split["neg_dst_all"], device=device)

    model.eval()
    with torch.no_grad():
        pos_s = model(gene_x, drug_x, pos_src, pos_dst)
        neg_s = model(gene_x, drug_x, neg_src, neg_dst)

    n_pos   = len(pos_src)
    n_neg_s = len(neg_src) // n_pos
    neg_mat = neg_s[:n_pos * n_neg_s].view(n_pos, n_neg_s)
    return compute_mrr_hits(pos_s, neg_mat)


# ── Training ──────────────────────────────────────────────────────────────────

def run_ablation(ablation_name: str, seed: int):
    cfg = ABLATIONS[ablation_name]
    print(f"\n[{ablation_name}] seed={seed}  ({cfg['desc']})")

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()
    print(f"  device={device}")

    # Load graph
    with open(cfg["graph"], "rb") as f:
        g_obj = pickle.load(f)
    with open(SPLITS, "rb") as f:
        splits = pickle.load(f)

    graph    = g_obj["hetero_data"]
    gene_x_  = graph["gene"].x.float().to(device)
    drug_x_  = graph["drug_class"].x.float().to(device)

    # For leaky ablation, also load drug fingerprints from new graph
    # (original graph uses 97-dim one-hot drug features; replace with real FP)
    if ablation_name == "fmlp_leaky":
        with open(GRAPH_NEW, "rb") as f:
            g_new = pickle.load(f)
        drug_x_ = g_new["hetero_data"]["drug_class"].x.float().to(device)

    # Apply feature zeroing
    if cfg.get("zero_gene"):
        gene_x_ = torch.zeros_like(gene_x_)
    if cfg.get("zero_drug"):
        drug_x_ = torch.zeros_like(drug_x_)

    gene_feat_dim = gene_x_.shape[1]
    drug_feat_dim = drug_x_.shape[1]

    model = FeatureMLPBaseline(
        gene_feat_dim=gene_feat_dim,
        drug_feat_dim=drug_feat_dim,
        hidden_dim=HP["hidden_dim"],
        dropout=HP["dropout"],
    ).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = Adam(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=HP["epochs"])

    # Prepare training data
    train_split = splits["train"]
    val_split   = splits["val"]
    zs_split    = splits["zeroshot"]

    pos_src_all = np.array(train_split["pos_src"])
    pos_dst_all = np.array(train_split["pos_dst"])
    neg_src_all = np.array(train_split["neg_src"])
    neg_dst_all = np.array(train_split["neg_dst"])
    n_pos       = len(pos_src_all)
    n_neg_pp    = HP["n_neg_train"]

    best_val_mrr = 0.0
    best_state   = None
    patience_cnt = 0
    history      = []
    import time; t0 = time.time()

    for epoch in range(1, HP["epochs"] + 1):
        model.train()
        # Shuffle
        perm     = np.random.permutation(n_pos)
        ps_shuf  = pos_src_all[perm]
        pd_shuf  = pos_dst_all[perm]
        neg_perm = np.random.permutation(len(neg_src_all))
        ns_all   = neg_src_all[neg_perm]
        nd_all   = neg_dst_all[neg_perm]

        batch_size = min(HP["batch_size"], n_pos)
        total_loss = 0.0
        n_batches  = 0

        for start in range(0, n_pos, batch_size):
            bi     = perm[start : start + batch_size]
            ps_t   = torch.tensor(ps_shuf[start:start+len(bi)],  device=device)
            pd_t   = torch.tensor(pd_shuf[start:start+len(bi)],  device=device)

            ns_start = (start * n_neg_pp) % len(ns_all)
            ns_end   = ns_start + len(bi) * n_neg_pp
            if ns_end <= len(ns_all):
                ns_idx = np.arange(ns_start, ns_end)
            else:
                ns_idx = np.concatenate([
                    np.arange(ns_start, len(ns_all)),
                    np.arange(0, ns_end - len(ns_all))
                ])
            ns_t = torch.tensor(ns_all[ns_idx], device=device)
            nd_t = torch.tensor(nd_all[ns_idx], device=device)

            pos_s = model(gene_x_, drug_x_, ps_t, pd_t)
            neg_s = model(gene_x_, drug_x_, ns_t, nd_t).view(len(bi), n_neg_pp)

            # Margin ranking loss
            loss = F.margin_ranking_loss(
                pos_s.unsqueeze(1).expand_as(neg_s).reshape(-1),
                neg_s.reshape(-1),
                torch.ones(len(bi) * n_neg_pp, device=device),
                margin=1.0,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        if epoch % 5 == 0:
            val_m = evaluate_split(model, gene_x_, drug_x_, val_split, device)
            val_mrr = val_m["mrr"]
            elapsed = int(time.time() - t0)
            print(f"  Epoch {epoch:4d}/{HP['epochs']}  "
                  f"loss={avg_loss:.4f}  val_MRR={val_mrr:.4f}  ({elapsed}s)")

            history.append({"epoch": epoch, "loss": avg_loss, "val_mrr": val_mrr})

            if val_mrr > best_val_mrr:
                best_val_mrr = val_mrr
                best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= HP["patience"]:
                    print(f"  Early stopping at epoch {epoch}")
                    break

    # Reload best
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    # Evaluate
    test_m   = evaluate_split(model, gene_x_, drug_x_, splits["test"],  device)
    zs_win_m = evaluate_split(model, gene_x_, drug_x_, splits["zeroshot"], device, n_neg=12)
    zs_all_m = evaluate_zs_all(model, gene_x_, drug_x_, splits, device)

    print(f"\n  [RESULTS] {ablation_name} seed={seed}")
    print(f"    Test        MRR={test_m['mrr']:.4f}  H@1={test_m['hits@1']:.4f}  "
          f"H@10={test_m['hits@10']:.4f}")
    print(f"    ZS-within   MRR={zs_win_m['mrr']:.4f}  H@10={zs_win_m['hits@10']:.4f}  "
          f"[12-way ranking]")
    print(f"    ZS-all      MRR={zs_all_m['mrr']:.4f}    H@10={zs_all_m['hits@10']:.4f}  "
          f"[all-class, PRIMARY METRIC]")

    ckpt_path = MODELS_DIR / f"{ablation_name}_seed{seed}.pt"
    torch.save({
        "model_name":        ablation_name,
        "seed":              seed,
        "state_dict":        best_state,
        "test_metrics":      test_m,
        "zs_within_metrics": zs_win_m,
        "zs_all_metrics":    zs_all_m,
        "history":           history,
        "hparams":           HP,
        "ablation_cfg":      {k: str(v) for k, v in cfg.items()},
        "gene_feat_dim":     gene_feat_dim,
        "drug_feat_dim":     drug_feat_dim,
    }, ckpt_path)
    print(f"  Saved → {ckpt_path}")

    return test_m, zs_win_m, zs_all_m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", type=str, default=None,
                        choices=list(ABLATIONS.keys()))
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--all",   action="store_true")
    args = parser.parse_args()

    if args.all:
        for name in ABLATIONS:
            for seed in SEEDS:
                run_ablation(name, seed)
    elif args.ablation:
        run_ablation(args.ablation, args.seed)
    else:
        print("Use --all or --ablation <name> --seed <seed>")
        print("Ablations:", list(ABLATIONS.keys()))


if __name__ == "__main__":
    main()
