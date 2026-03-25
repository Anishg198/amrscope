"""
Step 03d: Train CrossContrastAMR — cross-attention + structural alignment for zero-shot AMR.

Architecture:
  Gene encoder:  ESM-2(480) → MLP(256→128) → L2-norm
  Drug encoder:  MolFP(3245) → MLP(512→128) → L2-norm
  Cross-attention: gene queries over all 46 drug embeddings (4 heads)
  g_final = LayerNorm(g_base + g_ctx) → L2-norm

Zero-shot transfer mechanism:
  The drug structural alignment loss forces the learned drug embedding space to
  preserve Tanimoto fingerprint similarity.  This means:
    - ZS drug with similar fingerprint to training drug → similar embedding
    - Gene that attends to training drug representation will also attend to ZS drug
  This is the key difference from Feature-MLP: explicit metric-space enforcement.

Loss: α_rank * ListNetLoss(K=5)
    + α_struct * DrugStructuralAlignment(drug_emb, Tanimoto_matrix)

Primary metric: ZS-all MRR (rank ZS drug among all 46 classes, random = 0.022)
Baseline to beat: Feature-MLP  ZS-all MRR = 0.069 (3.1× random)
Target           : CrossContrastAMR ZS-all MRR > 0.080 (3.6× random)

Usage:
  python pipeline/03d_train_crossamr.py --seed 42
  python pipeline/03d_train_crossamr.py --seeds 42,1,2,3,4
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

from src.models.crossamr    import CrossContrastAMR
from src.training.losses    import ListNetLoss, StructuralAlignmentLoss

# ── Paths ─────────────────────────────────────────────────────────────────────
GRAPH_PATH  = ROOT / "data/processed/biomolamr_graph.pkl"
SPLITS_PATH = ROOT / "data/processed/extended_splits.pkl"
MODELS_DIR  = ROOT / "results/biomolamr/models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
HP = {
    "hidden_dim":    128,
    "n_heads":       4,
    "dropout":       0.3,
    "lr":            1e-3,
    "weight_decay":  1e-4,
    "epochs":        300,
    "patience":      10,       # patience in eval checks (each check = 5 epochs = 50 ep gap)
    "batch_size":    256,      # positive pairs per batch
    "n_neg_train":   5,        # K negatives per positive (from pre-computed splits)
    "alpha_rank":    0.5,      # weight for ListNet ranking loss
    "alpha_struct":  0.5,      # weight for drug structural alignment loss
    "listnet_tau":   0.1,      # ListNet temperature
    "listnet_margin": 1.0,
    "warmup_epochs": 20,       # ramp in structural alignment after warmup
}


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_mrr_hits(pos_scores: torch.Tensor, neg_mat: torch.Tensor):
    """pos_scores (N,), neg_mat (N, K) → MRR, H@1, H@3, H@10"""
    ps    = pos_scores.unsqueeze(1)
    ranks = (neg_mat >= ps).sum(dim=1).float() + 1.0
    return {
        "mrr":     (1.0 / ranks).mean().item(),
        "hits@1":  (ranks <= 1).float().mean().item(),
        "hits@3":  (ranks <= 3).float().mean().item(),
        "hits@10": (ranks <= 10).float().mean().item(),
    }


def evaluate_split(model, graph, splits, split_name, device, use_within_zs=False):
    split   = splits[split_name]
    pos_src = torch.tensor(split["pos_src"], dtype=torch.long, device=device)
    pos_dst = torch.tensor(split["pos_dst"], dtype=torch.long, device=device)

    if split_name == "zeroshot" and not use_within_zs:
        neg_src_k, neg_dst_k = "neg_src_all", "neg_dst_all"
    else:
        neg_src_k, neg_dst_k = "neg_src", "neg_dst"

    neg_src = torch.tensor(split[neg_src_k], dtype=torch.long, device=device)
    neg_dst = torch.tensor(split[neg_dst_k], dtype=torch.long, device=device)

    gene_x = graph["gene"].x
    drug_x = graph["drug_class"].x

    model.eval()
    with torch.no_grad():
        ps     = model(gene_x, drug_x, pos_src, pos_dst)
        n_pos  = len(pos_src)
        n_npp  = max(1, len(neg_src) // n_pos)
        ns     = model(gene_x, drug_x,
                       neg_src[:n_pos * n_npp],
                       neg_dst[:n_pos * n_npp])
        neg_mat = ns.view(n_pos, n_npp)

    return compute_mrr_hits(ps, neg_mat)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_crossamr(seed: int, graph_obj, splits):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n[CrossContrastAMR] seed={seed}  device={device}")

    graph    = graph_obj["hetero_data"].to(device)
    gene_x   = graph["gene"].x          # [N_g, 480]
    drug_x   = graph["drug_class"].x    # [46, 3245]
    gene_dim = gene_x.shape[1]
    drug_dim = drug_x.shape[1]

    model = CrossContrastAMR(
        gene_feat_dim=gene_dim,
        drug_feat_dim=drug_dim,
        hidden_dim=HP["hidden_dim"],
        n_heads=HP["n_heads"],
        dropout=HP["dropout"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(),
                            lr=HP["lr"], weight_decay=HP["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=HP["epochs"])

    rank_loss   = ListNetLoss(temperature=HP["listnet_tau"], margin=HP["listnet_margin"])
    struct_loss = StructuralAlignmentLoss()

    # Load Tanimoto fingerprint similarity matrix for drug structural alignment
    tanimoto = None
    tan_path = ROOT / "data/processed/drug_tanimoto.pt"
    if tan_path.exists():
        tanimoto = torch.load(tan_path, map_location=device)
        print(f"  Loaded Tanimoto matrix: {tanimoto.shape}")
    else:
        print("  WARNING: drug_tanimoto.pt not found — structural alignment disabled")

    split     = splits["train"]
    pos_src   = np.array(split["pos_src"])
    pos_dst   = np.array(split["pos_dst"])
    neg_src   = np.array(split["neg_src"])
    neg_dst   = np.array(split["neg_dst"])
    n_pos     = len(pos_src)
    K         = HP["n_neg_train"]
    BS        = HP["batch_size"]

    best_val_mrr = -1.0
    best_state   = None
    patience_ctr = 0
    history      = []
    t0           = time.time()

    for epoch in range(1, HP["epochs"] + 1):
        model.train()
        perm       = np.random.permutation(n_pos)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n_pos, BS):
            bi   = perm[start : start + BS]
            B    = len(bi)

            ps_t = torch.tensor(pos_src[bi], dtype=torch.long, device=device)
            pd_t = torch.tensor(pos_dst[bi], dtype=torch.long, device=device)

            # Negatives for ListNet (K per positive, cyclic)
            ns_start = (start * K) % len(neg_src)
            ns_end   = ns_start + B * K
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

            # ── Encode drugs once per batch ───────────────────────────────────
            all_d = model.encode_drugs(drug_x)              # [46, H]

            # ── Encode positive genes with drug context ───────────────────────
            g_pos = model.encode_genes_with_context(gene_x[ps_t], all_d)  # [B, H]
            d_pos = all_d[pd_t]                                            # [B, H]
            pos_scores = (g_pos * d_pos).sum(dim=-1)                       # [B]

            # ── ListNet: gene vs K explicit negatives ─────────────────────────
            g_neg = model.encode_genes_with_context(gene_x[ns_t], all_d)  # [B*K, H]
            d_neg = all_d[nd_t]                                            # [B*K, H]
            neg_scores = (g_neg * d_neg).sum(dim=-1)                       # [B*K]
            neg_mat    = neg_scores.view(B, K)                             # [B, K]
            l_rank     = rank_loss(pos_scores, neg_mat)

            # ── Drug structural alignment: emb cosine sim ≈ Tanimoto sim ─────
            # Ramp in after warmup to let encoders initialise first
            l_struct = torch.tensor(0.0, device=device)
            if tanimoto is not None and epoch > HP["warmup_epochs"]:
                scale = min(1.0, (epoch - HP["warmup_epochs"]) / 30)
                l_struct = scale * struct_loss(all_d, tanimoto)

            loss = HP["alpha_rank"] * l_rank + HP["alpha_struct"] * l_struct

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        # ── Evaluation ────────────────────────────────────────────────────────
        if epoch % 5 == 0 or epoch == 1:
            val_m   = evaluate_split(model, graph, splits, "val", device)
            val_mrr = val_m["mrr"]
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:4d}/{HP['epochs']}  "
                  f"loss={epoch_loss/max(n_batches,1):.4f}  "
                  f"val_MRR={val_mrr:.4f}  H@1={val_m['hits@1']:.4f}  "
                  f"H@10={val_m['hits@10']:.4f}  ({elapsed:.0f}s)")
            history.append({
                "epoch": epoch,
                "loss":  epoch_loss / max(n_batches, 1),
                **val_m,
            })

            if val_mrr > best_val_mrr + 1e-4:
                best_val_mrr = val_mrr
                best_state   = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= HP["patience"]:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(best val MRR={best_val_mrr:.4f})")
                    break

    # ── Load best checkpoint and evaluate ────────────────────────────────────
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

    test_m    = evaluate_split(model, graph, splits, "test",     device)
    zs_within = evaluate_split(model, graph, splits, "zeroshot", device, use_within_zs=True)
    zs_all    = evaluate_split(model, graph, splits, "zeroshot", device, use_within_zs=False)

    n_zs = splits["meta"]["n_zs_classes"]
    print(f"\n  ── RESULTS  CrossContrastAMR  seed={seed} ──────────────────────")
    print(f"    Test      MRR={test_m['mrr']:.4f}  "
          f"H@1={test_m['hits@1']:.4f}  H@10={test_m['hits@10']:.4f}")
    print(f"    ZS-within MRR={zs_within['mrr']:.4f}  "
          f"H@10={zs_within['hits@10']:.4f}  [{n_zs}-way]")
    print(f"    ZS-all    MRR={zs_all['mrr']:.4f}  "
          f"H@10={zs_all['hits@10']:.4f}  [all-class PRIMARY]")
    print(f"    Feature-MLP baseline: ZS-all MRR=0.069  (target >0.080)")

    ckpt_path = MODELS_DIR / f"crossamr_seed{seed}.pt"
    torch.save({
        "model_name":       "crossamr",
        "seed":             seed,
        "state_dict":       best_state,
        "test_metrics":     test_m,
        "zs_within_metrics": zs_within,
        "zs_all_metrics":   zs_all,
        "history":          history,
        "hparams":          HP,
    }, ckpt_path)
    print(f"  Saved → {ckpt_path}")

    return test_m, zs_within, zs_all


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds, e.g. 42,1,2,3,4")
    args = parser.parse_args()

    print("Loading graph and splits …")
    with open(GRAPH_PATH,  "rb") as f:
        graph_obj = pickle.load(f)
    with open(SPLITS_PATH, "rb") as f:
        splits = pickle.load(f)

    gene_dim = graph_obj["hetero_data"]["gene"].x.shape[1]
    drug_dim = graph_obj["hetero_data"]["drug_class"].x.shape[1]
    n_genes  = graph_obj["hetero_data"]["gene"].x.shape[0]
    n_dc     = graph_obj["hetero_data"]["drug_class"].x.shape[0]
    n_train  = len(splits["train"]["pos_src"])
    print(f"  Genes: {n_genes}  Drug classes: {n_dc}")
    print(f"  Gene dim: {gene_dim}  Drug dim: {drug_dim}")
    print(f"  Training positives: {n_train}")

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    results = []
    for s in seeds:
        tm, zw, za = train_crossamr(s, graph_obj, splits)
        results.append({"seed": s, "test": tm, "zs_within": zw, "zs_all": za})

    if len(results) > 1:
        zs_mrrs = [r["zs_all"]["mrr"] for r in results]
        print(f"\n── Summary ({len(results)} seeds) ──────────────────────────")
        print(f"  ZS-all MRR: mean={np.mean(zs_mrrs):.4f} ± {np.std(zs_mrrs):.4f}")
        print(f"  Range: [{min(zs_mrrs):.4f}, {max(zs_mrrs):.4f}]")
        print(f"  Feature-MLP baseline: 0.069 ± 0.006")

    return results


if __name__ == "__main__":
    main()
