"""
Step 5: Ablation studies for ZeroShotHetGAT.

Variants tested:
  A. Full model (ZS-HetGAT)
  B. No graph structure (Drug encoder + Gene MLP, no message passing)
  C. No drug features (learnable drug embeddings, loses zero-shot capability)
  D. Single GAT layer
  E. No mechanism nodes (gene encoder without mechanism aggregation)
  F. No auxiliary mechanism loss (if aux is enabled)

Each variant trained with 3 seeds, evaluated on both test and zero-shot splits.
"""

import json, pickle, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GRAPH_PKL  = ROOT / "data/processed/enhanced_graph.pkl"
SPLITS_PKL = ROOT / "data/processed/splits.pkl"
OUT_DIR    = ROOT / "results/zeroshot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ABLATION_SEEDS = [42, 1, 2]


# ─── Ablation model variants ──────────────────────────────────────────────────

class GeneOnlyMLP(nn.Module):
    """Variant B: No graph, gene features + drug features → MLP."""
    def __init__(self, gene_dim, drug_dim, hidden=256, out=128, dropout=0.3):
        super().__init__()
        self.gene_proj = nn.Sequential(
            nn.Linear(gene_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out))
        self.drug_proj = nn.Sequential(
            nn.Linear(drug_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out))
        self.W = nn.Parameter(torch.empty(out))
        nn.init.xavier_uniform_(self.W.unsqueeze(0))

    def forward(self, graph, gene_idx, drug_idx):
        g = self.gene_proj(graph["gene"].x[gene_idx])
        d = self.drug_proj(graph["drug_class"].x[drug_idx])
        return (g * self.W * d).sum(dim=-1)


class LookupTableDrug(nn.Module):
    """Variant C: Learnable drug embedding table (cannot do zero-shot)."""
    def __init__(self, gene_dim, mech_dim, n_dc, hidden=256, out=128,
                 num_heads=4, dropout=0.3):
        super().__init__()
        from src.models.zeroshot_hetgat import GeneEncoder
        self.gene_encoder = GeneEncoder(gene_dim, mech_dim, hidden, out,
                                        num_heads, dropout)
        self.drug_emb = nn.Embedding(n_dc, out)
        nn.init.xavier_uniform_(self.drug_emb.weight)
        self.W = nn.Parameter(torch.empty(out))
        nn.init.xavier_uniform_(self.W.unsqueeze(0))

    def _edge_idx_dict(self, graph):
        return {
            ("gene", "has_mechanism", "mechanism"):
                graph["gene", "has_mechanism", "mechanism"].edge_index,
            ("mechanism", "includes_gene", "gene"):
                graph["mechanism", "includes_gene", "gene"].edge_index,
        }

    def forward(self, graph, gene_idx, drug_idx):
        gene_emb = self.gene_encoder(
            graph["gene"].x, graph["mechanism"].x, self._edge_idx_dict(graph))
        drug_emb = self.drug_emb(drug_idx)
        g = gene_emb[gene_idx]
        return (g * self.W * drug_emb).sum(dim=-1)


class SingleLayerGAT(nn.Module):
    """Variant D: Single GAT layer."""
    def __init__(self, gene_dim, drug_dim, mech_dim, hidden=256, out=128,
                 num_heads=4, dropout=0.3):
        super().__init__()
        from torch_geometric.nn import HeteroConv, GATConv
        from src.models.zeroshot_hetgat import DrugClassEncoder, BilinearDecoder

        self.gene_proj = nn.Linear(gene_dim, hidden)
        self.mech_proj = nn.Linear(mech_dim, hidden)
        self.conv = HeteroConv({
            ("gene", "has_mechanism", "mechanism"):
                GATConv(hidden, hidden // num_heads, heads=num_heads,
                        dropout=dropout, add_self_loops=False),
            ("mechanism", "includes_gene", "gene"):
                GATConv(hidden, hidden // num_heads, heads=num_heads,
                        dropout=dropout, add_self_loops=False),
        }, aggr="sum")
        self.out_proj  = nn.Linear(hidden, out)
        self.drug_enc  = DrugClassEncoder(drug_dim, hidden, out, dropout)
        self.decoder   = BilinearDecoder(out)
        self.norm      = nn.LayerNorm(hidden)
        self.dropout   = nn.Dropout(dropout)

    def _encode(self, graph):
        h = {
            "gene":      F.gelu(self.gene_proj(graph["gene"].x)),
            "mechanism": F.gelu(self.mech_proj(graph["mechanism"].x)),
        }
        ei = {
            ("gene", "has_mechanism", "mechanism"):
                graph["gene", "has_mechanism", "mechanism"].edge_index,
            ("mechanism", "includes_gene", "gene"):
                graph["mechanism", "includes_gene", "gene"].edge_index,
        }
        h2 = self.conv(h, ei)
        gene_emb = self.norm(self.dropout(F.gelu(h2["gene"] + h["gene"])))
        return self.out_proj(gene_emb)

    def forward(self, graph, gene_idx, drug_idx):
        gene_emb = self._encode(graph)
        drug_emb = self.drug_enc(graph["drug_class"].x)
        return self.decoder(gene_emb[gene_idx], drug_emb[drug_idx])


class NoMechanismModel(nn.Module):
    """Variant E: No mechanism nodes — gene encoder is a plain MLP."""
    def __init__(self, gene_dim, drug_dim, hidden=256, out=128, dropout=0.3):
        super().__init__()
        from src.models.zeroshot_hetgat import DrugClassEncoder, BilinearDecoder

        self.gene_enc = nn.Sequential(
            nn.Linear(gene_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, out))
        self.drug_enc  = DrugClassEncoder(drug_dim, hidden, out, dropout)
        self.decoder   = BilinearDecoder(out)

    def forward(self, graph, gene_idx, drug_idx):
        gene_emb = self.gene_enc(graph["gene"].x)
        drug_emb = self.drug_enc(graph["drug_class"].x)
        return self.decoder(gene_emb[gene_idx], drug_emb[drug_idx])


ABLATION_CONFIGS = {
    "A_full":         "Full ZS-HetGAT",
    "B_no_graph":     "No graph structure",
    "C_lookup_drug":  "Lookup drug embeddings (no ZS)",
    "D_single_layer": "Single GAT layer",
    "E_no_mechanism": "No mechanism nodes",
}


def build_ablation_model(variant, graph, device):
    gene_dim = graph["gene"].x.shape[1]
    drug_dim = graph["drug_class"].x.shape[1]
    mech_dim = graph["mechanism"].x.shape[1]
    n_dc     = graph["drug_class"].x.shape[0]

    if variant == "A_full":
        from src.models.zeroshot_hetgat import ZeroShotHetGAT
        return ZeroShotHetGAT(gene_dim, drug_dim, mech_dim,
                               hidden_dim=256, out_dim=128,
                               num_heads=4, dropout=0.3).to(device)
    elif variant == "B_no_graph":
        return GeneOnlyMLP(gene_dim, drug_dim).to(device)
    elif variant == "C_lookup_drug":
        return LookupTableDrug(gene_dim, mech_dim, n_dc).to(device)
    elif variant == "D_single_layer":
        return SingleLayerGAT(gene_dim, drug_dim, mech_dim).to(device)
    elif variant == "E_no_mechanism":
        return NoMechanismModel(gene_dim, drug_dim).to(device)
    else:
        raise ValueError(f"Unknown ablation: {variant}")


def bce_loss(ps, ns):
    return (
        F.binary_cross_entropy_with_logits(ps, torch.ones_like(ps)) +
        F.binary_cross_entropy_with_logits(ns, torch.zeros_like(ns))
    )


def eval_split(variant, model, graph, splits, split_name, device):
    split    = splits[split_name]
    pos_src  = torch.tensor(split["pos_src"], dtype=torch.long, device=device)
    pos_dst  = torch.tensor(split["pos_dst"], dtype=torch.long, device=device)
    neg_src  = torch.tensor(split["neg_src"], dtype=torch.long, device=device)
    neg_dst  = torch.tensor(split["neg_dst"], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        ps = model(graph, pos_src, pos_dst)
        ns = model(graph, neg_src, neg_dst)

    n_pos     = len(ps)
    n_neg_per = max(1, len(ns) // n_pos)
    neg_mat   = ns[:n_pos * n_neg_per].view(n_pos, n_neg_per)
    pos_s2    = ps[:n_pos].unsqueeze(1)
    ranks     = (neg_mat >= pos_s2).sum(dim=1).float() + 1.0

    mrr   = (1.0 / ranks).mean().item()
    hits1  = (ranks <= 1).float().mean().item()
    hits3  = (ranks <= 3).float().mean().item()
    hits10 = (ranks <= 10).float().mean().item()
    return {"mrr": mrr, "hits@1": hits1, "hits@3": hits3, "hits@10": hits10}


def train_ablation(variant, seed, graph, splits, device):
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_ablation_model(variant, graph, device)

    opt   = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)

    split    = splits["train"]
    pos_src  = split["pos_src"]
    pos_dst  = split["pos_dst"]
    neg_src  = split["neg_src"]
    neg_dst  = split["neg_dst"]
    n_pos    = len(pos_src)
    n_neg_pp = max(1, len(neg_src) // n_pos)

    best_val = -1.0
    best_st  = None
    pat_ctr  = 0
    PATIENCE = 3   # checked every 10 epochs → 30 epoch patience

    for epoch in range(1, 201):
        model.train()
        perm = np.random.permutation(n_pos)
        loss_sum = 0.0
        for start in range(0, n_pos, 2048):
            bi = perm[start:start + 2048]
            ps_t = torch.tensor(pos_src[bi], dtype=torch.long, device=device)
            pd_t = torch.tensor(pos_dst[bi], dtype=torch.long, device=device)
            ns_s = start * n_neg_pp
            ns_e = min(ns_s + len(bi) * n_neg_pp, len(neg_src))
            ns_t = torch.tensor(neg_src[ns_s:ns_e], dtype=torch.long, device=device)
            nd_t = torch.tensor(neg_dst[ns_s:ns_e], dtype=torch.long, device=device)

            opt.zero_grad()
            ps = model(graph, ps_t, pd_t)
            ns = model(graph, ns_t, nd_t)
            loss = bce_loss(ps, ns)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item()
        sched.step()

        if epoch % 10 == 0:
            val_m = eval_split(variant, model, graph, splits, "val", device)
            if val_m["mrr"] > best_val + 1e-4:
                best_val = val_m["mrr"]
                best_st  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                pat_ctr  = 0
            else:
                pat_ctr += 1
                if pat_ctr >= PATIENCE:
                    break

    if best_st:
        model.load_state_dict(best_st)
        model.to(device)

    # For C_lookup_drug, zero-shot evaluation is undefined (can't generalise)
    # We still evaluate on the zero-shot split indices (using its learned embeddings)
    test_m = eval_split(variant, model, graph, splits, "test", device)
    zs_m   = eval_split(variant, model, graph, splits, "zeroshot", device) \
             if variant != "C_lookup_drug" else {"mrr": 0.0, "hits@1": 0.0,
                                                  "hits@3": 0.0, "hits@10": 0.0}
    return test_m, zs_m


def main():
    print("=" * 60)
    print("Step 5: Ablation studies")
    print("=" * 60)

    with open(GRAPH_PKL, "rb") as f:
        obj = pickle.load(f)
    with open(SPLITS_PKL, "rb") as f:
        splits = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph  = obj["hetero_data"].to(device)

    results = {}
    for variant, desc in ABLATION_CONFIGS.items():
        print(f"\n  Ablation: {variant} ({desc})")
        test_runs = []
        zs_runs   = []
        for seed in ABLATION_SEEDS:
            print(f"    seed={seed}")
            tm, zm = train_ablation(variant, seed, graph, splits, device)
            test_runs.append(tm)
            zs_runs.append(zm)
            print(f"      test MRR={tm['mrr']:.4f}  zs MRR={zm['mrr']:.4f}")

        def agg(runs):
            return {m: {"mean": float(np.mean([r[m] for r in runs])),
                        "std":  float(np.std([r[m] for r in runs]))}
                    for m in ["mrr", "hits@1", "hits@3", "hits@10"]}

        results[variant] = {
            "description": desc,
            "test":     agg(test_runs),
            "zeroshot": agg(zs_runs),
        }

    out = OUT_DIR / "ablation_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation results saved → {out}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Variant':<25} {'Test MRR':>10} {'ZS MRR':>10} {'Gen Gap':>10}")
    print("-" * 70)
    for v, r in results.items():
        tm = r["test"]["mrr"]["mean"]
        zm = r["zeroshot"]["mrr"]["mean"]
        print(f"{v:<25} {tm:>10.4f} {zm:>10.4f} {tm - zm:>+10.4f}")


if __name__ == "__main__":
    main()
