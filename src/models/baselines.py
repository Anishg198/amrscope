"""
Baseline models for zero-shot AMR prediction comparison.

Models
──────
1. DistMult       – classic transductive KGE (cannot handle zero-shot)
2. RGCN           – R-GCN with drug feature initialisation (can do zero-shot)
3. FeatureMLP     – pure MLP on concatenated gene+drug features (no graph)
4. TransE         – translational KGE (transductive)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, RGCNConv, SAGEConv


# ─────────────────────────────────────────────────────────────────────────────
# 1. DistMult (transductive)
# ─────────────────────────────────────────────────────────────────────────────

class DistMultBaseline(nn.Module):
    """
    Standard DistMult with learnable gene and drug embeddings.
    Cannot generalise to drug classes not seen during training.
    Used as an upper-bound transductive baseline.
    """

    def __init__(self, n_genes: int, n_drug_classes: int, emb_dim: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        self.gene_emb  = nn.Embedding(n_genes, emb_dim)
        self.drug_emb  = nn.Embedding(n_drug_classes, emb_dim)
        self.W = nn.Parameter(torch.empty(emb_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.gene_emb.weight)
        nn.init.xavier_uniform_(self.drug_emb.weight)
        nn.init.xavier_uniform_(self.W.unsqueeze(0))

    def forward(self, gene_idx, drug_idx):
        g = self.dropout(self.gene_emb(gene_idx))
        d = self.dropout(self.drug_emb(drug_idx))
        return (g * self.W * d).sum(dim=-1)

    def score_all(self, gene_idx=None, drug_idx=None):
        if gene_idx is None:
            g = self.gene_emb.weight
        else:
            g = self.gene_emb(gene_idx)
        if drug_idx is None:
            d = self.drug_emb.weight
        else:
            d = self.drug_emb(drug_idx)
        return (g * self.W).mm(d.t())


# ─────────────────────────────────────────────────────────────────────────────
# 2. TransE (transductive)
# ─────────────────────────────────────────────────────────────────────────────

class TransEBaseline(nn.Module):
    """
    TransE: score(g, d) = -||g + r - d||_2
    Transductive – cannot handle unseen drug classes.
    """

    def __init__(self, n_genes: int, n_drug_classes: int, emb_dim: int = 128,
                 margin: float = 1.0, dropout: float = 0.3):
        super().__init__()
        self.gene_emb  = nn.Embedding(n_genes, emb_dim)
        self.drug_emb  = nn.Embedding(n_drug_classes, emb_dim)
        self.relation  = nn.Parameter(torch.empty(emb_dim))
        self.margin    = margin
        self.dropout   = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.gene_emb.weight)
        nn.init.xavier_uniform_(self.drug_emb.weight)
        nn.init.uniform_(self.relation, -0.1, 0.1)

    def forward(self, gene_idx, drug_idx):
        g = F.normalize(self.dropout(self.gene_emb(gene_idx)))
        d = F.normalize(self.dropout(self.drug_emb(drug_idx)))
        r = F.normalize(self.relation.unsqueeze(0)).squeeze(0)
        # Negative L2 distance as score
        return -(g + r - d).pow(2).sum(dim=-1).sqrt()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature MLP baseline (inductive / zero-shot capable)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureMLPBaseline(nn.Module):
    """
    Concatenate gene features and drug features, pass through MLP.
    No graph structure.  Zero-shot capable (only needs feature vectors).
    """

    def __init__(self, gene_feat_dim: int, drug_feat_dim: int,
                 hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        in_dim = gene_feat_dim + drug_feat_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, gene_x: torch.Tensor, drug_x: torch.Tensor,
                gene_idx: torch.Tensor, drug_idx: torch.Tensor) -> torch.Tensor:
        g = gene_x[gene_idx]
        d = drug_x[drug_idx]
        return self.net(torch.cat([g, d], dim=-1)).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# 4. R-GCN with feature-based drug initialisation (inductive / zero-shot)
# ─────────────────────────────────────────────────────────────────────────────

class RGCNBaseline(nn.Module):
    """
    2-layer R-GCN.  Gene nodes: GCN aggregation.
    Drug nodes: feature projection (no lookup table → zero-shot capable).
    """

    def __init__(self, gene_feat_dim: int, drug_feat_dim: int, mech_feat_dim: int,
                 hidden_dim: int = 128, out_dim: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        self.gene_proj = nn.Linear(gene_feat_dim, hidden_dim)
        self.drug_proj = nn.Linear(drug_feat_dim, out_dim)
        self.mech_proj = nn.Linear(mech_feat_dim, hidden_dim)

        self.conv1 = HeteroConv({
            ("gene", "has_mechanism", "mechanism"):
                SAGEConv((hidden_dim, hidden_dim), hidden_dim, normalize=True),
            ("mechanism", "includes_gene", "gene"):
                SAGEConv((hidden_dim, hidden_dim), hidden_dim, normalize=True),
        }, aggr="mean")

        self.conv2 = HeteroConv({
            ("gene", "has_mechanism", "mechanism"):
                SAGEConv((hidden_dim, hidden_dim), out_dim, normalize=True),
            ("mechanism", "includes_gene", "gene"):
                SAGEConv((hidden_dim, hidden_dim), out_dim, normalize=True),
        }, aggr="mean")

        self.norm_gene = nn.LayerNorm(hidden_dim)
        self.norm_mech = nn.LayerNorm(hidden_dim)
        self.dropout   = nn.Dropout(dropout)

        # Bilinear decoder
        self.W = nn.Parameter(torch.empty(out_dim))
        nn.init.xavier_uniform_(self.W.unsqueeze(0))

    def encode(self, hetero_data):
        h_gene = F.gelu(self.gene_proj(hetero_data["gene"].x))
        h_mech = F.gelu(self.mech_proj(hetero_data["mechanism"].x))
        h_drug = self.drug_proj(hetero_data["drug_class"].x)

        ei = {
            ("gene", "has_mechanism", "mechanism"):
                hetero_data["gene", "has_mechanism", "mechanism"].edge_index,
            ("mechanism", "includes_gene", "gene"):
                hetero_data["mechanism", "includes_gene", "gene"].edge_index,
        }

        h = {"gene": h_gene, "mechanism": h_mech}
        h2 = self.conv1(h, ei)
        h["gene"]      = self.norm_gene(self.dropout(F.gelu(h2["gene"])))
        h["mechanism"] = self.norm_mech(self.dropout(F.gelu(h2["mechanism"])))
        h3 = self.conv2(h, ei)
        gene_emb = h3["gene"]
        return gene_emb, h_drug

    def forward(self, hetero_data, gene_idx, drug_idx):
        gene_emb, drug_emb = self.encode(hetero_data)
        g = gene_emb[gene_idx]
        d = drug_emb[drug_idx]
        return (g * self.W * d).sum(dim=-1)

    def score_all(self, hetero_data):
        gene_emb, drug_emb = self.encode(hetero_data)
        return (gene_emb * self.W).mm(drug_emb.t())
