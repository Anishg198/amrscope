"""
Zero-Shot Heterogeneous Graph Attention Network for AMR Prediction
==================================================================

Key design principle
────────────────────
Drug-class nodes are encoded PURELY from their feature vectors (no learnable
lookup table).  This means the model generalises to drug classes that were
*never seen* during training, provided they have feature vectors.

Architecture
────────────
  GeneEncoder   : R-GAT over (gene ↔ mechanism) neighbourhood
  DrugEncoder   : 2-layer MLP on drug-class feature vectors
  MechEncoder   : linear on mechanism feature vectors
  Decoder       : DistMult / bilinear scoring
  Aux head      : mechanism-type classification (multi-label) on gene embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, SAGEConv, Linear


# ─────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ─────────────────────────────────────────────────────────────────────────────

class DrugClassEncoder(nn.Module):
    """
    Pure feature-based drug encoder – no embedding table.
    Works at inference on any drug class described by a feature vector.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GeneEncoder(nn.Module):
    """
    Relation-aware GAT over the gene–mechanism sub-graph.
    Also accepts drug-class neighbour information via a second attention layer.
    """

    def __init__(self, gene_in: int, mech_in: int, hidden_dim: int,
                 out_dim: int, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.input_proj_gene = nn.Linear(gene_in, hidden_dim)
        self.input_proj_mech = nn.Linear(mech_in, hidden_dim)

        # Layer 1 – gene ↔ mechanism
        self.conv1 = HeteroConv({
            ("gene", "has_mechanism", "mechanism"):
                GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                        dropout=dropout, add_self_loops=False),
            ("mechanism", "includes_gene", "gene"):
                GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                        dropout=dropout, add_self_loops=False),
        }, aggr="sum")

        # Layer 2 – same edges
        self.conv2 = HeteroConv({
            ("gene", "has_mechanism", "mechanism"):
                GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                        dropout=dropout, add_self_loops=False),
            ("mechanism", "includes_gene", "gene"):
                GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                        dropout=dropout, add_self_loops=False),
        }, aggr="sum")

        self.norm1_gene = nn.LayerNorm(hidden_dim)
        self.norm1_mech = nn.LayerNorm(hidden_dim)
        self.norm2_gene = nn.LayerNorm(hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, gene_x, mech_x, edge_index_dict):
        # Input projections
        h = {
            "gene": F.gelu(self.input_proj_gene(gene_x)),
            "mechanism": F.gelu(self.input_proj_mech(mech_x)),
        }
        # Layer 1
        h2 = self.conv1(h, edge_index_dict)
        h["gene"]      = self.norm1_gene(h2["gene"] + h["gene"])
        h["mechanism"] = self.norm1_mech(h2["mechanism"] + h["mechanism"])
        h["gene"]      = self.dropout(F.gelu(h["gene"]))
        h["mechanism"] = self.dropout(F.gelu(h["mechanism"]))

        # Layer 2
        h3 = self.conv2(h, edge_index_dict)
        h["gene"] = self.norm2_gene(h3["gene"] + h["gene"])
        h["gene"] = self.dropout(F.gelu(h["gene"]))

        return self.out_proj(h["gene"])


class BilinearDecoder(nn.Module):
    """
    DistMult-style bilinear scoring:  score(g, d) = g^T diag(W) d
    """

    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(dim))
        nn.init.xavier_uniform_(self.W.unsqueeze(0))

    def forward(self, gene_emb: torch.Tensor, drug_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gene_emb : (B, dim)
            drug_emb : (B, dim)
        Returns:
            scores   : (B,)
        """
        return (gene_emb * self.W * drug_emb).sum(dim=-1)

    def score_all(self, gene_emb: torch.Tensor, drug_emb: torch.Tensor) -> torch.Tensor:
        """
        All-pairs scoring.
        gene_emb : (G, dim)
        drug_emb : (D, dim)
        Returns  : (G, D)
        """
        return (gene_emb * self.W).mm(drug_emb.t())


class MechanismClassifier(nn.Module):
    """Auxiliary multi-label classifier: predict resistance mechanism from gene embedding."""

    def __init__(self, in_dim: int, n_mechanisms: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_mechanisms)

    def forward(self, gene_emb: torch.Tensor) -> torch.Tensor:
        return self.fc(gene_emb)


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class ZeroShotHetGAT(nn.Module):
    """
    Zero-Shot Heterogeneous GAT for Antimicrobial Resistance Prediction.

    Can predict resistance scores for drug classes NEVER seen during training
    as long as their feature vectors are provided.
    """

    def __init__(
        self,
        gene_feat_dim:   int,
        drug_feat_dim:   int,
        mech_feat_dim:   int,
        hidden_dim:      int = 256,
        out_dim:         int = 128,
        num_heads:       int = 4,
        dropout:         float = 0.3,
        n_mechanisms:    int = 8,
    ):
        super().__init__()

        self.gene_encoder = GeneEncoder(
            gene_in=gene_feat_dim,
            mech_in=mech_feat_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.drug_encoder = DrugClassEncoder(
            in_dim=drug_feat_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
        )
        self.decoder = BilinearDecoder(out_dim)
        self.mech_classifier = MechanismClassifier(out_dim, n_mechanisms)

        self._out_dim = out_dim

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode_genes(self, hetero_data) -> torch.Tensor:
        edge_index_dict = {
            ("gene", "has_mechanism", "mechanism"):
                hetero_data["gene", "has_mechanism", "mechanism"].edge_index,
            ("mechanism", "includes_gene", "gene"):
                hetero_data["mechanism", "includes_gene", "gene"].edge_index,
        }
        return self.gene_encoder(
            hetero_data["gene"].x,
            hetero_data["mechanism"].x,
            edge_index_dict,
        )

    def encode_drug_classes(self, drug_x: torch.Tensor) -> torch.Tensor:
        """Encode drug classes from feature vectors (zero-shot capable)."""
        return self.drug_encoder(drug_x)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, hetero_data, gene_idx: torch.Tensor, drug_idx: torch.Tensor):
        """
        Score specific (gene, drug_class) pairs.

        Args:
            hetero_data : HeteroData object
            gene_idx    : (B,) gene indices
            drug_idx    : (B,) drug class indices
        Returns:
            scores      : (B,)
        """
        gene_emb = self.encode_genes(hetero_data)
        drug_emb = self.encode_drug_classes(hetero_data["drug_class"].x)

        g = gene_emb[gene_idx]
        d = drug_emb[drug_idx]
        return self.decoder(g, d)

    def forward_with_aux(self, hetero_data, gene_idx, drug_idx):
        """Forward pass returning both link scores and mechanism logits."""
        gene_emb = self.encode_genes(hetero_data)
        drug_emb = self.encode_drug_classes(hetero_data["drug_class"].x)

        g = gene_emb[gene_idx]
        d = drug_emb[drug_idx]
        scores = self.decoder(g, d)

        # Auxiliary: mechanism prediction on query genes
        mech_logits = self.mech_classifier(g)
        return scores, mech_logits, gene_emb, drug_emb

    # ── Zero-shot inference ───────────────────────────────────────────────────

    def predict_for_new_drug_class(
        self,
        hetero_data,
        new_drug_features: torch.Tensor,
        top_k: int = 50,
    ):
        """
        Zero-shot: predict resistance genes for a drug class never seen in training.

        Args:
            hetero_data        : graph (used for gene encoding only)
            new_drug_features  : (D_new, feat_dim) feature vectors for new drug classes
            top_k              : number of top resistance genes to return per drug
        Returns:
            top_gene_indices   : (D_new, top_k)
            top_scores         : (D_new, top_k)
        """
        self.eval()
        with torch.no_grad():
            gene_emb = self.encode_genes(hetero_data)
            drug_emb = self.encode_drug_classes(new_drug_features)
            # All-pairs scores: (n_genes, n_new_drugs)
            all_scores = self.decoder.score_all(gene_emb, drug_emb)   # (G, D_new)
            all_scores_t = all_scores.t()                               # (D_new, G)
            top_scores, top_idx = torch.topk(all_scores_t, k=top_k, dim=1)
        return top_idx, top_scores

    def score_all_pairs(self, hetero_data):
        """Return all gene × drug-class score matrix.  Shape: (G, D)."""
        gene_emb = self.encode_genes(hetero_data)
        drug_emb = self.encode_drug_classes(hetero_data["drug_class"].x)
        return self.decoder.score_all(gene_emb, drug_emb)

    def drug_class_contrastive_loss(
        self, hetero_data, drug_sim_matrix: torch.Tensor, temperature: float = 0.1
    ) -> torch.Tensor:
        """
        Contrastive regularisation on drug class embeddings.

        Similar drug classes (high sim_matrix value) should have similar
        embeddings; dissimilar ones should be far apart.

        drug_sim_matrix : (D, D) symmetric similarity matrix (e.g. cosine sim
                          of drug class feature vectors, pre-computed)
        """
        drug_emb = self.encode_drug_classes(hetero_data["drug_class"].x)
        drug_emb_n = torch.nn.functional.normalize(drug_emb, dim=-1)

        # Pairwise cosine similarity in embedding space
        emb_sim = drug_emb_n @ drug_emb_n.t()   # (D, D)

        # MSE between embedding similarity and feature similarity
        loss = ((emb_sim - drug_sim_matrix) ** 2).mean()
        return loss
