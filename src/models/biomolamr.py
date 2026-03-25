"""
BioMolAMR: Biologically-Grounded Multi-Modal Zero-Shot AMR Prediction

Architecture:
  GeneEncoder      — ESM-2 features + 2-layer mechanism-bottleneck GAT
  DrugEncoder      — Molecular fingerprints + drug-drug similarity SAGE
  MechBottleneck   — Shared mechanism space for gene-drug alignment
  Decoder          — Mechanism-weighted bilinear scoring

Novel contributions vs prior work:
  1. ESM-2 gene features eliminate circular drug-class dependency
  2. Molecular fingerprints enable chemical-similarity-based generalization
  3. Drug-drug similarity graph propagates structural neighbors' information
  4. Mechanism-weighted decoder explicitly models the causal path:
     gene → mechanism → drug (rather than direct gene → drug)
  5. Prototype contrastive loss (in losses.py) creates structured mechanism space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, HeteroConv


# ─── Gene Encoder ─────────────────────────────────────────────────────────────

class GeneEncoder(nn.Module):
    """
    Mechanism-bottleneck graph attention encoder for resistance genes.

    Message passing runs over the gene ↔ mechanism bipartite subgraph.
    The bottleneck forces gene representations to route through mechanism nodes,
    providing an inductive bias that matches the biological resistance pathway.

    Gene features: ESM-2 embeddings (480-dim, sequence-only, no drug label info)
    Mechanism features: 8-dim one-hot (clean, no drug class contamination)
    """

    def __init__(self, gene_feat_dim: int, mech_feat_dim: int,
                 hidden_dim: int = 256, out_dim: int = 128,
                 num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim    = out_dim
        self.num_layers = num_layers
        self.dropout    = dropout

        # Input projections
        self.gene_proj = nn.Sequential(
            nn.Linear(gene_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.mech_proj = nn.Sequential(
            nn.Linear(mech_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Mechanism-bottleneck GAT layers (heterogeneous)
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ("gene", "has_mechanism", "mechanism"):
                    GATConv(hidden_dim, hidden_dim // num_heads,
                            heads=num_heads, dropout=dropout, add_self_loops=False),
                ("mechanism", "includes_gene", "gene"):
                    GATConv(hidden_dim, hidden_dim // num_heads,
                            heads=num_heads, dropout=dropout, add_self_loops=False),
            }, aggr="sum")
            self.conv_layers.append(conv)

        self.layer_norms_gene = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.layer_norms_mech = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> torch.Tensor:
        """
        Returns gene embeddings (n_genes, out_dim).
        """
        h_gene = self.gene_proj(x_dict["gene"])          # (G, H)
        h_mech = self.mech_proj(x_dict["mechanism"])     # (M, H)

        h = {"gene": h_gene, "mechanism": h_mech}

        for i, conv in enumerate(self.conv_layers):
            h_new = conv(h, edge_index_dict)
            # Residual + LayerNorm + activation
            h["gene"] = F.gelu(
                self.layer_norms_gene[i](h_new["gene"] + h["gene"])
            )
            h["mechanism"] = F.gelu(
                self.layer_norms_mech[i](h_new["mechanism"] + h["mechanism"])
            )
            h["gene"] = F.dropout(h["gene"], p=self.dropout, training=self.training)

        return self.out_proj(h["gene"])   # (G, out_dim)


# ─── Drug Encoder ─────────────────────────────────────────────────────────────

class DrugEncoder(nn.Module):
    """
    Molecular fingerprint encoder with drug-drug chemical similarity graph.

    Drug features: Morgan(2048) + MACCS(167) + TopoTorsion(1024) + target(5) + betaL(1)
    Drug-drug graph: edges where Tanimoto similarity > 0.2

    The chemical similarity graph allows the encoder to leverage structural
    neighbors when encoding a drug class. For zero-shot drug classes:
      - Their molecular features are still available (from SMILES)
      - Their neighbors in chemical space have resistance profiles from training
      - Message passing aggregates this neighborhood signal

    CRITICAL: No learned embedding table is used — drug representations are
    computed entirely from molecular features → true zero-shot capability.
    """

    def __init__(self, drug_feat_dim: int, hidden_dim: int = 256,
                 out_dim: int = 128, dropout: float = 0.3,
                 use_drug_graph: bool = True):
        super().__init__()
        self.use_drug_graph = use_drug_graph

        # Input projection MLP
        self.input_proj = nn.Sequential(
            nn.Linear(drug_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Drug-drug similarity SAGE (optional, for chemical neighborhood aggregation)
        if use_drug_graph:
            self.drug_sage = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
            self.drug_sage_norm = nn.LayerNorm(hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, drug_feat: torch.Tensor,
                drug_drug_edge_index: torch.Tensor = None) -> torch.Tensor:
        """
        drug_feat          : (n_dc, feat_dim)  molecular fingerprints
        drug_drug_edge_index: (2, n_edges)     similarity edges (optional)
        Returns: (n_dc, out_dim)
        """
        h = self.input_proj(drug_feat)  # (n_dc, H)

        if self.use_drug_graph and drug_drug_edge_index is not None:
            h_neigh = self.drug_sage(h, drug_drug_edge_index)   # (n_dc, H)
            h = F.gelu(self.drug_sage_norm(h + h_neigh))        # residual
            h = F.dropout(h, p=0.3, training=self.training)

        return self.out_proj(h)   # (n_dc, out_dim)


# ─── Mechanism-Weighted Decoder ───────────────────────────────────────────────

class MechanismWeightedDecoder(nn.Module):
    """
    Mechanism-routing bilinear decoder.

    Scores a (gene, drug) pair by routing through mechanism space:

      w_m = softmax( MLP([z_gene, P_m]) )         # gate: which mechanism matters?
      z_gene_gated = sum_m w_m * P_m               # mechanism-weighted gene rep
      score = (z_gene + z_gene_gated)^T W z_drug   # bilinear with mechanism context

    where P_m are the mechanism prototype embeddings (from PrototypeContrastiveLoss).

    This is strictly more expressive than DistMult (diagonal W) and encodes
    the biological prior that resistance is mechanism-mediated.
    """

    def __init__(self, out_dim: int = 128, n_mechanisms: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.n_mech = n_mechanisms

        # Learnable mechanism embeddings (also used as prototypes in loss)
        self.mech_embeddings = nn.Embedding(n_mechanisms, out_dim)
        nn.init.xavier_uniform_(self.mech_embeddings.weight)

        # Gate MLP: (gene_emb || mech_emb) → scalar weight
        self.gate_mlp = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, 1),
        )

        # Full bilinear relation matrix (not diagonal like DistMult)
        self.W = nn.Parameter(torch.empty(out_dim, out_dim))
        nn.init.xavier_uniform_(self.W)

        self.drop = nn.Dropout(dropout)

    def forward(self, gene_emb: torch.Tensor,
                drug_emb: torch.Tensor) -> torch.Tensor:
        """
        gene_emb : (B, D)
        drug_emb : (B, D)
        Returns  : (B,)  scores
        """
        B, D = gene_emb.shape
        protos = self.mech_embeddings.weight  # (8, D)

        # Compute mechanism gates for each gene-mechanism pair
        gene_exp  = gene_emb.unsqueeze(1).expand(-1, self.n_mech, -1)  # (B, 8, D)
        proto_exp = protos.unsqueeze(0).expand(B, -1, -1)               # (B, 8, D)
        gate_in   = torch.cat([gene_exp, proto_exp], dim=-1)            # (B, 8, 2D)
        gates     = self.gate_mlp(gate_in).squeeze(-1)                  # (B, 8)
        weights   = F.softmax(gates, dim=-1)                            # (B, 8)

        # Mechanism-gated gene representation
        gene_gated = (weights.unsqueeze(-1) * proto_exp).sum(dim=1)     # (B, D)

        # Combine original and mechanism-gated
        gene_final = self.drop(gene_emb + gene_gated)                   # (B, D)

        # Bilinear scoring: (B, D) @ (D, D) = (B, D), element-wise with drug
        scores = (gene_final @ self.W * drug_emb).sum(dim=-1)           # (B,)
        return scores

    def score_all(self, gene_emb: torch.Tensor,
                  drug_emb: torch.Tensor) -> torch.Tensor:
        """
        gene_emb : (G, D)
        drug_emb : (D_dc, D)
        Returns  : (G, D_dc)  all-pairs score matrix
        """
        # Use mean mechanism gating for efficient all-pairs scoring
        protos = self.mech_embeddings.weight  # (8, D)
        mean_proto = protos.mean(0, keepdim=True)  # (1, D) — approximation

        gene_final = gene_emb + mean_proto.expand_as(gene_emb)   # (G, D)
        # Bilinear: gene (G, D) @ W (D, D) = (G, D) → (G, D_dc)
        gene_W = gene_final @ self.W                              # (G, D)
        return gene_W @ drug_emb.t()                              # (G, D_dc)

    def get_mechanism_prototypes(self) -> torch.Tensor:
        """Return mechanism prototype vectors for use in contrastive loss."""
        return self.mech_embeddings.weight


# ─── Mechanism Classifier Head ────────────────────────────────────────────────

class MechanismClassifier(nn.Module):
    """Multi-label classifier for resistance mechanism prediction (auxiliary task)."""

    def __init__(self, in_dim: int = 128, n_mechanisms: int = 8):
        super().__init__()
        self.head = nn.Linear(in_dim, n_mechanisms)

    def forward(self, gene_emb: torch.Tensor) -> torch.Tensor:
        """Returns (B, 8) raw logits for mechanism prediction."""
        return self.head(gene_emb)


# ─── Full BioMolAMR Model ─────────────────────────────────────────────────────

class BioMolAMR(nn.Module):
    """
    BioMolAMR: Full zero-shot AMR prediction model.

    Parameters
    ----------
    gene_feat_dim : int
        Dimensionality of gene features (480 for ESM-2 t12_35M).
    drug_feat_dim : int
        Dimensionality of drug features (3245 for Morgan+MACCS+TopoTorsion+target+betaL).
    mech_feat_dim : int
        Dimensionality of mechanism features (8 for one-hot identity).
    hidden_dim : int
        Hidden dimension throughout the model (default 256).
    out_dim : int
        Output embedding dimension (default 128).
    num_heads : int
        Number of attention heads in gene encoder GAT layers (default 4).
    num_gat_layers : int
        Number of mechanism-bottleneck GAT layers in gene encoder (default 2).
    dropout : float
        Dropout rate (default 0.3).
    n_mechanisms : int
        Number of resistance mechanisms (default 8, matches CARD annotation).
    use_drug_graph : bool
        Whether to use drug-drug chemical similarity graph (default True).
    """

    def __init__(
        self,
        gene_feat_dim:   int   = 480,
        drug_feat_dim:   int   = 3245,
        mech_feat_dim:   int   = 8,
        hidden_dim:      int   = 256,
        out_dim:         int   = 128,
        num_heads:       int   = 4,
        num_gat_layers:  int   = 2,
        dropout:         float = 0.3,
        n_mechanisms:    int   = 8,
        use_drug_graph:  bool  = True,
    ):
        super().__init__()
        self.out_dim   = out_dim
        self.n_mech    = n_mechanisms

        self.gene_encoder = GeneEncoder(
            gene_feat_dim, mech_feat_dim, hidden_dim, out_dim,
            num_heads, num_gat_layers, dropout
        )
        self.drug_encoder = DrugEncoder(
            drug_feat_dim, hidden_dim, out_dim, dropout, use_drug_graph
        )
        self.decoder = MechanismWeightedDecoder(out_dim, n_mechanisms, dropout=0.1)
        self.mech_classifier = MechanismClassifier(out_dim, n_mechanisms)

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode_genes(self, hetero_data) -> torch.Tensor:
        """Encode all genes in the graph. Returns (n_genes, out_dim)."""
        x_dict = {
            "gene":      hetero_data["gene"].x,
            "mechanism": hetero_data["mechanism"].x,
        }
        edge_index_dict = {
            ("gene", "has_mechanism", "mechanism"):
                hetero_data["gene", "has_mechanism", "mechanism"].edge_index,
            ("mechanism", "includes_gene", "gene"):
                hetero_data["mechanism", "includes_gene", "gene"].edge_index,
        }
        return self.gene_encoder(x_dict, edge_index_dict)

    def encode_drug_classes(self, hetero_data) -> torch.Tensor:
        """Encode all drug classes. Returns (n_dc, out_dim)."""
        drug_feat = hetero_data["drug_class"].x
        drug_drug_ei = None
        if ("drug_class", "similar_to", "drug_class") in hetero_data.edge_index_dict:
            drug_drug_ei = hetero_data["drug_class", "similar_to", "drug_class"].edge_index
        return self.drug_encoder(drug_feat, drug_drug_ei)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, hetero_data,
                gene_idx: torch.Tensor,
                drug_idx: torch.Tensor) -> torch.Tensor:
        """
        Score a batch of (gene, drug) pairs.

        Returns (B,) scores.
        """
        gene_emb = self.encode_genes(hetero_data)        # (G, D)
        drug_emb = self.encode_drug_classes(hetero_data) # (n_dc, D)

        g = gene_emb[gene_idx]   # (B, D)
        d = drug_emb[drug_idx]   # (B, D)

        return self.decoder(g, d)

    # ── Training helpers ──────────────────────────────────────────────────────

    def forward_with_extras(self, hetero_data, gene_idx, drug_idx):
        """
        Extended forward for training — returns scores + extras needed for losses.

        Returns:
          scores     : (B,)
          gene_emb   : (G, D)   full gene embeddings for structural loss
          drug_emb   : (n_dc, D) full drug embeddings for structural loss
          mech_logits: (B, 8)   mechanism predictions for focal loss
          batch_gene_emb: (B, D) gene embeddings for the batch (for proto loss)
        """
        gene_emb = self.encode_genes(hetero_data)
        drug_emb = self.encode_drug_classes(hetero_data)

        g_batch = gene_emb[gene_idx]   # (B, D)
        d_batch = drug_emb[drug_idx]   # (B, D)

        scores      = self.decoder(g_batch, d_batch)
        mech_logits = self.mech_classifier(g_batch)

        return scores, gene_emb, drug_emb, mech_logits, g_batch

    def predict_zeroshot(self, hetero_data,
                         new_drug_features: torch.Tensor,
                         top_k: int = 20) -> dict:
        """
        Zero-shot inference: predict top-k resistance genes for a new drug class.

        new_drug_features : (1, drug_feat_dim) molecular fingerprint of new drug
        Returns: dict with ranked gene indices and scores
        """
        self.eval()
        with torch.no_grad():
            gene_emb = self.encode_genes(hetero_data)  # (G, D)

            # Encode new drug (no graph needed — feature-based)
            drug_emb = self.drug_encoder(new_drug_features, None)  # (1, D)

            # Score all genes
            scores = self.decoder.score_all(gene_emb, drug_emb).squeeze(1)  # (G,)

            top_scores, top_idx = scores.topk(top_k)

        return {
            "gene_indices": top_idx.cpu().numpy(),
            "scores":       top_scores.cpu().numpy(),
        }

    def drug_class_contrastive_loss(self, hetero_data,
                                    tanimoto_matrix: torch.Tensor) -> torch.Tensor:
        """
        Structural alignment loss for drug embeddings (called during training).
        """
        drug_emb = self.encode_drug_classes(hetero_data)
        emb_norm = F.normalize(drug_emb, dim=-1)
        learned_sim = emb_norm @ emb_norm.t()
        return F.mse_loss(learned_sim, tanimoto_matrix.to(drug_emb.device))
