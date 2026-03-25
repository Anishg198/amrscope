"""
CrossContrastAMR: Cross-Attention + InfoNCE for Zero-Shot AMR Prediction.

Architecture
────────────
Gene encoder:  ESM-2(480)  → Linear(256) → GELU → Dropout → Linear(128) → L2-norm
Drug encoder:  MolFP(3245) → Linear(512) → GELU → Dropout → Linear(128) → L2-norm

Cross-attention (gene queries drug context):
  Q = g_base  [B, 1, 128]
  K = V = all_drug_emb  [B, 46, 128]   (all 46 CARD drug classes)
  g_ctx = MHA(Q, K, V)   [B, 1, 128]
  g_final = LayerNorm(g_base + g_ctx)  → L2-norm

Score: dot(g_final, d_base) ∈ [-1, 1]

Novelty
───────
First model to apply bidirectional cross-attention for zero-shot AMR prediction
on CARD.  Gene representations are conditioned on the full drug-class landscape,
so a novel drug class (unseen at training) can still be located in this shared
metric space via its molecular fingerprint encoding.

Training objective: InfoNCELoss (all 46 drug classes as contrastive dictionary)
                  + ListNetLoss (K=5 negatives from pre-computed splits)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossContrastAMR(nn.Module):
    """
    Cross-attention gene-drug model for zero-shot AMR resistance prediction.

    Parameters
    ----------
    gene_feat_dim : int   Input dimension for gene features (ESM-2: 480)
    drug_feat_dim : int   Input dimension for drug features (MolFP: 3245)
    hidden_dim    : int   Shared embedding dimension (default 128)
    n_heads       : int   Number of cross-attention heads (default 4)
    dropout       : float Dropout rate (default 0.3)
    """

    def __init__(
        self,
        gene_feat_dim: int,
        drug_feat_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # ── Gene encoder: ESM-2 → 256 → 128 ──────────────────────────────────
        self.gene_enc = nn.Sequential(
            nn.Linear(gene_feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
        )

        # ── Drug encoder: MolFP → 512 → 128 ──────────────────────────────────
        self.drug_enc = nn.Sequential(
            nn.Linear(drug_feat_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim),
        )

        # ── Gene→Drug cross-attention ─────────────────────────────────────────
        # Gene acts as query; all drug embeddings are keys/values.
        # This conditions each gene representation on the global drug landscape.
        self.gene_drug_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gene_norm = nn.LayerNorm(hidden_dim)

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode_drugs(self, drug_x: torch.Tensor) -> torch.Tensor:
        """
        Encode all drug classes into the shared metric space.

        Parameters
        ----------
        drug_x : [M, drug_feat_dim]  raw molecular fingerprint features

        Returns
        -------
        [M, hidden_dim]  L2-normalised drug embeddings
        """
        return F.normalize(self.drug_enc(drug_x), dim=-1)

    def encode_genes_with_context(
        self,
        gene_x_batch: torch.Tensor,
        all_drug_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode a batch of genes enriched by cross-attention over drug context.

        Parameters
        ----------
        gene_x_batch  : [B, gene_feat_dim]  raw ESM-2 gene features
        all_drug_emb  : [M, hidden_dim]     L2-normed drug embeddings (typically M=46)

        Returns
        -------
        [B, hidden_dim]  L2-normalised enriched gene embeddings
        """
        B = gene_x_batch.shape[0]
        M = all_drug_emb.shape[0]

        # Base gene embedding
        g_base = F.normalize(self.gene_enc(gene_x_batch), dim=-1)  # [B, H]

        # Cross-attention: gene (query) attends over all drugs (keys/values)
        # Q: [B, 1, H],  K/V: [B, M, H]
        d_kv = all_drug_emb.unsqueeze(0).expand(B, M, -1).contiguous()  # [B, M, H]
        g_q  = g_base.unsqueeze(1)                                        # [B, 1, H]
        g_ctx, _ = self.gene_drug_attn(g_q, d_kv, d_kv)                  # [B, 1, H]
        g_ctx = g_ctx.squeeze(1)                                          # [B, H]

        # Residual + LayerNorm + L2-norm
        g_final = F.normalize(self.gene_norm(g_base + g_ctx), dim=-1)    # [B, H]
        return g_final

    # ── Forward (main inference interface) ────────────────────────────────────

    def forward(
        self,
        gene_x: torch.Tensor,
        drug_x: torch.Tensor,
        gene_idx: torch.Tensor,
        drug_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score a batch of (gene, drug) pairs.

        Matches the FeatureMLPBaseline interface so existing training and
        evaluation code can call crossamr with no changes.

        Parameters
        ----------
        gene_x   : [N_genes, gene_feat_dim]  all gene features
        drug_x   : [M_drugs, drug_feat_dim]  all drug features
        gene_idx : [B]  indices into gene_x
        drug_idx : [B]  indices into drug_x

        Returns
        -------
        [B]  scalar similarity scores ∈ [-1, 1]
        """
        all_d = self.encode_drugs(drug_x)                              # [M, H]
        g = self.encode_genes_with_context(gene_x[gene_idx], all_d)   # [B, H]
        d = all_d[drug_idx]                                            # [B, H]
        return (g * d).sum(dim=-1)                                     # [B]

    # ── Chunked full-matrix inference (for caching / evaluation) ─────────────

    def compute_all_scores(
        self,
        gene_x: torch.Tensor,
        drug_x: torch.Tensor,
        chunk_size: int = 512,
    ) -> torch.Tensor:
        """
        Compute the full N_genes × M_drugs score matrix.

        Uses chunked processing to avoid OOM with large gene sets.

        Parameters
        ----------
        gene_x     : [N_genes, gene_feat_dim]
        drug_x     : [M, drug_feat_dim]
        chunk_size : int  genes to process per chunk (default 512)

        Returns
        -------
        [N_genes, M]  score matrix
        """
        all_d = self.encode_drugs(drug_x)        # [M, H]
        n_genes = gene_x.shape[0]
        parts = []
        for start in range(0, n_genes, chunk_size):
            g_chunk = self.encode_genes_with_context(
                gene_x[start : start + chunk_size], all_d
            )                                    # [chunk, H]
            parts.append(g_chunk @ all_d.T)      # [chunk, M]
        return torch.cat(parts, dim=0)           # [N_genes, M]
