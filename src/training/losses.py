"""
Novel training objectives for BioMolAMR.

Four complementary losses, each targeting a different failure mode:

1. ListNet Ranking Loss   — directly optimizes rank of positive drug above negatives
                            (BCE optimises each pair independently; ListNet optimises rank)
2. Structural Alignment   — drug embedding cosine similarity ≈ Tanimoto fingerprint similarity
                            (enforces chemical structure is preserved in embedding space)
3. Mechanism Classification — multi-label focal loss on resistance mechanisms
                            (auxiliary task that grounds gene embeddings biologically)
4. Prototype Contrastive  — genes cluster around mechanism prototypes
                            (zero-shot: new drug maps to mechanism space → finds genes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── 1. ListNet Ranking Loss ──────────────────────────────────────────────────

class ListNetLoss(nn.Module):
    """
    ListNet-style softmax ranking loss.

    For each positive (gene, drug^+) with K negative drugs (drug^-_1..K):
      logits = [score(g, d^+), score(g, d^-_1), ..., score(g, d^-_K)]
      loss   = -log softmax(logits / tau)[0]   # prob that d^+ is ranked first

    Directly optimizes MRR: with tau→0 this becomes max-margin; with tau=0.1
    it provides smooth gradients while keeping the positive at rank 1.

    Additionally adds a margin term to push positives above ALL negatives:
      L_margin = mean( max(0, gamma - score(d^+) + score(d^-_k)) )
    """

    def __init__(self, temperature: float = 0.1, margin: float = 1.0,
                 margin_weight: float = 0.5):
        super().__init__()
        self.tau           = temperature
        self.margin        = margin
        self.margin_weight = margin_weight

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        """
        pos_scores : (B,)    — scores for positive (gene, drug) pairs
        neg_scores : (B, K)  — scores for K negatives per positive
        """
        # ListNet: softmax over all scores, maximise P(positive is rank 1)
        all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)  # (B, K+1)
        log_probs  = F.log_softmax(all_scores / self.tau, dim=1)               # (B, K+1)
        l_listnet  = -log_probs[:, 0].mean()

        # Margin: max(0, margin - pos + neg) summed over negatives
        margin_violations = F.relu(
            self.margin - pos_scores.unsqueeze(1) + neg_scores
        )  # (B, K)
        l_margin = margin_violations.mean()

        return l_listnet + self.margin_weight * l_margin


# ─── 2. Structural Alignment Loss ────────────────────────────────────────────

class StructuralAlignmentLoss(nn.Module):
    """
    Enforces that pairwise cosine similarity between learned embeddings
    matches a pre-computed reference similarity matrix.

    For drugs: reference = Tanimoto fingerprint similarity (captures chemical structure)
    For genes: reference = ESM-2 embedding cosine similarity (captures sequence homology)

    This directly encodes the key biological prior:
      structurally similar drugs → similar resistance gene profiles (cross-resistance)
      sequence-similar genes → similar resistance mechanisms
    """

    def __init__(self):
        super().__init__()

    def forward(self, embeddings: torch.Tensor,
                reference_sim: torch.Tensor) -> torch.Tensor:
        """
        embeddings    : (N, D)   learned embeddings (will be L2-normalised internally)
        reference_sim : (N, N)   pre-computed pairwise similarity (e.g. Tanimoto)
        """
        emb_norm = F.normalize(embeddings, dim=-1)
        learned_sim = emb_norm @ emb_norm.t()  # (N, N)
        return F.mse_loss(learned_sim, reference_sim.to(embeddings.device))


# ─── 3. Mechanism Classification Loss (Multi-label Focal) ────────────────────

class MechanismFocalLoss(nn.Module):
    """
    Focal binary cross-entropy for multi-label mechanism classification.

    Standard BCE is dominated by easy negatives (7 of 8 mechanism bits are 0).
    Focal loss downweights easy examples and focuses training on hard ones:
      FL(p, y) = -alpha * (1-p)^gamma * y*log(p) - (1-alpha) * p^gamma * (1-y)*log(1-p)

    alpha=0.25 (standard), gamma=2 (standard from RetinaNet).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, 8) raw mechanism logits
        targets : (B, 8) multi-hot mechanism labels in {0, 1}
        """
        probs  = torch.sigmoid(logits)
        # Standard BCE per element
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        # Focal weight
        pt = probs * targets + (1 - probs) * (1 - targets)  # P(correct class)
        focal_weight = (1 - pt) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = (alpha_t * focal_weight * bce).mean()
        return loss


# ─── 4. Prototype Contrastive Loss ───────────────────────────────────────────

class PrototypeContrastiveLoss(nn.Module):
    """
    Mechanism-prototype contrastive loss.

    Maintains one learnable prototype vector per resistance mechanism.
    For each gene:
      - Push gene embedding towards its mechanism prototypes (positive)
      - Pull gene embedding away from other mechanism prototypes (negative)

    This creates a structured embedding space where:
      1. All genes using "antibiotic efflux" cluster around prototype P_efflux
      2. New drug classes can be mapped to mechanism space via molecular features
      3. Resistance prediction = find all genes near the drug's mechanism prototypes

    Formulation (margin-based):
      L = mean_g [ mean_{m+ in M_g}  ||z_g - P_{m+}||^2
                 - log sum_{m- not in M_g} exp(-||z_g - P_{m-}||^2 / sigma) ]
    """

    def __init__(self, n_mechanisms: int = 8, emb_dim: int = 128,
                 sigma: float = 1.0, min_examples: int = 1):
        super().__init__()
        self.n_mech       = n_mechanisms
        self.sigma        = sigma
        self.min_examples = min_examples
        # Prototypes are registered as parameters (trained end-to-end)
        self.prototypes = nn.Embedding(n_mechanisms, emb_dim)
        nn.init.xavier_uniform_(self.prototypes.weight)

    def forward(self, gene_emb: torch.Tensor,
                mechanism_labels: torch.Tensor) -> torch.Tensor:
        """
        gene_emb         : (B, D)   batch of gene embeddings
        mechanism_labels : (B, 8)   multi-hot mechanism assignments
        """
        B, D = gene_emb.shape
        protos = self.prototypes.weight  # (8, D)

        # Pairwise squared distances: (B, 8)
        dists_sq = torch.cdist(gene_emb, protos, p=2).pow(2)  # (B, 8)

        pos_mask = mechanism_labels.bool()      # (B, 8)
        neg_mask = ~pos_mask                    # (B, 8)

        # Positive loss: mean distance to own mechanism prototypes
        pos_count = pos_mask.sum(dim=1).clamp(min=1).float()
        pos_loss  = (dists_sq * pos_mask.float()).sum(dim=1) / pos_count  # (B,)

        # Negative loss: log-sum-exp of negative prototype distances (pulls away)
        neg_dists  = dists_sq / self.sigma  # scale
        neg_logsum = torch.logsumexp(-neg_dists * neg_mask.float() +
                                     (~neg_mask).float() * 1e9, dim=1)   # (B,)
        neg_loss = -neg_logsum  # (B,) — maximise distance to wrong prototypes

        # Only include samples that have at least one mechanism label
        has_label = pos_mask.any(dim=1)
        if not has_label.any():
            return torch.tensor(0.0, device=gene_emb.device, requires_grad=True)

        loss = (pos_loss[has_label] + neg_loss[has_label]).mean()
        return loss

    def get_prototypes(self) -> torch.Tensor:
        """Return current prototype embeddings (8, D)."""
        return self.prototypes.weight.detach()


# ─── Combined Loss ────────────────────────────────────────────────────────────

class BioMolAMRLoss(nn.Module):
    """
    Combined multi-objective loss for BioMolAMR.

    L = L_listnet
      + lambda_struct_drug  * L_struct_drug
      + lambda_struct_gene  * L_struct_gene    (batch-level approximation)
      + lambda_mech         * L_mech
      + lambda_proto        * L_proto

    Warm-up schedule: structural/proto losses are ramped in after `warmup_epochs`
    to avoid disrupting the link prediction task early in training.
    """

    def __init__(
        self,
        n_mechanisms:      int   = 8,
        emb_dim:           int   = 128,
        listnet_tau:       float = 0.1,
        listnet_margin:    float = 1.0,
        lambda_struct_drug: float = 0.5,
        lambda_struct_gene: float = 0.3,
        lambda_mech:        float = 0.2,
        lambda_proto:       float = 0.1,
        warmup_epochs:      int   = 50,
    ):
        super().__init__()
        self.lambda_struct_drug = lambda_struct_drug
        self.lambda_struct_gene = lambda_struct_gene
        self.lambda_mech        = lambda_mech
        self.lambda_proto       = lambda_proto
        self.warmup_epochs      = warmup_epochs

        self.listnet   = ListNetLoss(temperature=listnet_tau, margin=listnet_margin)
        self.struct    = StructuralAlignmentLoss()
        self.focal     = MechanismFocalLoss()
        self.proto     = PrototypeContrastiveLoss(n_mechanisms, emb_dim)

    def get_warmup_scale(self, epoch: int) -> float:
        """Linear warm-up for auxiliary losses."""
        if epoch < self.warmup_epochs:
            return epoch / self.warmup_epochs
        return 1.0

    def forward(
        self,
        pos_scores:      torch.Tensor,          # (B,)
        neg_scores:      torch.Tensor,          # (B, K)
        drug_embeddings: torch.Tensor,          # (n_dc, D)
        tanimoto_matrix: torch.Tensor,          # (n_dc, n_dc)
        gene_embeddings_batch: torch.Tensor,    # (B, D)
        esm2_sim_batch:  torch.Tensor,          # (B, B) ESM-2 gene similarities
        mech_logits:     torch.Tensor,          # (B, 8)
        mech_labels:     torch.Tensor,          # (B, 8)
        epoch:           int = 0,
    ) -> tuple[torch.Tensor, dict]:

        scale = self.get_warmup_scale(epoch)

        # 1. Main ranking loss (always active)
        l_link = self.listnet(pos_scores, neg_scores)

        # 2. Drug structural alignment
        l_struct_drug = self.struct(drug_embeddings, tanimoto_matrix)

        # 3. Gene structural alignment (batch-level)
        l_struct_gene = self.struct(gene_embeddings_batch, esm2_sim_batch)

        # 4. Mechanism classification
        l_mech = self.focal(mech_logits, mech_labels)

        # 5. Prototype contrastive
        l_proto = self.proto(gene_embeddings_batch, mech_labels)

        total = (l_link
                 + scale * self.lambda_struct_drug * l_struct_drug
                 + scale * self.lambda_struct_gene * l_struct_gene
                 + scale * self.lambda_mech        * l_mech
                 + scale * self.lambda_proto       * l_proto)

        breakdown = {
            "link":         l_link.item(),
            "struct_drug":  l_struct_drug.item(),
            "struct_gene":  l_struct_gene.item(),
            "mech":         l_mech.item(),
            "proto":        l_proto.item(),
            "warmup_scale": scale,
        }
        return total, breakdown
