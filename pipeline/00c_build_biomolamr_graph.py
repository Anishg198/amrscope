"""
Step 00c: Build the BioMolAMR heterogeneous graph with clean features.

Key differences from the original enhanced_graph.pkl:
  - Gene features: ESM-2 embeddings (480-dim) — NO drug-class membership
    The original 154-dim features included 46-dim drug class membership,
    creating a circular dependency for zero-shot prediction.
  - Drug features: Molecular fingerprints (3245-dim) — Morgan + MACCS + TopoTorsion
    The original 97-dim features used one-hot identity as the dominant component,
    making structurally related drug classes orthogonal.
  - Mechanism features: kept clean (8-dim one-hot, unchanged)
  - NEW: Drug-drug similarity edges based on Tanimoto similarity > threshold
    Allows chemical similarity information to propagate during drug encoding.

Output: data/processed/biomolamr_graph.pkl
  Contains:
    - hetero_data: HeteroData with gene/drug_class/mechanism nodes
    - dc2i:        drug_class_name → drug_class_index
    - gene2i:      model_id → gene_index
    - all_drug_classes: sorted list
    - gene_metadata: list of dicts per gene
"""

import json, pickle, sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CARD_JSON      = ROOT / "data/raw/card/card.json"
ORIG_GRAPH     = ROOT / "data/processed/enhanced_graph.pkl"
ESM2_EMB       = ROOT / "data/processed/esm2_embeddings.pt"
DRUG_FP        = ROOT / "data/processed/drug_fingerprints.pt"
DRUG_TAN       = ROOT / "data/processed/drug_tanimoto.pt"
DRUG_FP_META   = ROOT / "data/processed/drug_fp_meta.json"
OUT_PKL        = ROOT / "data/processed/biomolamr_graph.pkl"

TANIMOTO_EDGE_THRESHOLD = 0.20   # Tanimoto similarity threshold for drug-drug edges


def main():
    print("=" * 60)
    print("Step 00c: Building BioMolAMR heterogeneous graph")
    print("=" * 60)

    # ── Load original graph (for structure) ──────────────────────────────────
    with open(ORIG_GRAPH, "rb") as f:
        orig = pickle.load(f)

    orig_data  = orig["hetero_data"]
    dc2i       = orig["dc2i"]
    i2dc       = {v: k for k, v in dc2i.items()}
    all_dc     = orig["all_drug_classes"]
    gene_meta  = orig["gene_metadata"]
    n_genes    = orig_data["gene"].x.shape[0]
    n_dc       = orig_data["drug_class"].x.shape[0]
    n_mech     = orig_data["mechanism"].x.shape[0]

    print(f"Graph: {n_genes} genes, {n_dc} drug classes, {n_mech} mechanisms")

    # ── Load new features ─────────────────────────────────────────────────────
    print("\nLoading ESM-2 gene embeddings …")
    if not ESM2_EMB.exists():
        raise FileNotFoundError(
            f"{ESM2_EMB} not found. Run pipeline/00a_extract_esm2_embeddings.py first."
        )
    esm2_emb = torch.load(ESM2_EMB, map_location="cpu")   # (n_genes, 480)
    assert esm2_emb.shape[0] == n_genes, \
        f"ESM-2 embeddings have {esm2_emb.shape[0]} rows, expected {n_genes}"
    print(f"  Gene features (ESM-2): {esm2_emb.shape}")

    print("Loading molecular drug fingerprints …")
    if not DRUG_FP.exists():
        raise FileNotFoundError(
            f"{DRUG_FP} not found. Run pipeline/00b_compute_drug_fingerprints.py first."
        )
    drug_fp  = torch.load(DRUG_FP,  map_location="cpu")   # (n_dc, 3245)
    drug_tan = torch.load(DRUG_TAN, map_location="cpu")   # (n_dc, n_dc)
    assert drug_fp.shape[0] == n_dc, \
        f"Drug fingerprints have {drug_fp.shape[0]} rows, expected {n_dc}"
    print(f"  Drug features (mol FP): {drug_fp.shape}")

    # ── Mechanism features (keep original clean 8-dim one-hot) ───────────────
    mech_feat = orig_data["mechanism"].x   # (8, 46) or (8, 8), check shape
    # Rebuild clean one-hot: one dim per mechanism
    mech_feat_clean = torch.eye(n_mech, dtype=torch.float32)
    print(f"  Mechanism features (one-hot): {mech_feat_clean.shape}")

    # ── Build HeteroData ──────────────────────────────────────────────────────
    data = HeteroData()

    # Node features
    data["gene"].x         = esm2_emb.float()
    data["drug_class"].x   = drug_fp.float()
    data["mechanism"].x    = mech_feat_clean.float()

    # ── Copy existing edges ──────────────────────────────────────────────────
    # gene → drug_class (resistance edges — the main prediction target)
    res_ei = orig_data["gene", "confers_resistance_to", "drug_class"].edge_index
    data["gene", "confers_resistance_to", "drug_class"].edge_index = res_ei.clone()

    # gene → mechanism
    has_mech_ei = orig_data["gene", "has_mechanism", "mechanism"].edge_index
    data["gene", "has_mechanism", "mechanism"].edge_index = has_mech_ei.clone()

    # mechanism → gene (reverse, for bidirectional message passing)
    data["mechanism", "includes_gene", "gene"].edge_index = has_mech_ei[[1, 0]].clone()

    # ── NEW: Drug-drug chemical similarity edges ──────────────────────────────
    # Add edges between drug classes with Tanimoto similarity > threshold
    # This allows the drug encoder to leverage chemical neighborhood structure
    sim_mat = drug_tan.numpy()
    rows, cols = np.where(
        (sim_mat > TANIMOTO_EDGE_THRESHOLD) &
        (np.eye(n_dc, dtype=bool) == False)
    )
    drug_drug_ei = torch.tensor(np.stack([rows, cols], axis=0), dtype=torch.long)
    drug_drug_w  = torch.tensor(sim_mat[rows, cols], dtype=torch.float32)

    data["drug_class", "similar_to", "drug_class"].edge_index = drug_drug_ei
    data["drug_class", "similar_to", "drug_class"].edge_attr  = drug_drug_w.unsqueeze(1)

    n_drug_drug = drug_drug_ei.shape[1]
    print(f"\nDrug-drug similarity edges (Tanimoto > {TANIMOTO_EDGE_THRESHOLD}): {n_drug_drug}")
    print(f"  Average degree per drug class: {n_drug_drug / n_dc:.1f}")

    # Print some similar drug pairs
    print("\nSample drug-drug similarity pairs (>0.3):")
    for i in range(n_dc):
        for j in range(i+1, n_dc):
            if sim_mat[i, j] > 0.3:
                print(f"  {i2dc[i]} ↔ {i2dc[j]}: {sim_mat[i,j]:.3f}")

    # ── Statistics ────────────────────────────────────────────────────────────
    n_res_edges  = res_ei.shape[1]
    n_mech_edges = has_mech_ei.shape[1]
    print(f"\nGraph statistics:")
    print(f"  Gene nodes:              {n_genes} (features: {esm2_emb.shape[1]}-dim ESM-2)")
    print(f"  Drug class nodes:        {n_dc}   (features: {drug_fp.shape[1]}-dim mol FP)")
    print(f"  Mechanism nodes:         {n_mech}")
    print(f"  Resistance edges:        {n_res_edges}")
    print(f"  Gene-mechanism edges:    {n_mech_edges}")
    print(f"  Drug-drug sim edges:     {n_drug_drug}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "hetero_data":     data,
        "dc2i":            dc2i,
        "i2dc":            i2dc,
        "all_drug_classes": all_dc,
        "gene_metadata":   gene_meta,
        "gene2idx":        orig.get("gene2idx", orig.get("gene2i", {})),
        "tanimoto_matrix": drug_tan,
        "feature_info": {
            "gene_feat_dim":  esm2_emb.shape[1],
            "drug_feat_dim":  drug_fp.shape[1],
            "mech_feat_dim":  n_mech,
            "gene_features":  "ESM-2 esm2_t12_35M_UR50D 480-dim mean-pool",
            "drug_features":  "Morgan(2048) + MACCS(167) + TopoTorsion(1024) + target(5) + betaL(1)",
            "note": "NO drug-class membership in gene features (eliminates circular dependency)"
        }
    }

    with open(OUT_PKL, "wb") as f:
        pickle.dump(out, f)

    print(f"\nSaved BioMolAMR graph → {OUT_PKL}")
    print(f"File size: {OUT_PKL.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
