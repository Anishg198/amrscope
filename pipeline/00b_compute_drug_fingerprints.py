"""
Step 00b: Compute molecular fingerprints for all 46 CARD drug classes.

Features per drug class (4,100-dim total, projected to 256 in model):
  - Morgan fingerprint  radius=2, nBits=2048  (structural connectivity / circular)
  - MACCS keys          167-bit               (pharmacophoric keys)
  - Topological torsion nBits=1024            (conformational)
  - Biological target   5-dim                 (manual annotation from CARD)
  - beta-lactam flag    1-dim                 (superfamily membership)

Total: 2048 + 167 + 1024 + 5 + 1 = 3245 dim

Tanimoto similarity matrix (46×46) on Morgan fingerprints is also saved
for use in the structural alignment loss.

Output:
  data/processed/drug_fingerprints.pt   — (46, 3245) float32 tensor
  data/processed/drug_tanimoto.pt       — (46, 46)   float32 tensor
  data/processed/drug_fp_meta.json      — metadata (drug class order, dims)
"""

import json, pickle, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors
from rdkit import DataStructs

SMILES_FILE = ROOT / "data/raw/drug_smiles.json"
GRAPH_PKL   = ROOT / "data/processed/enhanced_graph.pkl"
OUT_DIR     = ROOT / "data/processed"

# Biological target encodings (same as original pipeline, kept clean)
DRUG_CLASS_TARGETS = {
    "aminocoumarin antibiotic":             [0, 0, 1, 0, 0],
    "aminoglycoside antibiotic":            [0, 1, 0, 0, 0],
    "antibacterial free fatty acids":       [0, 0, 0, 1, 0],
    "antibiotic without defined classification": [0, 0, 0, 0, 1],
    "bicyclomycin-like antibiotic":         [0, 0, 1, 0, 0],
    "carbapenem":                           [1, 0, 0, 0, 0],
    "cephalosporin":                        [1, 0, 0, 0, 0],
    "cycloserine-like antibiotic":          [1, 0, 0, 0, 0],
    "diaminopyrimidine antibiotic":         [0, 0, 0, 0, 1],
    "diarylquinoline antibiotic":           [0, 0, 0, 0, 1],
    "disinfecting agents and antiseptics":  [0, 0, 0, 1, 0],
    "elfamycin antibiotic":                 [0, 1, 0, 0, 0],
    "fluoroquinolone antibiotic":           [0, 0, 1, 0, 0],
    "fusidane antibiotic":                  [0, 1, 0, 0, 0],
    "glycopeptide antibiotic":              [1, 0, 0, 0, 0],
    "glycylcycline":                        [0, 1, 0, 0, 0],
    "isoniazid-like antibiotic":            [1, 0, 0, 0, 1],
    "lincosamide antibiotic":               [0, 1, 0, 0, 0],
    "macrolide antibiotic":                 [0, 1, 0, 0, 0],
    "moenomycin antibiotic":                [1, 0, 0, 0, 0],
    "monobactam":                           [1, 0, 0, 0, 0],
    "mupirocin-like antibiotic":            [0, 0, 0, 0, 1],
    "nitrofuran antibiotic":                [0, 0, 1, 0, 0],
    "nitroimidazole antibiotic":            [0, 0, 1, 0, 0],
    "nucleoside antibiotic":                [1, 0, 0, 0, 0],
    "orthosomycin antibiotic":              [0, 1, 0, 0, 0],
    "oxazolidinone antibiotic":             [0, 1, 0, 0, 0],
    "pactamycin-like antibiotic":           [0, 1, 0, 0, 0],
    "penicillin beta-lactam":               [1, 0, 0, 0, 0],
    "peptide antibiotic":                   [0, 0, 0, 1, 0],
    "phenicol antibiotic":                  [0, 1, 0, 0, 0],
    "phosphonic acid antibiotic":           [1, 0, 0, 0, 0],
    "pleuromutilin antibiotic":             [0, 1, 0, 0, 0],
    "polyamine antibiotic":                 [0, 0, 0, 1, 0],
    "pyrazine antibiotic":                  [0, 0, 0, 0, 1],
    "rifamycin antibiotic":                 [0, 0, 1, 0, 0],
    "salicylic acid antibiotic":            [0, 0, 0, 0, 1],
    "streptogramin A antibiotic":           [0, 1, 0, 0, 0],
    "streptogramin B antibiotic":           [0, 1, 0, 0, 0],
    "streptogramin antibiotic":             [0, 1, 0, 0, 0],
    "sulfonamide antibiotic":               [0, 0, 0, 0, 1],
    "sulfone antibiotic":                   [0, 0, 0, 0, 1],
    "tetracycline antibiotic":              [0, 1, 0, 0, 0],
    "thioamide antibiotic":                 [1, 0, 0, 0, 1],
    "thiosemicarbazone antibiotic":         [1, 0, 0, 0, 0],
    "zoliflodacin-like antibiotic":         [0, 0, 1, 0, 0],
}

BETA_LACTAM_CLASSES = {
    "carbapenem", "cephalosporin", "monobactam", "penicillin beta-lactam"
}

MORGAN_BITS   = 2048
MACCS_BITS    = 167
TOPO_BITS     = 1024
TARGET_BITS   = 5
BETAL_BITS    = 1
TOTAL_FP_BITS = MORGAN_BITS + MACCS_BITS + TOPO_BITS + TARGET_BITS + BETAL_BITS  # 3245


def smiles_to_fingerprints(smiles: str, dc_name: str) -> np.ndarray:
    """Convert a SMILES string to a concatenated fingerprint vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"  WARNING: could not parse SMILES for '{dc_name}', using zeros")
        return np.zeros(MORGAN_BITS + MACCS_BITS + TOPO_BITS, dtype=np.float32)

    # Morgan fingerprint
    fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=MORGAN_BITS)
    arr_morgan = np.zeros(MORGAN_BITS, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp_morgan, arr_morgan)

    # MACCS keys
    fp_maccs = MACCSkeys.GenMACCSKeys(mol)
    arr_maccs = np.zeros(MACCS_BITS, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp_maccs, arr_maccs)

    # Topological torsion
    fp_topo = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
        mol, nBits=TOPO_BITS
    )
    arr_topo = np.zeros(TOPO_BITS, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp_topo, arr_topo)

    return np.concatenate([arr_morgan, arr_maccs, arr_topo])


def compute_tanimoto(morgan_fps: list) -> np.ndarray:
    """Compute pairwise Tanimoto similarity matrix using Morgan fingerprints."""
    n = len(morgan_fps)
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = 1.0
            elif morgan_fps[i] is not None and morgan_fps[j] is not None:
                mat[i, j] = DataStructs.TanimotoSimilarity(morgan_fps[i], morgan_fps[j])
    return mat


def main():
    print("=" * 60)
    print("Step 00b: Computing molecular fingerprints for drug classes")
    print("=" * 60)

    with open(SMILES_FILE) as f:
        smiles_data = json.load(f)
    smiles_data.pop("_note", None)

    with open(GRAPH_PKL, "rb") as f:
        obj = pickle.load(f)
    all_dc = obj["all_drug_classes"]
    dc2i   = obj["dc2i"]
    n_dc   = len(all_dc)
    print(f"Drug classes: {n_dc}")

    fp_matrix   = np.zeros((n_dc, TOTAL_FP_BITS), dtype=np.float32)
    morgan_fps  = [None] * n_dc   # keep RDKit objects for Tanimoto

    for dc_name in all_dc:
        idx = dc2i[dc_name]
        # Fingerprint part
        if dc_name in smiles_data:
            smiles = smiles_data[dc_name]["smiles"]
            fp_vec = smiles_to_fingerprints(smiles, dc_name)
            # Also keep RDKit morgan object
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                morgan_fps[idx] = AllChem.GetMorganFingerprintAsBitVect(
                    mol, radius=2, nBits=MORGAN_BITS
                )
        else:
            print(f"  WARNING: no SMILES for '{dc_name}', using zeros")
            fp_vec = np.zeros(MORGAN_BITS + MACCS_BITS + TOPO_BITS, dtype=np.float32)

        # Biological target (5-dim)
        target = DRUG_CLASS_TARGETS.get(dc_name, [0, 0, 0, 0, 0])

        # Beta-lactam flag (1-dim)
        betal = [1.0] if dc_name in BETA_LACTAM_CLASSES else [0.0]

        fp_matrix[idx] = np.concatenate([fp_vec, target, betal])
        print(f"  [{idx:2d}] {dc_name}  nnz={int(fp_vec.sum())}")

    # L2-normalize
    norms = np.linalg.norm(fp_matrix, axis=1, keepdims=True) + 1e-8
    fp_matrix_norm = fp_matrix / norms

    # Tanimoto similarity matrix (Morgan FP only)
    print("\nComputing Tanimoto similarity matrix …")
    tanimoto = compute_tanimoto(morgan_fps)

    # Save
    fp_tensor  = torch.tensor(fp_matrix_norm, dtype=torch.float32)
    tan_tensor = torch.tensor(tanimoto,        dtype=torch.float32)

    torch.save(fp_tensor,  OUT_DIR / "drug_fingerprints.pt")
    torch.save(tan_tensor, OUT_DIR / "drug_tanimoto.pt")

    meta = {
        "drug_class_order": all_dc,
        "dc2i":             dc2i,
        "feature_dims": {
            "morgan":        MORGAN_BITS,
            "maccs":         MACCS_BITS,
            "topo_torsion":  TOPO_BITS,
            "bio_target":    TARGET_BITS,
            "beta_lactam":   BETAL_BITS,
            "total":         TOTAL_FP_BITS,
        },
    }
    with open(OUT_DIR / "drug_fp_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDrug fingerprint tensor : {fp_tensor.shape}")
    print(f"Tanimoto matrix         : {tan_tensor.shape}")
    print(f"Mean Tanimoto (off-diag): {(tanimoto.sum() - np.trace(tanimoto)) / (n_dc*(n_dc-1)):.4f}")
    print(f"\nSaved → {OUT_DIR}")

    # Sanity check: glycylcycline vs tetracycline similarity
    i_glycyl = dc2i.get("glycylcycline", -1)
    i_tet    = dc2i.get("tetracycline antibiotic", -1)
    if i_glycyl >= 0 and i_tet >= 0:
        sim = tanimoto[i_glycyl, i_tet]
        print(f"\nSanity check — Tanimoto(glycylcycline, tetracycline) = {sim:.4f}")
        print(f"  (should be > 0.3 for structurally related drugs)")

    i_linco = dc2i.get("lincosamide antibiotic", -1)
    i_macro = dc2i.get("macrolide antibiotic", -1)
    if i_linco >= 0 and i_macro >= 0:
        sim = tanimoto[i_linco, i_macro]
        print(f"Sanity check — Tanimoto(lincosamide, macrolide)    = {sim:.4f}")


if __name__ == "__main__":
    main()
