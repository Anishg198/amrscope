"""
Step 02b: Create extended zero-shot splits with 12 held-out drug classes.

Design principles:
  1. Original 5 ZS classes retained (glycylcycline, lincosamide, streptogramin A,
     rifamycin, glycopeptide) — allows direct comparison with baseline paper
  2. 7 additional ZS classes spanning different biological axes
  3. Minimum 10 edges per ZS class for reliable evaluation
  4. Classes chosen to cover all 5 biological target axes

12 Zero-shot drug classes and their biological justification:
  Original 5:
    glycylcycline          (100% tetracycline overlap)   — semi-synthetic tetracycline
    lincosamide antibiotic ( 56% macrolide overlap)      — MLSB cross-resistance
    streptogramin A        (100% streptogramin overlap)  — MLSB group
    rifamycin antibiotic   ( 47% fluoroquinolone overlap)— DNA-targeting
    glycopeptide antibiotic(  1% beta-lactam overlap)    — cell wall (hard case)
  New 7:
    aminocoumarin antibiotic   (DNA gyrase B)    related: fluoroquinolone
    oxazolidinone antibiotic   (50S ribosome)    related: phenicol antibiotic
    nitroimidazole antibiotic  (DNA/RNA)         related: nitrofuran antibiotic
    pleuromutilin antibiotic   (50S peptidyl)    related: lincosamide antibiotic
    phosphonic acid antibiotic (MurA cell wall)  related: penicillin beta-lactam
    isoniazid-like antibiotic  (InhA cell wall)  related: thioamide antibiotic
    fusidane antibiotic        (EF-G ribosomal)  related: elfamycin antibiotic

This gives 12 ZS classes and 34 training classes — a more rigorous evaluation.

Output: data/processed/extended_splits.pkl
"""

import pickle, random, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Support both old and new graph format
GRAPH_ORIG = ROOT / "data/processed/enhanced_graph.pkl"
GRAPH_NEW  = ROOT / "data/processed/biomolamr_graph.pkl"
OUT_PATH   = ROOT / "data/processed/extended_splits.pkl"

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── All 12 zero-shot drug classes ─────────────────────────────────────────────
ZEROSHOT_CLASSES_EXTENDED = [
    # Original 5
    "glycylcycline",
    "lincosamide antibiotic",
    "streptogramin A antibiotic",
    "rifamycin antibiotic",
    "glycopeptide antibiotic",
    # New 7
    "aminocoumarin antibiotic",
    "oxazolidinone antibiotic",
    "nitroimidazole antibiotic",
    "pleuromutilin antibiotic",
    "phosphonic acid antibiotic",
    "isoniazid-like antibiotic",
    "fusidane antibiotic",
]

# Related training classes (for overlap computation and reporting)
RELATED_TRAINING = {
    "glycylcycline":              "tetracycline antibiotic",
    "lincosamide antibiotic":     "macrolide antibiotic",
    "streptogramin A antibiotic": "streptogramin antibiotic",
    "rifamycin antibiotic":       "fluoroquinolone antibiotic",
    "glycopeptide antibiotic":    "penicillin beta-lactam",
    "aminocoumarin antibiotic":   "fluoroquinolone antibiotic",
    "oxazolidinone antibiotic":   "phenicol antibiotic",
    "nitroimidazole antibiotic":  "nitrofuran antibiotic",
    "pleuromutilin antibiotic":   "lincosamide antibiotic",
    "phosphonic acid antibiotic": "penicillin beta-lactam",
    "isoniazid-like antibiotic":  "thioamide antibiotic",
    "fusidane antibiotic":        "elfamycin antibiotic",
}

# ─────────────────────────────────────────────────────────────────────────────

def load_graph():
    """Load the BioMolAMR graph if available, fall back to original."""
    if GRAPH_NEW.exists():
        print(f"Loading BioMolAMR graph from {GRAPH_NEW}")
        with open(GRAPH_NEW, "rb") as f:
            return pickle.load(f), True
    print(f"BioMolAMR graph not found. Loading original from {GRAPH_ORIG}")
    with open(GRAPH_ORIG, "rb") as f:
        return pickle.load(f), False


def compute_gene_overlap(src, dst, zs_idx, related_idx):
    genes_zs  = set(src[dst == zs_idx])
    genes_rel = set(src[dst == related_idx])
    if not genes_zs:
        return 0.0
    return 100.0 * len(genes_zs & genes_rel) / len(genes_zs)


def neg_corrupt_drug(pos_src_arr, pos_dst_arr, n_neg_per, rng, pos_pairs, n_dcs, pool=None):
    """Corrupt drug side to generate negatives. Same protocol as original splits."""
    neg_src, neg_dst = [], []
    pool_arr = np.array(list(pool)) if pool else None
    for g, d in zip(pos_src_arr, pos_dst_arr):
        count = 0; tries = 0
        while count < n_neg_per and tries < 500:
            nd = int(rng.choice(pool_arr)) if pool_arr is not None else int(rng.integers(0, n_dcs))
            if (g, nd) not in pos_pairs:
                neg_src.append(g); neg_dst.append(nd)
                count += 1
            tries += 1
    return np.array(neg_src), np.array(neg_dst)


def make_extended_splits(obj):
    data      = obj["hetero_data"]
    dc2i      = obj["dc2i"]
    all_dc    = obj["all_drug_classes"]
    gene_meta = obj["gene_metadata"]

    edge_index = data["gene", "confers_resistance_to", "drug_class"].edge_index
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    n_total = len(src)
    n_genes = data["gene"].x.shape[0]
    n_dcs   = data["drug_class"].x.shape[0]

    print(f"Total edges: {n_total}")
    print(f"Genes: {n_genes},  Drug classes: {n_dcs}")

    idx2dc = {v: k for k, v in dc2i.items()}

    # ── Find valid ZS classes ─────────────────────────────────────────────────
    zs_indices = {}
    for dc in ZEROSHOT_CLASSES_EXTENDED:
        if dc not in dc2i:
            print(f"  SKIP: '{dc}' not in dc2i")
            continue
        idx = dc2i[dc]
        cnt = int((dst == idx).sum())
        if cnt < 10:
            print(f"  SKIP: '{dc}' only {cnt} edges (need ≥10)")
            continue
        zs_indices[dc] = idx

    print(f"\nZero-shot classes ({len(zs_indices)}):")
    for dc, idx in zs_indices.items():
        cnt     = int((dst == idx).sum())
        related = RELATED_TRAINING.get(dc)
        overlap = 0.0
        if related and related in dc2i:
            overlap = compute_gene_overlap(src, dst, idx, dc2i[related])
        print(f"  [{idx:2d}] {dc}: {cnt} edges  "
              f"(related: {related}, overlap: {overlap:.0f}%)")

    zs_idx_set = set(zs_indices.values())

    # ── Standard splits ───────────────────────────────────────────────────────
    seen_mask = ~np.isin(dst, list(zs_idx_set))
    seen_idx  = np.where(seen_mask)[0]

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(seen_idx))
    seen_idx = seen_idx[perm]

    n_seen  = len(seen_idx)
    n_val   = max(1, int(0.10 * n_seen))
    n_test  = max(1, int(0.15 * n_seen))
    n_train = n_seen - n_val - n_test

    train_idx = seen_idx[:n_train]
    val_idx   = seen_idx[n_train : n_train + n_val]
    test_idx  = seen_idx[n_train + n_val:]
    zs_raw    = np.where(~seen_mask)[0]

    print(f"\nSplit sizes:")
    print(f"  Train:      {len(train_idx):6d}")
    print(f"  Val:        {len(val_idx):6d}")
    print(f"  Test:       {len(test_idx):6d}")
    print(f"  Zero-shot:  {len(zs_raw):6d}  (across {len(zs_indices)} classes)")

    # Leakage check
    assert len(set(dst[train_idx]) & zs_idx_set) == 0, "Data leakage detected!"

    pos_pairs = set(zip(src.tolist(), dst.tolist()))
    rng2 = np.random.default_rng(SEED + 1)

    def sample_negatives(src_arr, dst_arr, n_neg, pool=None):
        return neg_corrupt_drug(src_arr, dst_arr, n_neg, rng2, pos_pairs, n_dcs, pool)

    print("Sampling negatives …")
    train_neg_s, train_neg_d = sample_negatives(src[train_idx], dst[train_idx], 5)
    val_neg_s,   val_neg_d   = sample_negatives(src[val_idx],   dst[val_idx],   10)
    test_neg_s,  test_neg_d  = sample_negatives(src[test_idx],  dst[test_idx],  10)

    # Within-ZS negatives (rank among all ZS drug classes)
    zs_neg_s_within, zs_neg_d_within = sample_negatives(
        src[zs_raw], dst[zs_raw], 99, pool=zs_idx_set)

    # All-class negatives (primary metric: rank among ALL 46 drug classes)
    zs_neg_s_all, zs_neg_d_all = sample_negatives(
        src[zs_raw], dst[zs_raw], 99, pool=None)

    print(f"  Train negatives: {len(train_neg_s)}")
    print(f"  ZS within:       {len(zs_neg_s_within)}")
    print(f"  ZS all-class:    {len(zs_neg_s_all)}")

    splits = {
        "train": {
            "pos_src": src[train_idx], "pos_dst": dst[train_idx],
            "neg_src": train_neg_s,    "neg_dst": train_neg_d,
        },
        "val": {
            "pos_src": src[val_idx],   "pos_dst": dst[val_idx],
            "neg_src": val_neg_s,      "neg_dst": val_neg_d,
        },
        "test": {
            "pos_src": src[test_idx],  "pos_dst": dst[test_idx],
            "neg_src": test_neg_s,     "neg_dst": test_neg_d,
        },
        "zeroshot": {
            "pos_src":     src[zs_raw],
            "pos_dst":     dst[zs_raw],
            "neg_src":     zs_neg_s_within,
            "neg_dst":     zs_neg_d_within,
            "neg_src_all": zs_neg_s_all,
            "neg_dst_all": zs_neg_d_all,
        },
        "meta": {
            "zs_drug_class_indices": sorted(zs_idx_set),
            "zs_drug_class_names":   [idx2dc[i] for i in sorted(zs_idx_set)],
            "zs_related_training":   {dc: RELATED_TRAINING.get(dc) for dc in zs_indices},
            "train_drug_class_indices": sorted(set(dst[train_idx])),
            "n_zs_classes": len(zs_indices),
            "n_genes": n_genes,
            "n_drug_classes": n_dcs,
            "pos_pairs": pos_pairs,
        }
    }
    return splits


def main():
    print("=" * 60)
    print("Step 02b: Creating extended 12-class zero-shot splits")
    print("=" * 60)

    obj, is_new = load_graph()
    splits = make_extended_splits(obj)

    with open(OUT_PATH, "wb") as f:
        pickle.dump(splits, f)

    meta = splits["meta"]
    print(f"\nZero-shot drug classes ({meta['n_zs_classes']}):")
    for name in meta["zs_drug_class_names"]:
        related = meta["zs_related_training"].get(name, "?")
        print(f"  {name}  ← related to {related}")

    print(f"\nTraining drug classes: {len(meta['train_drug_class_indices'])}")
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
