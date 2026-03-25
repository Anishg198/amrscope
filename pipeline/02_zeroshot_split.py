"""
Step 2: Create biologically-grounded zero-shot train/val/test splits.

Zero-shot design
────────────────
We hold out drug classes that have GENUINE cross-resistance relationships
with training drug classes.  This enables meaningful (non-trivial) zero-shot
evaluation, because resistance genes are SHARED between related drug classes.

Held-out (zero-shot) → Related training drug class → Gene overlap
  glycylcycline        → tetracycline               → 100 % overlap
  lincosamide          → macrolide                  →  56 % overlap
  streptogramin A      → streptogramin              → 100 % overlap
  rifamycin            → fluoroquinolone            →  47 % overlap
  glycopeptide         → penicillin beta-lactam     →   1 % overlap (hard baseline)

This gives a graded evaluation:
  • High-overlap classes → expect high zero-shot performance (cross-resistance)
  • Low-overlap classes  → expect low zero-shot performance (novel mechanism)

Evaluation protocols
────────────────────
  1. Standard test split  – seen drug classes (15 % held-out edges)
  2. Zero-shot split      – held-out drug classes
     a. Within-zero-shot ranking  – rank among the 5 zero-shot drug classes
     b. All-class filtered ranking – rank among all 46 drug classes (filtered)

Output: data/processed/splits.pkl
"""

import pickle, sys, random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GRAPH_PKL = ROOT / "data/processed/enhanced_graph.pkl"
OUT_PATH  = ROOT / "data/processed/splits.pkl"

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Zero-shot drug classes — biologically justified by cross-resistance
ZEROSHOT_CLASSES = [
    "glycylcycline",            # tetracycline derivative  (100 % gene overlap)
    "lincosamide antibiotic",   # MLSB resistance family   ( 56 % w/ macrolide)
    "streptogramin A antibiotic",# MLSB + streptogramin    (100 % w/ streptogramin)
    "rifamycin antibiotic",     # RNA polymerase inhibitor ( 47 % w/ fluoroquinolone)
    "glycopeptide antibiotic",  # cell-wall target          (  1 % – hard case)
]

# Related training classes (same biological axis)
RELATED_TRAINING = {
    "glycylcycline":             "tetracycline antibiotic",
    "lincosamide antibiotic":    "macrolide antibiotic",
    "streptogramin A antibiotic":"streptogramin antibiotic",
    "rifamycin antibiotic":      "fluoroquinolone antibiotic",
    "glycopeptide antibiotic":   "penicillin beta-lactam",
}


def load_graph():
    with open(GRAPH_PKL, "rb") as f:
        return pickle.load(f)


def make_splits(obj):
    data      = obj["hetero_data"]
    dc2i      = obj["dc2i"]
    all_dc    = obj["all_drug_classes"]
    gene_meta = obj["gene_metadata"]

    edge_index = data["gene", "confers_resistance_to", "drug_class"].edge_index
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    n_total = len(src)
    print(f"Total gene→drug_class edges: {n_total}")

    idx2dc = {v: k for k, v in dc2i.items()}

    # ── Zero-shot class indices ────────────────────────────────────────────
    zs_indices = {}
    for dc in ZEROSHOT_CLASSES:
        if dc not in dc2i:
            print(f"  WARNING: '{dc}' not found, skipping")
            continue
        idx = dc2i[dc]
        cnt = (dst == idx).sum()
        if cnt < 5:
            print(f"  Skipping '{dc}' (only {cnt} edges)")
            continue
        zs_indices[dc] = idx

    print(f"\nZero-shot classes ({len(zs_indices)}):")
    for dc, idx in zs_indices.items():
        cnt       = (dst == idx).sum()
        related   = RELATED_TRAINING.get(dc, "?")
        if related in dc2i:
            ri = dc2i[related]
            genes_zs  = set(src[dst == idx])
            genes_tr  = set(src[dst == ri])
            overlap   = len(genes_zs & genes_tr)
            pct       = 100.0 * overlap / max(1, len(genes_zs))
            print(f"  [{idx:2d}] {dc}: {cnt} edges  "
                  f"(related: {related}, gene overlap: {overlap}/{len(genes_zs)} = {pct:.0f}%)")
        else:
            print(f"  [{idx:2d}] {dc}: {cnt} edges")

    zs_idx_set = set(zs_indices.values())

    # ── Standard splits (seen drug classes) ───────────────────────────────
    seen_mask  = ~np.isin(dst, list(zs_idx_set))
    seen_idx   = np.where(seen_mask)[0]

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(seen_idx))
    seen_idx = seen_idx[perm]

    n_seen  = len(seen_idx)
    n_val   = max(1, int(0.10 * n_seen))
    n_test  = max(1, int(0.15 * n_seen))
    n_train = n_seen - n_val - n_test

    train_idx = seen_idx[:n_train]
    val_idx   = seen_idx[n_train:n_train + n_val]
    test_idx  = seen_idx[n_train + n_val:]
    zs_raw    = np.where(~seen_mask)[0]

    print(f"\nSplit sizes:")
    print(f"  Train:      {len(train_idx):6d}")
    print(f"  Val:        {len(val_idx):6d}")
    print(f"  Test:       {len(test_idx):6d}")
    print(f"  Zero-shot:  {len(zs_raw):6d}")

    # Leakage check
    assert len(set(dst[train_idx]) & zs_idx_set) == 0, "Data leakage!"

    pos_pairs = set(zip(src.tolist(), dst.tolist()))
    n_genes   = data["gene"].x.shape[0]
    n_dcs     = data["drug_class"].x.shape[0]

    # ── Negative sampling ─────────────────────────────────────────────────
    rng2 = np.random.default_rng(SEED + 1)

    def neg_corrupt_drug(pos_src_arr, pos_dst_arr, n_neg_per, pool=None):
        """Corrupt the drug class side (gene-side stays fixed)."""
        neg_src, neg_dst = [], []
        pool_arr = np.array(list(pool)) if pool else None
        for g, d in zip(pos_src_arr, pos_dst_arr):
            count = 0; tries = 0
            while count < n_neg_per and tries < 500:
                nd = int(rng2.choice(pool_arr)) if pool else int(rng2.integers(0, n_dcs))
                if (g, nd) not in pos_pairs:
                    neg_src.append(g); neg_dst.append(nd)
                    count += 1
                tries += 1
        return np.array(neg_src), np.array(neg_dst)

    print("Sampling negatives …")
    # Standard splits: negatives from ALL seen drug classes
    train_neg_s, train_neg_d = neg_corrupt_drug(
        src[train_idx], dst[train_idx], 5)
    val_neg_s,   val_neg_d   = neg_corrupt_drug(
        src[val_idx],   dst[val_idx],   10)
    test_neg_s,  test_neg_d  = neg_corrupt_drug(
        src[test_idx],  dst[test_idx],  10)

    # Zero-shot negatives (WITHIN zero-shot classes only — for fair evaluation)
    zs_neg_s_within, zs_neg_d_within = neg_corrupt_drug(
        src[zs_raw], dst[zs_raw], 49,
        pool=zs_idx_set)   # rank among ZS drug classes only

    # Zero-shot negatives (ALL classes, filtered) — for all-class ranking
    zs_neg_s_all, zs_neg_d_all = neg_corrupt_drug(
        src[zs_raw], dst[zs_raw], 99,
        pool=None)  # includes seen drug classes

    print(f"  Train negatives:       {len(train_neg_s)}")
    print(f"  Val negatives:         {len(val_neg_s)}")
    print(f"  Test negatives:        {len(test_neg_s)}")
    print(f"  ZS within negatives:   {len(zs_neg_s_within)}")
    print(f"  ZS all negatives:      {len(zs_neg_s_all)}")

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
            "zs_related_training":   {dc: RELATED_TRAINING[dc] for dc in zs_indices},
            "train_drug_class_indices": sorted(set(dst[train_idx])),
            "n_genes": n_genes,
            "n_drug_classes": n_dcs,
            "pos_pairs": pos_pairs,
        }
    }

    return splits


def main():
    print("=" * 60)
    print("Step 2: Creating biologically-grounded zero-shot splits")
    print("=" * 60)

    obj    = load_graph()
    splits = make_splits(obj)

    with open(OUT_PATH, "wb") as f:
        pickle.dump(splits, f)
    print(f"\nSaved splits → {OUT_PATH}")

    meta = splits["meta"]
    print(f"\nZero-shot drug classes: {meta['zs_drug_class_names']}")
    print(f"Training drug classes:  {len(meta['train_drug_class_indices'])}")


if __name__ == "__main__":
    main()
