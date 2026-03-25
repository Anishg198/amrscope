"""
Step 1: Build enhanced heterogeneous graph from CARD database.

Adds proper drug class feature vectors (from ARO descriptions + biological targets)
so the model can generalise to unseen antibiotics at inference time.

Output: data/processed/enhanced_graph.pkl
         data/processed/drug_classes.json   (drug-class metadata)
         data/processed/gene_metadata.csv
"""

import json
import pickle
import csv
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch_geometric.data import HeteroData
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CARD_JSON = ROOT / "data/raw/card/card.json"
ARO_CAT_IDX = ROOT / "data/raw/card/aro_categories_index.tsv"
OUT_DIR = ROOT / "data/processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── biological target encoding (manual, 5 dims) ──────────────────────────────
# Axes: cell-wall | ribosome | DNA/RNA | membrane | metabolic
DRUG_CLASS_TARGETS = {
    "aminocoumarin antibiotic":             [0, 0, 1, 0, 0],
    "aminoglycoside antibiotic":            [0, 1, 0, 0, 0],
    "antibacterial free fatty acids":       [0, 0, 0, 1, 0],
    "antibiotic without defined classification": [0, 0, 0, 0, 1],
    "bicyclomycin-like antibiotic":         [0, 0, 0, 0, 1],
    "carbapenem":                           [1, 0, 0, 0, 0],
    "cephalosporin":                        [1, 0, 0, 0, 0],
    "cycloserine-like antibiotic":          [1, 0, 0, 0, 0],
    "diaminopyrimidine antibiotic":         [0, 0, 0, 0, 1],
    "diarylquinoline antibiotic":           [0, 1, 0, 0, 0],
    "disinfecting agents and antiseptics":  [0, 0, 0, 1, 0],
    "elfamycin antibiotic":                 [0, 1, 0, 0, 0],
    "fluoroquinolone antibiotic":           [0, 0, 1, 0, 0],
    "fusidane antibiotic":                  [0, 1, 0, 0, 0],
    "glycopeptide antibiotic":              [1, 0, 0, 0, 0],
    "glycylcycline":                        [0, 1, 0, 0, 0],
    "isoniazid-like antibiotic":            [0, 0, 0, 0, 1],
    "lincosamide antibiotic":               [0, 1, 0, 0, 0],
    "macrolide antibiotic":                 [0, 1, 0, 0, 0],
    "moenomycin antibiotic":                [1, 0, 0, 0, 0],
    "monobactam":                           [1, 0, 0, 0, 0],
    "mupirocin-like antibiotic":            [0, 0, 0, 0, 1],
    "nitrofuran antibiotic":                [0, 0, 1, 0, 0],
    "nitroimidazole antibiotic":            [0, 0, 1, 0, 0],
    "nucleoside antibiotic":                [0, 1, 0, 0, 0],
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
    "tetracenomycin antibiotic":            [0, 1, 0, 0, 0],
    "tetracycline antibiotic":              [0, 1, 0, 0, 0],
    "thioamide antibiotic":                 [0, 0, 0, 0, 1],
    "thiosemicarbazone antibiotic":         [0, 0, 0, 0, 1],
    "zoliflodacin-like antibiotic":         [0, 0, 1, 0, 0],
    "N/A":                                  [0, 0, 0, 0, 0],
}

# β-lactam super-family flag
BETA_LACTAM_CLASSES = {"carbapenem", "cephalosporin", "monobactam", "penicillin beta-lactam"}

ALL_MECHANISMS = [
    "antibiotic efflux",
    "antibiotic inactivation",
    "antibiotic target alteration",
    "antibiotic target protection",
    "antibiotic target replacement",
    "reduced permeability to antibiotic",
    "resistance by absence",
    "resistance by host-dependent nutrient acquisition",
]
MECH2IDX = {m: i for i, m in enumerate(ALL_MECHANISMS)}


# ─────────────────────────────────────────────────────────────────────────────

def parse_card_json():
    """Extract genes, drug_classes, mechanisms and relationships from card.json."""
    print("Loading card.json …")
    with open(CARD_JSON) as f:
        card = json.load(f)
    print(f"  {len(card)} entries loaded")

    genes = {}          # gene_id -> {name, description, drug_classes, mechanisms, family}
    drug_classes_seen = set()
    mechanisms_seen = set()

    for entry_id, entry in card.items():
        if not isinstance(entry, dict):
            continue
        model_type = entry.get("model_type", "")
        # Only keep gene-level resistance models
        if model_type not in ("protein homolog model", "protein variant model",
                               "rRNA gene variant model", "protein knockout model",
                               "protein overexpression model"):
            continue

        gene_id = entry.get("ARO_accession", entry_id)
        gene_name = entry.get("ARO_name", "")
        gene_desc = entry.get("ARO_description", "")

        cats = entry.get("ARO_category", {})
        dcs = set()
        mechs = set()
        families = set()
        for cv in cats.values():
            cls_name = cv.get("category_aro_class_name", "")
            cat_name = cv.get("category_aro_name", "")
            if cls_name == "Drug Class":
                dcs.add(cat_name)
                drug_classes_seen.add(cat_name)
            elif cls_name == "Resistance Mechanism":
                mechs.add(cat_name)
                mechanisms_seen.add(cat_name)
            elif cls_name == "AMR Gene Family":
                families.add(cat_name)

        if not dcs:
            continue  # skip genes with no drug class association

        genes[gene_id] = {
            "name": gene_name,
            "description": gene_desc,
            "drug_classes": sorted(dcs),
            "mechanisms": sorted(mechs),
            "families": sorted(families),
        }

    print(f"  {len(genes)} resistance genes with drug associations")
    print(f"  {len(drug_classes_seen)} unique drug classes")
    print(f"  {len(mechanisms_seen)} unique mechanisms")
    return genes, sorted(drug_classes_seen), sorted(mechanisms_seen)


def load_aro_categories_index():
    """Load gene → drug_class and mechanism from aro_categories_index.tsv."""
    gene_drug = defaultdict(set)
    gene_mech = defaultdict(set)
    gene_family = defaultdict(set)
    with open(ARO_CAT_IDX) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            prot_acc = row.get("Protein Accession", "").strip()
            dcs = [d.strip() for d in row.get("Drug Class", "").split(";") if d.strip()]
            mechs = [m.strip() for m in row.get("Resistance Mechanism", "").split(";") if m.strip()]
            fam = row.get("AMR Gene Family", "").strip()
            for dc in dcs:
                if dc and dc != "N/A":
                    gene_drug[prot_acc].add(dc)
            for m in mechs:
                if m and m != "N/A":
                    gene_mech[prot_acc].add(m)
            if fam and fam != "N/A":
                gene_family[prot_acc].add(fam)
    return gene_drug, gene_mech, gene_family


def build_drug_class_features(all_drug_classes: list, descriptions: dict) -> np.ndarray:
    """
    Build drug-class feature matrix.
    Features per drug class:
      - One-hot drug class identity (len(all_drug_classes) dims)  [identity feature]
      - Biological target encoding (5 dims)
      - β-lactam super-family flag (1 dim)
      - TF-IDF description features (50 dims, reduced via SVD)
    Total: len(all_drug_classes) + 5 + 1 + 50 dims
    """
    n = len(all_drug_classes)
    dc2i = {dc: i for i, dc in enumerate(all_drug_classes)}

    # --- one-hot identity ---
    identity = np.eye(n, dtype=np.float32)

    # --- biological targets ---
    targets = np.array(
        [DRUG_CLASS_TARGETS.get(dc, [0, 0, 0, 0, 0]) for dc in all_drug_classes],
        dtype=np.float32,
    )

    # --- beta-lactam flag ---
    beta = np.array([[1.0] if dc in BETA_LACTAM_CLASSES else [0.0] for dc in all_drug_classes],
                    dtype=np.float32)

    # --- TF-IDF on descriptions ---
    texts = [descriptions.get(dc, dc) for dc in all_drug_classes]
    tfidf = TfidfVectorizer(max_features=200, sublinear_tf=True, ngram_range=(1, 2))
    tfidf_mat = tfidf.fit_transform(texts).toarray().astype(np.float32)

    # Reduce with truncated SVD to 50 dims
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=min(50, n - 1))
    tfidf_reduced = svd.fit_transform(tfidf_mat).astype(np.float32)

    features = np.concatenate([identity, targets, beta, tfidf_reduced], axis=1)
    features = normalize(features, norm="l2")
    print(f"  Drug class feature dim: {features.shape[1]}")
    return features, dc2i


def build_gene_features(genes: dict, all_drug_classes: list, dc2i: dict) -> tuple:
    """
    Gene features:
      - Multi-hot drug class vector (which classes this gene confers resistance to)
      - Multi-hot mechanism vector (8 dims)
      - TF-IDF on description (50 dims)
    """
    gene_ids = sorted(genes.keys())
    gene2idx = {g: i for i, g in enumerate(gene_ids)}
    n = len(gene_ids)
    n_dc = len(all_drug_classes)

    dc_onehot = np.zeros((n, n_dc), dtype=np.float32)
    mech_onehot = np.zeros((n, len(ALL_MECHANISMS)), dtype=np.float32)
    descriptions = []

    for i, gid in enumerate(gene_ids):
        g = genes[gid]
        for dc in g["drug_classes"]:
            if dc in dc2i:
                dc_onehot[i, dc2i[dc]] = 1.0
        for m in g["mechanisms"]:
            if m in MECH2IDX:
                mech_onehot[i, MECH2IDX[m]] = 1.0
        descriptions.append(g.get("description", g["name"]))

    # TF-IDF on gene descriptions
    tfidf = TfidfVectorizer(max_features=500, sublinear_tf=True, ngram_range=(1, 2))
    tfidf_mat = tfidf.fit_transform(descriptions).toarray().astype(np.float32)
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=min(100, n - 1))
    desc_feats = svd.fit_transform(tfidf_mat).astype(np.float32)

    gene_feats = np.concatenate([dc_onehot, mech_onehot, desc_feats], axis=1)
    gene_feats = normalize(gene_feats, norm="l2")
    print(f"  Gene feature dim: {gene_feats.shape[1]}")
    return gene_feats, gene_ids, gene2idx


def build_edges(genes: dict, gene2idx: dict, dc2i: dict) -> tuple:
    """Build gene→drug_class edges."""
    src, dst = [], []
    for gid, info in genes.items():
        if gid not in gene2idx:
            continue
        gi = gene2idx[gid]
        for dc in info["drug_classes"]:
            if dc in dc2i:
                src.append(gi)
                dst.append(dc2i[dc])
    src = np.array(src, dtype=np.int64)
    dst = np.array(dst, dtype=np.int64)
    return src, dst


def build_mechanism_features(all_drug_classes: list, dc2i: dict, genes: dict, gene2idx: dict) -> np.ndarray:
    """Mechanism features: which drug classes they affect (multi-hot)."""
    n_mech = len(ALL_MECHANISMS)
    n_dc = len(all_drug_classes)
    # For each mechanism, which drug classes are affected
    mech_dc = np.zeros((n_mech, n_dc), dtype=np.float32)
    for gid, info in genes.items():
        for m in info["mechanisms"]:
            if m in MECH2IDX:
                for dc in info["drug_classes"]:
                    if dc in dc2i:
                        mech_dc[MECH2IDX[m], dc2i[dc]] = 1.0
    mech_feats = normalize(mech_dc, norm="l2")
    return mech_feats


def build_gene_mechanism_edges(genes: dict, gene2idx: dict) -> tuple:
    """Build gene→mechanism edges."""
    src, dst = [], []
    for gid, info in genes.items():
        if gid not in gene2idx:
            continue
        gi = gene2idx[gid]
        for m in info["mechanisms"]:
            if m in MECH2IDX:
                src.append(gi)
                dst.append(MECH2IDX[m])
    return np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)


def get_drug_class_descriptions() -> dict:
    """Extract descriptions for each drug class from card.json."""
    with open(CARD_JSON) as f:
        card = json.load(f)
    desc = {}
    for entry in card.values():
        if not isinstance(entry, dict):
            continue
        cats = entry.get("ARO_category", {})
        for cv in cats.values():
            if cv.get("category_aro_class_name") == "Drug Class":
                name = cv.get("category_aro_name", "")
                description = cv.get("category_aro_description", "")
                if name and name not in desc:
                    desc[name] = description
    return desc


def main():
    print("=" * 60)
    print("Step 1: Building enhanced heterogeneous graph")
    print("=" * 60)

    # 1. Parse CARD
    genes, all_drug_classes, _ = parse_card_json()

    # 2. Get drug class descriptions for TF-IDF
    print("Extracting drug class descriptions …")
    dc_descriptions = get_drug_class_descriptions()

    # 3. Build drug class features
    print("Building drug class features …")
    dc_feats, dc2i = build_drug_class_features(all_drug_classes, dc_descriptions)

    # 4. Build gene features
    print("Building gene features …")
    gene_feats, gene_ids, gene2idx = build_gene_features(genes, all_drug_classes, dc2i)

    # 5. Build edges
    print("Building edges …")
    g2dc_src, g2dc_dst = build_edges(genes, gene2idx, dc2i)
    g2m_src, g2m_dst = build_gene_mechanism_edges(genes, gene2idx)

    # 6. Build mechanism features
    print("Building mechanism features …")
    mech_feats = build_mechanism_features(all_drug_classes, dc2i, genes, gene2idx)

    print(f"\nGraph statistics:")
    print(f"  Genes:        {len(gene_ids)}")
    print(f"  Drug classes: {len(all_drug_classes)}")
    print(f"  Mechanisms:   {len(ALL_MECHANISMS)}")
    print(f"  Gene→DrugClass edges: {len(g2dc_src)}")
    print(f"  Gene→Mechanism edges: {len(g2m_src)}")

    # 7. Build HeteroData
    data = HeteroData()

    data["gene"].x = torch.tensor(gene_feats, dtype=torch.float)
    data["gene"].node_ids = gene_ids

    data["drug_class"].x = torch.tensor(dc_feats, dtype=torch.float)
    data["drug_class"].node_ids = all_drug_classes

    data["mechanism"].x = torch.tensor(mech_feats, dtype=torch.float)
    data["mechanism"].node_ids = ALL_MECHANISMS

    data["gene", "confers_resistance_to", "drug_class"].edge_index = torch.tensor(
        np.stack([g2dc_src, g2dc_dst]), dtype=torch.long
    )
    data["drug_class", "resistance_conferred_by", "gene"].edge_index = torch.tensor(
        np.stack([g2dc_dst, g2dc_src]), dtype=torch.long
    )
    data["gene", "has_mechanism", "mechanism"].edge_index = torch.tensor(
        np.stack([g2m_src, g2m_dst]), dtype=torch.long
    )
    data["mechanism", "includes_gene", "gene"].edge_index = torch.tensor(
        np.stack([g2m_dst, g2m_src]), dtype=torch.long
    )

    # 8. Save
    out_path = OUT_DIR / "enhanced_graph.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({
            "hetero_data": data,
            "gene2idx": gene2idx,
            "dc2i": dc2i,
            "all_drug_classes": all_drug_classes,
            "mechanisms": ALL_MECHANISMS,
            "gene_metadata": {gid: genes[gid] for gid in gene_ids},
        }, f)
    print(f"\nSaved to {out_path}")

    # Save drug class metadata as JSON
    dc_meta = {}
    for dc in all_drug_classes:
        dc_meta[dc] = {
            "description": dc_descriptions.get(dc, ""),
            "target": DRUG_CLASS_TARGETS.get(dc, [0, 0, 0, 0, 0]),
            "is_beta_lactam": dc in BETA_LACTAM_CLASSES,
        }
    with open(OUT_DIR / "drug_classes.json", "w") as f:
        json.dump(dc_meta, f, indent=2)
    print(f"Saved drug class metadata to {OUT_DIR / 'drug_classes.json'}")


if __name__ == "__main__":
    main()
