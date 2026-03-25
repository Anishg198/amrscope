"""
Step 00a: Extract ESM-2 protein language model embeddings for CARD resistance genes.

Model: esm2_t12_35M_UR50D  (35M params, 480-dim embeddings)
       Chosen for CPU/MPS feasibility while providing strong sequence representations.

Strategy:
  - Load protein sequences from CARD json (protein_sequence field)
  - For genes without protein sequences (rRNA genes, knockout models), use zeros
  - Mean-pool over non-padding positions (excluding CLS/EOS tokens)
  - Match genes by ARO accession to the gene2idx mapping in enhanced_graph.pkl
  - Save as (n_genes, 480) float32 tensor

Output:
  data/processed/esm2_embeddings.pt   — (6397, 480) gene embeddings
  data/processed/esm2_meta.json       — coverage statistics

Runtime estimate: ~400 batches × 16 seqs × 0.1s/seq ≈ 40-60 min on CPU
                  With MPS (Apple Silicon): ~15-25 min
"""

import json, pickle, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CARD_DIR   = ROOT / "data/raw/card"
CARD_JSON  = CARD_DIR / "card.json"
GRAPH_PKL  = ROOT / "data/processed/enhanced_graph.pkl"
OUT_DIR    = ROOT / "data/processed"

# Protein FASTA files (all model types)
FASTA_FILES = [
    "protein_fasta_protein_homolog_model.fasta",
    "protein_fasta_protein_knockout_model.fasta",
    "protein_fasta_protein_overexpression_model.fasta",
    "protein_fasta_protein_variant_model.fasta",
]

ESM_MODEL  = "esm2_t12_35M_UR50D"   # 480-dim, fast, good quality
EMB_DIM    = 480
MAX_LEN    = 1022   # ESM-2 max context (1024 minus CLS/EOS)
BATCH_SIZE = 16


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_fasta(fasta_path: Path):
    """Parse a FASTA file → dict of header→sequence."""
    seqs = {}
    current_header = None
    current_seq = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header:
                    seqs[current_header] = "".join(current_seq)
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
    if current_header:
        seqs[current_header] = "".join(current_seq)
    return seqs


def extract_sequences_from_fasta(card_dir: Path):
    """Extract protein sequences from CARD FASTA files.

    FASTA headers look like:
      gb|ACT97415.1|ARO:3002999|CblA-1 [mixed culture bacterium ...]

    Returns:
      aro2seq  : ARO accession (e.g. 'ARO:3002999') → sequence
      name2seq : gene name → sequence
    """
    aro2seq  = {}
    name2seq = {}

    for fname in FASTA_FILES:
        fasta_path = card_dir / fname
        if not fasta_path.exists():
            print(f"  WARNING: FASTA file not found: {fasta_path}")
            continue

        seqs = parse_fasta(fasta_path)
        for header, seq in seqs.items():
            # Parse ARO accession from header: ...|ARO:XXXXXXX|...
            parts = header.split("|")
            aro = None
            gene_name = None
            for p in parts:
                if p.startswith("ARO:"):
                    aro = p.strip()
                    break
            # Gene name is typically the part after the ARO: field
            if aro and len(parts) >= 4:
                gene_name_raw = parts[3].split("[")[0].strip()
                gene_name = gene_name_raw

            if aro and seq:
                if aro not in aro2seq:  # first occurrence wins
                    aro2seq[aro] = seq
            if gene_name and seq:
                if gene_name not in name2seq:
                    name2seq[gene_name] = seq

        print(f"  Parsed {len(seqs)} sequences from {fname}")

    return aro2seq, name2seq


def extract_sequences_from_card(card_json_path: Path):
    """Extract gene sequences from CARD json, indexed by model_id.
    Returns id2seq (model_id→seq) and id2aro (model_id→aro_accession).
    Falls back gracefully if JSON values are not dicts.
    """
    with open(card_json_path) as f:
        card = json.load(f)

    id2seq = {}
    id2aro = {}
    for k, v in card.items():
        if k == "_version":
            continue
        if not isinstance(v, dict):
            continue
        model_id = str(v.get("model_id", ""))
        aro_acc  = str(v.get("ARO_accession", ""))
        ms = v.get("model_sequences", {})
        if not isinstance(ms, dict):
            continue
        seq_dict = ms.get("sequence", {})
        if not isinstance(seq_dict, dict):
            continue
        seq = ""
        for sid, sdata in seq_dict.items():
            if not isinstance(sdata, dict):
                continue
            ps = sdata.get("protein_sequence", {})
            if isinstance(ps, dict):
                ps = ps.get("sequence", "")
            else:
                ps = ""
            if ps:
                seq = ps
                break
        if seq:
            id2seq[model_id] = seq
            id2aro[model_id] = aro_acc

    return id2seq, id2aro


def main():
    print("=" * 60)
    print(f"Step 00a: Extracting ESM-2 embeddings ({ESM_MODEL})")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    # Load graph to get gene order
    with open(GRAPH_PKL, "rb") as f:
        obj = pickle.load(f)
    gene_meta = obj["gene_metadata"]   # dict: aro_id_str → {name, description, ...}
    gene2idx  = obj["gene2idx"]        # dict: aro_id_str → integer index in HeteroData
    n_genes   = len(gene2idx)
    # Build idx→aro mapping so we can look up by embedding index
    idx2aro   = {v: k for k, v in gene2idx.items()}
    print(f"Genes in graph: {n_genes}")

    # Extract sequences from FASTA files (primary source — most complete)
    print("\nParsing protein FASTA files …")
    aro2seq, name2seq = extract_sequences_from_fasta(CARD_DIR)
    print(f"Unique ARO sequences from FASTA: {len(aro2seq)}")
    print(f"Unique name sequences from FASTA: {len(name2seq)}")

    # Extract sequences from CARD JSON (fallback)
    print("\nParsing CARD JSON (fallback) …")
    id2seq, id2aro = extract_sequences_from_card(CARD_JSON)
    print(f"Sequences in CARD json: {len(id2seq)}")

    # Build reverse map: aro_accession → json sequence
    aro2seq_json = {}
    for mid, seq in id2seq.items():
        aro = id2aro.get(mid, "")
        if aro and aro not in aro2seq_json:
            aro2seq_json[aro] = seq

    # Map gene index → sequence
    # gene_meta is a dict: aro_id_str (e.g. "3000005") → {name, ...}
    # gene2idx  is a dict: aro_id_str → embedding index
    # aro2seq   is a dict: "ARO:3000005" → sequence (from FASTA)
    # name2seq  is a dict: gene_name → sequence (from FASTA)
    gene_seqs = [""] * n_genes
    n_found = 0
    for aro_id_str, idx in gene2idx.items():
        gm = gene_meta.get(aro_id_str, {})
        gm = gm if isinstance(gm, dict) else {}

        # Try 1: ARO accession key → FASTA (keys like "ARO:3000005")
        aro_full = f"ARO:{aro_id_str}"
        seq = aro2seq.get(aro_full, "")

        # Try 2: gene name → FASTA
        if not seq:
            gname = gm.get("name", "")
            seq = name2seq.get(gname, "")

        # Try 3: model_id → CARD JSON (model IDs are separate from ARO IDs in CARD)
        if not seq:
            # In CARD JSON, model IDs differ from ARO IDs; try both aro_id as model_id
            seq = id2seq.get(aro_id_str, "")

        if seq:
            n_found += 1
            if len(seq) > MAX_LEN:
                seq = seq[:MAX_LEN]
        gene_seqs[idx] = seq

    print(f"Genes with protein sequences: {n_found}/{n_genes} ({100*n_found/n_genes:.1f}%)")
    print(f"Genes without sequences (will use zeros): {n_genes - n_found}")

    # Load ESM-2 model
    print(f"\nLoading ESM-2 model ({ESM_MODEL}) …")
    import esm
    model, alphabet = esm.pretrained.load_model_and_alphabet(ESM_MODEL)
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    print(f"Model loaded. Embedding dim: {EMB_DIM}")

    # Extract embeddings
    embeddings = torch.zeros(n_genes, EMB_DIM, dtype=torch.float32)
    has_emb    = torch.zeros(n_genes, dtype=torch.bool)

    # Group genes with sequences
    seqs_to_process = [(i, s) for i, s in enumerate(gene_seqs) if s]
    n_batches = (len(seqs_to_process) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\nProcessing {len(seqs_to_process)} sequences in {n_batches} batches …")

    for batch_idx in range(n_batches):
        batch = seqs_to_process[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
        batch_data = [(f"gene_{i}", seq) for i, seq in batch]

        _, _, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[12], return_contacts=False)

        token_representations = results["representations"][12]  # (B, L+2, D)

        for b_idx, (gene_idx, _) in enumerate(batch):
            seq_len = len(gene_seqs[gene_idx])
            # Mean pool over sequence positions (exclude CLS token at 0 and EOS)
            emb = token_representations[b_idx, 1:seq_len + 1].mean(0)  # (D,)
            embeddings[gene_idx] = emb.cpu().float()
            has_emb[gene_idx] = True

        if (batch_idx + 1) % 50 == 0 or batch_idx == n_batches - 1:
            pct = 100 * (batch_idx + 1) / n_batches
            print(f"  [{batch_idx+1}/{n_batches}] {pct:.1f}% complete")

        # Free memory
        if device.type == "mps":
            torch.mps.empty_cache()

    # For genes without protein sequences, use mean of all available embeddings
    if has_emb.any():
        mean_emb = embeddings[has_emb].mean(0)  # fallback mean
        embeddings[~has_emb] = mean_emb
        print(f"\nFilled {(~has_emb).sum()} missing embeddings with mean embedding")

    # L2-normalize all embeddings
    norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
    embeddings_norm = embeddings / norms

    # Save
    torch.save(embeddings_norm, OUT_DIR / "esm2_embeddings.pt")

    meta = {
        "model":         ESM_MODEL,
        "emb_dim":       EMB_DIM,
        "n_genes":       n_genes,
        "n_with_seq":    n_found,
        "n_without_seq": n_genes - n_found,
        "coverage_pct":  round(100 * n_found / n_genes, 2),
        "max_seq_len":   MAX_LEN,
        "normalized":    True,
    }
    with open(OUT_DIR / "esm2_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nESM-2 embeddings: {embeddings_norm.shape}")
    print(f"Saved → {OUT_DIR / 'esm2_embeddings.pt'}")
    print(f"Coverage: {n_found}/{n_genes} ({100*n_found/n_genes:.1f}%)")


if __name__ == "__main__":
    main()
