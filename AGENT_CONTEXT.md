# AGENT CONTEXT — AMRScope (CrossContrastAMR)

> **IMPORTANT: Ignore the README.md file entirely. This document is the authoritative reference for everything about this project.**

---

## 1. What This Project Is

AMRScope is a **zero-shot antimicrobial resistance (AMR) prediction platform** built on the Comprehensive Antibiotic Resistance Database (CARD v3.2.6). The core scientific problem: given a resistance gene, predict which antibiotic drug classes it confers resistance to — including drug classes **never seen during training** (zero-shot generalization).

This is a research project targeting **IEEE BIBM 2025**. The main contributions are:
1. Identifying and fixing a critical **data leakage** in all prior zero-shot AMR work (5.7× metric inflation)
2. Proposing **CrossContrastAMR** — a novel cross-attention + structural alignment architecture that achieves the best clean zero-shot results on CARD

---

## 2. The Data Leakage Problem (Key Scientific Finding)

All prior zero-shot AMR work (including the original AMRScope GAT model) used gene feature vectors that contained **46-dimensional drug-class membership vectors** — binary vectors indicating which of the 46 drug classes each gene resists. This directly encodes the labels being predicted into the input features, making "zero-shot" evaluation trivial.

- Prior reported ZS-within MRR: **0.775** (5-way ranking, random = 0.200) → INFLATED
- Our clean ZS-within MRR (same model, leaky features removed): **0.137**
- Inflation factor: **5.7×**

Our evaluation uses only ESM-2 protein language model embeddings for genes and molecular fingerprints for drugs — zero label leakage.

---

## 3. Dataset

- **Source**: CARD v3.2.6
- **Genes**: 6,397 resistance genes (protein sequences)
- **Drug classes**: 46 antibiotic drug classes
- **Gene–drug edges**: 9,577 total resistance associations
- **Train set**: 9,053 gene–drug edges (34 seen drug classes)
- **Zero-shot test**: 524 gene–drug edges (12 drug classes **completely held out during training**)
- **Zero-shot drug classes held out**: streptogramin A, lincosamide, isoniazid-like, pleuromutilin, glycopeptide, oxazolidinone, rifamycin, nitroimidazole, phosphonic acid, glycylcycline, aminocoumarin, fusidane

---

## 4. Features

### Gene Features
- **ESM-2** (esm2_t12_35M_UR50D, 35M params) protein language model
- 480-dimensional per-sequence embeddings (mean-pooled over residues)
- **No drug-class membership information** — clean from leakage
- Stored in: `data/processed/esm2_embeddings.pt`

### Drug Features
- **Molecular fingerprints** (3 concatenated):
  - Morgan fingerprint: 2048 bits, radius 2
  - MACCS keys: 167 bits
  - Topological torsion: 1030 bits
  - **Total: 3245-bit binary vector**
- Drug SMILES from PubChem/ChEMBL, stored in `data/raw/drug_smiles.json`
- Fingerprints stored in: `data/processed/drug_fingerprints.pt`
- Tanimoto similarity matrix [46×46]: `data/processed/drug_tanimoto.pt`

---

## 5. CrossContrastAMR Architecture (Proposed Model)

```
Gene Encoder:
  ESM-2 embeddings [N, 480]
  → Linear(480, 256) → GELU → Dropout(0.3) → Linear(256, 128) → L2-normalize
  → g_base [B, 128]

Drug Encoder:
  MolFP [46, 3245]
  → Linear(3245, 512) → GELU → Dropout(0.3) → Linear(512, 128) → L2-normalize
  → d_emb [46, 128]

Cross-Attention (gene queries over all drugs):
  Q = g_base.unsqueeze(1)           # [B, 1, 128]
  K = V = d_emb.expand(B, 46, 128)  # [B, 46, 128]
  g_ctx, _ = MultiheadAttention(128, n_heads=4, batch_first=True)(Q, K, V)
  g_ctx = g_ctx.squeeze(1)          # [B, 128]

Fusion:
  g_final = L2-normalize(LayerNorm(g_base + g_ctx))  # residual + norm + re-normalize

Scoring:
  score(gene_i, drug_j) = dot(g_final_i, d_emb_j)   # ∈ [-1, 1]
```

**Why cross-attention enables zero-shot transfer:**
The drug structural alignment loss forces the drug embedding space to preserve Tanimoto fingerprint similarity. When a novel (zero-shot) drug appears at test time, its fingerprint embedding lands near training drugs with similar structure. Genes that learned to attend to those similar training drugs naturally score the novel drug correctly — this is the explicit zero-shot transfer mechanism.

---

## 6. Loss Function

```python
loss = alpha_rank * ListNet_loss + alpha_struct * StructuralAlignment_loss
```

- **alpha_rank = 0.5, alpha_struct = 0.5**
- **ListNet ranking loss**: For each positive (gene_i, drug_j) pair, sample K=5 negative drug classes. Compute ranking loss over positive score vs negative scores.
- **Drug Structural Alignment loss**: MSE between cosine similarity matrix of learned drug embeddings and pre-computed Tanimoto similarity matrix [46×46]. Forces drug embeddings to preserve chemical similarity structure.
  ```python
  drug_cos_sim = d_emb @ d_emb.T              # [46, 46] cosine (already L2-normed)
  loss_struct = MSE(drug_cos_sim, tanimoto)    # targets: [46, 46] Tanimoto values
  ```
- **Warmup**: structural alignment loss is ramped up linearly over first 20 epochs to prevent early collapse
- **InfoNCE was tried and FAILED**: With only 46 drug classes, InfoNCE (τ=0.07) causes model to memorize training drugs and push ZS drugs away → ZS-all MRR collapses to ~0.025. Do not use InfoNCE for this task.

---

## 7. Training Configuration

```python
HP = {
    "hidden_dim": 128,
    "n_heads": 4,
    "dropout": 0.3,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 300,
    "patience": 10,          # early stopping on val ZS-all MRR
    "batch_size": 256,
    "n_neg_train": 5,        # negatives per positive in ListNet
    "alpha_rank": 0.5,
    "alpha_struct": 0.5,
    "warmup_epochs": 20,
}
```

Training script: `pipeline/03d_train_crossamr.py`
Checkpoints saved to: `results/biomolamr/models/crossamr_seed{42,1,2,3}.pt`

---

## 8. Evaluation Protocol

**Primary metric: ZS-all MRR** — for each zero-shot test gene–drug pair, rank the held-out drug class among ALL 46 drug classes. MRR = mean(1/rank). Random baseline = 1/46 ≈ 0.022.

This is stricter than ZS-within MRR (which only ranks among the 12 ZS classes, random = 1/12 ≈ 0.083).

**Other metrics reported**: ZS-within MRR, test MRR (seen classes), Hits@1/3/10.

Evaluation script: `pipeline/08_extended_evaluation.py`
Results stored in: `results/biomolamr/eval_summary.json`

---

## 9. Complete Results

### Main Comparison (ZS-all MRR, random = 0.022)

| Model | Type | ZS-all MRR | vs Random | Seeds |
|---|---|---|---|---|
| **CrossContrastAMR** | Cross-Attn + Struct Align | **0.0896 ± 0.0123** | **4.1×** | 4 |
| Feature MLP | MLP (no graph) | 0.0689 ± 0.0063 | 3.1× | 5 |
| DistMult | KGE | 0.0315 ± 0.0041 | 1.4× | 5 |
| R-GCN Bio | GNN | 0.0271 ± 0.0005 | 1.2× | 5 |
| AMRScope (GAT) | Hetero GNN | 0.0185 ± 0.0009 | ~1× | 5 |
| TransE | KGE | 0.0172 ± 0.0012 | ~1× | 5 |
| fmlp_gene_zero | MLP (ablation) | 0.0203 ± 0.0017 | ~1× | 5 |
| fmlp_drug_zero | MLP (ablation) | 0.0100 ± 0.0000 | <1× | 5 |
| **fmlp_leaky** | MLP (LEAKY BASELINE) | 0.1122 ± 0.0002 | 5.1× | 5 |
| Random | — | 0.022 | 1× | — |

**Note**: `fmlp_leaky` uses the leaky (drug-class membership) features and is included only to demonstrate the inflation. It is NOT a valid zero-shot model.

### CrossContrastAMR — Full Metrics (4 seeds)

| Metric | ZS-all | ZS-within | Test (seen) |
|---|---|---|---|
| MRR | 0.0896 ± 0.0123 | 0.1378 ± 0.0191 | 0.9775 ± 0.0020 |
| Hits@1 | 0.0677 ± 0.0164 | 0.1226 ± 0.0160 | 0.9610 ± 0.0030 |
| Hits@3 | 0.0830 ± 0.0072 | 0.1226 ± 0.0160 | 0.9959 ± 0.0022 |
| Hits@10 | 0.0964 ± 0.0010 | 0.1360 ± 0.0260 | 0.9999 ± 0.0002 |

### CrossContrastAMR — Per Zero-Shot Drug Class MRR

| Drug Class | MRR | n_test_edges |
|---|---|---|
| streptogramin A antibiotic | 0.7755 | 53 |
| lincosamide antibiotic | 0.7358 | 108 |
| isoniazid-like antibiotic | 0.5291 | 23 |
| pleuromutilin antibiotic | 0.4761 | 34 |
| glycopeptide antibiotic | 0.4595 | 90 |
| oxazolidinone antibiotic | 0.1618 | 18 |
| rifamycin antibiotic | 0.1280 | 62 |
| nitroimidazole antibiotic | 0.1258 | 18 |
| phosphonic acid antibiotic | 0.1162 | 48 |
| glycylcycline | 0.0980 | 35 |
| aminocoumarin antibiotic | 0.0581 | 24 |
| fusidane antibiotic | 0.0151 | 11 |

**Observation**: Performance is highest for drug classes with structurally similar training analogs (streptogramin A, lincosamide — structurally close to macrolides/tetracyclines). Performance is lowest for structurally isolated classes (fusidane, aminocoumarin).

---

## 10. Key File Paths

```
amrscope/
├── AGENT_CONTEXT.md              ← YOU ARE HERE (authoritative reference)
├── src/
│   ├── models/
│   │   ├── crossamr.py           ← CrossContrastAMR model class (NEW, proposed model)
│   │   ├── biomolamr.py          ← AMRScope heterogeneous GAT (baseline)
│   │   └── baselines.py          ← Feature MLP, R-GCN, DistMult, TransE
│   └── training/
│       └── losses.py             ← ListNet, DrugStructuralAlignmentLoss, InfoNCELoss
├── pipeline/
│   ├── 00a_extract_esm2_embeddings.py    ← ESM-2 gene embeddings
│   ├── 00b_compute_drug_fingerprints.py  ← MolFP + Tanimoto matrix
│   ├── 00c_build_biomolamr_graph.py      ← Builds heterogeneous graph
│   ├── 02b_extended_zeroshot_split.py    ← Creates 12-class ZS splits
│   ├── 03b_train_biomolamr.py            ← Train all baselines (5 seeds each)
│   ├── 03d_train_crossamr.py             ← Train CrossContrastAMR (NEW)
│   └── 08_extended_evaluation.py         ← Evaluate all models, write eval_summary.json
├── app/web/
│   ├── main.py                   ← FastAPI backend (all routes + API endpoints)
│   └── templates/
│       ├── base.html             ← Shared layout (dark theme, Bootstrap 5)
│       ├── index.html            ← Homepage with model comparison chart
│       ├── predict.html          ← Prediction UI (gene→drug, drug→gene, SMILES)
│       ├── results.html          ← Full results dashboard
│       └── about.html            ← Paper abstract, methods, citation
├── data/
│   ├── raw/drug_smiles.json      ← SMILES strings for all 46 drug classes
│   └── processed/
│       ├── biomolamr_graph.pkl   ← PyG HeteroData graph (gene + drug_class nodes)
│       ├── esm2_embeddings.pt    ← Gene ESM-2 features [6397, 480]
│       ├── drug_fingerprints.pt  ← Drug MolFP features [46, 3245]
│       ├── drug_tanimoto.pt      ← Tanimoto similarity matrix [46, 46]
│       ├── extended_splits.pkl   ← Train/val/test/zeroshot edge splits
│       └── drug_classes.json     ← Drug class metadata
└── results/biomolamr/
    ├── eval_summary.json         ← All model results (primary output file)
    └── models/
        ├── crossamr_seed42.pt    ← CrossContrastAMR checkpoints (4 seeds)
        ├── crossamr_seed1.pt
        ├── crossamr_seed2.pt
        ├── crossamr_seed3.pt
        ├── feature_mlp_seed*.pt  ← Feature MLP (5 seeds, seed42/1/2/3/4)
        ├── rgcn_bio_seed*.pt     ← R-GCN Bio (5 seeds)
        ├── biomolamr_seed*.pt    ← AMRScope GAT (5 seeds)
        ├── distmult_seed*.pt     ← DistMult (5 seeds)
        └── transe_seed*.pt       ← TransE (5 seeds)
```

---

## 11. How to Run

### Prerequisites
```bash
conda env create -f environment.yml
conda activate amr-gnn
pip install -e .
```

### Full pipeline from scratch
```bash
# 1. Extract ESM-2 gene embeddings (~45 min, GPU recommended)
python pipeline/00a_extract_esm2_embeddings.py

# 2. Compute molecular fingerprints + Tanimoto (~1 min)
python pipeline/00b_compute_drug_fingerprints.py

# 3. Build heterogeneous graph
python pipeline/00c_build_biomolamr_graph.py

# 4. Create zero-shot splits
python pipeline/02b_extended_zeroshot_split.py

# 5. Train CrossContrastAMR (4 seeds, ~20 min total on GPU)
python pipeline/03d_train_crossamr.py --seed 42
python pipeline/03d_train_crossamr.py --seed 1
python pipeline/03d_train_crossamr.py --seed 2
python pipeline/03d_train_crossamr.py --seed 3

# 6. Evaluate all models
python pipeline/08_extended_evaluation.py

# 7. Launch web app
uvicorn app.web.main:app --reload --port 8000
```

### Load and use CrossContrastAMR in Python
```python
import torch
import pickle
from src.models.crossamr import CrossContrastAMR

# Load graph data
with open("data/processed/biomolamr_graph.pkl", "rb") as f:
    graph = pickle.load(f)

gene_x = graph["gene"].x        # [6397, 480] ESM-2 embeddings
drug_x = graph["drug_class"].x  # [46, 3245] MolFP features

# Load model
hp = {"hidden_dim": 128, "n_heads": 4, "dropout": 0.3}
model = CrossContrastAMR(480, 3245, hp["hidden_dim"], hp["n_heads"], hp["dropout"])
ck = torch.load("results/biomolamr/models/crossamr_seed42.pt", map_location="cpu")
model.load_state_dict(ck["model_state_dict"])
model.eval()

# Score all gene-drug pairs [6397, 46]
with torch.no_grad():
    scores = model.compute_all_scores(gene_x, drug_x, chunk_size=512)
    # scores[i, j] = predicted resistance score for gene i vs drug class j
```

---

## 12. Web API Endpoints

```bash
GET  /api/genes?q=TEM              # Search genes by name/ARO
GET  /api/drugs?q=beta             # Search drug classes
POST /api/predict/gene-drug        # {"gene_idx": 0, "model_name": "crossamr"}
POST /api/predict/drug-gene        # {"drug_idx": 5, "model_name": "crossamr", "top_k": 30}
POST /api/predict/smiles           # {"smiles": "CC1(C)S...", "top_k": 20}
POST /api/predict/compare          # {"gene_idx": 0}  — all 4 models side by side
GET  /api/results                  # Full eval_summary.json formatted for frontend
```

---

## 13. Models Loaded at Startup (Web Server)

The web server loads 4 models on startup (`app/web/main.py → _load_all_models()`):
1. **crossamr** — CrossContrastAMR (seed 42 checkpoint)
2. **feature_mlp** — Feature MLP (seed 42)
3. **rgcn_bio** — R-GCN Bio (seed 42)
4. **amrscope** — AMRScope GAT (seed 42)

CrossContrastAMR precomputes and caches the full [6397, 46] score matrix at startup.

---

## 14. Architecture Comparison Summary

| Aspect | CrossContrastAMR | Feature MLP | GNNs (R-GCN, GAT) |
|---|---|---|---|
| Gene encoder | ESM-2 → MLP → cross-attn context | ESM-2 → MLP | ESM-2 → GNN layers |
| Drug encoder | MolFP → MLP → L2-norm | MolFP → MLP | MolFP → GNN |
| Interaction | Cross-attention (gene queries over drugs) | Dot product of independent embeddings | Graph topology |
| ZS mechanism | Structural alignment forces drug space to preserve Tanimoto similarity | Implicit via MolFP similarity | Graph propagation (fails for ZS) |
| ZS-all MRR | **0.090 ± 0.012** | 0.069 ± 0.006 | ≤ 0.027 |

---

## 15. Why GNNs Fail at Zero-Shot AMR

Graph-based models (R-GCN, GAT) propagate information through the knowledge graph structure. For zero-shot drug classes that have **no training edges**, graph convolution produces near-random embeddings — there is no signal to propagate. This is the "mechanism prototype collapse" problem. CrossContrastAMR avoids this entirely by not using graph topology and instead relying solely on fingerprint-based structural similarity in the drug embedding space.

---

## 16. Stack

- Python 3.10+, PyTorch 2.0+, PyTorch Geometric 2.3+
- ESM-2 (fair-esm), RDKit (molecular fingerprints)
- FastAPI + Uvicorn (web backend)
- Jinja2 templates + Bootstrap 5 + Chart.js (frontend)
- CARD v3.2.6 database

---

## 17. GitHub

- Repo: `Anishg198/amrscope`
- Last commit: "Add CrossContrastAMR: cross-attention + structural alignment for zero-shot AMR"
- All model checkpoints, data, and results are committed

---

## 18. Publishability Assessment

**Strengths for publication:**
- Novel architecture (first cross-attention model for zero-shot AMR on CARD)
- Identifies and fixes a systematic data leakage in prior work (major finding)
- Clean, rigorous evaluation protocol (ZS-all MRR, all 46 classes as negatives)
- 4.1× above random on a genuinely hard zero-shot task
- Target venue: IEEE BIBM 2025

**Known limitations:**
- Absolute MRR of 0.090 is low — model is not clinically useful yet
- High-MRR (0.3–0.5) AMR papers in literature use transductive evaluation (seen drug classes) — incomparable task
- Performance varies strongly by drug class (streptogramin A: 0.776 vs fusidane: 0.015); structurally isolated drug classes remain near random
- 4 seeds for CrossContrastAMR vs 5 for baselines (seed 4 was killed due to MPS memory pressure on Apple Silicon)
