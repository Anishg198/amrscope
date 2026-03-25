<div align="center">

# 🦠 BioMolAMR

### Antimicrobial Resistance Prediction Platform

**Predict which resistance genes confer resistance to antibiotic drug classes — including novel drugs never seen during training.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-7952B3?style=flat&logo=bootstrap&logoColor=white)](https://getbootstrap.com)
[![CARD](https://img.shields.io/badge/Database-CARD%20v3.2.6-4CAF50?style=flat)](https://card.mcmaster.ca)

</div>

---

## What is BioMolAMR?

BioMolAMR is a **local web platform** for zero-shot antimicrobial resistance (AMR) prediction. Given a resistance gene, an antibiotic drug class, or even a novel drug SMILES string, it predicts resistance associations — including for antibiotic classes the model has **never seen during training**.

It combines:
- **ESM-2** protein language model embeddings for resistance genes (480-dim)
- **Molecular fingerprints** (Morgan 2048 + MACCS 167 + TopoTorsion 1030 = 3245-bit) for drug chemical structure
- **Three trained models** spanning MLP, relational GCN, and heterogeneous GAT architectures
- Built on **CARD v3.2.6** — 6,397 resistance genes, 46 antibiotic drug classes

---

## Features

| | Feature | Description |
|---|---|---|
| 🔍 | **Gene → Drug** | Search any resistance gene, rank all 46 drug classes by predicted resistance score |
| 💊 | **Drug → Gene** | Type any antibiotic class, find top resistance genes with confidence metrics |
| 🧪 | **Novel SMILES** | Paste any drug SMILES — even unknown drugs — and get resistance gene predictions |
| 📊 | **Model Comparison** | Run all 3 models side-by-side on the same gene, compare rankings instantly |
| ✅ | **Verdict System** | Plain-language confidence verdict with TP recovery stats after every prediction |
| 📈 | **Results Dashboard** | Full evaluation metrics, ablation study tables, and publication figures |

---

## Models

### 🥇 Feature MLP
A 2-layer MLP that scores gene–drug pairs using ESM-2 protein embeddings and molecular fingerprints — no graph. Best zero-shot performance (ZS-all MRR **0.069**, 3.1× random).

### 🔷 R-GCN Bio
Relational Graph Convolutional Network over the full CARD heterogeneous knowledge graph (genes, drug classes, resistance mechanisms). Typed edge convolutions per relation.

### 🔶 BioMolAMR
Heterogeneous Graph Attention Network — gene encoder (ESM-2 → GAT), drug encoder (MolFP → GraphSAGE over Tanimoto similarity graph), mechanism-weighted decoder.

| Model | ZS-all MRR | vs. Random (0.022) |
|---|---|---|
| Feature MLP | **0.069 ± 0.006** | **3.1×** |
| R-GCN Bio | 0.027 ± 0.001 | 1.2× |
| BioMolAMR | 0.019 ± 0.001 | ~1× |

> *ZS-all MRR: rank the zero-shot drug class among all 46 classes. Random baseline = 1/46 ≈ 0.022.*

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/yourusername/biomolamr.git
cd biomolamr

# Conda (recommended)
conda env create -f environment.yml
conda activate amr-gnn

# Or pip
pip install -r requirements.txt
pip install -e .
```

### 2. Build the graph

```bash
# Step 1 — Extract ESM-2 gene embeddings (~45 min, GPU recommended)
python pipeline/00a_extract_esm2_embeddings.py

# Step 2 — Compute molecular drug fingerprints (~1 min)
python pipeline/00b_compute_drug_fingerprints.py

# Step 3 — Build heterogeneous knowledge graph (~2 min)
python pipeline/00c_build_biomolamr_graph.py

# Step 4 — Create zero-shot splits
python pipeline/02b_extended_zeroshot_split.py
```

Or run the full pipeline at once:

```bash
python run_biomolamr_pipeline.py --skip_esm2   # if ESM-2 embeddings already exist
```

### 3. Train models

```bash
# Train all models (5 seeds each — ~2–3 hours)
python pipeline/03b_train_biomolamr.py

# Quick single run (seed 42 only)
python pipeline/03b_train_biomolamr.py --seeds 42
```

### 4. Evaluate

```bash
python pipeline/08_extended_evaluation.py
# → results/biomolamr/eval_summary.json
```

### 5. Launch the web app

```bash
uvicorn app.web.main:app --reload --port 8000
```

Open **[http://localhost:8000](http://localhost:8000)**

---

## Using the App

### Gene → Drug

Search a resistance gene by name or ARO number, select it, hit **Predict**. The model ranks all 46 drug classes by predicted resistance score and returns a confidence verdict.

**Try:** `TEM-1`, `mecA`, `vanA`, `blaCTX-M`, `tetM`, `aac(6')`

### Drug → Gene

Type an antibiotic class name — results auto-trigger on selection.

**Try:** `beta-lactam`, `aminoglycoside`, `tetracycline`, `carbapenem`, `macrolide`, `fluoroquinolone`

### Novel SMILES

Paste any SMILES string. The system computes Morgan + MACCS + TopoTorsion fingerprints and returns the top resistance genes plus a Tanimoto similarity score as a reliability indicator.

**Example SMILES:**

```
# Ampicillin
CC1(C)S[C@@H]2[C@H](NC(=O)[C@@H](N)c3ccccc3)C(=O)N2[C@H]1C(=O)O

# Ciprofloxacin
O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O

# Doxycycline
OC1=C(O)C(=O)[C@@]2(O)C(O)=C3C(=O)c4c(O)cccc4[C@@H](N(C)C)[C@H]3C[C@@H]2[C@@H]1C(N)=O
```

---

## Understanding Results

### Tags

| Tag | Meaning |
|---|---|
| `TP` | **True Positive** — confirmed gene–drug resistance association in CARD |
| `ZS` | **Zero-Shot** — drug class withheld during training; model never saw it |
| `ZS+TP` | **Both** — confirmed resistance to a drug class the model was never trained on |

### Verdict Confidence

| Level | Meaning |
|---|---|
| 🟢 **High** | Strong unambiguous signal — score > 2.0, large margin over #2 |
| 🔵 **Moderate** | Reasonable signal with some uncertainty |
| 🟡 **Low** | Weak signal — treat predictions as exploratory |

For SMILES mode, confidence reflects **Tanimoto similarity** to nearest training drug:
`≥ 0.5` high · `0.25–0.5` moderate · `< 0.25` exploratory

---

## API

All prediction endpoints accept and return JSON.

```bash
# Search genes
GET /api/genes?q=TEM

# Search drug classes
GET /api/drugs?q=beta

# Gene → ranked drugs
POST /api/predict/gene-drug
{"gene_idx": 0, "model_name": "feature_mlp"}

# Drug → top genes
POST /api/predict/drug-gene
{"drug_idx": 5, "model_name": "feature_mlp", "top_k": 30}

# Novel SMILES
POST /api/predict/smiles
{"smiles": "CC1(C)S[C@@H]2...", "top_k": 20}

# Compare all models
POST /api/predict/compare
{"gene_idx": 0}
```

---

## Project Structure

```
biomolamr/
├── app/web/                  # FastAPI web application
│   ├── main.py               # Backend — all routes and API endpoints
│   ├── templates/            # Jinja2 HTML pages
│   └── static/               # CSS + JS
├── src/
│   ├── models/               # Model implementations
│   │   ├── biomolamr.py      # BioMolAMR heterogeneous GAT
│   │   └── baselines.py      # Feature MLP, R-GCN, DistMult, TransE
│   └── training/
│       └── losses.py         # ListNet, structural alignment, focal losses
├── pipeline/                 # Data processing + training scripts
├── data/
│   ├── raw/drug_smiles.json  # SMILES for all 46 CARD drug classes
│   └── processed/            # Graph, embeddings, splits (generated)
├── results/biomolamr/        # Trained checkpoints + eval JSON + figures
├── environment.yml
└── requirements.txt
```

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- RDKit
- fair-esm
- FastAPI + Uvicorn
- 8 GB RAM (models run on CPU)

---

## Data

All gene and resistance data from the **Comprehensive Antibiotic Resistance Database (CARD) v3.2.6**.
Drug SMILES sourced from PubChem and ChEMBL.

---

<div align="center">

PyTorch · PyTorch Geometric · ESM-2 · RDKit · FastAPI · Bootstrap 5 · Chart.js

</div>
