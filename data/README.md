# Data Directory

This directory contains raw and processed data for the AMR prediction project.

## Structure

```
data/
├── raw/                    # Original downloaded datasets
│   ├── card/              # CARD database files
│   └── patric/            # PATRIC/BV-BRC database files
├── processed/             # Processed graph data
│   ├── graph.pkl          # Main heterogeneous graph
│   ├── train_split.pkl    # Training data split
│   ├── val_split.pkl      # Validation data split
│   └── test_split.pkl     # Test data split
└── README.md              # This file
```

## Data Download

### CARD Database

1. Visit https://card.mcmaster.ca/download
2. Download the latest data package
3. Extract to `data/raw/card/`

Required files:
- `aro.json` - ARO ontology
- `card.json` - Main database
- `nucleotide_fasta_protein_homolog_model.fasta` - Gene sequences
- `protein_fasta_protein_homolog_model.fasta` - Protein sequences

### PATRIC/BV-BRC Database

1. Visit https://www.bv-brc.org/
2. Download AMR phenotype data
3. Extract to `data/raw/patric/`

Required files:
- `AMR.csv` - AMR phenotype data
- `genome_metadata.csv` - Genome metadata
- `PATRIC_genome_feature.csv` - Feature annotations (large file)

## Usage

```python
from src.data_preprocessing.card_parser import CARDParser
from src.data_preprocessing.patric_parser import PATRICParser
from src.data_preprocessing.graph_builder import HeterogeneousGraphBuilder

# Parse data
card_parser = CARDParser('data/raw/card')
patric_parser = PATRICParser('data/raw/patric')

# Build graph
builder = HeterogeneousGraphBuilder(card_parser, patric_parser)
graph = builder.build()

# Save
builder.save('data/processed/graph.pkl')
```

## Data Citation

If you use these datasets, please cite:

**CARD:**
Alcock et al. (2023). CARD 2023: expanded curation, support for machine learning, and resistome prediction at the Comprehensive Antibiotic Resistance Database. *Nucleic Acids Research*.

**PATRIC/BV-BRC:**
Olson et al. (2023). Introducing the Bacterial and Viral Bioinformatics Resource Center (BV-BRC). *Nucleic Acids Research*.
