# Project Documentation: AMR Prediction using Graph Neural Networks

## Executive Summary

This project implements a novel heterogeneous graph attention network with explainable mechanisms for predicting antimicrobial resistance (AMR). The research combines:

1. **Data Integration**: CARD and PATRIC/BV-BRC databases
2. **Novel Architecture**: Heterogeneous GAT with relation-specific attention
3. **Explainability**: SHAP values and attention visualization
4. **Comprehensive Evaluation**: Standard metrics with statistical validation

## Research Objectives

### Primary Goals
1. Predict gene-drug resistance relationships using link prediction
2. Classify resistance mechanisms for interpretability
3. Provide clinically actionable explanations for predictions
4. Outperform state-of-the-art baseline methods

### Secondary Goals
1. Discover novel resistance pathways through attention analysis
2. Enable temporal tracking of resistance emergence (future work)
3. Create reproducible, open-source implementation
4. Establish benchmark for future AMR graph learning research

## Technical Architecture

### Graph Schema

**Node Types:**
- `gene`: Antimicrobial resistance genes (from CARD ARO)
- `drug`: Antibiotic compounds
- `species`: Bacterial species (from PATRIC)
- `protein`: Protein sequences
- `mechanism`: Resistance mechanisms (e.g., antibiotic efflux, target alteration)

**Edge Types:**
- `confers_resistance`: gene → drug (primary prediction target)
- `targets`: drug → protein
- `found_in`: gene → species
- `encoded_by`: protein → gene
- `belongs_to`: gene → mechanism

### Model Pipeline

```
Data Sources → Graph Construction → Feature Engineering → GNN Training → Evaluation → Explainability
    ↓                  ↓                    ↓                  ↓              ↓             ↓
  CARD            Heterogeneous         Node/Edge         Multi-task      MRR, Hits@k   Attention
  PATRIC           Multi-relational      Embeddings         Loss                       SHAP values
```

## Implementation Details

### Data Preprocessing

**CARD Parser** (`src/data_preprocessing/card_parser.py`):
- Parses ARO ontology JSON
- Extracts gene sequences and annotations
- Maps resistance mechanisms to ontology terms

**PATRIC Parser** (`src/data_preprocessing/patric_parser.py`):
- Processes AMR phenotype data
- Extracts genomic features
- Links genomes to bacterial species

**Graph Builder** (`src/data_preprocessing/graph_builder.py`):
- Constructs PyTorch Geometric HeteroData object
- Creates node and edge indices
- Generates train/val/test splits (80/10/10)
- Implements negative sampling for link prediction

### Model Implementations

**Baseline Models:**
1. **GCN** (`src/models/gcn_model.py`): Standard graph convolution
2. **GraphSAGE** (`src/models/graphsage_model.py`): Neighbor sampling aggregation
3. **GAT** (`src/models/gat_model.py`): Single-type attention mechanism

**Proposed Model** (`src/models/explainable_gat.py`):
- **Heterogeneous Attention**: Separate attention mechanisms per relation type
- **Node Type Transformations**: Type-specific embedding projections
- **Multi-Task Heads**:
  - Link prediction head (gene → drug)
  - Mechanism classification head
- **Explainability Module**:
  - Attention weight extraction
  - Subgraph explanation generation

### Training Pipeline

**Loss Function:**
```
L = α * L_link_prediction + β * L_classification + γ * L_regularization

where:
  L_link_prediction = Binary Cross-Entropy on link scores
  L_classification = Cross-Entropy on mechanism classes
  L_regularization = L2 penalty on attention weights
```

**Optimization:**
- Adam optimizer with learning rate scheduling
- Gradient clipping for stability
- Early stopping on validation MRR
- Mixed precision training for efficiency

**Negative Sampling:**
- Uniform random negative sampling
- Hard negative mining (high-scoring false edges)
- Balanced positive:negative ratio (1:5)

### Evaluation Protocol

**Metrics:**
1. **Link Prediction**:
   - Mean Reciprocal Rank (MRR)
   - Hits@1, Hits@3, Hits@10
   - Mean Rank (MR)

2. **Classification**:
   - AUC-ROC
   - Precision, Recall, F1-score
   - Confusion matrices per mechanism class

3. **Explainability**:
   - Top-k attention pathways
   - SHAP feature importance rankings
   - Case study validation against literature

**Statistical Validation:**
- 5-fold cross-validation
- Paired t-tests for model comparison
- Bootstrap confidence intervals
- Multiple testing correction (Bonferroni)

## Computational Requirements

**Minimum:**
- RAM: 16 GB
- Storage: 50 GB
- CPU: 8 cores

**Recommended:**
- RAM: 32 GB
- GPU: NVIDIA with 8+ GB VRAM (e.g., RTX 3070, V100)
- Storage: 100 GB SSD
- CPU: 16+ cores

**Training Time Estimates:**
- Data preprocessing: 1-2 hours
- Baseline models: 2-4 hours each
- Proposed model: 8-12 hours
- Total: ~24-48 hours on single GPU

## Reproducibility

### Random Seeds
All experiments use fixed random seeds:
- PyTorch: 42
- NumPy: 42
- Python: 42
- CUDA: deterministic mode enabled

### Version Control
- Git repository with tagged releases
- Docker container for environment isolation
- Requirements pinned to specific versions

### Data Versioning
- Raw data checksums (MD5/SHA256)
- Processed data snapshots with metadata
- Graph statistics logging

## Known Limitations

1. **Data Availability**: PATRIC data requires institutional access
2. **Computational Cost**: Large graphs require significant GPU memory
3. **Temporal Data**: Limited historical AMR data for temporal modeling
4. **Class Imbalance**: Some resistance mechanisms are rare
5. **Evaluation**: Limited ground truth for novel predictions

## Future Extensions

1. **Temporal Dynamics**: Add time-aware graph convolutions
2. **Multi-Modal Features**: Integrate protein structures, chemical descriptors
3. **Transfer Learning**: Pre-training on broader biomedical knowledge graphs
4. **Active Learning**: Prioritize experimental validation targets
5. **Federated Learning**: Privacy-preserving multi-institutional training

## References

See [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md) for comprehensive citations.

## Contact & Support

For technical questions:
1. Check documentation in `docs/`
2. Review example notebooks in `notebooks/`
3. Open GitHub issue with reproducible example
4. Email: [contact information]

## Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.
