# Changelog

All notable changes to the AMR-GNN project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-21

### Added
- Initial project structure and documentation
- Data preprocessing modules:
  - CARD database parser (`src/data_preprocessing/card_parser.py`)
  - PATRIC/BV-BRC database parser (`src/data_preprocessing/patric_parser.py`)
  - Heterogeneous graph builder (`src/data_preprocessing/graph_builder.py`)
- Model implementations:
  - Base GNN classes (`src/models/base_gnn.py`)
  - GCN baseline model (`src/models/gcn_model.py`)
  - GraphSAGE baseline model (`src/models/graphsage_model.py`)
  - GAT baseline model (`src/models/gat_model.py`)
  - Heterogeneous GNN (`src/models/heterogeneous_gnn.py`)
  - Explainable Heterogeneous GAT - main contribution (`src/models/explainable_gat.py`)
- Training infrastructure:
  - Configuration management (`src/training/config.py`)
  - Trainer with multi-task learning (`src/training/trainer.py`)
- Evaluation modules:
  - Link prediction metrics (MRR, Hits@k, AUC) (`src/evaluation/metrics.py`)
  - Classification metrics (`src/evaluation/metrics.py`)
- Visualization tools:
  - Attention weight visualization (`src/visualization/attention_viz.py`)
  - Training curves and metric comparisons (`src/visualization/results_plots.py`)
- Utilities:
  - Early stopping callback (`src/utils/early_stopping.py`)
  - Logger setup (`src/utils/logger.py`)
- Documentation:
  - Comprehensive project documentation (`docs/PROJECT_DOCUMENTATION.md`)
  - Detailed dataset description (`docs/DATA_DESCRIPTION.md`)
  - Methodology and mathematical formulation (`docs/METHODOLOGY.md`)
  - Literature review with citations (`docs/LITERATURE_REVIEW.md`)
- Example notebooks:
  - Data exploration notebook (`notebooks/01_data_exploration.ipynb`)
- Training scripts:
  - Main training script with CLI (`train.py`)
- Configuration files:
  - Example configuration for explainable GAT (`experiments/configs/explainable_gat_config.yaml`)
  - Python dependencies (`requirements.txt`)
  - Conda environment (`environment.yml`)

### Documentation
- README with quick start guide
- Comprehensive API documentation
- Example usage notebooks
- Configuration examples

### Features
- Heterogeneous graph construction from CARD and PATRIC databases
- Multi-entity support (genes, drugs, species, proteins, mechanisms)
- Relation-type-specific attention mechanisms
- Multi-task learning (link prediction + mechanism classification)
- Explainability through attention weight extraction
- Standard evaluation metrics (MRR, Hits@k, AUC, F1)
- Model checkpointing and loading
- Early stopping with validation monitoring
- Mixed precision training support
- Configurable hyperparameters via YAML
- Visualization of attention weights and training curves

## [Unreleased]

### Planned Features
- Temporal graph extensions for tracking resistance emergence
- Integration with OpenBioLink benchmark
- SHAP value computation for feature importance
- Cross-validation support
- Hyperparameter optimization (Bayesian/Random search)
- Additional baseline models (R-GCN, CompGCN)
- Interactive visualization dashboard
- REST API for model serving
- Docker container for reproducibility
- Pre-trained model checkpoints
- Additional evaluation notebooks
- Web interface for predictions

### Future Enhancements
- Transfer learning from general biomedical KGs
- Few-shot learning for rare resistance mechanisms
- Active learning for experimental prioritization
- Federated learning for privacy-preserving training
- Integration with protein structure data
- Chemical descriptor features for drugs
- Multi-species transfer learning
- Causal inference for mechanism discovery

## Version History

### Version 0.1.0 (Initial Release)
- Core functionality implemented
- Basic models and training pipeline
- Documentation and examples
- Ready for initial experiments

---

## Notes

### Breaking Changes
- None yet (initial release)

### Deprecations
- None yet (initial release)

### Known Issues
- Large PATRIC feature file requires significant memory
- Attention visualization limited to single predictions
- No GPU memory optimization for very large graphs

### Contributors
- AMR-GNN Research Team

### License
- MIT License (see LICENSE file)
