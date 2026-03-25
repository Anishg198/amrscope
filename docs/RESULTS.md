# Results

This document will contain experimental results once models are trained on actual data.

## Placeholder for Results

After downloading CARD and PATRIC data and training models, results will be documented here including:

### 1. Link Prediction Performance

**Table: Comparison of Models on Link Prediction Task**

| Model | MRR | Hits@1 | Hits@3 | Hits@10 | AUC | Mean Rank |
|-------|-----|--------|--------|---------|-----|-----------|
| GCN | TBD | TBD | TBD | TBD | TBD | TBD |
| GraphSAGE | TBD | TBD | TBD | TBD | TBD | TBD |
| GAT | TBD | TBD | TBD | TBD | TBD | TBD |
| Heterogeneous GNN | TBD | TBD | TBD | TBD | TBD | TBD |
| **Explainable GAT (Ours)** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

### 2. Mechanism Classification Performance

**Table: Multi-Class Classification Results**

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Explainable GAT | TBD | TBD | TBD | TBD |

### 3. Statistical Significance

**Table: Paired t-test Results (p-values)**

| Comparison | p-value | Significant? |
|------------|---------|--------------|
| Ours vs. GCN | TBD | TBD |
| Ours vs. GraphSAGE | TBD | TBD |
| Ours vs. GAT | TBD | TBD |
| Ours vs. Hetero GNN | TBD | TBD |

### 4. Ablation Studies

**Table: Impact of Model Components**

| Variant | MRR | Hits@10 | Notes |
|---------|-----|---------|-------|
| Full Model | TBD | TBD | All components |
| - Multi-task | TBD | TBD | Without mechanism classification |
| - Attention | TBD | TBD | Standard aggregation |
| - Heterogeneity | TBD | TBD | Homogeneous graph |

### 5. Attention Analysis

#### Top Attention Pathways

Examples of high-attention pathways for correct predictions:

1. **Gene X → Drug Y**
   - Attention Path 1: Gene → Mechanism → Drug (weight: TBD)
   - Attention Path 2: Gene → Species → Drug (weight: TBD)
   - Attention Path 3: Gene → Protein → Drug (weight: TBD)

### 6. Case Studies

#### Novel Predictions Validated by Literature

**Case 1: Gene-Drug Pair Example**
- Prediction: TBD
- Confidence: TBD
- Validation: TBD
- Attention Explanation: TBD

### 7. Computational Performance

**Table: Training and Inference Time**

| Model | Training Time | Inference Time | Parameters | GPU Memory |
|-------|---------------|----------------|------------|------------|
| GCN | TBD | TBD | TBD | TBD |
| GraphSAGE | TBD | TBD | TBD | TBD |
| GAT | TBD | TBD | TBD | TBD |
| Heterogeneous GNN | TBD | TBD | TBD | TBD |
| Explainable GAT | TBD | TBD | TBD | TBD |

### 8. Visualizations

Visualizations will be added here after training:

- Training curves (loss, MRR, Hits@k over epochs)
- Attention weight distributions
- Confusion matrices for mechanism classification
- ROC curves
- Precision-Recall curves
- Attention heatmaps for example predictions

### 9. Error Analysis

#### Common Error Patterns

Analysis of false positives and false negatives:

- **False Positives**: TBD
- **False Negatives**: TBD
- **Root Causes**: TBD

### 10. Cross-Validation Results

**Table: 5-Fold Cross-Validation Results**

| Fold | MRR | Hits@10 | AUC |
|------|-----|---------|-----|
| 1 | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD |
| 5 | TBD | TBD | TBD |
| **Mean ± Std** | **TBD ± TBD** | **TBD ± TBD** | **TBD ± TBD** |

---

## How to Generate Results

1. **Download Data**
   ```bash
   # See docs/DATA_DESCRIPTION.md for download instructions
   ```

2. **Preprocess Data**
   ```bash
   # Run data exploration notebook
   jupyter notebook notebooks/01_data_exploration.ipynb
   ```

3. **Train Models**
   ```bash
   # Train baseline models
   python train.py --model gcn --exp-name gcn_baseline
   python train.py --model sage --exp-name sage_baseline
   python train.py --model gat --exp-name gat_baseline

   # Train proposed model
   python train.py --model explainable_gat --exp-name explainable_gat_main
   ```

4. **Evaluate and Visualize**
   ```bash
   # Run results analysis notebook
   jupyter notebook notebooks/04_results_analysis.ipynb
   ```

5. **Generate Figures**
   ```bash
   # Figures will be saved to results/figures/
   ```

## Notes

Results will be updated as experiments are completed. Expected timeline:

- [ ] Data preprocessing (1-2 days)
- [ ] Baseline model training (2-3 days)
- [ ] Proposed model training (3-4 days)
- [ ] Evaluation and analysis (1-2 days)
- [ ] Visualization and documentation (1 day)

**Total estimated time**: 1-2 weeks for complete experimental results.
