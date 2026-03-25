# Methodology

## Overview

This document describes the methodology for antimicrobial resistance prediction using heterogeneous graph neural networks with explainable attention mechanisms.

## 1. Problem Formulation

### Link Prediction Task

Given a heterogeneous graph $G = (V, E, \mathcal{R})$ where:
- $V$ is the set of nodes of different types (genes, drugs, species, proteins, mechanisms)
- $E$ is the set of edges of different relation types
- $\mathcal{R}$ is the set of relation types

**Objective:** Predict missing edges between gene nodes and drug nodes (resistance relationships).

**Formulation:** For a gene-drug pair $(g_i, d_j)$, predict:
$$
\hat{y}_{ij} = f_\theta(z_g^{(i)}, z_d^{(j)})
$$

where:
- $z_g^{(i)}$ is the embedding of gene $g_i$
- $z_d^{(j)}$ is the embedding of drug $d_j$
- $f_\theta$ is a scoring function (e.g., dot product)
- $\theta$ are learnable parameters

### Multi-Task Learning

Additionally, predict resistance mechanisms for genes:
$$
\hat{m}_i = \text{softmax}(W_m z_g^{(i)} + b_m)
$$

where $\hat{m}_i$ is the predicted mechanism distribution for gene $g_i$.

## 2. Model Architecture

### 2.1 Heterogeneous Graph Attention Network

**Input Layer:**
- Node features: $X_v \in \mathbb{R}^{|V_v| \times d_{in}^v}$ for each node type $v$
- Edge indices: $E_r$ for each relation type $r \in \mathcal{R}$

**Type-Specific Transformations:**
For each node type $v$:
$$
H_v^{(0)} = \sigma(X_v W_{in}^v)
$$

where $W_{in}^v \in \mathbb{R}^{d_{in}^v \times d_{hidden}}$ is a type-specific linear transformation.

**Heterogeneous Attention Layers:**
For layer $l$ and relation type $r = (v_{src}, r_{type}, v_{dst})$:

$$
\alpha_{ij}^{(l, r)} = \frac{\exp(\text{LeakyReLU}(a_r^T [W_r^{src} h_i^{(l)} \| W_r^{dst} h_j^{(l)}]))}{\sum_{k \in \mathcal{N}_i^r} \exp(\text{LeakyReLU}(a_r^T [W_r^{src} h_i^{(l)} \| W_r^{dst} h_k^{(l)}]))}
$$

where:
- $\alpha_{ij}^{(l, r)}$ is the attention coefficient from node $i$ to node $j$ via relation $r$
- $a_r$ is a learnable attention vector for relation $r$
- $W_r^{src}, W_r^{dst}$ are relation-specific transformation matrices
- $\|$ denotes concatenation
- $\mathcal{N}_i^r$ is the set of neighbors of node $i$ via relation $r$

**Message Aggregation:**
$$
h_i^{(l+1)} = \sigma\left(\frac{1}{|\mathcal{R}_i|} \sum_{r \in \mathcal{R}_i} \sum_{j \in \mathcal{N}_i^r} \alpha_{ij}^{(l, r)} W_r^{dst} h_j^{(l)}\right)
$$

where $\mathcal{R}_i$ is the set of relation types connected to node $i$.

**Multi-Head Attention:**
Use $K$ independent attention heads and concatenate:
$$
h_i^{(l+1)} = \|_{k=1}^K \sigma\left(\sum_{r \in \mathcal{R}_i} \sum_{j \in \mathcal{N}_i^r} \alpha_{ij}^{(l, r, k)} W_r^{dst, k} h_j^{(l)}\right)
$$

**Residual Connections:**
$$
h_i^{(l+1)} = h_i^{(l)} + h_i^{(l+1)}
$$

### 2.2 Prediction Heads

**Link Prediction:**
$$
s_{ij} = \sigma(z_g^{(i)})^T \sigma(z_d^{(j)})
$$

where $\sigma$ is an optional MLP transformation.

**Mechanism Classification:**
$$
\hat{m}_i = \text{softmax}(MLP(z_g^{(i)}))
$$

## 3. Training

### 3.1 Loss Function

**Multi-Task Loss:**
$$
\mathcal{L} = \alpha \mathcal{L}_{link} + \beta \mathcal{L}_{mech} + \gamma \mathcal{L}_{reg}
$$

**Link Prediction Loss (Binary Cross-Entropy):**
$$
\mathcal{L}_{link} = -\frac{1}{|E_{pos}| + |E_{neg}|} \left[\sum_{(i,j) \in E_{pos}} \log \sigma(s_{ij}) + \sum_{(i,j) \in E_{neg}} \log(1 - \sigma(s_{ij}))\right]
$$

where:
- $E_{pos}$ are positive (observed) edges
- $E_{neg}$ are negative (sampled non-existent) edges

**Mechanism Classification Loss (Cross-Entropy):**
$$
\mathcal{L}_{mech} = -\frac{1}{|V_g|} \sum_{i=1}^{|V_g|} \sum_{c=1}^C y_{ic} \log \hat{m}_{ic}
$$

where:
- $V_g$ is the set of gene nodes with mechanism labels
- $C$ is the number of mechanism classes
- $y_{ic}$ is the true label (one-hot)

**Regularization (Attention Entropy):**
$$
\mathcal{L}_{reg} = \frac{1}{|E|} \sum_{(i,j) \in E} \sum_{r} \alpha_{ij}^r \log \alpha_{ij}^r
$$

This encourages sharper attention distributions.

### 3.2 Negative Sampling

**Strategy:** For each positive edge $(g_i, d_j)$, sample $k$ negative edges:

1. **Uniform Sampling:** Randomly sample drugs $d'$ that are not connected to $g_i$
2. **Hard Negative Mining:** Sample high-scoring negative edges (periodically)

**Ratio:** Use $k = 5$ negative samples per positive edge.

### 3.3 Optimization

**Optimizer:** Adam with learning rate $\eta = 0.001$

**Learning Rate Schedule:** ReduceLROnPlateau
- Factor: 0.5
- Patience: 10 epochs
- Monitor: Validation MRR

**Gradient Clipping:** Clip gradients to norm 1.0

**Early Stopping:**
- Patience: 20 epochs
- Metric: Validation MRR
- Min delta: 0.0001

**Batch Training:**
- Edge mini-batches of size 512
- Full graph message passing (no neighbor sampling)

## 4. Evaluation

### 4.1 Metrics

**Link Prediction:**

1. **Mean Reciprocal Rank (MRR):**
$$
\text{MRR} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{\text{rank}_q}
$$

where $\text{rank}_q$ is the rank of the true positive for query $q$.

2. **Hits@k:**
$$
\text{Hits@k} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \mathbb{I}[\text{rank}_q \leq k]
$$

3. **AUC-ROC:** Area under the ROC curve

**Classification:**

1. **Accuracy, Precision, Recall, F1-score**
2. **Confusion Matrix**

### 4.2 Evaluation Protocol

**Data Splits:**
- Training: 80%
- Validation: 10%
- Test: 10%

**Edge Splitting:**
- Hold out edges (not nodes) for validation and test
- Ensure no information leakage

**Cross-Validation:**
- 5-fold cross-validation for robust evaluation
- Report mean ± std across folds

**Statistical Testing:**
- Paired t-tests for model comparison
- Bonferroni correction for multiple testing
- Significance level: $\alpha = 0.05$

## 5. Explainability

### 5.1 Attention Weight Analysis

For a prediction $(g_i, d_j)$, extract attention weights $\alpha_{ij}^{(l, r)}$ for all layers and relations.

**Top-k Attention Paths:**
Identify the $k$ most important attention paths contributing to the prediction.

### 5.2 SHAP Values

Compute SHAP values for node features to identify important features:

$$
\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} [f(S \cup \{j\}) - f(S)]
$$

where:
- $F$ is the set of all features
- $S$ is a subset of features
- $\phi_j$ is the SHAP value for feature $j$

### 5.3 Subgraph Explanation

Extract k-hop subgraph around gene-drug pair and visualize attention flow.

## 6. Baseline Models

For comparison, implement:

1. **GCN:** Standard graph convolution
2. **GraphSAGE:** Neighborhood sampling and aggregation
3. **GAT:** Single-type graph attention
4. **Heterogeneous GNN:** Without explainability features

All baselines use the same:
- Training procedure
- Hyperparameters (where applicable)
- Evaluation protocol

## 7. Hyperparameters

| Parameter | Value | Search Range |
|-----------|-------|--------------|
| Hidden dim | 256 | [128, 256, 512] |
| Output dim | 128 | [64, 128, 256] |
| Num layers | 3 | [2, 3, 4] |
| Num heads | 8 | [4, 8, 16] |
| Dropout | 0.5 | [0.3, 0.5, 0.7] |
| Attention dropout | 0.2 | [0.1, 0.2, 0.3] |
| Learning rate | 0.001 | [0.0001, 0.001, 0.01] |
| Weight decay | 5e-4 | [1e-5, 5e-4, 1e-3] |
| $\alpha$ (link loss) | 1.0 | [0.5, 1.0, 2.0] |
| $\beta$ (mech loss) | 0.3 | [0.1, 0.3, 0.5] |
| $\gamma$ (reg loss) | 0.01 | [0.001, 0.01, 0.1] |

**Hyperparameter Tuning:**
- Random search or Bayesian optimization
- 50 trials
- Select best based on validation MRR

## 8. Implementation Details

**Framework:** PyTorch + PyTorch Geometric

**Hardware:**
- GPU: NVIDIA RTX 3080 or better
- RAM: 32 GB minimum
- Storage: 100 GB

**Training Time:**
- Baseline models: 2-4 hours
- Proposed model: 8-12 hours
- Total experiments: ~48 hours

**Reproducibility:**
- Fixed random seeds (42)
- Deterministic CUDA operations
- Version-controlled code
- Saved model checkpoints

## References

1. Veličković et al. (2018). Graph Attention Networks. ICLR.
2. Schlichtkrull et al. (2018). Modeling Relational Data with Graph Convolutional Networks. ESWC.
3. Hamilton et al. (2017). Inductive Representation Learning on Large Graphs. NeurIPS.
4. Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.
