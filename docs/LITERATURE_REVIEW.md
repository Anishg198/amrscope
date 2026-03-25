# Literature Review: AMR Prediction using Graph Neural Networks

## Overview

This document summarizes recent research (2024-2026) on antimicrobial resistance prediction using machine learning, with focus on graph-based methods.

## 1. Graph Neural Networks for AMR

### 1.1 Recent Work (2024-2026)

**Zhang et al. (2024): GNN for MIC Prediction in Salmonella**
- **Reference**: Leveraging Graph Neural Networks for MIC Prediction
- **Contribution**: Applied GNNs to predict minimum inhibitory concentrations
- **Dataset**: Salmonella genomes with MIC values
- **Performance**: Improved prediction accuracy over traditional ML
- **Limitation**: Single organism, homogeneous graph

**MFAGCN (2025): Molecular Fingerprints + Graph Attention**
- **Reference**: A machine learning method for predicting molecular antimicrobial activity
- **Contribution**: Combined molecular fingerprints with graph convolutional networks
- **Innovation**: Multi-view learning (structure + sequence)
- **Performance**: State-of-the-art on molecular AMR datasets
- **Limitation**: Molecule-centric, not genome-centric

**MSDeepAMR (2024): Multi-Species AMR Prediction**
- **Reference**: MSDeepAMR: antimicrobial resistance prediction
- **Contribution**: Multi-species model using deep learning
- **Dataset**: Multiple bacterial species
- **Performance**: Good generalization across species
- **Limitation**: Feature-based, not graph-based

### 1.2 Knowledge Graph Approaches

**E. coli ARG Discovery (2024)**
- **Contribution**: Iterative link prediction on AMR knowledge graph
- **Graph Size**: 651,758 triples
- **Discovery**: 6 novel resistance genes validated experimentally
- **Method**: TransE + iterative refinement
- **Key Insight**: Link prediction effective for gene discovery

## 2. Graph Neural Network Architectures

### 2.1 Graph Attention Networks (GAT)

**Veličković et al. (2018): Original GAT Paper**
- **Innovation**: Attention mechanism for graph convolution
- **Advantages**:
  - Adaptive edge weighting
  - Interpretable attention coefficients
  - Superior performance on complex networks
- **Application to AMR**: Natural fit for heterogeneous AMR networks

**Performance on Biomedical Graphs:**
- Outperforms GCN on heterophilic graphs
- Better on sparse, noisy biological networks
- Attention provides explainability

### 2.2 GraphSAGE

**Hamilton et al. (2017): Inductive Graph Learning**
- **Innovation**: Neighborhood sampling and aggregation
- **Advantages**:
  - Inductive learning (generalizes to unseen nodes)
  - Scalable to large graphs
  - Multiple aggregator options
- **Benchmarks**: 48% improvement over GCN on heterophilic datasets
- **Application to AMR**: Good for new gene/drug discovery

### 2.3 Heterogeneous GNNs

**Schlichtkrull et al. (2018): R-GCN**
- **Innovation**: Relation-type-specific transformations
- **Method**: Separate weight matrices per edge type
- **Application**: Knowledge graph completion

**H2GnnDTI (2024): Heterogeneous GNN for Drug-Target Interaction**
- **Innovation**: Dual-channel heterogeneous GNN
- **Performance**: State-of-the-art on drug discovery benchmarks
- **Relevance**: Similar structure to gene-drug prediction

**HGATDVA (2025): Heterogeneous GAT for Disease-Variant Association**
- **Innovation**: Multi-relational attention mechanisms
- **Dataset**: Biomedical knowledge graphs
- **Performance**: Superior to homogeneous methods
- **Key Insight**: Heterogeneity crucial for biological networks

## 3. Explainable AI for AMR

### 3.1 Importance of Explainability

**Clinical Adoption Requirement:**
- Predictions must be interpretable for clinical use
- Mechanism identification critical for treatment decisions
- Regulatory requirements (FDA AI/ML guidelines)

**Recent Work on Explainable AMR (2025)**
- **Reference**: Explainable AI for AMR challenges
- **Finding**: Black-box models hinder clinical adoption
- **Recommendation**: Integrate explainability from design phase

### 3.2 Explainability Techniques

**Attention-Based Explanation:**
- **Advantage**: Built into model architecture
- **Limitation**: May not reflect true importance
- **Application**: Identify important gene-drug pathways

**SHAP (Lundberg & Lee, 2017)**
- **Method**: Shapley values for feature importance
- **Advantage**: Theoretically grounded, model-agnostic
- **Application**: Identify important genomic features

**GNNExplainer (Ying et al., 2019)**
- **Method**: Subgraph explanation via optimization
- **Application**: Extract important subgraphs for predictions

**Integrated Gradients (Sundararajan et al., 2017)**
- **Method**: Path integral of gradients
- **Advantage**: Satisfies axioms of attribution
- **Application**: Feature importance for deep models

### 3.3 Applications to AMR

**Resistance Mechanism Discovery:**
- Attention weights identify novel resistance pathways
- SHAP values highlight critical genetic variants
- Subgraph explanations reveal multi-gene interactions

## 4. Evaluation Standards

### 4.1 Link Prediction Metrics

**Mean Reciprocal Rank (MRR)**
- Standard metric for knowledge graph completion
- Emphasizes top-ranked predictions
- Used in OpenBioLink benchmark

**Hits@k**
- Proportion of correct predictions in top-k
- Common values: k = 1, 3, 10
- Clinically relevant (shortlist for testing)

**Mean Rank (MR)**
- Average rank of correct prediction
- Sensitive to outliers
- Less commonly used

### 4.2 Benchmark Datasets

**OpenBioLink (Breit et al., 2020)**
- **Content**: Biomedical knowledge graph
- **Size**: 4.7M triples, 180K entities
- **Task**: Link prediction
- **Standard**: De facto benchmark for biomedical KGE
- **Evaluation Protocol**: Time-based split to prevent leakage

**Benchmark Best Practices (2023)**
- Negative sampling strategy crucial
- Avoid information leakage in splits
- Report confidence intervals
- Statistical significance testing

## 5. Temporal Graphs

### 5.1 Temporal Networks in Biology

**Longa et al. (2023): Temporal Networks Survey**
- **Finding**: Temporal information improves predictions
- **Methods**: TGN, TGAT, DyRep
- **Application**: Evolution of biological systems

### 5.2 Application to AMR

**Resistance Emergence Tracking:**
- Model temporal spread of resistance genes
- Predict future resistance patterns
- Identify early warning signals

**Challenges:**
- Limited temporal AMR data
- Varying sampling rates
- Missing data imputation

## 6. Datasets

### 6.1 CARD (Comprehensive Antibiotic Resistance Database)

**Alcock et al. (2023): CARD 2023**
- **Content**: 8,582 ontology terms, 6,442 sequences
- **Updates**: Quarterly
- **License**: Open access
- **Advantage**: Curated, high-quality annotations
- **Limitation**: Primarily reference sequences

### 6.2 PATRIC/BV-BRC

**Olson et al. (2023): BV-BRC**
- **Content**: 67,000+ genomes, 500,000+ AMR tests
- **Data Type**: Laboratory AST data
- **Advantage**: Real-world phenotypes
- **Limitation**: Noise, missing data

### 6.3 Other Resources

**NCBI AMRFinderPlus:**
- Resistance gene identification
- Updated monthly

**ResFinder:**
- Web-based resistance gene finder
- Validated against experimental data

**MEGARes:**
- Antimicrobial resistance database for high-throughput sequencing

## 7. Challenges and Open Problems

### 7.1 Data Challenges

**Class Imbalance:**
- Rare resistance mechanisms underrepresented
- Novel genes have limited training data
- Solution: Few-shot learning, meta-learning

**Data Quality:**
- Noisy AMR phenotypes (testing variability)
- Incomplete annotations
- Solution: Robust loss functions, semi-supervised learning

**Data Availability:**
- Proprietary clinical data
- Privacy concerns
- Solution: Federated learning, synthetic data

### 7.2 Modeling Challenges

**Scalability:**
- Large graphs (millions of nodes)
- Computational cost of attention
- Solution: Sampling strategies, efficient attention

**Generalization:**
- New organisms not in training data
- Novel resistance mechanisms
- Solution: Inductive learning, transfer learning

**Interpretability vs. Performance:**
- Tradeoff between accuracy and explainability
- Complex models harder to interpret
- Solution: Post-hoc explanation, inherently interpretable models

### 7.3 Validation Challenges

**Ground Truth:**
- Limited experimental validation
- Expensive to verify predictions
- Solution: Active learning to prioritize experiments

**Clinical Relevance:**
- Lab predictions may not reflect clinical outcomes
- In vitro vs. in vivo differences
- Solution: Integration with clinical data

## 8. Future Directions

### 8.1 Multi-Modal Learning

**Integration of Multiple Data Types:**
- Genomic sequences
- Protein structures
- Chemical properties
- Clinical metadata

**Methods:**
- Multi-view learning
- Graph neural networks with heterogeneous features
- Pre-training on large unlabeled datasets

### 8.2 Foundation Models

**Large-Scale Pre-Training:**
- Pre-train on general biomedical knowledge graphs
- Fine-tune for AMR prediction
- Examples: BioGPT, BioBERT, ESM-2 (proteins)

**Transfer Learning:**
- Leverage knowledge from related tasks
- Reduce data requirements
- Improve generalization

### 8.3 Causal Inference

**Move Beyond Correlation:**
- Identify causal resistance mechanisms
- Interventional predictions
- Counterfactual reasoning

**Methods:**
- Causal graph neural networks
- Do-calculus
- Structural equation models

### 8.4 Active Learning

**Experiment Prioritization:**
- Identify most informative genes to test
- Reduce validation cost
- Accelerate discovery

## 9. Positioning This Work

### 9.1 Novel Contributions

**Heterogeneous Graph Modeling:**
- Multi-entity integration (genes, drugs, species, mechanisms)
- Relation-type-specific attention
- **Gap Filled**: Most work uses homogeneous graphs

**Explainability:**
- Built-in attention mechanisms
- SHAP value integration
- Mechanistic interpretation
- **Gap Filled**: Most models are black boxes

**Multi-Task Learning:**
- Link prediction + mechanism classification
- Joint optimization
- **Gap Filled**: Single-task approaches dominant

### 9.2 Comparison to Related Work

| Work | Graph Type | Explainability | Multi-Task | Benchmark |
|------|------------|----------------|------------|-----------|
| Zhang et al. (2024) | Homogeneous | No | No | Salmonella |
| MFAGCN (2025) | Molecular | Partial | No | Molecules |
| KG-ARG (2024) | Homogeneous KG | No | No | E. coli |
| **This Work** | **Heterogeneous** | **Yes** | **Yes** | **Multi-species** |

### 9.3 Expected Impact

**Scientific Contribution:**
- Novel architecture combining heterogeneity + explainability
- Benchmark for future AMR graph learning research
- Open-source implementation for reproducibility

**Practical Impact:**
- Clinically interpretable predictions
- Discovery of novel resistance mechanisms
- Prioritization for experimental validation

## 10. Key References

### Foundational Papers

1. Veličković et al. (2018). Graph Attention Networks. ICLR.
2. Hamilton et al. (2017). Inductive Representation Learning on Large Graphs. NeurIPS.
3. Schlichtkrull et al. (2018). Modeling Relational Data with Graph Convolutional Networks. ESWC.

### AMR and GNNs

4. Zhang et al. (2024). Leveraging Graph Neural Networks for MIC Prediction. PubMed 40039779.
5. MFAGCN (2025). Machine learning for molecular AMR activity. Nature Scientific Reports.
6. Knowledge Graph ARG Discovery (2024). Frontiers in Microbiology.

### Explainability

7. Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.
8. Ying et al. (2019). GNNExplainer. NeurIPS.
9. Explainable AI for AMR (2025). Frontiers in Microbiology.

### Benchmarks and Evaluation

10. Breit et al. (2020). OpenBioLink: A benchmarking framework for biomedical link prediction. Bioinformatics.
11. Benchmark Best Practices (2023). PMC 7971091.
12. Heterogeneous GNN Benchmarks (2024). PMC 12448810.

### Datasets

13. Alcock et al. (2023). CARD 2023. Nucleic Acids Research.
14. Olson et al. (2023). BV-BRC. Nucleic Acids Research.

### Future Directions

15. Longa et al. (2023). Temporal Networks in Biology. PMC 9803903.
16. Challenges in AI for AMR (2025). PMC 11721440.
