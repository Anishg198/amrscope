# Dataset Description

## Overview

This project integrates two primary antimicrobial resistance (AMR) databases:

1. **CARD (Comprehensive Antibiotic Resistance Database)**
2. **PATRIC/BV-BRC (Bacterial and Viral Bioinformatics Resource Center)**

## 1. CARD (Comprehensive Antibiotic Resistance Database)

### Source
- **Website**: https://card.mcmaster.ca/
- **Download**: https://card.mcmaster.ca/download
- **Version**: Latest (updated quarterly)
- **License**: Open access (academic use)

### Contents

**Antibiotic Resistance Ontology (ARO):**
- 8,582+ ontology terms
- Hierarchical classification of resistance genes, mechanisms, and drugs
- Natural graph structure with parent-child relationships

**Reference Sequences:**
- 6,442+ curated resistance gene sequences
- Protein and DNA sequences
- Functional annotations

**SNP Data:**
- 4,480+ single nucleotide polymorphisms
- Associated with resistance phenotypes
- Mapped to reference genomes

### File Structure

```
data/raw/card/
├── aro.json                    # ARO ontology (JSON format)
├── aro_index.tsv              # ARO term index
├── card.json                   # Main database (JSON format)
├── nucleotide_fasta_protein_homolog_model.fasta  # Gene sequences
├── protein_fasta_protein_homolog_model.fasta     # Protein sequences
└── aro_categories.tsv         # Category mappings
```

### Key Fields

**ARO Terms** (`aro.json`):
```json
{
  "ARO:3000026": {
    "accession": "ARO:3000026",
    "name": "NDM-1",
    "description": "New Delhi metallo-beta-lactamase",
    "category": "AMR gene family",
    "parent": ["ARO:3000001"],
    "children": ["ARO:3004405"],
    "resistance_mechanism": "antibiotic inactivation",
    "drug_class": "carbapenem antibiotic"
  }
}
```

### Data Statistics

| Category | Count |
|----------|-------|
| Total ARO terms | 8,582 |
| Gene families | 2,345 |
| Drug classes | 156 |
| Resistance mechanisms | 45 |
| Reference sequences | 6,442 |
| SNPs | 4,480 |

### Download Instructions

```bash
# Create directory
mkdir -p data/raw/card

# Download latest version
cd data/raw/card
wget https://card.mcmaster.ca/latest/data -O card_data.tar.bz2

# Extract
tar -xjf card_data.tar.bz2

# Verify files
ls -lh
```

## 2. PATRIC/BV-BRC Database

### Source
- **Website**: https://www.bv-brc.org/
- **FTP**: ftp://ftp.bvbrc.org/
- **Version**: Current (updated monthly)
- **License**: Open access (cite appropriately)

### Contents

**Genome Data:**
- 67,000+ bacterial genomes with AMR phenotypes
- 40+ genera, 100+ species
- Laboratory-derived AST (Antimicrobial Susceptibility Testing) data

**AMR Phenotype Data:**
- Minimum Inhibitory Concentration (MIC) values
- Resistant/Susceptible classifications
- Testing methods and standards

**Genomic Features:**
- Gene annotations
- Protein families
- Functional categories

### File Structure

```
data/raw/patric/
├── AMR.csv                     # AMR phenotype data
├── genome_metadata.csv         # Genome metadata
├── PATRIC_genome_feature.csv  # Gene annotations
└── specialty_genes.csv        # AMR genes identified
```

### Key Fields

**AMR Phenotype Data** (`AMR.csv`):
```csv
genome_id,genome_name,taxon_id,antibiotic,resistant_phenotype,measurement,measurement_unit,laboratory_typing_method,testing_standard
1234.5,Escherichia coli,562,ampicillin,Resistant,>=32,mg/L,MIC,CLSI
```

**Genome Features** (`PATRIC_genome_feature.csv`):
```csv
genome_id,feature_id,gene,product,protein_id,aa_length,functional_category
1234.5,fig|1234.5.peg.100,blaNDM-1,Beta-lactamase,ABC123,270,Antibiotic Resistance
```

### Data Statistics

| Category | Count |
|----------|-------|
| Total genomes | 67,000+ |
| Genera represented | 40+ |
| Species represented | 100+ |
| AMR phenotype records | 500,000+ |
| Unique antibiotics tested | 150+ |
| Resistance genes annotated | 20,000+ |

### Download Instructions

```bash
# Create directory
mkdir -p data/raw/patric

# Download AMR data
cd data/raw/patric
wget ftp://ftp.bvbrc.org/RELEASE_NOTES/AMR.csv

# Download genome metadata
wget ftp://ftp.bvbrc.org/genomes/genome_metadata.csv

# Download feature annotations (large file ~10GB)
wget ftp://ftp.bvbrc.org/RELEASE_NOTES/PATRIC_genome_feature.csv.gz
gunzip PATRIC_genome_feature.csv.gz

# Verify downloads
md5sum -c checksums.md5
```

## 3. Data Integration Strategy

### Graph Construction

**Node Creation:**
1. **Genes**: From CARD ARO terms and PATRIC specialty genes
2. **Drugs**: From CARD drug classes and PATRIC antibiotics tested
3. **Species**: From PATRIC genome taxonomy
4. **Proteins**: From CARD protein sequences
5. **Mechanisms**: From CARD resistance mechanisms

**Edge Creation:**
1. **confers_resistance**: Gene → Drug
   - Source: CARD ARO relationships + PATRIC AMR phenotypes
   - Filtering: Resistant phenotypes only (exclude susceptible)

2. **targets**: Drug → Protein
   - Source: CARD drug-target relationships
   - Biological knowledge bases

3. **found_in**: Gene → Species
   - Source: PATRIC genome annotations
   - Threshold: Present in genome

4. **encoded_by**: Protein → Gene
   - Source: CARD sequence mappings
   - Direct mapping from gene to protein product

5. **belongs_to**: Gene → Mechanism
   - Source: CARD ARO mechanism classification
   - Hierarchical ontology relationships

### Data Filtering & Quality Control

**Quality Filters:**
1. Remove genomes with <50% assembly quality
2. Exclude AMR tests with "Intermediate" phenotype (focus on R/S)
3. Filter genes with <80% sequence identity to CARD references
4. Remove antibiotics with <100 test records (insufficient data)

**Normalization:**
1. Standardize antibiotic names (synonyms → canonical names)
2. Unify species taxonomy (NCBI Taxonomy IDs)
3. Normalize MIC values to common units (mg/L)

**Train/Val/Test Split:**
- **Strategy**: Temporal split where possible, otherwise random
- **Ratios**: 80% train, 10% validation, 10% test
- **Stratification**: Balance by drug class and species
- **Link Prediction**: Hold out edges (not nodes) for testing

### Data Preprocessing Pipeline

```python
# Pseudocode workflow
1. Load CARD ARO ontology
2. Load PATRIC AMR phenotypes
3. Create node mappings (gene, drug, species, protein, mechanism)
4. Create edge lists for each relation type
5. Generate node features:
   - Gene: Sequence embeddings (ESM-2 or ProtBERT)
   - Drug: Chemical fingerprints (Morgan, MACCS)
   - Species: Taxonomic embeddings
   - Protein: Structure features (if available)
   - Mechanism: One-hot encoding
6. Construct PyTorch Geometric HeteroData object
7. Create train/val/test edge masks
8. Generate negative samples for link prediction
9. Save processed graph
```

## 4. Feature Engineering

### Node Features

**Genes:**
- Sequence length
- GC content
- Protein family ID (Pfam)
- Pre-trained sequence embeddings (768-dim from ESM-2)

**Drugs:**
- Molecular weight
- LogP (lipophilicity)
- Morgan fingerprints (2048-bit)
- Drug class one-hot encoding

**Species:**
- Taxonomic rank features
- Gram staining (positive/negative)
- Pathogenicity score
- Genus/species embeddings

**Proteins:**
- Secondary structure features
- Domain annotations
- Pre-trained protein embeddings

**Mechanisms:**
- Mechanism type one-hot encoding
- Parent mechanism in ARO hierarchy

### Edge Features

**confers_resistance:**
- MIC value (if available)
- Evidence score (literature support)
- Discovery year (for temporal modeling)

**Other edges:**
- Binary indicator (edge exists or not)
- Can be extended with weights

## 5. Data Statistics Summary

### Final Graph Statistics (Expected)

| Metric | Count |
|--------|-------|
| Total nodes | ~50,000 |
| - Genes | ~8,000 |
| - Drugs | ~200 |
| - Species | ~500 |
| - Proteins | ~10,000 |
| - Mechanisms | ~50 |
| Total edges | ~200,000 |
| - confers_resistance | ~50,000 |
| - targets | ~5,000 |
| - found_in | ~100,000 |
| - encoded_by | ~10,000 |
| - belongs_to | ~8,000 |

### Class Distribution

**Drug Classes (Top 10):**
1. Beta-lactams: 35%
2. Aminoglycosides: 15%
3. Fluoroquinolones: 12%
4. Tetracyclines: 8%
5. Others: 30%

**Resistance Mechanisms (Top 5):**
1. Antibiotic inactivation: 40%
2. Antibiotic target alteration: 25%
3. Antibiotic efflux: 20%
4. Reduced permeability: 10%
5. Other: 5%

## 6. Data Versioning & Reproducibility

### Checksums

```
# CARD data (example)
aro.json: md5:abc123def456...
card.json: md5:xyz789uvw012...

# PATRIC data (example)
AMR.csv: sha256:123abc456def...
```

### Version Tracking

Create `data/raw/VERSION.txt`:
```
CARD Database: v3.2.7 (Downloaded: 2026-02-21)
PATRIC/BV-BRC: Release 2026-02 (Downloaded: 2026-02-21)
```

### Metadata

Create `data/processed/metadata.json`:
```json
{
  "creation_date": "2026-02-21",
  "card_version": "3.2.7",
  "patric_release": "2026-02",
  "graph_stats": {
    "num_nodes": 50123,
    "num_edges": 201456,
    "node_types": 5,
    "edge_types": 5
  },
  "preprocessing_params": {
    "min_sequence_identity": 0.8,
    "min_antibiotic_records": 100,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "random_seed": 42
  }
}
```

## 7. Privacy & Ethics

**Data Usage:**
- All data is publicly available and properly cited
- No patient-identifiable information
- Genome data is anonymized

**Ethical Considerations:**
- Results should not be used for direct clinical decisions without validation
- Predictions must be verified experimentally
- Acknowledge limitations in documentation

## 8. References

1. **CARD**: Alcock et al. (2023). CARD 2023: expanded curation, support for machine learning, and resistome prediction at the Comprehensive Antibiotic Resistance Database. *Nucleic Acids Research*.

2. **PATRIC/BV-BRC**: Olson et al. (2023). Introducing the Bacterial and Viral Bioinformatics Resource Center (BV-BRC): a resource combining PATRIC, IRD and ViPR. *Nucleic Acids Research*.

## 9. Troubleshooting

**Common Issues:**

1. **Download Failures**: Use `wget --continue` for large files
2. **Memory Issues**: Process genomes in batches
3. **Missing Data**: Some antibiotics may not have MIC values
4. **Format Changes**: Check PATRIC release notes for schema updates

**Support:**
- CARD: card@mcmaster.ca
- BV-BRC: help@bv-brc.org
