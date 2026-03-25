"""
Heterogeneous Graph Builder

Constructs a heterogeneous graph from CARD and PATRIC data for AMR prediction.
Creates nodes (genes, drugs, species, proteins, mechanisms) and edges (various relationships).
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import HeteroData
from sklearn.model_selection import train_test_split

from .card_parser import CARDParser
from .patric_parser import PATRICParser

logger = logging.getLogger(__name__)


class HeterogeneousGraphBuilder:
    """Builder for heterogeneous AMR graph."""

    def __init__(self, card_parser: Optional[CARDParser] = None,
                 patric_parser: Optional[PATRICParser] = None):
        """
        Initialize graph builder.

        Args:
            card_parser: Initialized CARD parser with loaded data
            patric_parser: Initialized PATRIC parser with loaded data
        """
        self.card_parser = card_parser
        self.patric_parser = patric_parser

        # Node mappings (original_id -> integer index)
        self.gene_to_idx: Dict[str, int] = {}
        self.drug_to_idx: Dict[str, int] = {}
        self.species_to_idx: Dict[str, int] = {}
        self.protein_to_idx: Dict[str, int] = {}
        self.mechanism_to_idx: Dict[str, int] = {}

        # Reverse mappings (integer index -> original_id)
        self.idx_to_gene: Dict[int, str] = {}
        self.idx_to_drug: Dict[int, str] = {}
        self.idx_to_species: Dict[int, str] = {}
        self.idx_to_protein: Dict[int, str] = {}
        self.idx_to_mechanism: Dict[int, str] = {}

        # Edge lists
        self.edges: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

        # Graph data
        self.graph: Optional[HeteroData] = None

        logger.info("Initialized heterogeneous graph builder")

    def build_node_mappings(self):
        """Create mappings from node IDs to integer indices."""
        logger.info("Building node mappings")

        # Map genes
        if self.card_parser and self.card_parser.genes:
            for idx, gene_id in enumerate(sorted(self.card_parser.genes.keys())):
                self.gene_to_idx[gene_id] = idx
                self.idx_to_gene[idx] = gene_id

        # Add PATRIC genes
        if self.patric_parser and self.patric_parser.amr_genes:
            current_idx = len(self.gene_to_idx)
            for gene_id in sorted(self.patric_parser.amr_genes.keys()):
                if gene_id not in self.gene_to_idx:
                    self.gene_to_idx[gene_id] = current_idx
                    self.idx_to_gene[current_idx] = gene_id
                    current_idx += 1

        # Map drugs
        if self.card_parser and self.card_parser.drugs:
            for idx, drug_id in enumerate(sorted(self.card_parser.drugs.keys())):
                self.drug_to_idx[drug_id] = idx
                self.idx_to_drug[idx] = drug_id

        # Map species
        if self.patric_parser and self.patric_parser.species:
            for idx, species_id in enumerate(sorted(self.patric_parser.species.keys())):
                self.species_to_idx[species_id] = idx
                self.idx_to_species[idx] = species_id

        # Map mechanisms
        if self.card_parser and self.card_parser.mechanisms:
            for idx, mech_id in enumerate(sorted(self.card_parser.mechanisms.keys())):
                self.mechanism_to_idx[mech_id] = idx
                self.idx_to_mechanism[idx] = mech_id

        logger.info(f"Created mappings: {len(self.gene_to_idx)} genes, "
                   f"{len(self.drug_to_idx)} drugs, {len(self.species_to_idx)} species, "
                   f"{len(self.mechanism_to_idx)} mechanisms")

    def build_edges(self):
        """Build edge lists for all relationship types."""
        logger.info("Building edge lists")

        # 1. Gene -> Drug (confers_resistance) - PRIMARY PREDICTION TARGET
        if self.card_parser:
            gene_drug_rels = self.card_parser.extract_gene_drug_relationships()
            for gene_id, drug_id, metadata in gene_drug_rels:
                if gene_id in self.gene_to_idx and drug_id in self.drug_to_idx:
                    gene_idx = self.gene_to_idx[gene_id]
                    drug_idx = self.drug_to_idx[drug_id]
                    self.edges['gene_confers_drug'].append((gene_idx, drug_idx))

        # Also from PATRIC phenotype data
        if self.patric_parser:
            # Map PATRIC antibiotics to CARD drug IDs (simplified mapping)
            genome_drug_rels = self.patric_parser.get_genome_drug_relationships()

            # For each genome, find its genes and create gene-drug edges
            genome_to_genes = defaultdict(list)
            for gene_id, gene_data in self.patric_parser.amr_genes.items():
                genome_id = gene_data.get('genome_id', '')
                if genome_id and gene_id in self.gene_to_idx:
                    genome_to_genes[genome_id].append(self.gene_to_idx[gene_id])

            for genome_id, drug_id, metadata in genome_drug_rels:
                if genome_id in genome_to_genes and drug_id in self.drug_to_idx:
                    drug_idx = self.drug_to_idx[drug_id]
                    for gene_idx in genome_to_genes[genome_id]:
                        self.edges['gene_confers_drug'].append((gene_idx, drug_idx))

        # 2. Gene -> Species (found_in)
        if self.patric_parser:
            gene_species_rels = self.patric_parser.get_gene_species_relationships()
            for gene_id, species_id in gene_species_rels:
                if gene_id in self.gene_to_idx and species_id in self.species_to_idx:
                    gene_idx = self.gene_to_idx[gene_id]
                    species_idx = self.species_to_idx[species_id]
                    self.edges['gene_found_in_species'].append((gene_idx, species_idx))

        # 3. Gene -> Mechanism (belongs_to)
        if self.card_parser:
            gene_mech_rels = self.card_parser.extract_gene_mechanism_relationships()
            for gene_id, mech_id in gene_mech_rels:
                if gene_id in self.gene_to_idx and mech_id in self.mechanism_to_idx:
                    gene_idx = self.gene_to_idx[gene_id]
                    mech_idx = self.mechanism_to_idx[mech_id]
                    self.edges['gene_belongs_to_mechanism'].append((gene_idx, mech_idx))

        # Remove duplicates
        for edge_type in self.edges:
            self.edges[edge_type] = list(set(self.edges[edge_type]))

        # Log statistics
        for edge_type, edge_list in self.edges.items():
            logger.info(f"  {edge_type}: {len(edge_list)} edges")

    def create_node_features(self) -> Dict[str, torch.Tensor]:
        """
        Create initial node features (simple one-hot or random for now).

        Returns:
            Dictionary mapping node types to feature tensors
        """
        logger.info("Creating node features")

        features = {}

        # Gene features (random initialization for now - can be replaced with embeddings)
        if len(self.gene_to_idx) > 0:
            features['gene'] = torch.randn(len(self.gene_to_idx), 128)

        # Drug features (random initialization - can be replaced with chemical descriptors)
        if len(self.drug_to_idx) > 0:
            features['drug'] = torch.randn(len(self.drug_to_idx), 64)

        # Species features (random initialization - can be replaced with taxonomic embeddings)
        if len(self.species_to_idx) > 0:
            features['species'] = torch.randn(len(self.species_to_idx), 32)

        # Mechanism features (one-hot encoding)
        if len(self.mechanism_to_idx) > 0:
            features['mechanism'] = torch.eye(len(self.mechanism_to_idx))

        logger.info(f"Created features for {len(features)} node types")
        return features

    def create_hetero_data(self) -> HeteroData:
        """
        Create PyTorch Geometric HeteroData object.

        Returns:
            HeteroData object with nodes and edges
        """
        logger.info("Creating HeteroData object")

        data = HeteroData()

        # Add node features
        features = self.create_node_features()
        for node_type, feat_tensor in features.items():
            data[node_type].x = feat_tensor
            logger.info(f"  {node_type}: {feat_tensor.shape}")

        # Add edges
        for edge_type, edge_list in self.edges.items():
            if len(edge_list) == 0:
                continue

            # Parse edge type (e.g., 'gene_confers_drug' -> ('gene', 'confers', 'drug'))
            parts = edge_type.split('_')
            if len(parts) >= 3:
                src_type = parts[0]
                relation = '_'.join(parts[1:-1])
                dst_type = parts[-1]
            else:
                continue

            # Convert to edge_index tensor
            edge_array = np.array(edge_list).T
            edge_index = torch.tensor(edge_array, dtype=torch.long)

            # Add to HeteroData
            data[src_type, relation, dst_type].edge_index = edge_index
            logger.info(f"  ({src_type}, {relation}, {dst_type}): {edge_index.shape}")

        self.graph = data
        return data

    def split_edges(self, edge_type: Tuple[str, str, str],
                   train_ratio: float = 0.8,
                   val_ratio: float = 0.1,
                   test_ratio: float = 0.1,
                   random_state: int = 42) -> HeteroData:
        """
        Split edges for link prediction task.

        Args:
            edge_type: Tuple of (src_type, relation, dst_type)
            train_ratio: Ratio of edges for training
            val_ratio: Ratio of edges for validation
            test_ratio: Ratio of edges for testing
            random_state: Random seed for reproducibility

        Returns:
            HeteroData object with train/val/test edge masks
        """
        logger.info(f"Splitting edges for {edge_type}")

        if self.graph is None:
            raise ValueError("Graph not created. Call create_hetero_data() first.")

        # Get edge index
        edge_index = self.graph[edge_type].edge_index

        # Create indices
        num_edges = edge_index.shape[1]
        indices = np.arange(num_edges)

        # Split indices
        train_idx, temp_idx = train_test_split(
            indices,
            train_size=train_ratio,
            random_state=random_state
        )

        val_size = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_size,
            random_state=random_state
        )

        # Create masks
        train_mask = torch.zeros(num_edges, dtype=torch.bool)
        val_mask = torch.zeros(num_edges, dtype=torch.bool)
        test_mask = torch.zeros(num_edges, dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        # Add to graph
        self.graph[edge_type].train_mask = train_mask
        self.graph[edge_type].val_mask = val_mask
        self.graph[edge_type].test_mask = test_mask

        logger.info(f"  Train: {train_mask.sum()} edges")
        logger.info(f"  Val: {val_mask.sum()} edges")
        logger.info(f"  Test: {test_mask.sum()} edges")

        return self.graph

    def build(self, split_target_edges: bool = True) -> HeteroData:
        """
        Build complete heterogeneous graph.

        Args:
            split_target_edges: Whether to split target edges for link prediction

        Returns:
            HeteroData object
        """
        logger.info("Building heterogeneous graph")

        self.build_node_mappings()
        self.build_edges()
        self.create_hetero_data()

        # Split gene-drug edges for link prediction
        if split_target_edges and ('gene', 'confers', 'drug') in self.graph.edge_types:
            self.split_edges(('gene', 'confers', 'drug'))

        logger.info("Graph building complete")
        return self.graph

    def save(self, filepath: str):
        """
        Save graph and mappings to file.

        Args:
            filepath: Path to save graph
        """
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'graph': self.graph,
            'gene_to_idx': self.gene_to_idx,
            'drug_to_idx': self.drug_to_idx,
            'species_to_idx': self.species_to_idx,
            'mechanism_to_idx': self.mechanism_to_idx,
            'idx_to_gene': self.idx_to_gene,
            'idx_to_drug': self.idx_to_drug,
            'idx_to_species': self.idx_to_species,
            'idx_to_mechanism': self.idx_to_mechanism,
        }

        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)

        logger.info(f"Saved graph to {save_path}")

    @staticmethod
    def load(filepath: str) -> 'HeterogeneousGraphBuilder':
        """
        Load graph and mappings from file.

        Args:
            filepath: Path to load graph from

        Returns:
            HeterogeneousGraphBuilder with loaded data
        """
        logger.info(f"Loading graph from {filepath}")

        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)

        builder = HeterogeneousGraphBuilder()
        builder.graph = save_data['graph']
        builder.gene_to_idx = save_data['gene_to_idx']
        builder.drug_to_idx = save_data['drug_to_idx']
        builder.species_to_idx = save_data['species_to_idx']
        builder.mechanism_to_idx = save_data['mechanism_to_idx']
        builder.idx_to_gene = save_data['idx_to_gene']
        builder.idx_to_drug = save_data['idx_to_drug']
        builder.idx_to_species = save_data['idx_to_species']
        builder.idx_to_mechanism = save_data['idx_to_mechanism']

        logger.info("Graph loaded successfully")
        return builder

    def get_statistics(self) -> Dict:
        """
        Get statistics about the graph.

        Returns:
            Dictionary of graph statistics
        """
        stats = {
            'num_node_types': 0,
            'num_edge_types': 0,
            'num_nodes': {},
            'num_edges': {},
        }

        if self.graph is not None:
            stats['num_node_types'] = len(self.graph.node_types)
            stats['num_edge_types'] = len(self.graph.edge_types)

            for node_type in self.graph.node_types:
                stats['num_nodes'][node_type] = self.graph[node_type].num_nodes

            for edge_type in self.graph.edge_types:
                stats['num_edges'][str(edge_type)] = self.graph[edge_type].num_edges

        return stats


def main():
    """Example usage of graph builder."""
    import sys

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # Initialize parsers
        card_parser = CARDParser('data/raw/card')
        patric_parser = PATRICParser('data/raw/patric')

        # Parse data
        logger.info("Parsing CARD data...")
        card_parser.parse_aro_ontology()
        card_parser.parse_card_database()
        card_parser.extract_genes()
        card_parser.extract_drugs()
        card_parser.extract_mechanisms()

        logger.info("Parsing PATRIC data...")
        patric_parser.parse_amr_data()
        patric_parser.parse_genome_metadata()
        patric_parser.parse_genome_features(sample_size=5000)  # Sample for testing
        patric_parser.extract_species()
        patric_parser.extract_amr_genes()
        patric_parser.extract_phenotypes()

        # Build graph
        logger.info("Building heterogeneous graph...")
        builder = HeterogeneousGraphBuilder(card_parser, patric_parser)
        graph = builder.build()

        # Print statistics
        stats = builder.get_statistics()
        print("\nGraph Statistics:")
        print(f"  Node types: {stats['num_node_types']}")
        print(f"  Edge types: {stats['num_edge_types']}")
        print("\nNodes:")
        for node_type, count in stats['num_nodes'].items():
            print(f"  {node_type}: {count}")
        print("\nEdges:")
        for edge_type, count in stats['num_edges'].items():
            print(f"  {edge_type}: {count}")

        # Save graph
        builder.save('data/processed/graph.pkl')

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.info("Please download data first. See docs/DATA_DESCRIPTION.md")
        sys.exit(1)


if __name__ == '__main__':
    main()
