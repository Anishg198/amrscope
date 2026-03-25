"""
CARD Database Parser

Parses the Comprehensive Antibiotic Resistance Database (CARD) to extract:
- Antimicrobial Resistance Ontology (ARO) terms
- Resistance genes and sequences
- Drug classes and resistance mechanisms
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

import pandas as pd
from Bio import SeqIO

logger = logging.getLogger(__name__)


class CARDParser:
    """Parser for CARD database files."""

    def __init__(self, data_dir: str):
        """
        Initialize CARD parser.

        Args:
            data_dir: Path to directory containing CARD data files
        """
        self.data_dir = Path(data_dir)
        self.aro_data: Dict = {}
        self.card_data: Dict = {}
        self.genes: Dict = {}
        self.drugs: Dict = {}
        self.mechanisms: Dict = {}
        self.proteins: Dict = {}

        logger.info(f"Initialized CARD parser for directory: {data_dir}")

    def parse_aro_ontology(self, aro_file: str = "aro.json") -> Dict:
        """
        Parse ARO ontology JSON file.

        Args:
            aro_file: Name of ARO JSON file

        Returns:
            Dictionary of ARO terms with structure and relationships
        """
        aro_path = self.data_dir / aro_file
        logger.info(f"Parsing ARO ontology from {aro_path}")

        if not aro_path.exists():
            raise FileNotFoundError(f"ARO file not found: {aro_path}")

        with open(aro_path, 'r') as f:
            self.aro_data = json.load(f)

        logger.info(f"Loaded {len(self.aro_data)} ARO terms")
        return self.aro_data

    def parse_card_database(self, card_file: str = "card.json") -> Dict:
        """
        Parse main CARD database JSON file.

        Args:
            card_file: Name of CARD JSON file

        Returns:
            Dictionary of CARD entries with gene and resistance data
        """
        card_path = self.data_dir / card_file
        logger.info(f"Parsing CARD database from {card_path}")

        if not card_path.exists():
            raise FileNotFoundError(f"CARD file not found: {card_path}")

        with open(card_path, 'r') as f:
            self.card_data = json.load(f)

        logger.info(f"Loaded {len(self.card_data)} CARD entries")
        return self.card_data

    def extract_genes(self) -> Dict[str, Dict]:
        """
        Extract resistance genes from CARD data.

        Returns:
            Dictionary mapping gene IDs to gene information
        """
        logger.info("Extracting resistance genes")

        # Extract from card_data directly since we don't have ARO separately
        for card_id, entry in self.card_data.items():
            if not isinstance(entry, dict):
                continue

            # Get ARO accession as gene ID
            aro_accession = entry.get('ARO_accession', '')
            if not aro_accession:
                continue

            # Check if this is a gene model
            model_type = entry.get('model_type', '')
            if 'protein' in model_type.lower() or 'gene' in model_type.lower():
                self.genes[aro_accession] = {
                    'id': aro_accession,
                    'name': entry.get('ARO_name', ''),
                    'description': entry.get('model_description', ''),
                    'accession': aro_accession,
                    'model_type': model_type,
                }

        logger.info(f"Extracted {len(self.genes)} resistance genes")
        return self.genes

    def extract_drugs(self) -> Dict[str, Dict]:
        """
        Extract antibiotic drugs from CARD data.

        Returns:
            Dictionary mapping drug IDs to drug information
        """
        logger.info("Extracting antibiotic drugs")

        drug_classes = set()

        for aro_id, entry in self.aro_data.items():
            # Extract drug class terms
            if 'ARO_category' in entry:
                categories = entry.get('ARO_category', {})

                is_drug = any(cat.get('category_aro_name') == 'antibiotic molecule'
                            for cat in categories.values())

                if is_drug:
                    drug_id = f"DRUG_{aro_id}"
                    self.drugs[drug_id] = {
                        'id': drug_id,
                        'aro_id': aro_id,
                        'name': entry.get('ARO_name', ''),
                        'description': entry.get('ARO_description', ''),
                        'drug_class': self._extract_drug_class(entry),
                    }

        # Also extract from model sequences in card.json
        for card_id, entry in self.card_data.items():
            if 'model_sequences' in entry:
                model_type = entry.get('model_type', '')
                if 'resistance' in model_type.lower():
                    # Extract drug information from ARO categories
                    aro_categories = entry.get('ARO_category', {})
                    for cat_id, cat_info in aro_categories.items():
                        if cat_info.get('category_aro_class_name') == 'Drug Class':
                            drug_name = cat_info.get('category_aro_name', '')
                            drug_id = f"DRUG_{cat_id}"

                            if drug_id not in self.drugs:
                                self.drugs[drug_id] = {
                                    'id': drug_id,
                                    'aro_id': cat_id,
                                    'name': drug_name,
                                    'drug_class': drug_name,
                                }

        logger.info(f"Extracted {len(self.drugs)} antibiotic drugs")
        return self.drugs

    def extract_mechanisms(self) -> Dict[str, Dict]:
        """
        Extract resistance mechanisms from CARD data.

        Returns:
            Dictionary mapping mechanism IDs to mechanism information
        """
        logger.info("Extracting resistance mechanisms")

        for aro_id, entry in self.aro_data.items():
            if 'ARO_category' in entry:
                categories = entry.get('ARO_category', {})

                is_mechanism = any(
                    cat.get('category_aro_class_name') == 'Resistance Mechanism'
                    for cat in categories.values()
                )

                if is_mechanism:
                    self.mechanisms[aro_id] = {
                        'id': aro_id,
                        'name': entry.get('ARO_name', ''),
                        'description': entry.get('ARO_description', ''),
                        'mechanism_type': self._extract_mechanism_type(entry),
                    }

        logger.info(f"Extracted {len(self.mechanisms)} resistance mechanisms")
        return self.mechanisms

    def extract_gene_drug_relationships(self) -> List[Tuple[str, str, Dict]]:
        """
        Extract gene-drug resistance relationships.

        Returns:
            List of (gene_id, drug_id, metadata) tuples
        """
        logger.info("Extracting gene-drug relationships")
        relationships = []

        for card_id, entry in self.card_data.items():
            if not isinstance(entry, dict):
                continue

            # Get gene ARO ID
            gene_aro = entry.get('ARO_accession', '')
            if not gene_aro:
                continue

            # Get resistance drugs from categories
            aro_categories = entry.get('ARO_category', {})
            if not isinstance(aro_categories, dict):
                continue

            for cat_id, cat_info in aro_categories.items():
                if not isinstance(cat_info, dict):
                    continue

                if cat_info.get('category_aro_class_name') == 'Drug Class':
                    drug_id = f"DRUG_{cat_id}"

                    relationships.append((
                        gene_aro,
                        drug_id,
                        {
                            'evidence': 'CARD database',
                            'model_type': entry.get('model_type', ''),
                        }
                    ))

        logger.info(f"Extracted {len(relationships)} gene-drug relationships")
        return relationships

    def extract_gene_mechanism_relationships(self) -> List[Tuple[str, str]]:
        """
        Extract gene-mechanism relationships.

        Returns:
            List of (gene_id, mechanism_id) tuples
        """
        logger.info("Extracting gene-mechanism relationships")
        relationships = []

        for card_id, entry in self.card_data.items():
            if not isinstance(entry, dict):
                continue

            gene_aro = entry.get('ARO_accession', '')
            if not gene_aro:
                continue

            # Extract mechanism from ARO categories
            aro_categories = entry.get('ARO_category', {})
            if not isinstance(aro_categories, dict):
                continue

            for cat_id, cat_info in aro_categories.items():
                if not isinstance(cat_info, dict):
                    continue

                if cat_info.get('category_aro_class_name') == 'Resistance Mechanism':
                    mechanism_id = cat_id
                    relationships.append((gene_aro, mechanism_id))

        logger.info(f"Extracted {len(relationships)} gene-mechanism relationships")
        return relationships

    def parse_sequences(self, fasta_file: str) -> Dict[str, str]:
        """
        Parse FASTA sequence file.

        Args:
            fasta_file: Name of FASTA file

        Returns:
            Dictionary mapping sequence IDs to sequences
        """
        fasta_path = self.data_dir / fasta_file
        logger.info(f"Parsing sequences from {fasta_path}")

        if not fasta_path.exists():
            logger.warning(f"FASTA file not found: {fasta_path}")
            return {}

        sequences = {}
        for record in SeqIO.parse(fasta_path, "fasta"):
            sequences[record.id] = str(record.seq)

        logger.info(f"Loaded {len(sequences)} sequences")
        return sequences

    def _extract_drug_class(self, entry: Dict) -> str:
        """Extract drug class from ARO entry."""
        categories = entry.get('ARO_category', {})

        for cat in categories.values():
            if cat.get('category_aro_class_name') == 'Drug Class':
                return cat.get('category_aro_name', 'Unknown')

        return 'Unknown'

    def _extract_mechanism_type(self, entry: Dict) -> str:
        """Extract mechanism type from ARO entry."""
        categories = entry.get('ARO_category', {})

        for cat in categories.values():
            if cat.get('category_aro_class_name') == 'Resistance Mechanism':
                return cat.get('category_aro_name', 'Unknown')

        return 'Unknown'

    def get_statistics(self) -> Dict:
        """
        Get statistics about parsed CARD data.

        Returns:
            Dictionary of statistics
        """
        return {
            'num_aro_terms': len(self.aro_data),
            'num_card_entries': len(self.card_data),
            'num_genes': len(self.genes),
            'num_drugs': len(self.drugs),
            'num_mechanisms': len(self.mechanisms),
        }

    def save_parsed_data(self, output_dir: str):
        """
        Save parsed data to files.

        Args:
            output_dir: Directory to save parsed data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save genes
        pd.DataFrame.from_dict(self.genes, orient='index').to_csv(
            output_path / 'card_genes.csv', index=False
        )

        # Save drugs
        pd.DataFrame.from_dict(self.drugs, orient='index').to_csv(
            output_path / 'card_drugs.csv', index=False
        )

        # Save mechanisms
        pd.DataFrame.from_dict(self.mechanisms, orient='index').to_csv(
            output_path / 'card_mechanisms.csv', index=False
        )

        logger.info(f"Saved parsed CARD data to {output_path}")


def main():
    """Example usage of CARD parser."""
    import sys

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Parse CARD data
    parser = CARDParser('data/raw/card')

    try:
        parser.parse_aro_ontology()
        parser.parse_card_database()
        parser.extract_genes()
        parser.extract_drugs()
        parser.extract_mechanisms()

        # Print statistics
        stats = parser.get_statistics()
        print("\nCARD Parsing Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Save parsed data
        parser.save_parsed_data('data/processed')

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.info("Please download CARD data first. See docs/DATA_DESCRIPTION.md")
        sys.exit(1)


if __name__ == '__main__':
    main()
