"""
PATRIC/BV-BRC Database Parser

Parses the Bacterial and Viral Bioinformatics Resource Center (BV-BRC) data to extract:
- AMR phenotype data
- Genomic features
- Bacterial species information
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class PATRICParser:
    """Parser for PATRIC/BV-BRC database files."""

    def __init__(self, data_dir: str):
        """
        Initialize PATRIC parser.

        Args:
            data_dir: Path to directory containing PATRIC data files
        """
        self.data_dir = Path(data_dir)
        self.amr_data: Optional[pd.DataFrame] = None
        self.genome_metadata: Optional[pd.DataFrame] = None
        self.genome_features: Optional[pd.DataFrame] = None

        self.species: Dict[str, Dict] = {}
        self.amr_genes: Dict[str, Dict] = {}
        self.phenotypes: List[Dict] = []

        logger.info(f"Initialized PATRIC parser for directory: {data_dir}")

    def parse_amr_data(self, amr_file: str = "AMR.csv") -> pd.DataFrame:
        """
        Parse AMR phenotype data.

        Args:
            amr_file: Name of AMR CSV file

        Returns:
            DataFrame with AMR phenotype data
        """
        amr_path = self.data_dir / amr_file
        logger.info(f"Parsing AMR data from {amr_path}")

        if not amr_path.exists():
            raise FileNotFoundError(f"AMR file not found: {amr_path}")

        # Read CSV with appropriate dtypes
        self.amr_data = pd.read_csv(amr_path, low_memory=False)

        logger.info(f"Loaded {len(self.amr_data)} AMR phenotype records")
        logger.info(f"Columns: {list(self.amr_data.columns)}")

        return self.amr_data

    def parse_genome_metadata(self, metadata_file: str = "genome_metadata.csv") -> pd.DataFrame:
        """
        Parse genome metadata.

        Args:
            metadata_file: Name of genome metadata CSV file

        Returns:
            DataFrame with genome metadata
        """
        metadata_path = self.data_dir / metadata_file
        logger.info(f"Parsing genome metadata from {metadata_path}")

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.genome_metadata = pd.read_csv(metadata_path, low_memory=False)

        logger.info(f"Loaded metadata for {len(self.genome_metadata)} genomes")

        return self.genome_metadata

    def parse_genome_features(self, features_file: str = "PATRIC_genome_feature.csv",
                             sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Parse genome features (large file, may sample).

        Args:
            features_file: Name of features CSV file
            sample_size: If specified, sample this many rows (for testing)

        Returns:
            DataFrame with genome features
        """
        features_path = self.data_dir / features_file
        logger.info(f"Parsing genome features from {features_path}")

        if not features_path.exists():
            logger.warning(f"Features file not found: {features_path}")
            return pd.DataFrame()

        # This file can be very large (>10GB), so use chunking if needed
        if sample_size:
            logger.info(f"Sampling {sample_size} features for testing")
            self.genome_features = pd.read_csv(
                features_path,
                nrows=sample_size,
                low_memory=False
            )
        else:
            # Read in chunks for memory efficiency
            logger.info("Reading features in chunks...")
            chunks = []
            for chunk in pd.read_csv(features_path, chunksize=100000, low_memory=False):
                # Filter for AMR-related features only
                amr_chunk = chunk[
                    chunk['product'].str.contains('resist|beta-lactam|efflux',
                                                  case=False, na=False) |
                    chunk['gene'].str.contains('bla|tet|erm|van|mec',
                                              case=False, na=False)
                ]
                if len(amr_chunk) > 0:
                    chunks.append(amr_chunk)

            if chunks:
                self.genome_features = pd.concat(chunks, ignore_index=True)
            else:
                self.genome_features = pd.DataFrame()

        logger.info(f"Loaded {len(self.genome_features)} AMR-related features")

        return self.genome_features

    def extract_species(self) -> Dict[str, Dict]:
        """
        Extract bacterial species information.

        Returns:
            Dictionary mapping species IDs to species information
        """
        logger.info("Extracting bacterial species")

        if self.genome_metadata is None:
            logger.warning("Genome metadata not loaded, loading now...")
            self.parse_genome_metadata()

        # Extract unique species
        if 'genome_name' in self.genome_metadata.columns and 'taxon_id' in self.genome_metadata.columns:
            species_data = self.genome_metadata[['genome_name', 'taxon_id', 'genome_id']].drop_duplicates('taxon_id')

            for _, row in species_data.iterrows():
                taxon_id = str(row['taxon_id'])
                species_id = f"SPECIES_{taxon_id}"

                self.species[species_id] = {
                    'id': species_id,
                    'taxon_id': taxon_id,
                    'name': row['genome_name'],
                    'genome_ids': [],  # Will populate later
                }

        # Group genomes by species
        if 'taxon_id' in self.genome_metadata.columns:
            genome_by_species = self.genome_metadata.groupby('taxon_id')['genome_id'].apply(list)

            for taxon_id, genome_ids in genome_by_species.items():
                species_id = f"SPECIES_{taxon_id}"
                if species_id in self.species:
                    self.species[species_id]['genome_ids'] = genome_ids

        logger.info(f"Extracted {len(self.species)} bacterial species")
        return self.species

    def extract_amr_genes(self) -> Dict[str, Dict]:
        """
        Extract AMR genes from genome features.

        Returns:
            Dictionary mapping gene IDs to gene information
        """
        logger.info("Extracting AMR genes from features")

        if self.genome_features is None or len(self.genome_features) == 0:
            logger.warning("No genome features loaded")
            return {}

        # Extract genes with AMR annotations
        for _, row in self.genome_features.iterrows():
            gene_id = row.get('feature_id', '')
            gene_name = row.get('gene', '')

            if gene_id:
                self.amr_genes[gene_id] = {
                    'id': gene_id,
                    'gene': gene_name,
                    'product': row.get('product', ''),
                    'genome_id': row.get('genome_id', ''),
                    'aa_length': row.get('aa_length', 0),
                }

        logger.info(f"Extracted {len(self.amr_genes)} AMR genes")
        return self.amr_genes

    def extract_phenotypes(self,
                          resistant_only: bool = True,
                          min_records: int = 10) -> List[Dict]:
        """
        Extract AMR phenotype data.

        Args:
            resistant_only: If True, only include resistant phenotypes
            min_records: Minimum number of records per antibiotic to include

        Returns:
            List of phenotype records
        """
        logger.info("Extracting AMR phenotypes")

        if self.amr_data is None:
            logger.warning("AMR data not loaded, loading now...")
            self.parse_amr_data()

        # Filter data
        df = self.amr_data.copy()

        # Filter for resistant phenotypes if requested
        if resistant_only and 'resistant_phenotype' in df.columns:
            df = df[df['resistant_phenotype'].str.lower().isin(['resistant', 'r'])]
            logger.info(f"Filtered to {len(df)} resistant phenotypes")

        # Filter antibiotics with too few records
        if 'antibiotic' in df.columns:
            antibiotic_counts = df['antibiotic'].value_counts()
            valid_antibiotics = antibiotic_counts[antibiotic_counts >= min_records].index
            df = df[df['antibiotic'].isin(valid_antibiotics)]
            logger.info(f"Filtered to {len(valid_antibiotics)} antibiotics with >={min_records} records")

        # Convert to list of dicts
        for _, row in df.iterrows():
            self.phenotypes.append({
                'genome_id': row.get('genome_id', ''),
                'antibiotic': row.get('antibiotic', ''),
                'resistant_phenotype': row.get('resistant_phenotype', ''),
                'measurement': row.get('measurement', ''),
                'measurement_unit': row.get('measurement_unit', ''),
                'laboratory_typing_method': row.get('laboratory_typing_method', ''),
            })

        logger.info(f"Extracted {len(self.phenotypes)} phenotype records")
        return self.phenotypes

    def get_gene_species_relationships(self) -> List[Tuple[str, str]]:
        """
        Extract gene-species relationships (found_in edges).

        Returns:
            List of (gene_id, species_id) tuples
        """
        logger.info("Extracting gene-species relationships")
        relationships = []

        if self.genome_features is None or len(self.genome_features) == 0:
            return relationships

        # Map genome_id to species_id
        genome_to_species = {}
        if self.genome_metadata is not None:
            for _, row in self.genome_metadata.iterrows():
                genome_id = row.get('genome_id', '')
                taxon_id = row.get('taxon_id', '')
                if genome_id and taxon_id:
                    genome_to_species[genome_id] = f"SPECIES_{taxon_id}"

        # Create gene-species edges
        for gene_id, gene_data in self.amr_genes.items():
            genome_id = gene_data.get('genome_id', '')
            if genome_id in genome_to_species:
                species_id = genome_to_species[genome_id]
                relationships.append((gene_id, species_id))

        logger.info(f"Extracted {len(relationships)} gene-species relationships")
        return relationships

    def get_genome_drug_relationships(self) -> List[Tuple[str, str, Dict]]:
        """
        Extract genome-drug resistance relationships from phenotype data.

        Returns:
            List of (genome_id, antibiotic, metadata) tuples
        """
        logger.info("Extracting genome-drug relationships")
        relationships = []

        for phenotype in self.phenotypes:
            genome_id = phenotype['genome_id']
            antibiotic = phenotype['antibiotic']

            # Standardize antibiotic name to match CARD drugs
            # (This would need a proper mapping in practice)
            drug_id = f"DRUG_{antibiotic.replace(' ', '_').upper()}"

            relationships.append((
                genome_id,
                drug_id,
                {
                    'phenotype': phenotype['resistant_phenotype'],
                    'measurement': phenotype['measurement'],
                    'measurement_unit': phenotype['measurement_unit'],
                    'method': phenotype['laboratory_typing_method'],
                }
            ))

        logger.info(f"Extracted {len(relationships)} genome-drug relationships")
        return relationships

    def get_statistics(self) -> Dict:
        """
        Get statistics about parsed PATRIC data.

        Returns:
            Dictionary of statistics
        """
        stats = {
            'num_species': len(self.species),
            'num_amr_genes': len(self.amr_genes),
            'num_phenotypes': len(self.phenotypes),
        }

        if self.amr_data is not None:
            stats['num_amr_records'] = len(self.amr_data)
            if 'antibiotic' in self.amr_data.columns:
                stats['num_unique_antibiotics'] = self.amr_data['antibiotic'].nunique()

        if self.genome_metadata is not None:
            stats['num_genomes'] = len(self.genome_metadata)

        return stats

    def save_parsed_data(self, output_dir: str):
        """
        Save parsed data to files.

        Args:
            output_dir: Directory to save parsed data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save species
        if self.species:
            pd.DataFrame.from_dict(self.species, orient='index').to_csv(
                output_path / 'patric_species.csv', index=False
            )

        # Save AMR genes
        if self.amr_genes:
            pd.DataFrame.from_dict(self.amr_genes, orient='index').to_csv(
                output_path / 'patric_amr_genes.csv', index=False
            )

        # Save phenotypes
        if self.phenotypes:
            pd.DataFrame(self.phenotypes).to_csv(
                output_path / 'patric_phenotypes.csv', index=False
            )

        logger.info(f"Saved parsed PATRIC data to {output_path}")


def main():
    """Example usage of PATRIC parser."""
    import sys

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Parse PATRIC data
    parser = PATRICParser('data/raw/patric')

    try:
        parser.parse_amr_data()
        parser.parse_genome_metadata()
        # Sample features for testing (full file is very large)
        parser.parse_genome_features(sample_size=10000)

        parser.extract_species()
        parser.extract_amr_genes()
        parser.extract_phenotypes()

        # Print statistics
        stats = parser.get_statistics()
        print("\nPATRIC Parsing Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Save parsed data
        parser.save_parsed_data('data/processed')

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.info("Please download PATRIC data first. See docs/DATA_DESCRIPTION.md")
        sys.exit(1)


if __name__ == '__main__':
    main()
