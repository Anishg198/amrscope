"""
Run complete data preprocessing pipeline.
"""

import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from src.data_preprocessing.card_parser import CARDParser
    from src.data_preprocessing.patric_parser import PATRICParser
    from src.data_preprocessing.graph_builder import HeterogeneousGraphBuilder

    logger.info("=" * 80)
    logger.info("AMR Data Preprocessing Pipeline")
    logger.info("=" * 80)

    # Parse CARD data
    logger.info("\n1. Parsing CARD database...")
    card_parser = CARDParser('data/raw/card')

    # Note: CARD doesn't have aro.json, only aro_index.tsv and card.json
    logger.info("   Loading card.json...")
    card_parser.parse_card_database()

    logger.info("   Extracting entities...")
    card_parser.extract_genes()
    card_parser.extract_drugs()
    card_parser.extract_mechanisms()

    stats = card_parser.get_statistics()
    logger.info(f"\n   CARD Statistics:")
    for key, value in stats.items():
        logger.info(f"     {key}: {value}")

    # Parse PATRIC data
    logger.info("\n2. Parsing PATRIC database...")
    patric_parser = PATRICParser('data/raw/patric')

    patric_parser.parse_amr_data()
    patric_parser.parse_genome_metadata()
    patric_parser.parse_genome_features()

    patric_parser.extract_species()
    patric_parser.extract_amr_genes()
    patric_parser.extract_phenotypes(resistant_only=True, min_records=5)

    stats = patric_parser.get_statistics()
    logger.info(f"\n   PATRIC Statistics:")
    for key, value in stats.items():
        logger.info(f"     {key}: {value}")

    # Build heterogeneous graph
    logger.info("\n3. Building heterogeneous graph...")
    builder = HeterogeneousGraphBuilder(card_parser, patric_parser)
    graph = builder.build(split_target_edges=True)

    # Get graph statistics
    graph_stats = builder.get_statistics()
    logger.info(f"\n   Graph Statistics:")
    logger.info(f"     Node types: {graph_stats['num_node_types']}")
    logger.info(f"     Edge types: {graph_stats['num_edge_types']}")
    logger.info(f"\n   Nodes:")
    for node_type, count in graph_stats['num_nodes'].items():
        logger.info(f"     {node_type}: {count}")
    logger.info(f"\n   Edges:")
    for edge_type, count in graph_stats['num_edges'].items():
        logger.info(f"     {edge_type}: {count}")

    # Save graph
    logger.info("\n4. Saving processed graph...")
    builder.save('data/processed/graph.pkl')

    # Save parsed data
    card_parser.save_parsed_data('data/processed')
    patric_parser.save_parsed_data('data/processed')

    logger.info("\n" + "=" * 80)
    logger.info("✅ Data preprocessing complete!")
    logger.info("=" * 80)
    logger.info(f"\nProcessed graph saved to: data/processed/graph.pkl")
    logger.info(f"Ready for model training!")

except Exception as e:
    logger.error(f"\n❌ Error during preprocessing: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
