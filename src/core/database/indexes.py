# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Database index creation and optimization for Qortfolio V2.
Optimizes performance for risk analytics and portfolio management.
"""

from typing import Dict, List
import logging
from .connection import get_database_async

logger = logging.getLogger(__name__)

# Database indexes for optimal performance
DATABASE_INDEXES = {
    "options_data": [
        [("symbol", 1)],
        [("underlying", 1), ("timestamp", -1)],
        [("expiry", 1), ("option_type", 1)],
        [("timestamp", -1)],
        [("strike", 1), ("option_type", 1)],
        [("underlying", 1), ("expiry", 1), ("strike", 1)]  # Compound for volatility surface
    ],
    "price_data": [
        [("symbol", 1), ("timestamp", -1)],
        [("timestamp", -1)],
        [("symbol", 1)],
        [("source", 1), ("timestamp", -1)]
    ],
    "portfolio_data": [
        [("portfolio_id", 1), ("timestamp", -1)],
        [("user_id", 1), ("timestamp", -1)],
        [("portfolio_id", 1)],
        [("timestamp", -1)]
    ],
    "risk_metrics": [
        [("portfolio_id", 1), ("timestamp", -1)],
        [("timestamp", -1)],
        [("portfolio_id", 1)],
        [("calculated_by", 1), ("timestamp", -1)]
    ],
    "performance_reports": [
        [("portfolio_id", 1), ("timestamp", -1)],
        [("timestamp", -1)],
        [("portfolio_id", 1)],
        [("generated_by", 1), ("timestamp", -1)]
    ],
    "performance_attribution": [
        [("portfolio_id", 1), ("attribution_type", 1), ("timestamp", -1)],
        [("portfolio_id", 1), ("timestamp", -1)],
        [("attribution_type", 1), ("timestamp", -1)],
        [("timestamp", -1)]
    ],
    "risk_adjusted_metrics": [
        [("portfolio_id", 1), ("timestamp", -1)],
        [("timestamp", -1)],
        [("portfolio_id", 1)],
        [("risk_free_rate", 1), ("timestamp", -1)]
    ],
    "comprehensive_tearsheets": [
        [("portfolio_id", 1), ("timestamp", -1)],
        [("timestamp", -1)],
        [("portfolio_id", 1)]
    ],
    "sector_risk_metrics": [
        [("portfolio_id", 1), ("timestamp", -1)],
        [("timestamp", -1)],
        [("portfolio_id", 1)]
    ],
    "portfolio_comparisons": [
        [("comparison_date", -1)],
        [("portfolios_compared", 1), ("comparison_date", -1)],
        [("lookback_days", 1), ("comparison_date", -1)]
    ],
    "volatility_surfaces": [
        [("currency", 1), ("timestamp", -1)],
        [("timestamp", -1)],
        [("currency", 1)],
        [("data_points_count", -1), ("timestamp", -1)]
    ],
    "volatility_surfaces_history": [
        [("currency", 1), ("timestamp", -1)],
        [("timestamp", -1)],
        [("currency", 1)]
    ],
    "options_chain_analytics": [
        [("currency", 1), ("timestamp", -1)],
        [("timestamp", -1)],
        [("currency", 1)],
        [("flow_direction", 1), ("timestamp", -1)]
    ],
    "greeks_snapshots": [
        [("portfolio_id", 1), ("timestamp", -1)],
        [("currency", 1), ("timestamp", -1)],
        [("timestamp", -1)],
        [("portfolio_id", 1), ("currency", 1)]
    ],
    "implied_volatility_points": [
        [("currency", 1), ("timestamp", -1)],
        [("symbol", 1), ("timestamp", -1)],
        [("currency", 1), ("moneyness", 1), ("timestamp", -1)],
        [("timestamp", -1)],
        [("volume", -1), ("timestamp", -1)]
    ]
}

async def create_database_indexes() -> Dict[str, List[str]]:
    """
    Create optimized database indexes for all collections

    Returns:
        Dictionary mapping collection names to created index names
    """
    try:
        db = await get_database_async()
        created_indexes = {}

        for collection_name, indexes in DATABASE_INDEXES.items():
            collection = getattr(db, collection_name)
            collection_indexes = []

            for index_fields in indexes:
                try:
                    # Create index
                    index_name = await collection.create_index(index_fields)
                    collection_indexes.append(index_name)
                    logger.info(f"Created index {index_name} on {collection_name}")
                except Exception as e:
                    # Index might already exist
                    if "duplicate key" in str(e).lower() or "already exists" in str(e).lower():
                        logger.info(f"Index on {collection_name} already exists: {index_fields}")
                    else:
                        logger.warning(f"Failed to create index on {collection_name}: {e}")

            created_indexes[collection_name] = collection_indexes

        logger.info(f"Database indexing completed for {len(created_indexes)} collections")
        return created_indexes

    except Exception as e:
        logger.error(f"Error creating database indexes: {e}")
        return {}

async def drop_collection_indexes(collection_name: str) -> bool:
    """
    Drop all indexes for a specific collection (except _id)

    Args:
        collection_name: Name of the collection

    Returns:
        True if successful, False otherwise
    """
    try:
        db = await get_database_async()
        collection = getattr(db, collection_name)

        # Get current indexes
        indexes = await collection.list_indexes().to_list(length=None)

        for index in indexes:
            index_name = index.get('name')
            if index_name and index_name != '_id_':  # Don't drop the default _id index
                try:
                    await collection.drop_index(index_name)
                    logger.info(f"Dropped index {index_name} from {collection_name}")
                except Exception as e:
                    logger.warning(f"Failed to drop index {index_name}: {e}")

        return True

    except Exception as e:
        logger.error(f"Error dropping indexes for {collection_name}: {e}")
        return False

async def get_collection_stats(collection_name: str) -> Dict:
    """
    Get performance statistics for a collection

    Args:
        collection_name: Name of the collection

    Returns:
        Dictionary with collection statistics
    """
    try:
        db = await get_database_async()
        collection = getattr(db, collection_name)

        # Get collection stats
        stats = await db.command("collStats", collection_name)

        # Get index information
        indexes = await collection.list_indexes().to_list(length=None)

        return {
            'collection': collection_name,
            'size_bytes': stats.get('size', 0),
            'document_count': stats.get('count', 0),
            'average_object_size': stats.get('avgObjSize', 0),
            'storage_size': stats.get('storageSize', 0),
            'index_count': len(indexes),
            'index_names': [idx.get('name') for idx in indexes],
            'total_index_size': stats.get('totalIndexSize', 0)
        }

    except Exception as e:
        logger.error(f"Error getting stats for {collection_name}: {e}")
        return {}

async def optimize_database_performance() -> Dict:
    """
    Comprehensive database performance optimization

    Returns:
        Optimization results and recommendations
    """
    try:
        # Create all indexes
        index_results = await create_database_indexes()

        # Get performance stats for key collections
        key_collections = ['portfolio_data', 'risk_metrics', 'performance_reports', 'price_data']
        collection_stats = {}

        for collection in key_collections:
            stats = await get_collection_stats(collection)
            collection_stats[collection] = stats

        # Calculate optimization metrics
        total_documents = sum(stats.get('document_count', 0) for stats in collection_stats.values())
        total_storage = sum(stats.get('storage_size', 0) for stats in collection_stats.values())
        total_indexes = sum(len(indexes) for indexes in index_results.values())

        optimization_results = {
            'indexes_created': index_results,
            'collection_stats': collection_stats,
            'summary': {
                'total_collections_optimized': len(index_results),
                'total_indexes_created': total_indexes,
                'total_documents': total_documents,
                'total_storage_mb': round(total_storage / (1024 * 1024), 2),
                'optimization_date': str(datetime.utcnow())
            },
            'recommendations': []
        }

        # Add performance recommendations
        for collection, stats in collection_stats.items():
            if stats.get('document_count', 0) > 10000 and stats.get('index_count', 0) < 3:
                optimization_results['recommendations'].append(
                    f"Consider adding more indexes to {collection} for better query performance"
                )

            if stats.get('average_object_size', 0) > 10000:  # 10KB+ documents
                optimization_results['recommendations'].append(
                    f"Large document size in {collection} - consider document restructuring"
                )

        logger.info("Database performance optimization completed")
        return optimization_results

    except Exception as e:
        logger.error(f"Error optimizing database performance: {e}")
        return {'error': str(e)}

# Performance monitoring queries
PERFORMANCE_QUERIES = {
    "portfolio_risk_lookup": {
        "collection": "risk_metrics",
        "query": {"portfolio_id": "test_portfolio"},
        "sort": [("timestamp", -1)],
        "limit": 1
    },
    "price_history_lookup": {
        "collection": "price_data",
        "query": {"symbol": "BTC"},
        "sort": [("timestamp", -1)],
        "limit": 365
    },
    "portfolio_performance_lookup": {
        "collection": "performance_reports",
        "query": {"portfolio_id": "test_portfolio"},
        "sort": [("timestamp", -1)],
        "limit": 1
    }
}

async def benchmark_query_performance() -> Dict:
    """
    Benchmark key query performance for risk analytics

    Returns:
        Query performance metrics
    """
    try:
        db = await get_database_async()
        benchmark_results = {}

        for query_name, query_config in PERFORMANCE_QUERIES.items():
            collection = getattr(db, query_config['collection'])

            # Time the query
            import time
            start_time = time.time()

            cursor = collection.find(query_config['query'])
            if 'sort' in query_config:
                cursor = cursor.sort(query_config['sort'])
            if 'limit' in query_config:
                cursor = cursor.limit(query_config['limit'])

            results = await cursor.to_list(length=query_config.get('limit', 100))

            end_time = time.time()
            query_time = (end_time - start_time) * 1000  # Convert to milliseconds

            benchmark_results[query_name] = {
                'query_time_ms': round(query_time, 2),
                'documents_returned': len(results),
                'collection': query_config['collection'],
                'performance_rating': 'Excellent' if query_time < 10 else 'Good' if query_time < 50 else 'Needs Optimization'
            }

        return benchmark_results

    except Exception as e:
        logger.error(f"Error benchmarking query performance: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    import asyncio
    from datetime import datetime

    async def main():
        print("Creating database indexes...")
        results = await create_database_indexes()
        print(f"Created indexes for {len(results)} collections")

        print("\nOptimizing database performance...")
        optimization = await optimize_database_performance()
        print(f"Optimization complete: {optimization.get('summary', {})}")

        print("\nBenchmarking query performance...")
        benchmarks = await benchmark_query_performance()
        for query, metrics in benchmarks.items():
            print(f"{query}: {metrics.get('query_time_ms', 0)}ms - {metrics.get('performance_rating', 'Unknown')}")

    asyncio.run(main())