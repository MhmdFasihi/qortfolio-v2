"""Database CRUD Operations for MongoDB"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pymongo import UpdateOne
from .connection import db_connection
import logging

logger = logging.getLogger(__name__)

class DatabaseOperations:
    """MongoDB CRUD operations handler"""
    
    def __init__(self):
        self.db = db_connection
        
    # CREATE Operations
    def insert_one(self, collection_name: str, document: Dict) -> Optional[str]:
        """Insert single document"""
        try:
            collection = self.db.get_collection(collection_name)
            if collection is not None:
                document['timestamp'] = datetime.now()
                result = collection.insert_one(document)
                return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Insert error: {e}")
        return None

    # === Async convenience queries used by UI layers ===
    async def get_latest_options(self, underlying: str, limit: int = 100) -> List[Dict]:
        """Fetch latest options for an underlying (async, motor)."""
        try:
            adb = await self.db.get_database_async()
            cursor = adb.options_data.find({"underlying": underlying}).sort("timestamp", -1).limit(limit)
            return await cursor.to_list(length=limit)
        except Exception as e:
            logger.error(f"get_latest_options error: {e}")
            return []
    
    def insert_many(self, collection_name: str, documents: List[Dict]) -> List[str]:
        """Insert multiple documents"""
        try:
            collection = self.db.get_collection(collection_name)
            if collection is not None:
                for doc in documents:
                    doc['timestamp'] = datetime.now()
                result = collection.insert_many(documents)
                return [str(id) for id in result.inserted_ids]
        except Exception as e:
            logger.error(f"Bulk insert error: {e}")
        return []
    
    # READ Operations
    def find_one(self, collection_name: str, query: Dict) -> Optional[Dict]:
        """Find single document"""
        try:
            collection = self.db.get_collection(collection_name)
            if collection is not None:
                return collection.find_one(query)
        except Exception as e:
            logger.error(f"Find error: {e}")
        return None
    
    def find_many(self, collection_name: str, query: Dict = None, 
                  limit: int = 100, sort: List = None) -> List[Dict]:
        """Find multiple documents"""
        try:
            collection = self.db.get_collection(collection_name)
            if collection is not None:
                query = query or {}
                cursor = collection.find(query).limit(limit)
                if sort:
                    cursor = cursor.sort(sort)
                return list(cursor)
        except Exception as e:
            logger.error(f"Find many error: {e}")
        return []
    
    # UPDATE Operations
    def update_one(self, collection_name: str, query: Dict, 
                   update: Dict, upsert: bool = False) -> bool:
        """Update single document"""
        try:
            collection = self.db.get_collection(collection_name)
            if collection is not None:
                update['$set'] = update.get('$set', {})
                update['$set']['updated_at'] = datetime.now()
                result = collection.update_one(query, update, upsert=upsert)
                return result.modified_count > 0 or result.upserted_id is not None
        except Exception as e:
            logger.error(f"Update error: {e}")
        return False
    
    def bulk_update(self, collection_name: str, updates: List[Dict]) -> int:
        """Bulk update operations"""
        try:
            collection = self.db.get_collection(collection_name)
            if collection is not None:
                operations = []
                for update in updates:
                    operations.append(
                        UpdateOne(
                            update['filter'],
                            {'$set': update['update']},
                            upsert=update.get('upsert', False)
                        )
                    )
                result = collection.bulk_write(operations)
                return result.modified_count
        except Exception as e:
            logger.error(f"Bulk update error: {e}")
        return 0
    
    # DELETE Operations
    def delete_one(self, collection_name: str, query: Dict) -> bool:
        """Delete single document"""
        try:
            collection = self.db.get_collection(collection_name)
            if collection is not None:
                result = collection.delete_one(query)
                return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Delete error: {e}")
        return False

    def delete_many(self, collection_name: str, query: Dict) -> int:
        """Delete multiple documents"""
        try:
            collection = self.db.get_collection(collection_name)
            if collection is not None:
                result = collection.delete_many(query)
                return result.deleted_count
        except Exception as e:
            logger.error(f"Delete many error: {e}")
        return 0

    # ANALYTICS-SPECIFIC OPERATIONS (Week 3)

    async def store_volatility_surface(self, currency: str, surface_data: Dict) -> bool:
        """Store volatility surface with history tracking."""
        try:
            adb = await self.db.get_database_async()

            # Store current surface (replace existing)
            await adb.volatility_surfaces.replace_one(
                {'currency': currency},
                surface_data,
                upsert=True
            )

            # Store in history
            surface_data['_id'] = f"{currency}_{surface_data['timestamp'].strftime('%Y%m%d_%H%M%S')}"
            await adb.volatility_surfaces_history.insert_one(surface_data.copy())

            return True
        except Exception as e:
            logger.error(f"Store volatility surface error: {e}")
            return False

    async def get_latest_volatility_surface(self, currency: str) -> Optional[Dict]:
        """Get latest volatility surface for currency."""
        try:
            adb = await self.db.get_database_async()
            return await adb.volatility_surfaces.find_one({'currency': currency})
        except Exception as e:
            logger.error(f"Get volatility surface error: {e}")
            return None

    async def store_options_chain_analytics(self, currency: str, analytics_data: Dict) -> bool:
        """Store options chain analytics."""
        try:
            adb = await self.db.get_database_async()

            # Replace current analytics
            await adb.options_chain_analytics.replace_one(
                {'currency': currency},
                analytics_data,
                upsert=True
            )

            return True
        except Exception as e:
            logger.error(f"Store chain analytics error: {e}")
            return False

    async def get_options_chain_analytics(self, currency: str) -> Optional[Dict]:
        """Get latest options chain analytics."""
        try:
            adb = await self.db.get_database_async()
            return await adb.options_chain_analytics.find_one({'currency': currency})
        except Exception as e:
            logger.error(f"Get chain analytics error: {e}")
            return None

    async def store_greeks_snapshot(self, portfolio_id: str, currency: str, greeks_data: Dict) -> bool:
        """Store portfolio Greeks snapshot."""
        try:
            adb = await self.db.get_database_async()

            greeks_data['portfolio_id'] = portfolio_id
            greeks_data['currency'] = currency

            await adb.greeks_snapshots.insert_one(greeks_data)

            return True
        except Exception as e:
            logger.error(f"Store Greeks snapshot error: {e}")
            return False

    async def get_latest_greeks_snapshot(self, portfolio_id: str, currency: str) -> Optional[Dict]:
        """Get latest Greeks snapshot for portfolio."""
        try:
            adb = await self.db.get_database_async()
            return await adb.greeks_snapshots.find_one(
                {'portfolio_id': portfolio_id, 'currency': currency},
                sort=[('timestamp', -1)]
            )
        except Exception as e:
            logger.error(f"Get Greeks snapshot error: {e}")
            return None

    async def store_iv_points_bulk(self, currency: str, iv_points: List[Dict]) -> bool:
        """Store implied volatility points in bulk."""
        try:
            adb = await self.db.get_database_async()

            if iv_points:
                await adb.implied_volatility_points.insert_many(iv_points)
                return True

            return False
        except Exception as e:
            logger.error(f"Store IV points error: {e}")
            return False

    async def get_iv_points_by_currency(self, currency: str,
                                       hours_back: int = 24) -> List[Dict]:
        """Get IV points for currency within time window."""
        try:
            adb = await self.db.get_database_async()

            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            cursor = adb.implied_volatility_points.find({
                'currency': currency,
                'timestamp': {'$gte': cutoff_time}
            }).sort('timestamp', -1)

            return await cursor.to_list(length=1000)
        except Exception as e:
            logger.error(f"Get IV points error: {e}")
            return []

    async def get_options_by_currency_expiry(self, currency: str,
                                          expiry_days: Optional[int] = None) -> List[Dict]:
        """Get options data filtered by currency and optional expiry."""
        try:
            adb = await self.db.get_database_async()

            query = {'underlying': currency}

            if expiry_days:
                cutoff_date = datetime.now() + timedelta(days=expiry_days)
                query['expiry'] = {'$lte': cutoff_date}

            cursor = adb.options_data.find(query).sort('timestamp', -1).limit(500)
            return await cursor.to_list(length=500)

        except Exception as e:
            logger.error(f"Get options by currency/expiry error: {e}")
            return []

    async def cleanup_old_analytics_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up old analytics data beyond retention period."""
        try:
            adb = await self.db.get_database_async()
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)

            cleanup_results = {}

            # Clean up old volatility surfaces history
            result = await adb.volatility_surfaces_history.delete_many({
                'timestamp': {'$lt': cutoff_date}
            })
            cleanup_results['volatility_surfaces_history'] = result.deleted_count

            # Clean up old IV points
            result = await adb.implied_volatility_points.delete_many({
                'timestamp': {'$lt': cutoff_date}
            })
            cleanup_results['iv_points'] = result.deleted_count

            # Clean up old Greeks snapshots
            result = await adb.greeks_snapshots.delete_many({
                'timestamp': {'$lt': cutoff_date}
            })
            cleanup_results['greeks_snapshots'] = result.deleted_count

            return cleanup_results

        except Exception as e:
            logger.error(f"Cleanup analytics data error: {e}")
            return {}

# Create singleton instance
db_ops = DatabaseOperations()
