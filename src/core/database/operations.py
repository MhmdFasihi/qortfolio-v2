"""Database CRUD Operations for MongoDB"""

from typing import Dict, List, Optional, Any
from datetime import datetime
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

# Create singleton instance
db_ops = DatabaseOperations()
