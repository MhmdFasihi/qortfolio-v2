# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Database operations for Qortfolio V2.
Provides CRUD operations for all collections.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from pymongo import ASCENDING, DESCENDING
from .connection import db_connection
from .models import OptionsData, PriceData, PortfolioPosition, RiskMetrics
from ..exceptions import DatabaseOperationError

logger = logging.getLogger(__name__)

class DatabaseOperations:
    """Database operations manager."""
    
    def __init__(self):
        self.connection = db_connection
        # Defer establishing a sync connection to avoid noisy failures at import time.
        # Tests and optional code paths already guard on None.
        self._db = None

    @property
    def db(self):
        """Expose sync database handle if available (for tests)."""
        return self._db
        
    # === Options Data Operations ===
    
    async def insert_options_data(self, options: OptionsData) -> str:
        """Insert options data."""
        try:
            db = await self.connection.get_database_async()
            result = await db.options_data.insert_one(options.to_dict())
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Failed to insert options data: {e}")
            raise DatabaseOperationError(f"Insert failed: {e}")
    
    async def get_latest_options(
        self, 
        underlying: str, 
        limit: int = 100
    ) -> List[Dict]:
        """Get latest options data for an underlying."""
        try:
            db = await self.connection.get_database_async()
            cursor = db.options_data.find(
                {"underlying": underlying}
            ).sort("timestamp", DESCENDING).limit(limit)
            
            return await cursor.to_list(length=limit)
        except Exception as e:
            logger.error(f"Failed to get options data: {e}")
            raise DatabaseOperationError(f"Query failed: {e}")
    
    # === Price Data Operations ===
    
    async def insert_price_data(self, price: PriceData) -> str:
        """Insert price data."""
        try:
            db = await self.connection.get_database_async()
            result = await db.price_data.insert_one(price.to_dict())
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Failed to insert price data: {e}")
            raise DatabaseOperationError(f"Insert failed: {e}")
    
    async def get_price_history(
        self,
        symbol: str,
        days: int = 30
    ) -> List[Dict]:
        """Get price history for a symbol."""
        try:
            db = await self.connection.get_database_async()
            start_date = datetime.utcnow() - timedelta(days=days)
            
            cursor = db.price_data.find({
                "symbol": symbol,
                "timestamp": {"$gte": start_date}
            }).sort("timestamp", ASCENDING)
            
            return await cursor.to_list(length=None)
        except Exception as e:
            logger.error(f"Failed to get price history: {e}")
            raise DatabaseOperationError(f"Query failed: {e}")
    
    # === Portfolio Operations ===
    
    async def update_portfolio_position(
        self,
        position: PortfolioPosition
    ) -> bool:
        """Update or insert portfolio position."""
        try:
            db = await self.connection.get_database_async()
            result = await db.portfolio_positions.update_one(
                {
                    "user_id": position.user_id,
                    "symbol": position.symbol
                },
                {"$set": position.to_dict()},
                upsert=True
            )
            return result.modified_count > 0 or result.upserted_id is not None
        except Exception as e:
            logger.error(f"Failed to update portfolio position: {e}")
            raise DatabaseOperationError(f"Update failed: {e}")
    
    # === Risk Metrics Operations ===
    
    async def store_risk_metrics(self, metrics: RiskMetrics) -> str:
        """Store risk metrics."""
        try:
            db = await self.connection.get_database_async()
            result = await db.risk_metrics.insert_one(metrics.to_dict())
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Failed to store risk metrics: {e}")
            raise DatabaseOperationError(f"Insert failed: {e}")
    
    # === Utility Operations ===
    
    async def create_indexes(self):
        """Create database indexes for performance."""
        try:
            db = await self.connection.get_database_async()
            
            # Options data indexes
            await db.options_data.create_index([
                ("underlying", ASCENDING),
                ("timestamp", DESCENDING)
            ])
            await db.options_data.create_index("expiry")
            
            # Price data indexes
            await db.price_data.create_index([
                ("symbol", ASCENDING),
                ("timestamp", DESCENDING)
            ])
            
            # Portfolio indexes
            await db.portfolio_positions.create_index([
                ("user_id", ASCENDING),
                ("symbol", ASCENDING)
            ])
            
            logger.info("✅ Database indexes created successfully")
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
            raise DatabaseOperationError(f"Index creation failed: {e}")

# Global database operations instance
db_ops = DatabaseOperations()

if __name__ == "__main__":
    # Test database operations
    async def test_operations():
        try:
            await db_ops.create_indexes()
            print("✅ Database operations initialized")
        except Exception as e:
            print(f"❌ Database operations test failed: {e}")
    
    asyncio.run(test_operations())
