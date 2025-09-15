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

    # === Risk Analytics Database Operations ===

    async def store_portfolio_data(self, portfolio_data: Dict) -> Optional[str]:
        """Store portfolio configuration and allocation data"""
        try:
            adb = await self.db.get_database_async()
            portfolio_data['timestamp'] = datetime.utcnow()
            result = await adb.portfolio_data.insert_one(portfolio_data)
            logger.info(f"Portfolio data stored for {portfolio_data.get('portfolio_id')}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error storing portfolio data: {e}")
            return None

    async def get_portfolio_data(self, portfolio_id: str) -> Optional[Dict]:
        """Get latest portfolio configuration"""
        try:
            adb = await self.db.get_database_async()
            portfolio_doc = await adb.portfolio_data.find_one(
                {'portfolio_id': portfolio_id},
                sort=[('timestamp', -1)]
            )
            return portfolio_doc
        except Exception as e:
            logger.error(f"Error fetching portfolio data: {e}")
            return None

    async def store_risk_metrics(self, portfolio_id: str, risk_metrics: Dict) -> Optional[str]:
        """Store calculated risk metrics"""
        try:
            adb = await self.db.get_database_async()
            risk_doc = {
                'portfolio_id': portfolio_id,
                'metrics': risk_metrics,
                'timestamp': datetime.utcnow(),
                'calculated_by': 'PortfolioRiskAnalyzer'
            }
            result = await adb.risk_metrics.insert_one(risk_doc)
            logger.info(f"Risk metrics stored for portfolio {portfolio_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error storing risk metrics: {e}")
            return None

    async def get_latest_risk_metrics(self, portfolio_id: str) -> Optional[Dict]:
        """Get latest risk metrics for a portfolio"""
        try:
            adb = await self.db.get_database_async()
            risk_doc = await adb.risk_metrics.find_one(
                {'portfolio_id': portfolio_id},
                sort=[('timestamp', -1)]
            )
            return risk_doc['metrics'] if risk_doc else None
        except Exception as e:
            logger.error(f"Error fetching risk metrics: {e}")
            return None

    async def store_performance_report(self, portfolio_id: str, performance_report: Dict) -> Optional[str]:
        """Store performance analysis report"""
        try:
            adb = await self.db.get_database_async()
            report_doc = {
                'portfolio_id': portfolio_id,
                'performance_report': performance_report,
                'timestamp': datetime.utcnow(),
                'generated_by': 'QuantStatsAnalyzer'
            }
            result = await adb.performance_reports.insert_one(report_doc)
            logger.info(f"Performance report stored for portfolio {portfolio_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error storing performance report: {e}")
            return None

    async def get_latest_performance_report(self, portfolio_id: str) -> Optional[Dict]:
        """Get latest performance report for a portfolio"""
        try:
            adb = await self.db.get_database_async()
            report_doc = await adb.performance_reports.find_one(
                {'portfolio_id': portfolio_id},
                sort=[('timestamp', -1)]
            )
            return report_doc['performance_report'] if report_doc else None
        except Exception as e:
            logger.error(f"Error fetching performance report: {e}")
            return None

    async def store_performance_attribution(self, portfolio_id: str, attribution_type: str, attribution_results: Dict) -> Optional[str]:
        """Store performance attribution analysis"""
        try:
            adb = await self.db.get_database_async()
            attribution_doc = {
                'portfolio_id': portfolio_id,
                'attribution_type': attribution_type,
                'attribution_results': attribution_results,
                'timestamp': datetime.utcnow()
            }
            result = await adb.performance_attribution.insert_one(attribution_doc)
            logger.info(f"Performance attribution ({attribution_type}) stored for portfolio {portfolio_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error storing performance attribution: {e}")
            return None

    async def get_price_data_for_asset(self, asset: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Get historical price data for an asset"""
        try:
            adb = await self.db.get_database_async()
            cursor = adb.price_data.find({
                'symbol': asset,
                'timestamp': {'$gte': start_date, '$lte': end_date}
            }).sort('timestamp', 1)
            return await cursor.to_list(length=None)
        except Exception as e:
            logger.error(f"Error fetching price data for {asset}: {e}")
            return []

    async def get_portfolio_list(self, user_id: str = None) -> List[Dict]:
        """Get list of available portfolios"""
        try:
            adb = await self.db.get_database_async()
            query = {'user_id': user_id} if user_id else {}
            cursor = adb.portfolio_data.find(query).sort('timestamp', -1)
            portfolios = await cursor.to_list(length=None)

            # Get unique portfolio IDs with latest data
            unique_portfolios = {}
            for portfolio in portfolios:
                portfolio_id = portfolio.get('portfolio_id')
                if portfolio_id not in unique_portfolios:
                    unique_portfolios[portfolio_id] = portfolio

            return list(unique_portfolios.values())
        except Exception as e:
            logger.error(f"Error fetching portfolio list: {e}")
            return []

    async def store_sector_risk_metrics(self, portfolio_id: str, sector_risk_data: Dict) -> Optional[str]:
        """Store sector-based risk analysis"""
        try:
            adb = await self.db.get_database_async()
            sector_doc = {
                'portfolio_id': portfolio_id,
                'sector_allocation_risk': sector_risk_data,
                'timestamp': datetime.utcnow()
            }
            result = await adb.sector_risk_metrics.insert_one(sector_doc)
            logger.info(f"Sector risk metrics stored for portfolio {portfolio_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error storing sector risk metrics: {e}")
            return None

    async def get_risk_metrics_history(self, portfolio_id: str, days: int = 30) -> List[Dict]:
        """Get historical risk metrics for trend analysis"""
        try:
            adb = await self.db.get_database_async()
            start_date = datetime.utcnow() - timedelta(days=days)
            cursor = adb.risk_metrics.find({
                'portfolio_id': portfolio_id,
                'timestamp': {'$gte': start_date}
            }).sort('timestamp', 1)
            return await cursor.to_list(length=None)
        except Exception as e:
            logger.error(f"Error fetching risk metrics history: {e}")
            return []

    async def store_comprehensive_tearsheet(self, portfolio_id: str, tearsheet_data: Dict) -> Optional[str]:
        """Store comprehensive performance tearsheet"""
        try:
            adb = await self.db.get_database_async()
            tearsheet_doc = {
                'portfolio_id': portfolio_id,
                'comprehensive_tearsheet': tearsheet_data,
                'timestamp': datetime.utcnow()
            }
            result = await adb.comprehensive_tearsheets.insert_one(tearsheet_doc)
            logger.info(f"Comprehensive tearsheet stored for portfolio {portfolio_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error storing comprehensive tearsheet: {e}")
            return None
    
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
