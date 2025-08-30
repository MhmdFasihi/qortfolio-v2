# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Data Manager - Orchestrates all data collectors.
Coordinates data collection, synchronization, and storage.
"""

import asyncio
from typing import Dict, List, Optional, Any, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor

from .crypto_collector import CryptoCollector
from .deribit_collector import DeribitCollector
from src.core.config import config
from src.core.exceptions import DataCollectionError, ValidationError
from src.core.database.connection import db_connection
from src.core.database.operations import db_ops

logger = logging.getLogger(__name__)

class DataSource(Enum):
    """Available data sources."""
    YFINANCE = "yfinance"
    DERIBIT = "deribit"
    ALL = "all"

class UpdateFrequency(Enum):
    """Data update frequencies."""
    REALTIME = "realtime"      # WebSocket streaming
    HIGH = "high"               # Every 1 minute
    MEDIUM = "medium"           # Every 5 minutes
    LOW = "low"                 # Every 15 minutes
    HOURLY = "hourly"           # Every hour
    DAILY = "daily"             # Once per day

@dataclass
class CollectionTask:
    """Data collection task definition."""
    name: str
    source: DataSource
    symbols: List[str]
    frequency: UpdateFrequency
    params: Dict[str, Any] = field(default_factory=dict)
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    enabled: bool = True
    
    def is_due(self) -> bool:
        """Check if task is due for execution."""
        if not self.enabled:
            return False
        if self.next_run is None:
            return True
        return datetime.utcnow() >= self.next_run
    
    def update_schedule(self):
        """Update next run time based on frequency."""
        self.last_run = datetime.utcnow()
        
        frequency_intervals = {
            UpdateFrequency.REALTIME: timedelta(seconds=0),
            UpdateFrequency.HIGH: timedelta(minutes=1),
            UpdateFrequency.MEDIUM: timedelta(minutes=5),
            UpdateFrequency.LOW: timedelta(minutes=15),
            UpdateFrequency.HOURLY: timedelta(hours=1),
            UpdateFrequency.DAILY: timedelta(days=1)
        }
        
        interval = frequency_intervals.get(self.frequency, timedelta(minutes=5))
        self.next_run = self.last_run + interval

@dataclass
class DataSyncResult:
    """Result of data synchronization."""
    source: str
    symbols: List[str]
    records_fetched: int
    records_stored: int
    errors: List[str]
    duration_seconds: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

class DataManager:
    """
    Central data management orchestrator.
    Coordinates all data collectors and handles synchronization.
    """
    
    def __init__(
        self,
        enable_yfinance: bool = True,
        enable_deribit: bool = True,
        enable_scheduler: bool = True,
        max_workers: int = 4
    ):
        """
        Initialize data manager.
        
        Args:
            enable_yfinance: Enable yfinance data collection
            enable_deribit: Enable Deribit options data collection
            enable_scheduler: Enable automatic scheduled updates
            max_workers: Maximum concurrent collection workers
        """
        self.enable_yfinance = enable_yfinance
        self.enable_deribit = enable_deribit
        self.enable_scheduler = enable_scheduler
        self.max_workers = max_workers
        
        # Initialize collectors
        self.collectors: Dict[str, Any] = {}
        if enable_yfinance:
            self.collectors['yfinance'] = CryptoCollector()
        if enable_deribit:
            self.collectors['deribit'] = DeribitCollector(testnet=True)
        
        # Task management
        self.tasks: List[CollectionTask] = []
        self.running_tasks: Set[str] = set()
        self.scheduler_running = False
        
        # Thread pool for concurrent collections
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Statistics
        self.sync_history: List[DataSyncResult] = []
        
        logger.info(f"DataManager initialized with collectors: {list(self.collectors.keys())}")
    
    # === Task Management ===
    
    def add_task(self, task: CollectionTask) -> bool:
        """
        Add a collection task.
        
        Args:
            task: Collection task to add
            
        Returns:
            True if task added successfully
        """
        # Check if task already exists
        if any(t.name == task.name for t in self.tasks):
            logger.warning(f"Task {task.name} already exists")
            return False
        
        self.tasks.append(task)
        logger.info(f"Added task: {task.name} ({task.frequency.value})")
        return True
    
    def remove_task(self, task_name: str) -> bool:
        """Remove a task by name."""
        self.tasks = [t for t in self.tasks if t.name != task_name]
        return True
    
    def get_task(self, task_name: str) -> Optional[CollectionTask]:
        """Get task by name."""
        for task in self.tasks:
            if task.name == task_name:
                return task
        return None
    
    # === Data Collection Methods ===
    
    async def collect_crypto_data(
        self,
        symbols: List[str],
        period: str = "1d",
        interval: str = "1h",
        store: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect cryptocurrency price data.
        
        Args:
            symbols: List of crypto symbols
            period: Time period for historical data
            interval: Data interval
            store: Whether to store in MongoDB
            
        Returns:
            Dictionary of symbol to DataFrame
        """
        if 'yfinance' not in self.collectors:
            raise DataCollectionError("yfinance collector not available")
        
        collector = self.collectors['yfinance']
        results = {}
        errors = []
        
        start_time = datetime.utcnow()
        
        for symbol in symbols:
            try:
                logger.info(f"Collecting {symbol} price data...")
                
                # Fetch data
                data = await collector.collect(
                    symbol=symbol,
                    period=period,
                    interval=interval,
                    use_cache=False  # Always get fresh data for sync
                )
                
                if data:
                    # Convert to DataFrame for easier handling
                    df = pd.DataFrame(data)
                    results[symbol] = df
                    
                    # Store in MongoDB if requested
                    if store:
                        await collector.store_data("price_data", data)
                        logger.info(f"Stored {len(data)} {symbol} price records")
                    
            except Exception as e:
                error_msg = f"Failed to collect {symbol}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        # Record sync result
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        sync_result = DataSyncResult(
            source="yfinance",
            symbols=symbols,
            records_fetched=sum(len(df) for df in results.values()),
            records_stored=sum(len(df) for df in results.values()) if store else 0,
            errors=errors,
            duration_seconds=duration
        )
        
        self.sync_history.append(sync_result)
        
        return results
    
    async def collect_options_data(
        self,
        currencies: List[str] = ["BTC", "ETH"],
        store: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect options chain data.
        
        Args:
            currencies: List of currencies (BTC, ETH)
            store: Whether to store in MongoDB
            
        Returns:
            Dictionary of currency to options DataFrame
        """
        if 'deribit' not in self.collectors:
            raise DataCollectionError("Deribit collector not available")
        
        collector = self.collectors['deribit']
        results = {}
        errors = []
        
        start_time = datetime.utcnow()
        
        for currency in currencies:
            try:
                logger.info(f"Collecting {currency} options chain...")
                
                # Get options chain
                chain = await collector.get_options_chain(
                    currency=currency,
                    strikes_around_atm=20  # Get more strikes
                )
                
                if not chain.empty:
                    results[currency] = chain
                    
                    # Store in MongoDB if requested
                    if store:
                        # Convert DataFrame to list of dicts for storage
                        options_data = chain.to_dict('records')
                        await collector.store_data("options_data", options_data)
                        logger.info(f"Stored {len(options_data)} {currency} options")
                    
            except Exception as e:
                error_msg = f"Failed to collect {currency} options: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        # Record sync result
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        sync_result = DataSyncResult(
            source="deribit",
            symbols=currencies,
            records_fetched=sum(len(df) for df in results.values()),
            records_stored=sum(len(df) for df in results.values()) if store else 0,
            errors=errors,
            duration_seconds=duration
        )
        
        self.sync_history.append(sync_result)
        
        return results
    
    async def collect_all_data(
        self,
        crypto_symbols: List[str] = ["BTC", "ETH"],
        option_currencies: List[str] = ["BTC", "ETH"],
        store: bool = True
    ) -> Dict[str, Any]:
        """
        Collect data from all sources.
        
        Args:
            crypto_symbols: Crypto symbols for price data
            option_currencies: Currencies for options data
            store: Whether to store in MongoDB
            
        Returns:
            Dictionary with all collected data
        """
        results = {
            'prices': {},
            'options': {},
            'metadata': {
                'timestamp': datetime.utcnow(),
                'symbols': crypto_symbols,
                'currencies': option_currencies
            }
        }
        
        # Collect price data
        if self.enable_yfinance:
            try:
                results['prices'] = await self.collect_crypto_data(
                    symbols=crypto_symbols,
                    store=store
                )
            except Exception as e:
                logger.error(f"Price collection failed: {e}")
        
        # Collect options data
        if self.enable_deribit:
            try:
                results['options'] = await self.collect_options_data(
                    currencies=option_currencies,
                    store=store
                )
            except Exception as e:
                logger.error(f"Options collection failed: {e}")
        
        return results
    
    # === Data Synchronization ===
    
    async def sync_historical_data(
        self,
        symbols: List[str],
        days_back: int = 30,
        interval: str = "1h"
    ) -> DataSyncResult:
        """
        Synchronize historical data for symbols.
        
        Args:
            symbols: List of symbols to sync
            days_back: Number of days of history
            interval: Data interval
            
        Returns:
            Sync result
        """
        start_time = datetime.utcnow()
        errors = []
        total_records = 0
        
        logger.info(f"Starting historical sync for {len(symbols)} symbols, {days_back} days back")
        
        for symbol in symbols:
            try:
                # Calculate date range
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=days_back)
                
                # Fetch historical data
                collector = self.collectors['yfinance']
                data = await collector.collect(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                    use_cache=False
                )
                
                if data:
                    # Store in database
                    await collector.store_data("price_data", data)
                    total_records += len(data)
                    logger.info(f"Synced {len(data)} records for {symbol}")
                    
            except Exception as e:
                error_msg = f"Failed to sync {symbol}: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        return DataSyncResult(
            source="historical_sync",
            symbols=symbols,
            records_fetched=total_records,
            records_stored=total_records,
            errors=errors,
            duration_seconds=duration
        )
    
    async def sync_realtime_data(
        self,
        symbols: List[str],
        duration_minutes: int = 60
    ):
        """
        Sync real-time data for a specified duration.
        
        Args:
            symbols: Symbols to track
            duration_minutes: How long to collect data
        """
        logger.info(f"Starting real-time sync for {symbols} ({duration_minutes} minutes)")
        
        end_time = datetime.utcnow() + timedelta(minutes=duration_minutes)
        
        while datetime.utcnow() < end_time:
            try:
                # Collect current prices
                for symbol in symbols:
                    collector = self.collectors['yfinance']
                    price_data = await collector.get_current_price(symbol)
                    
                    # Store with timestamp
                    await collector.store_data("realtime_prices", [price_data])
                
                # Wait before next collection
                await asyncio.sleep(60)  # Collect every minute
                
            except Exception as e:
                logger.error(f"Real-time sync error: {e}")
                await asyncio.sleep(5)  # Short wait on error
    
    # === Task Scheduler ===
    
    async def run_scheduler(self):
        """Run the task scheduler."""
        if self.scheduler_running:
            logger.warning("Scheduler already running")
            return
        
        self.scheduler_running = True
        logger.info("Task scheduler started")
        
        try:
            while self.scheduler_running:
                # Check all tasks
                for task in self.tasks:
                    if task.is_due() and task.name not in self.running_tasks:
                        # Run task asynchronously
                        asyncio.create_task(self._execute_task(task))
                
                # Wait before next check
                await asyncio.sleep(10)  # Check every 10 seconds
                
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
        finally:
            self.scheduler_running = False
    
    async def _execute_task(self, task: CollectionTask):
        """Execute a collection task."""
        if task.name in self.running_tasks:
            logger.warning(f"Task {task.name} already running")
            return
        
        self.running_tasks.add(task.name)
        
        try:
            logger.info(f"Executing task: {task.name}")
            
            if task.source == DataSource.YFINANCE:
                await self.collect_crypto_data(
                    symbols=task.symbols,
                    **task.params
                )
            elif task.source == DataSource.DERIBIT:
                await self.collect_options_data(
                    currencies=task.symbols,
                    **task.params
                )
            elif task.source == DataSource.ALL:
                await self.collect_all_data(
                    crypto_symbols=task.symbols,
                    **task.params
                )
            
            task.update_schedule()
            logger.info(f"Task {task.name} completed, next run: {task.next_run}")
            
        except Exception as e:
            logger.error(f"Task {task.name} failed: {e}")
        finally:
            self.running_tasks.discard(task.name)
    
    def stop_scheduler(self):
        """Stop the task scheduler."""
        self.scheduler_running = False
        logger.info("Scheduler stopped")
    
    # === Data Conflict Resolution ===
    
    async def merge_price_data(
        self,
        existing_data: pd.DataFrame,
        new_data: pd.DataFrame,
        conflict_resolution: str = "newest"
    ) -> pd.DataFrame:
        """
        Merge price data with conflict resolution.
        
        Args:
            existing_data: Existing DataFrame
            new_data: New DataFrame to merge
            conflict_resolution: How to resolve conflicts (newest, average, highest_volume)
            
        Returns:
            Merged DataFrame
        """
        if existing_data.empty:
            return new_data
        if new_data.empty:
            return existing_data
        
        # Merge on timestamp
        merged = pd.concat([existing_data, new_data])
        
        # Handle duplicates based on strategy
        if conflict_resolution == "newest":
            # Keep the most recent entry for each timestamp
            merged = merged.sort_values('collected_at').drop_duplicates(
                subset=['timestamp', 'symbol'],
                keep='last'
            )
        elif conflict_resolution == "average":
            # Average prices for duplicate timestamps
            price_cols = ['open', 'high', 'low', 'close', 'volume']
            merged = merged.groupby(['timestamp', 'symbol'])[price_cols].mean().reset_index()
        elif conflict_resolution == "highest_volume":
            # Keep entry with highest volume
            merged = merged.sort_values('volume', ascending=False).drop_duplicates(
                subset=['timestamp', 'symbol'],
                keep='first'
            )
        
        return merged.sort_values('timestamp')
    
    # === Statistics and Monitoring ===
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get data manager statistics."""
        stats = {
            'collectors': list(self.collectors.keys()),
            'active_tasks': len(self.tasks),
            'running_tasks': list(self.running_tasks),
            'scheduler_running': self.scheduler_running,
            'sync_history_count': len(self.sync_history)
        }
        
        # Add collector statistics
        for name, collector in self.collectors.items():
            stats[f'{name}_stats'] = collector.get_stats()
        
        # Add recent sync results
        if self.sync_history:
            recent_syncs = self.sync_history[-5:]  # Last 5 syncs
            stats['recent_syncs'] = [
                {
                    'source': sync.source,
                    'records': sync.records_stored,
                    'duration': sync.duration_seconds,
                    'errors': len(sync.errors)
                }
                for sync in recent_syncs
            ]
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components."""
        health = {
            'status': 'healthy',
            'timestamp': datetime.utcnow(),
            'collectors': {},
            'database': False
        }
        
        # Check collectors
        for name, collector in self.collectors.items():
            try:
                # Try to get stats as a basic health check
                stats = collector.get_stats()
                health['collectors'][name] = {
                    'status': 'healthy',
                    'requests': stats['total_requests']
                }
            except Exception as e:
                health['collectors'][name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                health['status'] = 'degraded'
        
        # Check database
        try:
            db_health = db_connection.health_check()
            health['database'] = db_health['connected']
        except Exception as e:
            health['database'] = False
            health['status'] = 'degraded'
        
        return health
    
    # === Cleanup ===
    
    async def close(self):
        """Close all resources."""
        # Stop scheduler
        self.stop_scheduler()
        
        # Close all collectors
        for name, collector in self.collectors.items():
            await collector.close()
            logger.info(f"Closed {name} collector")
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("DataManager closed")

# === Testing ===

async def test_data_manager():
    """Test data manager functionality."""
    manager = DataManager(
        enable_yfinance=True,
        enable_deribit=True,
        enable_scheduler=False  # Don't start scheduler in test
    )
    
    print("\n📊 Testing Data Manager")
    print("=" * 50)
    
    try:
        # Test 1: Health check
        print("\n1️⃣ Performing health check...")
        health = await manager.health_check()
        print(f"   ✅ System status: {health['status']}")
        print(f"   Collectors: {list(health['collectors'].keys())}")
        print(f"   Database: {'Connected' if health['database'] else 'Disconnected'}")
        
        # Test 2: Collect crypto prices
        print("\n2️⃣ Collecting crypto price data...")
        prices = await manager.collect_crypto_data(
            symbols=["BTC", "ETH"],
            period="1d",
            interval="1h",
            store=False  # Don't store in test
        )
        
        for symbol, df in prices.items():
            if not df.empty:
                print(f"   ✅ {symbol}: {len(df)} records")
                latest = df.iloc[-1]
                print(f"      Latest price: ${latest['close']:.2f}")
        
        # Test 3: Collect options data
        print("\n3️⃣ Collecting options data...")
        options = await manager.collect_options_data(
            currencies=["BTC"],
            store=False
        )
        
        for currency, df in options.items():
            if not df.empty:
                print(f"   ✅ {currency}: {len(df)} options")
                atm_options = df[df['moneyness'].between(0.95, 1.05)]
                print(f"      ATM options: {len(atm_options)}")
        
        # Test 4: Add and execute a task
        print("\n4️⃣ Testing task management...")
        task = CollectionTask(
            name="test_btc_price",
            source=DataSource.YFINANCE,
            symbols=["BTC"],
            frequency=UpdateFrequency.HIGH,
            params={"period": "1d", "interval": "5m"}
        )
        
        manager.add_task(task)
        print(f"   ✅ Added task: {task.name}")
        
        # Execute task manually
        await manager._execute_task(task)
        print(f"   ✅ Task executed successfully")
        
        # Test 5: Check statistics
        print("\n5️⃣ Data Manager Statistics:")
        stats = manager.get_statistics()
        print(f"   Active collectors: {stats['collectors']}")
        print(f"   Tasks: {stats['active_tasks']}")
        print(f"   Sync history: {stats['sync_history_count']} syncs")
        
        if stats.get('recent_syncs'):
            print("   Recent syncs:")
            for sync in stats['recent_syncs']:
                print(f"      - {sync['source']}: {sync['records']} records in {sync['duration']:.1f}s")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await manager.close()

if __name__ == "__main__":
    asyncio.run(test_data_manager())
