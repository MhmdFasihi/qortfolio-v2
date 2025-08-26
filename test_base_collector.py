#!/usr/bin/env python3
"""Test the base data collector functionality."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data.collectors.base_collector import BaseDataCollector

class TestCollector(BaseDataCollector):
    """Simple test implementation of base collector."""
    
    async def fetch_data(self, symbol: str = "TEST") -> dict:
        """Fetch test data."""
        return {
            "symbol": symbol,
            "price": 100.0,
            "timestamp": "2025-01-01T00:00:00Z"
        }
    
    async def validate_data(self, data: dict) -> bool:
        """Validate test data."""
        return "symbol" in data and "price" in data
    
    async def process_data(self, raw_data: dict) -> dict:
        """Process test data."""
        return {
            **raw_data,
            "processed": True
        }

async def test():
    """Test the base collector."""
    print("Testing Base Data Collector...")
    
    # Create test collector
    collector = TestCollector(
        name="test",
        rate_limit=2.0,  # 2 calls per second
        cache_ttl=60
    )
    
    # Test data collection
    data = await collector.collect(symbol="BTC")
    print(f"✅ Collected data: {data}")
    
    # Test caching (should hit cache)
    data2 = await collector.collect(symbol="BTC")
    print(f"✅ Cached data: {data2}")
    
    # Check stats
    stats = collector.get_stats()
    print(f"\n📊 Collector Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Cleanup
    await collector.close()
    
    print("\n✅ Base collector test passed!")

if __name__ == "__main__":
    asyncio.run(test())
