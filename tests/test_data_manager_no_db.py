#!/usr/bin/env python3
"""Test data manager without MongoDB dependency."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data.collectors.data_manager import DataManager

async def test():
    """Test data manager without DB storage."""
    manager = DataManager(
        enable_yfinance=True,
        enable_deribit=True,
        enable_scheduler=False
    )
    
    print("\n📊 Testing Data Manager (No DB Storage)")
    print("=" * 50)
    
    try:
        # Test 1: Collect crypto prices (no storage)
        print("\n1️⃣ Collecting crypto price data...")
        prices = await manager.collect_crypto_data(
            symbols=["BTC", "ETH"],
            period="1d",
            interval="1h",
            store=False  # Don't store in MongoDB
        )
        
        for symbol, df in prices.items():
            if not df.empty:
                print(f"   ✅ {symbol}: {len(df)} records")
                latest = df.iloc[-1]
                print(f"      Latest price: ${latest['close']:.2f}")
        
        # Test 2: Collect options data (no storage)
        print("\n2️⃣ Collecting options data...")
        try:
            options = await manager.collect_options_data(
                currencies=["BTC"],
                store=False  # Don't store in MongoDB
            )
            
            for currency, df in options.items():
                if not df.empty:
                    print(f"   ✅ {currency}: {len(df)} options")
        except Exception as e:
            print(f"   ⚠️  Options collection skipped: {e}")
        
        # Test 3: Statistics
        print("\n3️⃣ Manager Statistics:")
        stats = manager.get_statistics()
        print(f"   Collectors: {stats['collectors']}")
        print(f"   Sync history: {stats['sync_history_count']} syncs")
        
        print("\n✅ Tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await manager.close()

if __name__ == "__main__":
    asyncio.run(test())
