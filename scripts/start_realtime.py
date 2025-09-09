#!/usr/bin/env python3
"""Start background real-time data collection into MongoDB.

This launches DataManager and schedules periodic Deribit options snapshots
for BTC and ETH (every 1 minute by default). Requires MongoDB running.
"""

import asyncio
from src.data.collectors.data_manager import DataManager, CollectionTask, DataSource, UpdateFrequency


async def main():
    manager = DataManager(enable_yfinance=False, enable_deribit=True, enable_scheduler=True)

    # Add a 1-minute options task for BTC/ETH
    manager.add_task(
        CollectionTask(
            name="options-btc-eth-1m",
            source=DataSource.DERIBIT,
            symbols=["BTC", "ETH"],
            frequency=UpdateFrequency.HIGH,
            params={"store": True},
        )
    )

    # Run scheduler until Ctrl+C
    try:
        await manager.run_scheduler()
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop_scheduler()


if __name__ == "__main__":
    asyncio.run(main())

