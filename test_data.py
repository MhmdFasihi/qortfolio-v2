import asyncio

# Example integration with your existing data
from src.data.collectors.deribit_collector import DeribitCollector
from src.models.options.options_chain import OptionsChainProcessor


async def main():
    # Get options data (async)
    collector = DeribitCollector()
    try:
        df = await collector.get_options_chain('BTC')
    finally:
        await collector.close()

    # Convert DataFrame to the expected raw options format for the processor
    options_data = []
    if df is not None and not df.empty:
        records = df.to_dict('records')
        for r in records:
            options_data.append({
                'instrument_name': r.get('symbol') or r.get('instrument_name'),
                'underlying_price': r.get('underlying_price', 0),
                'index_price': r.get('underlying_price', 0),
                'mark_price': r.get('mark_price', 0),
                'mark_iv': r.get('mark_iv', 0),
                'best_bid_price': r.get('bid', 0),
                'best_ask_price': r.get('ask', 0),
                'volume': r.get('volume', 0),
                'open_interest': r.get('open_interest', 0),
            })

    # Fallback sample if no data available
    if not options_data:
        options_data = [
            {
                'instrument_name': 'BTC-28JUN24-50000-C',
                'underlying_price': 50000,
                'index_price': 50000,
                'mark_price': 0.0523,
                'mark_iv': 80,
                'best_bid_price': 0.0520,
                'best_ask_price': 0.0526,
                'volume': 125.5,
                'open_interest': 523.2,
            }
        ]

    # Process with Greeks
    processor = OptionsChainProcessor()
    processed_chain = processor.process_deribit_chain(options_data)

    # Analyze
    metrics = processor.analyze_chain_metrics(processed_chain)
    print(f"BTC Options: {len(processed_chain)} contracts")
    print(f"Average IV: {metrics.average_iv:.2%}")
    print(f"Put/Call Ratio: {metrics.put_call_ratio:.2f}")


if __name__ == '__main__':
    asyncio.run(main())