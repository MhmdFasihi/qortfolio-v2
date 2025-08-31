        try:
            # Clean data
            data = raw_data.copy()
            
            # Forward fill NaN values (common in crypto markets during low volume)
            data = data.ffill().bfill()
            
            # Drop any remaining rows with NaN
            data = data.dropna()
            
            # Process each row
            processed_data = []
            symbol = data['Symbol'].iloc[0] if 'Symbol' in data.columns else 'UNKNOWN'
            sector = data.attrs.get('info', {}).get('sector', 'Unknown')
            
            for timestamp, row in data.iterrows():
                price_doc = {
                    'symbol': symbol,
                    'timestamp': timestamp.to_pydatetime(),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': float(row['Volume']),
                    'source': 'yfinance',
                    'sector': sector,
                    'metadata': {
                        'yf_ticker': row.get('YFTicker', f"{symbol}-USD"),
                        'collected_at': datetime.utcnow()
                    }
                }
                
                # Validate using our PriceData model
                try:
                    price_data = PriceData(
                        symbol=price_doc['symbol'],
                        open=price_doc['open'],
                        high=price_doc['high'],
                        low=price_doc['low'],
                        close=price_doc['close'],
                        volume=price_doc['volume'],
                        timestamp=price_doc['timestamp'],
                        source=price_doc['source']
                    )
                    processed_data.append(price_data.to_dict())
                except Exception as e:
                    logger.warning(f"Skipping invalid row: {e}")
                    continue
            
            logger.info(f"Processed {len(processed_data)} price records for {symbol}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Data processing error: {e}")
            raise DataCollectionError(f"Failed to process data: {e}")
