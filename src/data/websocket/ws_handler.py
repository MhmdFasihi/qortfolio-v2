"""WebSocket Handler for Real-time Updates"""

import asyncio
import websockets
import json
from typing import Dict, Callable
import logging

logger = logging.getLogger(__name__)

class WebSocketHandler:
    """Handle WebSocket connections for real-time data"""
    
    def __init__(self):
        self.connections = set()
        self.callbacks = {}
        
    async def connect_deribit(self, callback: Callable):
        """Connect to Deribit WebSocket"""
        uri = "wss://www.deribit.com/ws/api/v2"
        
        try:
            async with websockets.connect(uri) as websocket:
                # Subscribe to channels
                await self.subscribe_to_channels(websocket)
                
                # Listen for messages
                async for message in websocket:
                    data = json.loads(message)
                    await self.handle_message(data, callback)
                    
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
    
    async def subscribe_to_channels(self, websocket):
        """Subscribe to market data channels"""
        # Subscribe to BTC options
        subscribe_msg = {
            "jsonrpc": "2.0",
            "method": "public/subscribe",
            "params": {
                "channels": [
                    "book.BTC-PERPETUAL.100ms",
                    "ticker.BTC-PERPETUAL.100ms",
                    "trades.BTC-PERPETUAL.100ms"
                ]
            },
            "id": 1
        }
        await websocket.send(json.dumps(subscribe_msg))
    
    async def handle_message(self, data: Dict, callback: Callable):
        """Process incoming WebSocket messages"""
        if "params" in data:
            channel = data["params"].get("channel", "")
            data_content = data["params"].get("data", {})
            
            # Process based on channel type
            if "ticker" in channel:
                await callback("ticker", data_content)
            elif "book" in channel:
                await callback("orderbook", data_content)
            elif "trades" in channel:
                await callback("trades", data_content)
    
    async def broadcast(self, message: Dict):
        """Broadcast message to all connected clients"""
        if self.connections:
            await asyncio.gather(
                *[ws.send(json.dumps(message)) for ws in self.connections]
            )

# Create singleton instance
ws_handler = WebSocketHandler()
