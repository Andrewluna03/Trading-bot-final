"""
Databento WebSocket client for real-time market data streaming
Connects to Databento's WebSocket API to stream Level 2, trades, and quotes data
"""
import asyncio
import json
import logging
import os
import ssl
import socket
import time
from typing import Dict, Any, List, Callable, Optional, Coroutine, Set, Union, Tuple
from datetime import datetime

import websockets
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configuration
REALTIME_ENABLED = True  # Force real data when possible

# Load environment variables
load_dotenv()
API_KEY = os.environ.get("DATABENTO_API_KEY")

class DatabentoWebSocket:
    """
    WebSocket client for the Databento real-time market data API
    """
    
    def __init__(self):
        """Initialize the WebSocket client"""
        self.uri = "wss://live.databento.com/v0/md/"
        self.websocket = None
        self.connected = False
        self.authenticated = False
        self.active_task = None
        self.subscriptions = {}  # symbol -> {schema: callback}
        self.stop_event = asyncio.Event()
        self.last_heartbeat = time.time()
        self.heartbeat_interval = 30  # seconds
        self.loop = None  # Store reference to the event loop
        self._pending_subscriptions = []  # Store pending subscriptions
        
        # Simulate mode for when actual connection isn't available
        self.simulated_mode = False
        self.simulation_task = None
        
        # Force real API key validation during initialization
        if not API_KEY:
            logger.warning("DATABENTO_API_KEY environment variable is not set. Live data will not be available.")
        else:
            logger.info("Databento API key found. Live data will be used when connecting.")
        
    def enable_simulated_mode(self, enable: bool = True) -> None:
        """
        Enable or disable simulated mode
        
        Args:
            enable (bool): True to enable simulated mode, False to disable
        """
        # Check if REALTIME_ENABLED is set to True, which forces real data
        if REALTIME_ENABLED and enable:
            logger.info("REALTIME_ENABLED is set to True - ignoring request to enable simulation mode")
            # Force simulation mode to False
            enable = False
        
        # Already in the desired state
        if self.simulated_mode == enable:
            return
            
        if enable:
            # Enable simulated mode
            self.simulated_mode = True
            logger.info("Enabled simulated mode")
            # If simulated mode is enabled, we want to make sure real websocket is disconnected
            if self.websocket:
                asyncio.create_task(self.disconnect())
        else:
            # Disable simulated mode
            # Just set the flag, actual simulation will be stopped in disconnect() if needed
            self.simulated_mode = False
            logger.info("Disabled simulated mode - will use real data")
            
        # The next call to connect() will start simulation if simulated_mode is True
        
    def is_connected(self) -> bool:
        """
        Check if the client is connected to the WebSocket
        
        Returns:
            bool: True if connected, False otherwise
        """
        # In simulated mode, we're always "connected"
        if self.simulated_mode:
            return True
            
        return self.connected and self.websocket is not None
        
    def _start_simulation(self) -> None:
        """
        Start the simulated market data generation
        """
        if self.simulation_task is not None:
            return
            
        logger.info("Starting simulated market data generation")
        
        # Create a new thread for simulation
        import threading
        
        def run_simulation_loop():
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the simulation coroutine
            try:
                loop.run_until_complete(self._simulate_market_data())
            except Exception as e:
                logger.error(f"Error in simulation loop: {str(e)}")
            finally:
                loop.close()
        
        # Start the simulation thread
        simulation_thread = threading.Thread(target=run_simulation_loop)
        simulation_thread.daemon = True  # Thread will exit when main thread exits
        simulation_thread.start()
        
        # Store a reference to cancel it later if needed
        self.simulation_task = simulation_thread
        
    def _stop_simulation(self) -> None:
        """
        Stop the simulated market data generation
        """
        if self.simulation_task is not None:
            # We can't actually stop the thread, but setting the stop event will
            # cause the simulation loop to exit on its next iteration
            self.stop_event.set()
            
            # Just remove the reference
            self.simulation_task = None
            logger.info("Stopped simulated market data generation")
    
    def get_active_subscriptions(self) -> List[Dict[str, Any]]:
        """
        Get a list of active subscriptions
        
        Returns:
            List[Dict]: Active subscriptions
        """
        subs = []
        for symbol, schemas in self.subscriptions.items():
            for schema in schemas:
                subs.append({
                    "symbol": symbol,
                    "schema": schema
                })
        return subs
    
    def subscribe_level2(self, symbol: str, callback: Any = None,
                         dataset: str = "XNAS.ITCH") -> bool:
        """
        Subscribe to Level 2 market depth data for a symbol
        
        Args:
            symbol (str): The symbol to subscribe to
            callback (callable): Callback function for received data (can be async)
            dataset (str): The dataset to use
            
        Returns:
            bool: Success status
        """
        return self._add_subscription(symbol, "mbp_10", callback, dataset)
    
    def subscribe_trades(self, symbol: str, callback: Any = None,
                        dataset: str = "XNAS.ITCH") -> bool:
        """
        Subscribe to trades (time & sales) data for a symbol
        
        Args:
            symbol (str): The symbol to subscribe to
            callback (callable): Callback function for received data (can be async)
            dataset (str): The dataset to use
            
        Returns:
            bool: Success status
        """
        return self._add_subscription(symbol, "trades", callback, dataset)
    
    def subscribe_quotes(self, symbol: str, callback: Any = None,
                        dataset: str = "XNAS.ITCH") -> bool:
        """
        Subscribe to quotes (BBO) data for a symbol
        
        Args:
            symbol (str): The symbol to subscribe to
            callback (callable): Callback function for received data (can be async)
            dataset (str): The dataset to use
            
        Returns:
            bool: Success status
        """
        return self._add_subscription(symbol, "bbo_1s", callback, dataset)
    
    def _add_subscription(self, symbol: str, schema: str, 
                         callback: Any = None,
                         dataset: str = "XNAS.ITCH") -> bool:
        """
        Add a subscription for a symbol and schema
        
        Args:
            symbol (str): The symbol to subscribe to
            schema (str): The data schema (mbp_10, trades, bbo_1s)
            callback (callable): Callback function for received data (can be async)
            dataset (str): The dataset to use
            
        Returns:
            bool: Success status
        """
        symbol = symbol.upper()
        
        if symbol not in self.subscriptions:
            self.subscriptions[symbol] = {}
            
        self.subscriptions[symbol][schema] = {
            "callback": callback,
            "dataset": dataset,
            "active": True
        }
        
        # If already connected, send subscription request
        if self.is_connected():
            if self.simulated_mode:
                # In simulated mode, no need to send subscription request
                logger.info(f"Added simulated subscription for {symbol} {schema}")
                return True
            else:
                try:
                    # Store for sending in the existing event loop
                    self._pending_subscriptions.append((symbol, schema, dataset))
                    logger.info(f"Queued subscription for {symbol} {schema} (pending)")
                except Exception as e:
                    logger.error(f"Error queuing subscription: {str(e)}")
            
        return True
    
    def unsubscribe(self, symbol: str, schema: Optional[str] = None) -> bool:
        """
        Unsubscribe from data for a symbol
        
        Args:
            symbol (str): The symbol to unsubscribe from
            schema (str, optional): The specific schema to unsubscribe from
                                   If None, unsubscribe from all schemas
            
        Returns:
            bool: Success status
        """
        symbol = symbol.upper()
        
        if symbol not in self.subscriptions:
            return False
            
        if schema is None:
            # Unsubscribe from all schemas
            self.subscriptions.pop(symbol)
        elif schema in self.subscriptions[symbol]:
            # Unsubscribe from specific schema
            self.subscriptions[symbol].pop(schema)
            
            # Remove symbol if no schemas left
            if not self.subscriptions[symbol]:
                self.subscriptions.pop(symbol)
        else:
            return False
            
        # If connected, send unsubscribe request
        if self.is_connected():
            asyncio.create_task(self._send_unsubscribe(symbol, schema))
            
        return True
    
    async def connect(self) -> bool:
        """
        Connect to the Databento WebSocket API
        
        Returns:
            bool: Success status
        """
        # Store the event loop for later use
        self.loop = asyncio.get_event_loop()
        
        # If simulated mode is already active and REALTIME_ENABLED is not True, we're good to go
        if self.simulated_mode and not REALTIME_ENABLED:
            logger.info("Already in simulated mode, starting simulation")
            # Start simulation task if not already running
            self._start_simulation()
            return True
        
        # Force real data mode if REALTIME_ENABLED is True
        if REALTIME_ENABLED and self.simulated_mode:
            logger.info("REALTIME_ENABLED is set to True - switching from simulation to real data mode")
            self.simulated_mode = False
            self._stop_simulation()
        
        # Check for API key before attempting connection to real API
        # Get it fresh from environment in case it's been updated since initialization
        current_api_key = os.environ.get("DATABENTO_API_KEY")
        if not current_api_key:
            logger.error("Missing Databento API key - cannot connect to real data API")
            if REALTIME_ENABLED:
                logger.critical("REALTIME_ENABLED is True but API key is missing - cannot continue")
                return False
            else:
                logger.warning("Falling back to simulated mode due to missing API key")
                self.simulated_mode = True
                self._start_simulation()
                return True
            
        # Reset stop event
        self.stop_event.clear()
        
        try:
            # Create SSL context
            ssl_context = ssl.create_default_context()
            
            # Configure connection options with appropriate timeouts
            connection_options = {
                'ssl': ssl_context,
                'ping_interval': 20,
                'ping_timeout': 10,
                'close_timeout': 5,
                'max_size': 10 * 1024 * 1024,  # 10MB max message size
                'max_queue': 32,
                'compression': None
            }
            
            logger.info(f"Attempting to connect to Databento WebSocket API at {self.uri}")
            
            # Check if DNS resolution works for the Databento domain
            try:
                # Try to resolve the domain to verify network connectivity
                hostname = "live.databento.com"
                logger.info(f"Checking DNS resolution for {hostname}...")
                addr_info = socket.getaddrinfo(hostname, 443, family=socket.AF_INET, type=socket.SOCK_STREAM)
                if addr_info:
                    ip_address = addr_info[0][4][0]
                    logger.info(f"DNS resolution successful: {hostname} -> {ip_address}")
                else:
                    logger.warning(f"DNS resolution returned empty result for {hostname}")
            except socket.gaierror as e:
                logger.error(f"DNS resolution failed for {hostname}: {str(e)}")
                # If we can't resolve the domain, we know the connection will fail
                if REALTIME_ENABLED:
                    logger.critical("DNS resolution failed - network connectivity issue detected")
                else:
                    logger.warning("DNS resolution failed - falling back to simulated mode")
                    self.simulated_mode = True
                    self._start_simulation()
                    return True
            
            # Try different endpoints if the primary one fails
            endpoints = [
                self.uri,
                "wss://live.databento.com/v0/md",  # Try without trailing slash
                "wss://ws.databento.com/v0/md/",   # Alternative subdomain
                "wss://feed.databento.com/v0/md/"  # Another alternative
            ]
            
            connection_exception = None
            for endpoint in endpoints:
                try:
                    logger.info(f"Trying to connect to {endpoint}")
                    self.websocket = await asyncio.wait_for(
                        websockets.connect(endpoint, **connection_options),
                        timeout=10.0  # 10 second connection timeout
                    )
                    logger.info(f"Successfully connected to {endpoint}")
                    # If we reach here, connection was successful
                    self.connected = True
                    self.uri = endpoint  # Save the working endpoint
                    break
                except Exception as e:
                    connection_exception = e
                    logger.warning(f"Failed to connect to {endpoint}: {str(e)}")
                    continue
            
            if not self.connected:
                # All connection attempts failed
                if connection_exception:
                    raise connection_exception
                else:
                    raise ConnectionError("Failed to connect to any Databento endpoint")
                
            logger.info("Connected to Databento WebSocket API")
            
            # Authenticate
            auth_message = json.dumps({
                "action": "auth",
                "key": current_api_key  # Use the fresh API key we just retrieved
            })
            
            await self.websocket.send(auth_message)
            response = await self.websocket.recv()
            response_data = json.loads(response)
            
            if response_data.get("status") == "auth_success":
                self.authenticated = True
                logger.info("Authenticated with Databento WebSocket API")
                # Print a clear visual indicator when authentication succeeds
                print("\n✅ DATABENTO LIVE AUTHENTICATION SUCCESSFUL\n")
                
                # Subscribe to all current subscriptions
                for symbol, schemas in self.subscriptions.items():
                    for schema, details in schemas.items():
                        await self._send_subscription(symbol, schema, details["dataset"])
                
                # Process any pending subscriptions
                if self._pending_subscriptions:
                    logger.info(f"Processing {len(self._pending_subscriptions)} pending subscriptions")
                    pending_subs = self._pending_subscriptions.copy()
                    self._pending_subscriptions = []  # Clear the list
                    
                    for symbol, schema, dataset in pending_subs:
                        logger.info(f"Sending pending subscription for {symbol} {schema}")
                        await self._send_subscription(symbol, schema, dataset)
                
                # Start message handler
                self.active_task = asyncio.create_task(self._message_handler())
                
                return True
            else:
                self.authenticated = False
                logger.error(f"Authentication failed: {response_data.get('message', 'Unknown error')}")
                await self.disconnect()
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to Databento WebSocket API: {str(e)}")
            self.connected = False
            self.websocket = None
            
            # Check if REALTIME_ENABLED is set
            if REALTIME_ENABLED:
                logger.critical("REALTIME_ENABLED is True but connection failed - cannot continue")
                return False
            else:
                # Fall back to simulation mode
                logger.warning("Connection to real API failed, falling back to simulated mode")
                self.simulated_mode = True
                self._start_simulation()
                return True
    
    async def disconnect(self) -> bool:
        """
        Disconnect from the Databento WebSocket API
        
        Returns:
            bool: Success status
        """
        # Signal to stop message handler
        self.stop_event.set()
        
        # Stop simulation if running
        if self.simulated_mode:
            self._stop_simulation()
            self.simulated_mode = False
            logger.info("Stopped simulated mode")
        
        # Cancel active task
        if self.active_task:
            self.active_task.cancel()
            self.active_task = None
        
        # Close WebSocket connection
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            
        self.connected = False
        self.authenticated = False
        logger.info("Disconnected from Databento WebSocket API")
        
        return True
    
    async def _send_subscription(self, symbol: str, schema: str, dataset: str) -> None:
        """
        Send a subscription request
        
        Args:
            symbol (str): The symbol to subscribe to
            schema (str): The data schema
            dataset (str): The dataset to use
        """
        if not self.is_connected() or not self.authenticated:
            logger.error("Cannot subscribe - not connected or authenticated")
            return
            
        try:
            # If we're already subscribed to another schema for this symbol, use a combined approach
            combined_schemas = set()
            if symbol in self.subscriptions:
                for existing_schema in self.subscriptions[symbol].keys():
                    combined_schemas.add(existing_schema)
            combined_schemas.add(schema)
            
            # If more than one schema, send a combined subscription message
            if len(combined_schemas) > 1:
                subscription_message = json.dumps({
                    "action": "subscribe",
                    "channels": [{
                        "dataset": dataset,
                        "symbols": [symbol],
                        "schema": list(combined_schemas)
                    }]
                })
                logger.info(f"Sending combined subscription for {symbol}: {combined_schemas}")
            else:
                # Single schema subscription
                subscription_message = json.dumps({
                    "action": "subscribe",
                    "channels": [{
                        "dataset": dataset,
                        "symbols": [symbol],
                        "schema": schema
                    }]
                })
            
            await self.websocket.send(subscription_message)
            logger.info(f"Subscribed to {symbol} {schema} data")
            
        except Exception as e:
            logger.error(f"Error subscribing to {symbol} {schema}: {str(e)}")
    
    async def _send_unsubscribe(self, symbol: str, schema: Optional[str] = None) -> None:
        """
        Send an unsubscribe request
        
        Args:
            symbol (str): The symbol to unsubscribe from
            schema (str, optional): The specific schema to unsubscribe from
        """
        if not self.is_connected() or not self.authenticated:
            logger.error("Cannot unsubscribe - not connected or authenticated")
            return
            
        try:
            schemas = [schema] if schema else ["mbp_10", "trades", "bbo_1s"]
            
            for sch in schemas:
                unsubscription_message = json.dumps({
                    "action": "unsubscribe",
                    "channels": [{
                        "symbols": [symbol],
                        "schema": sch
                    }]
                })
                
                await self.websocket.send(unsubscription_message)
                
            logger.info(f"Unsubscribed from {symbol}" + (f" {schema}" if schema else ""))
            
        except Exception as e:
            logger.error(f"Error unsubscribing from {symbol}: {str(e)}")
    
    async def _message_handler(self) -> None:
        """
        Handle incoming WebSocket messages
        """
        if not self.websocket:
            logger.error("WebSocket not connected")
            return
            
        try:
            while not self.stop_event.is_set():
                try:
                    # Wait for message with timeout (for heartbeat)
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=self.heartbeat_interval)
                    
                    # Process message
                    await self._process_message(message)
                    
                except asyncio.TimeoutError:
                    # Send heartbeat
                    await self._send_heartbeat()
                    
                except websockets.exceptions.ConnectionClosed:
                    logger.error("WebSocket connection closed")
                    break
                    
        except asyncio.CancelledError:
            logger.info("Message handler task cancelled")
            
        except Exception as e:
            logger.error(f"Error in message handler: {str(e)}")
            
        finally:
            # Ensure WebSocket is closed
            if self.websocket and not self.stop_event.is_set():
                # Unexpected closure, try to reconnect
                logger.info("Unexpected WebSocket closure, reconnecting...")
                self.active_task = None
                asyncio.create_task(self._reconnect())
    
    async def _reconnect(self) -> None:
        """
        Reconnect to the WebSocket API
        """
        # Close existing connection
        if self.websocket:
            try:
                await self.websocket.close()
            except:
                pass
            self.websocket = None
            
        self.connected = False
        self.authenticated = False
        
        # Wait before reconnecting
        await asyncio.sleep(5)
        
        # Try to reconnect
        for attempt in range(3):
            logger.info(f"Reconnection attempt {attempt + 1}/3")
            success = await self.connect()
            
            if success:
                logger.info("Reconnected successfully")
                return
                
            # Wait before next attempt
            await asyncio.sleep(5 * (attempt + 1))
            
        logger.error("Failed to reconnect after 3 attempts")
    
    async def _send_heartbeat(self) -> None:
        """
        Send a heartbeat message to keep the connection alive
        """
        if not self.is_connected():
            return
            
        try:
            heartbeat_message = json.dumps({
                "action": "ping"
            })
            
            await self.websocket.send(heartbeat_message)
            self.last_heartbeat = time.time()
            logger.debug("Sent heartbeat")
            
        except Exception as e:
            logger.error(f"Error sending heartbeat: {str(e)}")
    
    async def _process_message(self, message: str) -> None:
        """
        Process an incoming WebSocket message
        
        Args:
            message (str): The message to process
        """
        try:
            data = json.loads(message)
            message_type = data.get("action", "")
            
            # Handle different message types
            if message_type == "pong":
                # Heartbeat response
                logger.debug("Received heartbeat response")
                
            elif message_type == "md_update" or not message_type:
                # Market data update
                await self._handle_market_data(data)
                
            elif message_type == "error":
                # Error message
                logger.error(f"Received error: {data.get('message', 'Unknown error')}")
                
            else:
                # Unknown message type
                logger.warning(f"Received unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            logger.error(f"Received invalid JSON: {message}")
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
    
    async def _simulate_market_data(self) -> None:
        """
        Simulate market data updates for all active subscriptions
        """
        import random
        from datetime import datetime
        
        logger.info("Started simulated market data generator")
        
        try:
            # Run until stopped
            while not self.stop_event.is_set():
                # Check if we have any subscriptions
                if not self.subscriptions:
                    await asyncio.sleep(1)
                    continue
                
                # Process each subscription
                for symbol, schemas in self.subscriptions.items():
                    for schema, details in schemas.items():
                        # Get the callback
                        callback = details.get("callback")
                        if not callback:
                            continue
                        
                        # Generate simulated data based on schema
                        if schema == "mbp_10":  # Level 2 market depth
                            data = self._generate_simulated_level2(symbol)
                            if callback and data:
                                try:
                                    import inspect
                                    if inspect.iscoroutinefunction(callback):
                                        # If callback is async, create a task to run it
                                        asyncio.create_task(callback(data))
                                    else:
                                        # If callback is a regular function, just call it
                                        callback(data)
                                except Exception as e:
                                    logger.error(f"Error in L2 callback for {symbol}: {str(e)}")
                                    
                        elif schema == "trades":  # Trades
                            data = self._generate_simulated_trades(symbol)
                            if callback and data:
                                try:
                                    import inspect
                                    if inspect.iscoroutinefunction(callback):
                                        # If callback is async, create a task to run it
                                        asyncio.create_task(callback(data))
                                    else:
                                        # If callback is a regular function, just call it
                                        callback(data)
                                except Exception as e:
                                    logger.error(f"Error in trades callback for {symbol}: {str(e)}")
                                    
                        elif schema == "bbo_1s":  # Quotes (BBO)
                            data = self._generate_simulated_quotes(symbol)
                            if callback and data:
                                try:
                                    import inspect
                                    if inspect.iscoroutinefunction(callback):
                                        # If callback is async, create a task to run it
                                        asyncio.create_task(callback(data))
                                    else:
                                        # If callback is a regular function, just call it
                                        callback(data)
                                except Exception as e:
                                    logger.error(f"Error in quotes callback for {symbol}: {str(e)}")
                
                # Wait a short time before next update
                await asyncio.sleep(0.5)
                
        except asyncio.CancelledError:
            logger.info("Simulated market data generator task cancelled")
        except Exception as e:
            logger.error(f"Error in simulated market data generator: {str(e)}")
    
    def _generate_simulated_level2(self, symbol: str) -> Dict[str, Any]:
        """
        Generate simulated Level 2 market depth data
        
        Args:
            symbol (str): The symbol to generate data for
            
        Returns:
            Dict: Simulated market depth data in MBP-10 format that matches Databento's API
        """
        import random
        from datetime import datetime
        
        # Base price for the symbol (simulated)
        base_price = self._get_symbol_base_price(symbol)
        
        # Create the MBP-10 data structure with proper field names
        # This format matches what the handler expects from Databento API
        data = {
            "symbol": symbol,
            "schema": "mbp_10",
            "timestamp": datetime.now().timestamp(),
        }
        
        # Generate bid levels, adding proper field names
        for i in range(10):
            price = base_price - (i * 0.01) - (random.random() * 0.005)
            size = random.randint(100, 5000)
            
            # Use the exact field names expected by the handler
            bid_price_key = f'bid_px_{i:02d}'
            bid_size_key = f'bid_sz_{i:02d}'
            
            data[bid_price_key] = round(price, 2)
            data[bid_size_key] = size
        
        # Generate ask levels, adding proper field names
        for i in range(10):
            price = base_price + (i * 0.01) + (random.random() * 0.005)
            size = random.randint(100, 5000)
            
            # Use the exact field names expected by the handler
            ask_price_key = f'ask_px_{i:02d}'
            ask_size_key = f'ask_sz_{i:02d}'
            
            data[ask_price_key] = round(price, 2)
            data[ask_size_key] = size
        
        # Add a source field
        data["source"] = "databento_websocket"
        
        return data
    
    def _generate_simulated_trades(self, symbol: str) -> Dict[str, Any]:
        """
        Generate simulated trades data
        
        Args:
            symbol (str): The symbol to generate data for
            
        Returns:
            Dict: Simulated trades data in format compatible with the handler
        """
        import random
        import time
        from datetime import datetime
        
        # Base price for the symbol (simulated)
        base_price = self._get_symbol_base_price(symbol)
        
        # Random price variation
        price_variation = (random.random() - 0.5) * 0.1
        price = round(base_price + price_variation, 2)
        
        # Random size
        size = random.randint(100, 5000)
        
        # Random direction (buy/sell)
        side = random.choice(["B", "S"])  # B for buy, S for sell as per Databento API
        
        # Create the structure that matches what the handler expects
        data = {
            "symbol": symbol,
            "schema": "trades",
            "price": price,
            "size": size,
            "side": side,
            "timestamp": datetime.now().timestamp(),
            "exchange": "SIMX",
            "trade_id": f"{symbol}-{int(time.time())}-{random.randint(1000, 9999)}",
            "tape": "C",
            "is_buyer_maker": side == "B",
            "source": "databento_websocket"
        }
        
        return data
    
    def _generate_simulated_quotes(self, symbol: str) -> Dict[str, Any]:
        """
        Generate simulated quotes (BBO) data
        
        Args:
            symbol (str): The symbol to generate data for
            
        Returns:
            Dict: Simulated quotes data in BBO_1S format that matches Databento's API
        """
        import random
        from datetime import datetime
        
        # Base price for the symbol (simulated)
        base_price = self._get_symbol_base_price(symbol)
        
        # Generate bid (slightly lower than base)
        bid_price = round(base_price - (random.random() * 0.02), 2)
        bid_size = random.randint(100, 5000)
        
        # Generate ask (slightly higher than base)
        ask_price = round(base_price + (random.random() * 0.02), 2)
        ask_size = random.randint(100, 5000)
        
        # Create the data structure that matches what the handler expects
        data = {
            "symbol": symbol,
            "schema": "bbo_1s",
            "timestamp": datetime.now().timestamp(),
            "bid_px_00": bid_price,
            "bid_sz_00": bid_size,
            "ask_px_00": ask_price,
            "ask_sz_00": ask_size,
            "source": "databento_websocket"
        }
        
        return data
    
    def _get_symbol_base_price(self, symbol: str) -> float:
        """
        Get a simulated base price for a symbol
        
        Args:
            symbol (str): The symbol to get price for
            
        Returns:
            float: Simulated base price
        """
        # Use symbol hash to create a somewhat realistic and consistent price
        import hashlib
        import time
        
        # Get the hash of the symbol
        symbol_hash = int(hashlib.md5(symbol.encode()).hexdigest(), 16)
        
        # Use the hash to generate a base price between $10 and $1000
        base_price = 10 + (symbol_hash % 990)
        
        # Add some small variation based on time to make it change
        time_variation = (int(time.time()) % 100) / 1000
        
        return base_price + time_variation
    
    async def _handle_market_data(self, data: Dict[str, Any]) -> None:
        """
        Handle a market data update
        
        Args:
            data (Dict): The market data update
        """
        # Extract schema and symbol
        schema = data.get("schema")
        symbol = data.get("symbol")
        
        if not schema or not symbol:
            return
            
        # Check if we have a subscription for this data
        if symbol in self.subscriptions and schema in self.subscriptions[symbol]:
            # Get the callback
            callback = self.subscriptions[symbol][schema].get("callback")
            
            # Call the callback if it exists
            if callback:
                try:
                    import inspect
                    if inspect.iscoroutinefunction(callback):
                        # If callback is async, create a task to run it
                        asyncio.create_task(callback(data))
                    else:
                        # If callback is a regular function, just call it
                        callback(data)
                except Exception as e:
                    logger.error(f"Error in callback for {symbol} {schema}: {str(e)}")

# Simple test
async def main():
    client = DatabentoWebSocket()
    
    # Define callback for Level 2 data
    async def level2_callback(data):
        print(f"L2: {data.get('symbol')} - Received {len(data.get('data', []))} updates")
    
    # Define callback for trades data
    async def trades_callback(data):
        print(f"Trades: {data.get('symbol')} - {data.get('data')}")
    
    # Define callback for quotes data
    async def quotes_callback(data):
        print(f"Quotes: {data.get('symbol')} - {data.get('data')}")
    
    # Connect to the WebSocket API
    connected = await client.connect()
    if not connected:
        print("Failed to connect")
        return
    
    # Subscribe to data
    client.subscribe_level2("AAPL", level2_callback)
    client.subscribe_trades("AAPL", trades_callback)
    client.subscribe_quotes("AAPL", quotes_callback)
    
    # Wait for data
    await asyncio.sleep(60)
    
    # Unsubscribe and disconnect
    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())