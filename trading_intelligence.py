"""
Trading intelligence module for real-time market data analysis
Provides automated intelligence and insights based on order book patterns
"""
import logging
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingIntelligence:
    """
    Trading intelligence class that analyzes market data and provides insights
    """
    def __init__(self):
        """Initialize the trading intelligence system"""
        self.last_insights = {}
        self.cooldown_periods = {}  # Prevent repeated insights
        self.historical_context = {}  # Store historical context for deeper analysis
        self.pattern_recognition_history = {}  # Track pattern formations over time
    
    def analyze_order_book(self, symbol: str, bids: List[Dict[str, Any]], 
                           asks: List[Dict[str, Any]], trades: List[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Analyze order book (Level 2) data and generate trading insights
        
        Args:
            symbol (str): The trading symbol
            bids (List[Dict]): List of bid orders with price and size
            asks (List[Dict]): List of ask orders with price and size
            trades (List[Dict], optional): Recent trades to incorporate into analysis
            
        Returns:
            Dict or None: Trading insights if significant patterns detected
        """
        if not bids or not asks:
            return None
        
        # Apply cooldown to prevent repeated insights
        current_time = datetime.now()
        if symbol in self.cooldown_periods and \
           (current_time - self.cooldown_periods[symbol]).total_seconds() < 15:  # Reduced from 30s to 15s
            return None
        
        # Get top bid and ask
        top_bid = bids[0] if bids else {'price': 0, 'size': 0}
        top_ask = asks[0] if asks else {'price': 0, 'size': 0}
        
        bid_price = top_bid['price']
        bid_size = top_bid['size']
        ask_price = top_ask['price']
        ask_size = top_ask['size']
        
        # Calculate spread
        spread = round(ask_price - bid_price, 4) if ask_price > 0 and bid_price > 0 else 0
        spread_pct = round((spread / bid_price) * 100, 4) if bid_price > 0 else 0
        
        # Extract price and size arrays
        bid_prices = [b['price'] for b in bids[:5]]
        bid_sizes = [b['size'] for b in bids[:5]]
        ask_prices = [a['price'] for a in asks[:5]]
        ask_sizes = [a['size'] for a in asks[:5]]
        
        # Calculate averages
        avg_bid_size = sum(bid_sizes) / len(bid_sizes) if bid_sizes else 0
        avg_ask_size = sum(ask_sizes) / len(ask_sizes) if ask_sizes else 0
        
        # Calculate price jumps between levels
        bid_jumps = [abs(bid_prices[i] - bid_prices[i+1]) for i in range(len(bid_prices)-1)] if len(bid_prices) > 1 else []
        ask_jumps = [abs(ask_prices[i] - ask_prices[i+1]) for i in range(len(ask_prices)-1)] if len(ask_prices) > 1 else []
        avg_bid_jump = sum(bid_jumps) / len(bid_jumps) if bid_jumps else 0
        avg_ask_jump = sum(ask_jumps) / len(ask_jumps) if ask_jumps else 0
        
        # Initialize insights
        insights = {
            "symbol": symbol,
            "timestamp": current_time.isoformat(),
            "messages": [],
            "recommendation": None,
            "confidence": 0,
            "price_level": bid_price,
            "spread": spread,
            "spread_pct": spread_pct,
            "status": "stable"  # Default status
        }
        
        # Pattern 1: Tight spread (potential breakout signal)
        if spread_pct < 0.01:  # Extremely tight spread (less than 0.01%)
            insights["messages"].append("🟢 Spread tightening to minimum levels — breakout conditions building.")
            insights["status"] = "imminent_breakout"
            insights["confidence"] = 0.9
        elif spread_pct < 0.05:  # Very tight spread (less than 0.05%)
            insights["messages"].append("🟢 Spread is extremely tight. Market makers preparing for movement. Possible breakout imminent.")
            insights["status"] = "breakout_watch"
            insights["confidence"] = 0.8
        elif spread_pct < 0.1:  # Tight spread (less than 0.1%)
            insights["messages"].append("🟢 Spread is tightening. Market makers getting ready. Possible breakout soon.")
            insights["status"] = "breakout_watch"
            insights["confidence"] = 0.6
        
        # Pattern 2: Stacked bids (strong support)
        large_bids = [size for i, size in enumerate(bid_sizes) if size > 4 * avg_bid_size]
        if large_bids:
            if bid_sizes[0] > 3 * avg_bid_size:  # Very large bid at top of book
                insights["messages"].append(f"📈 Strong bid stacking — buyers loading below.")
                insights["status"] = "strong_support"
                insights["confidence"] = 0.85
            elif bid_sizes[0] > 2 * avg_bid_size:  # Large bid at top of book
                insights["messages"].append(f"📈 Big bid stacking detected. Buyers are stepping in hard.")
                insights["status"] = "strong_support"
                insights["confidence"] = 0.75
            else:
                insights["messages"].append(f"📈 Bid stacking detected at lower levels.")
                insights["status"] = "support_forming"
                insights["confidence"] = 0.65
        
        # Pattern 3: Heavy asks (resistance)
        large_asks = [size for i, size in enumerate(ask_sizes) if size > 3 * avg_ask_size]
        if large_asks:
            if ask_sizes[0] > 3 * avg_ask_size:  # Very large ask at top of book
                insights["messages"].append(f"🔻 Heavy selling into bid — watch for a flush.")
                insights["status"] = "strong_resistance"
                insights["confidence"] = 0.85
            elif ask_sizes[0] > 2 * avg_ask_size:  # Large ask at top of book
                insights["messages"].append(f"🔴 Heavy ask wall above. Could stall the breakout.")
                insights["status"] = "strong_resistance"
                insights["confidence"] = 0.75
            else:
                insights["messages"].append(f"🔴 Sell pressure building at higher levels.")
                insights["status"] = "resistance_forming"
                insights["confidence"] = 0.6
                insights["confidence"] = 0.65
        
        # Pattern 4: Bid size > Ask size (buying pressure)
        bid_to_ask_ratio = bid_size / ask_size if ask_size > 0 else 0
        if bid_to_ask_ratio > 2.5:
            insights["messages"].append(f"🚀 Strong buying pressure at current level. Bid/Ask ratio: {bid_to_ask_ratio:.1f}x")
            insights["status"] = "buying_pressure"
            insights["confidence"] = 0.7
        
        # Pattern 5: Ask size > Bid size (selling pressure)
        ask_to_bid_ratio = ask_size / bid_size if bid_size > 0 else 0
        if ask_to_bid_ratio > 2.5:
            insights["messages"].append(f"📉 Strong selling pressure at current level. Ask/Bid ratio: {ask_to_bid_ratio:.1f}x")
            insights["status"] = "selling_pressure"
            insights["confidence"] = 0.7
        
        # Pattern 6: Price gaps in order book (potential volatility)
        if any(jump > 2 * avg_bid_jump for jump in bid_jumps) or any(jump > 2 * avg_ask_jump for jump in ask_jumps):
            insights["messages"].append("⚠️ Price gaps detected. Prepare for volatility.")
            insights["status"] = "volatile"
            insights["confidence"] = 0.6
            
        # Analyze trades if provided
        if trades and len(trades) >= 3:
            # Extract trade types (buyer or seller initiated)
            ask_hits = [t for t in trades if t.get('side', '') == 'buy']
            bid_hits = [t for t in trades if t.get('side', '') == 'sell']
            
            # Look at recent aggressive trades
            if len(ask_hits) > 2 * len(bid_hits) and sum(t.get('size', 0) for t in ask_hits) > 5 * avg_bid_size:
                insights["messages"].append("🚀 Ask is getting lifted fast. Buyers might be absorbing supply. Watch for breakout.")
                insights["status"] = "bullish_momentum"
                insights["confidence"] = 0.8
                
            elif len(bid_hits) > 2 * len(ask_hits) and sum(t.get('size', 0) for t in bid_hits) > 5 * avg_ask_size:
                insights["messages"].append("📉 Selling into bids. Supply overwhelming demand.")
                insights["status"] = "bearish_momentum"
                insights["confidence"] = 0.8
        
        # Generate trading recommendation based on insights
        if insights["status"] in ["breakout_watch", "strong_support", "buying_pressure", "bullish_momentum"]:
            insights["recommendation"] = "Buy"
            insights["confidence"] = min(insights["confidence"] + 0.1, 0.9)
        elif insights["status"] in ["strong_resistance", "selling_pressure", "bearish_momentum"]:
            insights["recommendation"] = "Sell"
            insights["confidence"] = min(insights["confidence"] + 0.1, 0.9)
        elif insights["status"] == "volatile":
            insights["recommendation"] = "Hold"
        else:
            insights["recommendation"] = "Wait"
            
        # Only return insights if we have messages
        if insights["messages"]:
            # Update cooldown period
            self.cooldown_periods[symbol] = current_time
            # Store last insights
            self.last_insights[symbol] = insights
            return insights
        
        return None
    
    def analyze_trades(self, symbol: str, trades: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Analyze time & sales (trades) data and generate trading insights
        
        Args:
            symbol (str): The trading symbol
            trades (List[Dict]): List of recent trades with price, size, etc.
            
        Returns:
            Dict or None: Trading insights if significant patterns detected
        """
        if not trades or len(trades) < 5:
            return None
        
        # Apply cooldown
        current_time = datetime.now()
        if symbol in self.cooldown_periods and \
           (current_time - self.cooldown_periods[symbol]).total_seconds() < 15:  # Reduced from 30s to 15s
            return None
        
        # Process trades
        trade_prices = [t['price'] for t in trades]
        trade_sizes = [t['size'] for t in trades]
        
        # Get trade sides (buy/sell)
        trade_sides = []
        for t in trades:
            side = t.get('side', None)
            if side is None:
                # Fall back to is_buyer_maker if side isn't directly specified
                is_buyer_maker = t.get('is_buyer_maker', False)
                side = 'buy' if is_buyer_maker else 'sell'
            trade_sides.append(side)
            
        # Calculate metrics
        avg_price = sum(trade_prices) / len(trade_prices)
        avg_size = sum(trade_sizes) / len(trade_sizes)
        price_trend = trade_prices[0] - trade_prices[-1]  # First trade is most recent
        
        # Calculate buy/sell volumes based on sides
        buy_volume = sum(s for i, s in enumerate(trade_sizes) if trade_sides[i] == 'buy')
        sell_volume = sum(s for i, s in enumerate(trade_sizes) if trade_sides[i] == 'sell')
        
        # Count buy and sell trades
        buy_count = sum(1 for side in trade_sides if side == 'buy')
        sell_count = sum(1 for side in trade_sides if side == 'sell')
        
        # Detect large trades
        large_trades = [s for s in trade_sizes if s > 3 * avg_size]
        
        # Initialize insights
        insights = {
            "symbol": symbol,
            "timestamp": current_time.isoformat(),
            "messages": [],
            "recommendation": None,
            "confidence": 0,
            "price_level": trade_prices[0],
            "status": "stable"  # Default status
        }
        
        # Pattern 1: Large trades
        if large_trades:
            insights["messages"].append(f"👀 Large trade(s) detected: {large_trades[0]} shares at ${trade_prices[0]}")
            insights["confidence"] = 0.6
            
        # Pattern 2: Price momentum
        if abs(price_trend) / avg_price > 0.005:  # 0.5% price change in recent trades
            if price_trend > 0:
                insights["messages"].append(f"🚀 Strong upward price movement: +${price_trend:.2f} in recent trades")
                insights["status"] = "uptrend"
                insights["confidence"] = 0.7
            else:
                insights["messages"].append(f"📉 Strong downward price movement: -${abs(price_trend):.2f} in recent trades")
                insights["status"] = "downtrend"
                insights["confidence"] = 0.7
                
        # Pattern 3: Buy/Sell imbalance
        if buy_volume > 0 and sell_volume > 0:
            buy_to_sell_ratio = buy_volume / sell_volume if sell_volume > 0 else float('inf')
            sell_to_buy_ratio = sell_volume / buy_volume if buy_volume > 0 else float('inf')
            
            if buy_to_sell_ratio > 2.5:
                insights["messages"].append(f"🚀 Ask is getting lifted fast. Buyers might be absorbing supply. Watch for breakout.")
                insights["status"] = "buying_flow"
                insights["confidence"] = 0.75
            elif sell_to_buy_ratio > 2.5:
                insights["messages"].append(f"📉 Selling into bids. Supply overwhelming demand.")
                insights["status"] = "selling_flow"
                insights["confidence"] = 0.75
        
        # Pattern 4: Trade frequency analysis
        if len(trades) >= 10:
            # Check if there's a burst of trades (high activity)
            timestamps = [datetime.fromisoformat(t.get('timestamp', current_time.isoformat())) 
                        for t in trades[:5]]  # Check most recent 5 trades
            
            if timestamps and len(timestamps) > 1:
                time_diffs = [(timestamps[i] - timestamps[i+1]).total_seconds() 
                            for i in range(len(timestamps)-1)]
                avg_time_diff = sum(time_diffs) / len(time_diffs) if time_diffs else 0
                
                if avg_time_diff < 0.5:  # Less than 0.5 seconds between trades
                    insights["messages"].append("⚡ High-frequency trading detected. Increased volatility likely.")
                    insights["status"] = "high_activity"
                    insights["confidence"] = 0.65
            
        # Generate recommendation
        if insights["status"] in ["uptrend", "buying_flow", "high_activity"] and buy_count > sell_count:
            insights["recommendation"] = "Buy"
            insights["confidence"] = min(insights["confidence"] + 0.1, 0.9)
        elif insights["status"] in ["downtrend", "selling_flow"] or (insights["status"] == "high_activity" and sell_count > buy_count):
            insights["recommendation"] = "Sell" 
            insights["confidence"] = min(insights["confidence"] + 0.1, 0.9)
        else:
            insights["recommendation"] = "Wait"
            
        # Only return insights if we have messages
        if insights["messages"]:
            # Update cooldown period
            self.cooldown_periods[symbol] = current_time
            # Update last insights
            if symbol not in self.last_insights:
                self.last_insights[symbol] = insights
            else:
                # Merge with existing insights
                self.last_insights[symbol]["messages"].extend(insights["messages"])
                if insights["confidence"] > self.last_insights[symbol]["confidence"]:
                    self.last_insights[symbol]["recommendation"] = insights["recommendation"]
                    self.last_insights[symbol]["confidence"] = insights["confidence"]
            
            return insights
        
        return None
    
    def get_last_insights(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the last insights for a symbol
        
        Args:
            symbol (str): The trading symbol
            
        Returns:
            Dict or None: Last trading insights for the symbol
        """
        return self.last_insights.get(symbol.upper())
    
    def analyze_market_context(self, symbol: str, market_data: Dict[str, Any], 
                              trades: List[Dict[str, Any]], order_book: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the full market context by combining order book, trades, and historical context
        Provides a deeper, more contextual analysis than individual components
        
        Args:
            symbol (str): The trading symbol
            market_data (Dict): Current market data (bid/ask)
            trades (List[Dict]): Recent trades
            order_book (Dict): Level 2 order book data
            
        Returns:
            Dict: Deep context analysis with advanced insights
        """
        symbol = symbol.upper()
        current_time = datetime.now()
        
        # Initialize context analysis result
        context_analysis = {
            "symbol": symbol,
            "timestamp": current_time.isoformat(),
            "market_phase": "unknown",
            "price_structure": "neutral",
            "support_resistance": [],
            "volume_profile": "normal",
            "volatility_state": "normal",
            "key_levels": [],
            "advanced_insights": [],
            "deep_recommendation": "Wait",
            "confidence": 0.0,
            "reasoning": []
        }
        
        # Extract order book data
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])
        
        # Extract market data
        bid_price = market_data.get('bid_price', 0)
        ask_price = market_data.get('ask_price', 0)
        bid_size = market_data.get('bid_size', 0)
        ask_size = market_data.get('ask_size', 0)
        last_price = trades[0].get('price', 0) if trades else 0
        
        # Calculate current spread
        spread = round(ask_price - bid_price, 4) if ask_price > 0 and bid_price > 0 else 0
        spread_pct = round((spread / bid_price) * 100, 4) if bid_price > 0 else 0
        
        # Add symbol to historical context if not present
        if symbol not in self.historical_context:
            self.historical_context[symbol] = {
                "price_levels": [],
                "volume_nodes": {},
                "previous_insights": [],
                "support_resistance": [],
                "last_update": current_time.isoformat()
            }
        
        # Update historical price levels
        if last_price > 0:
            self.historical_context[symbol]["price_levels"].append(last_price)
            # Keep only recent price levels (last 1000)
            self.historical_context[symbol]["price_levels"] = self.historical_context[symbol]["price_levels"][-1000:]
        
        # Calculate volume profile
        if trades:
            for trade in trades:
                price = trade.get('price', 0)
                size = trade.get('size', 0)
                price_key = str(round(price, 2))
                
                if price_key not in self.historical_context[symbol]["volume_nodes"]:
                    self.historical_context[symbol]["volume_nodes"][price_key] = 0
                    
                self.historical_context[symbol]["volume_nodes"][price_key] += size
        
        # Identify support/resistance levels based on volume nodes
        if self.historical_context[symbol]["volume_nodes"]:
            volume_items = list(self.historical_context[symbol]["volume_nodes"].items())
            volume_items.sort(key=lambda x: float(x[0]))  # Sort by price
            
            # Find high volume nodes (potential support/resistance)
            prices = [float(p) for p, v in volume_items]
            volumes = [v for p, v in volume_items]
            avg_volume = sum(volumes) / len(volumes) if volumes else 0
            
            # Find local maxima in volume profile
            high_volume_nodes = []
            for i, (price_str, volume) in enumerate(volume_items):
                price = float(price_str)
                if volume > 2 * avg_volume:
                    # Check if it's a local maximum
                    is_local_max = True
                    window = 3  # Check 3 prices on each side
                    
                    for j in range(max(0, i-window), min(len(volume_items), i+window+1)):
                        if j != i and volume_items[j][1] > volume:
                            is_local_max = False
                            break
                    
                    if is_local_max:
                        high_volume_nodes.append((price, volume))
            
            # Update support/resistance levels
            self.historical_context[symbol]["support_resistance"] = [price for price, _ in high_volume_nodes]
            context_analysis["support_resistance"] = [price for price, _ in high_volume_nodes]
            
            # Identify nearest support and resistance
            current_price = last_price or bid_price
            support_levels = [p for p in self.historical_context[symbol]["support_resistance"] if p < current_price]
            resistance_levels = [p for p in self.historical_context[symbol]["support_resistance"] if p > current_price]
            
            support_levels.sort(reverse=True)  # Higher values first
            resistance_levels.sort()  # Lower values first
            
            if support_levels:
                context_analysis["key_levels"].append({
                    "type": "support",
                    "price": support_levels[0],
                    "distance": round(current_price - support_levels[0], 2),
                    "strength": "strong" if high_volume_nodes[0][1] > 3 * avg_volume else "moderate"
                })
                
            if resistance_levels:
                context_analysis["key_levels"].append({
                    "type": "resistance",
                    "price": resistance_levels[0],
                    "distance": round(resistance_levels[0] - current_price, 2),
                    "strength": "strong" if high_volume_nodes[0][1] > 3 * avg_volume else "moderate"
                })
        
        # Determine market phase
        if trades and len(trades) >= 10:
            # Look at price action in the most recent trades
            recent_prices = [t.get('price', 0) for t in trades[:10]]
            price_changes = [recent_prices[i] - recent_prices[i+1] for i in range(len(recent_prices)-1)]
            
            up_moves = sum(1 for change in price_changes if change > 0)
            down_moves = sum(1 for change in price_changes if change < 0)
            
            if up_moves >= 7:  # Strong uptrend
                context_analysis["market_phase"] = "accumulation"
                context_analysis["price_structure"] = "bullish"
                context_analysis["advanced_insights"].append("🚀 Strong accumulation phase - buyers stepping in consistently")
                
            elif down_moves >= 7:  # Strong downtrend
                context_analysis["market_phase"] = "distribution"
                context_analysis["price_structure"] = "bearish"
                context_analysis["advanced_insights"].append("📉 Distribution phase - sellers unloading positions")
                
            elif up_moves >= 5 and down_moves >= 3:  # Choppy with upward bias
                context_analysis["market_phase"] = "markup"
                context_analysis["price_structure"] = "bullish"
                context_analysis["advanced_insights"].append("📈 Markup phase - price making higher highs with pullbacks")
                
            elif down_moves >= 5 and up_moves >= 3:  # Choppy with downward bias
                context_analysis["market_phase"] = "markdown"
                context_analysis["price_structure"] = "bearish"
                context_analysis["advanced_insights"].append("🔻 Markdown phase - price making lower lows with bounces")
                
            else:  # Mixed signals
                context_analysis["market_phase"] = "consolidation"
                context_analysis["price_structure"] = "neutral"
                context_analysis["advanced_insights"].append("⏸️ Consolidation phase - price bouncing within a range")
        
        # Detect volatility state
        if trades and len(trades) >= 20:
            prices = [t.get('price', 0) for t in trades[:20]]
            if prices:
                avg_price = sum(prices) / len(prices)
                price_deviations = [(p - avg_price) ** 2 for p in prices]
                volatility = (sum(price_deviations) / len(price_deviations)) ** 0.5 / avg_price * 100
                
                if volatility > 0.2:  # High volatility
                    context_analysis["volatility_state"] = "high"
                    context_analysis["advanced_insights"].append("⚡ High volatility detected - prepare for significant price swings")
                elif volatility > 0.1:  # Moderate volatility
                    context_analysis["volatility_state"] = "moderate"
                    context_analysis["advanced_insights"].append("〰️ Moderate volatility - tactical trading opportunities present")
                else:  # Low volatility
                    context_analysis["volatility_state"] = "low"
                    context_analysis["advanced_insights"].append("🔄 Low volatility - potential buildup for a move, be patient")
        
        # Generate advanced volume profile analysis
        if bids and asks and trades:
            # Analyze volume profile imbalance
            bid_volumes = sum(b.get('size', 0) for b in bids[:3])
            ask_volumes = sum(a.get('size', 0) for a in asks[:3])
            
            if bid_volumes > 2 * ask_volumes:
                context_analysis["volume_profile"] = "bullish_imbalance"
                context_analysis["advanced_insights"].append("🟢 Strong buy-side volume imbalance indicating absorption of supply")
                context_analysis["reasoning"].append("Buy orders significantly outweigh sell orders at top of book")
                
            elif ask_volumes > 2 * bid_volumes:
                context_analysis["volume_profile"] = "bearish_imbalance"
                context_analysis["advanced_insights"].append("🔴 Strong sell-side volume imbalance indicating overwhelming supply")
                context_analysis["reasoning"].append("Sell orders significantly outweigh buy orders at top of book")
            
            # Analyze recent trade volume vs. book volume
            trade_volume = sum(t.get('size', 0) for t in trades[:10])
            book_volume = bid_volumes + ask_volumes
            
            if trade_volume > 3 * book_volume:
                context_analysis["volume_profile"] = "active_higher_time_frame"
                context_analysis["advanced_insights"].append("👁️ Higher time frame players active - trade volume exceeding book volume")
                context_analysis["reasoning"].append("Recent trades show unusually high volume compared to displayed liquidity")
        
        # Generate deep market interpretations
        price_action_insight = ""
        order_book_insight = ""
        
        # Price action interpretation
        if context_analysis["price_structure"] == "bullish":
            if context_analysis["volatility_state"] == "high":
                price_action_insight = "Bullish momentum with high volatility suggests strong buying interest and potential for rapid moves upward."
            else:
                price_action_insight = "Bullish trend with controlled volatility suggests methodical accumulation and likely continued upward movement."
        elif context_analysis["price_structure"] == "bearish":
            if context_analysis["volatility_state"] == "high":
                price_action_insight = "Bearish momentum with high volatility suggests strong selling pressure and potential for sharp declines."
            else:
                price_action_insight = "Bearish trend with controlled volatility suggests methodical distribution and likely continued downward movement."
        else:
            price_action_insight = "Neutral price structure suggests market indecision - watch for a breakout from this consolidation range."
        
        # Order book interpretation
        if spread_pct < 0.05:
            order_book_insight = "Extremely tight spread indicates high liquidity and potential for a decisive move soon."
        elif bids and asks:
            if len(bids) > 2 * len(asks):
                order_book_insight = "Book depth shows significant bid support compared to asks, suggesting stronger buying interest."
            elif len(asks) > 2 * len(bids):
                order_book_insight = "Book depth shows significant ask pressure compared to bids, suggesting stronger selling interest."
            else:
                order_book_insight = "Balanced order book depth suggests equilibrium between buyers and sellers at current levels."
        
        # Add the interpretations
        if price_action_insight:
            context_analysis["advanced_insights"].append(price_action_insight)
        if order_book_insight:
            context_analysis["advanced_insights"].append(order_book_insight)
        
        # Generate final comprehensive recommendation
        trading_signals = []
        
        # Add different signals based on our analysis
        if context_analysis["price_structure"] == "bullish":
            trading_signals.append(("buy", 0.6, "Bullish price structure"))
        elif context_analysis["price_structure"] == "bearish":
            trading_signals.append(("sell", 0.6, "Bearish price structure"))
            
        if context_analysis["market_phase"] == "accumulation":
            trading_signals.append(("buy", 0.7, "Accumulation phase detected"))
        elif context_analysis["market_phase"] == "distribution":
            trading_signals.append(("sell", 0.7, "Distribution phase detected"))
        elif context_analysis["market_phase"] == "consolidation":
            trading_signals.append(("wait", 0.5, "Consolidation phase - awaiting breakout"))
            
        if context_analysis["volume_profile"] == "bullish_imbalance":
            trading_signals.append(("buy", 0.65, "Bullish volume imbalance"))
        elif context_analysis["volume_profile"] == "bearish_imbalance":
            trading_signals.append(("sell", 0.65, "Bearish volume imbalance"))
            
        if context_analysis["volatility_state"] == "high":
            trading_signals.append(("reduce_size", 0.5, "High volatility - reduce position size"))
            
        # Support/resistance proximity signals
        for level in context_analysis["key_levels"]:
            if level["type"] == "resistance" and level["distance"] < 0.05:
                trading_signals.append(("caution", 0.6, f"Price approaching strong resistance at {level['price']}"))
            elif level["type"] == "support" and level["distance"] < 0.05:
                trading_signals.append(("buy_dip", 0.6, f"Price near strong support at {level['price']}"))
                
        # Count signal directions
        buy_signals = [s for s in trading_signals if s[0] in ("buy", "buy_dip")]
        sell_signals = [s for s in trading_signals if s[0] in ("sell")]
        neutral_signals = [s for s in trading_signals if s[0] in ("wait", "caution", "reduce_size")]
        
        # Calculate confidence weighted by signal strengths
        buy_confidence = sum(signal[1] for signal in buy_signals) / len(buy_signals) if buy_signals else 0
        sell_confidence = sum(signal[1] for signal in sell_signals) / len(sell_signals) if sell_signals else 0
        
        # Generate final recommendation
        if len(buy_signals) > len(sell_signals) and buy_confidence > 0.6:
            context_analysis["deep_recommendation"] = "Buy"
            context_analysis["confidence"] = buy_confidence
            context_analysis["reasoning"].extend([signal[2] for signal in buy_signals])
        elif len(sell_signals) > len(buy_signals) and sell_confidence > 0.6:
            context_analysis["deep_recommendation"] = "Sell"
            context_analysis["confidence"] = sell_confidence
            context_analysis["reasoning"].extend([signal[2] for signal in sell_signals])
        elif context_analysis["volatility_state"] == "high":
            context_analysis["deep_recommendation"] = "Reduce Risk"
            context_analysis["confidence"] = 0.7
            context_analysis["reasoning"].append("High volatility environment suggests caution")
        else:
            context_analysis["deep_recommendation"] = "Wait"
            context_analysis["confidence"] = 0.5
            context_analysis["reasoning"].append("Mixed or insufficient signals")
        
        return context_analysis
    
    def get_advanced_insights(self, symbol: str, market_data: Dict[str, Any], 
                             time_and_sales: List[Dict[str, Any]] = None,
                             order_book: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get comprehensive advanced insights by combining all analysis methods
        
        Args:
            symbol (str): Stock symbol
            market_data (Dict): Current market data 
            time_and_sales (List): Recent trades
            order_book (Dict): Level 2 order book
            
        Returns:
            Dict: Comprehensive advanced insights
        """
        symbol = symbol.upper()
        
        # Ensure we have valid data
        if not market_data:
            market_data = {"bid_price": 0, "ask_price": 0, "bid_size": 0, "ask_size": 0}
            
        if not time_and_sales:
            time_and_sales = []
            
        if not order_book:
            order_book = {"bids": [], "asks": []}
            
        # First, get basic order book analysis if we have order book data
        order_book_insights = None
        if order_book.get("bids") and order_book.get("asks"):
            order_book_insights = self.analyze_order_book(
                symbol, 
                order_book.get("bids", []), 
                order_book.get("asks", []), 
                time_and_sales
            )
        
        # Next, get trade analysis if we have trade data
        trade_insights = None
        if time_and_sales and len(time_and_sales) >= 5:
            trade_insights = self.analyze_trades(symbol, time_and_sales)
        
        # Finally, get deep market context
        context_analysis = self.analyze_market_context(
            symbol, 
            market_data,
            time_and_sales, 
            order_book
        )
        
        # Combine all insights
        combined_insights = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "market_data": market_data,
            "order_book_insights": order_book_insights,
            "trade_insights": trade_insights,
            "deep_analysis": context_analysis,
            "combined_messages": [],
            "final_recommendation": {
                "action": "Wait",
                "confidence": 0.0,
                "reasoning": []
            }
        }
        
        # Collect all messages
        if order_book_insights:
            combined_insights["combined_messages"].extend(order_book_insights.get("messages", []))
            
        if trade_insights:
            combined_insights["combined_messages"].extend(trade_insights.get("messages", []))
            
        if context_analysis:
            combined_insights["combined_messages"].extend(context_analysis.get("advanced_insights", []))
        
        # Generate final recommendation by combining all insights
        recommendations = []
        
        if order_book_insights and order_book_insights.get("recommendation"):
            recommendations.append((
                order_book_insights["recommendation"],
                order_book_insights.get("confidence", 0),
                "Order Book Analysis"
            ))
            
        if trade_insights and trade_insights.get("recommendation"):
            recommendations.append((
                trade_insights["recommendation"],
                trade_insights.get("confidence", 0),
                "Trade Analysis"
            ))
            
        if context_analysis and context_analysis.get("deep_recommendation"):
            recommendations.append((
                context_analysis["deep_recommendation"],
                context_analysis.get("confidence", 0),
                "Context Analysis"
            ))
        
        # Count recommendation types
        if recommendations:
            buy_recs = [r for r in recommendations if r[0] == "Buy"]
            sell_recs = [r for r in recommendations if r[0] == "Sell"]
            other_recs = [r for r in recommendations if r[0] not in ("Buy", "Sell")]
            
            # Find majority recommendation
            if buy_recs and len(buy_recs) >= len(sell_recs):
                # Choose the buy recommendation with highest confidence
                best_rec = max(buy_recs, key=lambda x: x[1])
                combined_insights["final_recommendation"]["action"] = "Buy"
                combined_insights["final_recommendation"]["confidence"] = best_rec[1]
                combined_insights["final_recommendation"]["reasoning"].append(f"Buy signal from {best_rec[2]}")
                if context_analysis.get("reasoning"):
                    combined_insights["final_recommendation"]["reasoning"].extend(context_analysis.get("reasoning"))
                    
            elif sell_recs:
                # Choose the sell recommendation with highest confidence
                best_rec = max(sell_recs, key=lambda x: x[1])
                combined_insights["final_recommendation"]["action"] = "Sell"
                combined_insights["final_recommendation"]["confidence"] = best_rec[1]
                combined_insights["final_recommendation"]["reasoning"].append(f"Sell signal from {best_rec[2]}")
                if context_analysis.get("reasoning"):
                    combined_insights["final_recommendation"]["reasoning"].extend(context_analysis.get("reasoning"))
                    
            elif other_recs:
                # Choose the other recommendation with highest confidence
                best_rec = max(other_recs, key=lambda x: x[1])
                combined_insights["final_recommendation"]["action"] = best_rec[0]
                combined_insights["final_recommendation"]["confidence"] = best_rec[1]
                combined_insights["final_recommendation"]["reasoning"].append(f"{best_rec[0]} signal from {best_rec[2]}")
        
        # Update last insights
        self.last_insights[symbol] = combined_insights
        
        return combined_insights


# Create a global instance of the trading intelligence system
trading_intelligence = TradingIntelligence()