import os
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

# Initialize SQLAlchemy with the base
db = SQLAlchemy(model_class=Base)

class ScreenAnalysis(db.Model):
    """Model for storing screen analysis results"""
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    extracted_text = db.Column(db.Text, nullable=True)
    analysis_result = db.Column(db.Text, nullable=True)
    
    # Trading-specific fields
    symbol = db.Column(db.String(20), nullable=True)
    bid_price = db.Column(db.Float, nullable=True)
    ask_price = db.Column(db.Float, nullable=True)
    spread = db.Column(db.Float, nullable=True)
    recommendation = db.Column(db.String(20), nullable=True)
    
    def __repr__(self):
        return f'<ScreenAnalysis {self.id} {self.timestamp}>'
        
    def to_dict(self):
        """Convert the model instance to a dictionary"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'extracted_text': self.extracted_text,
            'analysis_result': self.analysis_result,
            'symbol': self.symbol,
            'bid_price': self.bid_price,
            'ask_price': self.ask_price,
            'spread': self.spread,
            'recommendation': self.recommendation
        }


class MarketData(db.Model):
    """Model for storing real-time market data from Polygon.io"""
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    symbol = db.Column(db.String(20), nullable=False, index=True)
    
    # Level 1 data (latest quote)
    bid_price = db.Column(db.Float, nullable=True)
    bid_size = db.Column(db.Integer, nullable=True)
    ask_price = db.Column(db.Float, nullable=True)
    ask_size = db.Column(db.Integer, nullable=True)
    spread = db.Column(db.Float, nullable=True)
    last_price = db.Column(db.Float, nullable=True)
    
    # Additional market data
    volume = db.Column(db.Integer, nullable=True)
    vwap = db.Column(db.Float, nullable=True)  # Volume-weighted average price
    open_price = db.Column(db.Float, nullable=True)
    high_price = db.Column(db.Float, nullable=True)
    low_price = db.Column(db.Float, nullable=True)
    close_price = db.Column(db.Float, nullable=True)
    
    def __repr__(self):
        return f'<MarketData {self.symbol} @ {self.timestamp}>'
        
    def to_dict(self):
        """Convert the model instance to a dictionary"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'bid_price': self.bid_price,
            'bid_size': self.bid_size,
            'ask_price': self.ask_price,
            'ask_size': self.ask_size,
            'spread': self.spread,
            'last_price': self.last_price,
            'volume': self.volume,
            'vwap': self.vwap,
            'open_price': self.open_price,
            'high_price': self.high_price,
            'low_price': self.low_price,
            'close_price': self.close_price
        }


class TradeData(db.Model):
    """Model for storing Time & Sales data (individual trades)"""
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, index=True)
    symbol = db.Column(db.String(20), nullable=False, index=True)
    price = db.Column(db.Float, nullable=False)
    size = db.Column(db.Integer, nullable=False)
    exchange = db.Column(db.String(10), nullable=True)
    trade_id = db.Column(db.String(50), nullable=True, unique=True)
    tape = db.Column(db.String(2), nullable=True)  # Tape A, B, or C
    
    # Derived fields for analysis
    is_buyer_maker = db.Column(db.Boolean, nullable=True)  # True if trade hit the bid
    
    def __repr__(self):
        return f'<TradeData {self.symbol} {self.price} x {self.size} @ {self.timestamp}>'
        
    def to_dict(self):
        """Convert the model instance to a dictionary"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'price': self.price,
            'size': self.size,
            'exchange': self.exchange,
            'trade_id': self.trade_id,
            'tape': self.tape,
            'is_buyer_maker': self.is_buyer_maker
        }