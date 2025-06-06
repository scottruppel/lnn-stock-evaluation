#!/usr/bin/env python3
"""
Enhanced Trading Strategy Module for LNN Model
This creates sophisticated trading logic that goes beyond simple buy/sell signals.

Save this as: src/models/trading_strategy.py on your Jetson Orin Nano
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class TradeAction(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0
    CLOSE_LONG = -2
    CLOSE_SHORT = 2

@dataclass
class TradingSignal:
    """Comprehensive trading signal with position sizing and risk management."""
    action: TradeAction
    confidence: float  # 0-1, how confident the model is
    position_size: float  # What percentage of portfolio to use
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: str = ""

@dataclass
class MarketState:
    """Current market conditions that affect trading decisions."""
    volatility_regime: str  # "low", "normal", "high"
    trend_direction: str   # "up", "down", "sideways"
    trend_strength: float  # 0-1
    volume_profile: str    # "low", "normal", "high"
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None

class EnhancedTradingStrategy:
    """
    Sophisticated trading strategy that converts LNN predictions into actionable trades.
    This goes far beyond simple buy/sell based on price direction.
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 max_position_size: float = 0.2,  # Max 20% of portfolio per trade
                 risk_per_trade: float = 0.02,    # Risk max 2% per trade
                 volatility_window: int = 20,
                 confidence_threshold: float = 0.6):
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = max_position_size
        self.risk_per_trade = risk_per_trade
        self.volatility_window = volatility_window
        self.confidence_threshold = confidence_threshold
        
        # Trading state
        self.current_position = 0.0  # Current position size (-1 to +1)
        self.entry_price = None
        self.trade_history = []
        self.consecutive_losses = 0
        
        # Risk management
        self.max_drawdown = 0.15  # Max 15% drawdown
        self.daily_loss_limit = 0.05  # Max 5% loss per day
        self.position_correlation_limit = 0.8  # Don't over-correlate positions
        
    def analyze_market_state(self, 
                           prices: np.ndarray, 
                           volumes: Optional[np.ndarray] = None,
                           additional_features: Optional[Dict[str, np.ndarray]] = None) -> MarketState:
        """
        Analyze current market conditions to inform trading decisions.
        This is where your LNN's abstracted features become really valuable.
        """
        
        # Calculate volatility regime
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns[-self.volatility_window:]) * np.sqrt(252)
        
        if volatility < 0.15:
            volatility_regime = "low"
        elif volatility > 0.30:
            volatility_regime = "high"
        else:
            volatility_regime = "normal"
        
        # Determine trend
        short_ma = np.mean(prices[-5:])
        long_ma = np.mean(prices[-20:])
        
        if short_ma > long_ma * 1.02:
            trend_direction = "up"
            trend_strength = min((short_ma / long_ma - 1) * 10, 1.0)
        elif short_ma < long_ma * 0.98:
            trend_direction = "down"
            trend_strength = min((1 - short_ma / long_ma) * 10, 1.0)
        else:
            trend_direction = "sideways"
            trend_strength = 0.1
        
        # Calculate support/resistance (simplified)
        recent_prices = prices[-50:]
        support_level = np.percentile(recent_prices, 10)
        resistance_level = np.percentile(recent_prices, 90)
        
        # Volume analysis (if available)
        volume_profile = "normal"
        if volumes is not None:
            avg_volume = np.mean(volumes[-20:])
            recent_volume = np.mean(volumes[-5:])
            if recent_volume > avg_volume * 1.5:
                volume_profile = "high"
            elif recent_volume < avg_volume * 0.5:
                volume_profile = "low"
        
        return MarketState(
            volatility_regime=volatility_regime,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            volume_profile=volume_profile,
            support_level=support_level,
            resistance_level=resistance_level
        )
    
    def calculate_position_size(self, 
                              prediction: float,
                              confidence: float,
                              current_price: float,
                              market_state: MarketState) -> float:
        """
        Calculate optimal position size based on prediction, confidence, and risk management.
        This is where sophisticated money management happens.
        """
        
        # Base position size from Kelly Criterion (simplified)
        base_size = min(abs(prediction) * confidence, self.max_position_size)
        
        # Adjust for market volatility
        if market_state.volatility_regime == "high":
            base_size *= 0.5  # Reduce size in high volatility
        elif market_state.volatility_regime == "low":
            base_size *= 1.2  # Slightly increase in low volatility
        
        # Adjust for trend strength
        if market_state.trend_direction != "sideways":
            # Increase size when trading with strong trends
            base_size *= (1 + market_state.trend_strength * 0.3)
        
        # Risk management adjustments
        if self.consecutive_losses >= 3:
            base_size *= 0.5  # Reduce size after consecutive losses
        
        # Portfolio heat check (don't risk too much total capital)
        max_risk = self.current_capital * self.risk_per_trade
        if base_size * current_price > max_risk:
            base_size = max_risk / current_price
        
        return min(base_size, self.max_position_size)
    
    def calculate_stop_loss_take_profit(self, 
                                      entry_price: float,
                                      prediction: float,
                                      market_state: MarketState,
                                      position_direction: int) -> Tuple[float, float]:
        """
        Calculate dynamic stop loss and take profit levels.
        """
        
        # Base stop loss (2-3% depending on volatility)
        if market_state.volatility_regime == "high":
            stop_loss_pct = 0.03
        else:
            stop_loss_pct = 0.02
        
        # Take profit based on prediction magnitude and market conditions
        take_profit_pct = abs(prediction) * 2  # Target 2x the predicted move
        
        # Adjust for support/resistance levels
        if position_direction > 0:  # Long position
            stop_loss = entry_price * (1 - stop_loss_pct)
            take_profit = entry_price * (1 + take_profit_pct)
            
            # Adjust for support/resistance
            if market_state.support_level and stop_loss > market_state.support_level:
                stop_loss = market_state.support_level * 0.99
            if market_state.resistance_level and take_profit > market_state.resistance_level:
                take_profit = market_state.resistance_level * 0.99
                
        else:  # Short position
            stop_loss = entry_price * (1 + stop_loss_pct)
            take_profit = entry_price * (1 - take_profit_pct)
            
            # Adjust for support/resistance
            if market_state.resistance_level and stop_loss < market_state.resistance_level:
                stop_loss = market_state.resistance_level * 1.01
            if market_state.support_level and take_profit < market_state.support_level:
                take_profit = market_state.support_level * 1.01
        
        return stop_loss, take_profit
    
    def generate_trading_signal(self, 
                              lnn_prediction: float,
                              current_price: float,
                              price_history: np.ndarray,
                              volumes: Optional[np.ndarray] = None,
                              model_confidence: Optional[float] = None) -> TradingSignal:
        """
        Main function: Convert LNN prediction into sophisticated trading signal.
        This is where the magic happens - sophisticated decision making.
        """
        
        # Analyze current market state
        market_state = self.analyze_market_state(price_history, volumes)
        
        # Determine model confidence (if not provided)
        if model_confidence is None:
            # Use prediction magnitude as proxy for confidence
            model_confidence = min(abs(lnn_prediction) * 2, 1.0)
        
        # Check if we should trade at all
        if model_confidence < self.confidence_threshold:
            return TradingSignal(
                action=TradeAction.HOLD,
                confidence=model_confidence,
                position_size=0.0,
                reasoning="Model confidence too low"
            )
        
        # Risk management checks
        if self._check_risk_limits():
            return TradingSignal(
                action=TradeAction.HOLD,
                confidence=model_confidence,
                position_size=0.0,
                reasoning="Risk limits exceeded"
            )
        
        # Determine base action
        if lnn_prediction > 0.005:  # Positive prediction > 0.5%
            base_action = TradeAction.BUY
            position_direction = 1
        elif lnn_prediction < -0.005:  # Negative prediction < -0.5%
            base_action = TradeAction.SELL
            position_direction = -1
        else:
            base_action = TradeAction.HOLD
            position_direction = 0
        
        if base_action == TradeAction.HOLD:
            return TradingSignal(
                action=TradeAction.HOLD,
                confidence=model_confidence,
                position_size=0.0,
                reasoning="Prediction magnitude too small"
            )
        
        # Calculate position size
        position_size = self.calculate_position_size(
            lnn_prediction, model_confidence, current_price, market_state
        )
        
        # Calculate stop loss and take profit
        stop_loss, take_profit = self.calculate_stop_loss_take_profit(
            current_price, lnn_prediction, market_state, position_direction
        )
        
        # Create reasoning string
        reasoning = f"Pred: {lnn_prediction:.3f}, Conf: {model_confidence:.2f}, "
        reasoning += f"Market: {market_state.trend_direction} trend, "
        reasoning += f"{market_state.volatility_regime} vol"
        
        return TradingSignal(
            action=base_action,
            confidence=model_confidence,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning
        )
    
    def _check_risk_limits(self) -> bool:
        """Check if we're hitting any risk management limits."""
        
        # Check max drawdown
        current_drawdown = (self.initial_capital - self.current_capital) / self.initial_capital
        if current_drawdown > self.max_drawdown:
            return True
        
        # Check daily loss limit (simplified)
        if self.consecutive_losses >= 5:
            return True
        
        return False
    
    def update_position(self, signal: TradingSignal, execution_price: float):
        """Update our position and trading state after executing a trade."""
        
        if signal.action in [TradeAction.BUY, TradeAction.SELL]:
            self.current_position = signal.position_size * signal.action.value
            self.entry_price = execution_price
            
            # Record trade
            trade_record = {
                'timestamp': pd.Timestamp.now(),
                'action': signal.action.name,
                'price': execution_price,
                'size': signal.position_size,
                'confidence': signal.confidence,
                'reasoning': signal.reasoning
            }
            self.trade_history.append(trade_record)
        
        elif signal.action == TradeAction.HOLD:
            pass  # No position change
        
        # Reset consecutive losses on successful trade (simplified)
        if signal.action != TradeAction.HOLD:
            # You'd implement proper P&L tracking here
            pass
    
    def get_strategy_stats(self) -> Dict[str, float]:
        """Get comprehensive strategy performance statistics."""
        
        if not self.trade_history:
            return {'total_trades': 0}
        
        df = pd.DataFrame(self.trade_history)
        
        stats = {
            'total_trades': len(df),
            'buy_trades': len(df[df['action'] == 'BUY']),
            'sell_trades': len(df[df['action'] == 'SELL']),
            'avg_confidence': df['confidence'].mean(),
            'avg_position_size': df['size'].mean(),
            'current_position': self.current_position,
            'consecutive_losses': self.consecutive_losses
        }
        
        return stats

# Example usage function for your run_analysis.py
def integrate_enhanced_strategy_with_lnn():
    """
    Example of how to integrate this enhanced strategy with your existing LNN model.
    Add this to your evaluation pipeline in evaluate_model.py
    """
    
    print("🎯 ENHANCED TRADING STRATEGY INTEGRATION")
    print("=" * 50)
    
    # Initialize the enhanced strategy
    strategy = EnhancedTradingStrategy(
        initial_capital=100000,
        max_position_size=0.15,  # Max 15% per trade
        risk_per_trade=0.02      # Risk 2% per trade
    )
    
    # Example of how to use it with your LNN predictions
    # (This would be integrated into your existing evaluation code)
    
    # Mock data for demonstration
    mock_predictions = np.array([0.015, -0.008, 0.003, 0.022, -0.012])
    mock_prices = np.array([150.0, 151.5, 148.2, 152.1, 149.8])
    mock_price_history = np.random.randn(100) * 2 + 150  # 100 days of mock data
    
    print("Generating trading signals for sample predictions...")
    
    for i, (prediction, current_price) in enumerate(zip(mock_predictions, mock_prices)):
        # Get the price history up to this point
        history_end = min(50 + i * 10, len(mock_price_history))
        price_history = mock_price_history[:history_end]
        
        # Generate sophisticated trading signal
        signal = strategy.generate_trading_signal(
            lnn_prediction=prediction,
            current_price=current_price,
            price_history=price_history
        )
        
        print(f"\nPrediction {i+1}:")
        print(f"  LNN Prediction: {prediction:.3f}")
        print(f"  Action: {signal.action.name}")
        print(f"  Confidence: {signal.confidence:.2f}")
        print(f"  Position Size: {signal.position_size:.3f}")
        if signal.stop_loss:
            print(f"  Stop Loss: ${signal.stop_loss:.2f}")
        if signal.take_profit:
            print(f"  Take Profit: ${signal.take_profit:.2f}")
        print(f"  Reasoning: {signal.reasoning}")
        
        # Update position (in real trading, this would happen after execution)
        strategy.update_position(signal, current_price)
    
    # Get strategy statistics
    stats = strategy.get_strategy_stats()
    print(f"\n📊 STRATEGY STATISTICS:")
    print("-" * 30)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print(f"\n✅ Enhanced strategy integration example complete!")
    return strategy

if __name__ == "__main__":
    # Run the integration example
    integrate_enhanced_strategy_with_lnn()