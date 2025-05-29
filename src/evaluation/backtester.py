#!/usr/bin/env python3
"""
Robust Backtesting System for LNN Pipeline
Integrates with your existing training and evaluation pipeline

Save this as: src/evaluation/backtester.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import warnings
from dataclasses import dataclass, asdict
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add your existing paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Your existing imports
from models.lnn_model import LiquidNetwork, ModelConfig, create_sequences
from models.trainer import LNNTrainer
from data.data_loader import StockDataLoader
from analysis.feature_engineering import AdvancedFeatureEngineer
from analysis.market_abstraction_pipeline import EnhancedFeatureEngineer
from utils.metrics import StockPredictionMetrics

class BacktestMode(Enum):
    """Different backtesting approaches"""
    WALK_FORWARD = "walk_forward"           # Realistic time-series validation
    EXPANDING_WINDOW = "expanding_window"   # Growing training window
    ROLLING_WINDOW = "rolling_window"       # Fixed-size rolling window
    TIME_SERIES_SPLIT = "time_series_split" # Simple train/validation split

@dataclass
class BacktestPeriod:
    """Represents a single backtesting period"""
    period_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_start_date: str
    train_end_date: str
    test_start_date: str
    test_end_date: str

@dataclass
class PeriodResults:
    """Results for a single backtesting period"""
    period: BacktestPeriod
    predictions: np.ndarray
    actuals: np.ndarray
    returns: np.ndarray
    strategy_returns: np.ndarray
    benchmark_returns: np.ndarray
    metrics: Dict[str, float]
    model_path: Optional[str] = None
    training_time: float = 0.0
    prediction_time: float = 0.0

@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    mode: BacktestMode = BacktestMode.WALK_FORWARD
    initial_train_days: int = 252        # 1 year initial training
    retrain_frequency: int = 21          # Retrain every 21 days (monthly)
    test_period_days: int = 21           # Test on next 21 days
    min_train_samples: int = 100         # Minimum training samples
    max_train_samples: Optional[int] = None  # Maximum training samples (None = unlimited)
    overlap_days: int = 0                # Overlap between periods
    
    # Trading parameters
    transaction_cost: float = 0.001      # 0.1% transaction cost
    slippage: float = 0.0005            # 0.05% slippage
    position_sizing: str = "fixed"       # "fixed", "volatility", "kelly"
    max_position: float = 1.0           # Maximum position size
    
    # Model parameters
    retrain_threshold: float = 0.05      # Retrain if performance degrades by 5%
    ensemble_size: int = 1              # Number of models in ensemble
    early_stopping: bool = True         # Use early stopping in training
    
    # Output parameters
    save_models: bool = False           # Save models for each period
    save_predictions: bool = True       # Save predictions
    plot_results: bool = True          # Generate plots

class WalkForwardBacktester:
    """
    Comprehensive walk-forward backtesting system.
    Simulates realistic trading by training on historical data and testing on future data.
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        
        # Results storage
        self.period_results: List[PeriodResults] = []
        self.overall_results: Dict[str, Any] = {}
        
        # Data storage
        self.price_data: Optional[Dict[str, np.ndarray]] = None
        self.features: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None
        self.targets: Optional[np.ndarray] = None
        self.dates: Optional[List[str]] = None
        
        # Model components
        self.feature_engineer: Optional[Union[AdvancedFeatureEngineer, EnhancedFeatureEngineer]] = None
        self.model_config: Optional[ModelConfig] = None
        
        print(f"Initialized WalkForwardBacktester with {self.config.mode.value} mode")
    
    def prepare_data(self, 
                    price_data: Dict[str, np.ndarray],
                    target_ticker: str,
                    config: Dict,
                    start_date: str = None,
                    end_date: str = None,
                    use_enhanced_features: bool = True) -> None:
        """
        Prepare data for backtesting using your existing pipeline.
        
        Args:
            price_data: Your price data from data_loader.get_closing_prices()
            target_ticker: Target ticker for prediction  
            config: Your configuration dictionary
            start_date: Start date for backtesting
            end_date: End date for backtesting
            use_enhanced_features: Whether to use market abstraction features
        """
        print("🔧 Preparing data for backtesting...")
        
        self.price_data = price_data
        self.target_ticker = target_ticker
        
        # Initialize feature engineer
        if use_enhanced_features:
            self.feature_engineer = EnhancedFeatureEngineer(use_abstractions=True)
        else:
            self.feature_engineer = AdvancedFeatureEngineer()
        
        # Create OHLCV approximation (same as your run_analysis.py)
        target_prices = price_data[target_ticker]
        ohlcv_data = {
            'close': target_prices,
            'high': target_prices * 1.02,
            'low': target_prices * 0.98,
            'open': target_prices,
            'volume': np.ones_like(target_prices) * 1000000
        }
        
        # Generate features
        if use_enhanced_features:
            features, feature_names = self.feature_engineer.create_features_with_abstractions(
                price_data=price_data,
                target_ticker=target_ticker,
                ohlcv_data=ohlcv_data
            )
        else:
            features, feature_names = self.feature_engineer.create_comprehensive_features(
                ohlcv_data, include_advanced=True
            )
        
        self.features = features
        self.feature_names = feature_names
        
        # Create targets (next-day returns)
        target_prices_flat = target_prices.flatten()
        if len(target_prices_flat) > len(features):
            target_prices_flat = target_prices_flat[-len(features):]
        elif len(features) > len(target_prices_flat):
            self.features = features[-len(target_prices_flat):]
        
        # Calculate forward returns for prediction
        forward_returns = np.zeros(len(target_prices_flat))
        for i in range(len(target_prices_flat) - 1):
            forward_returns[i] = (target_prices_flat[i + 1] - target_prices_flat[i]) / target_prices_flat[i]
        
        self.targets = forward_returns[:-1]  # Remove last element (no future return)
        self.features = self.features[:-1]   # Align features
        
        # Generate date index (simplified)
        self.dates = [f"Day_{i}" for i in range(len(self.targets))]
        
        # Initialize model config
        self.model_config = ModelConfig(
            input_size=self.features.shape[1],
            hidden_size=config.get('model', {}).get('hidden_size', 50),
            output_size=1,
            sequence_length=config.get('model', {}).get('sequence_length', 30),
            learning_rate=config.get('model', {}).get('learning_rate', 0.001),
            batch_size=config.get('model', {}).get('batch_size', 32),
            num_epochs=config.get('model', {}).get('num_epochs', 100),
            patience=config.get('model', {}).get('patience', 10)
        )
        
        print(f"✓ Data prepared: {len(self.features)} samples, {len(self.feature_names)} features")
        print(f"✓ Target: {self.target_ticker} next-day returns")
        print(f"✓ Model config: {self.model_config.sequence_length}-day sequences, {self.model_config.hidden_size} hidden units")
    
    def generate_backtest_periods(self) -> List[BacktestPeriod]:
        """
        Generate backtesting periods based on configuration.
        """
        if self.features is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        total_samples = len(self.features)
        periods = []
        period_id = 0
        
        print(f"🗓️  Generating {self.config.mode.value} periods...")
        
        if self.config.mode == BacktestMode.WALK_FORWARD:
            # Walk-forward: train on expanding window, test on next period
            train_start = 0
            
            while train_start + self.config.initial_train_days + self.config.test_period_days < total_samples:
                train_end = train_start + self.config.initial_train_days
                test_start = train_end
                test_end = min(test_start + self.config.test_period_days, total_samples)
                
                if test_end - test_start < 5:  # Need at least 5 test samples
                    break
                
                period = BacktestPeriod(
                    period_id=period_id,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    train_start_date=self.dates[train_start],
                    train_end_date=self.dates[train_end-1],
                    test_start_date=self.dates[test_start],
                    test_end_date=self.dates[test_end-1]
                )
                periods.append(period)
                
                # Move to next period
                train_start = test_start
                period_id += 1
                
                # Limit for expanding window mode
                if self.config.max_train_samples and train_end - train_start > self.config.max_train_samples:
                    train_start = train_end - self.config.max_train_samples
        
        elif self.config.mode == BacktestMode.ROLLING_WINDOW:
            # Rolling window: fixed training window size, test on next period
            current_pos = 0
            
            while current_pos + self.config.initial_train_days + self.config.test_period_days < total_samples:
                train_start = current_pos
                train_end = train_start + self.config.initial_train_days
                test_start = train_end
                test_end = min(test_start + self.config.test_period_days, total_samples)
                
                if test_end - test_start < 5:
                    break
                
                period = BacktestPeriod(
                    period_id=period_id,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    train_start_date=self.dates[train_start],
                    train_end_date=self.dates[train_end-1],
                    test_start_date=self.dates[test_start],
                    test_end_date=self.dates[test_end-1]
                )
                periods.append(period)
                
                # Move forward by retrain frequency
                current_pos += self.config.retrain_frequency
                period_id += 1
        
        elif self.config.mode == BacktestMode.TIME_SERIES_SPLIT:
            # Simple split: single train/test split
            split_point = int(total_samples * 0.8)  # 80% train, 20% test
            
            period = BacktestPeriod(
                period_id=0,
                train_start=0,
                train_end=split_point,
                test_start=split_point,
                test_end=total_samples,
                train_start_date=self.dates[0],
                train_end_date=self.dates[split_point-1],
                test_start_date=self.dates[split_point],
                test_end_date=self.dates[-1]
            )
            periods.append(period)
        
        print(f"✓ Generated {len(periods)} backtesting periods")
        
        return periods
    
    def train_model_for_period(self, period: BacktestPeriod) -> Tuple[LiquidNetwork, float]:
        """
        Train a model for a specific period.
        """
        print(f"🏋️  Training model for period {period.period_id}...")
        
        # Extract training data
        X_train = self.features[period.train_start:period.train_end]
        y_train = self.targets[period.train_start:period.train_end]
        
        # Create sequences for LNN
        X_sequences, y_sequences = create_sequences(
            X_train, y_train.reshape(-1, 1), 
            self.model_config.sequence_length
        )
        
        if len(X_sequences) < 10:  # Need minimum sequences
            raise ValueError(f"Insufficient sequences for training: {len(X_sequences)}")
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_sequences)
        y_tensor = torch.FloatTensor(y_sequences)
        
        # Initialize model
        model = LiquidNetwork(
            input_size=self.model_config.input_size,
            hidden_size=self.model_config.hidden_size,
            output_size=self.model_config.output_size,
            dropout_rate=0.1
        )
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=self.model_config.learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        model.train()
        start_time = datetime.now()
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.model_config.num_epochs):
            total_loss = 0.0
            
            # Mini-batch training
            batch_size = self.model_config.batch_size
            n_batches = len(X_tensor) // batch_size
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                batch_X = X_tensor[start_idx:end_idx]
                batch_y = y_tensor[start_idx:end_idx]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / n_batches if n_batches > 0 else total_loss
            
            # Early stopping
            if self.config.early_stopping:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.model_config.patience:
                        print(f"   Early stopping at epoch {epoch}")
                        break
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        print(f"   Training completed in {training_time:.1f}s, final loss: {avg_loss:.6f}")
        
        return model, training_time
    
    def generate_predictions(self, model: LiquidNetwork, period: BacktestPeriod) -> Tuple[np.ndarray, float]:
        """
        Generate predictions for a test period.
        """
        print(f"🔮 Generating predictions for period {period.period_id}...")
        
        model.eval()
        start_time = datetime.now()
        
        predictions = []
        
        # Use expanding window for predictions (more realistic)
        for i in range(period.test_start, period.test_end):
            # Use data up to current point for prediction
            lookback_start = max(0, i - self.model_config.sequence_length)
            X_current = self.features[lookback_start:i]
            
            if len(X_current) >= self.model_config.sequence_length:
                # Take last sequence_length observations
                X_seq = X_current[-self.model_config.sequence_length:].reshape(1, self.model_config.sequence_length, -1)
                X_tensor = torch.FloatTensor(X_seq)
                
                with torch.no_grad():
                    pred = model(X_tensor)
                    predictions.append(pred.item())
            else:
                # Not enough data, use zero prediction
                predictions.append(0.0)
        
        prediction_time = (datetime.now() - start_time).total_seconds()
        
        print(f"   Generated {len(predictions)} predictions in {prediction_time:.1f}s")
        
        return np.array(predictions), prediction_time
    
    def calculate_trading_returns(self, 
                                predictions: np.ndarray, 
                                actuals: np.ndarray, 
                                benchmark_returns: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Calculate trading returns based on predictions.
        """
        # Generate trading signals
        signals = np.sign(predictions)  # Simple: long if positive prediction, short if negative
        
        # Apply position sizing
        if self.config.position_sizing == "fixed":
            positions = signals * self.config.max_position
        elif self.config.position_sizing == "volatility":
            # Scale by inverse volatility
            vol = np.std(actuals) if len(actuals) > 5 else 0.02
            vol_scaling = 0.02 / max(vol, 0.001)  # Target 2% volatility
            positions = signals * min(vol_scaling, self.config.max_position)
        else:  # Fixed for now
            positions = signals * self.config.max_position
        
        # Calculate strategy returns
        strategy_returns = positions * actuals
        
        # Apply transaction costs (when position changes)
        position_changes = np.abs(np.diff(np.concatenate(([0], positions))))
        transaction_costs = position_changes * self.config.transaction_cost
        
        # Adjust returns for costs
        adjusted_returns = strategy_returns.copy()
        adjusted_returns[1:] -= transaction_costs  # Apply costs from second period onward
        
        # Calculate metrics
        metrics = {
            'total_return': np.sum(adjusted_returns),
            'annualized_return': np.sum(adjusted_returns) * 252 / len(adjusted_returns),
            'volatility': np.std(adjusted_returns) * np.sqrt(252),
            'sharpe_ratio': np.mean(adjusted_returns) / np.std(adjusted_returns) * np.sqrt(252) if np.std(adjusted_returns) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(np.cumsum(adjusted_returns)),
            'hit_rate': np.mean((adjusted_returns > 0).astype(float)),
            'avg_win': np.mean(adjusted_returns[adjusted_returns > 0]) if np.any(adjusted_returns > 0) else 0,
            'avg_loss': np.mean(adjusted_returns[adjusted_returns < 0]) if np.any(adjusted_returns < 0) else 0,
            'profit_factor': abs(np.sum(adjusted_returns[adjusted_returns > 0]) / np.sum(adjusted_returns[adjusted_returns < 0])) if np.any(adjusted_returns < 0) else float('inf'),
            'benchmark_return': np.sum(benchmark_returns),
            'excess_return': np.sum(adjusted_returns) - np.sum(benchmark_returns),
            'information_ratio': (np.mean(adjusted_returns) - np.mean(benchmark_returns)) / np.std(adjusted_returns - benchmark_returns) if np.std(adjusted_returns - benchmark_returns) > 0 else 0
        }
        
        return adjusted_returns, metrics
    
    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown from cumulative returns."""
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / np.maximum(peak, 1e-10)
        return np.min(drawdown)
    
    def run_backtest(self) -> Dict[str, Any]:
        """
        Run the complete backtesting pipeline.
        """
        if self.features is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        print("🚀 Starting comprehensive backtesting...")
        print("=" * 60)
        
        # Generate periods
        periods = self.generate_backtest_periods()
        
        if not periods:
            raise ValueError("No backtesting periods generated")
        
        # Run backtesting for each period
        all_predictions = []
        all_actuals = []
        all_strategy_returns = []
        all_benchmark_returns = []
        
        for period in periods:
            try:
                # Train model
                model, training_time = self.train_model_for_period(period)
                
                # Generate predictions
                predictions, prediction_time = self.generate_predictions(model, period)
                
                # Get actual returns
                actuals = self.targets[period.test_start:period.test_end]
                
                # Calculate benchmark returns (buy and hold)
                benchmark_returns = actuals.copy()  # Simple benchmark
                
                # Calculate trading returns
                strategy_returns, trading_metrics = self.calculate_trading_returns(
                    predictions, actuals, benchmark_returns
                )
                
                # Calculate prediction metrics
                prediction_metrics = {
                    'mse': mean_squared_error(actuals, predictions),
                    'mae': mean_absolute_error(actuals, predictions),
                    'rmse': np.sqrt(mean_squared_error(actuals, predictions)),
                    'directional_accuracy': np.mean((np.sign(predictions) == np.sign(actuals)).astype(float)),
                    'correlation': np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0
                }
                
                # Combine metrics
                all_metrics = {**prediction_metrics, **trading_metrics}
                
                # Store results
                period_result = PeriodResults(
                    period=period,
                    predictions=predictions,
                    actuals=actuals,
                    returns=actuals,
                    strategy_returns=strategy_returns,
                    benchmark_returns=benchmark_returns,
                    metrics=all_metrics,
                    training_time=training_time,
                    prediction_time=prediction_time
                )
                
                self.period_results.append(period_result)
                
                # Accumulate for overall statistics
                all_predictions.extend(predictions)
                all_actuals.extend(actuals)
                all_strategy_returns.extend(strategy_returns)
                all_benchmark_returns.extend(benchmark_returns)
                
                # Print period summary
                print(f"Period {period.period_id}: Strategy Return={all_metrics['total_return']:.3f}, "
                      f"Sharpe={all_metrics['sharpe_ratio']:.2f}, "
                      f"Hit Rate={all_metrics['hit_rate']:.1%}")
                
            except Exception as e:
                print(f"❌ Error in period {period.period_id}: {e}")
                continue
        
        # Calculate overall results
        self.overall_results = self._calculate_overall_results(
            all_predictions, all_actuals, all_strategy_returns, all_benchmark_returns
        )
        
        print("=" * 60)
        print("🏁 Backtesting completed!")
        self._print_backtest_summary()
        
        return self.overall_results
    
    def _calculate_overall_results(self, 
                                 all_predictions: List[float],
                                 all_actuals: List[float], 
                                 all_strategy_returns: List[float],
                                 all_benchmark_returns: List[float]) -> Dict[str, Any]:
        """Calculate aggregate results across all periods."""
        
        predictions = np.array(all_predictions)
        actuals = np.array(all_actuals)
        strategy_returns = np.array(all_strategy_returns)
        benchmark_returns = np.array(all_benchmark_returns)
        
        # Overall prediction metrics
        overall_prediction_metrics = {
            'total_predictions': len(predictions),
            'overall_mse': mean_squared_error(actuals, predictions),
            'overall_mae': mean_absolute_error(actuals, predictions),
            'overall_rmse': np.sqrt(mean_squared_error(actuals, predictions)),
            'overall_directional_accuracy': np.mean((np.sign(predictions) == np.sign(actuals)).astype(float)),
            'overall_correlation': np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0
        }
        
        # Overall trading metrics
        overall_trading_metrics = {
            'total_strategy_return': np.sum(strategy_returns),
            'total_benchmark_return': np.sum(benchmark_returns),
            'annualized_strategy_return': np.sum(strategy_returns) * 252 / len(strategy_returns),
            'annualized_benchmark_return': np.sum(benchmark_returns) * 252 / len(benchmark_returns),
            'strategy_volatility': np.std(strategy_returns) * np.sqrt(252),
            'benchmark_volatility': np.std(benchmark_returns) * np.sqrt(252),
            'strategy_sharpe': np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0,
            'benchmark_sharpe': np.mean(benchmark_returns) / np.std(benchmark_returns) * np.sqrt(252) if np.std(benchmark_returns) > 0 else 0,
            'excess_return': np.sum(strategy_returns) - np.sum(benchmark_returns),
            'strategy_max_drawdown': self._calculate_max_drawdown(np.cumsum(strategy_returns)),
            'benchmark_max_drawdown': self._calculate_max_drawdown(np.cumsum(benchmark_returns)),
            'hit_rate': np.mean((strategy_returns > 0).astype(float)),
            'avg_period_return': np.mean([r.metrics['total_return'] for r in self.period_results]),
            'win_rate': np.mean([1 if r.metrics['total_return'] > 0 else 0 for r in self.period_results]),
        }
        
        # Period-level statistics
        period_stats = {
            'num_periods': len(self.period_results),
            'avg_training_time': np.mean([r.training_time for r in self.period_results]),
            'avg_prediction_time': np.mean([r.prediction_time for r in self.period_results]),
            'successful_periods': len([r for r in self.period_results if r.metrics['total_return'] > 0]),
            'failed_periods': len([r for r in self.period_results if r.metrics['total_return'] <= 0])
        }
        
        # Combine all results
        overall_results = {
            'backtest_config': asdict(self.config),
            'prediction_metrics': overall_prediction_metrics,
            'trading_metrics': overall_trading_metrics,
            'period_statistics': period_stats,
            'period_results': [asdict(r) for r in self.period_results],
            'backtest_completed': datetime.now().isoformat(),
            'data_info': {
                'target_ticker': getattr(self, 'target_ticker', 'Unknown'),
                'total_samples': len(all_predictions),
                'feature_count': len(self.feature_names) if self.feature_names else 0
            }
        }
        
        return overall_results
    
    def _print_backtest_summary(self):
        """Print comprehensive backtest summary."""
        
        if not self.overall_results:
            return
        
        print("\n📊 BACKTESTING SUMMARY")
        print("=" * 50)
        
        # Basic info
        trading_metrics = self.overall_results['trading_metrics']
        prediction_metrics = self.overall_results['prediction_metrics']
        period_stats = self.overall_results['period_statistics']
        
        print(f"Target Asset: {self.overall_results['data_info']['target_ticker']}")
        print(f"Backtesting Mode: {self.config.mode.value}")
        print(f"Number of Periods: {period_stats['num_periods']}")
        print(f"Total Predictions: {prediction_metrics['total_predictions']}")
        
        print(f"\n📈 TRADING PERFORMANCE:")
        print(f"Strategy Return: {trading_metrics['total_strategy_return']:.1%}")
        print(f"Benchmark Return: {trading_metrics['total_benchmark_return']:.1%}")
        print(f"Excess Return: {trading_metrics['excess_return']:.1%}")
        print(f"Strategy Sharpe: {trading_metrics['strategy_sharpe']:.2f}")
        print(f"Max Drawdown: {trading_metrics['strategy_max_drawdown']:.1%}")
        print(f"Hit Rate: {trading_metrics['hit_rate']:.1%}")
        print(f"Win Rate: {trading_metrics['win_rate']:.1%}")
        
        print(f"\n🎯 PREDICTION ACCURACY:")
        print(f"Directional Accuracy: {prediction_metrics['overall_directional_accuracy']:.1%}")
        print(f"Correlation: {prediction_metrics['overall_correlation']:.3f}")
        print(f"RMSE: {prediction_metrics['overall_rmse']:.4f}")
        
        print(f"\n⏱️  PERFORMANCE:")
        print(f"Avg Training Time: {period_stats['avg_training_time']:.1f}s")
        print(f"Avg Prediction Time: {period_stats['avg_prediction_time']:.1f}s")
        print(f"Successful Periods: {period_stats['successful_periods']}/{period_stats['num_periods']}")
        
        # Performance verdict
        if trading_metrics['excess_return'] > 0.05:  # 5% outperformance
            verdict = "🟢 STRONG PERFORMANCE"
        elif trading_metrics['excess_return'] > 0:
            verdict = "🟡 MODEST OUTPERFORMANCE"
        elif trading_metrics['excess_return'] > -0.05:
            verdict = "🟠 NEUTRAL PERFORMANCE"
        else:
            verdict = "🔴 UNDERPERFORMANCE"
        
        print(f"\n{verdict}")
    
    def save_backtest_results(self, filepath: str = None):
        """Save comprehensive backtest results to file."""
        
        if not self.overall_results:
            print("No backtest results to save")
            return
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target = getattr(self, 'target_ticker', 'unknown')
            filepath = f"results/backtests/backtest_{target}_{self.config.mode.value}_{timestamp}.json"
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.overall_results, f, indent=2, default=str)
        
        print(f"✓ Backtest results saved to: {filepath}")
        return filepath
    
    def plot_results(self, save_path: str = None):
        """Generate comprehensive backtesting plots."""
        
        if not self.period_results:
            print("No results to plot")
            return
        
        # Prepare data for plotting
        all_strategy_returns = []
        all_benchmark_returns = []
        all_predictions = []
        all_actuals = []
        period_labels = []
        
        for result in self.period_results:
            all_strategy_returns.extend(result.strategy_returns)
            all_benchmark_returns.extend(result.benchmark_returns)
            all_predictions.extend(result.predictions)
            all_actuals.extend(result.actuals)
            period_labels.extend([f"P{result.period.period_id}"] * len(result.strategy_returns))
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Backtesting Results - {getattr(self, "target_ticker", "Unknown")}', fontsize=16)
        
        # 1. Cumulative Returns
        strategy_cumret = np.cumsum(all_strategy_returns)
        benchmark_cumret = np.cumsum(all_benchmark_returns)
        
        axes[0, 0].plot(strategy_cumret, label='Strategy', linewidth=2)
        axes[0, 0].plot(benchmark_cumret, label='Benchmark', linewidth=2)
        axes[0, 0].set_title('Cumulative Returns')
        axes[0, 0].set_xlabel('Days')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Prediction vs Actual
        axes[0, 1].scatter(all_actuals, all_predictions, alpha=0.6, s=20)
        axes[0, 1].plot([min(all_actuals), max(all_actuals)], [min(all_actuals), max(all_actuals)], 'r--', linewidth=2)
        axes[0, 1].set_title('Predictions vs Actual Returns')
        axes[0, 1].set_xlabel('Actual Returns')
        axes[0, 1].set_ylabel('Predicted Returns')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add correlation text
        correlation = np.corrcoef(all_predictions, all_actuals)[0, 1]
        axes[0, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                       transform=axes[0, 1].transAxes, fontsize=12, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 3. Period-by-Period Returns
        period_returns = [result.metrics['total_return'] for result in self.period_results]
        period_ids = [result.period.period_id for result in self.period_results]
        
        colors = ['green' if ret > 0 else 'red' for ret in period_returns]
        axes[1, 0].bar(period_ids, period_returns, color=colors, alpha=0.7)
        axes[1, 0].set_title('Period Returns')
        axes[1, 0].set_xlabel('Period')
        axes[1, 0].set_ylabel('Return')
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Rolling Sharpe Ratio
        window = 21  # 21-day rolling window
        if len(all_strategy_returns) > window:
            rolling_sharpe = []
            for i in range(window, len(all_strategy_returns)):
                returns_window = all_strategy_returns[i-window:i]
                if len(returns_window) > 0 and np.std(returns_window) > 0:
                    sharpe = np.mean(returns_window) / np.std(returns_window) * np.sqrt(252)
                    rolling_sharpe.append(sharpe)
                else:
                    rolling_sharpe.append(0)
            
            axes[1, 1].plot(range(window, len(all_strategy_returns)), rolling_sharpe, linewidth=2)
            axes[1, 1].set_title(f'{window}-Day Rolling Sharpe Ratio')
            axes[1, 1].set_xlabel('Days')
            axes[1, 1].set_ylabel('Sharpe Ratio')
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1, 1].axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Sharpe = 1')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor rolling Sharpe', 
                           ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plots saved to: {save_path}")
        
        plt.show()
        return fig
    
    def get_feature_importance_over_time(self) -> Dict[str, List[float]]:
        """
        Analyze how feature importance changes over time (if using tree-based models).
        For LNN, this returns prediction correlation with features over time.
        """
        if not self.feature_names or not self.period_results:
            return {}
        
        print("📊 Analyzing feature importance over time...")
        
        feature_importance = {name: [] for name in self.feature_names}
        
        for result in self.period_results:
            period = result.period
            predictions = result.predictions
            
            # Calculate correlation between predictions and each feature
            for i, feature_name in enumerate(self.feature_names):
                feature_values = self.features[period.test_start:period.test_end, i]
                
                if len(feature_values) == len(predictions) and np.std(feature_values) > 0:
                    correlation = np.corrcoef(predictions, feature_values)[0, 1]
                    feature_importance[feature_name].append(abs(correlation))
                else:
                    feature_importance[feature_name].append(0.0)
        
        return feature_importance
    
    def analyze_regime_performance(self) -> Dict[str, Any]:
        """
        Analyze performance across different market regimes.
        """
        if not self.period_results:
            return {}
        
        print("📈 Analyzing regime-based performance...")
        
        # Simple regime classification based on benchmark returns
        regime_results = {'bull': [], 'bear': [], 'sideways': []}
        
        for result in self.period_results:
            benchmark_return = result.metrics['benchmark_return']
            
            if benchmark_return > 0.02:  # >2% benchmark return
                regime = 'bull'
            elif benchmark_return < -0.02:  # <-2% benchmark return
                regime = 'bear'
            else:
                regime = 'sideways'
            
            regime_results[regime].append({
                'period_id': result.period.period_id,
                'strategy_return': result.metrics['total_return'],
                'benchmark_return': benchmark_return,
                'excess_return': result.metrics['excess_return'],
                'sharpe_ratio': result.metrics['sharpe_ratio'],
                'hit_rate': result.metrics['hit_rate']
            })
        
        # Calculate regime statistics
        regime_stats = {}
        for regime, results in regime_results.items():
            if results:
                regime_stats[regime] = {
                    'count': len(results),
                    'avg_strategy_return': np.mean([r['strategy_return'] for r in results]),
                    'avg_benchmark_return': np.mean([r['benchmark_return'] for r in results]),
                    'avg_excess_return': np.mean([r['excess_return'] for r in results]),
                    'avg_sharpe': np.mean([r['sharpe_ratio'] for r in results]),
                    'avg_hit_rate': np.mean([r['hit_rate'] for r in results]),
                    'win_rate': np.mean([1 if r['strategy_return'] > 0 else 0 for r in results])
                }
            else:
                regime_stats[regime] = {'count': 0}
        
        return {
            'regime_classification': regime_results,
            'regime_statistics': regime_stats
        }


class BacktestIntegrator:
    """
    Integration helpers to connect backtesting with your existing pipeline.
    """
    
    @staticmethod
    def integrate_with_run_analysis(analyzer_class):
        """
        Add backtesting capabilities to your ComprehensiveAnalyzer class.
        """
        
        def run_comprehensive_backtest(self, backtest_config: BacktestConfig = None):
            """
            Add this method to your ComprehensiveAnalyzer class.
            
            Run comprehensive backtesting as part of your analysis pipeline.
            """
            print("=" * 70)
            print("PHASE 3.5: COMPREHENSIVE BACKTESTING")
            print("=" * 70)
            
            # Use default config if none provided
            if backtest_config is None:
                backtest_config = BacktestConfig(
                    mode=BacktestMode.WALK_FORWARD,
                    initial_train_days=252,    # 1 year
                    retrain_frequency=21,      # Monthly
                    test_period_days=21        # Test on next month
                )
            
            # Initialize backtester
            backtester = WalkForwardBacktester(backtest_config)
            
            # Prepare data (use your existing data)
            if not hasattr(self, 'data_loader') or not self.data_loader:
                raise ValueError("Data not loaded. Run analyze_raw_data() first.")
            
            price_data = self.data_loader.get_closing_prices()
            target_ticker = self.config['data']['target_ticker']
            
            # Prepare backtesting data
            backtester.prepare_data(
                price_data=price_data,
                target_ticker=target_ticker,
                config=self.config,
                use_enhanced_features=True  # Use your market abstraction features
            )
            
            # Run backtest
            backtest_results = backtester.run_backtest()
            
            # Store results
            self.analysis_results['backtesting'] = backtest_results
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backtest_file = f"results/backtests/backtest_{self.experiment_name}_{timestamp}.json"
            backtester.save_backtest_results(backtest_file)
            
            # Generate plots
            if backtest_config.plot_results:
                plot_file = f"results/backtests/backtest_plots_{self.experiment_name}_{timestamp}.png"
                backtester.plot_results(plot_file)
            
            print(f"✅ Comprehensive backtesting completed!")
            print(f"   Results saved to: {backtest_file}")
            
            return backtest_results
        
        # Add the method to the class
        analyzer_class.run_comprehensive_backtest = run_comprehensive_backtest
        return analyzer_class
    
    @staticmethod
    def create_backtest_only_script():
        """
        Create a standalone backtesting script for your pipeline.
        """
        
        script_content = '''#!/usr/bin/env python3
"""
Standalone Backtesting Script for LNN Pipeline
Run this to backtest your models without full analysis pipeline.

Usage on Jetson Nano:
    python scripts/run_backtest.py --config config/config.yaml --target AAPL
    python scripts/run_backtest.py --quick  # Quick backtest with default settings
"""

import os
import sys
import argparse
import yaml
from datetime import datetime

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_loader import StockDataLoader
from evaluation.backtester import WalkForwardBacktester, BacktestConfig, BacktestMode

def main():
    parser = argparse.ArgumentParser(description='Run backtesting for LNN models')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file path')
    parser.add_argument('--target', type=str, help='Target ticker (overrides config)')
    parser.add_argument('--mode', type=str, default='walk_forward', 
                       choices=['walk_forward', 'rolling_window', 'time_series_split'],
                       help='Backtesting mode')
    parser.add_argument('--quick', action='store_true', help='Quick backtest with minimal periods')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override target if specified
    if args.target:
        config['data']['target_ticker'] = args.target
    
    # Create backtest config
    if args.quick:
        backtest_config = BacktestConfig(
            mode=BacktestMode(args.mode),
            initial_train_days=100,  # Shorter for quick test
            retrain_frequency=10,
            test_period_days=10,
            plot_results=args.plot
        )
    else:
        backtest_config = BacktestConfig(
            mode=BacktestMode(args.mode),
            initial_train_days=252,
            retrain_frequency=21,
            test_period_days=21,
            plot_results=args.plot
        )
    
    print(f"🚀 Starting backtesting for {config['data']['target_ticker']}")
    print(f"   Mode: {args.mode}")
    print(f"   Quick: {args.quick}")
    
    try:
        # Load data
        data_loader = StockDataLoader(
            tickers=config['data']['tickers'],
            start_date=config['data']['start_date'],
            end_date=config['data']['end_date']
        )
        
        raw_data = data_loader.download_data()
        price_data = data_loader.get_closing_prices()
        
        # Initialize backtester
        backtester = WalkForwardBacktester(backtest_config)
        
        # Prepare data
        backtester.prepare_data(
            price_data=price_data,
            target_ticker=config['data']['target_ticker'],
            config=config,
            use_enhanced_features=True
        )
        
        # Run backtest
        results = backtester.run_backtest()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results/backtests/standalone_backtest_{timestamp}.json"
        backtester.save_backtest_results(results_file)
        
        if args.plot:
            plot_file = f"results/backtests/standalone_plots_{timestamp}.png"
            backtester.plot_results(plot_file)
        
        print(f"\\n✅ Backtesting completed successfully!")
        print(f"   Results: {results_file}")
        
    except Exception as e:
        print(f"❌ Backtesting failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
'''
        
        return script_content


# Example usage and testing functions
def test_backtesting_system():
    """Test the backtesting system with sample data."""
    
    print("🧪 Testing Backtesting System...")
    print("=" * 40)
    
    try:
        # Load sample data
        from data.data_loader import StockDataLoader
        
        data_loader = StockDataLoader(['AAPL', 'SPY'], '2022-01-01', '2024-01-01')
        raw_data = data_loader.download_data()
        price_data = data_loader.get_closing_prices()
        
        # Test config
        config = {
            'data': {'target_ticker': 'AAPL'},
            'model': {
                'hidden_size': 32,
                'sequence_length': 20,
                'learning_rate': 0.001,
                'batch_size': 16,
                'num_epochs': 50,
                'patience': 5
            }
        }
        
        # Quick backtest config
        backtest_config = BacktestConfig(
            mode=BacktestMode.WALK_FORWARD,
            initial_train_days=100,
            retrain_frequency=20,
            test_period_days=10,
            early_stopping=True,
            plot_results=False  # Don't plot during test
        )
        
        # Initialize and run backtester
        backtester = WalkForwardBacktester(backtest_config)
        backtester.prepare_data(price_data, 'AAPL', config, use_enhanced_features=False)
        
        results = backtester.run_backtest()
        
        print(f"✅ Backtesting test completed!")
        print(f"   Strategy Return: {results['trading_metrics']['total_strategy_return']:.3f}")
        print(f"   Sharpe Ratio: {results['trading_metrics']['strategy_sharpe']:.2f}")
        print(f"   Directional Accuracy: {results['prediction_metrics']['overall_directional_accuracy']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Backtesting test failed: {e}")
        return False


if __name__ == "__main__":
    # Run test if script is executed directly
    success = test_backtesting_system()
    
    if success:
        print("\n🎉 Backtesting system is ready for integration!")
        print("\nNext steps:")
        print("1. Save this as src/evaluation/backtester.py")
        print("2. Create results/backtests/ directory")
        print("3. Add backtesting to your run_analysis.py")
        print("4. Run: python scripts/run_analysis.py --experiment-name 'backtest_test'")
    else:
        print("\n⚠️  Fix the issues above before integrating backtesting")
