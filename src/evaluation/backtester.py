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
        print(f"Avg Training Time: {period_stats['avg_training_time
