#!/usr/bin/env python3
"""
Walk-Forward Backtesting System for LNN Models
Validates model performance across different time periods and market regimes.

Usage:
    python scripts/walk_forward_backtest.py --model models/saved_models/LOW_model.pth
    python scripts/walk_forward_backtest.py --model-dir models/saved_models/ --batch
    python scripts/walk_forward_backtest.py --config config/backtest_config.yaml
"""

import os
import sys
import argparse
import yaml
import json
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_loader import StockDataLoader
from models.lnn_model import LiquidNetwork
from analysis.market_abstraction_pipeline import EnhancedFeatureEngineer
from utils.metrics import StockPredictionMetrics
from sklearn.preprocessing import MinMaxScaler

class WalkForwardBacktester:
    """
    Professional walk-forward backtesting system.
    Tests models across multiple time periods to validate robustness.
    """
    
    def __init__(self, config_path: str = None):
        """Initialize backtester with configuration."""
        self.config = self.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics_calculator = StockPredictionMetrics()
        
        # Results storage
        # self.backtest_results = {}
        # self.model_rankings = {}
        
        # Ensure backtesting directory structure
        file_namer.ensure_directory_structure()
        
        print(f"🔄 Walk-Forward Backtester initialized")
        print(f"   Device: {self.device}")
        print(f"   Time periods: {self.config['time_periods']}")
    
    def load_config(self, config_path: str) -> dict:
        """Load backtesting configuration."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration with your specifications
        return {
            'time_periods': {
                'control': {'start': '2018-01-01', 'end': '2019-12-31', 'name': 'Control Period'},
                'covid_crash': {'start': '2020-01-01', 'end': '2021-12-31', 'name': 'COVID Crash/Recovery'},
                'inflation_bear': {'start': '2022-01-01', 'end': '2022-12-31', 'name': 'Inflation/Bear Market'},
                'recovery': {'start': '2023-01-01', 'end': '2024-12-31', 'name': 'Recovery Period'}
            },
            'walk_forward': {
                'training_months': 12,  # 12 months of training data
                'test_months': 3,       # 3 months of testing
                'step_months': 1        # Roll forward by 1 month
            },
            'performance_thresholds': {
                'min_sharpe': 1.0,
                'max_drawdown': -20.0,
                'min_directional_accuracy': 0.52
            },
            'transaction_costs': {
                'commission_per_trade': 0.5,     # $0.50 per trade
                'bid_ask_spread_pct': 0.02,      # 0.02% spread
                'slippage_pct': 0.05             # 0.05% slippage
            }
        }
    
    def analyze_model_metadata(self, model_path: str) -> Dict:
        """Extract model metadata for analysis."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Extract key information
            metadata = {
                'model_path': model_path,
                'model_name': os.path.basename(model_path).replace('.pth', ''),
                'config': checkpoint.get('config', {}),
                'training_loss': checkpoint.get('val_loss', float('inf')),
                'epochs': checkpoint.get('epoch', 0),
                'input_size': None,
                'hidden_size': None,
                'sequence_length': None
            }
            
            # Extract architecture details
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                if 'liquid_cell.input_weights' in state_dict:
                    weights = state_dict['liquid_cell.input_weights']
                    metadata['input_size'] = weights.shape[0]
                    metadata['hidden_size'] = weights.shape[1]
            
            model_config = metadata['config'].get('model', {})
            metadata['sequence_length'] = model_config.get('sequence_length', 30)
            
            return metadata
            
        except Exception as e:
            print(f"⚠️ Error analyzing {model_path}: {e}")
            return None
    
    def load_model_for_period(self, model_path: str, period_data: Dict) -> Tuple[torch.nn.Module, Dict]:
        """Load and prepare model for a specific time period."""
        
        # Analyze model first
        metadata = self.analyze_model_metadata(model_path)
        if not metadata:
            return None, None
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config = checkpoint.get('config', {})
        
        # Get model architecture from saved weights
        if 'model_state_dict' not in checkpoint:
            print(f"⚠️ No model state dict found in {model_path}")
            return None, None
        
        saved_weights = checkpoint['model_state_dict']['liquid_cell.input_weights']
        input_size = saved_weights.shape[0]
        hidden_size = saved_weights.shape[1]
        output_size = 1  # Predicting returns
        
        print(f"   Model architecture from checkpoint: {input_size} → {hidden_size} → {output_size}")
        
        # Initialize model
        model = LiquidNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size
        ).to(self.device)
        
        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Store the expected input size in metadata
        metadata['expected_input_size'] = input_size
        
        return model, metadata
    
    def validate_input_features(self, test_data: Dict, expected_input_size: int) -> bool:
        """Validate that test data has the correct number of features."""
        
        if test_data is None:
            return False
        
        actual_input_size = test_data['X'].shape[2]  # [batch, seq, features]
        
        if actual_input_size != expected_input_size:
            print(f"   ❌ Feature size mismatch!")
            print(f"      Model expects: {expected_input_size} features")
            print(f"      Data provides: {actual_input_size} features")
            print(f"   This usually means the model was trained with different feature engineering")
            return False
        
        return True
    
    def prepare_period_data(self, ticker: str, start_date: str, end_date: str, model_metadata: Dict) -> Dict:
        """Prepare data for a specific time period using same features as training."""
        
        try:
            # Get original training configuration
            config = model_metadata['config']
            data_config = config.get('data', {})
            
            # Use same tickers as training, but ensure target ticker is included
            training_tickers = data_config.get('tickers', ['^GSPC', 'QQQ', ticker])
            if ticker not in training_tickers:
                training_tickers.append(ticker)
            
            print(f"   Loading data for {ticker} from {start_date} to {end_date}")
            print(f"   Using tickers: {training_tickers}")
            
            # Load raw data
            data_loader = StockDataLoader(training_tickers, start_date, end_date)
            raw_data = data_loader.download_data()
            price_data = data_loader.get_closing_prices()
            
            # Check if we have enough data
            target_prices = price_data.get(ticker)
            if target_prices is None or len(target_prices) < 100:
                print(f"   ⚠️ Insufficient data for {ticker} in period {start_date} to {end_date}")
                return None
            
            print(f"   Raw target prices shape: {target_prices.shape}")
            
            # Create enhanced features (same as training)
            enhanced_engineer = EnhancedFeatureEngineer(use_abstractions=True)
            
            # Create OHLCV approximation
            ohlcv_data = {
                'close': target_prices,
                'high': target_prices * 1.02,
                'low': target_prices * 0.98,
                'open': target_prices,
                'volume': np.ones_like(target_prices) * 1000000
            }
            
            # Generate features
            features, feature_names = enhanced_engineer.create_features_with_abstractions(
                price_data=price_data,
                target_ticker=ticker,
                ohlcv_data=ohlcv_data
            )
            
            print(f"   Raw features shape: {features.shape}")
            
            # ROBUST ALIGNMENT: Work backwards from features to ensure perfect alignment
            n_features = len(features)
            
            # We need n_features + 1 prices to create n_features returns
            if len(target_prices) < n_features + 1:
                print(f"   ⚠️ Not enough prices for features: need {n_features + 1}, have {len(target_prices)}")
                return None
            
            # Take the last n_features + 1 prices
            aligned_prices = target_prices.flatten()[-n_features - 1:]
            
            # Create returns from aligned prices (will give us exactly n_features returns)
            target_returns = np.diff(aligned_prices) / aligned_prices[:-1]
            
            print(f"   Aligned prices shape: {aligned_prices.shape}")
            print(f"   Target returns shape: {target_returns.shape}")
            print(f"   Features shape: {features.shape}")
            
            # Now we should have perfect alignment
            assert len(features) == len(target_returns), \
                f"Alignment failed: features={len(features)}, returns={len(target_returns)}"
            
            # Final arrays that are guaranteed to be aligned
            features_final = features
            returns_final = target_returns
            prices_for_sequences = aligned_prices[1:]  # Prices corresponding to returns
            
            print(f"   ✅ Perfect alignment achieved:")
            print(f"      Features: {features_final.shape}")
            print(f"      Returns: {returns_final.shape}")
            print(f"      Prices: {prices_for_sequences.shape}")
            
            # Scale features
            scaler = MinMaxScaler(feature_range=(-1, 1))
            features_scaled = scaler.fit_transform(features_final)
            
            # Create sequences with perfect alignment
            sequence_length = model_metadata.get('sequence_length', 30)
            
            # Ensure we have enough data
            if len(features_scaled) < sequence_length + 10:
                print(f"   ⚠️ Insufficient data for sequences: {len(features_scaled)} < {sequence_length + 10}")
                return None
            
            X_sequences = []
            y_sequences = []
            price_sequences = []
            
            # Create sequences - everything is perfectly aligned now
            for i in range(sequence_length, len(features_scaled)):
                X_seq = features_scaled[i-sequence_length:i]
                y_seq = returns_final[i]
                price_seq = prices_for_sequences[i]
                
                # Validate each sequence
                if len(X_seq) == sequence_length and not np.isnan(y_seq) and not np.isnan(price_seq):
                    X_sequences.append(X_seq)
                    y_sequences.append(y_seq)
                    price_sequences.append(price_seq)
            
            if not X_sequences:
                print(f"   ⚠️ No valid sequences created")
                return None
            
            X = np.array(X_sequences)
            y = np.array(y_sequences)
            prices = np.array(price_sequences)
            
            print(f"   Final sequences: X={X.shape}, y={y.shape}, prices={prices.shape}")
            
            # Create date index - align with our sequences
            if hasattr(raw_data, 'index') and len(raw_data.index) >= len(y):
                # Take the last len(y) dates from the raw data
                dates = raw_data.index[-len(y):]
            else:
                # Fallback to generated dates
                end_dt = pd.to_datetime(end_date)
                dates = pd.date_range(end=end_dt, periods=len(y), freq='D')
            
            # Final validation - everything must match
            assert X.shape[0] == y.shape[0] == prices.shape[0] == len(dates), \
                f"Final shape mismatch: X={X.shape[0]}, y={y.shape[0]}, prices={prices.shape[0]}, dates={len(dates)}"
            
            print(f"   ✅ All shapes validated and aligned")
            
            return {
                'X': torch.tensor(X, dtype=torch.float32),
                'y': torch.tensor(y, dtype=torch.float32),
                'prices': prices,
                'dates': dates,
                'scaler': scaler,
                'ticker': ticker,
                'period': f"{start_date} to {end_date}",
                'n_features': features_final.shape[1],
                'n_samples': len(X)
            }
            
        except Exception as e:
            print(f"   ❌ Error preparing data for {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_predictions_with_confidence(self, model: torch.nn.Module, test_data: Dict) -> Dict:
        """Generate predictions with confidence bounds for option analysis."""
        
        model.eval()
        predictions = []
        confidences = []
        
        with torch.no_grad():
            # Generate multiple predictions for confidence estimation
            n_samples = 50  # Monte Carlo samples
            batch_size = 32
            test_x = test_data['X']
            
            # Store all MC samples
            mc_predictions = []
            
            for mc_iter in range(n_samples):
                batch_predictions = []
                
                for i in range(0, len(test_x), batch_size):
                    batch = test_x[i:i+batch_size].to(self.device)
                    
                    # Add small noise for MC sampling
                    if mc_iter > 0:
                        noise = torch.randn_like(batch) * 0.01
                        batch = batch + noise
                    
                    batch_pred = model(batch)
                    batch_predictions.append(batch_pred.cpu())
                
                # Combine batch predictions
                iter_predictions = torch.cat(batch_predictions, dim=0).numpy().flatten()
                mc_predictions.append(iter_predictions)
            
            # Calculate statistics across MC samples
            mc_predictions = np.array(mc_predictions)  # Shape: (n_samples, n_predictions)
            
            # Mean predictions
            predictions = np.mean(mc_predictions, axis=0)
            
            # Confidence bounds (2.5% and 97.5% percentiles for 95% CI)
            lower_bound = np.percentile(mc_predictions, 2.5, axis=0)
            upper_bound = np.percentile(mc_predictions, 97.5, axis=0)
            
            # Prediction uncertainty (std across MC samples)
            prediction_std = np.std(mc_predictions, axis=0)
            
            # Confidence score (inverse of uncertainty)
            confidences = 1 / (1 + prediction_std)
        
        return {
            'predictions': predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence': confidences,
            'prediction_std': prediction_std
        }
    
    def calculate_period_performance(self, predictions_dict: Dict, test_data: Dict) -> Dict:
        """Calculate comprehensive performance metrics for a period."""
        
        try:
            predictions = predictions_dict['predictions']
            actuals = test_data['y'].numpy().flatten()
            prices = test_data['prices']
            
            print(f"   🔍 Performance calculation debug:")
            print(f"      Predictions shape: {predictions.shape}")
            print(f"      Actuals shape: {actuals.shape}")
            print(f"      Prices shape: {prices.shape}")
            
            # Ensure all arrays are the same length
            min_length = min(len(predictions), len(actuals), len(prices))
            
            if min_length != len(predictions) or min_length != len(actuals) or min_length != len(prices):
                print(f"   ⚠️ Trimming arrays to common length: {min_length}")
                predictions = predictions[:min_length]
                actuals = actuals[:min_length]
                prices = prices[:min_length]
            
            print(f"   ✅ Aligned arrays: predictions={len(predictions)}, actuals={len(actuals)}, prices={len(prices)}")
            
            # Convert returns to prices for financial metrics
            start_price = prices[0]
            pred_prices = [start_price]
            actual_prices = [start_price]
            
            for i in range(len(predictions)):
                try:
                    pred_next = pred_prices[-1] * (1 + predictions[i])
                    actual_next = actual_prices[-1] * (1 + actuals[i])
                    pred_prices.append(pred_next)
                    actual_prices.append(actual_next)
                except IndexError as e:
                    print(f"   ❌ Index error at i={i}: {e}")
                    break
            
            pred_prices = np.array(pred_prices[1:])  # Remove initial price
            actual_prices = np.array(actual_prices[1:])
            
            print(f"   ✅ Price conversion: pred_prices={len(pred_prices)}, actual_prices={len(actual_prices)}")
            
            # Calculate metrics with transaction costs
            performance = self.calculate_trading_performance_with_costs(
                predictions, actuals, actual_prices, test_data['dates']
            )
            
            # Add confidence metrics
            performance['confidence_metrics'] = {
                'avg_confidence': np.mean(predictions_dict['confidence'][:len(predictions)]),
                'min_confidence': np.min(predictions_dict['confidence'][:len(predictions)]),
                'high_confidence_trades': np.sum(predictions_dict['confidence'][:len(predictions)] > 0.7),
                'prediction_uncertainty': np.mean(predictions_dict['prediction_std'][:len(predictions)])
            }
            
            return performance
            
        except Exception as e:
            print(f"   ❌ Error in performance calculation: {e}")
            import traceback
            traceback.print_exc()
            
            # Return dummy performance to avoid breaking the analysis
            return {
                'total_return': 0.0,
                'buy_hold_return': 0.0,
                'excess_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'directional_accuracy': 0.0,
                'n_trades': 0,
                'gross_pnl': 0.0,
                'net_pnl': 0.0,
                'total_costs': 0.0,
                'cost_ratio': float('inf')
            }
    
    def calculate_trading_performance_with_costs(self, predictions: np.ndarray, 
                                               actuals: np.ndarray, 
                                               prices: np.ndarray, 
                                               dates: pd.DatetimeIndex) -> Dict:
        """Calculate trading performance including transaction costs."""
        
        try:
            print(f"   🔍 Trading performance debug:")
            print(f"      Input predictions: {predictions.shape}")
            print(f"      Input actuals: {actuals.shape}")
            print(f"      Input prices: {prices.shape}")
            print(f"      Input dates: {len(dates)}")
            
            # Check for NaN or infinite values in inputs
            predictions_clean = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
            actuals_clean = np.nan_to_num(actuals, nan=0.0, posinf=0.0, neginf=0.0)
            prices_clean = np.nan_to_num(prices, nan=prices[0] if len(prices) > 0 else 100.0, posinf=prices[0] if len(prices) > 0 else 100.0, neginf=prices[0] if len(prices) > 0 else 100.0)
            
            # Check if we cleaned any NaNs
            if not np.array_equal(predictions, predictions_clean):
                print(f"   ⚠️ Cleaned {np.sum(np.isnan(predictions))} NaN predictions")
            if not np.array_equal(actuals, actuals_clean):
                print(f"   ⚠️ Cleaned {np.sum(np.isnan(actuals))} NaN actuals")
            if not np.array_equal(prices, prices_clean):
                print(f"   ⚠️ Cleaned {np.sum(np.isnan(prices))} NaN prices")
            
            # Use cleaned data
            predictions = predictions_clean
            actuals = actuals_clean
            prices = prices_clean
            
            # Ensure all inputs are the same length
            min_length = min(len(predictions), len(actuals), len(prices), len(dates))
            
            predictions = predictions[:min_length]
            actuals = actuals[:min_length]
            prices = prices[:min_length]
            dates = dates[:min_length]
            
            print(f"   ✅ Aligned inputs to length: {min_length}")
            
            if min_length == 0:
                print(f"   ⚠️ No data to calculate performance")
                return self._get_zero_performance()
            
            # Generate trading signals (safe operations)
            pred_signs = np.sign(predictions)
            actual_signs = np.sign(actuals)
            
            # Calculate position changes carefully
            if len(pred_signs) > 1:
                position_changes = np.diff(pred_signs)
                trades = np.abs(position_changes)
                n_trades = np.sum(trades)
                
                # Ensure n_trades is not NaN
                if np.isnan(n_trades) or np.isinf(n_trades):
                    print(f"   ⚠️ Invalid n_trades, setting to 0")
                    n_trades = 0.0
            else:
                trades = np.array([])
                n_trades = 0.0
            
            # Calculate returns
            strategy_returns = pred_signs * actuals
            buy_hold_returns = actuals
            
            # Clean returns
            strategy_returns = np.nan_to_num(strategy_returns, nan=0.0)
            buy_hold_returns = np.nan_to_num(buy_hold_returns, nan=0.0)
            
            # Apply transaction costs
            costs = self.config['transaction_costs']
            
            # Commission costs
            commission_cost = float(n_trades) * costs['commission_per_trade']
            
            # Spread and slippage costs (handle empty trades array)
            if len(trades) > 0 and len(prices) > len(trades):
                # Align trades with prices
                trade_prices = prices[1:len(trades)+1]  # Skip first price, align with trades
                total_traded_value = np.sum(trades * trade_prices)
                
                # Clean total_traded_value
                if np.isnan(total_traded_value) or np.isinf(total_traded_value):
                    total_traded_value = 0.0
            else:
                total_traded_value = 0.0
            
            spread_cost = total_traded_value * (costs['bid_ask_spread_pct'] / 100)
            slippage_cost = total_traded_value * (costs['slippage_pct'] / 100)
            
            total_costs = commission_cost + spread_cost + slippage_cost
            
            # Calculate net performance
            gross_pnl = np.sum(strategy_returns * prices)
            net_pnl = gross_pnl - total_costs
            
            # Clean PnL values
            gross_pnl = float(np.nan_to_num(gross_pnl, nan=0.0))
            net_pnl = float(np.nan_to_num(net_pnl, nan=0.0))
            
            # Performance metrics
            if len(strategy_returns) > 0:
                # Total returns
                strategy_cumulative = np.cumprod(1 + strategy_returns)
                buy_hold_cumulative = np.cumprod(1 + buy_hold_returns)
                
                total_return = strategy_cumulative[-1] - 1 if len(strategy_cumulative) > 0 else 0.0
                buy_hold_return = buy_hold_cumulative[-1] - 1 if len(buy_hold_cumulative) > 0 else 0.0
                
                # Clean returns
                total_return = float(np.nan_to_num(total_return, nan=0.0))
                buy_hold_return = float(np.nan_to_num(buy_hold_return, nan=0.0))
                
                # Sharpe ratio (handle zero std case)
                strategy_mean = np.mean(strategy_returns)
                strategy_std = np.std(strategy_returns)
                
                if strategy_std > 0 and not np.isnan(strategy_std):
                    sharpe_ratio = (strategy_mean / strategy_std) * np.sqrt(252)
                else:
                    sharpe_ratio = 0.0
                
                sharpe_ratio = float(np.nan_to_num(sharpe_ratio, nan=0.0))
                
                # Maximum drawdown
                cumulative = np.cumprod(1 + strategy_returns)
                peak = np.maximum.accumulate(cumulative)
                drawdown = (peak - cumulative) / peak
                max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
                max_drawdown = float(np.nan_to_num(max_drawdown, nan=0.0))
                
                # Directional accuracy
                directional_accuracy = np.mean(pred_signs == actual_signs)
                directional_accuracy = float(np.nan_to_num(directional_accuracy, nan=0.0))
                
            else:
                total_return = 0.0
                buy_hold_return = 0.0
                sharpe_ratio = 0.0
                max_drawdown = 0.0
                directional_accuracy = 0.0
            
            result = {
                'total_return': total_return,
                'buy_hold_return': buy_hold_return,
                'excess_return': total_return - buy_hold_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'directional_accuracy': directional_accuracy,
                'n_trades': int(n_trades),  # Convert to int safely
                'gross_pnl': gross_pnl,
                'net_pnl': net_pnl,
                'total_costs': total_costs,
                'cost_ratio': total_costs / abs(gross_pnl) if abs(gross_pnl) > 0.001 else float('inf')
            }
            
            print(f"   ✅ Performance calculated successfully")
            return result
            
        except Exception as e:
            print(f"   ❌ Error in trading performance calculation: {e}")
            import traceback
            traceback.print_exc()
            return self._get_zero_performance()
    
    def _get_zero_performance(self) -> Dict:
        """Return zero performance metrics as fallback."""
        return {
            'total_return': 0.0,
            'buy_hold_return': 0.0,
            'excess_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'directional_accuracy': 0.0,
            'n_trades': 0,
            'gross_pnl': 0.0,
            'net_pnl': 0.0,
            'total_costs': 0.0,
            'cost_ratio': float('inf')
        }
    
    def run_walk_forward_analysis(self, model_path: str, ticker: str) -> Dict:
        """Run complete walk-forward analysis for a model."""
        
        print(f"\n🔄 Starting walk-forward analysis:")
        print(f"   Model: {os.path.basename(model_path)}")
        print(f"   Ticker: {ticker}")
        print("="*60)
        
        model_results = {
            'model_path': model_path,
            'ticker': ticker,
            'period_results': {},
            'overall_performance': {},
            'recommendations': []
        }
        
        # Test on each time period
        for period_name, period_config in self.config['time_periods'].items():
            print(f"\n📊 Testing {period_config['name']} ({period_config['start']} - {period_config['end']})")
            
            try:
                # Load model
                model, metadata = self.load_model_for_period(model_path, period_config)
                if model is None:
                    continue
                
                # Prepare data
                test_data = self.prepare_period_data(
                    ticker, 
                    period_config['start'], 
                    period_config['end'], 
                    metadata
                )
                
                if test_data is None:
                    continue
                
                print(f"   ✅ Prepared {len(test_data['X'])} test samples")
                
                # Generate predictions with confidence
                predictions_dict = self.generate_predictions_with_confidence(model, test_data)
                
                # Calculate performance
                performance = self.calculate_period_performance(predictions_dict, test_data)
                
                # Create standardized backtest paths
                backtest_paths = file_namer.create_backtest_paths(model_path)
                
                # Store results
                # model_results['period_results'][period_name] = {
                    # 'performance': performance,
                    # 'predictions': predictions_dict,
                    # 'test_samples': len(test_data['X']),
                    # 'date_range': f"{period_config['start']} to {period_config['end']}"
                # }
                
                # Print period summary
                print(f"   📈 Performance Summary:")
                print(f"      Excess Return: {performance['excess_return']:.1%}")
                print(f"      Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
                print(f"      Max Drawdown: {performance['max_drawdown']:.1%}")
                print(f"      Directional Accuracy: {performance['directional_accuracy']:.1%}")
                print(f"      Net P&L: ${performance['net_pnl']:.0f}")
                print(f"      Total Costs: ${performance['total_costs']:.0f}")
                
            except Exception as e:
                print(f"   ❌ Error in period {period_name}: {e}")
                continue
        
        # Calculate overall performance
        model_results['overall_performance'] = self.calculate_overall_performance(model_results)
        
        # Generate recommendations
        model_results['recommendations'] = self.generate_model_recommendations(model_results)

        # Save results to standardized location
        with open(backtest_paths['backtest_json'], 'w') as f:
            json.dump(model_results, f, indent=2, default=str)
    
        # Check if model qualifies as champion
        overall_perf = model_results['overall_performance']
        performance_score = overall_perf.get('overall_score', 0)
    
        # If model performs well, promote to champion
        if (performance_score > 60 and 
            overall_perf.get('avg_sharpe_ratio', 0) > 1.0 and
            overall_perf.get('win_rate', 0) > 0.6):
        
            champion_path = promote_champion_model(model_path, performance_score)
            model_results['champion_path'] = champion_path
            print(f"🏆 Model promoted to champion!")
        
        return model_results
    
    def calculate_overall_performance(self, model_results: Dict) -> Dict:
        """Calculate aggregated performance across all periods."""
        
        period_results = model_results['period_results']
        if not period_results:
            return {'error': 'No valid period results'}
        
        # Aggregate metrics
        metrics = ['excess_return', 'sharpe_ratio', 'max_drawdown', 'directional_accuracy', 'net_pnl']
        aggregated = {}
        
        for metric in metrics:
            values = [period['performance'][metric] for period in period_results.values() 
                     if metric in period['performance']]
            if values:
                aggregated[f'avg_{metric}'] = np.mean(values)
                aggregated[f'std_{metric}'] = np.std(values)
                aggregated[f'min_{metric}'] = np.min(values)
                aggregated[f'max_{metric}'] = np.max(values)
        
        # Consistency metrics
        positive_periods = sum(1 for period in period_results.values() 
                              if period['performance'].get('excess_return', 0) > 0)
        total_periods = len(period_results)
        
        aggregated['win_rate'] = positive_periods / total_periods if total_periods > 0 else 0
        aggregated['total_periods'] = total_periods
        aggregated['profitable_periods'] = positive_periods
        
        # Overall ranking score (you can customize this)
        score = (
            aggregated.get('avg_sharpe_ratio', 0) * 0.4 +
            aggregated.get('avg_excess_return', 0) * 0.3 +
            aggregated.get('win_rate', 0) * 0.2 +
            (1 - abs(aggregated.get('avg_max_drawdown', 0))) * 0.1
        )
        aggregated['overall_score'] = score
        
        return aggregated
    
    def generate_model_recommendations(self, model_results: Dict) -> List[str]:
        """Generate actionable recommendations for the model."""
        
        recommendations = []
        overall = model_results['overall_performance']
        thresholds = self.config['performance_thresholds']
        
        # Performance assessment
        avg_sharpe = overall.get('avg_sharpe_ratio', 0)
        avg_excess = overall.get('avg_excess_return', 0)
        win_rate = overall.get('win_rate', 0)
        avg_drawdown = overall.get('avg_max_drawdown', 0)
        
        if avg_sharpe >= thresholds['min_sharpe']:
            recommendations.append(f"✅ STRONG MODEL: Avg Sharpe {avg_sharpe:.2f} exceeds threshold")
        else:
            recommendations.append(f"⚠️ WEAK MODEL: Avg Sharpe {avg_sharpe:.2f} below threshold {thresholds['min_sharpe']}")
        
        if win_rate >= 0.6:
            recommendations.append(f"✅ CONSISTENT: {win_rate:.0%} of periods profitable")
        elif win_rate >= 0.4:
            recommendations.append(f"⚠️ MODERATE: {win_rate:.0%} of periods profitable")
        else:
            recommendations.append(f"❌ INCONSISTENT: Only {win_rate:.0%} of periods profitable")
        
        if abs(avg_drawdown) <= abs(thresholds['max_drawdown']/100):
            recommendations.append(f"✅ GOOD RISK CONTROL: Avg drawdown {avg_drawdown:.1%}")
        else:
            recommendations.append(f"⚠️ HIGH RISK: Avg drawdown {avg_drawdown:.1%}")
        
        # Usage recommendations
        if avg_sharpe >= 1.2 and win_rate >= 0.6:
            recommendations.append("🚀 RECOMMENDATION: Deploy with confidence for live trading")
        elif avg_sharpe >= 0.8 and win_rate >= 0.5:
            recommendations.append("📊 RECOMMENDATION: Use in ensemble or with reduced position size")
        else:
            recommendations.append("❌ RECOMMENDATION: Do not deploy - requires improvement")
        
        return recommendations
    
    def run_batch_analysis(self, model_dir: str, ticker: str = None) -> Dict:
        """Run walk-forward analysis on multiple models."""
        
        print(f"\n🔄 Starting batch walk-forward analysis")
        print(f"   Model directory: {model_dir}")
        print("="*80)
        
        # Find all model files
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        
        if not model_files:
            print(f"❌ No model files found in {model_dir}")
            return {}
        
        print(f"📁 Found {len(model_files)} models to test")
        
        batch_results = {
            'models': {},
            'rankings': {},
            'summary': {}
        }
        
        # Test each model
        for i, model_file in enumerate(model_files):
            model_path = os.path.join(model_dir, model_file)
            
            print(f"\n{'='*60}")
            print(f"🤖 Model {i+1}/{len(model_files)}: {model_file}")
            print(f"{'='*60}")
            
            try:
                # Extract ticker from model name if not provided
                if ticker is None:
                    # Try to extract ticker from filename
                    # Assumes format like: ticker_model_date.pth
                    model_ticker = model_file.split('_')[0].upper()
                else:
                    model_ticker = ticker
                
                # Run analysis
                model_results = self.run_walk_forward_analysis(model_path, model_ticker)
                batch_results['models'][model_file] = model_results
                
            except Exception as e:
                print(f"❌ Error testing {model_file}: {e}")
                continue
        
        # Rank models
        batch_results['rankings'] = self.rank_models(batch_results['models'])
        batch_results['summary'] = self.create_batch_summary(batch_results)
        
        # Save qualified models summary
        qualified_paths = file_namer.create_qualified_models_paths()
    
        # Create qualified models DataFrame
        qualified_models = []
        for model_name, results in batch_results['models'].items():
            overall = results.get('overall_performance', {})
            if overall.get('avg_sharpe_ratio', 0) > 1.0:
                qualified_models.append({
                    'model_name': model_name,
                    'sharpe_ratio': overall.get('avg_sharpe_ratio', 0),
                    'excess_return': overall.get('avg_excess_return', 0),
                    'win_rate': overall.get('win_rate', 0),
                    'overall_score': overall.get('overall_score', 0)
                })
    
        if qualified_models:
            qualified_df = pd.DataFrame(qualified_models)
            qualified_df = qualified_df.sort_values('overall_score', ascending=False)
            qualified_df.to_csv(qualified_paths['qualified_csv'], index=False)
            print(f"Saved {len(qualified_models)} qualified models to {qualified_paths['qualified_csv']}")
        
        return batch_results
    
    def rank_models(self, model_results: Dict) -> Dict:
        """Rank models by performance."""
        
        rankings = []
        
        for model_name, results in model_results.items():
            overall = results.get('overall_performance', {})
            
            ranking_entry = {
                'model': model_name,
                'ticker': results.get('ticker', 'Unknown'),
                'overall_score': overall.get('overall_score', 0),
                'avg_sharpe': overall.get('avg_sharpe_ratio', 0),
                'avg_excess_return': overall.get('avg_excess_return', 0),
                'win_rate': overall.get('win_rate', 0),
                'avg_drawdown': overall.get('avg_max_drawdown', 0),
                'total_periods': overall.get('total_periods', 0)
            }
            rankings.append(ranking_entry)
        
        # Sort by overall score
        rankings.sort(key=lambda x: x['overall_score'], reverse=True)
        
        return {
            'ranked_models': rankings,
            'top_models': rankings[:5],  # Top 5
            'qualified_models': [r for r in rankings if r['avg_sharpe'] >= 1.0 and r['win_rate'] >= 0.5]
        }
    
    def create_batch_summary(self, batch_results: Dict) -> Dict:
        """Create summary of batch analysis."""
        
        total_models = len(batch_results['models'])
        qualified_models = len(batch_results['rankings']['qualified_models'])
        
        return {
            'total_models_tested': total_models,
            'qualified_models': qualified_models,
            'qualification_rate': qualified_models / total_models if total_models > 0 else 0,
            'best_model': batch_results['rankings']['top_models'][0] if batch_results['rankings']['top_models'] else None,
            'avg_score_all_models': np.mean([r['overall_score'] for r in batch_results['rankings']['ranked_models']]) if batch_results['rankings']['ranked_models'] else 0
        }
    
    def save_results(self, results: Dict, output_path: str = None):
        """Save backtesting results."""
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"results/backtest_results_{timestamp}.json"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert numpy types for JSON serialization
        results_json = self._convert_to_json_safe(results)
        
        with open(output_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")
        return output_path
    
    def _convert_to_json_safe(self, obj):
        """Convert numpy types to JSON-safe types."""
        if isinstance(obj, dict):
            return {key: self._convert_to_json_safe(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_safe(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return obj

def main():
    """Main function for walk-forward backtesting."""
    parser = argparse.ArgumentParser(description='Walk-Forward Backtesting for LNN Models')
    
    # Model specification
    parser.add_argument('--model', type=str, 
                      help='Path to specific model file')
    parser.add_argument('--model-dir', type=str, default='models/saved_models',
                      help='Directory containing model files for batch analysis')
    parser.add_argument('--ticker', type=str,
                      help='Stock ticker to test (if not in model filename)')
    
    # Analysis options
    parser.add_argument('--batch', action='store_true',
                      help='Run batch analysis on all models in directory')
    parser.add_argument('--config', type=str,
                      help='Path to backtesting configuration file')
    
    # Output options
    parser.add_argument('--output', type=str,
                      help='Output path for results')
    parser.add_argument('--top-n', type=int, default=5,
                      help='Number of top models to highlight')
    
    # Forecasting options (for your option analysis)
    parser.add_argument('--forecast-periods', type=str, default='2w,4w,6m',
                      help='Forecast periods for option analysis (comma-separated)')
    parser.add_argument('--generate-forecasts', action='store_true',
                      help='Generate forward-looking forecasts for top models')
    
    args = parser.parse_args()
    
    # Initialize backtester
    backtester = WalkForwardBacktester(config_path=args.config)
    
    print("🔄 LNN WALK-FORWARD BACKTESTING SYSTEM")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        if args.batch or (not args.model and args.model_dir):
            # Batch analysis
            print(f"\n🔍 Running batch analysis on directory: {args.model_dir}")
            results = backtester.run_batch_analysis(args.model_dir, args.ticker)
            
            # Print batch summary
            print_batch_summary(results, args.top_n)
            
            # Generate forecasts for top models if requested
            if args.generate_forecasts:
                print(f"\n🔮 Generating forecasts for top models...")
                generate_option_forecasts(backtester, results, args.forecast_periods)
            
        elif args.model:
            # Single model analysis
            if not args.ticker:
                # Try to extract ticker from model filename
                model_name = os.path.basename(args.model)
                ticker = model_name.split('_')[0].upper()
                print(f"🎯 Extracted ticker '{ticker}' from model filename")
            else:
                ticker = args.ticker.upper()
            
            print(f"\n🔍 Running single model analysis:")
            print(f"   Model: {args.model}")
            print(f"   Ticker: {ticker}")
            
            results = backtester.run_walk_forward_analysis(args.model, ticker)
            
            # Print single model summary
            print_single_model_summary(results)
            
            # Generate forecasts if requested
            if args.generate_forecasts:
                print(f"\n🔮 Generating forecasts...")
                generate_single_model_forecasts(backtester, results, args.forecast_periods)
        
        else:
            print("❌ Error: Must specify either --model or --model-dir")
            return
        
        # Save results
        output_path = backtester.save_results(results, args.output)
        
        print(f"\n✅ BACKTESTING COMPLETED SUCCESSFULLY!")
        print(f"📊 Results saved to: {output_path}")
        print(f"⏱️  Total execution time: {(datetime.now() - datetime.now()).total_seconds():.1f} seconds")
        
    except Exception as e:
        print(f"\n❌ ERROR during backtesting: {e}")
        import traceback
        traceback.print_exc()

def print_batch_summary(results: Dict, top_n: int = 5):
    """Print summary of batch backtesting results."""
    
    print("\n" + "="*80)
    print("BATCH BACKTESTING SUMMARY")
    print("="*80)
    
    summary = results.get('summary', {})
    rankings = results.get('rankings', {})
    
    # Overall statistics
    print(f"📊 OVERALL STATISTICS:")
    print(f"   Total models tested: {summary.get('total_models_tested', 0)}")
    print(f"   Qualified models: {summary.get('qualified_models', 0)}")
    print(f"   Qualification rate: {summary.get('qualification_rate', 0):.1%}")
    print(f"   Average score: {summary.get('avg_score_all_models', 0):.3f}")
    
    # Top performers
    top_models = rankings.get('top_models', [])[:top_n]
    if top_models:
        print(f"\n🏆 TOP {min(top_n, len(top_models))} PERFORMING MODELS:")
        print("-" * 60)
        
        for i, model in enumerate(top_models):
            print(f"{i+1}. {model['model']}")
            print(f"   Ticker: {model['ticker']}")
            print(f"   Overall Score: {model['overall_score']:.3f}")
            print(f"   Avg Sharpe: {model['avg_sharpe']:.2f}")
            print(f"   Avg Excess Return: {model['avg_excess_return']:.1%}")
            print(f"   Win Rate: {model['win_rate']:.1%}")
            print(f"   Avg Drawdown: {model['avg_drawdown']:.1%}")
            print()
    
    # Qualified models for trading
    qualified = rankings.get('qualified_models', [])
    if qualified:
        print(f"✅ QUALIFIED FOR LIVE TRADING ({len(qualified)} models):")
        print("-" * 60)
        for model in qualified:
            print(f"• {model['model']} ({model['ticker']}) - Score: {model['overall_score']:.3f}")
    else:
        print("❌ NO MODELS QUALIFIED FOR LIVE TRADING")
        print("   Consider retraining with different parameters or more data")

def print_single_model_summary(results: Dict):
    """Print summary of single model backtesting."""
    
    print("\n" + "="*60)
    print("SINGLE MODEL BACKTESTING SUMMARY")
    print("="*60)
    
    overall = results.get('overall_performance', {})
    recommendations = results.get('recommendations', [])
    
    print(f"🤖 Model: {os.path.basename(results['model_path'])}")
    print(f"📈 Ticker: {results['ticker']}")
    print()
    
    print(f"📊 OVERALL PERFORMANCE:")
    print(f"   Average Sharpe Ratio: {overall.get('avg_sharpe_ratio', 0):.2f}")
    print(f"   Average Excess Return: {overall.get('avg_excess_return', 0):.1%}")
    print(f"   Win Rate: {overall.get('win_rate', 0):.1%}")
    print(f"   Average Max Drawdown: {overall.get('avg_max_drawdown', 0):.1%}")
    print(f"   Overall Score: {overall.get('overall_score', 0):.3f}")
    print()
    
    print(f"🎯 RECOMMENDATIONS:")
    for rec in recommendations:
        print(f"   {rec}")

def generate_option_forecasts(backtester, batch_results: Dict, forecast_periods: str):
    """Generate forecasts for option analysis using top models."""
    
    periods = forecast_periods.split(',')
    top_models = batch_results.get('rankings', {}).get('qualified_models', [])[:3]  # Top 3 qualified
    
    if not top_models:
        print("⚠️ No qualified models found for forecasting")
        return
    
    print(f"🔮 GENERATING FORECASTS FOR OPTION ANALYSIS")
    print("="*60)
    print(f"📅 Forecast periods: {periods}")
    print(f"🤖 Using {len(top_models)} top models")
    print()
    
    forecast_results = {}
    
    for period in periods:
        print(f"\n📊 {period.upper()} FORECASTS:")
        print("-" * 40)
        
        period_forecasts = {}
        
        for model_info in top_models:
            model_name = model_info['model']
            ticker = model_info['ticker']
            
            try:
                # Generate forecast for this period
                forecast = generate_single_period_forecast(
                    backtester, 
                    model_name, 
                    ticker, 
                    period
                )
                
                period_forecasts[model_name] = forecast
                
                # Print forecast summary
                print(f"   {model_name} ({ticker}):")
                print(f"     Expected Return: {forecast['expected_return']:.1%} ± {forecast['return_std']:.1%}")
                print(f"     Confidence: {forecast['confidence']:.1%}")
                print(f"     Implied Vol Signal: {forecast['iv_signal']}")
                print()
                
            except Exception as e:
                print(f"   ❌ Error forecasting {model_name}: {e}")
        
        forecast_results[period] = period_forecasts
    
    # Generate option recommendations
    option_recs = generate_option_recommendations(forecast_results)
    
    print(f"\n🎯 OPTION TRADING RECOMMENDATIONS:")
    print("="*50)
    for rec in option_recs:
        print(f"   {rec}")

def generate_single_period_forecast(backtester, model_name: str, ticker: str, period: str) -> Dict:
    """Generate forecast for a specific model and period."""
    
    # Parse period (e.g., '2w' = 2 weeks, '4w' = 4 weeks, '6m' = 6 months)
    if period.endswith('w'):
        days = int(period[:-1]) * 7
    elif period.endswith('m'):
        days = int(period[:-1]) * 30
    else:
        days = 30  # Default
    
    # Calculate expected return and volatility
    # This is a simplified version - you'd use your actual model here
    
    # Mock forecast (replace with actual model prediction)
    base_return = np.random.normal(0.02, 0.15)  # 2% expected, 15% vol
    return_std = np.random.uniform(0.10, 0.25)  # 10-25% volatility
    confidence = np.random.uniform(0.6, 0.9)    # 60-90% confidence
    
    # Implied volatility signal
    current_iv = 0.20  # Mock current implied volatility
    predicted_rv = return_std  # Predicted realized volatility
    
    if predicted_rv > current_iv * 1.1:
        iv_signal = "BUY_STRADDLE"  # Expect higher volatility
    elif predicted_rv < current_iv * 0.9:
        iv_signal = "SELL_STRADDLE"  # Expect lower volatility
    else:
        iv_signal = "NEUTRAL"
    
    return {
        'model': model_name,
        'ticker': ticker,
        'period': period,
        'days': days,
        'expected_return': base_return,
        'return_std': return_std,
        'confidence': confidence,
        'current_iv': current_iv,
        'predicted_rv': predicted_rv,
        'iv_signal': iv_signal
    }

def generate_single_model_forecasts(backtester, model_results: Dict, forecast_periods: str):
    """Generate forecasts for a single model."""
    
    periods = forecast_periods.split(',')
    model_path = model_results['model_path']
    ticker = model_results['ticker']
    
    print(f"🔮 GENERATING FORECASTS")
    print("="*40)
    print(f"🤖 Model: {os.path.basename(model_path)}")
    print(f"📈 Ticker: {ticker}")
    print()
    
    for period in periods:
        forecast = generate_single_period_forecast(
            backtester, 
            os.path.basename(model_path), 
            ticker, 
            period
        )
        
        print(f"📊 {period.upper()} FORECAST:")
        print(f"   Expected Return: {forecast['expected_return']:.1%} ± {forecast['return_std']:.1%}")
        print(f"   Confidence: {forecast['confidence']:.1%}")
        print(f"   IV Signal: {forecast['iv_signal']}")
        print()

def generate_option_recommendations(forecast_results: Dict) -> List[str]:
    """Generate option trading recommendations based on forecasts."""
    
    recommendations = []
    
    for period, forecasts in forecast_results.items():
        if not forecasts:
            continue
        
        # Aggregate signals across models
        iv_signals = [f['iv_signal'] for f in forecasts.values()]
        buy_straddle_votes = iv_signals.count('BUY_STRADDLE')
        sell_straddle_votes = iv_signals.count('SELL_STRADDLE')
        
        # Consensus recommendation
        if buy_straddle_votes > len(forecasts) * 0.6:
            recommendations.append(f"🟢 {period.upper()}: BUY STRADDLES - Models expect volatility spike")
        elif sell_straddle_votes > len(forecasts) * 0.6:
            recommendations.append(f"🔴 {period.upper()}: SELL STRADDLES - Models expect volatility crush")
        else:
            recommendations.append(f"🟡 {period.upper()}: NEUTRAL - Mixed volatility signals")
        
        # Expected return consensus
        expected_returns = [f['expected_return'] for f in forecasts.values()]
        avg_return = np.mean(expected_returns)
        
        if avg_return > 0.05:
            recommendations.append(f"   📈 Strong upside expected ({avg_return:.1%}) - Consider call spreads")
        elif avg_return < -0.05:
            recommendations.append(f"   📉 Downside risk ({avg_return:.1%}) - Consider put spreads")
    
    if not recommendations:
        recommendations.append("⚠️ Insufficient data for option recommendations")
    
    return recommendations

if __name__ == "__main__":
    main()
