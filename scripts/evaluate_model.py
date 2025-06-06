#!/usr/bin/env python3
"""
Evaluation script for trained Liquid Neural Network models.
Performs comprehensive evaluation including financial metrics, pattern analysis, and visualization.

Usage:
    python scripts/evaluate_model.py --model models/saved_models/best_lnn_model_20241223_143022.pth
    python scripts/evaluate_model.py --model models/saved_models/best_lnn_model_20241223_143022.pth --detailed
    python scripts/evaluate_model.py --compare-experiments exp1,exp2,exp3
"""

import os
import sys
import argparse
import torch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from models.trading_strategy import EnhancedTradingStrategy, TradingSignal
    ENHANCED_STRATEGY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Enhanced trading strategy not available: {e}")
    ENHANCED_STRATEGY_AVAILABLE = False

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import our modules
# NEW (working):
from data.data_loader import StockDataLoader
from data.preprocessor import StockDataPreprocessor, prepare_model_data
from models.lnn_model import LiquidNetwork
from analysis.pattern_recognition import PatternRecognizer
from analysis.feature_engineering import AdvancedFeatureEngineer
from analysis.dimensionality_reduction import DimensionalityReducer
from analysis.temporal_analysis import TemporalAnalyzer
from utils.metrics import StockPredictionMetrics
from utils.experiment_tracker import ExperimentTracker
from models.trading_strategy import EnhancedTradingStrategy, TradingSignal
from utils.file_naming import file_namer, create_evaluation_paths, parse_model_info

try:
    from models.trading_strategy import EnhancedTradingStrategy, TradingSignal
    ENHANCED_STRATEGY_AVAILABLE = True
    print("✓ Enhanced trading strategy available")
except ImportError as e:
    print(f"⚠️ Enhanced trading strategy not available: {e}")
    ENHANCED_STRATEGY_AVAILABLE = False

class ModelEvaluator:
    """
    Comprehensive model evaluation class.
    Loads trained models and performs detailed analysis including:
    - Financial performance metrics
    - Pattern recognition analysis
    - Feature importance analysis
    - Temporal analysis
    - Visualization and reporting
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to saved model file
        """
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.model = None
        self.config = None
        self.preprocessor = None
        self.metrics_calculator = StockPredictionMetrics()
        self.experiment_tracker = ExperimentTracker()
        
        # Data storage
        self.test_data = None
        self.predictions = None
        self.actuals = None
        self.evaluation_results = {}
        
        # Analysis components
        self.pattern_recognizer = PatternRecognizer()
        self.temporal_analyzer = TemporalAnalyzer()
        self.dim_reducer = DimensionalityReducer()
        
        if model_path:
            # Create standardized evaluation paths
            self.eval_paths = create_evaluation_paths(model_path)
            self.model_info = parse_model_info(os.path)
            
            print(f"Evaluation paths created:")
            print(f"  Results: {self.eval_paths['evaluation_json']}")
            print(f"  Trading: {self.eval_paths['trading_performance']}")
        
    def load_model_and_config(self):
        """Load the trained model and its configuration."""
        print("="*50)
        print("LOADING MODEL AND CONFIGURATION")
        print("="*50)
    
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
    
        # Load model checkpoint FIRST
        print(f"Loading model from: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.checkpoint = checkpoint  # Store checkpoint BEFORE calling prepare_test_data
    
        # Extract configuration
        self.config = checkpoint.get('config', {})
        print(f"Model trained for {checkpoint.get('epoch', 'unknown')} epochs")
        print(f"Best validation loss: {checkpoint.get('val_loss', 'unknown'):.6f}")
    
        # NOW prepare test data (checkpoint is available)
        self.prepare_test_data()
    
        # Get the correct input size from the saved model
        saved_input_weights = checkpoint['model_state_dict']['liquid_cell.input_weights']
        input_size = saved_input_weights.shape[0]
        output_size = self.test_data['y'].shape[1] if len(self.test_data['y'].shape) > 1 else 1
        hidden_size = self.config.get('model', {}).get('hidden_size', 50)
    
        print(f"Model was trained with {input_size} input features")
    
        # Initialize model with correct architecture
        self.model = LiquidNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size
        ).to(self.device)
    
        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
        print(f"Model loaded successfully!")
        print(f"Architecture: {input_size} → {hidden_size} → {output_size}")
        
    def prepare_enhanced_test_data(self):
        """Prepare test data using enhanced features (same as training)."""
        print("Preparing enhanced test data to match training...")
    
        # Load data using same configuration as training
        data_config = self.config.get('data', {})
        tickers = data_config.get('tickers', ['^GSPC', 'AGG', 'QQQ', 'AAPL'])
        start_date = data_config.get('start_date', '2020-01-01')
        end_date = data_config.get('end_date', '2024-12-31')
        target_ticker = data_config.get('target_ticker', 'AAPL')
    
        # Load raw data
        data_loader = StockDataLoader(tickers, start_date, end_date)
        raw_data = data_loader.download_data()
        price_data = data_loader.get_closing_prices()
    
        # Check if enhanced features were used in training
        use_enhanced = self.config.get('analysis', {}).get('use_advanced_features', False)
        use_abstractions = self.config.get('analysis', {}).get('use_abstractions', False)
    
        if use_enhanced or use_abstractions:
            print("🧠 Recreating enhanced features for evaluation...")
        
            # Import and use the same enhanced feature engineer
            try:
                from analysis.market_abstraction_pipeline import EnhancedFeatureEngineer
            
                enhanced_engineer = EnhancedFeatureEngineer(use_abstractions=use_abstractions)
            
                # Create OHLCV approximation (same as training)
                target_prices = price_data[target_ticker]
                ohlcv_data = {
                    'close': target_prices,
                    'high': target_prices * 1.02,
                    'low': target_prices * 0.98,
                    'open': target_prices,
                    'volume': np.ones_like(target_prices) * 1000000
                }
            
                # Generate enhanced features (same as training)
                features, feature_names = enhanced_engineer.create_features_with_abstractions(
                    price_data=price_data,
                    target_ticker=target_ticker,
                    ohlcv_data=ohlcv_data
                )
            
                print(f"✅ Enhanced features recreated: {features.shape}")
            
                # Create target
                target_returns = np.diff(target_prices.flatten()) / target_prices.flatten()[:-1]
            
                # Align features with returns
                min_length = min(len(features), len(target_returns))
                features_aligned = features[-min_length:]
                target_aligned = target_returns[-min_length:]
            
                # Scale features using saved scaler parameters
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler(feature_range=(-1, 1))
            
                # Restore scaler state from checkpoint if available
                if 'scaler_params' in self.checkpoint:
                    scaler_params = self.checkpoint['scaler_params']
                    scaler.data_min_ = scaler_params['data_min_']
                    scaler.data_max_ = scaler_params['data_max_']
                    scaler.data_range_ = scaler_params['data_range_']
                    scaler.scale_ = scaler_params['scale_']
                    scaler.min_ = scaler_params['min_']
                    features_scaled = scaler.transform(features_aligned)
                    print("✅ Using saved scaler parameters")
                else:
                    features_scaled = scaler.fit_transform(features_aligned)
                    print("⚠️ Scaler parameters not found, fitting new scaler")
            
                # Create sequences
                sequence_length = self.config.get('model', {}).get('sequence_length', 30)
                X_sequences = []
                y_sequences = []
            
                for i in range(sequence_length, len(features_scaled)):
                    X_seq = features_scaled[i-sequence_length:i]
                    y_seq = target_aligned[i]
                    X_sequences.append(X_seq)
                    y_sequences.append(y_seq)
            
                X = np.array(X_sequences)
                y = np.array(y_sequences)
            
                # Train/test split (use same ratio as training)
                train_size = int(0.8 * len(X))
                val_size = int(0.1 * len(X))
            
                X_test = X[train_size+val_size:]
                y_test = y[train_size+val_size:]
            
                print(f"✅ Enhanced test data prepared: {X_test.shape}")
            
                return {
                    'X': torch.tensor(X_test, dtype=torch.float32),
                    'y': torch.tensor(y_test, dtype=torch.float32),
                    'raw_prices': target_prices[-len(y_test):] if len(y_test) > 0 else target_prices,
                    'dates': raw_data.index[-len(y_test):] if hasattr(raw_data, 'index') and len(y_test) > 0 else None
                }
            
            except ImportError as e:
                print(f"⚠️ Enhanced features not available: {e}")
                print("   Falling back to basic features...")
                return self.prepare_basic_test_data()
    
        else:
            print("📊 Using basic features (enhanced features disabled)")
            return self.prepare_basic_test_data()
        
    def prepare_test_data(self):
        """Prepare test data using the same preprocessing as training."""
        print("Preparing test data...")
    
        # Load data using same configuration as training
        data_config = self.config.get('data', {})
        tickers = data_config.get('tickers', ['^GSPC', 'AGG', 'QQQ', 'AAPL'])
        start_date = data_config.get('start_date', '2020-01-01')
        end_date = data_config.get('end_date', '2024-12-31')
        target_ticker = data_config.get('target_ticker', 'AAPL')
    
        # Check what kind of features were used in training
        use_enhanced = self.config.get('analysis', {}).get('use_advanced_features', False)
        use_abstractions = self.config.get('analysis', {}).get('use_abstractions', False)
    
        if use_enhanced or use_abstractions:
            print("🧠 Recreating REAL enhanced features for evaluation...")
        
            try:
                # Import the enhanced feature engineer
                from analysis.market_abstraction_pipeline import EnhancedFeatureEngineer
            
                # Load the raw data
                data_loader = StockDataLoader(tickers, start_date, end_date)
                raw_data = data_loader.download_data()
                price_data = data_loader.get_closing_prices()
            
                # Create enhanced features (EXACTLY like training)
                enhanced_engineer = EnhancedFeatureEngineer(use_abstractions=use_abstractions)
            
                # Create OHLCV approximation (same as training)
                target_prices = price_data[target_ticker]
                ohlcv_data = {
                    'close': target_prices,
                    'high': target_prices * 1.02,
                    'low': target_prices * 0.98,
                    'open': target_prices,
                    'volume': np.ones_like(target_prices) * 1000000
                }
            
                # Generate enhanced features (same as training)
                features, feature_names = enhanced_engineer.create_features_with_abstractions(
                    price_data=price_data,
                    target_ticker=target_ticker,
                    ohlcv_data=ohlcv_data
                )
            
                print(f"✅ REAL enhanced features recreated: {features.shape}")
            
                # Create target returns
                target_returns = np.diff(target_prices.flatten()) / target_prices.flatten()[:-1]
            
                # Align features with returns
                min_length = min(len(features), len(target_returns))
                features_aligned = features[-min_length:]
                target_aligned = target_returns[-min_length:]
            
                # Scale features using saved scaler parameters if available
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler(feature_range=(-1, 1))
            
                if 'scaler_params' in self.checkpoint:
                    print("✅ Using saved scaler parameters from training")
                    scaler_params = self.checkpoint['scaler_params']
                    scaler.data_min_ = scaler_params['data_min_']
                    scaler.data_max_ = scaler_params['data_max_']
                    scaler.data_range_ = scaler_params['data_range_']
                    scaler.scale_ = scaler_params['scale_']
                    scaler.min_ = scaler_params['min_']
                    features_scaled = scaler.transform(features_aligned)
                else:
                    print("⚠️ No saved scaler parameters, fitting new scaler")
                    features_scaled = scaler.fit_transform(features_aligned)
            
                # Store the preprocessor for later use
                self.preprocessor = type('MockPreprocessor', (), {
                    'inverse_transform_single': lambda self, ticker, data: data  # Simple pass-through
                })()
            
                # Create sequences (same as training)
                sequence_length = self.config.get('model', {}).get('sequence_length', 30)
                X_sequences = []
                y_sequences = []
            
                for i in range(sequence_length, len(features_scaled)):
                    X_seq = features_scaled[i-sequence_length:i]
                    y_seq = target_aligned[i]
                    X_sequences.append(X_seq)
                    y_sequences.append(y_seq)
            
                X = np.array(X_sequences)
                y = np.array(y_sequences)
            
                print(f"✅ Sequences created: X={X.shape}, y={y.shape}")
            
                # Train/val/test split (same as training)
                train_size = int(0.8 * len(X))
                val_size = int(0.1 * len(X))
                X_test = X[train_size+val_size:]
                y_test = y[train_size+val_size:]
            
                # Store test data with real features
                self.test_data = {
                    'X': torch.tensor(X_test, dtype=torch.float32),
                    'y': torch.tensor(y_test, dtype=torch.float32),
                    'raw_prices': target_prices.flatten()[-len(y_test):],
                    'dates': raw_data.index[-len(y_test):] if hasattr(raw_data, 'index') else None
                }
            
                print(f"✅ REAL test data prepared: {self.test_data['X'].shape}")
                print(f"   Features: {self.test_data['X'].shape[2]} (should match training)")
            
            except Exception as e:
                print(f"❌ Error creating real enhanced features: {e}")
                print("   Falling back to dummy features...")
                self.prepare_dummy_test_data()
    
        else:
            print("📊 Using basic features...")
            self.prepare_basic_test_data()
            
    def generate_predictions(self):
        """Generate predictions on test data."""
        print("="*50)
        print("GENERATING PREDICTIONS")
        print("="*50)

        self.model.eval()
        all_predictions = []

        with torch.no_grad():
            # Process in batches to handle memory efficiently
            batch_size = 64
            test_x = self.test_data['X']
        
            for i in range(0, len(test_x), batch_size):
                batch = test_x[i:i+batch_size].to(self.device)
                batch_pred = self.model(batch)
                all_predictions.append(batch_pred.cpu())
        
            # Combine all predictions
            self.predictions = torch.cat(all_predictions, dim=0)
            self.actuals = self.test_data['y']

        print(f"Generated {len(self.predictions)} predictions")

        # CORRECTED: Consistent handling of returns vs prices
        target_ticker = self.config.get('data', {}).get('target_ticker', 'AAPL')

        # Your model predicts RETURNS (percentage changes), not absolute prices
        self.predictions_returns = self.predictions.numpy().flatten()
        self.actuals_returns = self.actuals.numpy().flatten()

        print(f"Return predictions range: {self.predictions_returns.min():.4f} to {self.predictions_returns.max():.4f}")
        print(f"Actual returns range: {self.actuals_returns.min():.4f} to {self.actuals_returns.max():.4f}")

        # Convert returns to price series for visualization and trading simulation
        if 'raw_prices' in self.test_data and self.test_data['raw_prices'] is not None:
            raw_prices = self.test_data['raw_prices']
        
            if len(raw_prices) > 0:
                # Use the first test price as starting point
                start_price = raw_prices[0]
            
                # Build predicted price series from returns
                pred_prices = [start_price]
                actual_prices = [start_price]
            
                for i in range(len(self.predictions_returns)):
                    # Predicted next price: P_t+1 = P_t * (1 + return_t+1)
                    pred_next = pred_prices[-1] * (1 + self.predictions_returns[i])
                    pred_prices.append(pred_next)
                
                    # Actual next price using actual returns
                    actual_next = actual_prices[-1] * (1 + self.actuals_returns[i])
                    actual_prices.append(actual_next)
            
                # Store price series (remove initial price since we added it)
                self.predictions_prices = np.array(pred_prices[1:])
                self.actuals_prices = np.array(actual_prices[1:])
              
                # For backward compatibility, store as "unscaled" (these are the price versions)
                self.predictions_unscaled = self.predictions_prices
                self.actuals_unscaled = self.actuals_prices
            
                print(f"✅ Converted returns to prices successfully")
                print(f"Predicted prices range: ${self.predictions_prices.min():.2f} - ${self.predictions_prices.max():.2f}")
                print(f"Actual prices range: ${self.actuals_prices.min():.2f} - ${self.actuals_prices.max():.2f}")
            
            else:
                print("⚠️ Raw prices array is empty, using returns directly")
                self.predictions_unscaled = self.predictions_returns
                self.actuals_unscaled = self.actuals_returns
                self.predictions_prices = None
                self.actuals_prices = None
        else:
            print("⚠️ No raw prices available, using returns directly")
            self.predictions_unscaled = self.predictions_returns
            self.actuals_unscaled = self.actuals_returns
            self.predictions_prices = None
            self.actuals_prices = None
            
    def calculate_comprehensive_metrics(self):
        """Calculate all evaluation metrics."""
        print("="*50)
        print("CALCULATING COMPREHENSIVE METRICS")
        print("="*50)
    
        # CORRECTED: Use appropriate data for different metric types
    
        # For basic regression metrics, use RETURNS (what the model actually predicts)
        # This gives you meaningful R², RMSE, etc. on the prediction task
        basic_metrics_data = {
            'y_true': self.actuals_returns,
            'y_pred': self.predictions_returns
        }
    
        # For financial/trading metrics, use PRICES (what traders care about)
        # This gives you meaningful dollar amounts, trading returns, etc.
        if self.predictions_prices is not None and self.actuals_prices is not None:
            financial_metrics_data = {
                'y_true': self.actuals_prices,
                'y_pred': self.predictions_prices
            }
            print("Using price data for financial metrics")
        else:
            # Fallback to returns if no prices available
            financial_metrics_data = basic_metrics_data
            print("Using return data for financial metrics (no prices available)")
    
        # Calculate metrics using the appropriate data for each type
        self.evaluation_results['metrics'] = self.metrics_calculator.comprehensive_evaluation(
            y_true=financial_metrics_data['y_true'],
            y_pred=financial_metrics_data['y_pred'],
            dates=self.test_data['dates'],
            # Pass return data separately for directional accuracy
            return_predictions=self.predictions_returns,
            actual_returns=self.actuals_returns
        )
    
        # Print key metrics
        metrics = self.evaluation_results['metrics']
        print("KEY PERFORMANCE METRICS:")
        print("-" * 30)
    
        if 'basic_metrics' in metrics:
            basic = metrics['basic_metrics']
        
            # Check if we're using price or return metrics
            if self.predictions_prices is not None:
                print("📊 REGRESSION METRICS (on returns):")
                # Calculate return-based metrics for display
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                rmse_returns = np.sqrt(mean_squared_error(self.actuals_returns, self.predictions_returns))
                mae_returns = mean_absolute_error(self.actuals_returns, self.predictions_returns)
                r2_returns = r2_score(self.actuals_returns, self.predictions_returns)
                mape_returns = np.mean(np.abs((self.actuals_returns - self.predictions_returns) / 
                                        (self.actuals_returns + 1e-8))) * 100
            
                print(f"  RMSE (returns): {rmse_returns:.6f}")
                print(f"  MAE (returns): {mae_returns:.6f}")
                print(f"  MAPE (returns): {mape_returns:.2f}%")
                print(f"  R² (returns): {r2_returns:.4f}")
            
                print("💰 FINANCIAL METRICS (on prices):")
                print(f"  RMSE (price): ${basic.get('rmse', 'N/A'):.2f}")
                print(f"  MAE (price): ${basic.get('mae', 'N/A'):.2f}")
            else:
                print(f"RMSE: {basic.get('rmse', 'N/A'):.6f}")
                print(f"MAE: {basic.get('mae', 'N/A'):.6f}")
                print(f"MAPE: {basic.get('mape', 'N/A'):.2f}%")
                print(f"R²: {basic.get('r2', 'N/A'):.4f}")
    
        if 'directional_metrics' in metrics:
            direction = metrics['directional_metrics']
            print(f"🎯 Directional Accuracy: {direction.get('directional_accuracy', 'N/A'):.1%}")
    
        if 'trading_metrics' in metrics:
            trading = metrics['trading_metrics']
            print(f"📈 TRADING PERFORMANCE:")
            print(f"  Strategy Return: {trading.get('total_return', 'N/A'):.1%}")
            print(f"  Buy & Hold Return: {trading.get('buy_hold_return', 'N/A'):.1%}")
            print(f"  Excess Return: {trading.get('total_return', 0) - trading.get('buy_hold_return', 0):.1%}")
            print(f"  Sharpe Ratio: {trading.get('sharpe_ratio', 'N/A'):.2f}")
            print(f"  Max Drawdown: {trading.get('max_drawdown', 'N/A'):.1%}")
    
    def analyze_patterns(self):
        """Perform pattern recognition analysis on predictions and actuals."""
        print("="*50)
        print("PATTERN RECOGNITION ANALYSIS")
        print("="*50)
        
        # Analyze patterns in actual prices
        actual_patterns = self.pattern_recognizer.get_pattern_summary(self.actuals_unscaled)
        
        # Analyze patterns in predicted prices
        predicted_patterns = self.pattern_recognizer.get_pattern_summary(self.predictions_unscaled)
        
        self.evaluation_results['patterns'] = {
            'actual_patterns': actual_patterns,
            'predicted_patterns': predicted_patterns
        }
        
        # Print pattern summary
        print("PATTERN DETECTION RESULTS:")
        print("-" * 30)
        
        actual_counts = actual_patterns.get('pattern_count', {})
        predicted_counts = predicted_patterns.get('pattern_count', {})
        
        for pattern_type in ['support_levels', 'resistance_levels', 'triangles', 'double_patterns']:
            actual_count = actual_counts.get(pattern_type, 0)
            predicted_count = predicted_counts.get(pattern_type, 0)
            print(f"{pattern_type.replace('_', ' ').title()}: Actual={actual_count}, Predicted={predicted_count}")
        
        # Trend analysis
        actual_trend = actual_patterns.get('trend', {}).get('direction', 'unknown')
        predicted_trend = predicted_patterns.get('trend', {}).get('direction', 'unknown')
        print(f"Trend Direction: Actual={actual_trend}, Predicted={predicted_trend}")
    
    def analyze_temporal_features(self):
        """Perform temporal analysis to understand time-series characteristics."""
        print("="*50)
        print("TEMPORAL ANALYSIS")
        print("="*50)
        
        # Analyze actual price series
        actual_temporal = self.temporal_analyzer.get_comprehensive_analysis(
            data=self.actuals_unscaled,
            returns=None  # Will be calculated automatically
        )
        
        # Analyze prediction series
        predicted_temporal = self.temporal_analyzer.get_comprehensive_analysis(
            data=self.predictions_unscaled,
            returns=None
        )
        
        self.evaluation_results['temporal'] = {
            'actual_temporal': actual_temporal,
            'predicted_temporal': predicted_temporal
        }
        
        # Print temporal analysis summary
        print("TEMPORAL CHARACTERISTICS:")
        print("-" * 30)
        
        # Seasonality
        actual_seasonal = actual_temporal.get('seasonality', {})
        predicted_seasonal = predicted_temporal.get('seasonality', {})
        
        print(f"Seasonality Detected:")
        print(f"  Actual: {actual_seasonal.get('is_seasonal', False)}")
        print(f"  Predicted: {predicted_seasonal.get('is_seasonal', False)}")
        
        # Regime changes
        actual_regimes = actual_temporal.get('regime_changes', {})
        predicted_regimes = predicted_temporal.get('regime_changes', {})
        
        if actual_regimes:
            print(f"Regime Changes:")
            print(f"  Actual: {len(actual_regimes.get('change_points', []))} detected")
            print(f"  Predicted: {len(predicted_regimes.get('change_points', []))} detected")
    
    def create_visualizations(self, save_plots: bool = True):
        """Create comprehensive visualizations of model performance."""
        print("="*50)
        print("CREATING VISUALIZATIONS")
        print("="*50)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create plots directory
        if save_plots:
            os.makedirs('results/plots', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Price prediction plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Full prediction vs actual
        ax1 = axes[0, 0]
        ax1.plot(self.actuals_unscaled.flatten(), label='Actual', alpha=0.8)
        ax1.plot(self.predictions_unscaled.flatten(), label='Predicted', alpha=0.8)
        ax1.set_title('Full Test Set: Predictions vs Actual')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Last 60 days detail
        ax2 = axes[0, 1]
        last_n = min(60, len(self.actuals_unscaled))
        ax2.plot(self.actuals_unscaled[-last_n:].flatten(), label='Actual', alpha=0.8)
        ax2.plot(self.predictions_unscaled[-last_n:].flatten(), label='Predicted', alpha=0.8)
        ax2.set_title(f'Last {last_n} Days Detail')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Price ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Scatter plot of predictions vs actuals
        ax3 = axes[1, 0]
        ax3.scatter(self.actuals_unscaled.flatten(), self.predictions_unscaled.flatten(), alpha=0.6)
        min_val = min(self.actuals_unscaled.min(), self.predictions_unscaled.min())
        max_val = max(self.actuals_unscaled.max(), self.predictions_unscaled.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        ax3.set_xlabel('Actual Price ($)')
        ax3.set_ylabel('Predicted Price ($)')
        ax3.set_title('Prediction Accuracy Scatter')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Residuals
        ax4 = axes[1, 1]
        residuals = self.actuals_unscaled.flatten() - self.predictions_unscaled.flatten()
        ax4.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        ax4.axvline(0, color='red', linestyle='--', alpha=0.8)
        ax4.set_xlabel('Prediction Error ($)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Prediction Error Distribution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'results/plots/model_evaluation_{timestamp}.png', dpi=300, bbox_inches='tight')
            print(f"Saved evaluation plots to results/plots/model_evaluation_{timestamp}.png")
        else:
            plt.show()
        
        # 2. Trading performance plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Calculate returns for trading visualization
        actual_returns = np.diff(self.actuals_unscaled.flatten()) / self.actuals_unscaled.flatten()[:-1]
        predicted_directions = np.sign(np.diff(self.predictions_unscaled.flatten()))
        strategy_returns = predicted_directions * actual_returns
        
        # Remove NaN values
        strategy_returns = strategy_returns[~np.isnan(strategy_returns)]
        actual_returns = actual_returns[~np.isnan(actual_returns)]
        
        # Cumulative returns
        strategy_cumulative = np.cumprod(1 + strategy_returns) - 1
        buy_hold_cumulative = np.cumprod(1 + actual_returns[:len(strategy_returns)]) - 1
        
        # Plot cumulative returns
        ax1 = axes[0]
        ax1.plot(strategy_cumulative, label='LNN Strategy', linewidth=2)
        ax1.plot(buy_hold_cumulative, label='Buy & Hold', linewidth=2)
        ax1.set_title('Cumulative Returns Comparison')
        ax1.set_xlabel('Trading Days')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot rolling Sharpe ratio
        ax2 = axes[1]
        window = 30
        if len(strategy_returns) > window:
            rolling_sharpe = pd.Series(strategy_returns).rolling(window).apply(
                lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
            )
            ax2.plot(rolling_sharpe, label=f'{window}-day Rolling Sharpe')
            ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Sharpe = 1.0')
            ax2.set_title('Rolling Sharpe Ratio')
            ax2.set_xlabel('Trading Days')
            ax2.set_ylabel('Sharpe Ratio')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            # Use standardized plot paths
            prediction_plot_path = self.eval_paths['prediction_plot']
            trading_plot_path = self.eval_paths['trading_plot']
        
            # Save prediction accuracy plot
            plt.figure(1)  # First figure
            plt.savefig(prediction_plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved prediction plot to {prediction_plot_path}")
        
            # Save trading performance plot  
            plt.figure(2)  # Second figure
            plt.savefig(trading_plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved trading plot to {trading_plot_path}")
        else:
            plt.show()
        
        plt.close('all')  # Free memory
    
    def generate_report(self, save_report: bool = True):
        """Generate a comprehensive evaluation report."""
        print("="*50)
        print("GENERATING EVALUATION REPORT")
        print("="*50)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create report content
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("LIQUID NEURAL NETWORK MODEL EVALUATION REPORT")
        report_lines.append("="*60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Model: {os.path.basename(self.model_path)}")
        report_lines.append("")
        
        # Model configuration
        report_lines.append("MODEL CONFIGURATION:")
        report_lines.append("-" * 30)
        model_config = self.config.get('model', {})
        for key, value in model_config.items():
            report_lines.append(f"{key}: {value}")
        report_lines.append("")
        
        # Performance metrics
        if 'metrics' in self.evaluation_results:
            report_lines.append("PERFORMANCE METRICS:")
            report_lines.append("-" * 30)
            report_lines.append(self.metrics_calculator.get_metric_summary(self.evaluation_results['metrics']))
            report_lines.append("")
        
        # Pattern analysis
        if 'patterns' in self.evaluation_results:
            report_lines.append("PATTERN ANALYSIS:")
            report_lines.append("-" * 30)
            patterns = self.evaluation_results['patterns']
            actual_counts = patterns.get('actual_patterns', {}).get('pattern_count', {})
            predicted_counts = patterns.get('predicted_patterns', {}).get('pattern_count', {})
            
            for pattern_type in actual_counts.keys():
                report_lines.append(f"{pattern_type}: Actual={actual_counts.get(pattern_type, 0)}, "
                                  f"Predicted={predicted_counts.get(pattern_type, 0)}")
            report_lines.append("")
        
        # Temporal analysis
        if 'temporal' in self.evaluation_results:
            report_lines.append("TEMPORAL ANALYSIS:")
            report_lines.append("-" * 30)
            temporal = self.evaluation_results['temporal']
            
            actual_seasonal = temporal.get('actual_temporal', {}).get('seasonality', {})
            predicted_seasonal = temporal.get('predicted_temporal', {}).get('seasonality', {})
            
            report_lines.append(f"Seasonality - Actual: {actual_seasonal.get('is_seasonal', False)}, "
                              f"Predicted: {predicted_seasonal.get('is_seasonal', False)}")
            report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS:")
        report_lines.append("-" * 30)
        report_lines.extend(self._generate_recommendations())
        
        report_content = "\n".join(report_lines)
        
        # Save report
        if save_report:
            os.makedirs('results/reports', exist_ok=True)
            report_path = f'results/reports/evaluation_report_{timestamp}.txt'
            with open(report_path, 'w') as f:
                f.write(report_content)
            print(f"Evaluation report saved to {report_path}")
        
        # Print summary to console
        print(report_content)
        
        return report_content
    
    def _generate_recommendations(self) -> list:
        """Generate actionable recommendations based on evaluation results."""
        recommendations = []
        
        if 'metrics' in self.evaluation_results:
            metrics = self.evaluation_results['metrics']
            
            # Check basic metrics
            basic = metrics.get('basic_metrics', {})
            mape = basic.get('mape', float('inf'))
            r2 = basic.get('r2', 0)
            
            if mape > 10:
                recommendations.append("• High MAPE suggests model needs improvement. Try:")
                recommendations.append("  - Increase hidden layer size")
                recommendations.append("  - Add more features")
                recommendations.append("  - Increase sequence length")
            
            if r2 < 0.5:
                recommendations.append("• Low R² indicates poor fit. Consider:")
                recommendations.append("  - Different preprocessing (standardization vs minmax)")
                recommendations.append("  - Feature engineering")
                recommendations.append("  - Longer training")
            
            # Check trading performance
            trading = metrics.get('trading_metrics', {})
            sharpe = trading.get('sharpe_ratio', 0)
            max_dd = trading.get('max_drawdown', 0)
            
            if sharpe < 1.0:
                recommendations.append("• Low Sharpe ratio suggests risk-adjusted returns need improvement")
            
            if abs(max_dd) > 0.2:
                recommendations.append("• High drawdown indicates need for better risk management")
        
        if not recommendations:
            recommendations.append("• Model performance looks good! Consider:")
            recommendations.append("  - Testing on different time periods")
            recommendations.append("  - Adding more sophisticated features")
            recommendations.append("  - Ensemble methods")
        
        return recommendations
    
    def evaluate_model(self, detailed: bool = False, save_outputs: bool = True):
        """
        Run complete model evaluation pipeline.
    
        Args:
            detailed: Whether to run detailed analysis (slower)
            save_outputs: Whether to save plots and reports
        """
        print("="*60)
        print("STARTING COMPREHENSIVE MODEL EVALUATION")
        print("="*60)
    
        try:
            # Core evaluation steps
            self.load_model_and_config()
            self.generate_predictions()
            self.calculate_comprehensive_metrics()
        
            # Optional detailed analysis
            if detailed:
                print("\nRunning detailed analysis...")
                self.analyze_patterns()
                self.analyze_temporal_features()
        
            # Generate outputs
            self.create_visualizations(save_plots=save_outputs)
            self.generate_report(save_report=save_outputs)
        
            # Save evaluation results to standardized location (MOVED INSIDE TRY, BEFORE RETURN)
            if save_outputs:
                # Save evaluation results to standardized location
                with open(self.eval_paths['evaluation_json'], 'w') as f:
                    json.dump(self.evaluation_results, f, indent=2, default=str)
        
                # Save trading performance separately for easy access
                if 'trading_metrics' in self.evaluation_results.get('metrics', {}):
                    trading_results = {
                        'model_info': self.model_info,
                        'trading_metrics': self.evaluation_results['metrics']['trading_metrics'],
                        'timestamp': datetime.now().isoformat()
                    }
            
                    with open(self.eval_paths['trading_performance'], 'w') as f:
                        json.dump(trading_results, f, indent=2, default=str)
        
                print(f"Evaluation results saved to {self.eval_paths['evaluation_json']}")
        
            print("="*60)
            print("EVALUATION COMPLETED SUCCESSFULLY!")
            print("="*60)
          
            return self.evaluation_results
        
        except Exception as e:
            print(f"ERROR during evaluation: {e}")
            import traceback
            traceback.print_exc()
            raise
            
class EnhancedModelEvaluator(ModelEvaluator):  # Extend your existing evaluator
    """
    Enhanced evaluator that uses sophisticated trading strategy instead of simple buy/sell.
    """
    
    def __init__(self, model_path: str):
        super().__init__(model_path)
        # Initialize the enhanced trading strategy
        self.trading_strategy = EnhancedTradingStrategy(
            initial_capital=100000,
            max_position_size=0.2,
            risk_per_trade=0.02
        )
    
    def evaluate_trading_performance(self, predictions, actual_prices, dates):
        """
        Enhanced trading evaluation using sophisticated strategy.
        This replaces your simple buy/sell logic.
        """
        
        print("🎯 Evaluating with Enhanced Trading Strategy...")
        
        portfolio_value = self.trading_strategy.initial_capital
        positions = []
        trades = []
        portfolio_history = [portfolio_value]
        
        for i in range(len(predictions)):
            current_price = actual_prices[i]
            lnn_prediction = predictions[i]
            
            # Get price history up to this point
            history_start = max(0, i - 50)  # Last 50 days
            price_history = actual_prices[history_start:i+1]
            
            if len(price_history) < 10:  # Need some history
                portfolio_history.append(portfolio_value)
                continue
            
            # Generate sophisticated trading signal
            signal = self.trading_strategy.generate_trading_signal(
                lnn_prediction=lnn_prediction,
                current_price=current_price,
                price_history=price_history
            )
            
            # Execute trade if signal is not HOLD
            if signal.action.name in ['BUY', 'SELL']:
                # Calculate position value
                position_value = portfolio_value * signal.position_size
                shares = position_value / current_price
                
                if signal.action.name == 'BUY':
                    # Long position
                    positions.append({
                        'type': 'LONG',
                        'entry_price': current_price,
                        'shares': shares,
                        'entry_date': dates[i] if dates else i,
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit
                    })
                    
                    trades.append({
                        'date': dates[i] if dates else i,
                        'action': 'BUY',
                        'price': current_price,
                        'shares': shares,
                        'value': position_value,
                        'confidence': signal.confidence,
                        'reasoning': signal.reasoning
                    })
                
                elif signal.action.name == 'SELL':
                    # Short position (simplified - assume we can short)
                    positions.append({
                        'type': 'SHORT',
                        'entry_price': current_price,
                        'shares': shares,
                        'entry_date': dates[i] if dates else i,
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit
                    })
                    
                    trades.append({
                        'date': dates[i] if dates else i,
                        'action': 'SELL',
                        'price': current_price,
                        'shares': shares,
                        'value': position_value,
                        'confidence': signal.confidence,
                        'reasoning': signal.reasoning
                    })
            
            # Update portfolio value based on current positions
            current_portfolio_value = self.trading_strategy.initial_capital
            
            for position in positions:
                if position['type'] == 'LONG':
                    position_pnl = (current_price - position['entry_price']) * position['shares']
                else:  # SHORT
                    position_pnl = (position['entry_price'] - current_price) * position['shares']
                
                current_portfolio_value += position_pnl
            
            portfolio_history.append(current_portfolio_value)
            
            # Check stop losses and take profits (simplified)
            positions = self._check_exit_conditions(positions, current_price, dates[i] if dates else i)
        
        # Calculate enhanced performance metrics
        enhanced_metrics = self._calculate_enhanced_metrics(
            portfolio_history, trades, actual_prices
        )
        
        return enhanced_metrics
    
    def _check_exit_conditions(self, positions, current_price, current_date):
        """Check if any positions should be closed due to stop loss or take profit."""
        active_positions = []
        
        for position in positions:
            should_close = False
            
            if position['type'] == 'LONG':
                # Check stop loss
                if position.get('stop_loss') and current_price <= position['stop_loss']:
                    should_close = True
                # Check take profit
                elif position.get('take_profit') and current_price >= position['take_profit']:
                    should_close = True
            
            elif position['type'] == 'SHORT':
                # Check stop loss
                if position.get('stop_loss') and current_price >= position['stop_loss']:
                    should_close = True
                # Check take profit
                elif position.get('take_profit') and current_price <= position['take_profit']:
                    should_close = True
            
            if not should_close:
                active_positions.append(position)
        
        return active_positions
    
    def _calculate_enhanced_metrics(self, portfolio_history, trades, actual_prices):
        """Calculate sophisticated trading metrics."""
        
        portfolio_returns = np.diff(portfolio_history) / portfolio_history[:-1]
        market_returns = np.diff(actual_prices) / actual_prices[:-1]
        
        # Align lengths
        min_length = min(len(portfolio_returns), len(market_returns))
        portfolio_returns = portfolio_returns[:min_length]
        market_returns = market_returns[:min_length]
        
        # Calculate enhanced metrics
        total_return = (portfolio_history[-1] - portfolio_history[0]) / portfolio_history[0]
        buy_hold_return = (actual_prices[-1] - actual_prices[0]) / actual_prices[0]
        
        # Sharpe ratio
        portfolio_sharpe = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252) if np.std(portfolio_returns) > 0 else 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_history)
        drawdown = (peak - portfolio_history) / peak
        max_drawdown = np.max(drawdown)
        
        # Win rate
        if trades:
            profitable_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
            win_rate = profitable_trades / len(trades)
        else:
            win_rate = 0
        
        # Average trade confidence
        avg_confidence = np.mean([trade['confidence'] for trade in trades]) if trades else 0
        
        enhanced_metrics = {
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'excess_return': total_return - buy_hold_return,
            'sharpe_ratio': portfolio_sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'avg_confidence': avg_confidence,
            'final_portfolio_value': portfolio_history[-1],
            'portfolio_history': portfolio_history,
            'trade_details': trades
        }
        
        return enhanced_metrics


def compare_multiple_models(model_paths: list):
    """Compare multiple models side by side."""
    print("="*60)
    print("COMPARING MULTIPLE MODELS")
    print("="*60)
    
    comparison_results = {}
    
    for i, model_path in enumerate(model_paths):
        print(f"\nEvaluating model {i+1}/{len(model_paths)}: {os.path.basename(model_path)}")
        
        evaluator = ModelEvaluator(model_path)
        results = evaluator.evaluate_model(detailed=False, save_outputs=False)
        
        model_name = os.path.basename(model_path).replace('.pth', '')
        comparison_results[model_name] = results
    
    # Create comparison summary
    metrics_calculator = StockPredictionMetrics()
    comparison_df = metrics_calculator.compare_models(comparison_results)
    
    print("\nMODEL COMPARISON SUMMARY:")
    print("="*50)
    print(comparison_df.to_string(index=False))
    
    # Save comparison
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('results/reports', exist_ok=True)
    comparison_df.to_csv(f'results/reports/model_comparison_{timestamp}.csv', index=False)
    print(f"\nComparison saved to results/reports/model_comparison_{timestamp}.csv")
    
    return comparison_df

def main():
    """Main function for model evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate trained LNN models')
    parser.add_argument('--model', type=str, required=False,
                      help='Path to saved model file')
    parser.add_argument('--detailed', action='store_true',
                      help='Run detailed analysis (pattern recognition, temporal analysis)')
    parser.add_argument('--no-save', action='store_true',
                      help='Do not save plots and reports')
    parser.add_argument('--compare-models', type=str,
                      help='Comma-separated list of model paths to compare')
    
    args = parser.parse_args()
    
    if args.compare_models:
        # Compare multiple models
        model_paths = [path.strip() for path in args.compare_models.split(',')]
        compare_multiple_models(model_paths)
    
    elif args.model:
        # Evaluate single model
        evaluator = ModelEvaluator(args.model)
        evaluator.evaluate_model(
            detailed=args.detailed,
            save_outputs=not args.no_save
        )
    
    else:
        # Find most recent model
        models_dir = 'models/saved_models'
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
            if model_files:
                latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(models_dir, x)))
                model_path = os.path.join(models_dir, latest_model)
                print(f"No model specified, using latest: {model_path}")
                
                evaluator = ModelEvaluator(model_path)
                evaluator.evaluate_model(
                    detailed=args.detailed,
                    save_outputs=not args.no_save
                )
            else:
                print("No trained models found in models/saved_models/")
        else:
            print("No models directory found. Train a model first.")
            
def create_enhanced_evaluator(model_path: str):
    """
    Factory function to create the appropriate evaluator.
    
    This function determines whether to use enhanced or basic evaluation
    based on what's available in your system.
    
    Args:
        model_path: Path to the trained model
        
    Returns:
        ModelEvaluator or EnhancedModelEvaluator instance
    """
    
    if ENHANCED_STRATEGY_AVAILABLE and 'EnhancedModelEvaluator' in globals():
        print("🚀 Creating Enhanced Model Evaluator")
        return EnhancedModelEvaluator(model_path)
    else:
        print("📊 Creating Basic Model Evaluator (enhanced features not available)")
        return ModelEvaluator(model_path)

# Also add this helper function if it's not already there
def test_enhanced_evaluator():
    """
    Test function to verify enhanced evaluator is working.
    You can run this to test your setup.
    """
    print("🧪 Testing Enhanced Evaluator Setup...")
    
    try:
        # Test the factory function with a dummy path
        evaluator = create_enhanced_evaluator("dummy_model.pth")
        print(f"✅ Successfully created evaluator: {type(evaluator).__name__}")
        
        # Check available methods
        if hasattr(evaluator, 'evaluate_trading_performance_enhanced'):
            print("✅ Enhanced trading evaluation available")
        else:
            print("⚠️ Only basic trading evaluation available")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced evaluator test failed: {e}")
        return False

if __name__ == "__main__":
    # Test the enhanced evaluator when run directly
    test_enhanced_evaluator()
