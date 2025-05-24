#!/usr/bin/env python3
"""
Enhanced comprehensive stock analysis script with advanced forecasting capabilities.
This integrates your existing LNN pipeline with the new advanced forecasting features.

Usage:
    python run_analysis.py                                    # Full pipeline with advanced forecasting
    python run_analysis.py --data-only                        # Just data analysis
    python run_analysis.py --train-only                       # Just training
    python run_analysis.py --forecast-only                    # Just advanced forecasting
    python run_analysis.py --config config/custom_config.yaml # Custom config
    python run_analysis.py --experiment-name "my_experiment"  # Named experiment
    python run_analysis.py --quick                            # Fast analysis
"""

import os
import sys
import argparse
import yaml
import time
import numpy as np
import pandas as pd
import json
import torch
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif hasattr(obj, 'item'):
            return obj.item()
        return super(NumpyEncoder, self).default(obj)

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import data components
from data.data_loader import StockDataLoader
from data.preprocessor import StockDataPreprocessor

# Import analysis components
from analysis.pattern_recognition import PatternRecognizer
from analysis.feature_engineering import AdvancedFeatureEngineer
from analysis.dimensionality_reduction import DimensionalityReducer
from analysis.temporal_analysis import TemporalAnalyzer

# Import model components
from models.lnn_model import LiquidNeuralNetwork

# Import utility components
from utils.experiment_tracker import ExperimentTracker
from utils.metrics import StockPredictionMetrics

# Import the new advanced analysis components
try:
    from analysis.advanced_forecasting import AdvancedForecaster, OptionsAnalyzer
    from analysis.portfolio_optimizer import PortfolioOptimizer
    from analysis.backtester import Backtester
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Advanced features not available: {e}")
    print("Some functionality will be limited. Consider implementing missing modules.")
    ADVANCED_FEATURES_AVAILABLE = False

class EnhancedComprehensiveAnalyzer:
    """
    Enhanced analyzer that combines your existing LNN pipeline with advanced forecasting.
    This maintains compatibility with your current structure while adding new capabilities.
    """
    
    def __init__(self, config_path: str = "config/config.yaml", experiment_name: str = None):
        """
        Initialize enhanced comprehensive analyzer.
        
        Args:
            config_path: Path to configuration file
            experiment_name: Name for this analysis run
        """
        self.config_path = config_path
        self.experiment_name = experiment_name or f"enhanced_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Load configuration
        self.config = self.load_config()
        
        # Initialize your existing components
        self.data_loader = None
        self.preprocessor = StockDataPreprocessor()
        self.raw_data = None
        self.processed_data = None
        self.trained_models = {}
        
        # Analysis components (your existing ones)
        self.pattern_recognizer = PatternRecognizer()
        self.feature_engineer = AdvancedFeatureEngineer()
        self.dim_reducer = DimensionalityReducer()
        self.temporal_analyzer = TemporalAnalyzer()
        
        # New advanced components (if available)
        if ADVANCED_FEATURES_AVAILABLE:
            self.advanced_forecaster = None  # Will be initialized per model
            self.options_analyzer = OptionsAnalyzer()
            self.portfolio_optimizer = PortfolioOptimizer()
            self.backtester = Backtester()
        
        # Results storage
        self.analysis_results = {}
        self.training_results = {}
        self.forecasting_results = {}
        self.portfolio_results = {}
        
        # Experiment tracking
        self.experiment_tracker = ExperimentTracker()
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        print(f"🔬 Initialized enhanced analyzer for experiment: {self.experiment_name}")
    
    def load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ Loaded configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            print(f"⚠️  Config file {self.config_path} not found. Using default configuration.")
            return self.get_enhanced_default_config()
    
    def get_enhanced_default_config(self) -> dict:
        """Provide enhanced default configuration with advanced forecasting settings."""
        return {
            'data': {
                'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA'],
                'start_date': '2020-01-01',
                'end_date': '2024-12-31',
                'target_ticker': 'AAPL',
                'lookback_days': 252
            },
            'model': {
                'sequence_length': 30,
                'hidden_size': 64,
                'learning_rate': 0.001,
                'batch_size': 32,
                'num_epochs': 100,
                'patience': 15,
                'dropout_rate': 0.2
            },
            'analysis': {
                'use_advanced_features': True,
                'n_components_pca': 10,
                'umap_n_neighbors': 15,
                'pattern_analysis': True,
                'temporal_analysis': True,
                'dimensionality_reduction': True
            },
            'forecasting': {
                'forecast_horizons': [30, 60, 90],
                'n_simulations': 500,
                'confidence_levels': [0.68, 0.95, 0.99],
                'enable_options_analysis': True
            },
            'portfolio': {
                'optimization_method': 'mean_variance',
                'risk_tolerance': 0.1,
                'max_position_size': 0.3,
                'rebalance_frequency': 'monthly'
            }
        }
    
    def collect_and_analyze_data(self):
        """
        Phase 1: Data collection and initial analysis.
        This maintains your existing data analysis approach.
        """
        print("=" * 70)
        print("PHASE 1: DATA COLLECTION & ANALYSIS")
        print("=" * 70)
        
        # 1. Initialize data loader and collect data
        print("📊 Collecting market data...")
        self.data_loader = StockDataLoader(
            tickers=self.config['data']['tickers'],
            start_date=self.config['data']['start_date'],
            end_date=self.config['data']['end_date']
        )
        
        # Download raw data
        self.raw_data = self.data_loader.download_data()
        price_data = self.data_loader.get_closing_prices()
        
        print(f"✅ Loaded {len(price_data)} assets")
        print(f"✅ Date range: {self.config['data']['start_date']} to {self.config['data']['end_date']}")
        
        # 2. Basic statistical analysis
        print("\n📈 Calculating basic statistics...")
        data_stats = self._calculate_basic_statistics(price_data)
        self.analysis_results['data_statistics'] = data_stats
        
        # Print summary
        target_ticker = self.config['data']['target_ticker']
        if target_ticker in data_stats:
            stats = data_stats[target_ticker]
            print(f"\n{target_ticker} SUMMARY:")
            print(f"  Total Return: {stats['total_return']:.1%}")
            print(f"  Volatility: {stats['annualized_volatility']:.1f}%")
            print(f"  Sharpe Ratio: {stats['sharpe_estimate']:.2f}")
        
        # 3. Pattern recognition analysis
        if self.config.get('analysis', {}).get('pattern_analysis', True):
            print("\n🔍 Running pattern recognition...")
            target_prices = price_data[target_ticker]
            pattern_results = self.pattern_recognizer.get_pattern_summary(target_prices)
            self.analysis_results['patterns'] = pattern_results
            
            # Print pattern summary
            pattern_counts = pattern_results.get('pattern_count', {})
            trend_info = pattern_results.get('trend', {})
            print(f"  Patterns detected: {sum(pattern_counts.values())}")
            print(f"  Trend: {trend_info.get('direction', 'unknown').title()} "
                  f"(strength: {trend_info.get('strength', 0):.3f})")
        
        # 4. Temporal analysis
        if self.config.get('analysis', {}).get('temporal_analysis', True):
            print("\n⏰ Running temporal analysis...")
            target_prices = price_data[target_ticker]
            temporal_results = self.temporal_analyzer.get_comprehensive_analysis(target_prices)
            self.analysis_results['temporal'] = temporal_results
            
            # Print temporal insights
            seasonality = temporal_results.get('seasonality', {})
            if seasonality.get('is_seasonal', False):
                period = seasonality.get('dominant_period', 'unknown')
                print(f"  Seasonality: {period} days")
            else:
                print("  No significant seasonality detected")
        
        print("✅ Data collection and analysis completed")
    
    def engineer_and_analyze_features(self):
        """
        Phase 2: Feature engineering and analysis.
        Uses your existing feature engineering pipeline.
        """
        print("=" * 70)
        print("PHASE 2: FEATURE ENGINEERING & ANALYSIS")
        print("=" * 70)
        
        if not self.config.get('analysis', {}).get('use_advanced_features', True):
            print("⚠️  Advanced feature analysis disabled. Skipping...")
            return
        
        # Get price data for target ticker
        price_data = self.data_loader.get_closing_prices()
        target_ticker = self.config['data']['target_ticker']
        target_prices = price_data[target_ticker]
        
        print("🔧 Creating advanced features...")
        
        # Create OHLCV approximation from closing prices
        ohlcv_data = self._create_ohlcv_approximation(target_prices)
        
        # Generate comprehensive features
        features, feature_names = self.feature_engineer.create_comprehensive_features(
            ohlcv_data, include_advanced=True
        )
        
        print(f"✅ Created {len(feature_names)} features")
        print(f"✅ Feature matrix shape: {features.shape}")
        
        # Store feature information
        self.analysis_results['features'] = {
            'feature_names': feature_names,
            'feature_matrix_shape': features.shape,
            'n_features': len(feature_names)
        }
        
        # Feature categorization
        feature_categories = self.feature_engineer.get_feature_importance_by_category()
        self.analysis_results['feature_categories'] = feature_categories
        
        print("\n📊 Feature categories:")
        for category, feature_list in feature_categories.items():
            print(f"  {category.title()}: {len(feature_list)} features")
        
        # Dimensionality reduction analysis
        if self.config.get('analysis', {}).get('dimensionality_reduction', True):
            print("\n🔍 Running dimensionality reduction analysis...")
            
            # Prepare target for analysis
            target_returns = np.diff(target_prices.flatten()) / target_prices.flatten()[:-1]
            features_aligned = features[1:]  # Align with returns
            
            # Run dimensionality reduction
            dim_results = self.dim_reducer.compare_dimensionality_methods(
                features_aligned, 
                target_returns.reshape(-1, 1),
                feature_names
            )
            
            self.analysis_results['dimensionality_reduction'] = dim_results
            
            # Print summary
            if 'pca' in dim_results and dim_results['pca']:
                pca_result = dim_results['pca']
                print(f"  PCA: {pca_result['n_components']} components "
                      f"explain {pca_result['total_variance_explained']:.1%} of variance")
        
        print("✅ Feature engineering and analysis completed")
    
    def prepare_data_for_training(self):
        """
        Phase 3: Data preprocessing for model training.
        This uses your existing preprocessor.
        """
        print("=" * 70)
        print("PHASE 3: DATA PREPROCESSING FOR TRAINING")
        print("=" * 70)
        
        print("🔧 Preprocessing data for model training...")
        
        # Get raw price data
        price_data = self.data_loader.get_closing_prices()
        
        # Convert to DataFrame format expected by your preprocessor
        combined_data = []
        for ticker, prices in price_data.items():
            ticker_df = pd.DataFrame({
                'date': pd.date_range(start=self.config['data']['start_date'], 
                                    periods=len(prices), freq='D'),
                'ticker': ticker,
                'close': prices.flatten(),
                'high': prices.flatten() * 1.02,  # Approximate
                'low': prices.flatten() * 0.98,   # Approximate
                'open': prices.flatten(),         # Use close as open
                'volume': 1000000                 # Dummy volume
            })
            combined_data.append(ticker_df)
        
        combined_df = pd.concat(combined_data, ignore_index=True)
        
        # Fit preprocessor
        self.preprocessor.fit(combined_df)
        
        # Process each ticker
        self.processed_data = {}
        for ticker in self.config['data']['tickers']:
            ticker_data = combined_df[combined_df['ticker'] == ticker].copy()
            processed = self.preprocessor.transform(ticker_data)
            
            if processed is not None:
                self.processed_data[ticker] = processed
                print(f"✅ {ticker}: Preprocessed {len(processed['features'])} sequences")
            else:
                print(f"❌ {ticker}: Preprocessing failed")
        
        print(f"✅ Data preprocessing completed for {len(self.processed_data)} tickers")
    
    def train_models(self):
        """
        Phase 4: Model training.
        Train LNN models for each ticker.
        """
        print("=" * 70)
        print("PHASE 4: MODEL TRAINING")
        print("=" * 70)
        
        print(f"🧠 Training LNN models on {self.device}...")
        
        for ticker in self.processed_data.keys():
            print(f"\n🎯 Training model for {ticker}...")
            
            try:
                # Get processed data
                processed = self.processed_data[ticker]
                features = processed['features']
                targets = processed['targets']
                
                # Initialize model
                input_size = features.shape[2] if len(features.shape) > 2 else features.shape[1]
                model = LiquidNeuralNetwork(
                    input_size=input_size,
                    hidden_size=self.config['model']['hidden_size'],
                    output_size=1,
                    sequence_length=self.config['model']['sequence_length']
                ).to(self.device)
                
                # Training setup
                criterion = torch.nn.MSELoss()
                optimizer = torch.optim.Adam(
                    model.parameters(), 
                    lr=self.config['model']['learning_rate']
                )
                
                # Convert to tensors
                X_tensor = torch.FloatTensor(features).to(self.device)
                y_tensor = torch.FloatTensor(targets).to(self.device)
                
                # Training loop
                model.train()
                train_losses = []
                best_loss = float('inf')
                patience_counter = 0
                
                for epoch in range(self.config['model']['num_epochs']):
                    optimizer.zero_grad()
                    outputs = model(X_tensor)
                    loss = criterion(outputs, y_tensor)
                    loss.backward()
                    optimizer.step()
                    
                    train_losses.append(loss.item())
                    
                    # Early stopping
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        patience_counter = 0
                        # Save best model state
                        best_model_state = model.state_dict().copy()
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= self.config['model']['patience']:
                        print(f"    Early stopping at epoch {epoch+1}")
                        break
                    
                    if (epoch + 1) % 20 == 0:
                        print(f"    Epoch {epoch+1}: Loss = {loss.item():.6f}")
                
                # Restore best model
                model.load_state_dict(best_model_state)
                
                # Store trained model
                self.trained_models[ticker] = {
                    'model': model,
                    'final_loss': best_loss,
                    'training_losses': train_losses,
                    'sequence_length': self.config['model']['sequence_length']
                }
                
                print(f"✅ {ticker}: Training completed (Best Loss: {best_loss:.6f})")
                
            except Exception as e:
                print(f"❌ {ticker}: Training failed - {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n✅ Model training completed for {len(self.trained_models)} models")
    
    def run_backtesting(self):
        """
        Phase 5: Backtesting trained models.
        """
        print("=" * 70)
        print("PHASE 5: MODEL BACKTESTING")
        print("=" * 70)
        
        if not ADVANCED_FEATURES_AVAILABLE:
            print("⚠️  Backtesting module not available. Skipping...")
            return
        
        print("📈 Running backtests for trained models...")
        
        backtest_results = {}
        
        for ticker, model_data in self.trained_models.items():
            print(f"\n🎯 Backtesting {ticker}...")
            
            try:
                # Get data for backtesting
                processed = self.processed_data[ticker]
                features = processed['features']
                targets = processed['targets']
                
                # Use last 60 days for backtesting
                test_size = min(60, len(features) // 4)
                test_features = features[-test_size:]
                test_targets = targets[-test_size:]
                
                # Run backtest
                results = self.backtester.run_backtest(
                    model_data['model'], 
                    test_features, 
                    test_targets, 
                    ticker,
                    preprocessor=self.preprocessor
                )
                
                backtest_results[ticker] = results
                
                # Print results
                print(f"✅ {ticker}: Backtest completed")
                if 'sharpe_ratio' in results:
                    print(f"    Sharpe Ratio: {results['sharpe_ratio']:.3f}")
                if 'total_return' in results:
                    print(f"    Total Return: {results['total_return']:.1%}")
                if 'max_drawdown' in results:
                    print(f"    Max Drawdown: {results['max_drawdown']:.1%}")
                
            except Exception as e:
                print(f"❌ {ticker}: Backtesting failed - {e}")
        
        self.analysis_results['backtesting'] = backtest_results
        print("✅ Backtesting completed")
    
    def generate_advanced_forecasts(self):
        """
        Phase 6: Generate advanced multi-horizon forecasts.
        This is where your advanced_forecasting.py comes into play.
        """
        print("=" * 70)
        print("PHASE 6: ADVANCED FORECASTING")
        print("=" * 70)
        
        if not ADVANCED_FEATURES_AVAILABLE:
            print("⚠️  Advanced forecasting module not available. Skipping...")
            return
        
        print("🔮 Generating advanced multi-horizon forecasts...")
        
        for ticker, model_data in self.trained_models.items():
            print(f"\n🎯 Forecasting for {ticker}...")
            
            try:
                # Initialize advanced forecaster
                advanced_forecaster = AdvancedForecaster(
                    model=model_data['model'],
                    preprocessor=self.preprocessor,
                    device=self.device
                )
                
                # Get last sequence for forecasting
                processed = self.processed_data[ticker]
                features = processed['features']
                sequence_length = model_data['sequence_length']
                last_sequence = features[-sequence_length:]
                
                # Generate forecasts
                forecasts = advanced_forecaster.generate_multi_horizon_forecasts(
                    last_sequence=last_sequence,
                    target_ticker=ticker,
                    n_simulations=self.config.get('forecasting', {}).get('n_simulations', 500)
                )
                
                self.forecasting_results[ticker] = forecasts
                
                # Print forecast summary
                print(f"✅ {ticker}: Advanced forecasts generated")
                for horizon in ['30_day', '60_day', '90_day']:
                    if horizon in forecasts:
                        uncertainty = forecasts[horizon]['uncertainty_metrics']['mean_uncertainty']
                        reliability = forecasts[horizon]['uncertainty_metrics']['forecast_reliability']
                        print(f"    {horizon}: Reliability {reliability:.3f}, Uncertainty {uncertainty:.3f}")
                
            except Exception as e:
                print(f"❌ {ticker}: Advanced forecasting failed - {e}")
                import traceback
                traceback.print_exc()
        
        print("✅ Advanced forecasting completed")
    
    def analyze_options_opportunities(self):
        """
        Phase 7: Options analysis vs forecasts.
        """
        print("=" * 70)
        print("PHASE 7: OPTIONS ANALYSIS")
        print("=" * 70)
        
        if not ADVANCED_FEATURES_AVAILABLE:
            print("⚠️  Options analysis module not available. Skipping...")
            return
        
        if not self.config.get('forecasting', {}).get('enable_options_analysis', True):
            print("⚠️  Options analysis disabled in config. Skipping...")
            return
        
        print("💰 Analyzing options vs forecast opportunities...")
        
        options_results = {}
        
        for ticker in self.forecasting_results.keys():
            print(f"\n🎯 Analyzing options for {ticker}...")
            
            try:
                # Fetch options data
                options_data = self.options_analyzer.fetch_options_data(ticker)
                
                if options_data:
                    # Compare with forecasts
                    mismatches = self.options_analyzer.identify_forecast_option_mismatches(
                        forecasts=self.forecasting_results[ticker],
                        options_data=options_data,
                        current_price=options_data['current_price']
                    )
                    
                    options_results[ticker] = {
                        'current_price': options_data['current_price'],
                        'mismatches': mismatches,
                        'analysis_timestamp': datetime.now().isoformat()
                    }
                    
                    # Print opportunities
                    opportunity_count = 0
                    for expiry, mismatch in mismatches.items():
                        if isinstance(mismatch, dict) and 'trading_signal' in mismatch:
                            if mismatch['trading_signal'] not in ['NEUTRAL', 'WEAK_SIGNAL']:
                                opportunity_count += 1
                                signal = mismatch['trading_signal']
                                vol_diff = mismatch.get('percentage_difference', 0)
                                print(f"    📊 {expiry}: {signal} (Vol diff: {vol_diff:.1f}%)")
                    
                    if opportunity_count == 0:
                        print(f"    ℹ️  No significant options opportunities detected")
                    else:
                        print(f"✅ {ticker}: {opportunity_count} options opportunities identified")
                else:
                    print(f"⚠️  {ticker}: No options data available")
                    
            except Exception as e:
                print(f"❌ {ticker}: Options analysis failed - {e}")
        
        self.analysis_results['options'] = options_results
        print("✅ Options analysis completed")
    
    def optimize_portfolio(self):
        """
        Phase 8: Portfolio optimization using all insights.
        """
        print("=" * 70)
        print("PHASE 8: PORTFOLIO OPTIMIZATION")
        print("=" * 70)
        
        if not ADVANCED_FEATURES_AVAILABLE:
            print("⚠️  Portfolio optimization module not available. Skipping...")
            return
        
        print("🎯 Optimizing portfolio with all available insights...")
        
        try:
            # Prepare portfolio inputs
            portfolio_inputs = {}
            
            for ticker in self.config['data']['tickers']:
                if ticker in self.forecasting_results:
                    forecast_30d = self.forecasting_results[ticker]['30_day']
                    
                    # Calculate expected return from 30-day forecast
                    point_forecast = forecast_30d['point_forecast']
                    if len(point_forecast) > 0:
                        expected_return = (point_forecast[-1] - point_forecast[0]) / point_forecast[0]
                    else:
                        expected_return = 0.0
                    
                    portfolio_inputs[ticker] = {
                        'expected_return': float(expected_return[0]) if isinstance(expected_return, np.ndarray) else float(expected_return),
                        'volatility': forecast_30d['uncertainty_metrics']['mean_uncertainty'],
                        'reliability': forecast_30d['uncertainty_metrics']['forecast_reliability']
                    }
                elif ticker in self.analysis_results.get('data_statistics', {}):
                    # Fallback to historical statistics
                    stats = self.analysis_results['data_statistics'][ticker]
                    portfolio_inputs[ticker] = {
                        'expected_return': stats['return_mean'] * 30,  # 30-day expected return
                        'volatility': stats['return_std'],
                        'reliability': 0.5  # Neutral reliability
                    }
            
            if portfolio_inputs:
                # Run portfolio optimization
                optimal_weights = self.portfolio_optimizer.optimize_portfolio(
                    portfolio_inputs,
                    method=self.config.get('portfolio', {}).get('optimization_method', 'mean_variance'),
                    risk_tolerance=self.config.get('portfolio', {}).get('risk_tolerance', 0.1)
                )
                
                self.portfolio_results = {
                    'optimal_weights': optimal_weights,
                    'portfolio_inputs': portfolio_inputs,
                    'optimization_config': self.config.get('portfolio', {})
                }
                
                print("✅ Portfolio optimization completed")
                
                # Print top positions
                if optimal_weights:
                    sorted_weights = sorted(optimal_weights.items(), key=lambda x: x[1], reverse=True)
                    print("\n📊 Optimal Portfolio Allocation:")
                    for ticker, weight in sorted_weights:
                        if weight > 0.01:  # Only show weights > 1%
                            print(f"    {ticker}: {weight:.1%}")
                else:
                    print("⚠️  No optimal weights generated")
            else:
                print("⚠️  No data available for portfolio optimization")
                
        except Exception as e:
            print(f"❌ Portfolio optimization failed - {e}")
            import traceback
            traceback.print_exc()
    
    def generate_comprehensive_report(self):
        """
        Phase 9: Generate comprehensive analysis report.
        """
        print("=" * 70)
        print("PHASE 9: COMPREHENSIVE REPORTING")
        print("=" * 70)
        
        print("📋 Generating comprehensive analysis report...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create comprehensive report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ENHANCED STOCK MARKET ANALYSIS REPORT")
        report_lines.append("WITH ADVANCED FORECASTING")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Experiment: {self.experiment_name}")
        report_lines.append(f"Target Asset: {self.config['data']['target_ticker']}")
        report_lines.append(f"Analysis Period: {self.config['data']['start_date']} to {self.config['data']['end_date']}")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("EXECUTIVE SUMMARY:")
        report_lines.append("-" * 40)
        report_lines.extend(self._generate_executive_summary())
        report_lines.append("")
        
        # Model Performance
        report_lines.append("MODEL PERFORMANCE:")
        report_lines.append("-" * 40)
        for ticker, model_data in self.trained_models.items():
            report_lines.append(f"{ticker}: Training Loss = {model_data['final_loss']:.6f}")
        report_lines.append("")
        
        # Forecasting Results
        if self.forecasting_results:
            report_lines.append("FORECASTING INSIGHTS:")
            report_lines.append("-" * 40)
            for ticker, forecasts in self.forecasting_results.items():
                report_lines.append(f"\n{ticker}:")
                for horizon in ['30_day', '60_day', '90_day']:
                    if horizon in forecasts:
                        metrics = forecasts[horizon]['uncertainty_metrics']
                        reliability = metrics['forecast_reliability']
                        uncertainty = metrics['mean_uncertainty']
                        report_lines.append(f"  {horizon}: Reliability {reliability:.3f}, Uncertainty {uncertainty:.3f}")
        
        # Options Opportunities
        if self.analysis_results.get('options'):
            report_lines.append("\nOPTIONS TRADING OPPORTUNITIES:")
            report_lines.append("-" * 40)
            for ticker, analysis in self.analysis_results['options'].items():
                if 'current_price' in analysis:
                    report_lines.append(f"\n{ticker} (${analysis['current_price']:.2f}):")
                    if 'mismatches' in analysis:
                        opportunity_count = 0
                        for expiry, mismatch in analysis['mismatches'].items():
                            if isinstance(mismatch, dict) and 'trading_signal' in mismatch:
                                if mismatch['trading_signal'] not in ['NEUTRAL', 'WEAK_SIGNAL']:
                                    opportunity_count += 1
                                    signal = mismatch['trading_signal']
                                    vol_diff = mismatch.get('percentage_difference', 0)
                                    report_lines.append(f"  {expiry}: {signal} (Vol diff: {vol_diff:.1f}%)")
                        if opportunity_count == 0:
                            report_lines.append("  No significant opportunities detected")
        
        # Portfolio Optimization
        if self.portfolio_results:
            report_lines.append("\nPORTFOLIO OPTIMIZATION:")
            report_lines.append("-" * 40)
            optimal_weights = self.portfolio_results.get('optimal_weights', {})
            if optimal_weights:
                sorted_weights = sorted(optimal_weights.items(), key=lambda x: x[1], reverse=True)
                report_lines.append("Optimal Allocation:")
                for ticker, weight in sorted_weights:
                    if weight > 0.01:
                        report_lines.append(f"  {ticker}: {weight:.1%}")
        
        # Technical Configuration
        report_lines.append("\nTECHNICAL CONFIGURATION:")
        report_lines.append("-" * 40)
        model_config = self.config.get('model', {})
        report_lines.append(f"Model: LNN with {model_config.get('hidden_size', 64)} hidden units")
        report_lines.append(f"Sequence Length: {model_config.get('sequence_length', 30)} days")
        report_lines.append(f"Training Epochs: {model_config.get('num_epochs', 100)}")
        report_lines.append(f"Learning Rate: {model_config.get('learning_rate', 0.001)}")
        
        forecasting_config = self.config.get('forecasting', {})
        report_lines.append(f"Forecast Horizons: {forecasting_config.get('forecast_horizons', [30, 60, 90])}")
        report_lines.append(f"Monte Carlo Simulations: {forecasting_config.get('n_simulations', 500)}")
        
        # Combine report
        report_content = "\n".join(report_lines)
        
        # Ensure results directory exists
        os.makedirs('results/reports', exist_ok=True)
        
        # Save comprehensive report
        report_path = f'results/reports/enhanced_analysis_{self.experiment_name}_{timestamp}.txt'
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        # Save analysis results as JSON
        results_path = f'results/reports/analysis_data_{self.experiment_name}_{timestamp}.json'
        with open(results_path, 'w') as f:
            # Combine all results
            complete_results = {
                'analysis_results': self.analysis_results,
                'training_results': {ticker: {'final_loss': data['final_loss']} 
                                   for ticker, data in self.trained_models.items()},
                'forecasting_results': self.forecasting_results,
                'portfolio_results': self.portfolio_results,
                'config': self.config,
                'experiment_metadata': {
                    'experiment_name': self.experiment_name,
                    'timestamp': timestamp,
                    'device_used': str(self.device),
                    'advanced_features_available': ADVANCED_FEATURES_AVAILABLE
                }
            }
            
            # Convert to JSON-safe format
            json_safe_results = self._convert_to_json_safe(complete_results)
            json.dump(json_safe_results, f, indent=2, cls=NumpyEncoder)
        
        print(f"✅ Comprehensive report saved to: {report_path}")
        print(f"✅ Analysis data saved to: {results_path}")
        
        # Print executive summary to console
        print("\n" + "=" * 80)
        print("ENHANCED ANALYSIS COMPLETE - EXECUTIVE SUMMARY")
        print("=" * 80)
        for line in self._generate_executive_summary():
            print(line)
        print("=" * 80)
        
        return report_path, results_path
    
    def _calculate_basic_statistics(self, price_data: dict) -> dict:
        """Calculate basic statistics for each ticker."""
        stats = {}
        
        for ticker, prices in price_data.items():
            prices_flat = prices.flatten()
            returns = np.diff(prices_flat) / prices_flat[:-1]
            
            stats[ticker] = {
                'price_mean': float(np.mean(prices_flat)),
                'price_std': float(np.std(prices_flat)),
                'price_min': float(np.min(prices_flat)),
                'price_max': float(np.max(prices_flat)),
                'return_mean': float(np.mean(returns)),
                'return_std': float(np.std(returns)),
                'total_return': float((prices_flat[-1] - prices_flat[0]) / prices_flat[0]),
                'annualized_volatility': float(np.std(returns) * np.sqrt(252) * 100),
                'sharpe_estimate': float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
            }
        
        return stats
    
    def _create_ohlcv_approximation(self, prices: np.ndarray) -> dict:
        """Create OHLCV approximation from closing prices."""
        prices_flat = prices.flatten()
        
        return {
            'close': prices,
            'high': prices_flat * 1.02,    # Approximate 2% higher
            'low': prices_flat * 0.98,     # Approximate 2% lower
            'open': prices,                # Use close as open
            'volume': np.ones_like(prices_flat) * 1000000  # Dummy volume
        }
    
    def _generate_executive_summary(self) -> list:
        """Generate executive summary based on all analysis results."""
        summary = []
        
        # Basic data insights
        target_ticker = self.config['data']['target_ticker']
        if 'data_statistics' in self.analysis_results and target_ticker in self.analysis_results['data_statistics']:
            stats = self.analysis_results['data_statistics'][target_ticker]
            total_return = stats['total_return']
            volatility = stats['annualized_volatility']
            sharpe = stats['sharpe_estimate']
            
            summary.append(f"• {target_ticker} delivered {total_return:.1%} total return with {volatility:.1f}% annualized volatility")
            summary.append(f"• Historical Sharpe ratio: {sharpe:.2f}")
        
        # Model training results
        if self.trained_models:
            n_models = len(self.trained_models)
            avg_loss = np.mean([data['final_loss'] for data in self.trained_models.values()])
            summary.append(f"• Successfully trained {n_models} LNN models with average final loss: {avg_loss:.6f}")
        
        # Forecasting insights
        if self.forecasting_results:
            n_forecasts = len(self.forecasting_results)
            
            # Calculate average reliability
            reliabilities = []
            for forecasts in self.forecasting_results.values():
                if '30_day' in forecasts:
                    reliability = forecasts['30_day']['uncertainty_metrics']['forecast_reliability']
                    reliabilities.append(reliability)
            
            if reliabilities:
                avg_reliability = np.mean(reliabilities)
                summary.append(f"• Generated forecasts for {n_forecasts} assets with average 30-day reliability: {avg_reliability:.3f}")
        
        # Options opportunities
        if self.analysis_results.get('options'):
            total_opportunities = 0
            for analysis in self.analysis_results['options'].values():
                if 'mismatches' in analysis:
                    for mismatch in analysis['mismatches'].values():
                        if isinstance(mismatch, dict) and 'trading_signal' in mismatch:
                            if mismatch['trading_signal'] not in ['NEUTRAL', 'WEAK_SIGNAL']:
                                total_opportunities += 1
            
            if total_opportunities > 0:
                summary.append(f"• Identified {total_opportunities} potential options trading opportunities")
            else:
                summary.append("• No significant options mispricings detected")
        
        # Portfolio optimization
        if self.portfolio_results and 'optimal_weights' in self.portfolio_results:
            weights = self.portfolio_results['optimal_weights']
            if weights:
                max_allocation = max(weights.values())
                diversification = len([w for w in weights.values() if w > 0.05])  # Count allocations > 5%
                summary.append(f"• Optimal portfolio shows {diversification} significant positions (max allocation: {max_allocation:.1%})")
        
        # Pattern insights
        if 'patterns' in self.analysis_results:
            trend_info = self.analysis_results['patterns'].get('trend', {})
            direction = trend_info.get('direction', 'unknown')
            strength = trend_info.get('strength', 0)
            
            if direction != 'unknown':
                summary.append(f"• Market trend analysis shows {direction} direction with {strength:.1%} confidence")
        
        if not summary:
            summary.append("• Enhanced analysis completed successfully with comprehensive market intelligence")
        
        return summary
    
    def _convert_to_json_safe(self, obj):
        """Convert numpy types and other non-JSON types to JSON-safe formats."""
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
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif hasattr(obj, 'item'):  # For numpy scalars
            return obj.item()
        else:
            return obj
    
    def run_complete_pipeline(self, phases: list = None):
        """
        Run the complete enhanced analysis pipeline.
        
        Args:
            phases: List of phases to run. If None, runs all phases.
                   Options: ['data', 'features', 'preprocessing', 'training', 
                           'backtesting', 'forecasting', 'options', 'portfolio', 'report']
        """
        start_time = time.time()
        
        if phases is None:
            phases = ['data', 'features', 'preprocessing', 'training', 'backtesting', 
                     'forecasting', 'options', 'portfolio', 'report']
        
        print("=" * 80)
        print("ENHANCED STOCK MARKET ANALYSIS PIPELINE")
        print("WITH ADVANCED FORECASTING CAPABILITIES")
        print("=" * 80)
        print(f"Experiment: {self.experiment_name}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Phases to run: {', '.join(phases)}")
        print(f"Advanced features available: {ADVANCED_FEATURES_AVAILABLE}")
        print("=" * 80)
        
        try:
            # Phase 1: Data Collection & Analysis
            if 'data' in phases:
                phase_start = time.time()
                self.collect_and_analyze_data()
                phase_duration = time.time() - phase_start
                print(f"✅ Data analysis completed in {phase_duration:.1f} seconds\n")
            
            # Phase 2: Feature Engineering & Analysis
            if 'features' in phases:
                phase_start = time.time()
                self.engineer_and_analyze_features()
                phase_duration = time.time() - phase_start
                print(f"✅ Feature analysis completed in {phase_duration:.1f} seconds\n")
            
            # Phase 3: Data Preprocessing
            if 'preprocessing' in phases:
                phase_start = time.time()
                self.prepare_data_for_training()
                phase_duration = time.time() - phase_start
                print(f"✅ Data preprocessing completed in {phase_duration:.1f} seconds\n")
            
            # Phase 4: Model Training
            if 'training' in phases:
                phase_start = time.time()
                self.train_models()
                phase_duration = time.time() - phase_start
                print(f"✅ Model training completed in {phase_duration:.1f} seconds\n")
            
            # Phase 5: Backtesting
            if 'backtesting' in phases and ADVANCED_FEATURES_AVAILABLE:
                phase_start = time.time()
                self.run_backtesting()
                phase_duration = time.time() - phase_start
                print(f"✅ Backtesting completed in {phase_duration:.1f} seconds\n")
            
            # Phase 6: Advanced Forecasting
            if 'forecasting' in phases and ADVANCED_FEATURES_AVAILABLE:
                phase_start = time.time()
                self.generate_advanced_forecasts()
                phase_duration = time.time() - phase_start
                print(f"✅ Advanced forecasting completed in {phase_duration:.1f} seconds\n")
            
            # Phase 7: Options Analysis
            if 'options' in phases and ADVANCED_FEATURES_AVAILABLE:
                phase_start = time.time()
                self.analyze_options_opportunities()
                phase_duration = time.time() - phase_start
                print(f"✅ Options analysis completed in {phase_duration:.1f} seconds\n")
            
            # Phase 8: Portfolio Optimization
            if 'portfolio' in phases and ADVANCED_FEATURES_AVAILABLE:
                phase_start = time.time()
                self.optimize_portfolio()
                phase_duration = time.time() - phase_start
                print(f"✅ Portfolio optimization completed in {phase_duration:.1f} seconds\n")
            
            # Phase 9: Comprehensive Reporting
            if 'report' in phases:
                phase_start = time.time()
                report_path, data_path = self.generate_comprehensive_report()
                phase_duration = time.time() - phase_start
                print(f"✅ Comprehensive reporting completed in {phase_duration:.1f} seconds\n")
            
            total_duration = time.time() - start_time
            
            print("=" * 80)
            print("🎉 ENHANCED ANALYSIS PIPELINE COMPLETED SUCCESSFULLY! 🎉")
            print("=" * 80)
            print(f"Total execution time: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
            print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            if self.trained_models:
                print(f"Models trained: {len(self.trained_models)}")
            
            if self.forecasting_results:
                print(f"Forecasts generated: {len(self.forecasting_results)}")
            
            if 'report' in phases:
                print(f"Report saved to: {report_path}")
                print(f"Data saved to: {data_path}")
            
            print("=" * 80)
            
            return {
                'analysis_results': self.analysis_results,
                'training_results': self.trained_models,
                'forecasting_results': self.forecasting_results,
                'portfolio_results': self.portfolio_results
            }
            
        except Exception as e:
            print(f"\n❌ ERROR during enhanced analysis pipeline: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Main function for running enhanced comprehensive analysis."""
    parser = argparse.ArgumentParser(description='Run enhanced stock market analysis with LNN and advanced forecasting')
    
    # Pipeline control arguments
    parser.add_argument('--config', type=str, default='config/config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--experiment-name', type=str, default=None,
                      help='Name for this analysis experiment')
    
    # Phase control arguments
    parser.add_argument('--data-only', action='store_true',
                      help='Run only data analysis phase')
    parser.add_argument('--features-only', action='store_true',
                      help='Run only feature engineering phase')
    parser.add_argument('--train-only', action='store_true',
                      help='Run only model training phase')
    parser.add_argument('--forecast-only', action='store_true',
                      help='Run only advanced forecasting phase (requires trained models)')
    parser.add_argument('--portfolio-only', action='store_true',
                      help='Run only portfolio optimization phase')
    parser.add_argument('--report-only', action='store_true',
                      help='Run only reporting phase')
    
    # Analysis control arguments
    parser.add_argument('--quick', action='store_true',
                      help='Run quick analysis with reduced complexity')
    parser.add_argument('--no-patterns', action='store_true',
                      help='Skip pattern recognition analysis')
    parser.add_argument('--no-temporal', action='store_true',
                      help='Skip temporal analysis')
    parser.add_argument('--no-features', action='store_true',
                      help='Skip advanced feature engineering')
    parser.add_argument('--no-options', action='store_true',
                      help='Skip options analysis')
    
    # Performance arguments
    parser.add_argument('--fast-forecast', action='store_true',
                      help='Use fewer simulations for faster forecasting')
    parser.add_argument('--gpu', action='store_true',
                      help='Force GPU usage (will error if not available)')
    
    args = parser.parse_args()
    
    # Determine which phases to run
    phases = []
    if args.data_only:
        phases = ['data']
    elif args.features_only:
        phases = ['features']
    elif args.train_only:
        phases = ['preprocessing', 'training']
    elif args.forecast_only:
        phases = ['forecasting']
    elif args.portfolio_only:
        phases = ['portfolio']
    elif args.report_only:
        phases = ['report']
    else:
        # Default: run all phases
        phases = ['data', 'features', 'preprocessing', 'training', 'backtesting', 
                 'forecasting', 'options', 'portfolio', 'report']
    
    # Remove phases based on flags
    if args.no_options and 'options' in phases:
        phases.remove('options')
    
    # Generate experiment name if not provided
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.quick:
            experiment_name = f"enhanced_quick_{timestamp}"
        elif len(phases) == 1:
            experiment_name = f"enhanced_{phases[0]}_{timestamp}"
        else:
            experiment_name = f"enhanced_full_{timestamp}"
    
    # Handle GPU argument
    if args.gpu:
        if not torch.cuda.is_available():
            print("❌ GPU requested but CUDA not available!")
            return None
        print("🔥 GPU mode forced")
    
    print("=" * 80)
    print("🚀 ENHANCED LIQUID NEURAL NETWORK ANALYSIS")
    print("   WITH ADVANCED FORECASTING CAPABILITIES")
    print("=" * 80)
    print(f"Experiment: {experiment_name}")
    print(f"Configuration: {args.config}")
    print(f"Phases: {', '.join(phases)}")
    print(f"Quick mode: {args.quick}")
    print(f"GPU available: {torch.cuda.is_available()}")
    print(f"Advanced features available: {ADVANCED_FEATURES_AVAILABLE}")
    
    try:
        # Initialize and run enhanced analyzer
        analyzer = EnhancedComprehensiveAnalyzer(
            config_path=args.config,
            experiment_name=experiment_name
        )
        
        # Modify config for quick mode
        if args.quick:
            analyzer.config['model']['num_epochs'] = 50
            analyzer.config['forecasting']['n_simulations'] = 100
        
        if args.fast_forecast:
            analyzer.config['forecasting']['n_simulations'] = 100
        
        if args.no_patterns:
            analyzer.config['analysis']['pattern_analysis'] = False
        
        if args.no_temporal:
            analyzer.config['analysis']['temporal_analysis'] = False
        
        if args.no_features:
            analyzer.config['analysis']['use_advanced_features'] = False
        
        if args.no_options:
            analyzer.config['forecasting']['enable_options_analysis'] = False
        
        # Run enhanced analysis pipeline
        results = analyzer.run_complete_pipeline(phases=phases)
        
        print("\n🎊 ENHANCED ANALYSIS COMPLETED SUCCESSFULLY! 🎊")
        
        # Quick summary
        if results:
            if 'training_results' in results and results['training_results']:
                print(f"📊 Trained {len(results['training_results'])} models")
            
            if 'forecasting_results' in results and results['forecasting_results']:
                print(f"🔮 Generated forecasts for {len(results['forecasting_results'])} assets")
            
            if 'portfolio_results' in results and results['portfolio_results']:
                print("🎯 Portfolio optimization completed")
        
        return results
        
    except KeyboardInterrupt:
        print("\n⚠️  Enhanced analysis interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ ERROR during enhanced analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
