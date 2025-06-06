#!/usr/bin/env python3
"""
Main analysis script for comprehensive stock market analysis using Liquid Neural Networks.
This script orchestrates the complete pipeline from data analysis to model training and evaluation.

Usage:
    python scripts/run_analysis.py                                    # Full pipeline
    python scripts/run_analysis.py --data-only                        # Just data analysis
    python scripts/run_analysis.py --train-only                       # Just training
    python scripts/run_analysis.py --analyze-only                     # Just analysis
    python scripts/run_analysis.py --config config/custom_config.yaml # Custom config
    python scripts/run_analysis.py --experiment-name "my_experiment"  # Named experiment
    python scripts/run_analysis.py --quick                            # Fast analysis
"""

import os
import sys
import argparse
import yaml
import time
import numpy as np
import pandas as pd
import json
from datetime import datetime
import warnings
import torch
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
        return super (NumpyEncoder, self).default(obj)

# FIXED: Smart path detection for imports
def setup_import_paths():
    """Setup import paths to work from both scripts/ and project root."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Go up one level from scripts/
    src_path = os.path.join(project_root, 'src')
    
    # Add src to Python path if it exists and isn't already there
    if os.path.exists(src_path) and src_path not in sys.path:
        sys.path.insert(0, src_path)
        print(f"Added to Python path: {src_path}")
    
    # Also add the scripts directory for when run from elsewhere
    scripts_path = current_dir
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    
    return src_path

# Setup paths before any src imports
setup_import_paths()

# Now the original src imports should work
from data.data_loader import StockDataLoader
from data.preprocessor import StockDataPreprocessor
from analysis.pattern_recognition import PatternRecognizer
from analysis.feature_engineering import AdvancedFeatureEngineer
from analysis.dimensionality_reduction import DimensionalityReducer
from analysis.temporal_analysis import TemporalAnalyzer
from analysis.market_abstraction_pipeline import EnhancedFeatureEngineer
from utils.experiment_tracker import ExperimentTracker
from utils.metrics import StockPredictionMetrics
from utils.file_naming import file_namer, create_model_paths

# Import training and evaluation scripts as modules
from integrated_enhanced_trainer import LNNTrainer  # Uses enhanced pipeline
from evaluate_model import ModelEvaluator, create_enhanced_evaluator

class ComprehensiveAnalyzer:
    """
    Master orchestrator for complete stock market analysis pipeline.
    Combines data analysis, model training, and evaluation into a unified workflow.
    """
    
    def __init__(self, config_path: str = "config/config.yaml", experiment_name: str = None):
        """
        Initialize comprehensive analyzer.
        
        Args:
            config_path: Path to configuration file
            experiment_name: Name for this analysis run
        """
        self.config_path = config_path
        self.experiment_name = experiment_name or f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Load configuration
        self.config = self.load_config()
        
        # Initialize components
        self.data_loader = None
        self.raw_data = None
        self.processed_data = None
        
        # Analysis components
        self.pattern_recognizer = PatternRecognizer()
        self.feature_engineer = AdvancedFeatureEngineer()
        self.enhanced_engineer = EnhancedFeatureEngineer(use_abstractions=True)
        self.dim_reducer = DimensionalityReducer()
        self.temporal_analyzer = TemporalAnalyzer()
        
        # Results storage
        self.analysis_results = {}
        self.model_path = None
        self.evaluation_results = None
        
        # NEW: Create standardized file paths
        data_config = self.config.get('data', {})
        model_config = self.config.get('model', {})
        target_ticker = data_config.get('target_ticker', 'UNKNOWN')
        hidden_size = model_config.get('hidden_size', 50)
        sequence_length = model_config.get('sequence_length', 30)
    
        self.file_paths = create_model_paths(target_ticker, hidden_size, sequence_length, experiment_name)
    
        print(f"Standardized paths created:")
        print(f"  Model: {self.file_paths['model_path']}")
        print(f"  Analysis: {self.file_paths['analysis_json']}")
        
        # Experiment tracking
        self.experiment_tracker = ExperimentTracker()
        
        print(f"Initialized analyzer for experiment: {self.experiment_name}")
    
    def load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Loaded configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            print(f"Config file {self.config_path} not found. Using default configuration.")
            return self.get_default_config()
    
    def get_default_config(self) -> dict:
        """Provide comprehensive default configuration."""
        return {
            'data': {
                'tickers': ['^GSPC', 'AGG', 'QQQ', 'AAPL'],
                'start_date': '2020-01-01',
                'end_date': '2024-12-31',
                'target_ticker': 'AAPL'
            },
            'model': {
                'sequence_length': 30,
                'hidden_size': 50,
                'learning_rate': 0.001,
                'batch_size': 32,
                'num_epochs': 100,
                'patience': 10
            },
            'analysis': {
                'use_advanced_features': True,
                'n_components_pca': 10,
                'umap_n_neighbors': 15,
                'pattern_analysis': True,
                'temporal_analysis': True,
                'dimensionality_reduction': True
            }
        }
    
    def analyze_raw_data(self):
        """
        Comprehensive analysis of raw market data.
        This is your "market intelligence" phase - understanding the data before modeling.
        """
        print("=" * 70)
        print("PHASE 1: RAW DATA ANALYSIS")
        print("=" * 70)
    
        # 1. Load raw market data
        print("Loading market data...")
        self.data_loader = StockDataLoader(
            tickers=self.config['data']['tickers'],
            start_date=self.config['data']['start_date'],
            end_date=self.config['data']['end_date']
        )
    
        self.raw_data = self.data_loader.download_data()
        price_data = self.data_loader.get_closing_prices()
    
        # Validate data
        if not price_data:
            print("ERROR: No price data loaded!")
            raise ValueError("Failed to load any price data. Check your internet connection and ticker symbols.")
    
        # Check for empty data
        valid_tickers = []
        for ticker, prices in price_data.items():
            if prices is None or (hasattr(prices, 'size') and prices.size == 0):
                print(f"⚠️  WARNING: No data for {ticker}")
            else:
                valid_tickers.append(ticker)
    
        if not valid_tickers:
            print("ERROR: All tickers have empty data!")
            raise ValueError("No valid price data found for any ticker.")
    
        print(f"✓ Loaded {len(valid_tickers)} assets with valid data")
        print(f"✓ Date range: {self.config['data']['start_date']} to {self.config['data']['end_date']}")
    
        # Get length of first valid ticker's data
        first_valid_ticker = valid_tickers[0]
        first_valid_data = price_data[first_valid_ticker]
        data_length = len(first_valid_data) if hasattr(first_valid_data, '__len__') else first_valid_data.size
        print(f"✓ Total observations per asset: {data_length}")
    
        # 2. Basic data statistics
        print("\nCalculating basic statistics...")
        data_stats = {}
    
        for ticker, prices in price_data.items():
            # Skip empty data
            if prices is None or (hasattr(prices, 'size') and prices.size == 0):
                print(f"   Skipping {ticker} - no data")
                continue
            
            try:
                # Ensure prices is a 1D array
                if hasattr(prices, 'ndim') and prices.ndim > 1:
                    prices_flat = prices.flatten()
                else:
                    prices_flat = np.array(prices).flatten()
            
                # Check if we have enough data
                if len(prices_flat) < 2:
                    print(f"   Skipping {ticker} - insufficient data points")
                    continue
            
                # Calculate returns
                returns = (prices_flat[1:] - prices_flat[:-1]) / prices_flat[:-1]
            
                # Remove any NaN or infinite values
                returns_clean = returns[np.isfinite(returns)]
            
                if len(returns_clean) == 0:
                    print(f"   Skipping {ticker} - no valid returns")
                    continue
            
                data_stats[ticker] = {
                    'price_mean': float(np.mean(prices_flat)),
                    'price_std': float(np.std(prices_flat)),
                    'price_min': float(np.min(prices_flat)),
                    'price_max': float(np.max(prices_flat)),
                    'return_mean': float(np.mean(returns_clean)),
                    'return_std': float(np.std(returns_clean)),
                    'total_return': float((prices_flat[-1] - prices_flat[0]) / prices_flat[0]),
                    'sharpe_estimate': float(np.mean(returns_clean) / np.std(returns_clean) * np.sqrt(252)) if np.std(returns_clean) > 0 else 0
                }
            
            except Exception as e:
                print(f"   Error processing {ticker}: {e}")
                continue
    
        if not data_stats:
            print("ERROR: Could not calculate statistics for any ticker!")
            raise ValueError("Failed to process any ticker data.")
    
        self.analysis_results['data_statistics'] = data_stats
    
        # Print summary
        print("\nBASIC DATA SUMMARY:")
        print("-" * 40)
        for ticker, stats in data_stats.items():
            print(f"{ticker}: Total Return={stats['total_return']:.1%}, "
                f"Volatility={stats['return_std']*100:.1f}%, "
                f"Sharpe≈{stats['sharpe_estimate']:.2f}")
    
        # 3. Pattern Recognition Analysis
        if self.config.get('analysis', {}).get('pattern_analysis', True):
            print("\nRunning pattern recognition analysis...")
            target_ticker = self.config['data']['target_ticker']
        
        # Check if target ticker has valid data
        if target_ticker not in data_stats:
            print(f"⚠️  WARNING: Target ticker {target_ticker} has no valid data")
            # Try to use first valid ticker instead
            if valid_tickers:
                target_ticker = valid_tickers[0]
                print(f"   Using {target_ticker} as alternative target")
                self.config['data']['target_ticker'] = target_ticker
            else:
                print("   Skipping pattern analysis - no valid data")
                self.analysis_results['patterns'] = {'error': 'No valid target data'}
                return
        
        target_prices = price_data[target_ticker]
        
        try:
            pattern_results = self.pattern_recognizer.get_pattern_summary(target_prices)
            self.analysis_results['patterns'] = pattern_results
            
            # Print pattern summary
            print(f"\nPATTERN ANALYSIS FOR {target_ticker}:")
            print("-" * 40)
            pattern_counts = pattern_results.get('pattern_count', {})
            for pattern_type, count in pattern_counts.items():
                print(f"{pattern_type.replace('_', ' ').title()}: {count}")
            
            trend_info = pattern_results.get('trend', {})
            print(f"Overall Trend: {trend_info.get('direction', 'unknown').title()} "
                  f"(strength: {trend_info.get('strength', 0):.3f})")
        except Exception as e:
            print(f"   Error in pattern recognition: {e}")
            self.analysis_results['patterns'] = {'error': str(e)}
    
        # 4. Temporal Analysis
        if self.config.get('analysis', {}).get('temporal_analysis', True):
            print("\nRunning temporal analysis...")
            target_ticker = self.config['data']['target_ticker']
        
            if target_ticker in price_data and price_data[target_ticker] is not None:
                target_prices = price_data[target_ticker]
            
                try:
                    temporal_results = self.temporal_analyzer.get_comprehensive_analysis(target_prices)
                    self.analysis_results['temporal'] = temporal_results
                
                    # Print temporal summary
                    print(f"\nTEMPORAL ANALYSIS:")
                    print("-" * 40)
                
                    # Seasonality
                    seasonality = temporal_results.get('seasonality', {})
                    if seasonality.get('is_seasonal', False):
                        period = seasonality.get('dominant_period', 'unknown')
                        print(f"Seasonality detected with period: {period} days")
                    else:
                        print("No significant seasonality detected")
                
                    # Regime changes
                    regime_info = temporal_results.get('regime_changes', {})
                    if regime_info:
                        n_regimes = regime_info.get('n_regimes', 0)
                        change_points = regime_info.get('change_points', [])
                        print(f"Detected {n_regimes} market regimes with {len(change_points)} transitions")
                
                    # Autocorrelation
                    autocorr_info = temporal_results.get('autocorrelation', {})
                    if autocorr_info:
                        significant_lags = autocorr_info.get('significant_lags', [])
                        if significant_lags:
                            print(f"Significant autocorrelations at lags: {significant_lags[:5]}")
                        else:
                            print("No significant autocorrelations detected")
                except Exception as e:
                    print(f"   Error in temporal analysis: {e}")
                    self.analysis_results['temporal'] = {'error': str(e)}
            else:
                print("   Skipping temporal analysis - no valid target data")
                self.analysis_results['temporal'] = {'error': 'No valid target data'}
    
        print(f"\n✓ Raw data analysis completed")
    
    def analyze_features(self):
        """
        Advanced feature engineering and analysis.
        This creates and evaluates sophisticated features for modeling.
        """
        print("=" * 70)
        print("PHASE 2: FEATURE ENGINEERING & ANALYSIS")
        print("=" * 70)
        
        if not self.config.get('analysis', {}).get('use_advanced_features', True):
            print("Advanced feature analysis disabled in config. Skipping...")
            return
        
        # Get price data
        price_data = self.data_loader.get_closing_prices()
        target_ticker = self.config['data']['target_ticker']
        
        # 1. Create comprehensive features
        print("Creating advanced features...")
        
        # Create OHLCV structure (approximated from close prices)
        ohlcv_data = {
            'close': price_data[target_ticker],
            'high': price_data[target_ticker] * 1.02,    # Approximate high
            'low': price_data[target_ticker] * 0.98,     # Approximate low
            'open': price_data[target_ticker],           # Use close as open
            'volume': np.ones_like(price_data[target_ticker]) * 1000000  # Dummy volume
        }
        
        # Generate features
        features, feature_names = self.feature_engineer.create_comprehensive_features(
            ohlcv_data, include_advanced=True
        )
        features, feature_names = self.enhanced_engineer.create_features_with_abstractions(
            price_data=price_data,        # Your existing price_data 
            target_ticker=target_ticker,  # Your target ticker
            ohlcv_data=ohlcv_data        # Your existing OHLCV approximation
        )
        
        print(f"✓ Created {len(feature_names)} features")
        print(f"✓ Feature matrix shape: {features.shape}")
        
        # 2. Feature categorization and importance
        feature_categories = self.feature_engineer.get_feature_importance_by_category()
        self.analysis_results['feature_categories'] = feature_categories
        
        print("\nFEATURE CATEGORIES:")
        print("-" * 40)
        for category, feature_list in feature_categories.items():
            print(f"{category.title()}: {len(feature_list)} features")
        
        # 3. Dimensionality reduction analysis
        if self.config.get('analysis', {}).get('dimensionality_reduction', True):
            print("\nRunning dimensionality reduction analysis...")
            
            # Prepare target for feature selection
            target_prices = price_data[target_ticker].flatten()
            target_returns = np.diff(target_prices) / target_prices[:-1]
            
            # Align features with returns (remove first observation)
            features_aligned = features[1:]
            
            # Run comprehensive dimensionality reduction
            dim_results = self.dim_reducer.compare_dimensionality_methods(
                features_aligned, 
                target_returns.reshape(-1, 1),
                feature_names
            )
            
            self.analysis_results['dimensionality_reduction'] = dim_results
            
            # Print results
            print("\nDIMENSIONALITY REDUCTION RESULTS:")
            print("-" * 40)
            
            # PCA results
            if 'pca' in dim_results and dim_results['pca']:
                pca_result = dim_results['pca']
                n_components = pca_result['n_components']
                variance_explained = pca_result['total_variance_explained']
                print(f"PCA: {n_components} components explain {variance_explained:.1%} of variance")
            
            # Feature selection results
            if 'feature_selection' in dim_results and dim_results['feature_selection']:
                fs_result = dim_results['feature_selection']
                selected_features = fs_result['k']
                print(f"Feature Selection: Top {selected_features} features identified")
            
            # Print top features if available
            rankings = dim_results.get('feature_rankings', {})
            if rankings:
                for method, ranking_df in rankings.items():
                    if not ranking_df.empty:
                        top_features = ranking_df.head(5)['feature_name'].tolist()
                        print(f"Top 5 features ({method}): {', '.join(top_features)}")
        
        print(f"\n✓ Feature analysis completed")
    
    def train_model(self):
        """
        Train the Liquid Neural Network model using enhanced pipeline.
        """
        print("=" * 70)
        print("PHASE 3: MODEL TRAINING (ENHANCED)")
        print("=" * 70)

        # Check if enhanced features are enabled
        use_enhanced = self.config.get('analysis', {}).get('use_advanced_features', False)
        use_abstractions = self.config.get('analysis', {}).get('use_abstractions', False)

        if use_enhanced or use_abstractions:
            print("🧠 Using Enhanced Feature Pipeline")
            print("   ✅ Advanced features enabled")
            print("   ✅ Market abstractions enabled" if use_abstractions else "   ⚪ Market abstractions disabled")
        else:
            print("📊 Using Basic Feature Pipeline")

        # Initialize enhanced trainer  
        trainer = LNNTrainer(config_path=self.config_path)

        # CORRECTED: Use standardized model path 
        # Set the trainer to save to our standardized location
        original_model_path = trainer.train_model(experiment_name=self.experiment_name)
    
        # Copy to standardized location if different
        if original_model_path != self.file_paths['model_path']:
            import shutil
            shutil.copy2(original_model_path, self.file_paths['model_path'])
            print(f"Model copied to standardized location: {self.file_paths['model_path']}")
    
        # Use standardized path
        self.model_path = self.file_paths['model_path']

        # Store training results
        training_summary = trainer.get_training_summary()
        self.analysis_results['training'] = training_summary

        print(f"✅ Model training completed")
        print(f"✅ Model saved to: {self.model_path}")
    
    def evaluate_model(self):
        """
        Comprehensive model evaluation.
        """
        print("=" * 70)
        print("PHASE 4: MODEL EVALUATION")
        print("=" * 70)
        
        if not self.model_path:
            print("ERROR: No trained model available for evaluation")
            return
        
        # Initialize evaluator
        evaluator = ModelEvaluator(self.model_path)
        
        # Run comprehensive evaluation
        print("Starting model evaluation pipeline...")
        self.evaluation_results = evaluator.evaluate_model(
            detailed=True,  # Include pattern and temporal analysis
            save_outputs=True
        )
        
        # Store evaluation results
        self.analysis_results['evaluation'] = self.evaluation_results
        
        print(f"✓ Model evaluation completed")

    def evaluate_model_enhanced(self):
        """
        Enhanced model evaluation using sophisticated trading strategy.
        This is an ADDITIONAL method, don't replace your existing evaluate_model method.
        """
        print("=" * 70)
        print("PHASE 4: ENHANCED MODEL EVALUATION WITH SOPHISTICATED TRADING")
        print("=" * 70)
        
        if not self.model_path:
            print("ERROR: No trained model available for evaluation")
            return
        
        # Use the enhanced evaluator factory function
        try:
            enhanced_evaluator = create_enhanced_evaluator(self.model_path)
            print("✓ Enhanced evaluator created successfully")
        except Exception as e:
            print(f"ERROR creating enhanced evaluator: {e}")
            print("Falling back to basic evaluation...")
            # Fall back to your existing evaluation
            return self.evaluate_model()
        
        # Run enhanced evaluation
        print("Starting enhanced model evaluation pipeline...")
        
        try:
            # For now, we'll run a simplified version that integrates with your existing data
            self.evaluation_results = self._run_simplified_enhanced_evaluation(enhanced_evaluator)
            
            # Store evaluation results
            self.analysis_results['enhanced_evaluation'] = self.evaluation_results
            
            print(f"✓ Enhanced model evaluation completed")
            
        except Exception as e:
            print(f"ERROR during enhanced evaluation: {e}")
            print("Falling back to basic evaluation...")
            return self.evaluate_model()
    
    def validate_enhanced_integration(self):
        """Validate that enhanced features are working properly."""
    
        print("\n🔍 VALIDATING ENHANCED INTEGRATION")
        print("=" * 50)
    
        # Check configuration
        use_enhanced = self.config.get('analysis', {}).get('use_advanced_features', False)
        use_abstractions = self.config.get('analysis', {}).get('use_abstractions', False)
    
        print(f"Configuration check:")
        print(f"  use_advanced_features: {use_enhanced}")
        print(f"  use_abstractions: {use_abstractions}")
    
        # Check training results
        if 'training' in self.analysis_results:
            training = self.analysis_results['training']
            feature_count = training.get('feature_count', 0)
        
            print(f"Training results:")
            print(f"  Feature count: {feature_count}")
            print(f"  Model path: {training.get('model_path', 'Not found')}")
        
            # Validation checks
            if use_enhanced and feature_count < 50:
                print("❌ WARNING: Enhanced features enabled but low feature count")
                return False
            elif not use_enhanced and feature_count > 50:
                print("❌ WARNING: Enhanced features disabled but high feature count")
                return False
            else:
                print("✅ Feature configuration matches training results")
                return True
    
        return False
    
    def _run_simplified_enhanced_evaluation(self, enhanced_evaluator):
        """
        Simplified version of enhanced evaluation that works with your existing pipeline.
        """
        print("Running simplified enhanced evaluation...")
        
        # Get your existing data
        price_data = self.data_loader.get_closing_prices()
        target_ticker = self.config['data']['target_ticker']
        target_prices = price_data[target_ticker]
        
        # Create some mock predictions for testing
        # In a real implementation, you'd load your trained model and make actual predictions
        print("⚠️ Using mock predictions for testing - replace with actual model predictions")
        
        # Generate mock predictions (replace this with actual model predictions)
        n_predictions = min(100, len(target_prices) - 50)  # Last 100 days or available data
        mock_predictions = np.random.randn(n_predictions) * 0.02  # Random predictions ±2%
        actual_prices = target_prices[-n_predictions:]
        
        print(f"Testing enhanced strategy with {n_predictions} prediction points...")
        
        # Run enhanced trading evaluation
        if hasattr(enhanced_evaluator, 'evaluate_trading_performance_enhanced'):
            enhanced_results = enhanced_evaluator.evaluate_trading_performance_enhanced(
                predictions=mock_predictions,
                actual_prices=actual_prices
            )
        else:
            # Fall back to basic evaluation
            enhanced_results = enhanced_evaluator.evaluate_trading_performance_basic(
                predictions=mock_predictions,
                actual_prices=actual_prices
            )
        
        # Print results
        self._print_enhanced_results(enhanced_results)
        
        return enhanced_results
    
    def _print_enhanced_results(self, results):
        """Print the enhanced evaluation results."""
        
        print("\n" + "=" * 60)
        print("ENHANCED TRADING STRATEGY RESULTS")
        print("=" * 60)
        
        if 'total_return' in results:
            print(f"\n📈 RETURN METRICS:")
            print("-" * 30)
            print(f"Strategy Return:     {results['total_return']:.1%}")
            print(f"Buy & Hold Return:   {results['buy_hold_return']:.1%}")
            print(f"Excess Return:       {results['excess_return']:.1%}")
            
            if 'sharpe_ratio' in results:
                print(f"\n⚖️ RISK METRICS:")
                print("-" * 30)
                print(f"Sharpe Ratio:        {results['sharpe_ratio']:.2f}")
                if 'max_drawdown' in results:
                    print(f"Max Drawdown:        {results['max_drawdown']:.1%}")
            
            if 'total_trades' in results:
                print(f"\n🎯 TRADING METRICS:")
                print("-" * 30)
                print(f"Total Trades:        {results['total_trades']}")
                if 'avg_confidence' in results:
                    print(f"Avg Confidence:      {results['avg_confidence']:.2f}")
            
            print(f"\n💰 PORTFOLIO METRICS:")
            print("-" * 30)
            print(f"Final Portfolio Value: ${results['final_portfolio_value']:,.2f}")
            
            # Performance summary
            print(f"\n🏆 PERFORMANCE SUMMARY:")
            print("-" * 30)
            
            if results['excess_return'] > 0:
                print(f"✅ Strategy OUTPERFORMED buy-and-hold by {results['excess_return']:.1%}")
            else:
                print(f"❌ Strategy UNDERPERFORMED buy-and-hold by {abs(results['excess_return']):.1%}")
        
        else:
            print("⚠️ Enhanced results format not recognized")
            print("Available keys:", list(results.keys()))
    
    # MODIFY your existing run_complete_analysis method
    # Find this part and ADD the enhanced evaluation option:
    
    def run_complete_analysis(self, phases: list = None):
        """
        Run the complete analysis pipeline.
        MODIFIED to include enhanced evaluation option.
        """
        start_time = time.time()
        
        if phases is None:
            phases = ['data', 'features', 'training', 'evaluation', 'report']
        
        # Phase 3: Model Training
        if 'training' in phases:
            phase_start = time.time()
            self.train_model()
    
            # ADD THIS VALIDATION:
            validation_passed = self.validate_enhanced_integration()
            if not validation_passed:
                print("⚠️  Enhanced integration validation failed - continuing anyway")
    
            phase_duration = time.time() - phase_start
            print(f"✓ Model training completed in {phase_duration:.1f} seconds")
        
        # Phase 4: Model Evaluation - MODIFY this section
        if 'evaluation' in phases:
            phase_start = time.time()
            
            # Check if enhanced evaluation is requested
            if '--enhanced' in sys.argv or os.getenv('USE_ENHANCED_EVALUATION'):
                print("🚀 Using Enhanced Trading Evaluation")
                self.evaluate_model_enhanced()
            else:
                print("📊 Using Standard Evaluation")
                self.evaluate_model()
            
            phase_duration = time.time() - phase_start
            print(f"✓ Model evaluation completed in {phase_duration:.1f} seconds")
    
    def generate_comprehensive_report(self):
        """
        Generate a comprehensive analysis report combining all phases.
        """
        print("=" * 70)
        print("PHASE 5: COMPREHENSIVE REPORTING")
        print("=" * 70)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # NEW: Add this section for standardized naming  DELETE IF IT FAILS
        # from src.utils.file_naming import create_file_paths
        # file_paths = create_file_paths(self.config, self.experiment_name)
    
        # Use standardized paths (but keep legacy as backup)
        # report_path = file_paths['comprehensive_analysis']['standard_path']
        # results_path = file_paths['analysis_data']['standard_path']
        
        # Create comprehensive report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("COMPREHENSIVE STOCK MARKET ANALYSIS REPORT")
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
        
        # Data Analysis Summary
        if 'data_statistics' in self.analysis_results:
            report_lines.append("DATA ANALYSIS:")
            report_lines.append("-" * 40)
            target_ticker = self.config['data']['target_ticker']
            if target_ticker in self.analysis_results['data_statistics']:
                stats = self.analysis_results['data_statistics'][target_ticker]
                report_lines.append(f"Total Return: {stats['total_return']:.1%}")
                report_lines.append(f"Annualized Volatility: {stats['return_std']*100*(252**0.5):.1f}%")
                report_lines.append(f"Estimated Sharpe Ratio: {stats['sharpe_estimate']:.2f}")
            report_lines.append("")
        
        # Pattern Analysis Summary
        if 'patterns' in self.analysis_results:
            report_lines.append("PATTERN ANALYSIS:")
            report_lines.append("-" * 40)
            pattern_counts = self.analysis_results['patterns'].get('pattern_count', {})
            for pattern_type, count in pattern_counts.items():
                if count > 0:
                    report_lines.append(f"{pattern_type.replace('_', ' ').title()}: {count}")
            
            trend_info = self.analysis_results['patterns'].get('trend', {})
            direction = trend_info.get('direction', 'unknown')
            strength = trend_info.get('strength', 0)
            report_lines.append(f"Market Trend: {direction.title()} (strength: {strength:.3f})")
            report_lines.append("")
        
        # Model Performance Summary
        if 'evaluation' in self.analysis_results and self.evaluation_results:
            report_lines.append("MODEL PERFORMANCE:")
            report_lines.append("-" * 40)
            
            # Get key metrics
            metrics = self.evaluation_results.get('metrics', {})
            
            if 'basic_metrics' in metrics:
                basic = metrics['basic_metrics']
                report_lines.append(f"RMSE: ${basic.get('rmse', 'N/A'):.2f}")
                report_lines.append(f"MAPE: {basic.get('mape', 'N/A'):.1f}%")
                report_lines.append(f"R²: {basic.get('r2', 'N/A'):.3f}")
            
            if 'directional_metrics' in metrics:
                direction = metrics['directional_metrics']
                report_lines.append(f"Directional Accuracy: {direction.get('directional_accuracy', 'N/A'):.1%}")
            
            if 'trading_metrics' in metrics:
                trading = metrics['trading_metrics']
                report_lines.append(f"Strategy Return: {trading.get('total_return', 'N/A'):.1%}")
                report_lines.append(f"Sharpe Ratio: {trading.get('sharpe_ratio', 'N/A'):.2f}")
                report_lines.append(f"Max Drawdown: {trading.get('max_drawdown', 'N/A'):.1%}")
            
            report_lines.append("")
        
        # Feature Analysis Summary
        if 'feature_categories' in self.analysis_results:
            report_lines.append("FEATURE ANALYSIS:")
            report_lines.append("-" * 40)
            categories = self.analysis_results['feature_categories']
            total_features = sum(len(features) for features in categories.values())
            report_lines.append(f"Total Features Created: {total_features}")
            
            for category, features in categories.items():
                report_lines.append(f"{category.title()}: {len(features)} features")
            report_lines.append("")
        
        # Strategic Recommendations
        report_lines.append("STRATEGIC RECOMMENDATIONS:")
        report_lines.append("-" * 40)
        report_lines.extend(self._generate_strategic_recommendations())
        report_lines.append("")
        
        # Technical Configuration
        report_lines.append("TECHNICAL CONFIGURATION:")
        report_lines.append("-" * 40)
        report_lines.append(f"Model Architecture: LNN with {self.config.get('model', {}).get('hidden_size', 50)} hidden units")
        report_lines.append(f"Sequence Length: {self.config.get('model', {}).get('sequence_length', 30)} days")
        report_lines.append(f"Training Epochs: {self.config.get('model', {}).get('num_epochs', 100)}")
        report_lines.append(f"Learning Rate: {self.config.get('model', {}).get('learning_rate', 0.001)}")
        
        # Combine report
        report_content = "\n".join(report_lines)
        
        # Save comprehensive report
        # os.makedirs('results/reports', exist_ok=True)
        # report_path = f'results/reports/comprehensive_analysis_{self.experiment_name}_{timestamp}.txt'
        
        # with open(report_path, 'w') as f:
            # f.write(report_content)
        
        # Save analysis results as JSON
        results_path = self.file_paths['analysis_json']  # Use standardized path
        with open(results_path, 'w') as f:
            # Convert numpy types for JSON serialization
            json_safe_results = self._convert_to_json_safe(self.analysis_results)
            json.dump(json_safe_results, f, indent=2, cls=NumpyEncoder)
            
        # Use standardized paths instead of timestamp-based names 5 Jun Add
        report_path = self.file_paths['analysis_report']
        results_path = self.file_paths['analysis_json']
    
        # Save files to standardized locations 5 Jun Add
        with open(report_path, 'w') as f:
            f.write(report_content)
    
        with open(results_path, 'w') as f:
            json.dump(json_safe_results, f, indent=2, cls=NumpyEncoder)
        
        print(f"✓ Comprehensive report saved to: {report_path}")
        print(f"✓ Analysis data saved to: {results_path}")
        
        # Print executive summary to console
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE - EXECUTIVE SUMMARY")
        print("=" * 80)
        for line in self._generate_executive_summary():
            print(line)
        print("=" * 80)
        
        return report_path, results_path
    
    def _generate_executive_summary(self) -> list:
        """Generate executive summary based on all analysis results."""
        summary = []
        
        # Data overview
        target_ticker = self.config['data']['target_ticker']
        if 'data_statistics' in self.analysis_results and target_ticker in self.analysis_results['data_statistics']:
            stats = self.analysis_results['data_statistics'][target_ticker]
            total_return = stats['total_return']
            volatility = stats['return_std'] * 100 * (252**0.5)
            
            summary.append(f"• {target_ticker} delivered {total_return:.1%} total return with {volatility:.1f}% annualized volatility")
        
        # Pattern insights
        if 'patterns' in self.analysis_results:
            trend_info = self.analysis_results['patterns'].get('trend', {})
            direction = trend_info.get('direction', 'unknown')
            strength = trend_info.get('strength', 0)
            
            if direction != 'unknown':
                summary.append(f"• Market shows {direction} trend with {strength:.1%} statistical confidence")
        
        # Temporal insights
        if 'temporal' in self.analysis_results:
            temporal = self.analysis_results['temporal']
            if temporal.get('seasonality', {}).get('is_seasonal', False):
                period = temporal['seasonality'].get('dominant_period', 'unknown')
                summary.append(f"• Significant seasonality detected with {period}-day cycle")
            
            if 'regime_changes' in temporal and temporal['regime_changes']:
                n_regimes = temporal['regime_changes'].get('n_regimes', 0)
                summary.append(f"• Market exhibits {n_regimes} distinct behavioral regimes")
        
        # Model performance
        if 'evaluation' in self.analysis_results and self.evaluation_results:
            metrics = self.evaluation_results.get('metrics', {})
            
            # Trading performance
            if 'trading_metrics' in metrics:
                trading = metrics['trading_metrics']
                strategy_return = trading.get('total_return', 0)
                buy_hold_return = trading.get('buy_hold_return', 0)
                sharpe = trading.get('sharpe_ratio', 0)
                
                if strategy_return > buy_hold_return:
                    outperformance = strategy_return - buy_hold_return
                    summary.append(f"• LNN strategy outperformed buy-and-hold by {outperformance:.1%}")
                else:
                    underperformance = buy_hold_return - strategy_return
                    summary.append(f"• LNN strategy underperformed buy-and-hold by {underperformance:.1%}")
                
                summary.append(f"• Strategy achieved {sharpe:.2f} Sharpe ratio")
            
            # Prediction accuracy
            if 'directional_metrics' in metrics:
                directional_acc = metrics['directional_metrics'].get('directional_accuracy', 0)
                summary.append(f"• Model correctly predicted price direction {directional_acc:.1%} of the time")
        
        # Feature insights
        if 'feature_categories' in self.analysis_results:
            categories = self.analysis_results['feature_categories']
            total_features = sum(len(features) for features in categories.values())
            summary.append(f"• Analysis incorporated {total_features} engineered features across {len(categories)} categories")
        
        if not summary:
            summary.append("• Analysis completed successfully with comprehensive market intelligence generated")
        
        return summary
    
    def _generate_strategic_recommendations(self) -> list:
        """Generate strategic recommendations based on analysis."""
        recommendations = []
        
        # Model performance recommendations
        if 'evaluation' in self.analysis_results and self.evaluation_results:
            metrics = self.evaluation_results.get('metrics', {})
            
            # Check Sharpe ratio
            if 'trading_metrics' in metrics:
                sharpe = metrics['trading_metrics'].get('sharpe_ratio', 0)
                if sharpe < 1.0:
                    recommendations.append("• Sharpe ratio below 1.0 suggests risk management improvements needed")
                    recommendations.append("  - Consider position sizing based on volatility")
                    recommendations.append("  - Implement stop-loss mechanisms")
                elif sharpe > 1.5:
                    recommendations.append("• Strong Sharpe ratio suggests strategy is viable for live trading")
                    recommendations.append("  - Consider scaling up position sizes")
                    recommendations.append("  - Monitor performance during different market regimes")
            
            # Check directional accuracy
            if 'directional_metrics' in metrics:
                directional_acc = metrics['directional_metrics'].get('directional_accuracy', 0)
                if directional_acc < 0.55:
                    recommendations.append("• Low directional accuracy suggests model refinement needed")
                    recommendations.append("  - Increase sequence length for longer-term patterns")
                    recommendations.append("  - Add regime-aware features")
                elif directional_acc > 0.6:
                    recommendations.append("• High directional accuracy indicates strong predictive power")
                    recommendations.append("  - Consider more aggressive trading strategies")
        
        # Pattern-based recommendations
        if 'patterns' in self.analysis_results:
            pattern_counts = self.analysis_results['patterns'].get('pattern_count', {})
            
            if pattern_counts.get('support_levels', 0) > 3:
                recommendations.append("• Multiple support levels detected - consider range trading strategies")
            
            if pattern_counts.get('triangles', 0) > 0:
                recommendations.append("• Triangle patterns detected - monitor for breakout opportunities")
            
            trend_info = self.analysis_results['patterns'].get('trend', {})
            direction = trend_info.get('direction', 'unknown')
            strength = trend_info.get('strength', 0)
            
            if direction == 'upward' and strength > 0.7:
                recommendations.append("• Strong upward trend confirmed - favor long positions")
            elif direction == 'downward' and strength > 0.7:
                recommendations.append("• Strong downward trend confirmed - consider short strategies")
            elif direction == 'sideways':
                recommendations.append("• Sideways market detected - range trading may be optimal")
        
        # Temporal recommendations
        if 'temporal' in self.analysis_results:
            temporal = self.analysis_results['temporal']
            
            if temporal.get('seasonality', {}).get('is_seasonal', False):
                recommendations.append("• Seasonality detected - incorporate calendar effects in strategy")
            
            if 'regime_changes' in temporal and temporal['regime_changes']:
                recommendations.append("• Multiple market regimes identified - consider regime-switching models")
        
        # Technical recommendations
        if 'training' in self.analysis_results:
            training = self.analysis_results['training']
            final_val_loss = training.get('final_val_loss', float('inf'))
            
            if final_val_loss > 0.01:
                recommendations.append("• High validation loss suggests model underfitting")
                recommendations.append("  - Increase model capacity (hidden units)")
                recommendations.append("  - Train for more epochs")
                recommendations.append("  - Add more features")
        
        # Default recommendations if none generated
        if not recommendations:
            recommendations.extend([
                "• Model shows reasonable performance - consider these enhancements:",
                "  - Test on out-of-sample data from different time periods",
                "  - Implement ensemble methods combining multiple models",
                "  - Add alternative data sources (sentiment, macro indicators)",
                "  - Develop risk management overlays"
            ])
        
        return recommendations
    
    def _convert_to_json_safe(self, obj):
        """Convert numpy types and other non-JSON types to JSON-safe formats."""
        import numpy as np
        
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
        elif hasattr(obj, 'item'):  # For numpy scalars
            return obj.item()
        else:
            return obj
    
    def run_complete_analysis(self, phases: list = None):
        """
        Run the complete analysis pipeline.
        
        Args:
            phases: List of phases to run. If None, runs all phases.
                   Options: ['data', 'features', 'training', 'evaluation', 'report']
        """
        start_time = time.time()
        
        if phases is None:
            phases = ['data', 'features', 'training', 'evaluation', 'report']
        
        print("=" * 80)
        print("COMPREHENSIVE STOCK MARKET ANALYSIS PIPELINE")
        print("=" * 80)
        print(f"Experiment: {self.experiment_name}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Phases to run: {', '.join(phases)}")
        print("=" * 80)
        
        try:
            # Phase 1: Raw Data Analysis
            if 'data' in phases:
                phase_start = time.time()
                self.analyze_raw_data()
                phase_duration = time.time() - phase_start
                print(f"✓ Data analysis completed in {phase_duration:.1f} seconds")
            
            # Phase 2: Feature Engineering & Analysis
            if 'features' in phases:
                phase_start = time.time()
                self.analyze_features()
                phase_duration = time.time() - phase_start
                print(f"✓ Feature analysis completed in {phase_duration:.1f} seconds")
            
            # Phase 3: Model Training
            if 'training' in phases:
                phase_start = time.time()
                self.train_model()
                phase_duration = time.time() - phase_start
                print(f"✓ Model training completed in {phase_duration:.1f} seconds")
            
            # Phase 4: Model Evaluation
            if 'evaluation' in phases:
                phase_start = time.time()
                self.evaluate_model()
                phase_duration = time.time() - phase_start
                print(f"✓ Model evaluation completed in {phase_duration:.1f} seconds")
            
            # Phase 5: Comprehensive Reporting
            if 'report' in phases:
                phase_start = time.time()
                report_path, data_path = self.generate_comprehensive_report()
                phase_duration = time.time() - phase_start
                print(f"✓ Comprehensive reporting completed in {phase_duration:.1f} seconds")
            
            # Log the complete experiment
            self._log_complete_experiment()
            
            total_duration = time.time() - start_time
            print("\n" + "=" * 80)
            print("ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"Total execution time: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
            print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            if self.model_path:
                print(f"Trained model: {self.model_path}")
            
            if 'report' in phases:
                print(f"Comprehensive report: {report_path}")
                print(f"Analysis data: {data_path}")
            
            print("=" * 80)
            
            return self.analysis_results
            
        except Exception as e:
            print(f"\nERROR during analysis pipeline: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _log_complete_experiment(self):
        """Log the complete experiment with all results."""
        
        # Prepare comprehensive config
        complete_config = {
            'experiment_name': self.experiment_name,
            'data_config': self.config['data'],
            'model_config': self.config.get('model', {}),
            'analysis_config': self.config.get('analysis', {}),
            'phases_completed': list(self.analysis_results.keys())
        }
        
        # Prepare comprehensive metrics
        complete_metrics = {}
        
        # Add data statistics
        if 'data_statistics' in self.analysis_results:
            target_ticker = self.config['data']['target_ticker']
            if target_ticker in self.analysis_results['data_statistics']:
                stats = self.analysis_results['data_statistics'][target_ticker]
                complete_metrics.update({
                    'data_total_return': stats['total_return'],
                    'data_volatility': stats['return_std'],
                    'data_sharpe_estimate': stats['sharpe_estimate']
                })
        
        # Add training metrics
        if 'training' in self.analysis_results:
            training = self.analysis_results['training']
            complete_metrics.update({
                'training_final_loss': training.get('final_val_loss', None),
                'training_best_loss': training.get('best_val_loss', None),
                'training_epochs': training.get('total_epochs', None)
            })
        
        # Add evaluation metrics
        if 'evaluation' in self.analysis_results and self.evaluation_results:
            eval_metrics = self.evaluation_results.get('metrics', {})
            
            # Basic metrics
            if 'basic_metrics' in eval_metrics:
                basic = eval_metrics['basic_metrics']
                complete_metrics.update({
                    'eval_rmse': basic.get('rmse', None),
                    'eval_mae': basic.get('mae', None),
                    'eval_mape': basic.get('mape', None),
                    'eval_r2': basic.get('r2', None)
                })
            
            # Directional metrics
            if 'directional_metrics' in eval_metrics:
                directional = eval_metrics['directional_metrics']
                complete_metrics.update({
                    'eval_directional_accuracy': directional.get('directional_accuracy', None)
                })
            
            # Trading metrics
            if 'trading_metrics' in eval_metrics:
                trading = eval_metrics['trading_metrics']
                complete_metrics.update({
                    'eval_strategy_return': trading.get('total_return', None),
                    'eval_buy_hold_return': trading.get('buy_hold_return', None),
                    'eval_sharpe_ratio': trading.get('sharpe_ratio', None),
                    'eval_max_drawdown': trading.get('max_drawdown', None)
                })
        
        # Add pattern analysis results
        if 'patterns' in self.analysis_results:
            pattern_counts = self.analysis_results['patterns'].get('pattern_count', {})
            complete_metrics.update({
                'patterns_support_levels': pattern_counts.get('support_levels', 0),
                'patterns_resistance_levels': pattern_counts.get('resistance_levels', 0),
                'patterns_triangles': pattern_counts.get('triangles', 0)
            })
            
            trend_info = self.analysis_results['patterns'].get('trend', {})
            complete_metrics.update({
                'patterns_trend_direction': trend_info.get('direction', 'unknown'),
                'patterns_trend_strength': trend_info.get('strength', 0)
            })
        
        # Log experiment
        experiment_id = self.experiment_tracker.log_experiment(
            experiment_name=f"Complete Analysis: {self.experiment_name}",
            config=complete_config,
            metrics=complete_metrics,
            model_path=self.model_path,
            notes=f"Comprehensive analysis pipeline run on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            tags=['comprehensive', 'analysis', 'lnn', self.config['data']['target_ticker'].lower()]
        )
        
        print(f"✓ Complete experiment logged with ID: {experiment_id}")
        return experiment_id

def main():
    """Main function for running comprehensive analysis."""
    parser = argparse.ArgumentParser(description='Run comprehensive stock market analysis with LNN')
    
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
    parser.add_argument('--evaluate-only', action='store_true',
                      help='Run only model evaluation phase (requires trained model)')
    parser.add_argument('--report-only', action='store_true',
                      help='Run only reporting phase (requires previous analysis)')
    
    # Analysis control arguments
    parser.add_argument('--quick', action='store_true',
                      help='Run quick analysis (skip advanced features and detailed evaluation)')
    parser.add_argument('--no-patterns', action='store_true',
                      help='Skip pattern recognition analysis')
    parser.add_argument('--no-temporal', action='store_true',
                      help='Skip temporal analysis')
    parser.add_argument('--no-features', action='store_true',
                      help='Skip advanced feature engineering')
    
    # Output control
    parser.add_argument('--output-dir', type=str, default='results',
                      help='Directory for output files')
    parser.add_argument('--enhanced', action='store_true',
                      help='Use enhanced trading strategy evaluation')
    
    
    args = parser.parse_args()
    
    # Determine which phases to run
    phases = []
    if args.data_only:
        phases = ['data']
    elif args.features_only:
        phases = ['features']
    elif args.train_only:
        phases = ['training']
    elif args.evaluate_only:
        phases = ['evaluation']
    elif args.report_only:
        phases = ['report']
    else:
        # Default: run all phases
        phases = ['data', 'features', 'training', 'evaluation', 'report']
    
    # Modify config based on arguments
    if args.quick or args.no_features:
        # Load config and modify
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        except:
            config = {}
        
        if 'analysis' not in config:
            config['analysis'] = {}
        
        if args.quick or args.no_features:
            config['analysis']['use_advanced_features'] = False
        if args.quick or args.no_patterns:
            config['analysis']['pattern_analysis'] = False
        if args.quick or args.no_temporal:
            config['analysis']['temporal_analysis'] = False
        
        # Save modified config
        temp_config_path = 'config/temp_config.yaml'
        os.makedirs('config', exist_ok=True)
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        config_path = temp_config_path
    else:
        config_path = args.config
    
    # Generate experiment name if not provided
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.quick:
            experiment_name = f"quick_analysis_{timestamp}"
        elif len(phases) == 1:
            experiment_name = f"{phases[0]}_only_{timestamp}"
        else:
            experiment_name = f"full_analysis_{timestamp}"
    
    print("=" * 80)
    print("LIQUID NEURAL NETWORK COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    print(f"Experiment: {experiment_name}")
    print(f"Configuration: {config_path}")
    print(f"Phases: {', '.join(phases)}")
    print(f"Quick mode: {args.quick}")
    
    try:
        # Initialize and run analyzer
        analyzer = ComprehensiveAnalyzer(
            config_path=config_path,
            experiment_name=experiment_name
        )
        
        # Run analysis pipeline
        results = analyzer.run_complete_analysis(phases=phases)
        
        # Clean up temp config if created
        if config_path.endswith('temp_config.yaml') and os.path.exists(config_path):
            os.remove(config_path)
        
        print("\n🎉 ANALYSIS COMPLETED SUCCESSFULLY! 🎉")
        return results
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ ERROR during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
