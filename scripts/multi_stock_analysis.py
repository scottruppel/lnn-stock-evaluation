#!/usr/bin/env python3
"""
Comprehensive Multi-Stock LNN Analysis Script
Analyzes 20 stocks with 10 different hyperparameter combinations each (200 total experiments)
Builds comprehensive dataset for data mining and pattern discovery.

Usage on Jetson Orin Nano:
    # FROM PROJECT ROOT DIRECTORY:
    cd /path/to/your/lnn/project
    python scripts/multi_stock_analysis.py
    
    # Or with custom parameters:
    python scripts/multi_stock_analysis.py --quick-mode  # Faster testing
    python scripts/multi_stock_analysis.py --stocks-only "AAPL,TSLA,NVDA"  # Specific stocks
"""

import os
import sys
import json
import time
import yaml
import argparse
import itertools
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# === PATH SETUP ===
# Auto-detect project root and fix paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Go up one level from scripts/

print("=== MULTI-STOCK LNN ANALYSIS STARTING ===")
print(f"Script location: {SCRIPT_DIR}")
print(f"Project root: {PROJECT_ROOT}")
print(f"Current working directory: {os.getcwd()}")

# Change to project root if not already there
if os.getcwd() != PROJECT_ROOT:
    print(f"Changing to project root: {PROJECT_ROOT}")
    os.chdir(PROJECT_ROOT)

# Verify project structure
required_dirs = ['src', 'config', 'scripts', 'results']
missing_dirs = []
for directory in required_dirs:
    if os.path.exists(directory):
        print(f"✓ Found {directory}/")
    else:
        print(f"❌ Missing {directory}/")
        missing_dirs.append(directory)

if missing_dirs:
    print(f"ERROR: Missing directories: {missing_dirs}")
    print("Please run this script from your LNN project root directory.")
    sys.exit(1)

# Add src to Python path
src_path = os.path.join(PROJECT_ROOT, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
    print(f"✓ Added {src_path} to Python path")

# GPU monitoring imports (for Jetson)
try:
    import psutil
    GPU_MONITORING = True
    print("✓ psutil available for system monitoring")
except ImportError:
    GPU_MONITORING = False
    print("⚠️  psutil not available - system monitoring disabled")

# Test PyTorch and CUDA
try:
    import torch
    print(f"✓ PyTorch {torch.__version__} loaded")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU device: {torch.cuda.get_device_name(0)}")
    else:
        print("  ⚠️  Running on CPU - training will be slower")
except ImportError as e:
    print(f"❌ PyTorch import failed: {e}")

print("=== IMPORTING PROJECT MODULES ===")

# Test project imports
try:
    from data.data_loader import StockDataLoader
    print("✓ StockDataLoader imported")
except ImportError as e:
    print(f"❌ StockDataLoader import failed: {e}")
    print("  Make sure you have the data module in src/data/")

try:
    from run_analysis import ComprehensiveAnalyzer
    print("✓ ComprehensiveAnalyzer imported")
except ImportError as e:
    print(f"❌ ComprehensiveAnalyzer import failed: {e}")
    print("  Make sure run_analysis.py is in the scripts/ directory")

print("=== INITIALIZATION COMPLETE ===\n")

class MultiStockAnalyzer:
    """
    Comprehensive analyzer for testing LNN performance across multiple stocks
    and hyperparameter combinations. Configured via YAML files for maximum flexibility.
    """
    
    def __init__(self, config_path: str = "config/config.yaml", output_dir: str = None):
        print(f"Initializing MultiStockAnalyzer...")
        print(f"Config path: {config_path}")
        
        # Ensure we're working with absolute paths
        if not os.path.isabs(config_path):
            config_path = os.path.join(PROJECT_ROOT, config_path)
        
        self.config_path = config_path
        
        # Load configuration
        self.config = self.load_config()
        
        # Set output directory from config or parameter
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = self.config.get('multi_stock_analysis', {}).get(
                'output_dir', 'results/multi_stock_analysis'
            )
        
        # Ensure output directory is absolute
        if not os.path.isabs(self.output_dir):
            self.output_dir = os.path.join(PROJECT_ROOT, self.output_dir)
        
        self.results_file = os.path.join(self.output_dir, "comprehensive_results.json")
        self.summary_file = os.path.join(self.output_dir, "analysis_summary.json")
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load existing results if they exist
        self.all_results = self.load_existing_results()
        
        # Performance tracking
        self.start_time = None
        self.total_experiments = 0
        self.completed_experiments = 0
        
        print(f"✓ MultiStockAnalyzer initialized")
        print(f"  Configuration: {self.config_path}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Existing results loaded: {len(self.all_results)} experiments")
    
    def load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✓ Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            print(f"Warning: Config file {self.config_path} not found. Using minimal default.")
            return self.get_minimal_default_config()
        except Exception as e:
            print(f"Error loading config: {e}. Using minimal default.")
            return self.get_minimal_default_config()
    
    def get_minimal_default_config(self) -> Dict:
        """Provide minimal default configuration if config file is not available."""
        return {
            'multi_stock_analysis': {
                'active_universe': 'default',
                'active_grid': 'default',
                'output_dir': 'results/multi_stock_analysis'
            },
            'data': {
                'start_date': '2020-01-01',
                'end_date': '2024-12-31'
            },
            'model': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'num_epochs': 50,
                'patience': 8
            }
        }
    
    def get_stock_universe_from_config(self, config: Dict) -> List[Tuple[str, str]]:
        """
        Get stock universe from configuration file.
        Returns list of (ticker, description) tuples.
        """
        multi_stock_config = config.get('multi_stock_analysis', {})
        
        # Check if stocks are defined in config
        if 'stock_universes' in multi_stock_config:
            universe_name = multi_stock_config.get('active_universe', 'default')
            universes = multi_stock_config['stock_universes']
            
            if universe_name in universes:
                stock_list = universes[universe_name]
                # Convert to (ticker, description) tuples
                if isinstance(stock_list[0], dict):
                    # Format: [{"ticker": "AAPL", "description": "Apple Inc."}]
                    return [(stock['ticker'], stock.get('description', stock['ticker'])) 
                           for stock in stock_list]
                elif isinstance(stock_list[0], list):
                    # Format: [["AAPL", "Apple Inc."], ["MSFT", "Microsoft"]]
                    return [(stock[0], stock[1] if len(stock) > 1 else stock[0]) 
                           for stock in stock_list]
                else:
                    # Format: ["AAPL", "MSFT", "GOOGL"]
                    return [(ticker, ticker) for ticker in stock_list]
            else:
                print(f"Warning: Universe '{universe_name}' not found in config. Using default.")
        
        # Fallback to default universe if not in config
        print("Using default stock universe from config fallback.")
        return self.get_default_stock_universe()
    
    def get_default_stock_universe(self) -> List[Tuple[str, str]]:
        """
        Fallback default stock universe if not specified in config.
        """
        return [
            ("AAPL", "Apple - Consumer Electronics"),
            ("MSFT", "Microsoft - Software"),
            ("GOOGL", "Google - Internet Services"),
            ("NVDA", "NVIDIA - Semiconductors"),
            ("TSLA", "Tesla - Electric Vehicles"),
            ("JPM", "JPMorgan Chase - Banking"),
            ("JNJ", "Johnson & Johnson - Healthcare"),
            ("AMZN", "Amazon - E-commerce"),
            ("XOM", "ExxonMobil - Oil & Gas"),
            ("SPY", "S&P 500 ETF - Market Index")
        ]
    
    def get_hyperparameter_grid_from_config(self, config: Dict) -> List[Dict]:
        """
        Get hyperparameter combinations from configuration file.
        Returns list of parameter dictionaries.
        """
        multi_stock_config = config.get('multi_stock_analysis', {})
        
        # Check if hyperparameters are defined in config
        if 'hyperparameter_grids' in multi_stock_config:
            grid_name = multi_stock_config.get('active_grid', 'default')
            grids = multi_stock_config['hyperparameter_grids']
            
            if grid_name in grids:
                return grids[grid_name]
            else:
                print(f"Warning: Grid '{grid_name}' not found in config. Using default.")
        
        # Fallback to default grid
        print("Using default hyperparameter grid.")
        return self.get_default_hyperparameter_grid()
    
    def get_default_hyperparameter_grid(self) -> List[Dict]:
        """
        Fallback default hyperparameter grid if not specified in config.
        """
        return [
            {"hidden_size": 32, "sequence_length": 20, "description": "Small-Short"},
            {"hidden_size": 64, "sequence_length": 30, "description": "Medium-Medium"},
            {"hidden_size": 128, "sequence_length": 30, "description": "Large-Medium"},
            {"hidden_size": 64, "sequence_length": 50, "description": "Medium-Long"},
            {"hidden_size": 256, "sequence_length": 30, "description": "XLarge-Medium"}
        ]
    
    def load_existing_results(self) -> List[Dict]:
        """Load existing results from JSON file if it exists."""
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'r') as f:
                    return json.load(f)
            except:
                print("Warning: Could not load existing results file")
                return []
        return []
    
    def save_results(self):
        """Save current results to JSON file."""
        try:
            # Save main results
            with open(self.results_file, 'w') as f:
                json.dump(self.all_results, f, indent=2, default=str)
            
            # Save summary
            summary = self.generate_summary()
            with open(self.summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
                
            print(f"✓ Results saved to {self.results_file}")
            print(f"✓ Summary saved to {self.summary_file}")
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def experiment_exists(self, ticker: str, params: Dict) -> bool:
        """Check if experiment already exists in results."""
        for result in self.all_results:
            if (result.get('ticker') == ticker and 
                result.get('hidden_size') == params['hidden_size'] and
                result.get('sequence_length') == params['sequence_length']):
                return True
        return False
    
    def run_single_experiment(self, ticker: str, description: str, params: Dict) -> Dict:
        """
        Run a single LNN experiment for one stock with one parameter set.
        """
        experiment_start = time.time()
        experiment_id = f"{ticker}_{params['hidden_size']}h_{params['sequence_length']}s"
        
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {experiment_id}")
        print(f"Stock: {ticker} ({description})")
        print(f"Parameters: {params['description']}")
        print(f"Hidden Size: {params['hidden_size']}, Sequence Length: {params['sequence_length']}")
        print(f"{'='*60}")
        
        # Initialize result structure
        result = {
            'experiment_id': experiment_id,
            'ticker': ticker,
            'stock_description': description,
            'hidden_size': params['hidden_size'],
            'sequence_length': params['sequence_length'],
            'param_description': params['description'],
            'timestamp': datetime.now().isoformat(),
            'status': 'started',
            'error': None,
            'metrics': {},
            'training_info': {},
            'system_info': {}
        }
        
        try:
            # Monitor system resources at start
            if GPU_MONITORING:
                result['system_info']['cpu_percent_start'] = psutil.cpu_percent()
                result['system_info']['memory_percent_start'] = psutil.virtual_memory().percent
                result['system_info']['disk_usage_start'] = psutil.disk_usage('/').percent
            
            # Create custom config for this experiment
            config = self.create_experiment_config(ticker, params)
            config_path = os.path.join(PROJECT_ROOT, "config", f"temp_config_{experiment_id}.yaml")
            
            # Save temporary config
            temp_config_dir = os.path.join(PROJECT_ROOT, "config")
            os.makedirs(temp_config_dir, exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            # Import and run the comprehensive analyzer
            print("Importing ComprehensiveAnalyzer...")
            sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
            from run_analysis import ComprehensiveAnalyzer
            
            # Create analyzer with custom config
            analyzer = ComprehensiveAnalyzer(
                config_path=config_path,
                experiment_name=experiment_id
            )
            
            print("Running comprehensive analysis pipeline...")
            
            # Run all phases of analysis
            analysis_results = analyzer.run_complete_analysis()
            
            # Extract key metrics from analysis results
            result['metrics'] = self.extract_key_metrics(analysis_results)
            result['training_info'] = self.extract_training_info(analysis_results)
            
            # Monitor system resources at end
            if GPU_MONITORING:
                result['system_info']['cpu_percent_end'] = psutil.cpu_percent()
                result['system_info']['memory_percent_end'] = psutil.virtual_memory().percent
                result['system_info']['disk_usage_end'] = psutil.disk_usage('/').percent
            
            result['status'] = 'completed'
            result['duration_seconds'] = time.time() - experiment_start
            
            print(f"✓ Experiment {experiment_id} completed successfully")
            print(f"  Duration: {result['duration_seconds']:.1f} seconds")
            
            # Clean up temporary config
            if os.path.exists(config_path):
                os.remove(config_path)
            
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            result['duration_seconds'] = time.time() - experiment_start
            
            print(f"❌ Experiment {experiment_id} failed: {e}")
            
            # Clean up temporary config on error
            config_path = f"config/temp_config_{experiment_id}.yaml"
            if os.path.exists(config_path):
                os.remove(config_path)
        
        return result
    
    def create_experiment_config(self, ticker: str, params: Dict) -> Dict:
        """Create configuration for a specific experiment."""
        return {
            'data': {
                'tickers': ['^GSPC', 'AGG', 'QQQ', ticker],  # Include market context
                'start_date': '2020-01-01',
                'end_date': '2024-12-31',
                'target_ticker': ticker
            },
            'model': {
                'sequence_length': params['sequence_length'],
                'hidden_size': params['hidden_size'],
                'learning_rate': 0.001,
                'batch_size': 32,
                'num_epochs': 50,  # Reduced for faster experiments
                'patience': 8
            },
            'analysis': {
                'use_advanced_features': True,
                'n_components_pca': min(10, params['hidden_size'] // 4),
                'umap_n_neighbors': 15,
                'pattern_analysis': True,
                'temporal_analysis': True,
                'dimensionality_reduction': True
            }
        }
    
    def extract_key_metrics(self, analysis_results: Dict) -> Dict:
        """Extract key metrics from comprehensive analysis results."""
        metrics = {}
        
        try:
            # Data statistics
            if 'data_statistics' in analysis_results:
                data_stats = analysis_results['data_statistics']
                for ticker, stats in data_stats.items():
                    if ticker != '^GSPC':  # Skip market index
                        metrics['data_total_return'] = stats.get('total_return', None)
                        metrics['data_volatility'] = stats.get('return_std', None)
                        metrics['data_sharpe_estimate'] = stats.get('sharpe_estimate', None)
                        break
            
            # Training metrics
            if 'training' in analysis_results:
                training = analysis_results['training']
                metrics['training_final_loss'] = training.get('final_val_loss', None)
                metrics['training_best_loss'] = training.get('best_val_loss', None)
                metrics['training_epochs_completed'] = training.get('total_epochs', None)
            
            # Evaluation metrics
            if 'evaluation' in analysis_results:
                eval_results = analysis_results['evaluation']
                if isinstance(eval_results, dict) and 'metrics' in eval_results:
                    eval_metrics = eval_results['metrics']
                    
                    # Basic prediction metrics
                    if 'basic_metrics' in eval_metrics:
                        basic = eval_metrics['basic_metrics']
                        metrics['rmse'] = basic.get('rmse', None)
                        metrics['mae'] = basic.get('mae', None)
                        metrics['mape'] = basic.get('mape', None)
                        metrics['r2_score'] = basic.get('r2', None)
                    
                    # Directional accuracy
                    if 'directional_metrics' in eval_metrics:
                        directional = eval_metrics['directional_metrics']
                        metrics['directional_accuracy'] = directional.get('directional_accuracy', None)
                    
                    # Trading performance
                    if 'trading_metrics' in eval_metrics:
                        trading = eval_metrics['trading_metrics']
                        metrics['strategy_return'] = trading.get('total_return', None)
                        metrics['buy_hold_return'] = trading.get('buy_hold_return', None)
                        metrics['sharpe_ratio'] = trading.get('sharpe_ratio', None)
                        metrics['max_drawdown'] = trading.get('max_drawdown', None)
                        metrics['win_rate'] = trading.get('win_rate', None)
            
            # Pattern analysis
            if 'patterns' in analysis_results:
                patterns = analysis_results['patterns']
                if 'trend' in patterns:
                    trend = patterns['trend']
                    metrics['trend_direction'] = trend.get('direction', None)
                    metrics['trend_strength'] = trend.get('strength', None)
                
                if 'pattern_count' in patterns:
                    pattern_counts = patterns['pattern_count']
                    metrics['support_levels'] = pattern_counts.get('support_levels', 0)
                    metrics['resistance_levels'] = pattern_counts.get('resistance_levels', 0)
                    metrics['triangles'] = pattern_counts.get('triangles', 0)
            
            # Temporal analysis
            if 'temporal' in analysis_results:
                temporal = analysis_results['temporal']
                if 'seasonality' in temporal:
                    seasonality = temporal['seasonality']
                    metrics['is_seasonal'] = seasonality.get('is_seasonal', False)
                    metrics['dominant_period'] = seasonality.get('dominant_period', None)
                
                if 'regime_changes' in temporal:
                    regime = temporal['regime_changes']
                    metrics['n_regimes'] = regime.get('n_regimes', None)
        
        except Exception as e:
            print(f"Warning: Error extracting metrics: {e}")
        
        return metrics
    
    def extract_training_info(self, analysis_results: Dict) -> Dict:
        """Extract training information from analysis results."""
        training_info = {}
        
        try:
            if 'training' in analysis_results:
                training = analysis_results['training']
                
                # Copy relevant training information
                training_info = {
                    'epochs_completed': training.get('total_epochs', None),
                    'final_train_loss': training.get('final_train_loss', None),
                    'final_val_loss': training.get('final_val_loss', None),
                    'best_val_loss': training.get('best_val_loss', None),
                    'early_stopping': training.get('early_stopping_triggered', False),
                    'convergence_epoch': training.get('best_epoch', None)
                }
        
        except Exception as e:
            print(f"Warning: Error extracting training info: {e}")
        
        return training_info
    
    def run_comprehensive_analysis(self, 
                                 custom_stocks: List[str] = None,
                                 custom_universe: str = None,
                                 custom_grid: str = None,
                                 skip_existing: bool = True) -> Dict:
        """
        Run comprehensive multi-stock analysis using configuration.
        
        Args:
            custom_stocks: List of specific tickers to analyze (overrides config)
            custom_universe: Name of stock universe from config to use
            custom_grid: Name of hyperparameter grid from config to use
            skip_existing: Whether to skip experiments that already exist
        """
        
        self.start_time = time.time()
        
        # Get stocks to analyze
        if custom_stocks:
            stocks = [(ticker, f"Custom - {ticker}") for ticker in custom_stocks]
            print(f"Using custom stock list: {custom_stocks}")
        else:
            # Temporarily override config if custom universe specified
            if custom_universe:
                self.config['multi_stock_analysis']['active_universe'] = custom_universe
            stocks = self.get_stock_universe_from_config(self.config)
        
        # Get hyperparameter combinations
        if custom_grid:
            self.config['multi_stock_analysis']['active_grid'] = custom_grid
        param_combinations = self.get_hyperparameter_grid_from_config(self.config)
        
        # Calculate total experiments
        self.total_experiments = len(stocks) * len(param_combinations)
        self.completed_experiments = len(self.all_results) if skip_existing else 0
        
        print("=" * 80)
        print("MULTI-STOCK LNN COMPREHENSIVE ANALYSIS")
        print("=" * 80)
        print(f"Stocks to analyze: {len(stocks)}")
        print(f"Parameter combinations: {len(param_combinations)}")
        print(f"Total experiments: {self.total_experiments}")
        print(f"Existing experiments: {len(self.all_results)}")
        print(f"New experiments to run: {self.total_experiments - self.completed_experiments}")
        print(f"Skip existing: {skip_existing}")
        print("=" * 80)
        
        # List stocks
        print("\nSTOCKS TO ANALYZE:")
        for i, (ticker, description) in enumerate(stocks, 1):
            print(f"{i:2d}. {ticker:6s} - {description}")
        
        # List parameter combinations
        print(f"\nPARAMETER COMBINATIONS:")
        for i, params in enumerate(param_combinations, 1):
            print(f"{i:2d}. {params['description']:15s} - "
                  f"Hidden: {params['hidden_size']:3d}, Sequence: {params['sequence_length']:2d}")
        
        print("\n" + "=" * 80)
        print("STARTING ANALYSIS...")
        print("=" * 80)
        
        # Run experiments
        experiment_count = 0
        success_count = 0
        error_count = 0
        
        for stock_idx, (ticker, description) in enumerate(stocks, 1):
            print(f"\n🏢 STOCK {stock_idx}/{len(stocks)}: {ticker} ({description})")
            print("-" * 60)
            
            for param_idx, params in enumerate(param_combinations, 1):
                experiment_count += 1
                
                # Check if experiment already exists
                if skip_existing and self.experiment_exists(ticker, params):
                    print(f"⏭️  Experiment {experiment_count}/{self.total_experiments}: "
                          f"{ticker} + {params['description']} - SKIPPED (exists)")
                    continue
                
                print(f"🧪 Experiment {experiment_count}/{self.total_experiments}: "
                      f"{ticker} + {params['description']}")
                
                # Run the experiment
                result = self.run_single_experiment(ticker, description, params)
                
                # Add to results
                self.all_results.append(result)
                
                # Track success/failure
                if result['status'] == 'completed':
                    success_count += 1
                    print(f"✅ Success! (Total: {success_count}/{experiment_count})")
                else:
                    error_count += 1
                    print(f"❌ Failed! (Total errors: {error_count}/{experiment_count})")
                
                # Save results periodically (every 5 experiments)
                if experiment_count % 5 == 0:
                    self.save_results()
                    print(f"💾 Progress saved ({experiment_count}/{self.total_experiments} experiments)")
                
                # Show progress estimate
                elapsed = time.time() - self.start_time
                if experiment_count > 0:
                    avg_time_per_experiment = elapsed / experiment_count
                    remaining_experiments = self.total_experiments - experiment_count
                    estimated_remaining = avg_time_per_experiment * remaining_experiments
                    print(f"⏱️  Estimated time remaining: {estimated_remaining/60:.1f} minutes")
        
        # Final save
        self.save_results()
        
        # Generate final summary
        total_time = time.time() - self.start_time
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"Total experiments: {self.total_experiments}")
        print(f"Successful: {success_count}")
        print(f"Failed: {error_count}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Average time per experiment: {total_time/experiment_count:.1f} seconds")
        print(f"Results saved to: {self.results_file}")
        print(f"Summary saved to: {self.summary_file}")
        print("=" * 80)
        
        return {
            'total_experiments': self.total_experiments,
            'successful': success_count,
            'failed': error_count,
            'total_time_minutes': total_time / 60,
            'results_file': self.results_file,
            'summary_file': self.summary_file
        }
    
    def generate_summary(self) -> Dict:
        """Generate summary statistics from all results."""
        if not self.all_results:
            return {}
        
        summary = {
            'generated_at': datetime.now().isoformat(),
            'total_experiments': len(self.all_results),
            'successful_experiments': len([r for r in self.all_results if r['status'] == 'completed']),
            'failed_experiments': len([r for r in self.all_results if r['status'] == 'failed']),
            'stocks_analyzed': len(set(r['ticker'] for r in self.all_results)),
            'parameter_combinations': len(set(f"{r['hidden_size']}_{r['sequence_length']}" 
                                            for r in self.all_results)),
        }
        
        # Performance statistics
        successful_results = [r for r in self.all_results if r['status'] == 'completed']
        
        if successful_results:
            # Extract key metrics for summary
            sharpe_ratios = [r['metrics'].get('sharpe_ratio') for r in successful_results 
                           if r['metrics'].get('sharpe_ratio') is not None]
            directional_accuracies = [r['metrics'].get('directional_accuracy') for r in successful_results 
                                    if r['metrics'].get('directional_accuracy') is not None]
            strategy_returns = [r['metrics'].get('strategy_return') for r in successful_results 
                              if r['metrics'].get('strategy_return') is not None]
            
            summary['performance_stats'] = {
                'sharpe_ratio': {
                    'count': len(sharpe_ratios),
                    'mean': np.mean(sharpe_ratios) if sharpe_ratios else None,
                    'std': np.std(sharpe_ratios) if sharpe_ratios else None,
                    'min': np.min(sharpe_ratios) if sharpe_ratios else None,
                    'max': np.max(sharpe_ratios) if sharpe_ratios else None
                },
                'directional_accuracy': {
                    'count': len(directional_accuracies),
                    'mean': np.mean(directional_accuracies) if directional_accuracies else None,
                    'std': np.std(directional_accuracies) if directional_accuracies else None,
                    'min': np.min(directional_accuracies) if directional_accuracies else None,
                    'max': np.max(directional_accuracies) if directional_accuracies else None
                },
                'strategy_return': {
                    'count': len(strategy_returns),
                    'mean': np.mean(strategy_returns) if strategy_returns else None,
                    'std': np.std(strategy_returns) if strategy_returns else None,
                    'min': np.min(strategy_returns) if strategy_returns else None,
                    'max': np.max(strategy_returns) if strategy_returns else None
                }
            }
            
            # Best performing experiments
            if sharpe_ratios:
                best_sharpe_idx = np.argmax(sharpe_ratios)
                best_sharpe_experiment = successful_results[best_sharpe_idx]
                summary['best_sharpe_experiment'] = {
                    'experiment_id': best_sharpe_experiment['experiment_id'],
                    'ticker': best_sharpe_experiment['ticker'],
                    'sharpe_ratio': best_sharpe_experiment['metrics']['sharpe_ratio'],
                    'hidden_size': best_sharpe_experiment['hidden_size'],
                    'sequence_length': best_sharpe_experiment['sequence_length']
                }
            
            if directional_accuracies:
                best_accuracy_idx = np.argmax(directional_accuracies)
                best_accuracy_experiment = successful_results[best_accuracy_idx]
                summary['best_accuracy_experiment'] = {
                    'experiment_id': best_accuracy_experiment['experiment_id'],
                    'ticker': best_accuracy_experiment['ticker'],
                    'directional_accuracy': best_accuracy_experiment['metrics']['directional_accuracy'],
                    'hidden_size': best_accuracy_experiment['hidden_size'],
                    'sequence_length': best_accuracy_experiment['sequence_length']
                }
        
        return summary
    
    def generate_analysis_report(self) -> str:
        """Generate a comprehensive text report from all results."""
        if not self.all_results:
            return "No results available for report generation."
        
        summary = self.generate_summary()
        
        # Create report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MULTI-STOCK LNN ANALYSIS COMPREHENSIVE REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Experiments: {summary.get('total_experiments', 0)}")
        report_lines.append(f"Successful: {summary.get('successful_experiments', 0)}")
        report_lines.append(f"Failed: {summary.get('failed_experiments', 0)}")
        report_lines.append(f"Stocks Analyzed: {summary.get('stocks_analyzed', 0)}")
        report_lines.append("")
        
        # Performance overview
        if 'performance_stats' in summary:
            perf = summary['performance_stats']
            report_lines.append("PERFORMANCE OVERVIEW:")
            report_lines.append("-" * 40)
            
            if perf['sharpe_ratio']['count'] > 0:
                sr = perf['sharpe_ratio']
                report_lines.append(f"Sharpe Ratio - Mean: {sr['mean']:.3f}, "
                                  f"Std: {sr['std']:.3f}, Range: {sr['min']:.3f} to {sr['max']:.3f}")
            
            if perf['directional_accuracy']['count'] > 0:
                da = perf['directional_accuracy']
                report_lines.append(f"Directional Accuracy - Mean: {da['mean']:.1%}, "
                                  f"Std: {da['std']:.1%}, Range: {da['min']:.1%} to {da['max']:.1%}")
            
            if perf['strategy_return']['count'] > 0:
                sr = perf['strategy_return']
                report_lines.append(f"Strategy Return - Mean: {sr['mean']:.1%}, "
                                  f"Std: {sr['std']:.1%}, Range: {sr['min']:.1%} to {sr['max']:.1%}")
            
            report_lines.append("")
        
        # Best experiments
        if 'best_sharpe_experiment' in summary:
            best = summary['best_sharpe_experiment']
            report_lines.append("BEST SHARPE RATIO EXPERIMENT:")
            report_lines.append("-" * 40)
            report_lines.append(f"Stock: {best['ticker']}")
            report_lines.append(f"Sharpe Ratio: {best['sharpe_ratio']:.3f}")
            report_lines.append(f"Architecture: {best['hidden_size']} hidden units, {best['sequence_length']} sequence length")
            report_lines.append(f"Experiment ID: {best['experiment_id']}")
            report_lines.append("")
        
        if 'best_accuracy_experiment' in summary:
            best = summary['best_accuracy_experiment']
            report_lines.append("BEST DIRECTIONAL ACCURACY EXPERIMENT:")
            report_lines.append("-" * 40)
            report_lines.append(f"Stock: {best['ticker']}")
            report_lines.append(f"Directional Accuracy: {best['directional_accuracy']:.1%}")
            report_lines.append(f"Architecture: {best['hidden_size']} hidden units, {best['sequence_length']} sequence length")
            report_lines.append(f"Experiment ID: {best['experiment_id']}")
            report_lines.append("")
        
        # Stock performance breakdown
        successful_results = [r for r in self.all_results if r['status'] == 'completed']
        if successful_results:
            report_lines.append("TOP PERFORMING STOCKS:")
            report_lines.append("-" * 40)
            
            # Calculate average performance by stock
            stock_performance = {}
            for result in successful_results:
                ticker = result['ticker']
                if ticker not in stock_performance:
                    stock_performance[ticker] = []
                
                # Collect key metrics
                metrics = result.get('metrics', {})
                if metrics.get('sharpe_ratio') is not None:
                    stock_performance[ticker].append({
                        'sharpe_ratio': metrics['sharpe_ratio'],
                        'directional_accuracy': metrics.get('directional_accuracy', 0),
                        'strategy_return': metrics.get('strategy_return', 0)
                    })
            
            # Calculate averages and sort
            stock_averages = []
            for ticker, performances in stock_performance.items():
                if performances:
                    avg_sharpe = np.mean([p['sharpe_ratio'] for p in performances])
                    avg_accuracy = np.mean([p['directional_accuracy'] for p in performances if p['directional_accuracy']])
                    avg_return = np.mean([p['strategy_return'] for p in performances if p['strategy_return']])
                    
                    stock_averages.append({
                        'ticker': ticker,
                        'avg_sharpe': avg_sharpe,
                        'avg_accuracy': avg_accuracy,
                        'avg_return': avg_return,
                        'experiments': len(performances)
                    })
            
            # Sort by Sharpe ratio and show top 10
            stock_averages.sort(key=lambda x: x['avg_sharpe'], reverse=True)
            for i, stock in enumerate(stock_averages[:10], 1):
                report_lines.append(f"{i:2d}. {stock['ticker']:6s} - "
                                  f"Sharpe: {stock['avg_sharpe']:6.3f}, "
                                  f"Accuracy: {stock['avg_accuracy']:5.1%}, "
                                  f"Return: {stock['avg_return']:6.1%} "
                                  f"({stock['experiments']} experiments)")
            
            report_lines.append("")
        
        # Parameter analysis
        if successful_results:
            report_lines.append("PARAMETER COMBINATION ANALYSIS:")
            report_lines.append("-" * 40)
            
            # Calculate average performance by parameter combination
            param_performance = {}
            for result in successful_results:
                param_key = f"{result['hidden_size']}h_{result['sequence_length']}s"
                if param_key not in param_performance:
                    param_performance[param_key] = []
                
                metrics = result.get('metrics', {})
                if metrics.get('sharpe_ratio') is not None:
                    param_performance[param_key].append({
                        'sharpe_ratio': metrics['sharpe_ratio'],
                        'directional_accuracy': metrics.get('directional_accuracy', 0),
                        'description': result.get('param_description', param_key)
                    })
            
            # Calculate averages and sort
            param_averages = []
            for param_key, performances in param_performance.items():
                if performances:
                    avg_sharpe = np.mean([p['sharpe_ratio'] for p in performances])
                    avg_accuracy = np.mean([p['directional_accuracy'] for p in performances if p['directional_accuracy']])
                    description = performances[0]['description']
                    
                    param_averages.append({
                        'param_key': param_key,
                        'description': description,
                        'avg_sharpe': avg_sharpe,
                        'avg_accuracy': avg_accuracy,
                        'experiments': len(performances)
                    })
            
            # Sort by Sharpe ratio
            param_averages.sort(key=lambda x: x['avg_sharpe'], reverse=True)
            for i, param in enumerate(param_averages, 1):
                report_lines.append(f"{i:2d}. {param['description']:15s} - "
                                  f"Sharpe: {param['avg_sharpe']:6.3f}, "
                                  f"Accuracy: {param['avg_accuracy']:5.1%} "
                                  f"({param['experiments']} experiments)")
        
        # Technical summary
        report_lines.append("")
        report_lines.append("TECHNICAL SUMMARY:")
        report_lines.append("-" * 40)
        failed_results = [r for r in self.all_results if r['status'] == 'failed']
        if failed_results:
            report_lines.append(f"Failed experiments: {len(failed_results)}")
            # Show common failure reasons
            error_counts = {}
            for result in failed_results:
                error = result.get('error', 'Unknown error')
                # Simplify error message
                if 'CUDA' in error or 'GPU' in error:
                    error_key = 'GPU/CUDA Error'
                elif 'memory' in error.lower() or 'out of memory' in error.lower():
                    error_key = 'Out of Memory'
                elif 'timeout' in error.lower() or 'connection' in error.lower():
                    error_key = 'Network/Timeout'
                else:
                    error_key = 'Other Error'
                
                error_counts[error_key] = error_counts.get(error_key, 0) + 1
            
            for error_type, count in error_counts.items():
                report_lines.append(f"  {error_type}: {count} experiments")
        
        if successful_results:
            durations = [r.get('duration_seconds', 0) for r in successful_results if r.get('duration_seconds')]
            if durations:
                avg_duration = np.mean(durations)
                total_duration = sum(durations)
                report_lines.append(f"Average experiment duration: {avg_duration:.1f} seconds")
                report_lines.append(f"Total successful experiment time: {total_duration/3600:.1f} hours")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"analysis_report_{timestamp}.txt")
        
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✓ Comprehensive report saved to: {report_path}")
        return '\n'.join(report_lines)

# Add the missing import at the top of the file if not already there
try:
    import numpy as np
except ImportError:
    print("Warning: numpy not available for advanced statistics")
    np = None
