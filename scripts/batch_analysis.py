#!/usr/bin/env python3
"""
Batch Analysis Script for LNN Stock Market Analysis
Runs comprehensive analysis across multiple stocks and parameter combinations.

This script automates the process of testing different model configurations
across various stock portfolios to optimize hyperparameters.

Usage:
    python scripts/batch_analysis.py                           # Run all portfolios
    python scripts/batch_analysis.py --portfolio tech          # Run specific portfolio
    python scripts/batch_analysis.py --quick                   # Run quick analysis
    python scripts/batch_analysis.py --dry-run                 # Show what will be run
"""
import subprocess
import tempfile
import os
import sys
import argparse
import yaml
import json
import time
import itertools
import pandas as pd
import numpy as np

# Set matplotlib backend before any plotting imports
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for headless operation

from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# FIXED: Proper path setup for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
scripts_path = os.path.join(project_root, 'scripts')

# Add both src and scripts to Python path
for path in [src_path, scripts_path, project_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

print(f"Project root: {project_root}")
print(f"Added paths: {[src_path, scripts_path]}")
from utils.file_naming import file_namer

# FIXED: Import run_analysis as a module and extract the class
def import_comprehensive_analyzer():
    """Import ComprehensiveAnalyzer with proper error handling."""
    try:
        # Change to the scripts directory temporarily
        original_cwd = os.getcwd()
        os.chdir(scripts_path)
        
        # Import run_analysis module
        import importlib.util
        spec = importlib.util.spec_from_file_location("run_analysis", 
                                                     os.path.join(scripts_path, "run_analysis.py"))
        run_analysis_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_analysis_module)
        
        # Get the ComprehensiveAnalyzer class
        ComprehensiveAnalyzer = run_analysis_module.ComprehensiveAnalyzer
        
        # Return to original directory
        os.chdir(original_cwd)
        
        print("✅ Successfully imported ComprehensiveAnalyzer")
        return ComprehensiveAnalyzer
        
    except Exception as e:
        print(f"❌ Failed to import ComprehensiveAnalyzer: {e}")
        
        # Return to original directory in case of error
        try:
            os.chdir(original_cwd)
        except:
            pass
        
        return None

# Import the analyzer
ComprehensiveAnalyzer = "subprocess"

if ComprehensiveAnalyzer is None:
    print("❌ Could not import ComprehensiveAnalyzer")
    print("Make sure run_analysis.py is working correctly")
    sys.exit(1)

class BatchAnalyzer:
    """
    Batch analyzer for running comprehensive analysis across multiple stocks and parameters.
    
    This class orchestrates running your existing analysis pipeline across different
    combinations of stocks, hidden sizes, and sequence lengths to generate
    comparative performance data.
    """
    
    def __init__(self, config_path: str = "config/config2.yaml"):
        """
        Initialize batch analyzer.
        
        Args:
            config_path: Path to batch configuration file
        """
        self.config_path = config_path
        self.config = self.load_batch_config()
        self.results = []
        self.start_time = None
        
        # Create results directory structure
        # self.results_dir = self.config.get('output', {}).get('results_dir', 'results/batch_analysis')
        # os.makedirs(self.results_dir, exist_ok=True)
        # os.makedirs(f"{self.results_dir}/individual_runs", exist_ok=True)
        
        # Use standardized directory structure
        file_namer.ensure_directory_structure()
        self.results_dir = str(file_namer.base_dir / "01_training" / "batch_results")
        
        print(f"Batch analyzer initialized")
        print(f"Results will be saved to: {self.results_dir}")
    
    def load_batch_config(self) -> Dict:
        """Load batch configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Loaded batch configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            print(f"Batch config file {self.config_path} not found!")
            print("Creating default batch configuration...")
            return self.create_default_batch_config()
    
    def create_default_batch_config(self) -> Dict:
        """Create default batch configuration if file doesn't exist."""
        default_config = {
            'portfolios': {
                'tech': ['AAPL', 'MSFT', 'GOOGL', 'NVDA'],
                'finance': ['JPM', 'BAC', 'WFC', 'GS'],
                'market': ['^GSPC', 'QQQ', 'AGG', 'AAPL']
            },
            'parameters': {
                'hidden_sizes': [32, 50, 64],
                'sequence_lengths': [20, 30, 40]
            },
            'base_config': {
                'start_date': '2020-01-01',
                'end_date': '2024-12-31',
                'learning_rate': 0.001,
                'batch_size': 32,
                'num_epochs': 50,  # Reduced for batch analysis
                'patience': 10,
                'use_advanced_features': True,
                'pattern_analysis': True,
                'temporal_analysis': True,
                'dimensionality_reduction': True,
                'n_components_pca': 10,
                'umap_n_neighbors': 15
            },
            'output': {
                'results_dir': 'results/batch_analysis'
            }
        }
        
        # Save default config
        os.makedirs('config', exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        print(f"Created default batch config at {self.config_path}")
        return default_config
    
    def generate_parameter_combinations(self) -> List[Dict]:
        """
        Generate all combinations of parameters to test.
        
        Returns:
            List of parameter dictionaries to test
        """
        param_config = self.config.get('parameters', {})
        
        hidden_sizes = param_config.get('hidden_sizes', [50])
        sequence_lengths = param_config.get('sequence_lengths', [30])
        
        combinations = []
        for hidden_size in hidden_sizes:
            for seq_length in sequence_lengths:
                combinations.append({
                    'hidden_size': hidden_size,
                    'sequence_length': seq_length
                })
        
        print(f"Generated {len(combinations)} parameter combinations:")
        for i, combo in enumerate(combinations):
            print(f"  {i+1}: Hidden={combo['hidden_size']}, Sequence={combo['sequence_length']}")
        
        return combinations
    
    def get_portfolios_to_run(self, portfolio_filter: str = None) -> Dict[str, List[str]]:
        """
        Get portfolios to analyze with data validation.
        
        Args:
            portfolio_filter: If provided, only run this specific portfolio
        
        Returns:
            Dictionary of portfolio names to validated stock lists
        """
        all_portfolios = self.config.get('portfolios', {})
        base_config = self.config.get('base_config', {})
        start_date = base_config.get('start_date', '2020-01-01')
        end_date = base_config.get('end_date', '2024-12-31')
        
        if portfolio_filter:
            if portfolio_filter in all_portfolios:
                selected_portfolios = {portfolio_filter: all_portfolios[portfolio_filter]}
            else:
                print(f"Portfolio '{portfolio_filter}' not found in config!")
                print(f"Available portfolios: {list(all_portfolios.keys())}")
                return {}
        else:
            selected_portfolios = all_portfolios
        
        # For now, skip validation to make it faster - you can add this back if needed
        print(f"Portfolios to run: {list(selected_portfolios.keys())}")
        for name, stocks in selected_portfolios.items():
            print(f"  {name}: {stocks}")
        
        return selected_portfolios
    
    def create_run_config(self, 
                         portfolio_name: str, 
                         target_stock: str, 
                         context_stocks: List[str],
                         hidden_size: int, 
                         sequence_length: int) -> Dict:
        """
        Create configuration for a single analysis run.
        
        Args:
            portfolio_name: Name of the portfolio being analyzed
            target_stock: Stock to predict
            context_stocks: Additional stocks to use as features
            hidden_size: Number of hidden units in LNN
            sequence_length: Length of input sequences
        
        Returns:
            Configuration dictionary for run_analysis.py
        """
        base_config = self.config.get('base_config', {})
        
        # Create the stock list (target + context stocks, removing duplicates)
        all_stocks = [target_stock] + [s for s in context_stocks if s != target_stock]
        all_stocks = list(dict.fromkeys(all_stocks))  # Remove duplicates while preserving order
        
        run_config = {
            'data': {
                'tickers': all_stocks,
                'start_date': base_config.get('start_date', '2020-01-01'),
                'end_date': base_config.get('end_date', '2024-12-31'),
                'target_ticker': target_stock
            },
            'model': {
                'sequence_length': sequence_length,
                'hidden_size': hidden_size,
                'learning_rate': base_config.get('learning_rate', 0.001),
                'batch_size': base_config.get('batch_size', 32),
                'num_epochs': base_config.get('num_epochs', 50),  # Reduced for batch
                'patience': base_config.get('patience', 10)
            },
            'analysis': {
                'use_advanced_features': base_config.get('use_advanced_features', True),
                'n_components_pca': base_config.get('n_components_pca', 10),
                'umap_n_neighbors': base_config.get('umap_n_neighbors', 15),
                'pattern_analysis': base_config.get('pattern_analysis', True),
                'temporal_analysis': base_config.get('temporal_analysis', True),
                'dimensionality_reduction': base_config.get('dimensionality_reduction', True)
            }
        }
        
        return run_config

    def run_single_analysis(self, 
                          portfolio_name: str,
                          target_stock: str,
                          context_stocks: List[str],
                          hidden_size: int,
                          sequence_length: int,
                          run_index: int,
                          total_runs: int) -> Dict:
        """
        Run a single analysis configuration using subprocess.
        This avoids all import issues by calling run_analysis.py directly.
        
        Returns:
            Dictionary containing analysis results and metadata
        """
        run_start = time.time()
        
        # Create unique experiment name
        experiment_name = f"batch_{portfolio_name}_{target_stock}_h{hidden_size}_s{sequence_length}"
        
        print(f"\n{'='*80}")
        print(f"RUN {run_index}/{total_runs}: {experiment_name}")
        print(f"Portfolio: {portfolio_name}")
        print(f"Target: {target_stock}")
        print(f"Context: {context_stocks}")
        print(f"Hidden Size: {hidden_size}, Sequence Length: {sequence_length}")
        print(f"{'='*80}")
        
        try:
            # Create run configuration
            run_config = self.create_run_config(
                portfolio_name, target_stock, context_stocks,
                hidden_size, sequence_length
            )
            
            # Save temporary config file
            temp_config_path = f"config/temp_batch_config_{run_index}.yaml"
            os.makedirs('config', exist_ok=True)
            with open(temp_config_path, 'w') as f:
                yaml.dump(run_config, f)
            
            # Run analysis as subprocess
            cmd = [
                sys.executable,  # Use the same Python interpreter
                'scripts/run_analysis.py',
                '--config', temp_config_path,
                '--experiment-name', experiment_name
            ]
            
            print(f"Running command: {' '.join(cmd)}")
            
            # Execute the analysis
            result = subprocess.run(
                cmd,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # Project root
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            if result.returncode == 0:
                print("✓ Analysis completed successfully")
                
                # Try to extract results from STANDARDIZED LOCATIONS
                standardized_results_dir = 'results/01_training/analysis_reports'
                legacy_results_dir = 'results/reports'

                results_found = False
                analysis_results = {}
                metrics = {}  # Initialize metrics

                # CORRECTED: Search based on model parameters, not experiment name
                # The standardized files use format: analysis_TICKER_HIDDENH_SEQUENCES_TIMESTAMP.json
                search_pattern = f"analysis_{target_stock}_{hidden_size}H_{sequence_length}S_"
                
                print(f"Debug: Looking for files matching pattern: {search_pattern}")

                # First, try standardized location
                if os.path.exists(standardized_results_dir):
                    analysis_files = [f for f in os.listdir(standardized_results_dir) 
                                    if f.startswith(search_pattern) and f.endswith('.json')]

                    if analysis_files:
                        # Get the most recent file (highest timestamp)
                        latest_file = max(analysis_files, key=lambda x: os.path.getmtime(os.path.join(standardized_results_dir, x)))
                        results_path = os.path.join(standardized_results_dir, latest_file)

                        try:
                            with open(results_path, 'r') as f:
                                analysis_results = json.load(f)
                            metrics = self.extract_key_metrics(analysis_results)
                            results_found = True
                            print(f"✓ Found results in standardized location: {latest_file}")
                        except Exception as e:
                            print(f"Warning: Could not load results file: {e}")
                    else:
                        print(f"Debug: No files found matching pattern '{search_pattern}' in standardized dir")

                # Fallback to legacy location if not found (keep original logic for legacy)
                if not results_found and os.path.exists(legacy_results_dir):
                    analysis_files = [f for f in os.listdir(legacy_results_dir) 
                                    if f.startswith(f'analysis_data_{experiment_name}') and f.endswith('.json')]

                    if analysis_files:
                        latest_file = max(analysis_files, key=lambda x: os.path.getmtime(os.path.join(legacy_results_dir, x)))
                        results_path = os.path.join(legacy_results_dir, latest_file)

                        try:
                            with open(results_path, 'r') as f:
                                analysis_results = json.load(f)
                            metrics = self.extract_key_metrics(analysis_results)
                            results_found = True
                            print(f"✓ Found results in legacy location: {latest_file}")
                        except Exception as e:
                            print(f"Warning: Could not load results file: {e}")

                if not results_found:
                    print("Warning: No results file found in either location")
                    print(f"Debug: Expected pattern '{search_pattern}' not found")
                    print(f"Debug: Available files: {os.listdir(standardized_results_dir)if os.path.exists(standardized_results_dir) else 'None'}")
                    if os.path.exists(standardized_results_dir):
                        all_files = os.listdir(standardized_results_dir)
                        print(f"All files in standardized dir: {all_files}")
                        matching_files = [f for f in all_files if experiment_name in f]
                        print(f"Files containing '{experiment_name}': {matching_files}")
                    
                    print(f"Debug: Checking legacy dir: {legacy_results_dir}")
                    if os.path.exists(legacy_results_dir):
                        all_files = os.listdir(legacy_results_dir)
                        print(f"All files in legacy dir: {all_files}")
                        matching_files = [f for f in all_files if experiment_name in f]
                        print(f"Files containing '{experiment_name}': {matching_files}")
                
                # Compile result summary
                result_dict = {
                    'metadata': {
                        'portfolio_name': portfolio_name,
                        'target_stock': target_stock,
                        'context_stocks': context_stocks,
                        'hidden_size': hidden_size,
                        'sequence_length': sequence_length,
                        'experiment_name': experiment_name,
                        'run_time_seconds': time.time() - run_start,
                        'timestamp': datetime.now().isoformat()
                    },
                    'metrics': metrics,
                    'config': run_config,
                    'stdout': result.stdout[-1000:],  # Last 1000 chars of output
                    'stderr': result.stderr[-1000:] if result.stderr else ""
                }
                
            else:
                print(f"❌ Analysis failed with return code {result.returncode}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                result_dict = {
                    'metadata': {
                        'portfolio_name': portfolio_name,
                        'target_stock': target_stock,
                        'context_stocks': context_stocks,
                        'hidden_size': hidden_size,
                        'sequence_length': sequence_length,
                        'experiment_name': experiment_name,
                        'run_time_seconds': time.time() - run_start,
                        'timestamp': datetime.now().isoformat(),
                        'error': f"Process failed with code {result.returncode}"
                    },
                    'metrics': {},
                    'config': run_config,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            
            # Clean up temporary config
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
            
            return result_dict
            
        except subprocess.TimeoutExpired:
            print("❌ Analysis timed out after 30 minutes")
            return {
                'metadata': {
                    'portfolio_name': portfolio_name,
                    'target_stock': target_stock,
                    'context_stocks': context_stocks,
                    'hidden_size': hidden_size,
                    'sequence_length': sequence_length,
                    'experiment_name': experiment_name,
                    'run_time_seconds': time.time() - run_start,
                    'timestamp': datetime.now().isoformat(),
                    'error': "Timeout after 30 minutes"
                },
                'metrics': {},
                'config': {}
            }
            
        except Exception as e:
            print(f"❌ Run failed with error: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'metadata': {
                    'portfolio_name': portfolio_name,
                    'target_stock': target_stock,
                    'context_stocks': context_stocks,
                    'hidden_size': hidden_size,
                    'sequence_length': sequence_length,
                    'experiment_name': experiment_name,
                    'run_time_seconds': time.time() - run_start,
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                },
                'metrics': {},
                'config': {}
            }
            
    def extract_key_metrics(self, analysis_results: Dict) -> Dict:
        """
        Extract key performance metrics from analysis results.
        
        Args:
            analysis_results: Results from ComprehensiveAnalyzer
        
        Returns:
            Dictionary of key metrics for comparison
        """
        metrics = {}
        
        if not analysis_results:
            return metrics
        
        # Data statistics
        if 'data_statistics' in analysis_results:
            data_stats = analysis_results['data_statistics']
            
            # Get target ticker stats
            for ticker, stats in data_stats.items():
                if stats:  # Make sure stats exist
                    metrics['data'] = {
                        'total_return': stats.get('total_return', None),
                        'volatility': stats.get('return_std', None),
                        'sharpe_estimate': stats.get('sharpe_estimate', None)
                    }
                    break  # Use first available stats
        
        # Model training metrics
        if 'training' in analysis_results:
            training = analysis_results['training']
            metrics['training'] = {
                'final_val_loss': training.get('final_val_loss', None),
                'best_val_loss': training.get('best_val_loss', None),
                'total_epochs': training.get('total_epochs', None),
                'training_time': training.get('training_time', None)
            }
        
        # Model evaluation metrics
        if 'evaluation' in analysis_results:
            evaluation = analysis_results['evaluation']
            eval_metrics = evaluation.get('metrics', {})
            
            # Basic metrics
            if 'basic_metrics' in eval_metrics:
                basic = eval_metrics['basic_metrics']
                metrics['prediction'] = {
                    'rmse': basic.get('rmse', None),
                    'mae': basic.get('mae', None),
                    'mape': basic.get('mape', None),
                    'r2': basic.get('r2', None)
                }
            
            # Directional accuracy
            if 'directional_metrics' in eval_metrics:
                directional = eval_metrics['directional_metrics']
                metrics['directional'] = {
                    'accuracy': directional.get('directional_accuracy', None)
                }
            
            # Trading performance
            if 'trading_metrics' in eval_metrics:
                trading = eval_metrics['trading_metrics']
                metrics['trading'] = {
                    'strategy_return': trading.get('total_return', None),
                    'buy_hold_return': trading.get('buy_hold_return', None),
                    'sharpe_ratio': trading.get('sharpe_ratio', None),
                    'max_drawdown': trading.get('max_drawdown', None),
                    'win_rate': trading.get('win_rate', None)
                }
                
                # Calculate excess return
                if trading.get('total_return') and trading.get('buy_hold_return'):
                    metrics['trading']['excess_return'] = (
                        trading['total_return'] - trading['buy_hold_return']
                    )
        
        return metrics
    
    def run_batch_analysis(self, 
                          portfolio_filter: str = None, 
                          dry_run: bool = False,
                          quick: bool = False) -> str:
        """
        Run comprehensive batch analysis.
        
        Args:
            portfolio_filter: Optional filter to run only specific portfolio
            dry_run: If True, just show what would be run without executing
            quick: If True, run with reduced parameters for faster execution
        
        Returns:
            Path to results file
        """
        self.start_time = time.time()
        
        # Get portfolios to analyze
        portfolios = self.get_portfolios_to_run(portfolio_filter)
        
        if not portfolios:
            print("No portfolios to analyze!")
            return None
        
        # Get parameter combinations
        if quick:
            # For quick mode, use reduced parameter set
            param_combinations = [{'hidden_size': 50, 'sequence_length': 30}]
            print("Quick mode: Using single parameter combination")
        else:
            param_combinations = self.generate_parameter_combinations()
        
        # Calculate total runs
        total_runs = 0
        runs_to_execute = []
        
        for portfolio_name, stocks in portfolios.items():
            for target_stock in stocks:
                context_stocks = [s for s in stocks if s != target_stock]
                for params in param_combinations:
                    total_runs += 1
                    runs_to_execute.append({
                        'portfolio_name': portfolio_name,
                        'target_stock': target_stock,
                        'context_stocks': context_stocks,
                        'hidden_size': params['hidden_size'],
                        'sequence_length': params['sequence_length']
                    })
        
        print(f"\n{'='*80}")
        print(f"BATCH ANALYSIS PLAN")
        print(f"{'='*80}")
        print(f"Portfolios: {list(portfolios.keys())}")
        print(f"Parameter combinations: {len(param_combinations)}")
        print(f"Total runs planned: {total_runs}")
        print(f"Estimated time: {total_runs * 2:.0f}-{total_runs * 5:.0f} minutes")
        print(f"{'='*80}")
        
        if dry_run:
            print("\nDRY RUN - Showing planned runs:")
            for i, run in enumerate(runs_to_execute, 1):
                print(f"  {i:2d}: {run['portfolio_name']:10s} | {run['target_stock']:6s} | "
                      f"H={run['hidden_size']:3d} | S={run['sequence_length']:2d}")
            return None
        
        # Confirm execution
        if total_runs > 5:
            response = input(f"\nThis will run {total_runs} analyses. Continue? (y/N): ")
            if response.lower() != 'y':
                print("Batch analysis cancelled.")
                return None
        
        # Execute runs
        print(f"\nStarting batch analysis execution...")
        
        for i, run in enumerate(runs_to_execute, 1):
            result = self.run_single_analysis(
                run['portfolio_name'],
                run['target_stock'],
                run['context_stocks'],
                run['hidden_size'],
                run['sequence_length'],
                i,
                total_runs
            )
            
            self.results.append(result)
            
            # Save intermediate results every 3 runs
            if i % 3 == 0:
                self.save_intermediate_results(i)
        
        # Save final results
        results_path = self.save_final_results()
        
        # Generate summary
        self.generate_summary_report()
        
        total_time = time.time() - self.start_time
        print(f"\n{'='*80}")
        print(f"BATCH ANALYSIS COMPLETED!")
        print(f"{'='*80}")
        print(f"Total runs: {len(self.results)}")
        print(f"Successful runs: {sum(1 for r in self.results if 'error' not in r['metadata'])}")
        print(f"Failed runs: {sum(1 for r in self.results if 'error' in r['metadata'])}")
        print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Results saved to: {results_path}")
        print(f"{'='*80}")
        
        return results_path
    
    def save_intermediate_results(self, run_number: int):
        """Save intermediate results during batch run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"intermediate_results_run_{run_number}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"  → Saved intermediate results to {filename}")
    
    def save_final_results(self, portfolio_name: str) -> str:
        """Save final batch analysis results."""
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # filename = f"batch_analysis_results_{timestamp}.json"
        # filepath = os.path.join(self.results_dir, filename)
        
        batch_paths = file_namer.create_batch_analysis_paths(portfolio_name)
        
        # Prepare complete results package
        results_package = {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'total_runs': len(self.results),
                'successful_runs': sum(1 for r in self.results if 'error' not in r['metadata']),
                'failed_runs': sum(1 for r in self.results if 'error' in r['metadata']),
                'total_time_seconds': time.time() - self.start_time if self.start_time else None,
                'config_file': self.config_path
            },
            'config': self.config,
            'results': self.results
        }
        
        # Save detailed JSON
        with open(batch_paths['detailed_json'], 'w') as f:
            json.dump(results_package, f, indent=2, default=str)
    
        # Create and save summary CSV
        summary_df = self.create_summary_dataframe()
        summary_df.to_csv(batch_paths['summary_csv'], index=False)
    
        return batch_paths['detailed_json']
        
        # return filepath
    
    def generate_summary_report(self):
        """Generate a summary report of batch analysis results."""
        if not self.results:
            return
        
        # Filter successful results
        successful_results = [r for r in self.results if 'error' not in r['metadata']]
        
        if not successful_results:
            print("No successful results to summarize.")
            return
        
        # Create summary DataFrame
        summary_data = []
        
        for result in successful_results:
            meta = result['metadata']
            metrics = result['metrics']
            
            row = {
                'portfolio': meta['portfolio_name'],
                'target_stock': meta['target_stock'],
                'hidden_size': meta['hidden_size'],
                'sequence_length': meta['sequence_length'],
                'run_time': meta['run_time_seconds']
            }
            
            # Add trading metrics if available
            if 'trading' in metrics:
                trading = metrics['trading']
                row.update({
                    'strategy_return': trading.get('strategy_return'),
                    'buy_hold_return': trading.get('buy_hold_return'),
                    'excess_return': trading.get('excess_return'),
                    'sharpe_ratio': trading.get('sharpe_ratio'),
                    'max_drawdown': trading.get('max_drawdown')
                })
            
            # Add prediction metrics if available
            if 'prediction' in metrics:
                pred = metrics['prediction']
                row.update({
                    'rmse': pred.get('rmse'),
                    'r2': pred.get('r2'),
                    'mape': pred.get('mape')
                })
            
            if 'directional' in metrics:
                row['directional_accuracy'] = metrics['directional'].get('accuracy')
            
            summary_data.append(row)
        
        if not summary_data:
            print("No summary data to report.")
            return
        
        # Create DataFrame and save
        summary_df = pd.DataFrame(summary_data)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(self.results_dir, f"batch_summary_{timestamp}.csv")
        summary_df.to_csv(summary_path, index=False)
        
        # Print summary statistics
        print(f"\n{'='*60}")
        print("BATCH ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Total successful runs: {len(summary_df)}")
        print(f"Portfolios analyzed: {summary_df['portfolio'].nunique()}")
        print(f"Stocks analyzed: {summary_df['target_stock'].nunique()}")
        print(f"Average run time: {summary_df['run_time'].mean():.1f} seconds")
        
        # Show best performing configurations
        numeric_columns = summary_df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 3:  # More than just run_time and dimensions
            print(f"\nTop 3 configurations by available metrics:")
            for col in ['excess_return', 'sharpe_ratio', 'directional_accuracy', 'r2']:
                if col in summary_df.columns:
                    top_3 = summary_df.nlargest(3, col)[
                        ['portfolio', 'target_stock', 'hidden_size', 'sequence_length', col]
                    ]
                    print(f"\nBy {col}:")
                    print(top_3.to_string(index=False, float_format='%.3f'))
                    break
        
        print(f"\nSummary saved to: {summary_path}")

def main():
    """Main function for batch analysis."""
    parser = argparse.ArgumentParser(description='Run batch analysis across multiple stocks and parameters')
    
    parser.add_argument('--config', type=str, default='config/config2.yaml',
                      help='Path to batch configuration file')
    parser.add_argument('--portfolio', type=str, default=None,
                      help='Run only specific portfolio (e.g., "tech", "finance")')
    parser.add_argument('--dry-run', action='store_true',
                      help='Show planned runs without executing')
    parser.add_argument('--quick', action='store_true',
                      help='Run quick analysis with reduced parameters')
    
    args = parser.parse_args()
    
    print("="*80)
    print("LNN BATCH ANALYSIS SYSTEM")
    print("="*80)
    print(f"Configuration: {args.config}")
    if args.portfolio:
        print(f"Portfolio filter: {args.portfolio}")
    if args.dry_run:
        print("Mode: DRY RUN (no actual execution)")
    if args.quick:
        print("Mode: QUICK (reduced parameters)")
    print("="*80)
    
    try:
        # Initialize batch analyzer
        batch_analyzer = BatchAnalyzer(config_path=args.config)
        
        # Run batch analysis
        results_path = batch_analyzer.run_batch_analysis(
            portfolio_filter=args.portfolio,
            dry_run=args.dry_run,
            quick=args.quick
        )
        
        if results_path:
            print(f"\n🎉 Batch analysis completed successfully!")
            print(f"Results saved to: {results_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Batch analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during batch analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
