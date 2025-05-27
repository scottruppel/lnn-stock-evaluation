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

# Add src to Python path - handle different project structures
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
possible_paths = [
    os.path.join(project_root, 'src'),
    os.path.join(project_root, 'scripts'),
    os.path.join(os.getcwd(), 'src'),
    os.path.join(os.getcwd(), 'scripts'),
    project_root,
    os.getcwd()
]

for path in possible_paths:
    if path not in sys.path:
        sys.path.insert(0, path)

print(f"Python paths added: {[p for p in possible_paths if os.path.exists(p)]}")

# Import your existing analysis pipeline with multiple fallbacks
ComprehensiveAnalyzer = None

# Try different import methods
import_attempts = [
    lambda: __import__('run_analysis', fromlist=['ComprehensiveAnalyzer']).ComprehensiveAnalyzer,
    lambda: __import__('scripts.run_analysis', fromlist=['ComprehensiveAnalyzer']).ComprehensiveAnalyzer,
]

for attempt in import_attempts:
    try:
        ComprehensiveAnalyzer = attempt()
        print(f"✅ Successfully imported ComprehensiveAnalyzer")
        break
    except ImportError as e:
        print(f"Import attempt failed: {e}")
        continue

if ComprehensiveAnalyzer is None:
    print("❌ Could not import ComprehensiveAnalyzer")
    print("Make sure run_analysis.py is in the scripts/ directory")
    print("Available files in scripts/:")
    scripts_dir = os.path.join(os.getcwd(), 'scripts')
    if os.path.exists(scripts_dir):
        for f in os.listdir(scripts_dir):
            if f.endswith('.py'):
                print(f"  {f}")
    print("Available files in current directory:")
    for f in os.listdir('.'):
        if f.endswith('.py') and 'analysis' in f:
            print(f"  {f}")
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
        self.results_dir = self.config.get('output', {}).get('results_dir', 'results/batch_analysis')
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(f"{self.results_dir}/individual_runs", exist_ok=True)
        
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
            raise
    
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
        
        # Validate data for each portfolio
        validated_portfolios = {}
        
        for name, stocks in selected_portfolios.items():
            if not stocks or len(stocks) == 0:
                continue
                
            print(f"\nValidating portfolio: {name}")
            valid_stocks, invalid_stocks = self.validate_stock_data(stocks, start_date, end_date)
            
            if len(valid_stocks) < 2:  # Need at least 2 stocks for analysis
                print(f"  ⚠️  Portfolio '{name}' has insufficient valid stocks ({len(valid_stocks)}). Skipping.")
                if invalid_stocks:
                    print(f"     Invalid stocks: {', '.join(invalid_stocks)}")
                continue
            
            validated_portfolios[name] = valid_stocks
            
            if invalid_stocks:
                print(f"  ⚠️  Removed {len(invalid_stocks)} stocks with insufficient data: {', '.join(invalid_stocks)}")
            
            print(f"  ✅ Portfolio '{name}': {len(valid_stocks)} valid stocks")
        
        return validated_portfolios
    
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
                'num_epochs': base_config.get('num_epochs', 100),
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
    
    def validate_stock_data(self, stocks: List[str], start_date: str, end_date: str) -> Tuple[List[str], List[str]]:
        """
        Validate that stocks have sufficient data for the analysis period.
        
        Args:
            stocks: List of stock tickers to validate
            start_date: Analysis start date
            end_date: Analysis end date
            
        Returns:
            Tuple of (valid_stocks, invalid_stocks)
        """
        import yfinance as yf
        from datetime import datetime, timedelta
        
        valid_stocks = []
        invalid_stocks = []
        
        # Calculate minimum data points needed (roughly 2 years of data minimum)
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        min_date = start_dt + timedelta(days=730)  # Need at least 2 years of data
        
        print(f"Validating data availability for {len(stocks)} stocks...")
        
        for stock in stocks:
            try:
                # Download a small sample to check data availability
                ticker = yf.Ticker(stock)
                hist = ticker.history(start=start_date, end=end_date)
                
                if len(hist) < 500:  # Less than ~2 years of trading days
                    print(f"  ❌ {stock}: Insufficient data ({len(hist)} days)")
                    invalid_stocks.append(stock)
                elif hist.index[0].date() > min_date.date():
                    print(f"  ❌ {stock}: Data starts too late ({hist.index[0].date()})")
                    invalid_stocks.append(stock)
                else:
                    print(f"  ✅ {stock}: Valid ({len(hist)} days from {hist.index[0].date()})")
                    valid_stocks.append(stock)
                    
            except Exception as e:
                print(f"  ❌ {stock}: Error downloading data - {e}")
                invalid_stocks.append(stock)
        
        return valid_stocks, invalid_stocks

    def run_single_analysis(self, 
                          portfolio_name: str,
                          target_stock: str,
                          context_stocks: List[str],
                          hidden_size: int,
                          sequence_length: int,
                          run_index: int,
                          total_runs: int) -> Dict:
        """
        Run a single analysis configuration.
        
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
            
            # Initialize and run analyzer
            analyzer = ComprehensiveAnalyzer(
                config_path=temp_config_path,
                experiment_name=experiment_name
            )
            
            # Run the analysis pipeline
            analysis_results = analyzer.run_complete_analysis()
            
            # Extract key metrics
            metrics = self.extract_key_metrics(analysis_results)
            
            # Compile result summary
            result = {
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
                'config': run_config
            }
            
            # Clean up temporary config
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
            
            print(f"✓ Run completed successfully in {time.time() - run_start:.1f} seconds")
            return result
            
        except Exception as e:
            print(f"❌ Run failed with error: {e}")
            import traceback
            traceback.print_exc()
            
            # Return error result
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
        
        # Data statistics
        if 'data_statistics' in analysis_results:
            data_stats = analysis_results['data_statistics']
            target_ticker = None
            
            # Find the target ticker stats
            for ticker, stats in data_stats.items():
                if ticker in ['^GSPC', 'AAPL', 'QQQ', 'AGG']:  # Look for likely target
                    target_ticker = ticker
                    break
            
            if target_ticker and target_ticker in data_stats:
                stats = data_stats[target_ticker]
                metrics['data'] = {
                    'total_return': stats.get('total_return', None),
                    'volatility': stats.get('return_std', None),
                    'sharpe_estimate': stats.get('sharpe_estimate', None)
                }
        
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
        
        # Pattern analysis
        if 'patterns' in analysis_results:
            patterns = analysis_results['patterns']
            trend_info = patterns.get('trend', {})
            metrics['patterns'] = {
                'trend_direction': trend_info.get('direction', None),
                'trend_strength': trend_info.get('strength', None)
            }
        
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
        print(f"Estimated time: {total_runs * 3:.0f}-{total_runs * 8:.0f} minutes")
        print(f"{'='*80}")
        
        if dry_run:
            print("\nDRY RUN - Showing planned runs:")
            for i, run in enumerate(runs_to_execute, 1):
                print(f"  {i:2d}: {run['portfolio_name']:10s} | {run['target_stock']:6s} | "
                      f"H={run['hidden_size']:3d} | S={run['sequence_length']:2d}")
            return None
        
        # Confirm execution
        if total_runs > 10:
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
            
            # Save intermediate results every 5 runs
            if i % 5 == 0:
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
    
    def save_final_results(self) -> str:
        """Save final batch analysis results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_analysis_results_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
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
        
        with open(filepath, 'w') as f:
            json.dump(results_package, f, indent=2, default=str)
        
        return filepath
    
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
        
        # Create DataFrame and save
        summary_df = pd.DataFrame(summary_data)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(self.results_dir, f"batch_summary_{timestamp}.csv")
        summary_df.to_csv(summary_path, index=False)
        
        # Print top performers
        print(f"\n{'='*60}")
        print("BATCH ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        if 'excess_return' in summary_df.columns:
            print("\nTOP 5 PERFORMERS BY EXCESS RETURN:")
            top_excess = summary_df.nlargest(5, 'excess_return')[
                ['portfolio', 'target_stock', 'hidden_size', 'sequence_length', 'excess_return', 'sharpe_ratio']
            ]
            print(top_excess.to_string(index=False, float_format='%.3f'))
        
        if 'sharpe_ratio' in summary_df.columns:
            print("\nTOP 5 PERFORMERS BY SHARPE RATIO:")
            top_sharpe = summary_df.nlargest(5, 'sharpe_ratio')[
                ['portfolio', 'target_stock', 'hidden_size', 'sequence_length', 'sharpe_ratio', 'excess_return']
            ]
            print(top_sharpe.to_string(index=False, float_format='%.3f'))
        
        if 'directional_accuracy' in summary_df.columns:
            print("\nTOP 5 PERFORMERS BY DIRECTIONAL ACCURACY:")
            top_acc = summary_df.nlargest(5, 'directional_accuracy')[
                ['portfolio', 'target_stock', 'hidden_size', 'sequence_length', 'directional_accuracy', 'r2']
            ]
            print(top_acc.to_string(index=False, float_format='%.3f'))
        
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
