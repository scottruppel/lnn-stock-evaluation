#!/usr/bin/env python3
"""
Batch Results Analysis Script
Analyzes and visualizes results from batch_analysis.py runs.

This script helps you understand which parameter combinations work best
across different stocks and portfolios.

Usage:
    python scripts/analyze_batch_results.py results/batch_analysis/batch_analysis_results_TIMESTAMP.json
    python scripts/analyze_batch_results.py --latest                    # Analyze most recent results
    python scripts/analyze_batch_results.py --compare file1.json file2.json  # Compare two result sets
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np

# Force matplotlib to use non-interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class BatchResultsAnalyzer:
    """
    Analyzer for batch analysis results.
    Provides comprehensive analysis and visualization of parameter optimization results.
    """
    
    def __init__(self):
        self.results_data = None
        self.results_df = None
        self.successful_results = None
        
    def load_results(self, results_path: str) -> bool:
        """
        Load batch analysis results from JSON file.
        
        Args:
            results_path: Path to batch results JSON file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(results_path, 'r') as f:
                self.results_data = json.load(f)
            
            print(f"Loaded results from: {results_path}")
            print(f"Total runs: {self.results_data['metadata']['total_runs']}")
            print(f"Successful runs: {self.results_data['metadata']['successful_runs']}")
            print(f"Failed runs: {self.results_data['metadata']['failed_runs']}")
            
            # Convert to DataFrame for easier analysis
            self.results_df = self._create_results_dataframe()
            self.successful_results = self.results_df[~self.results_df['has_error']].copy()
            
            print(f"DataFrame created with {len(self.successful_results)} successful results")
            return True
            
        except Exception as e:
            print(f"Error loading results: {e}")
            return False
    
    def _create_results_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame for analysis."""
        rows = []
        
        for result in self.results_data['results']:
            meta = result['metadata']
            metrics = result['metrics']
            
            row = {
                # Metadata
                'portfolio': meta['portfolio_name'],
                'target_stock': meta['target_stock'],
                'hidden_size': meta['hidden_size'],
                'sequence_length': meta['sequence_length'],
                'experiment_name': meta['experiment_name'],
                'run_time': meta['run_time_seconds'],
                'timestamp': meta['timestamp'],
                'has_error': 'error' in meta,
                'error_message': meta.get('error', ''),
                
                # Initialize all metrics as None
                'strategy_return': None,
                'buy_hold_return': None,
                'excess_return': None,
                'sharpe_ratio': None,
                'max_drawdown': None,
                'directional_accuracy': None,
                'rmse': None,
                'r2': None,
                'mape': None,
                'final_val_loss': None,
                'total_epochs': None,
                'data_total_return': None,
                'data_volatility': None,
                'trend_direction': None,
                'trend_strength': None
            }
            
            # Helper function to safely convert to numeric
            def safe_numeric(value):
                """Safely convert value to numeric, handling various data types."""
                if value is None:
                    return None
                
                # If it's already a number, return it
                if isinstance(value, (int, float)):
                    return float(value)
                
                # If it's a string, try to convert
                if isinstance(value, str):
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        # If conversion fails, return None
                        return None
                
                # If it's a list or numpy array, take the first element
                if isinstance(value, (list, np.ndarray)):
                    if len(value) > 0:
                        return safe_numeric(value[0])
                    else:
                        return None
                
                # For any other type, try direct float conversion
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return None
            
            # Extract metrics if available with safe conversion
            if 'trading' in metrics:
                trading = metrics['trading']
                row.update({
                    'strategy_return': safe_numeric(trading.get('strategy_return')),
                    'buy_hold_return': safe_numeric(trading.get('buy_hold_return')),
                    'excess_return': safe_numeric(trading.get('excess_return')),
                    'sharpe_ratio': safe_numeric(trading.get('sharpe_ratio')),
                    'max_drawdown': safe_numeric(trading.get('max_drawdown')),
                })
            
            if 'directional' in metrics:
                row['directional_accuracy'] = safe_numeric(metrics['directional'].get('accuracy'))
            
            if 'prediction' in metrics:
                pred = metrics['prediction']
                row.update({
                    'rmse': safe_numeric(pred.get('rmse')),
                    'r2': safe_numeric(pred.get('r2')),
                    'mape': safe_numeric(pred.get('mape')),
                })
            
            if 'training' in metrics:
                training = metrics['training']
                row.update({
                    'final_val_loss': safe_numeric(training.get('final_val_loss')),
                    'total_epochs': safe_numeric(training.get('total_epochs')),
                })
            
            if 'data' in metrics:
                data = metrics['data']
                row.update({
                    'data_total_return': safe_numeric(data.get('total_return')),
                    'data_volatility': safe_numeric(data.get('volatility')),
                })
            
            if 'patterns' in metrics:
                patterns = metrics['patterns']
                row.update({
                    'trend_direction': patterns.get('trend_direction'),  # Keep as string
                    'trend_strength': safe_numeric(patterns.get('trend_strength')),
                })
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Ensure numeric columns are properly typed
        numeric_columns = [
            'strategy_return', 'buy_hold_return', 'excess_return', 'sharpe_ratio', 
            'max_drawdown', 'directional_accuracy', 'rmse', 'r2', 'mape', 
            'final_val_loss', 'total_epochs', 'data_total_return', 
            'data_volatility', 'trend_strength', 'run_time'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def generate_comprehensive_report(self, output_dir: str = None):
        """Generate comprehensive analysis report with visualizations."""
        if self.successful_results is None or len(self.successful_results) == 0:
            print("No successful results to analyze!")
            return
        
        if output_dir is None:
            output_dir = "results/batch_analysis/analysis_reports"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("Generating comprehensive analysis report...")
        
        # 1. Parameter Performance Analysis
        self._analyze_parameter_performance(output_dir, timestamp)
        
        # 2. Portfolio Performance Comparison
        self._analyze_portfolio_performance(output_dir, timestamp)
        
        # 3. Stock-specific Analysis
        self._analyze_stock_performance(output_dir, timestamp)
        
        # 4. Generate summary statistics
        self._generate_summary_statistics(output_dir, timestamp)
        
        # 5. Create visualizations
        self._create_visualizations(output_dir, timestamp)
        
        print(f"Analysis complete! Reports saved to: {output_dir}")
    
    def _analyze_parameter_performance(self, output_dir: str, timestamp: str):
        """Analyze how different parameter combinations perform."""
        print("Analyzing parameter performance...")
        
        try:
            # Check what data we have
            print(f"  Successful results: {len(self.successful_results)}")
            print(f"  Columns available: {list(self.successful_results.columns)}")
            
            # Check for numeric columns
            numeric_cols = ['excess_return', 'sharpe_ratio', 'directional_accuracy', 'r2', 'run_time']
            available_cols = {}
            
            for col in numeric_cols:
                if col in self.successful_results.columns:
                    non_null_count = self.successful_results[col].notna().sum()
                    print(f"  {col}: {non_null_count} non-null values")
                    if non_null_count > 0:
                        available_cols[col] = ['mean', 'std', 'count']
                else:
                    print(f"  {col}: column not found")
            
            if not available_cols:
                print("  ⚠️  No numeric metrics available for parameter analysis")
                return
            
            # Group by parameter combinations
            param_groups = self.successful_results.groupby(['hidden_size', 'sequence_length'])
            
            # Only aggregate columns that have data
            param_summary = param_groups.agg(available_cols).round(4)
            
            # Flatten column names
            param_summary.columns = ['_'.join(col).strip() for col in param_summary.columns]
            param_summary = param_summary.reset_index()
            
            # Save parameter analysis
            param_path = os.path.join(output_dir, f"parameter_analysis_{timestamp}.csv")
            param_summary.to_csv(param_path, index=False)
            print(f"  ✅ Parameter analysis saved to: {param_path}")
            
            # Find best parameter combinations
            best_params = {}
            metrics = [col for col in param_summary.columns if col.endswith('_mean')]
            
            for metric in metrics:
                if metric in param_summary.columns and param_summary[metric].notna().any():
                    best_idx = param_summary[metric].idxmax()
                    best_params[metric] = {
                        'hidden_size': int(param_summary.loc[best_idx, 'hidden_size']),
                        'sequence_length': int(param_summary.loc[best_idx, 'sequence_length']),
                        'value': float(param_summary.loc[best_idx, metric])
                    }
            
            # Save best parameters with custom JSON encoder
            if best_params:
                best_params_path = os.path.join(output_dir, f"best_parameters_{timestamp}.json")
                
                # Convert numpy types to native Python types
                def convert_numpy_types(obj):
                    if isinstance(obj, dict):
                        return {key: convert_numpy_types(value) for key, value in obj.items()}
                    elif isinstance(obj, (np.integer, np.int64)):
                        return int(obj)
                    elif isinstance(obj, (np.floating, np.float64)):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    else:
                        return obj
                
                best_params_clean = convert_numpy_types(best_params)
                
                with open(best_params_path, 'w') as f:
                    json.dump(best_params_clean, f, indent=2)
                print(f"  ✅ Best parameters saved to: {best_params_path}")
            
        except Exception as e:
            print(f"  ❌ Error in parameter analysis: {e}")
            import traceback
            traceback.print_exc()
    
    def _analyze_portfolio_performance(self, output_dir: str, timestamp: str):
        """Analyze performance across different portfolios."""
        print("Analyzing portfolio performance...")
        
        portfolio_summary = self.successful_results.groupby('portfolio').agg({
            'excess_return': ['mean', 'std', 'count'],
            'sharpe_ratio': ['mean', 'std'],
            'directional_accuracy': ['mean', 'std'],
            'r2': ['mean', 'std'],
            'data_total_return': ['mean', 'std'],
            'data_volatility': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        portfolio_summary.columns = ['_'.join(col).strip() for col in portfolio_summary.columns]
        portfolio_summary = portfolio_summary.reset_index()
        
        # Save portfolio analysis
        portfolio_path = os.path.join(output_dir, f"portfolio_analysis_{timestamp}.csv")
        portfolio_summary.to_csv(portfolio_path, index=False)
    
    def _analyze_stock_performance(self, output_dir: str, timestamp: str):
        """Analyze performance of individual stocks."""
        print("Analyzing individual stock performance...")
        
        stock_summary = self.successful_results.groupby(['portfolio', 'target_stock']).agg({
            'excess_return': ['mean', 'std', 'min', 'max'],
            'sharpe_ratio': ['mean', 'std', 'min', 'max'],
            'directional_accuracy': ['mean', 'std', 'min', 'max'],
            'r2': ['mean', 'std', 'min', 'max'],
        }).round(4)
        
        # Flatten column names
        stock_summary.columns = ['_'.join(col).strip() for col in stock_summary.columns]
        stock_summary = stock_summary.reset_index()
        
        # Save stock analysis
        stock_path = os.path.join(output_dir, f"stock_analysis_{timestamp}.csv")
        stock_summary.to_csv(stock_path, index=False)
        
        # Find top performing stocks
        top_stocks = {}
        metrics = ['excess_return_mean', 'sharpe_ratio_mean', 'directional_accuracy_mean', 'r2_mean']
        
        for metric in metrics:
            if metric in stock_summary.columns:
                top_5 = stock_summary.nlargest(5, metric)[['portfolio', 'target_stock', metric]]
                top_stocks[metric] = top_5.to_dict('records')
        
        # Save top stocks
        top_stocks_path = os.path.join(output_dir, f"top_performing_stocks_{timestamp}.json")
        with open(top_stocks_path, 'w') as f:
            json.dump(top_stocks, f, indent=2)
    
    def _generate_summary_statistics(self, output_dir: str, timestamp: str):
        """Generate overall summary statistics."""
        print("Generating summary statistics...")
        
        summary_stats = {}
        
        # Overall performance statistics
        numeric_columns = ['excess_return', 'sharpe_ratio', 'directional_accuracy', 'r2', 'run_time']
        
        for col in numeric_columns:
            if col in self.successful_results.columns:
                col_data = self.successful_results[col].dropna()
                if len(col_data) > 0:
                    summary_stats[col] = {
                        'count': len(col_data),
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'q25': float(col_data.quantile(0.25)),
                        'q50': float(col_data.quantile(0.50)),
                        'q75': float(col_data.quantile(0.75))
                    }
        
        # Performance distribution
        if 'excess_return' in self.successful_results.columns:
            excess_returns = self.successful_results['excess_return'].dropna()
            summary_stats['performance_distribution'] = {
                'positive_excess_return_count': int((excess_returns > 0).sum()),
                'positive_excess_return_pct': float((excess_returns > 0).mean() * 100),
                'excellent_performance_count': int((excess_returns > 0.15).sum()),
                'good_performance_count': int((excess_returns > 0.05).sum()),
                'poor_performance_count': int((excess_returns < -0.05).sum())
            }
        
        # Save summary statistics
        summary_path = os.path.join(output_dir, f"summary_statistics_{timestamp}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
    
    def _create_visualizations(self, output_dir: str, timestamp: str):
        """Create comprehensive visualizations."""
        print("Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Parameter heatmap for excess return
        plt.subplot(3, 3, 1)
        if 'excess_return' in self.successful_results.columns:
            param_pivot = self.successful_results.pivot_table(
                values='excess_return', 
                index='hidden_size', 
                columns='sequence_length', 
                aggfunc='mean'
            )
            sns.heatmap(param_pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0)
            plt.title('Excess Return by Parameters')
        
        # 2. Parameter heatmap for Sharpe ratio
        plt.subplot(3, 3, 2)
        if 'sharpe_ratio' in self.successful_results.columns:
            sharpe_pivot = self.successful_results.pivot_table(
                values='sharpe_ratio', 
                index='hidden_size', 
                columns='sequence_length', 
                aggfunc='mean'
            )
            sns.heatmap(sharpe_pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=1.0)
            plt.title('Sharpe Ratio by Parameters')
        
        # 3. Portfolio performance comparison
        plt.subplot(3, 3, 3)
        if 'excess_return' in self.successful_results.columns:
            portfolio_data = self.successful_results.groupby('portfolio')['excess_return'].mean().sort_values(ascending=True)
            portfolio_data.plot(kind='barh')
            plt.title('Average Excess Return by Portfolio')
            plt.xlabel('Excess Return')
        
        # 4. Distribution of excess returns
        plt.subplot(3, 3, 4)
        if 'excess_return' in self.successful_results.columns:
            self.successful_results['excess_return'].hist(bins=30, alpha=0.7)
            plt.axvline(0, color='red', linestyle='--', label='Break-even')
            plt.title('Distribution of Excess Returns')
            plt.xlabel('Excess Return')
            plt.ylabel('Frequency')
            plt.legend()
        
        # 5. Sharpe ratio vs excess return scatter
        plt.subplot(3, 3, 5)
        if 'sharpe_ratio' in self.successful_results.columns and 'excess_return' in self.successful_results.columns:
            scatter_data = self.successful_results[['excess_return', 'sharpe_ratio']].dropna()
            plt.scatter(scatter_data['excess_return'], scatter_data['sharpe_ratio'], alpha=0.6)
            plt.xlabel('Excess Return')
            plt.ylabel('Sharpe Ratio')
            plt.title('Sharpe Ratio vs Excess Return')
            plt.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Sharpe = 1.0')
            plt.axvline(0, color='red', linestyle='--', alpha=0.5, label='Break-even')
            plt.legend()
        
        # 6. Directional accuracy distribution
        plt.subplot(3, 3, 6)
        if 'directional_accuracy' in self.successful_results.columns:
            self.successful_results['directional_accuracy'].hist(bins=20, alpha=0.7)
            plt.axvline(0.5, color='red', linestyle='--', label='Random (50%)')
            plt.title('Distribution of Directional Accuracy')
            plt.xlabel('Directional Accuracy')
            plt.ylabel('Frequency')
            plt.legend()
        
        # 7. Model performance metrics by portfolio
        plt.subplot(3, 3, 7)
        if 'r2' in self.successful_results.columns:
            portfolio_r2 = self.successful_results.groupby('portfolio')['r2'].mean().sort_values(ascending=True)
            portfolio_r2.plot(kind='barh')
            plt.title('Average R² by Portfolio')
            plt.xlabel('R² Score')
        
        # 8. Runtime analysis
        plt.subplot(3, 3, 8)
        if 'run_time' in self.successful_results.columns:
            runtime_by_params = self.successful_results.groupby(['hidden_size', 'sequence_length'])['run_time'].mean()
            runtime_by_params.plot(kind='bar')
            plt.title('Average Runtime by Parameters')
            plt.ylabel('Runtime (seconds)')
            plt.xticks(rotation=45)
        
        # 9. Success rate by parameters
        plt.subplot(3, 3, 9)
        all_results_df = self.results_df
        success_rate = all_results_df.groupby(['hidden_size', 'sequence_length']).apply(
            lambda x: (~x['has_error']).mean()
        ).reset_index()
        success_rate.columns = ['hidden_size', 'sequence_length', 'success_rate']
        success_pivot = success_rate.pivot(
            index='hidden_size', 
            columns='sequence_length', 
            values='success_rate'
        )
        sns.heatmap(success_pivot, annot=True, fmt='.2f', cmap='RdYlGn')
        plt.title('Success Rate by Parameters')
        
        plt.tight_layout()
        
        # Save the comprehensive plot
        plot_path = os.path.join(output_dir, f"comprehensive_analysis_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to: {plot_path}")
    
    def find_optimal_parameters(self, metric: str = 'excess_return') -> Dict:
        """Find optimal parameters based on specified metric."""
        if self.successful_results is None or len(self.successful_results) == 0:
            return {}
        
        if metric not in self.successful_results.columns:
            print(f"Metric '{metric}' not found in results")
            return {}
        
        # Group by parameters and calculate mean performance
        param_performance = self.successful_results.groupby(['hidden_size', 'sequence_length'])[metric].agg(['mean', 'std', 'count']).reset_index()
        
        # Find the best performing combination
        best_idx = param_performance['mean'].idxmax()
        optimal_params = {
            'hidden_size': int(param_performance.loc[best_idx, 'hidden_size']),
            'sequence_length': int(param_performance.loc[best_idx, 'sequence_length']),
            'mean_performance': float(param_performance.loc[best_idx, 'mean']),
            'std_performance': float(param_performance.loc[best_idx, 'std']),
            'num_runs': int(param_performance.loc[best_idx, 'count']),
            'metric_used': metric
        }
        
        return optimal_params

def find_latest_results_file(results_dir: str = "results/batch_analysis") -> Optional[str]:
    """Find the most recent batch analysis results file."""
    if not os.path.exists(results_dir):
        return None
    
    result_files = [f for f in os.listdir(results_dir) if f.startswith('batch_analysis_results_') and f.endswith('.json')]
    
    if not result_files:
        return None
    
    # Sort by modification time and return the most recent
    result_files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
    return os.path.join(results_dir, result_files[0])

def main():
    """Main function for batch results analysis."""
    parser = argparse.ArgumentParser(description='Analyze batch analysis results')
    
    parser.add_argument('results_file', nargs='?', 
                      help='Path to batch results JSON file')
    parser.add_argument('--latest', action='store_true',
                      help='Analyze the most recent results file')
    parser.add_argument('--output-dir', type=str, 
                      default='results/batch_analysis/analysis_reports',
                      help='Directory for analysis outputs')
    parser.add_argument('--metric', type=str, default='excess_return',
                      choices=['excess_return', 'sharpe_ratio', 'directional_accuracy', 'r2'],
                      help='Metric to optimize for')
    
    args = parser.parse_args()
    
    # Determine which results file to analyze
    if args.latest:
        results_file = find_latest_results_file()
        if not results_file:
            print("No batch analysis results found!")
            return
    elif args.results_file:
        results_file = args.results_file
    else:
        print("Please specify a results file or use --latest")
        return
    
    print("="*80)
    print("BATCH RESULTS ANALYSIS")
    print("="*80)
    print(f"Results file: {results_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Optimization metric: {args.metric}")
    print("="*80)
    
    try:
        # Initialize analyzer and load results
        analyzer = BatchResultsAnalyzer()
        
        if not analyzer.load_results(results_file):
            return
        
        # Generate comprehensive analysis
        analyzer.generate_comprehensive_report(args.output_dir)
        
        # Find optimal parameters
        optimal_params = analyzer.find_optimal_parameters(args.metric)
        
        if optimal_params:
            print(f"\n{'='*60}")
            print("OPTIMAL PARAMETERS")
            print(f"{'='*60}")
            print(f"Metric: {optimal_params['metric_used']}")
            print(f"Hidden Size: {optimal_params['hidden_size']}")
            print(f"Sequence Length: {optimal_params['sequence_length']}")
            print(f"Mean Performance: {optimal_params['mean_performance']:.4f}")
            print(f"Std Performance: {optimal_params['std_performance']:.4f}")
            print(f"Number of runs: {optimal_params['num_runs']}")
            print(f"{'='*60}")
        
        print(f"\n🎉 Analysis completed successfully!")
        print(f"Reports saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
