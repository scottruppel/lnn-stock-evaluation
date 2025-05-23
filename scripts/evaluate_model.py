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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

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
        
    def load_model_and_config(self):
        """Load the trained model and its configuration."""
        print("="*50)
        print("LOADING MODEL AND CONFIGURATION")
        print("="*50)
        
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load model checkpoint
        print(f"Loading model from: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract configuration
        self.config = checkpoint.get('config', {})
        print(f"Model trained for {checkpoint.get('epoch', 'unknown')} epochs")
        print(f"Best validation loss: {checkpoint.get('val_loss', 'unknown'):.6f}")
        
        # Reconstruct data to get model architecture info
        self.prepare_test_data()
        
        # Initialize model with correct architecture
        input_size = self.test_data['X'].shape[2]
        output_size = self.test_data['y'].shape[1] if len(self.test_data['y'].shape) > 1 else 1
        hidden_size = self.config.get('model', {}).get('hidden_size', 50)
        
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
        
    def prepare_test_data(self):
        """Prepare test data using the same preprocessing as training."""
        print("Preparing test data...")
        
        # Load data using same configuration as training
        data_config = self.config.get('data', {})
        tickers = data_config.get('tickers', ['^GSPC', 'AGG', 'QQQ', 'AAPL'])
        start_date = data_config.get('start_date', '2020-01-01')
        end_date = data_config.get('end_date', '2024-12-31')
        target_ticker = data_config.get('target_ticker', 'AAPL')
        
        # Load and preprocess data
        data_loader = StockDataLoader(tickers, start_date, end_date)
        raw_data = data_loader.download_data()
        price_data = data_loader.get_closing_prices()
        
        # Apply same preprocessing
        self.preprocessor = StockDataPreprocessor(scaling_method='minmax', feature_range=(-1, 1))
        scaled_data = self.preprocessor.fit_transform(price_data)
        
        # Prepare sequences
        model_config = self.config.get('model', {})
        sequence_length = model_config.get('sequence_length', 30)
        
        X_train, y_train, X_test, y_test = prepare_model_data(
            price_data=scaled_data,
            target_ticker=target_ticker,
            sequence_length=sequence_length,
            train_ratio=0.8,
            add_features=False
        )
        
        # Store test data
        self.test_data = {
            'X': torch.tensor(X_test, dtype=torch.float32),
            'y': torch.tensor(y_test, dtype=torch.float32),
            'raw_prices': price_data[target_ticker],
            'dates': raw_data.index if hasattr(raw_data, 'index') else None
        }
        
        print(f"Test data prepared: {self.test_data['X'].shape}")
        
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
        
        # Convert back to original scale for analysis
        target_ticker = self.config.get('data', {}).get('target_ticker', 'AAPL')
        self.predictions_unscaled = self.preprocessor.inverse_transform_single(
            target_ticker, self.predictions.numpy()
        )
        self.actuals_unscaled = self.preprocessor.inverse_transform_single(
            target_ticker, self.actuals.numpy()
        )
        
        print(f"Predictions range: ${self.predictions_unscaled.min():.2f} - ${self.predictions_unscaled.max():.2f}")
        print(f"Actuals range: ${self.actuals_unscaled.min():.2f} - ${self.actuals_unscaled.max():.2f}")
    
    def calculate_comprehensive_metrics(self):
        """Calculate all evaluation metrics."""
        print("="*50)
        print("CALCULATING COMPREHENSIVE METRICS")
        print("="*50)
        
        # Basic and financial metrics
        self.evaluation_results['metrics'] = self.metrics_calculator.comprehensive_evaluation(
            y_true=self.actuals_unscaled,
            y_pred=self.predictions_unscaled,
            dates=self.test_data['dates']
        )
        
        # Print key metrics
        metrics = self.evaluation_results['metrics']
        print("KEY PERFORMANCE METRICS:")
        print("-" * 30)
        
        if 'basic_metrics' in metrics:
            basic = metrics['basic_metrics']
            print(f"RMSE: ${basic.get('rmse', 'N/A'):.2f}")
            print(f"MAE: ${basic.get('mae', 'N/A'):.2f}")
            print(f"MAPE: {basic.get('mape', 'N/A'):.1f}%")
            print(f"R²: {basic.get('r2', 'N/A'):.3f}")
        
        if 'directional_metrics' in metrics:
            direction = metrics['directional_metrics']
            print(f"Directional Accuracy: {direction.get('directional_accuracy', 'N/A'):.1%}")
        
        if 'trading_metrics' in metrics:
            trading = metrics['trading_metrics']
            print(f"Strategy Return: {trading.get('total_return', 'N/A'):.1%}")
            print(f"Buy & Hold Return: {trading.get('buy_hold_return', 'N/A'):.1%}")
            print(f"Sharpe Ratio: {trading.get('sharpe_ratio', 'N/A'):.2f}")
            print(f"Max Drawdown: {trading.get('max_drawdown', 'N/A'):.1%}")
    
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
            plt.savefig(f'results/plots/trading_performance_{timestamp}.png', dpi=300, bbox_inches='tight')
            print(f"Saved trading plots to results/plots/trading_performance_{timestamp}.png")
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
            
            print("="*60)
            print("EVALUATION COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            return self.evaluation_results
            
        except Exception as e:
            print(f"ERROR during evaluation: {e}")
            import traceback
            traceback.print_exc()
            raise

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

if __name__ == "__main__":
    main()
