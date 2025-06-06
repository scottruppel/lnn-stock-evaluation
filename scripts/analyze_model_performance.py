#!/usr/bin/env python3
"""
Model Performance Analyzer - Extract and analyze performance from trained LNN models
Updated for Scott's backup drive structure
"""

import torch
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add project paths
sys.path.append('src')

class ModelPerformanceAnalyzer:
    def __init__(self, models_dir: str = None):
        # Use standardized model directory if none provided
        if models_dir is None:
            models_dir = str(file_namer.base_dir / "01_training" / "models")
        
        self.models_dir = models_dir
        
        # Search standardized reports directories
        self.reports_dirs = [
            str(file_namer.base_dir / "01_training" / "analysis_reports"),
            str(file_namer.base_dir / "02_evaluation" / "model_performance"),
            str(file_namer.base_dir / "03_backtesting" / "individual_backtests"),
            '/media/scott/Data/reports',  # Keep backup drive as fallback
            'results/reports'  # Legacy fallback
        ]
    
    def extract_model_info(self, model_path: str) -> dict:
        """Extract performance metrics and config from a model file."""
        try:
            print(f"  Analyzing: {os.path.basename(model_path)}")
            
            # Load the model file
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Extract key information
            model_info = {
                'model_file': os.path.basename(model_path),
                'file_path': model_path,
                'file_size_mb': round(os.path.getsize(model_path) / (1024*1024), 2),
                'modified_date': datetime.fromtimestamp(os.path.getmtime(model_path)),
            }
            
            # Extract configuration if available
            if 'config' in checkpoint:
                config = checkpoint['config']
                model_info.update({
                    'target_ticker': config.get('data', {}).get('target_ticker', 'Unknown'),
                    'start_date': config.get('data', {}).get('start_date', 'Unknown'),
                    'end_date': config.get('data', {}).get('end_date', 'Unknown'),
                    'sequence_length': config.get('model', {}).get('sequence_length', 0),
                    'hidden_size': config.get('model', {}).get('hidden_size', 0),
                    'learning_rate': config.get('model', {}).get('learning_rate', 0),
                    'epochs_trained': checkpoint.get('epoch', 0),
                })
            
            # Extract training metrics
            model_info['final_val_loss'] = checkpoint.get('val_loss', float('inf'))
            
            # Extract optimizer state for additional insights
            if 'optimizer_state_dict' in checkpoint:
                # Count parameters that were actually trained
                param_groups = checkpoint['optimizer_state_dict'].get('param_groups', [])
                if param_groups:
                    model_info['optimizer_lr'] = param_groups[0].get('lr', 0)
            
            # Look for companion results files
            companion_data = self._find_companion_files(model_path)
            model_info.update(companion_data)
            
            print(f"    ✓ Extracted basic info from model")
            
            return model_info
            
        except Exception as e:
            print(f"    ❌ Error analyzing {os.path.basename(model_path)}: {str(e)}")
            return {
                'model_file': os.path.basename(model_path),
                'error': str(e),
                'file_path': model_path
            }
    
    def _find_companion_files(self, model_path: str) -> dict:
        # Use standardized filename parsing
        model_info = parse_model_info(os.path.basename(model_path))
    
        # Search for related files using standardized approach
        related_files = file_namer.find_related_files(model_path)
    
        companion_info = {
            'companion_files': [],
            'companion_count': 0
        }
    
        # Combine all related files
        for stage_files in related_files.values():
            companion_info['companion_files'].extend(stage_files)
    
        companion_info['companion_count'] = len(companion_info['companion_files'])
    
        # Extract performance from related files
        for file_path in companion_info['companion_files']:
            if file_path.endswith('.json'):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                
                    extracted_perf = self._extract_performance_from_json(data)
                    if extracted_perf:
                        companion_info.update(extracted_perf)
                        break
                    
                except Exception as e:
                    continue
    
        return companion_info
    
    def _extract_performance_from_json(self, data: dict) -> dict:
        """Extract key performance metrics from results JSON."""
        performance = {}
        
        # Look for evaluation results
        if 'evaluation' in data:
            eval_data = data['evaluation']
            if 'metrics' in eval_data:
                metrics = eval_data['metrics']
                
                # Basic metrics
                if 'basic_metrics' in metrics:
                    basic = metrics['basic_metrics']
                    performance.update({
                        'rmse': basic.get('rmse'),
                        'mae': basic.get('mae'),
                        'mape': basic.get('mape'),
                        'r2_score': basic.get('r2')
                    })
                
                # Trading metrics (MOST IMPORTANT)
                if 'trading_metrics' in metrics:
                    trading = metrics['trading_metrics']
                    performance.update({
                        'strategy_return': trading.get('total_return'),
                        'buy_hold_return': trading.get('buy_hold_return'),
                        'sharpe_ratio': trading.get('sharpe_ratio'),
                        'max_drawdown': trading.get('max_drawdown'),
                        'win_rate': trading.get('win_rate'),
                        'avg_win': trading.get('avg_win'),
                        'avg_loss': trading.get('avg_loss')
                    })
                    
                    # Calculate excess return
                    if trading.get('total_return') is not None and trading.get('buy_hold_return') is not None:
                        performance['excess_return'] = trading.get('total_return') - trading.get('buy_hold_return')
                
                # Directional accuracy
                if 'directional_metrics' in metrics:
                    directional = metrics['directional_metrics']
                    performance.update({
                        'directional_accuracy': directional.get('directional_accuracy')
                    })
        
        # Look for training results
        if 'training' in data:
            training = data['training']
            performance.update({
                'best_val_loss': training.get('best_val_loss'),
                'final_val_loss': training.get('final_val_loss'),
                'total_epochs': training.get('total_epochs')
            })
        
        # Look for data statistics
        if 'data_statistics' in data:
            data_stats = data['data_statistics']
            # Get stats for the target ticker if available
            for ticker, stats in data_stats.items():
                if isinstance(stats, dict):
                    performance.update({
                        f'{ticker}_total_return': stats.get('total_return'),
                        f'{ticker}_sharpe_estimate': stats.get('sharpe_estimate'),
                        f'{ticker}_volatility': stats.get('return_std')
                    })
                    break  # Just take the first one for now
        
        return performance
    
    def analyze_all_models(self) -> pd.DataFrame:
        """Analyze all models in the models directory."""
        print(f"🔍 Scanning models directory: {self.models_dir}")
        
        if not os.path.exists(self.models_dir):
            print(f"❌ Models directory {self.models_dir} not found!")
            print("Available drives:")
            if os.path.exists('/media/scott'):
                for drive in os.listdir('/media/scott'):
                    drive_path = f'/media/scott/{drive}'
                    if os.path.isdir(drive_path):
                        print(f"  - {drive_path}")
            return pd.DataFrame()
        
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pth')]
        
        if not model_files:
            print(f"❌ No model files found in {self.models_dir}")
            return pd.DataFrame()
        
        print(f"📊 Found {len(model_files)} model files to analyze")
        print(f"📁 Will search for companion files in:")
        for reports_dir in self.reports_dirs:
            exists = "✓" if os.path.exists(reports_dir) else "❌"
            print(f"  {exists} {reports_dir}")
        print()
        
        for i, model_file in enumerate(model_files, 1):
            print(f"[{i}/{len(model_files)}] Processing {model_file}")
            model_path = os.path.join(self.models_dir, model_file)
            model_info = self.extract_model_info(model_path)
            self.performance_data.append(model_info)
            print()
        
        # Convert to DataFrame
        df = pd.DataFrame(self.performance_data)
        
        # Calculate performance scores
        df = self._calculate_performance_scores(df)
        
        return df
    
    def _calculate_performance_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate composite performance scores."""
        
        print("📈 Calculating performance scores...")
        
        # Initialize performance score
        df['performance_score'] = 0
        
        # Sharpe ratio contribution (40% weight)
        if 'sharpe_ratio' in df.columns:
            valid_sharpe = df['sharpe_ratio'].notna()
            sharpe_score = df['sharpe_ratio'].fillna(0) * 40
            df.loc[valid_sharpe, 'performance_score'] += sharpe_score[valid_sharpe]
            print(f"  ✓ Applied Sharpe ratio scoring to {valid_sharpe.sum()} models")
        
        # Excess return contribution (30% weight)  
        if 'excess_return' in df.columns:
            valid_excess = df['excess_return'].notna()
            excess_score = df['excess_return'].fillna(0) * 100 * 30  # Convert to percentage
            df.loc[valid_excess, 'performance_score'] += excess_score[valid_excess]
            print(f"  ✓ Applied excess return scoring to {valid_excess.sum()} models")
        
        # Directional accuracy contribution (20% weight)
        if 'directional_accuracy' in df.columns:
            valid_direction = df['directional_accuracy'].notna()
            direction_score = (df['directional_accuracy'].fillna(0.5) - 0.5) * 2 * 20  # Normalize around 50%
            df.loc[valid_direction, 'performance_score'] += direction_score[valid_direction]
            print(f"  ✓ Applied directional accuracy scoring to {valid_direction.sum()} models")
        
        # Low drawdown bonus (10% weight)
        if 'max_drawdown' in df.columns:
            valid_drawdown = df['max_drawdown'].notna()
            drawdown_score = (1 - df['max_drawdown'].fillna(1).abs()) * 10
            df.loc[valid_drawdown, 'performance_score'] += drawdown_score[valid_drawdown]
            print(f"  ✓ Applied drawdown scoring to {valid_drawdown.sum()} models")
        
        # Create performance categories
        df['performance_category'] = pd.cut(df['performance_score'], 
                                          bins=[-float('inf'), 0, 20, 40, 60, float('inf')],
                                          labels=['Poor', 'Below Average', 'Average', 'Good', 'Excellent'])
        
        return df
    
    def get_top_performers(self, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """Get top N performing models."""
        # Sort by performance score, then by sharpe ratio as tiebreaker
        if 'sharpe_ratio' in df.columns:
            top_models = df.nlargest(n, ['performance_score', 'sharpe_ratio'])
        else:
            top_models = df.nlargest(n, 'performance_score')
        
        return top_models
    
    def generate_performance_report(self, df: pd.DataFrame) -> str:
        """Generate a comprehensive performance report."""
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("LNN MODEL PERFORMANCE ANALYSIS REPORT")
        report_lines.append("Scott's Backup Drive Analysis")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Models Directory: {self.models_dir}")
        report_lines.append(f"Total Models Found: {len(df)}")
        
        # Count models with performance data
        has_performance = df['sharpe_ratio'].notna().sum()
        report_lines.append(f"Models with Performance Data: {has_performance}")
        report_lines.append("")
        
        # Performance summary
        if 'performance_category' in df.columns:
            report_lines.append("PERFORMANCE SUMMARY:")
            report_lines.append("-" * 40)
            performance_counts = df['performance_category'].value_counts()
            for category, count in performance_counts.items():
                report_lines.append(f"{category}: {count} models")
            report_lines.append("")
        
        # Models with data vs without
        with_data = df[df['companion_count'] > 0]
        without_data = df[df['companion_count'] == 0]
        
        report_lines.append(f"Models with companion files: {len(with_data)}")
        report_lines.append(f"Models without companion files: {len(without_data)}")
        report_lines.append("")
        
        # Top performers
        top_models = self.get_top_performers(df, 5)
        if not top_models.empty:
            report_lines.append("🏆 TOP 5 PERFORMING MODELS:")
            report_lines.append("-" * 40)
            
            for rank, (idx, model) in enumerate(top_models.iterrows(), 1):
                report_lines.append(f"#{rank}. {model['model_file']}")
                
                if 'target_ticker' in model and pd.notna(model['target_ticker']):
                    report_lines.append(f"    Stock: {model['target_ticker']}")
                
                if 'strategy_return' in model and pd.notna(model['strategy_return']):
                    report_lines.append(f"    Strategy Return: {model['strategy_return']:.1%}")
                
                if 'buy_hold_return' in model and pd.notna(model['buy_hold_return']):
                    report_lines.append(f"    Buy & Hold Return: {model['buy_hold_return']:.1%}")
                
                if 'excess_return' in model and pd.notna(model['excess_return']):
                    report_lines.append(f"    Excess Return: {model['excess_return']:.1%}")
                
                if 'sharpe_ratio' in model and pd.notna(model['sharpe_ratio']):
                    report_lines.append(f"    Sharpe Ratio: {model['sharpe_ratio']:.2f}")
                
                if 'directional_accuracy' in model and pd.notna(model['directional_accuracy']):
                    report_lines.append(f"    Direction Accuracy: {model['directional_accuracy']:.1%}")
                
                if 'max_drawdown' in model and pd.notna(model['max_drawdown']):
                    report_lines.append(f"    Max Drawdown: {model['max_drawdown']:.1%}")
                
                if 'performance_score' in model:
                    report_lines.append(f"    Performance Score: {model['performance_score']:.1f}")
                
                report_lines.append(f"    File Size: {model['file_size_mb']:.1f} MB")
                report_lines.append(f"    Modified: {model['modified_date'].strftime('%Y-%m-%d %H:%M')}")
                report_lines.append("")
        
        # Best by specific metrics
        metrics_to_highlight = [
            ('sharpe_ratio', 'HIGHEST SHARPE RATIO'),
            ('strategy_return', 'HIGHEST STRATEGY RETURNS'),
            ('excess_return', 'HIGHEST EXCESS RETURNS'),
            ('directional_accuracy', 'BEST DIRECTION PREDICTION')
        ]
        
        for metric, title in metrics_to_highlight:
            if metric in df.columns and df[metric].notna().any():
                best_idx = df[metric].idxmax()
                best_model = df.loc[best_idx]
                
                report_lines.append(f"🎯 {title}:")
                report_lines.append(f"Model: {best_model['model_file']}")
                if pd.notna(best_model[metric]):
                    if metric in ['strategy_return', 'excess_return', 'directional_accuracy']:
                        report_lines.append(f"Value: {best_model[metric]:.1%}")
                    else:
                        report_lines.append(f"Value: {best_model[metric]:.3f}")
                
                if 'target_ticker' in best_model and pd.notna(best_model['target_ticker']):
                    report_lines.append(f"Stock: {best_model['target_ticker']}")
                report_lines.append("")
        
        # Summary statistics
        if has_performance > 0:
            report_lines.append("📊 SUMMARY STATISTICS:")
            report_lines.append("-" * 40)
            
            for metric in ['sharpe_ratio', 'strategy_return', 'directional_accuracy']:
                if metric in df.columns and df[metric].notna().any():
                    values = df[metric].dropna()
                    
                    if metric in ['strategy_return', 'directional_accuracy']:
                        report_lines.append(f"{metric.replace('_', ' ').title()}:")
                        report_lines.append(f"  Average: {values.mean():.1%}")
                        report_lines.append(f"  Median: {values.median():.1%}")
                        report_lines.append(f"  Best: {values.max():.1%}")
                        report_lines.append(f"  Worst: {values.min():.1%}")
                    else:
                        report_lines.append(f"{metric.replace('_', ' ').title()}:")
                        report_lines.append(f"  Average: {values.mean():.2f}")
                        report_lines.append(f"  Median: {values.median():.2f}")
                        report_lines.append(f"  Best: {values.max():.2f}")
                        report_lines.append(f"  Worst: {values.min():.2f}")
                    report_lines.append("")
        
        return "\n".join(report_lines)
    
    def create_executive_summary_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a focused executive summary CSV with only the most relevant metrics."""
        
        print("📋 Creating executive summary CSV...")
        
        # Select only the most important columns
        summary_columns = [
            'model_file',
            'target_ticker',
            'performance_score',
            'performance_category',
            'strategy_return',
            'buy_hold_return', 
            'excess_return',
            'sharpe_ratio',
            'directional_accuracy',
            'max_drawdown',
            'modified_date',
            'file_size_mb',
            'companion_count'
        ]
        
        # Keep only columns that exist in the dataframe
        available_columns = [col for col in summary_columns if col in df.columns]
        summary_df = df[available_columns].copy()
        
        # Sort by performance score (best to worst)
        summary_df = summary_df.sort_values('performance_score', ascending=False).reset_index(drop=True)
        
        # Add rank column (1 = best)
        summary_df.insert(0, 'rank', range(1, len(summary_df) + 1))
        
        # Format columns for better readability
        if 'strategy_return' in summary_df.columns:
            summary_df['strategy_return_pct'] = (summary_df['strategy_return'] * 100).round(1)
        
        if 'buy_hold_return' in summary_df.columns:
            summary_df['buy_hold_return_pct'] = (summary_df['buy_hold_return'] * 100).round(1)
            
        if 'excess_return' in summary_df.columns:
            summary_df['excess_return_pct'] = (summary_df['excess_return'] * 100).round(1)
        
        if 'directional_accuracy' in summary_df.columns:
            summary_df['directional_accuracy_pct'] = (summary_df['directional_accuracy'] * 100).round(1)
            
        if 'max_drawdown' in summary_df.columns:
            summary_df['max_drawdown_pct'] = (summary_df['max_drawdown'] * 100).round(1)
        
        if 'sharpe_ratio' in summary_df.columns:
            summary_df['sharpe_ratio'] = summary_df['sharpe_ratio'].round(2)
        
        if 'performance_score' in summary_df.columns:
            summary_df['performance_score'] = summary_df['performance_score'].round(1)
        
        # Clean up column names and order
        final_columns = ['rank', 'model_file', 'target_ticker', 'performance_category', 'performance_score']
        
        # Add performance metrics in order of importance
        if 'sharpe_ratio' in summary_df.columns:
            final_columns.append('sharpe_ratio')
        if 'strategy_return_pct' in summary_df.columns:
            final_columns.append('strategy_return_pct')
        if 'excess_return_pct' in summary_df.columns:
            final_columns.append('excess_return_pct')
        if 'directional_accuracy_pct' in summary_df.columns:
            final_columns.append('directional_accuracy_pct')
        if 'max_drawdown_pct' in summary_df.columns:
            final_columns.append('max_drawdown_pct')
        if 'buy_hold_return_pct' in summary_df.columns:
            final_columns.append('buy_hold_return_pct')
        
        # Add metadata
        final_columns.extend(['modified_date', 'file_size_mb', 'companion_count'])
        
        # Keep only available columns
        final_columns = [col for col in final_columns if col in summary_df.columns]
        summary_df = summary_df[final_columns]
        
        # Rename columns for clarity
        column_renames = {
            'strategy_return_pct': 'strategy_return_%',
            'buy_hold_return_pct': 'buy_hold_return_%',
            'excess_return_pct': 'excess_return_%',
            'directional_accuracy_pct': 'directional_accuracy_%',
            'max_drawdown_pct': 'max_drawdown_%',
            'performance_score': 'perf_score',
            'performance_category': 'grade',
            'companion_count': 'data_files_found',
            'file_size_mb': 'size_mb'
        }
        
        summary_df = summary_df.rename(columns=column_renames)
        
        # Fill NaN values with clear indicators
        for col in summary_df.columns:
            if col.endswith('_%') or col in ['sharpe_ratio', 'perf_score']:
                summary_df[col] = summary_df[col].fillna('N/A')
        
        return summary_df

def main():
    """Main function to analyze model performance from Scott's backup drive."""
    
    print("🚀 LNN Model Performance Analyzer")
    print("Scott's Backup Drive Edition")
    print("=" * 60)
    
    # Initialize analyzer with Scott's backup drive paths
    analyzer = ModelPerformanceAnalyzer(
        models_dir="/media/scott/Data/saved_models"
    )
    
    # Analyze all models
    df = analyzer.analyze_all_models()
    
    if df.empty:
        print("❌ No models found to analyze!")
        print("\nTroubleshooting:")
        print("1. Check if backup drive is mounted: ls /media/scott/")
        print("2. Check if models directory exists: ls /media/scott/Data/")
        print("3. Verify model files exist: ls /media/scott/Data/saved_models/*.pth")
        return
    
    # Generate report
    report = analyzer.generate_performance_report(df)
    print("\n" + report)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    output_dir = "results/model_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed CSV (full data)
    csv_path = f"{output_dir}/model_performance_detailed_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Detailed analysis saved to: {csv_path}")
    
    # Create and save executive summary CSV
    summary_csv = analyzer.create_executive_summary_csv(df)
    summary_csv_path = f"{output_dir}/model_performance_summary_{timestamp}.csv"
    summary_csv.to_csv(summary_csv_path, index=False)
    print(f"✓ Executive summary CSV saved to: {summary_csv_path}")
    
    # Save summary report
    report_path = f"{output_dir}/model_performance_report_{timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✓ Performance report saved to: {report_path}")
    
    # Show actionable next steps
    top_models = analyzer.get_top_performers(df, 3)
    if not top_models.empty and top_models['sharpe_ratio'].notna().any():
        print("\n🎯 RECOMMENDED ACTIONS:")
        print("-" * 40)
        
        stellar_models = top_models[
            (top_models['sharpe_ratio'] > 1.0) & 
            (top_models['directional_accuracy'] > 0.55)
        ]
        
        if not stellar_models.empty:
            print("🌟 STELLAR MODELS READY FOR DEPLOYMENT:")
            for idx, model in stellar_models.iterrows():
                print(f"  • {model['model_file']}")
                print(f"    Sharpe: {model['sharpe_ratio']:.2f} | Return: {model.get('strategy_return', 0):.1%}")
            
            print("\nNext Steps:")
            print("1. Copy stellar models to production directory")
            print("2. Set up paper trading with top performer")
            print("3. Create live signal generation system")
        else:
            print("📈 DEVELOPING MODELS (need more training):")
            for idx, model in top_models.head(3).iterrows():
                print(f"  • {model['model_file']}")
                if pd.notna(model.get('sharpe_ratio')):
                    print(f"    Sharpe: {model['sharpe_ratio']:.2f}")
                else:
                    print("    (No performance data found)")
            
            print("\nNext Steps:")
            print("1. Retrain models with longer sequences")
            print("2. Add more features for better prediction")
            print("3. Try different hyperparameters")
    
    print(f"\n📊 Analysis complete! Found {len(df)} models, {df['sharpe_ratio'].notna().sum()} with performance data.")

if __name__ == "__main__":
    main()
