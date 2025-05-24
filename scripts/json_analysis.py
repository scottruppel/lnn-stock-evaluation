def analyze_financial_performance(self):
        """
        Analyze financial-specific metrics like Sharpe ratio, returns, drawdowns.
        """
        print("\n💰 FINANCIAL PERFORMANCE ANALYSIS")
        print("=" * 50)
        
        financial_metrics = ['eval_r2', 'eval_sharpe_ratio', 'eval_strategy_return', 
                           'eval_buy_hold_return', 'eval_max_drawdown', 'eval_directional_accuracy']
        
        available_metrics = [m for m in financial_metrics if m in self.df.columns]
        
        if not available_metrics:
            print("⚠️  No financial metrics found.")
            return None
        
        # Summary statistics
        print("📊 Financial Metrics Summary:")
        summary = self.df[available_metrics].describe().round(4)
        print(summary)
        
        # Find best performing experiments
        if 'eval_sharpe_ratio' in self.df.columns:
            best_sharpe_idx = self.df['eval_sharpe_ratio'].idxmax()
            best_sharpe_exp = self.df.loc[best_sharpe_idx]
            
            print(f"\n🏆 Best Sharpe Ratio: {best_sharpe_exp['eval_sharpe_ratio']:.4f}")
            print(f"   📋 Experiment: {best_sharpe_exp['experiment_name']}")
            print(f"   🎯 Ticker: {best_sharpe_exp['target_ticker']}")
            print(f"   ⚙️  Config: {best_sharpe_exp['sequence_length']} seq, {best_sharpe_exp['hidden_size']} hidden")
        
        if 'eval_r2' in self.df.columns:
            best_r2_idx = self.df['eval_r2'].idxmax()
            best_r2_exp = self.df.loc[best_r2_idx]
            
            print(f"\n📈 Best R² Score: {best_r2_exp['eval_r2']:.4f}")
            print(f"   📋 Experiment: {best_r2_exp['experiment_name']}")
            print(f"   🎯 Ticker: {best_r2_exp['target_ticker']}")
            print(f"   ⚙️  Config: {best_r2_exp['sequence_length']} seq, {best_r2_exp['hidden_size']} hidden")
        
        # Strategy vs Buy-and-Hold comparison
        if 'eval_strategy_return' in self.df.columns and 'eval_buy_hold_return' in self.df.columns:
            self.df['strategy_outperformance'] = self.df['eval_strategy_return'] - self.df['eval_buy_hold_return']
            
            outperforming = (self.df['strategy_outperformance'] > 0).sum()
            total = len(self.df)
            
            print(f"\n📈 Strategy Performance:")
            print(f"   🎯 Models beating buy-and-hold: {outperforming}/{total} ({outperforming/total*100:.1f}%)")
            print(f"   📊 Average outperformance: {self.df['strategy_outperformance'].mean():.4f}")
            
            # Best outperforming model
            best_outperf_idx = self.df['strategy_outperformance'].idxmax()
            best_outperf = self.df.loc[best_outperf_idx]
            print(f"   🏆 Best outperformance: {best_outperf['strategy_outperformance']:.4f}")
            print(f"     📋 Experiment: {best_outperf['experiment_name']}")
        
        # Risk-adjusted analysis
        if 'eval_max_drawdown' in self.df.columns and 'eval_sharpe_ratio' in self.df.columns:
            # Risk score (lower is better): combines max drawdown and inverse Sharpe
            self.df['risk_score'] = abs(self.df['eval_max_drawdown']) - self.df['eval_sharpe_ratio']
            
            best_risk_idx = self.df['risk_score'].idxmin()
            best_risk_exp = self.df.loc[best_risk_idx]
            
            print(f"\n🛡️  Best Risk-Adjusted Performance:")
            print(f"   📋 Experiment: {best_risk_exp['experiment_name']}")
            print(f"   📊 Sharpe Ratio: {best_risk_exp['eval_sharpe_ratio']:.4f}")
            print(f"   📉 Max Drawdown: {best_risk_exp['eval_max_drawdown']:.4f}")
        
        # Create financial performance visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Sharpe ratio distribution
        if 'eval_sharpe_ratio' in self.df.columns:
            axes[0, 0].hist(self.df['eval_sharpe_ratio'], bins=15, alpha=0.7)
            axes[0, 0].set_title('Sharpe Ratio Distribution')
            axes[0, 0].axvline(self.df['eval_sharpe_ratio'].mean(), color='red', linestyle='--', label='Mean')
            axes[0, 0].legend()
        
        # R² vs Sharpe ratio
        if 'eval_r2' in self.df.columns and 'eval_sharpe_ratio' in self.df.columns:
            scatter = axes[0, 1].scatter(self.df['eval_r2'], self.df['eval_sharpe_ratio'], 
                                       c=self.df['target_ticker'].astype('category').cat.codes, alpha=0.7)
            axes[0, 1].set_xlabel('R² Score')
            axes[0, 1].set_ylabel('Sharpe Ratio')
            axes[0, 1].set_title('R² vs Sharpe Ratio')
        
        # Strategy vs Buy-and-Hold returns
        if 'eval_strategy_return' in self.df.columns and 'eval_buy_hold_return' in self.df.columns:
            axes[0, 2].scatter(self.df['eval_buy_hold_return'], self.df['eval_strategy_return'], alpha=0.7)
            # Add diagonal line (where strategy = buy-and-hold)
            min_val = min(self.df['eval_buy_hold_return'].min(), self.df['eval_strategy_return'].min())
            max_val = max(self.df['eval_buy_hold_return'].max(), self.df['eval_strategy_return'].max())
            axes[0, 2].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            axes[0, 2].set_xlabel('Buy-and-Hold Return')
            axes[0, 2].set_ylabel('Strategy Return')
            axes[0, 2].set_title('Strategy vs Buy-and-Hold Returns')
        
        # Max drawdown analysis
        if 'eval_max_drawdown' in self.df.columns:
            ticker_drawdowns = self.df.groupby('target_ticker')['eval_max_drawdown'].mean()
            axes[1, 0].bar(ticker_drawdowns.index, ticker_drawdowns.values)
            axes[1, 0].set_title('Average Max Drawdown by Ticker')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Directional accuracy by ticker
        if 'eval_directional_accuracy' in self.df.columns:
            ticker_accuracy = self.df.groupby('target_ticker')['eval_directional_accuracy'].mean()
            axes[1, 1].bar(ticker_accuracy.index, ticker_accuracy.values)
            axes[1, 1].set_title('Average Directional Accuracy by Ticker')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Risk-return scatter
        if 'eval_sharpe_ratio' in self.df.columns and 'eval_max_drawdown' in self.df.columns:
            for ticker in self.df['target_ticker'].unique():
                ticker_data = self.df[self.df['target_ticker'] == ticker]
                axes[1, 2].scatter(ticker_data['eval_max_drawdown'], ticker_data['eval_sharpe_ratio'], 
                                 label=ticker, alpha=0.7)
            axes[1, 2].set_xlabel('Max Drawdown')
            axes[1, 2].set_ylabel('Sharpe Ratio')
            axes[1, 2].set_title('Risk-Return Profile by Ticker')
            axes[1, 2].legend()
        
        plt.tight_layout()
        plt.show()
        
        return {
            'summary': summary,
            'best_configs': {
                'sharpe': best_sharpe_exp if 'eval_sharpe_ratio' in self.df.columns else None,
                'r2': best_r2_exp if 'eval_r2' in self.df.columns else None
            }
        }import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class FinancialModelAnalysisInsights:
    """
    Analyzes JSON files containing model training/analysis results to derive insights
    about optimal hyperparameters and architectural choices.
    """
    
    def __init__(self, json_file_path):
        """
        Initialize the analyzer with your JSON file.
        
        Args:
            json_file_path (str): Path to your JSON file containing analysis results
        """
        self.json_file_path = json_file_path
        self.data = None
        self.df = None
        
    def load_and_parse_data(self):
        """
        Load the JSONL file and convert it to a pandas DataFrame for easier analysis.
        Your file is in JSONL format (one JSON object per line).
        """
        try:
            # Read JSONL file (one JSON object per line)
            records = []
            with open(self.json_file_path, 'r') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        records.append(json.loads(line))
            
            self.data = records
            
            # Flatten the nested structure for easier analysis
            flattened_records = []
            for record in records:
                flat_record = {
                    'experiment_id': record.get('experiment_id'),
                    'experiment_name': record.get('experiment_name'),
                    'timestamp': record.get('timestamp'),
                    # Flatten config
                    'target_ticker': record.get('config', {}).get('data_config', {}).get('target_ticker'),
                    'num_tickers': len(record.get('config', {}).get('data_config', {}).get('tickers', [])),
                    'input_size': record.get('config', {}).get('model_config', {}).get('input_size'),
                    'hidden_size': record.get('config', {}).get('model_config', {}).get('hidden_size'),
                    'sequence_length': record.get('config', {}).get('model_config', {}).get('sequence_length'),
                    'learning_rate': record.get('config', {}).get('model_config', {}).get('learning_rate'),
                    'batch_size': record.get('config', {}).get('model_config', {}).get('batch_size'),
                    'patience': record.get('config', {}).get('model_config', {}).get('patience'),
                    'device': record.get('config', {}).get('training_config', {}).get('device', 'cpu'),
                    'total_parameters': record.get('config', {}).get('training_config', {}).get('total_parameters'),
                }
                
                # Add all metrics directly
                metrics = record.get('metrics', {})
                flat_record.update(metrics)
                
                flattened_records.append(flat_record)
            
            self.df = pd.DataFrame(flattened_records)
            
            # Filter out incomplete experiments (focus on comprehensive analyses)
            self.df = self.df.dropna(subset=['eval_r2', 'eval_sharpe_ratio'])
            
            print(f"✅ Loaded {len(self.df)} complete analysis records from {self.json_file_path}")
            print(f"📊 Key metrics available: R² scores, Sharpe ratios, directional accuracy")
            print(f"🎯 Target tickers analyzed: {sorted(self.df['target_ticker'].unique())}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading JSON file: {e}")
            return False
    
    def analyze_abstraction_layers(self, performance_metric='eval_r2'):
        """
        Analyze which input configurations and data abstractions work better.
        Since you don't have explicit abstraction layers, we'll analyze input patterns.
        """
        print("\n🔍 INPUT CONFIGURATION ANALYSIS")
        print("=" * 50)
        
        # Analyze input size vs performance
        if 'input_size' in self.df.columns:
            input_stats = self.df.groupby('input_size')[performance_metric].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(4)
            
            print("📈 Performance by Input Size (Number of Features):")
            print(input_stats)
            
            best_input_size = input_stats['mean'].idxmax()
            print(f"\n🎯 Optimal input size: {best_input_size} features")
        
        # Analyze ticker combinations
        print(f"\n📊 Performance by Target Ticker:")
        ticker_stats = self.df.groupby('target_ticker')[performance_metric].agg([
            'count', 'mean', 'std'
        ]).round(4)
        print(ticker_stats)
        
        best_ticker = ticker_stats['mean'].idxmax()
        print(f"\n🏆 Best performing ticker: {best_ticker}")
        
        # Analyze number of input tickers
        if 'num_tickers' in self.df.columns:
            print(f"\n📈 Performance by Number of Input Tickers:")
            ticker_count_stats = self.df.groupby('num_tickers')[performance_metric].agg([
                'count', 'mean', 'std'
            ]).round(4)
            print(ticker_count_stats)
        
        # Create comprehensive visualization
        plt.figure(figsize=(16, 12))
        
        # Input size analysis
        plt.subplot(2, 3, 1)
        if 'input_size' in self.df.columns:
            self.df.boxplot(column=performance_metric, by='input_size', ax=plt.gca())
            plt.title('Performance by Input Size')
            plt.suptitle('')
        
        # Ticker performance
        plt.subplot(2, 3, 2)
        ticker_means = self.df.groupby('target_ticker')[performance_metric].mean()
        plt.bar(ticker_means.index, ticker_means.values)
        plt.title('Average Performance by Ticker')
        plt.xticks(rotation=45)
        
        # Number of tickers vs performance
        plt.subplot(2, 3, 3)
        if 'num_tickers' in self.df.columns:
            self.df.boxplot(column=performance_metric, by='num_tickers', ax=plt.gca())
            plt.title('Performance by Number of Input Tickers')
            plt.suptitle('')
        
        # Input size vs parameters
        plt.subplot(2, 3, 4)
        if 'total_parameters' in self.df.columns and 'input_size' in self.df.columns:
            self.df.plot.scatter(x='input_size', y='total_parameters',
                               c=performance_metric, colormap='viridis', ax=plt.gca())
            plt.title('Model Size vs Input Size')
        
        # Ticker complexity analysis
        plt.subplot(2, 3, 5)
        # Create a heatmap of ticker vs input_size performance
        if 'input_size' in self.df.columns:
            pivot_data = self.df.pivot_table(values=performance_metric, 
                                           index='target_ticker', 
                                           columns='input_size', 
                                           aggfunc='mean')
            sns.heatmap(pivot_data, annot=True, cmap='viridis', ax=plt.gca())
            plt.title('Performance Heatmap: Ticker vs Input Size')
        
        # Device performance comparison
        plt.subplot(2, 3, 6)
        if 'device' in self.df.columns:
            device_perf = self.df.groupby(['device', 'target_ticker'])[performance_metric].mean().unstack(fill_value=0)
            device_perf.plot(kind='bar', ax=plt.gca())
            plt.title('Performance by Device & Ticker')
            plt.xticks(rotation=0)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'input_stats': input_stats if 'input_size' in self.df.columns else None,
            'ticker_stats': ticker_stats,
            'ticker_count_stats': ticker_count_stats if 'num_tickers' in self.df.columns else None
        }
    
    def analyze_sequence_lengths(self, performance_metric='eval_r2'):
        """
        Analyze optimal sequence lengths for your financial models.
        """
        print("\n🔍 SEQUENCE LENGTH ANALYSIS")
        print("=" * 50)
        
        if 'sequence_length' not in self.df.columns:
            print("⚠️  Sequence length data not found.")
            return None
        
        # Group by sequence length and calculate statistics
        seq_stats = self.df.groupby('sequence_length')[performance_metric].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(4)
        
        print("📈 Performance by Sequence Length:")
        print(seq_stats)
        
        # Find optimal sequence length
        best_seq_length = seq_stats['mean'].idxmax()
        best_performance = seq_stats['mean'].max()
        
        print(f"\n🎯 Optimal sequence length: {best_seq_length}")
        print(f"📊 Average {performance_metric}: {best_performance:.4f}")
        
        # Analyze by ticker as well
        print(f"\n📈 Sequence Length Performance by Ticker:")
        ticker_seq_analysis = self.df.groupby(['target_ticker', 'sequence_length'])[performance_metric].mean().unstack(fill_value=0)
        print(ticker_seq_analysis.round(4))
        
        # Create visualization
        plt.figure(figsize=(14, 8))
        
        # Main plot: sequence length vs performance
        plt.subplot(2, 2, 1)
        self.df.boxplot(column=performance_metric, by='sequence_length', ax=plt.gca())
        plt.title('Performance Distribution by Sequence Length')
        plt.suptitle('')
        
        # Ticker-specific analysis
        plt.subplot(2, 2, 2)
        for ticker in self.df['target_ticker'].unique():
            ticker_data = self.df[self.df['target_ticker'] == ticker]
            if len(ticker_data) > 1:
                seq_means = ticker_data.groupby('sequence_length')[performance_metric].mean()
                plt.plot(seq_means.index, seq_means.values, marker='o', label=ticker)
        plt.xlabel('Sequence Length')
        plt.ylabel(performance_metric.replace('_', ' ').title())
        plt.title('Performance by Sequence Length & Ticker')
        plt.legend()
        
        # Training efficiency
        plt.subplot(2, 2, 3)
        if 'training_epochs' in self.df.columns:
            self.df.plot.scatter(x='sequence_length', y='training_epochs', 
                               c=performance_metric, colormap='viridis', ax=plt.gca())
            plt.title('Training Epochs vs Sequence Length')
        
        # Parameter efficiency
        plt.subplot(2, 2, 4)
        if 'total_parameters' in self.df.columns:
            self.df.plot.scatter(x='sequence_length', y='total_parameters',
                               c=performance_metric, colormap='viridis', ax=plt.gca())
            plt.title('Model Size vs Sequence Length')
        
        plt.tight_layout()
        plt.show()
        
        return seq_stats
    
    def analyze_hidden_layers(self, performance_metric='eval_r2'):
        """
        Analyze optimal hidden layer sizes for financial prediction.
        """
        print("\n🔍 HIDDEN LAYER SIZE ANALYSIS")
        print("=" * 50)
        
        if 'hidden_size' not in self.df.columns:
            print("⚠️  Hidden size data not found.")
            return None
        
        # Group by hidden size
        hidden_stats = self.df.groupby('hidden_size')[performance_metric].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(4)
        
        print("📈 Performance by Hidden Layer Size:")
        print(hidden_stats)
        
        # Find optimal hidden size
        best_hidden_size = hidden_stats['mean'].idxmax()
        best_performance = hidden_stats['mean'].max()
        
        print(f"\n🎯 Optimal hidden layer size: {best_hidden_size}")
        print(f"📊 Average {performance_metric}: {best_performance:.4f}")
        
        # Analyze parameter efficiency
        if 'total_parameters' in self.df.columns:
            self.df['param_efficiency'] = self.df[performance_metric] / (self.df['total_parameters'] / 1000)
            
            print(f"\n⚡ Parameter Efficiency Analysis (Performance per 1K parameters):")
            param_eff = self.df.groupby('hidden_size')['param_efficiency'].mean().round(4)
            print(param_eff)
            
            most_efficient_size = param_eff.idxmax()
            print(f"🏆 Most parameter-efficient hidden size: {most_efficient_size}")
        
        # Performance vs computational cost analysis
        plt.figure(figsize=(15, 10))
        
        # Main performance plot
        plt.subplot(2, 3, 1)
        self.df.boxplot(column=performance_metric, by='hidden_size', ax=plt.gca())
        plt.title('Performance by Hidden Size')
        plt.suptitle('')
        
        # Parameter count vs performance
        plt.subplot(2, 3, 2)
        if 'total_parameters' in self.df.columns:
            self.df.plot.scatter(x='total_parameters', y=performance_metric, 
                               c='hidden_size', colormap='viridis', ax=plt.gca())
            plt.title('Performance vs Total Parameters')
        
        # Training time analysis
        plt.subplot(2, 3, 3)
        if 'training_epochs' in self.df.columns:
            self.df.plot.scatter(x='hidden_size', y='training_epochs',
                               c=performance_metric, colormap='viridis', ax=plt.gca())
            plt.title('Training Epochs vs Hidden Size')
        
        # Ticker-specific analysis
        plt.subplot(2, 3, 4)
        for ticker in self.df['target_ticker'].unique():
            ticker_data = self.df[self.df['target_ticker'] == ticker]
            if len(ticker_data) > 1:
                hidden_means = ticker_data.groupby('hidden_size')[performance_metric].mean()
                plt.plot(hidden_means.index, hidden_means.values, marker='s', label=ticker)
        plt.xlabel('Hidden Size')
        plt.ylabel(performance_metric.replace('_', ' ').title())
        plt.title('Performance by Hidden Size & Ticker')
        plt.legend()
        
        # Device performance comparison
        plt.subplot(2, 3, 5)
        if 'device' in self.df.columns:
            device_perf = self.df.groupby(['device', 'hidden_size'])[performance_metric].mean().unstack(fill_value=0)
            device_perf.plot(kind='bar', ax=plt.gca())
            plt.title('Performance by Device & Hidden Size')
            plt.xticks(rotation=0)
        
        # Parameter efficiency plot
        plt.subplot(2, 3, 6)
        if 'param_efficiency' in self.df.columns:
            param_eff_by_size = self.df.groupby('hidden_size')['param_efficiency'].mean()
            plt.bar(param_eff_by_size.index, param_eff_by_size.values)
            plt.xlabel('Hidden Size')
            plt.ylabel('Performance per 1K Parameters')
            plt.title('Parameter Efficiency by Hidden Size')
        
        plt.tight_layout()
        plt.show()
        
        return hidden_stats
    
    def generate_optimization_recommendations(self, performance_metric='eval_r2'):
        """
        Generate comprehensive recommendations based on financial model analysis.
        """
        print("\n🚀 FINANCIAL MODEL OPTIMIZATION RECOMMENDATIONS")
        print("=" * 60)
        
        recommendations = []
        
        # Find correlations with performance
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1 and performance_metric in numeric_cols:
            corr_matrix = self.df[numeric_cols].corr()
            performance_corrs = corr_matrix[performance_metric].abs().sort_values(ascending=False)
            
            print("📊 Strongest correlations with R² performance:")
            for param, corr in performance_corrs.head(8).items():
                if param != performance_metric and corr > 0.1:
                    direction = "positively" if corr_matrix[performance_metric][param] > 0 else "negatively"
                    print(f"   • {param}: {corr:.3f} ({direction} correlated)")
                    
                    if corr > 0.3:  # Strong correlation
                        if corr_matrix[performance_metric][param] > 0:
                            recommendations.append(f"Consider increasing {param} for better R² scores")
                        else:
                            recommendations.append(f"Consider decreasing {param} for better R² scores")
        
        # Ticker-specific recommendations
        if 'target_ticker' in self.df.columns:
            ticker_performance = self.df.groupby('target_ticker')[performance_metric].mean().sort_values(ascending=False)
            best_ticker = ticker_performance.index[0]
            worst_ticker = ticker_performance.index[-1]
            
            print(f"\n🎯 Ticker Performance Ranking:")
            for ticker, score in ticker_performance.items():
                print(f"   • {ticker}: {score:.4f}")
            
            recommendations.append(f"Focus optimization efforts on {worst_ticker} (lowest R²: {ticker_performance[worst_ticker]:.4f})")
            
            # Best configs for each ticker
            print(f"\n🏆 Best configurations by ticker:")
            for ticker in self.df['target_ticker'].unique():
                ticker_data = self.df[self.df['target_ticker'] == ticker]
                if len(ticker_data) > 0:
                    best_idx = ticker_data[performance_metric].idxmax()
                    best_config = ticker_data.loc[best_idx]
                    print(f"   • {ticker}: seq_len={best_config['sequence_length']}, hidden={best_config['hidden_size']}, R²={best_config[performance_metric]:.4f}")
        
        # Sequence length recommendations
        if 'sequence_length' in self.df.columns:
            seq_performance = self.df.groupby('sequence_length')[performance_metric].mean()
            best_seq = seq_performance.idxmax()
            recommendations.append(f"Optimal sequence length appears to be {best_seq}")
        
        # Hidden size recommendations
        if 'hidden_size' in self.df.columns:
            hidden_performance = self.df.groupby('hidden_size')[performance_metric].mean()
            best_hidden = hidden_performance.idxmax()
            recommendations.append(f"Optimal hidden size appears to be {best_hidden}")
        
        # Financial performance recommendations
        if 'eval_sharpe_ratio' in self.df.columns:
            sharpe_threshold = self.df['eval_sharpe_ratio'].quantile(0.75)
            good_sharpe_models = self.df[self.df['eval_sharpe_ratio'] > sharpe_threshold]
            
            if len(good_sharpe_models) > 0:
                print(f"\n💰 High Sharpe Ratio Models (>{sharpe_threshold:.2f}):")
                for _, model in good_sharpe_models.head(3).iterrows():
                    print(f"   • {model['experiment_name']}: Sharpe={model['eval_sharpe_ratio']:.4f}, R²={model.get(performance_metric, 'N/A')}")
        
        # Device performance analysis
        if 'device' in self.df.columns:
            device_performance = self.df.groupby('device')[performance_metric].agg(['mean', 'count'])
            print(f"\n🖥️  Device Performance Comparison:")
            print(device_performance.round(4))
        
        # Risk-adjusted recommendations
        if 'eval_max_drawdown' in self.df.columns and 'eval_sharpe_ratio' in self.df.columns:
            # Find models with good risk-return profile
            self.df['risk_adjusted_score'] = self.df['eval_sharpe_ratio'] / abs(self.df['eval_max_drawdown'])
            best_risk_adjusted = self.df.loc[self.df['risk_adjusted_score'].idxmax()]
            
            recommendations.append(f"Best risk-adjusted model: {best_risk_adjusted['experiment_name']} "
                                 f"(Sharpe/|Drawdown|: {best_risk_adjusted['risk_adjusted_score']:.2f})")
        
        print(f"\n💡 KEY ACTIONABLE RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        # Next experiment suggestions
        print(f"\n🔬 SUGGESTED NEXT EXPERIMENTS:")
        
        if 'sequence_length' in self.df.columns:
            tested_seq_lengths = sorted(self.df['sequence_length'].unique())
            print(f"   • Test sequence lengths: {[x for x in [15, 45, 90, 120] if x not in tested_seq_lengths]}")
        
        if 'hidden_size' in self.df.columns:
            tested_hidden_sizes = sorted(self.df['hidden_size'].unique())
            print(f"   • Test hidden sizes: {[x for x in [75, 200, 300] if x not in tested_hidden_sizes]}")
        
        untested_tickers = ['SPY', 'QQQ', 'VTI', 'IWM', 'NVDA', 'MSFT']
        if 'target_ticker' in self.df.columns:
            tested_tickers = self.df['target_ticker'].unique()
            suggestions = [t for t in untested_tickers if t not in tested_tickers]
            if suggestions:
                print(f"   • Test additional tickers: {suggestions[:3]}")
        
        print(f"   • Try different patience values: [5, 15, 25] for early stopping optimization")
        print(f"   • Experiment with batch sizes: [16, 64] to see if it affects convergence")
        
        return recommendations
    
    def create_comprehensive_report(self, performance_metric='eval_r2'):
        """
        Generate a complete analysis report with all insights for financial models.
        """
        print("🔬 COMPREHENSIVE FINANCIAL MODEL ANALYSIS REPORT")
        print("=" * 70)
        
        if not self.load_and_parse_data():
            return
        
        # Run all analyses
        self.analyze_abstraction_layers(performance_metric)
        self.analyze_sequence_lengths(performance_metric)
        self.analyze_hidden_layers(performance_metric)
        self.analyze_financial_performance()
        self.generate_optimization_recommendations(performance_metric)
        
        # Summary statistics
        print(f"\n📋 EXECUTIVE SUMMARY")
        print("=" * 30)
        print(f"📊 Total experiments analyzed: {len(self.df)}")
        print(f"🎯 Tickers tested: {', '.join(sorted(self.df['target_ticker'].unique()))}")
        print(f"📈 Best R² score: {self.df[performance_metric].max():.4f}")
        print(f"📊 Average R² score: {self.df[performance_metric].mean():.4f}")
        print(f"📉 R² standard deviation: {self.df[performance_metric].std():.4f}")
        
        if 'eval_sharpe_ratio' in self.df.columns:
            print(f"💰 Best Sharpe ratio: {self.df['eval_sharpe_ratio'].max():.4f}")
            print(f"💰 Average Sharpe ratio: {self.df['eval_sharpe_ratio'].mean():.4f}")
        
        if 'eval_strategy_return' in self.df.columns and 'eval_buy_hold_return' in self.df.columns:
            outperforming = ((self.df['eval_strategy_return'] - self.df['eval_buy_hold_return']) > 0).sum()
            print(f"🏆 Models beating buy-and-hold: {outperforming}/{len(self.df)} ({outperforming/len(self.df)*100:.1f}%)")

# Example usage function for financial models
def analyze_financial_results(json_file_path, performance_metric='eval_r2'):
    """
    Main function to run financial model analysis on your JSONL file.
    
    Args:
        json_file_path (str): Path to your experiments.json file (JSONL format)
        performance_metric (str): Primary metric to optimize ('eval_r2', 'eval_sharpe_ratio', etc.)
    
    Usage Examples:
        # On your local machine (Windows/Mac/Linux):
        analyzer = analyze_financial_results('experiments.json', 'eval_r2')
        
        # On your Jetson Nano:
        analyzer = analyze_financial_results('/path/to/experiments.json', 'eval_sharpe_ratio')
        
        # Focus on different metrics:
        analyzer = analyze_financial_results('experiments.json', 'eval_directional_accuracy')
    """
    analyzer = FinancialModelAnalysisInsights(json_file_path)
    analyzer.create_comprehensive_report(performance_metric)
    return analyzer

# Quick setup function for different environments
def setup_environment():
    """
    Check and install required packages if needed.
    Run this first if you get import errors.
    """
    try:
        import pandas, matplotlib, seaborn
        print("✅ All required packages are installed!")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("\nTo install missing packages:")
        print("💻 On your local machine: pip install pandas matplotlib seaborn")
        print("🤖 On Jetson Nano: sudo pip3 install pandas matplotlib seaborn")
        print("☁️  On Google Colab: !pip install pandas matplotlib seaborn")

if __name__ == "__main__":
    # Run this to check your setup
    setup_environment()
    
    # Example usage for your specific file structure
    # analyzer = analyze_financial_results('experiments.json', 'eval_r2')
    
    print("\n" + "="*60)
    print("🚀 READY TO ANALYZE YOUR FINANCIAL MODEL EXPERIMENTS!")
    print("="*60)
    print("\nTo get started, save this script and run:")
    print("  python analyze_results.py")
    print("\nOr in Python:")
    print("  from analyze_results import analyze_financial_results")
    print("  analyzer = analyze_financial_results('experiments.json')")
    print("\nKey metrics you can analyze:")
    print("  • eval_r2 (R-squared for prediction accuracy)")
    print("  • eval_sharpe_ratio (risk-adjusted returns)")
    print("  • eval_directional_accuracy (trend prediction)")
    print("  • eval_strategy_return (model trading performance)")
    print("="*60)
