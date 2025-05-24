# src/analysis/backtester.py

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class Backtester:
    """
    Comprehensive backtesting framework for neural network trading models.
    Evaluates model performance using various trading strategies and metrics.
    """
    
    def __init__(self):
        self.trading_strategies = [
            'direction_only',     # Simple directional trading
            'threshold_based',    # Trade only when confidence > threshold
            'mean_reversion',     # Mean reversion strategy
            'momentum',           # Momentum strategy
            'volatility_adjusted' # Position sizing based on volatility
        ]
        
        # Default trading parameters
        self.default_params = {
            'transaction_cost': 0.001,  # 0.1% per trade
            'confidence_threshold': 0.6,  # For threshold-based strategy
            'rebalance_frequency': 1,     # Daily rebalancing
            'max_position_size': 1.0,     # 100% max allocation
            'stop_loss': 0.05,            # 5% stop loss
            'take_profit': 0.10           # 10% take profit
        }
    
    def run_backtest(self, 
                     model: torch.nn.Module,
                     test_features: np.ndarray,
                     test_targets: np.ndarray,
                     ticker: str,
                     preprocessor=None,
                     strategy: str = 'direction_only',
                     trading_params: Optional[Dict] = None) -> Dict:
        """
        Run comprehensive backtest for a trained model.
        
        Args:
            model: Trained PyTorch model
            test_features: Test feature sequences
            test_targets: Test target values
            ticker: Stock ticker symbol
            preprocessor: Data preprocessor for inverse transformation
            strategy: Trading strategy to use
            trading_params: Trading parameters (uses defaults if None)
        
        Returns:
            Dictionary with comprehensive backtest results
        """
        print(f"📈 Running backtest for {ticker} using {strategy} strategy...")
        
        # Merge trading parameters
        params = self.default_params.copy()
        if trading_params:
            params.update(trading_params)
        
        try:
            # Generate model predictions
            predictions = self._generate_predictions(model, test_features)
            
            # Convert back to original scale if preprocessor available
            if preprocessor:
                try:
                    actual_prices = preprocessor.inverse_transform_single(ticker, test_targets)
                    predicted_prices = preprocessor.inverse_transform_single(ticker, predictions)
                except:
                    # Fallback if inverse transform fails
                    print("⚠️  Inverse transform failed, using normalized values")
                    actual_prices = test_targets.flatten()
                    predicted_prices = predictions.flatten()
            else:
                actual_prices = test_targets.flatten()
                predicted_prices = predictions.flatten()
            
            # Generate trading signals
            signals = self._generate_trading_signals(
                actual_prices, predicted_prices, strategy, params
            )
            
            # Simulate trading
            trading_results = self._simulate_trading(
                actual_prices, signals, params
            )
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                actual_prices, predicted_prices, trading_results
            )
            
            # Calculate prediction accuracy metrics
            prediction_metrics = self._calculate_prediction_metrics(
                actual_prices, predicted_prices
            )
            
            # Combine all results
            backtest_results = {
                'ticker': ticker,
                'strategy': strategy,
                'trading_params': params,
                'performance_metrics': performance_metrics,
                'prediction_metrics': prediction_metrics,
                'trading_results': trading_results,
                'n_trades': len([s for s in signals if s != 0]),
                'n_test_days': len(actual_prices),
                'backtest_timestamp': datetime.now().isoformat()
            }
            
            # Add summary metrics to top level for easy access
            backtest_results.update({
                'total_return': performance_metrics.get('total_return', 0.0),
                'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0.0),
                'max_drawdown': performance_metrics.get('max_drawdown', 0.0),
                'win_rate': performance_metrics.get('win_rate', 0.0),
                'prediction_accuracy': prediction_metrics.get('directional_accuracy', 0.0)
            })
            
            print(f"✅ Backtest completed for {ticker}")
            return backtest_results
            
        except Exception as e:
            print(f"❌ Backtest failed for {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'ticker': ticker,
                'error': str(e),
                'backtest_timestamp': datetime.now().isoformat()
            }
    
    def _generate_predictions(self, model: torch.nn.Module, test_features: np.ndarray) -> np.ndarray:
        """Generate predictions from the model."""
        model.eval()
        
        with torch.no_grad():
            # Convert to tensor
            if isinstance(test_features, np.ndarray):
                X_tensor = torch.FloatTensor(test_features)
            else:
                X_tensor = test_features
            
            # Move to same device as model
            device = next(model.parameters()).device
            X_tensor = X_tensor.to(device)
            
            # Generate predictions
            predictions = model(X_tensor)
            
            # Convert back to numpy
            predictions_np = predictions.cpu().numpy()
            
        return predictions_np
    
    def _generate_trading_signals(self, 
                                actual_prices: np.ndarray,
                                predicted_prices: np.ndarray,
                                strategy: str,
                                params: Dict) -> np.ndarray:
        """
        Generate trading signals based on predictions and strategy.
        
        Returns:
            Array of trading signals: 1 = buy, -1 = sell, 0 = hold
        """
        
        if strategy == 'direction_only':
            return self._direction_only_signals(actual_prices, predicted_prices)
        elif strategy == 'threshold_based':
            return self._threshold_based_signals(actual_prices, predicted_prices, params)
        elif strategy == 'mean_reversion':
            return self._mean_reversion_signals(actual_prices, predicted_prices, params)
        elif strategy == 'momentum':
            return self._momentum_signals(actual_prices, predicted_prices, params)
        elif strategy == 'volatility_adjusted':
            return self._volatility_adjusted_signals(actual_prices, predicted_prices, params)
        else:
            print(f"⚠️  Unknown strategy {strategy}, using direction_only")
            return self._direction_only_signals(actual_prices, predicted_prices)
    
    def _direction_only_signals(self, actual_prices: np.ndarray, predicted_prices: np.ndarray) -> np.ndarray:
        """Simple directional trading signals."""
        signals = np.zeros(len(predicted_prices))
        
        for i in range(1, len(predicted_prices)):
            # Predict direction of next price change
            if predicted_prices[i] > actual_prices[i-1]:
                signals[i] = 1  # Buy signal
            elif predicted_prices[i] < actual_prices[i-1]:
                signals[i] = -1  # Sell signal
            # else: hold (signal = 0)
        
        return signals
    
    def _threshold_based_signals(self, actual_prices: np.ndarray, predicted_prices: np.ndarray, params: Dict) -> np.ndarray:
        """Trading signals based on prediction confidence threshold."""
        signals = np.zeros(len(predicted_prices))
        threshold = params['confidence_threshold']
        
        for i in range(1, len(predicted_prices)):
            # Calculate predicted return
            predicted_return = (predicted_prices[i] - actual_prices[i-1]) / actual_prices[i-1]
            
            # Only trade if confidence (absolute predicted return) exceeds threshold
            if abs(predicted_return) > threshold:
                if predicted_return > 0:
                    signals[i] = 1  # Strong buy signal
                else:
                    signals[i] = -1  # Strong sell signal
        
        return signals
    
    def _mean_reversion_signals(self, actual_prices: np.ndarray, predicted_prices: np.ndarray, params: Dict) -> np.ndarray:
        """Mean reversion trading strategy."""
        signals = np.zeros(len(predicted_prices))
        
        # Calculate rolling mean (lookback window)
        window = min(20, len(actual_prices) // 4)  # 20-day or 1/4 of data
        
        for i in range(window, len(predicted_prices)):
            # Calculate recent mean
            recent_mean = np.mean(actual_prices[i-window:i])
            current_price = actual_prices[i-1] if i > 0 else actual_prices[0]
            
            # Mean reversion logic: buy when below mean, sell when above
            deviation = (current_price - recent_mean) / recent_mean
            
            if deviation < -0.02:  # 2% below mean
                signals[i] = 1  # Buy (expect reversion up)
            elif deviation > 0.02:  # 2% above mean
                signals[i] = -1  # Sell (expect reversion down)
        
        return signals
    
    def _momentum_signals(self, actual_prices: np.ndarray, predicted_prices: np.ndarray, params: Dict) -> np.ndarray:
        """Momentum trading strategy."""
        signals = np.zeros(len(predicted_prices))
        
        # Calculate momentum (rate of change)
        momentum_window = min(10, len(actual_prices) // 6)
        
        for i in range(momentum_window, len(predicted_prices)):
            # Calculate price momentum
            price_momentum = (actual_prices[i-1] - actual_prices[i-momentum_window]) / actual_prices[i-momentum_window]
            
            # Calculate predicted momentum
            if i < len(predicted_prices):
                pred_momentum = (predicted_prices[i] - actual_prices[i-1]) / actual_prices[i-1]
            else:
                pred_momentum = 0
            
            # Momentum strategy: follow the trend if model agrees
            if price_momentum > 0.01 and pred_momentum > 0:  # Upward momentum
                signals[i] = 1
            elif price_momentum < -0.01 and pred_momentum < 0:  # Downward momentum
                signals[i] = -1
        
        return signals
    
    def _volatility_adjusted_signals(self, actual_prices: np.ndarray, predicted_prices: np.ndarray, params: Dict) -> np.ndarray:
        """Volatility-adjusted position sizing."""
        signals = np.zeros(len(predicted_prices))
        
        # Calculate rolling volatility
        vol_window = min(20, len(actual_prices) // 4)
        
        for i in range(vol_window, len(predicted_prices)):
            # Calculate recent volatility
            recent_returns = np.diff(actual_prices[i-vol_window:i]) / actual_prices[i-vol_window:i-1]
            volatility = np.std(recent_returns)
            
            # Calculate predicted return
            if i > 0:
                predicted_return = (predicted_prices[i] - actual_prices[i-1]) / actual_prices[i-1]
            else:
                predicted_return = 0
            
            # Adjust position size based on volatility (higher vol = smaller position)
            if volatility > 0:
                position_size = min(abs(predicted_return) / volatility, params['max_position_size'])
                
                if predicted_return > 0.005:  # 0.5% minimum threshold
                    signals[i] = position_size
                elif predicted_return < -0.005:
                    signals[i] = -position_size
        
        return signals
    
    def _simulate_trading(self, prices: np.ndarray, signals: np.ndarray, params: Dict) -> Dict:
        """
        Simulate trading based on signals and calculate returns.
        """
        portfolio_value = 1.0  # Start with $1
        position = 0.0  # Current position (-1 to 1)
        cash = 1.0
        transaction_costs = 0.0
        trades = []
        portfolio_values = [portfolio_value]
        
        for i in range(1, len(prices)):
            prev_price = prices[i-1]
            current_price = prices[i]
            signal = signals[i]
            
            # Calculate return if we have a position
            if position != 0:
                price_return = (current_price - prev_price) / prev_price
                position_return = position * price_return
                portfolio_value *= (1 + position_return)
            
            # Process trading signal
            if signal != 0 and abs(signal - position) > 0.01:  # Minimum change threshold
                # Calculate trade
                trade_size = signal - position
                trade_cost = abs(trade_size) * params['transaction_cost']
                
                # Record trade
                trades.append({
                    'day': i,
                    'price': current_price,
                    'signal': signal,
                    'trade_size': trade_size,
                    'trade_cost': trade_cost,
                    'portfolio_value_before': portfolio_value
                })
                
                # Apply transaction cost
                portfolio_value *= (1 - trade_cost)
                transaction_costs += trade_cost
                
                # Update position
                position = signal
            
            portfolio_values.append(portfolio_value)
        
        # Calculate final metrics
        total_return = portfolio_value - 1.0
        buy_hold_return = (prices[-1] - prices[0]) / prices[0]
        
        return {
            'portfolio_values': portfolio_values,
            'final_portfolio_value': portfolio_value,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'excess_return': total_return - buy_hold_return,
            'transaction_costs': transaction_costs,
            'trades': trades,
            'final_position': position
        }
    
    def _calculate_performance_metrics(self, actual_prices: np.ndarray, predicted_prices: np.ndarray, trading_results: Dict) -> Dict:
        """Calculate comprehensive performance metrics."""
        
        portfolio_values = np.array(trading_results['portfolio_values'])
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Basic return metrics
        total_return = trading_results['total_return']
        buy_hold_return = trading_results['buy_hold_return']
        excess_return = total_return - buy_hold_return
        
        # Risk metrics
        volatility = np.std(portfolio_returns) * np.sqrt(252) if len(portfolio_returns) > 1 else 0
        
        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = (np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)) if np.std(portfolio_returns) > 0 else 0
        
        # Maximum drawdown
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - running_max) / running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        # Win rate and trade analysis
        trades = trading_results['trades']
        winning_trades = 0
        losing_trades = 0
        
        if len(trades) > 1:
            for i in range(1, len(trades)):
                prev_trade = trades[i-1]
                current_trade = trades[i]
                
                # Calculate trade return
                entry_price = prev_trade['price']
                exit_price = current_trade['price']
                position = prev_trade['signal']
                
                if position > 0:  # Long position
                    trade_return = (exit_price - entry_price) / entry_price
                elif position < 0:  # Short position
                    trade_return = (entry_price - exit_price) / entry_price
                else:
                    trade_return = 0
                
                if trade_return > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1
        
        total_trades = winning_trades + losing_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calmar ratio (return / max drawdown)
        calmar_ratio = abs(total_return / max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio (downside deviation)
        negative_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 1 else 0
        sortino_ratio = (np.mean(portfolio_returns) * np.sqrt(252) / downside_std) if downside_std > 0 else 0
        
        return {
            'total_return': float(total_return),
            'buy_hold_return': float(buy_hold_return),
            'excess_return': float(excess_return),
            'annualized_volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'calmar_ratio': float(calmar_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'total_trades': int(total_trades),
            'winning_trades': int(winning_trades),
            'losing_trades': int(losing_trades),
            'transaction_costs': float(trading_results['transaction_costs'])
        }
    
    def _calculate_prediction_metrics(self, actual_prices: np.ndarray, predicted_prices: np.ndarray) -> Dict:
        """Calculate prediction accuracy metrics."""
        
        # Basic error metrics
        errors = predicted_prices.flatten() - actual_prices.flatten()
        mae = np.mean(np.abs(errors))
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs(errors) / np.abs(actual_prices)) * 100
        
        # R-squared
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((actual_prices - np.mean(actual_prices)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Directional accuracy
        actual_directions = np.sign(np.diff(actual_prices))
        predicted_directions = np.sign(np.diff(predicted_prices.flatten()))
        
        directional_accuracy = np.mean(actual_directions == predicted_directions) if len(actual_directions) > 0 else 0
        
        # Theil's U statistic (forecast accuracy relative to naive forecast)
        if len(actual_prices) > 1:
            naive_forecast = actual_prices[:-1]  # Use previous price as forecast
            naive_errors = actual_prices[1:] - naive_forecast
            naive_mse = np.mean(naive_errors ** 2)
            
            model_errors = predicted_prices.flatten()[1:] - actual_prices[1:]
            model_mse = np.mean(model_errors ** 2)
            
            theil_u = np.sqrt(model_mse) / np.sqrt(naive_mse) if naive_mse > 0 else float('inf')
        else:
            theil_u = float('inf')
        
        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            'r_squared': float(r_squared),
            'directional_accuracy': float(directional_accuracy),
            'theil_u': float(theil_u)
        }
    
    def run_strategy_comparison(self, 
                              model: torch.nn.Module,
                              test_features: np.ndarray,
                              test_targets: np.ndarray,
                              ticker: str,
                              preprocessor=None) -> Dict[str, Dict]:
        """
        Compare performance across all available trading strategies.
        
        Returns:
            Dictionary with results for each strategy
        """
        print(f"🔍 Comparing trading strategies for {ticker}...")
        
        strategy_results = {}
        
        for strategy in self.trading_strategies:
            print(f"  Testing {strategy} strategy...")
            
            try:
                results = self.run_backtest(
                    model=model,
                    test_features=test_features,
                    test_targets=test_targets,
                    ticker=ticker,
                    preprocessor=preprocessor,
                    strategy=strategy
                )
                
                strategy_results[strategy] = results
                
                # Print quick summary
                if 'total_return' in results:
                    total_ret = results['total_return']
                    sharpe = results.get('sharpe_ratio', 0)
                    max_dd = results.get('max_drawdown', 0)
                    print(f"    ✅ {strategy}: Return={total_ret:.1%}, Sharpe={sharpe:.2f}, MaxDD={max_dd:.1%}")
                else:
                    print(f"    ❌ {strategy}: Failed")
                    
            except Exception as e:
                print(f"    ❌ {strategy}: Error - {e}")
                strategy_results[strategy] = {'error': str(e)}
        
        # Find best strategy
        valid_strategies = {k: v for k, v in strategy_results.items() if 'total_return' in v}
        
        if valid_strategies:
            # Best by Sharpe ratio
            best_sharpe_strategy = max(valid_strategies.keys(), 
                                     key=lambda s: valid_strategies[s].get('sharpe_ratio', -999))
            
            # Best by total return
            best_return_strategy = max(valid_strategies.keys(),
                                     key=lambda s: valid_strategies[s].get('total_return', -999))
            
            print(f"\n🏆 Best strategy by Sharpe ratio: {best_sharpe_strategy}")
            print(f"🏆 Best strategy by total return: {best_return_strategy}")
            
            strategy_results['summary'] = {
                'best_sharpe_strategy': best_sharpe_strategy,
                'best_return_strategy': best_return_strategy,
                'best_sharpe_value': valid_strategies[best_sharpe_strategy]['sharpe_ratio'],
                'best_return_value': valid_strategies[best_return_strategy]['total_return']
            }
        
        return strategy_results
    
    def generate_backtest_report(self, backtest_results: Dict) -> str:
        """Generate a detailed backtest report."""
        
        if 'error' in backtest_results:
            return f"Backtest failed: {backtest_results['error']}"
        
        ticker = backtest_results.get('ticker', 'Unknown')
        strategy = backtest_results.get('strategy', 'Unknown')
        
        # Extract metrics
        perf_metrics = backtest_results.get('performance_metrics', {})
        pred_metrics = backtest_results.get('prediction_metrics', {})
        trading_results = backtest_results.get('trading_results', {})
        
        report = []
        report.append("BACKTEST REPORT")
        report.append("=" * 50)
        report.append(f"Ticker: {ticker}")
        report.append(f"Strategy: {strategy}")
        report.append(f"Test Period: {backtest_results.get('n_test_days', 'N/A')} days")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Performance Summary
        report.append("PERFORMANCE SUMMARY:")
        report.append("-" * 30)
        report.append(f"Total Return: {perf_metrics.get('total_return', 0):.2%}")
        report.append(f"Buy & Hold Return: {perf_metrics.get('buy_hold_return', 0):.2%}")
        report.append(f"Excess Return: {perf_metrics.get('excess_return', 0):.2%}")
        report.append(f"Annualized Volatility: {perf_metrics.get('annualized_volatility', 0):.2%}")
        report.append(f"Sharpe Ratio: {perf_metrics.get('sharpe_ratio', 0):.3f}")
        report.append(f"Sortino Ratio: {perf_metrics.get('sortino_ratio', 0):.3f}")
        report.append(f"Calmar Ratio: {perf_metrics.get('calmar_ratio', 0):.3f}")
        report.append(f"Maximum Drawdown: {perf_metrics.get('max_drawdown', 0):.2%}")
        report.append("")
        
        # Trading Statistics
        report.append("TRADING STATISTICS:")
        report.append("-" * 30)
        report.append(f"Total Trades: {perf_metrics.get('total_trades', 0)}")
        report.append(f"Winning Trades: {perf_metrics.get('winning_trades', 0)}")
        report.append(f"Losing Trades: {perf_metrics.get('losing_trades', 0)}")
        report.append(f"Win Rate: {perf_metrics.get('win_rate', 0):.1%}")
        report.append(f"Transaction Costs: {perf_metrics.get('transaction_costs', 0):.3%}")
        report.append("")
        
        # Prediction Accuracy
        report.append("PREDICTION ACCURACY:")
        report.append("-" * 30)
        report.append(f"RMSE: ${pred_metrics.get('rmse', 0):.2f}")
        report.append(f"MAE: ${pred_metrics.get('mae', 0):.2f}")
        report.append(f"MAPE: {pred_metrics.get('mape', 0):.1f}%")
        report.append(f"R-Squared: {pred_metrics.get('r_squared', 0):.3f}")
        report.append(f"Directional Accuracy: {pred_metrics.get('directional_accuracy', 0):.1%}")
        report.append(f"Theil's U: {pred_metrics.get('theil_u', 0):.3f}")
        report.append("")
        
        # Strategy Assessment
        report.append("STRATEGY ASSESSMENT:")
        report.append("-" * 30)
        
        total_return = perf_metrics.get('total_return', 0)
        buy_hold = perf_metrics.get('buy_hold_return', 0)
        sharpe = perf_metrics.get('sharpe_ratio', 0)
        win_rate = perf_metrics.get('win_rate', 0)
        
        if total_return > buy_hold:
            report.append("✅ Strategy outperformed buy-and-hold")
        else:
            report.append("❌ Strategy underperformed buy-and-hold")
        
        if sharpe > 1.0:
            report.append("✅ Strong risk-adjusted returns (Sharpe > 1.0)")
        elif sharpe > 0.5:
            report.append("⚠️  Moderate risk-adjusted returns (Sharpe 0.5-1.0)")
        else:
            report.append("❌ Poor risk-adjusted returns (Sharpe < 0.5)")
        
        if win_rate > 0.55:
            report.append("✅ Good predictive accuracy (Win rate > 55%)")
        elif win_rate > 0.45:
            report.append("⚠️  Moderate predictive accuracy (Win rate 45-55%)")
        else:
            report.append("❌ Poor predictive accuracy (Win rate < 45%)")
        
        return "\n".join(report)
    
    def calculate_rolling_metrics(self, 
                                portfolio_values: List[float], 
                                window: int = 20) -> Dict[str, List[float]]:
        """
        Calculate rolling performance metrics.
        
        Args:
            portfolio_values: List of portfolio values over time
            window: Rolling window size
        
        Returns:
            Dictionary with rolling metrics
        """
        values = np.array(portfolio_values)
        returns = np.diff(values) / values[:-1]
        
        rolling_metrics = {
            'rolling_return': [],
            'rolling_volatility': [],
            'rolling_sharpe': [],
            'rolling_drawdown': []
        }
        
        for i in range(window, len(values)):
            # Rolling return
            period_return = (values[i] - values[i-window]) / values[i-window]
            rolling_metrics['rolling_return'].append(period_return)
            
            # Rolling volatility
            period_returns = returns[i-window:i]
            period_vol = np.std(period_returns) * np.sqrt(252) if len(period_returns) > 1 else 0
            rolling_metrics['rolling_volatility'].append(period_vol)
            
            # Rolling Sharpe
            if period_vol > 0:
                period_sharpe = np.mean(period_returns) / np.std(period_returns) * np.sqrt(252)
            else:
                period_sharpe = 0
            rolling_metrics['rolling_sharpe'].append(period_sharpe)
            
            # Rolling drawdown
            period_values = values[i-window:i+1]
            period_peak = np.max(period_values[:-1])
            current_dd = (values[i] - period_peak) / period_peak if period_peak > 0 else 0
            rolling_metrics['rolling_drawdown'].append(current_dd)
        
        return rolling_metrics
    
    def optimize_strategy_parameters(self, 
                                   model: torch.nn.Module,
                                   test_features: np.ndarray,
                                   test_targets: np.ndarray,
                                   ticker: str,
                                   strategy: str = 'threshold_based',
                                   preprocessor=None) -> Dict:
        """
        Optimize strategy parameters using grid search.
        
        Returns:
            Best parameters and their performance
        """
        print(f"🔧 Optimizing {strategy} strategy parameters for {ticker}...")
        
        if strategy == 'threshold_based':
            param_grid = {
                'confidence_threshold': [0.01, 0.02, 0.03, 0.05, 0.07, 0.10],
                'transaction_cost': [0.0005, 0.001, 0.002]
            }
        elif strategy == 'volatility_adjusted':
            param_grid = {
                'max_position_size': [0.5, 0.7, 1.0, 1.5],
                'transaction_cost': [0.0005, 0.001, 0.002]
            }
        else:
            # Default grid for other strategies
            param_grid = {
                'transaction_cost': [0.0005, 0.001, 0.002, 0.003]
            }
        
        best_sharpe = -999
        best_params = None
        best_results = None
        
        # Generate parameter combinations
        from itertools import product
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for param_combination in product(*param_values):
            params = dict(zip(param_names, param_combination))
            
            try:
                # Run backtest with these parameters
                results = self.run_backtest(
                    model=model,
                    test_features=test_features,
                    test_targets=test_targets,
                    ticker=ticker,
                    preprocessor=preprocessor,
                    strategy=strategy,
                    trading_params=params
                )
                
                # Check if this is the best result
                current_sharpe = results.get('sharpe_ratio', -999)
                if current_sharpe > best_sharpe:
                    best_sharpe = current_sharpe
                    best_params = params
                    best_results = results
                    
            except Exception as e:
                print(f"    Parameter optimization failed for {params}: {e}")
                continue
        
        if best_params:
            print(f"✅ Best parameters found:")
            for param, value in best_params.items():
                print(f"    {param}: {value}")
            print(f"    Best Sharpe ratio: {best_sharpe:.3f}")
        else:
            print("❌ Parameter optimization failed")
        
        return {
            'best_parameters': best_params,
            'best_sharpe_ratio': best_sharpe,
            'best_results': best_results,
            'optimization_strategy': strategy
        }
