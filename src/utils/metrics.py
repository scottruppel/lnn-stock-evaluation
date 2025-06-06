import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

class StockPredictionMetrics:
    """
    Comprehensive metrics for evaluating stock prediction models.
    Includes traditional ML metrics and finance-specific metrics.
    """
    
    def __init__(self):
        self.metric_history = []
        
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate basic regression metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
        
        Returns:
            Dictionary with basic metrics
        """
        # Flatten arrays to ensure 1D
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Remove any NaN values
        mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
        y_true_clean = y_true_flat[mask]
        y_pred_clean = y_pred_flat[mask]
        
        if len(y_true_clean) == 0:
            return {metric: np.nan for metric in ['mse', 'rmse', 'mae', 'mape', 'r2']}
        
        # Calculate metrics
        mse = mean_squared_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        
        # MAPE (avoid division by zero)
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / np.where(y_true_clean != 0, y_true_clean, 1))) * 100
        
        # R-squared
        r2 = r2_score(y_true_clean, y_pred_clean)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2
        }
    
    def calculate_directional_accuracy(self, actual_returns, predicted_returns):
        """
        Calculate directional accuracy using return data.
    
        Args:
            actual_returns: Array of actual returns
            predicted_returns: Array of predicted returns
        """
        # Remove any NaN or infinite values
        mask = np.isfinite(actual_returns) & np.isfinite(predicted_returns)
        actual_clean = actual_returns[mask]
        predicted_clean = predicted_returns[mask]
    
        if len(actual_clean) == 0:
            return {'directional_accuracy': 0.0, 'error': 'No valid data for directional accuracy'}
    
        # Calculate direction (up/down)
        actual_direction = np.sign(actual_clean)
        predicted_direction = np.sign(predicted_clean)
    
        # Calculate accuracy
        directional_accuracy = np.mean(actual_direction == predicted_direction)
    
        # Additional directional metrics
        up_predictions = np.sum(predicted_direction > 0)
        down_predictions = np.sum(predicted_direction < 0)
        neutral_predictions = np.sum(predicted_direction == 0)
    
        # Precision and recall for up movements
        actual_up = actual_direction > 0
        predicted_up = predicted_direction > 0
    
        up_precision = np.sum(actual_up & predicted_up) / np.sum(predicted_up) if np.sum(predicted_up) > 0 else 0
        up_recall = np.sum(actual_up & predicted_up) / np.sum(actual_up) if np.sum(actual_up) > 0 else 0
    
        return {
            'directional_accuracy': directional_accuracy,
            'up_predictions': up_predictions,
            'down_predictions': down_predictions,
            'neutral_predictions': neutral_predictions,
            'up_precision': up_precision,
            'up_recall': up_recall,
            'total_predictions': len(actual_clean)
        }
    
    def calculate_trading_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                initial_capital: float = 10000) -> Dict[str, float]:
        """
        Calculate trading strategy performance metrics.
        
        Args:
            y_true: Actual prices
            y_pred: Predicted prices
            initial_capital: Starting capital for trading simulation
        
        Returns:
            Dictionary with trading performance metrics
        """
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        if len(y_true_flat) < 2:
            return {metric: np.nan for metric in ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']}
        
        # Calculate actual returns
        actual_returns = np.diff(y_true_flat) / y_true_flat[:-1]
        
        # Calculate predicted directions
        predicted_directions = np.sign(np.diff(y_pred_flat))
        
        # Simple trading strategy: go long when model predicts up, short when predicts down
        strategy_returns = predicted_directions * actual_returns
        
        # Remove NaN values
        strategy_returns = strategy_returns[~np.isnan(strategy_returns)]
        actual_returns = actual_returns[~np.isnan(actual_returns)]
        
        if len(strategy_returns) == 0:
            return {metric: np.nan for metric in ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']}
        
        # Calculate cumulative returns
        strategy_cumulative = np.cumprod(1 + strategy_returns) - 1
        actual_cumulative = np.cumprod(1 + actual_returns) - 1
        
        # Total return
        total_return = strategy_cumulative[-1]
        buy_hold_return = actual_cumulative[-1]
        
        # Sharpe ratio (assume risk-free rate = 0)
        if np.std(strategy_returns) > 0:
            sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        max_drawdown = self._calculate_max_drawdown(strategy_cumulative)
        
        # Win rate
        win_rate = np.sum(strategy_returns > 0) / len(strategy_returns)
        
        # Alpha (excess return over buy-and-hold)
        alpha = total_return - buy_hold_return
        
        return {
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'alpha': alpha,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'volatility': np.std(strategy_returns) * np.sqrt(252)  # Annualized
        }
    
    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown from cumulative returns."""
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative_returns)
        
        # Calculate drawdown at each point
        drawdown = (cumulative_returns - running_max) / (1 + running_max)
        
        # Return maximum drawdown (most negative value)
        return np.min(drawdown)
    
    def calculate_time_based_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   dates: Optional[pd.DatetimeIndex] = None) -> Dict[str, Dict]:
        """
        Calculate metrics broken down by time periods.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            dates: DateTime index for time-based grouping
        
        Returns:
            Dictionary with metrics for different time periods
        """
        if dates is None:
            # Create simple date range if not provided
            dates = pd.date_range(start='2020-01-01', periods=len(y_true), freq='D')
        
        # Create DataFrame
        df = pd.DataFrame({
            'actual': y_true.flatten(),
            'predicted': y_pred.flatten(),
            'date': dates[:len(y_true.flatten())]
        })
        
        results = {}
        
        # Monthly metrics
        try:
            monthly_data = df.set_index('date').resample('M').mean().dropna()
            if len(monthly_data) > 1:
                monthly_metrics = self.calculate_basic_metrics(
                    monthly_data['actual'].values, 
                    monthly_data['predicted'].values
                )
                results['monthly'] = monthly_metrics
        except Exception as e:
            print(f"Could not calculate monthly metrics: {e}")
            results['monthly'] = {}
        
        # Quarterly metrics
        try:
            quarterly_data = df.set_index('date').resample('Q').mean().dropna()
            if len(quarterly_data) > 1:
                quarterly_metrics = self.calculate_basic_metrics(
                    quarterly_data['actual'].values,
                    quarterly_data['predicted'].values
                )
                results['quarterly'] = quarterly_metrics
        except Exception as e:
            print(f"Could not calculate quarterly metrics: {e}")
            results['quarterly'] = {}
        
        return results
    
    def calculate_residual_analysis(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Analyze prediction residuals for model diagnostics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
        
        Returns:
            Dictionary with residual analysis metrics
        """
        # Calculate residuals
        residuals = y_true.flatten() - y_pred.flatten()
        residuals = residuals[~np.isnan(residuals)]
        
        if len(residuals) == 0:
            return {metric: np.nan for metric in ['residual_mean', 'residual_std', 'residual_skew', 'residual_kurt']}
        
        # Basic residual statistics
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        
        # Skewness and kurtosis
        if len(residuals) > 2:
            residual_skew = self._calculate_skewness(residuals)
            residual_kurt = self._calculate_kurtosis(residuals)
        else:
            residual_skew = np.nan
            residual_kurt = np.nan
        
        # Autocorrelation of residuals (lag 1)
        if len(residuals) > 1:
            residual_autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
            if np.isnan(residual_autocorr):
                residual_autocorr = 0
        else:
            residual_autocorr = np.nan
        
        return {
            'residual_mean': residual_mean,
            'residual_std': residual_std,
            'residual_skew': residual_skew,
            'residual_kurt': residual_kurt,
            'residual_autocorr': residual_autocorr
        }
        
    def calculate_risk_metrics(self, y_true, y_pred):
        """
        Calculate risk-related metrics for the predictions.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            dict: Risk metrics
        """
        # Convert to numpy arrays
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Remove any NaN or infinite values
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return {'error': 'No valid data for risk metrics'}
        
        # Calculate prediction errors
        errors = y_pred_clean - y_true_clean
        
        # Risk metrics using your existing helper methods
        risk_metrics = {
            'prediction_volatility': np.std(errors),
            'prediction_skewness': self._calculate_skewness(errors),
            'prediction_kurtosis': self._calculate_kurtosis(errors),
            'error_percentiles': {
                '5th': np.percentile(errors, 5),
                '25th': np.percentile(errors, 25),
                '75th': np.percentile(errors, 75),
                '95th': np.percentile(errors, 95)
            },
            'maximum_error': np.max(np.abs(errors)),
            'error_range': np.max(errors) - np.min(errors)
        }
        
        return risk_metrics
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0
        skew = np.mean(((data - mean_val) / std_val) ** 3)
        return skew
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate excess kurtosis of data."""
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0
        kurt = np.mean(((data - mean_val) / std_val) ** 4) - 3  # Excess kurtosis
        return kurt
    
    def comprehensive_evaluation(self, y_true, y_pred, dates=None, return_predictions=None, actual_returns=None):
        """
        Comprehensive evaluation supporting both price and return metrics.
    
        Args:
            y_true: True values (can be prices or returns)
            y_pred: Predicted values (can be prices or returns)
            dates: Date index for the data
            return_predictions: Predicted returns (optional, for directional accuracy)
            actual_returns: Actual returns (optional, for directional accuracy)
        """
        results = {}
    
        # Basic regression metrics (on whatever data is provided)
        results['basic_metrics'] = self.calculate_basic_metrics(y_true, y_pred)
    
        # Directional accuracy (use returns if provided, otherwise use differences)
        if return_predictions is not None and actual_returns is not None:
            # Use the provided return data for directional accuracy
            results['directional_metrics'] = self.calculate_directional_accuracy(
                actual_returns, return_predictions
            )
        else:
            # Fall back to using differences of the main data
            actual_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            directional_accuracy = np.mean(actual_direction == pred_direction)
            results['directional_metrics'] = {'directional_accuracy': directional_accuracy}
    
        # Trading metrics (use main data, whether prices or returns)
        results['trading_metrics'] = self.calculate_trading_metrics(y_true, y_pred, dates)
    
        # Risk metrics
        results['risk_metrics'] = self.calculate_risk_metrics(y_true, y_pred)
    
        return results
        
    def compare_models(self, results_dict: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple model results.
        
        Args:
            results_dict: Dictionary with model names as keys and metrics as values
        
        Returns:
            DataFrame comparing models across key metrics
        """
        comparison_data = []
        
        for model_name, metrics in results_dict.items():
            row = {'model': model_name}
            
            # Extract key metrics
            if 'basic_metrics' in metrics:
                row.update({f"basic_{k}": v for k, v in metrics['basic_metrics'].items()})
            
            if 'directional_metrics' in metrics:
                row.update({f"directional_{k}": v for k, v in metrics['directional_metrics'].items()})
            
            if 'trading_metrics' in metrics:
                row.update({f"trading_{k}": v for k, v in metrics['trading_metrics'].items()})
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def get_metric_summary(self, metrics: Dict) -> str:
        """
        Get a human-readable summary of metrics.
        
        Args:
            metrics: Dictionary of calculated metrics
        
        Returns:
            Formatted string summary
        """
        summary_lines = []
        summary_lines.append("=== Model Performance Summary ===")
        
        if 'basic_metrics' in metrics:
            basic = metrics['basic_metrics']
            summary_lines.append(f"RMSE: {basic.get('rmse', 'N/A'):.4f}")
            summary_lines.append(f"MAE: {basic.get('mae', 'N/A'):.4f}")
            summary_lines.append(f"MAPE: {basic.get('mape', 'N/A'):.2f}%")
            summary_lines.append(f"R²: {basic.get('r2', 'N/A'):.4f}")
        
        if 'directional_metrics' in metrics:
            direction = metrics['directional_metrics']
            summary_lines.append(f"Directional Accuracy: {direction.get('directional_accuracy', 'N/A'):.2f}")
        
        if 'trading_metrics' in metrics:
            trading = metrics['trading_metrics']
            summary_lines.append(f"Total Return: {trading.get('total_return', 'N/A'):.2%}")
            summary_lines.append(f"Sharpe Ratio: {trading.get('sharpe_ratio', 'N/A'):.4f}")
            summary_lines.append(f"Max Drawdown: {trading.get('max_drawdown', 'N/A'):.2%}")
        
        return "\n".join(summary_lines)

class MetricTracker:
    """
    Simple metric tracker for monitoring training progress.
    """
    
    def __init__(self):
        self.train_metrics = []
        self.val_metrics = []
        self.epochs = []
    
    def update(self, epoch: int, train_loss: float, val_loss: float):
        """Update metrics for current epoch."""
        self.epochs.append(epoch)
        self.train_metrics.append(train_loss)
        self.val_metrics.append(val_loss)
    
    def get_best_epoch(self) -> Tuple[int, float]:
        """Get epoch with best validation loss."""
        if not self.val_metrics:
            return 0, float('inf')
        
        best_idx = np.argmin(self.val_metrics)
        return self.epochs[best_idx], self.val_metrics[best_idx]
    
    def get_training_summary(self) -> Dict:
        """Get summary of training progress."""
        if not self.train_metrics:
            return {}
        
        return {
            'final_train_loss': self.train_metrics[-1],
            'final_val_loss': self.val_metrics[-1],
            'best_val_loss': min(self.val_metrics),
            'best_epoch': self.get_best_epoch()[0],
            'total_epochs': len(self.epochs)
        }
