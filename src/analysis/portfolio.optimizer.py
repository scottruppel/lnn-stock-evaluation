# src/analysis/portfolio_optimizer.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    """
    Portfolio optimization using modern portfolio theory and advanced methods.
    Incorporates forecast reliability and uncertainty into optimization.
    """
    
    def __init__(self):
        self.optimization_methods = [
            'mean_variance',
            'risk_parity', 
            'max_sharpe',
            'min_volatility',
            'reliability_weighted'
        ]
    
    def optimize_portfolio(self, 
                         asset_data: Dict[str, Dict],
                         method: str = 'mean_variance',
                         risk_tolerance: float = 0.1,
                         max_position_size: float = 0.4,
                         min_position_size: float = 0.0) -> Dict[str, float]:
        """
        Optimize portfolio allocation based on expected returns, volatility, and reliability.
        
        Args:
            asset_data: Dictionary with asset data including expected_return, volatility, reliability
            method: Optimization method to use
            risk_tolerance: Risk tolerance parameter (0 = risk averse, 1 = risk seeking)
            max_position_size: Maximum allocation to any single asset
            min_position_size: Minimum allocation to any single asset
        
        Returns:
            Dictionary of optimal weights for each asset
        """
        print(f"🎯 Optimizing portfolio using {method} method...")
        
        if not asset_data:
            print("❌ No asset data provided for optimization")
            return {}
        
        # Extract data
        assets = list(asset_data.keys())
        n_assets = len(assets)
        
        expected_returns = np.array([asset_data[asset]['expected_return'] for asset in assets])
        volatilities = np.array([asset_data[asset]['volatility'] for asset in assets])
        reliabilities = np.array([asset_data[asset].get('reliability', 0.5) for asset in assets])
        
        # Handle edge cases
        if np.any(volatilities <= 0):
            print("⚠️  Some assets have zero or negative volatility, setting to minimum")
            volatilities = np.maximum(volatilities, 0.001)
        
        try:
            if method == 'mean_variance':
                weights = self._mean_variance_optimization(
                    expected_returns, volatilities, risk_tolerance,
                    max_position_size, min_position_size, n_assets
                )
            elif method == 'risk_parity':
                weights = self._risk_parity_optimization(
                    volatilities, max_position_size, min_position_size, n_assets
                )
            elif method == 'max_sharpe':
                weights = self._max_sharpe_optimization(
                    expected_returns, volatilities, max_position_size, min_position_size, n_assets
                )
            elif method == 'min_volatility':
                weights = self._min_volatility_optimization(
                    volatilities, max_position_size, min_position_size, n_assets
                )
            elif method == 'reliability_weighted':
                weights = self._reliability_weighted_optimization(
                    expected_returns, volatilities, reliabilities,
                    risk_tolerance, max_position_size, min_position_size, n_assets
                )
            else:
                print(f"⚠️  Unknown optimization method: {method}. Using mean_variance.")
                weights = self._mean_variance_optimization(
                    expected_returns, volatilities, risk_tolerance,
                    max_position_size, min_position_size, n_assets
                )
            
            # Create result dictionary
            result = {assets[i]: weights[i] for i in range(n_assets)}
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(
                weights, expected_returns, volatilities, reliabilities
            )
            
            print("✅ Portfolio optimization completed")
            print(f"   Expected Return: {portfolio_metrics['expected_return']:.2%}")
            print(f"   Expected Volatility: {portfolio_metrics['volatility']:.2%}")
            print(f"   Expected Sharpe: {portfolio_metrics['sharpe_ratio']:.3f}")
            print(f"   Average Reliability: {portfolio_metrics['avg_reliability']:.3f}")
            
            return result
            
        except Exception as e:
            print(f"❌ Portfolio optimization failed: {e}")
            # Return equal weights as fallback
            equal_weight = 1.0 / n_assets
            return {asset: equal_weight for asset in assets}
    
    def _mean_variance_optimization(self, 
                                  expected_returns: np.ndarray,
                                  volatilities: np.ndarray,
                                  risk_tolerance: float,
                                  max_pos: float,
                                  min_pos: float,
                                  n_assets: int) -> np.ndarray:
        """Classic mean-variance optimization."""
        
        # Create covariance matrix (diagonal - assumes no correlation)
        cov_matrix = np.diag(volatilities ** 2)
        
        # Objective function: maximize utility = return - (risk_aversion/2) * variance
        risk_aversion = (1 - risk_tolerance) * 10  # Scale risk aversion
        
        def objective(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            utility = portfolio_return - (risk_aversion / 2) * portfolio_variance
            return -utility  # Minimize negative utility
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]
        
        # Bounds
        bounds = [(min_pos, max_pos) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x
        else:
            print("⚠️  Mean-variance optimization failed, using equal weights")
            return np.ones(n_assets) / n_assets
    
    def _risk_parity_optimization(self, 
                                volatilities: np.ndarray,
                                max_pos: float,
                                min_pos: float,
                                n_assets: int) -> np.ndarray:
        """Risk parity optimization - equal risk contribution."""
        
        # Inverse volatility weighting (simple risk parity)
        inv_vol = 1.0 / volatilities
        weights = inv_vol / np.sum(inv_vol)
        
        # Apply position limits
        weights = np.clip(weights, min_pos, max_pos)
        
        # Renormalize
        weights = weights / np.sum(weights)
        
        return weights
    
    def _max_sharpe_optimization(self, 
                               expected_returns: np.ndarray,
                               volatilities: np.ndarray,
                               max_pos: float,
                               min_pos: float,
                               n_assets: int) -> np.ndarray:
        """Maximize Sharpe ratio optimization."""
        
        # Create covariance matrix (diagonal)
        cov_matrix = np.diag(volatilities ** 2)
        
        def objective(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_std = np.sqrt(portfolio_variance)
            
            if portfolio_std == 0:
                return -np.inf
            
            sharpe_ratio = portfolio_return / portfolio_std
            return -sharpe_ratio  # Minimize negative Sharpe
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        # Bounds
        bounds = [(min_pos, max_pos) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x
        else:
            print("⚠️  Max Sharpe optimization failed, using equal weights")
            return np.ones(n_assets) / n_assets
    
    def _min_volatility_optimization(self, 
                                   volatilities: np.ndarray,
                                   max_pos: float,
                                   min_pos: float,
                                   n_assets: int) -> np.ndarray:
        """Minimize portfolio volatility."""
        
        # Create covariance matrix (diagonal)
        cov_matrix = np.diag(volatilities ** 2)
        
        def objective(weights):
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            return portfolio_variance
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        # Bounds
        bounds = [(min_pos, max_pos) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x
        else:
            print("⚠️  Min volatility optimization failed, using equal weights")
            return np.ones(n_assets) / n_assets
    
    def _reliability_weighted_optimization(self, 
                                         expected_returns: np.ndarray,
                                         volatilities: np.ndarray,
                                         reliabilities: np.ndarray,
                                         risk_tolerance: float,
                                         max_pos: float,
                                         min_pos: float,
                                         n_assets: int) -> np.ndarray:
        """
        Advanced optimization incorporating forecast reliability.
        This is unique to your advanced forecasting system.
        """
        
        # Adjust expected returns by reliability
        # Higher reliability = more confidence in the forecast
        adjusted_returns = expected_returns * reliabilities
        
        # Adjust volatilities by reliability (less reliable = higher perceived risk)
        # Lower reliability = increase perceived volatility
        reliability_adjustment = 1.0 + (1.0 - reliabilities) * 0.5  # Up to 50% increase
        adjusted_volatilities = volatilities * reliability_adjustment
        
        # Create covariance matrix
        cov_matrix = np.diag(adjusted_volatilities ** 2)
        
        # Objective function with reliability weighting
        risk_aversion = (1 - risk_tolerance) * 10
        
        def objective(weights):
            portfolio_return = np.sum(weights * adjusted_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            
            # Add reliability bonus - higher weight for more reliable assets
            reliability_bonus = np.sum(weights * reliabilities) * 0.1
            
            utility = portfolio_return - (risk_aversion / 2) * portfolio_variance + reliability_bonus
            return -utility
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        # Bounds
        bounds = [(min_pos, max_pos) for _ in range(n_assets)]
        
        # Initial guess weighted by reliability
        reliability_weights = reliabilities / np.sum(reliabilities)
        x0 = reliability_weights
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x
        else:
            print("⚠️  Reliability-weighted optimization failed, using reliability weights")
            return reliability_weights
    
    def _calculate_portfolio_metrics(self, 
                                   weights: np.ndarray,
                                   expected_returns: np.ndarray,
                                   volatilities: np.ndarray,
                                   reliabilities: np.ndarray) -> Dict[str, float]:
        """Calculate portfolio-level metrics."""
        
        portfolio_return = np.sum(weights * expected_returns)
        
        # Portfolio volatility (assuming no correlation)
        portfolio_variance = np.sum((weights * volatilities) ** 2)
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Sharpe ratio
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Average reliability
        avg_reliability = np.sum(weights * reliabilities)
        
        # Concentration (Herfindahl index)
        concentration = np.sum(weights ** 2)
        
        return {
            'expected_return': float(portfolio_return),
            'volatility': float(portfolio_volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'avg_reliability': float(avg_reliability),
            'concentration': float(concentration),
            'diversification_ratio': float(1.0 / concentration)  # Higher = more diversified
        }
    
    def compare_optimization_methods(self, 
                                   asset_data: Dict[str, Dict],
                                   risk_tolerance: float = 0.1) -> Dict[str, Dict]:
        """
        Compare different optimization methods and return results for all.
        Useful for understanding trade-offs between methods.
        """
        print("🔍 Comparing optimization methods...")
        
        results = {}
        
        for method in self.optimization_methods:
            print(f"\n  Testing {method}...")
            try:
                weights = self.optimize_portfolio(
                    asset_data, 
                    method=method, 
                    risk_tolerance=risk_tolerance
                )
                
                if weights:
                    # Calculate metrics for this allocation
                    assets = list(asset_data.keys())
                    expected_returns = np.array([asset_data[asset]['expected_return'] for asset in assets])
                    volatilities = np.array([asset_data[asset]['volatility'] for asset in assets])
                    reliabilities = np.array([asset_data[asset].get('reliability', 0.5) for asset in assets])
                    
                    weights_array = np.array([weights[asset] for asset in assets])
                    
                    metrics = self._calculate_portfolio_metrics(
                        weights_array, expected_returns, volatilities, reliabilities
                    )
                    
                    results[method] = {
                        'weights': weights,
                        'metrics': metrics
                    }
                    
                    print(f"    ✅ {method}: Sharpe={metrics['sharpe_ratio']:.3f}, "
                          f"Return={metrics['expected_return']:.2%}, "
                          f"Vol={metrics['volatility']:.2%}")
                else:
                    print(f"    ❌ {method}: Failed to generate weights")
                    
            except Exception as e:
                print(f"    ❌ {method}: Error - {e}")
        
        # Find best method by Sharpe ratio
        if results:
            best_method = max(results.keys(), 
                            key=lambda m: results[m]['metrics']['sharpe_ratio'])
            print(f"\n🏆 Best method by Sharpe ratio: {best_method}")
            
            results['recommended'] = results[best_method]
        
        return results
    
    def rebalance_portfolio(self, 
                          current_weights: Dict[str, float],
                          target_weights: Dict[str, float],
                          rebalance_threshold: float = 0.05,
                          transaction_cost: float = 0.001) -> Dict[str, Dict]:
        """
        Calculate rebalancing trades needed to reach target allocation.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            rebalance_threshold: Minimum drift before rebalancing
            transaction_cost: Transaction cost per trade (as decimal)
        
        Returns:
            Dictionary with rebalancing information
        """
        print("⚖️  Calculating portfolio rebalancing...")
        
        # Align assets (only consider assets in both portfolios)
        common_assets = set(current_weights.keys()) & set(target_weights.keys())
        
        if not common_assets:
            print("❌ No common assets between current and target portfolios")
            return {}
        
        rebalancing_info = {
            'trades': {},
            'total_turnover': 0.0,
            'estimated_costs': 0.0,
            'assets_to_rebalance': [],
            'should_rebalance': False
        }
        
        total_drift = 0.0
        
        for asset in common_assets:
            current_weight = current_weights.get(asset, 0.0)
            target_weight = target_weights.get(asset, 0.0)
            
            drift = abs(target_weight - current_weight)
            total_drift += drift
            
            if drift > rebalance_threshold:
                trade_amount = target_weight - current_weight
                
                rebalancing_info['trades'][asset] = {
                    'current_weight': current_weight,
                    'target_weight': target_weight,
                    'trade_amount': trade_amount,
                    'action': 'BUY' if trade_amount > 0 else 'SELL'
                }
                
                rebalancing_info['assets_to_rebalance'].append(asset)
        
        # Calculate turnover and costs
        rebalancing_info['total_turnover'] = total_drift / 2  # Divide by 2 to avoid double counting
        rebalancing_info['estimated_costs'] = rebalancing_info['total_turnover'] * transaction_cost
        rebalancing_info['should_rebalance'] = len(rebalancing_info['assets_to_rebalance']) > 0
        
        if rebalancing_info['should_rebalance']:
            print(f"✅ Rebalancing recommended for {len(rebalancing_info['assets_to_rebalance'])} assets")
            print(f"   Total turnover: {rebalancing_info['total_turnover']:.1%}")
            print(f"   Estimated costs: {rebalancing_info['estimated_costs']:.3%}")
        else:
            print("ℹ️  No rebalancing needed - all assets within threshold")
        
        return rebalancing_info
    
    def generate_portfolio_report(self, 
                                weights: Dict[str, float],
                                asset_data: Dict[str, Dict]) -> str:
        """Generate a detailed portfolio analysis report."""
        
        if not weights or not asset_data:
            return "No portfolio data available for report generation."
        
        # Calculate portfolio metrics
        assets = list(weights.keys())
        expected_returns = np.array([asset_data[asset]['expected_return'] for asset in assets])
        volatilities = np.array([asset_data[asset]['volatility'] for asset in assets])
        reliabilities = np.array([asset_data[asset].get('reliability', 0.5) for asset in assets])
        weights_array = np.array([weights[asset] for asset in assets])
        
        metrics = self._calculate_portfolio_metrics(
            weights_array, expected_returns, volatilities, reliabilities
        )
        
        # Generate report
        report = []
        report.append("PORTFOLIO OPTIMIZATION REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Portfolio summary
        report.append("PORTFOLIO SUMMARY:")
        report.append("-" * 30)
        report.append(f"Expected Annual Return: {metrics['expected_return']:.2%}")
        report.append(f"Expected Annual Volatility: {metrics['volatility']:.2%}")
        report.append(f"Expected Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        report.append(f"Average Forecast Reliability: {metrics['avg_reliability']:.3f}")
        report.append(f"Portfolio Concentration: {metrics['concentration']:.3f}")
        report.append(f"Diversification Ratio: {metrics['diversification_ratio']:.2f}")
        report.append("")
        
        # Individual holdings
        report.append("INDIVIDUAL HOLDINGS:")
        report.append("-" * 30)
        report.append(f"{'Asset':<8} {'Weight':<8} {'Return':<8} {'Vol':<8} {'Reliability':<12}")
        report.append("-" * 50)
        
        for asset in assets:
            weight = weights[asset]
            ret = asset_data[asset]['expected_return']
            vol = asset_data[asset]['volatility']
            rel = asset_data[asset].get('reliability', 0.5)
            
            report.append(f"{asset:<8} {weight:<8.1%} {ret:<8.2%} {vol:<8.2%} {rel:<12.3f}")
        
        report.append("")
        
        # Risk analysis
        report.append("RISK ANALYSIS:")
        report.append("-" * 30)
        
        # Contribution to portfolio risk
        risk_contributions = (weights_array * volatilities) ** 2
        risk_contributions = risk_contributions / np.sum(risk_contributions)
        
        report.append("Risk Contribution by Asset:")
        for i, asset in enumerate(assets):
            contrib = risk_contributions[i]
            report.append(f"  {asset}: {contrib:.1%}")
        
        report.append("")
        
        # Top holdings
        sorted_holdings = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_holdings[:3]
        
        report.append("TOP 3 HOLDINGS:")
        for i, (asset, weight) in enumerate(top_3, 1):
            report.append(f"  {i}. {asset}: {weight:.1%}")
        
        return "\n".join(report)
