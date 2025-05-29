#!/usr/bin/env python3
"""
Comprehensive Data Validation System for LNN Pipeline
Integrates with your existing data_loader.py and run_analysis.py

Save this as: src/data/data_validator.py
"""

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
import json
from dataclasses import dataclass
from enum import Enum

class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    INFO = "INFO"
    WARNING = "WARNING" 
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class ValidationIssue:
    """Structure for validation issues"""
    ticker: str
    issue_type: str
    severity: ValidationSeverity
    message: str
    data_affected: Optional[int] = None  # Number of data points affected
    recommendation: Optional[str] = None

class DataQualityChecker:
    """
    Comprehensive data quality checker that integrates with your existing pipeline.
    Checks for data integrity, completeness, and trading viability.
    """
    
    def __init__(self, 
                 min_data_points: int = 100,
                 max_daily_return: float = 0.25,  # 25% max daily move
                 min_price: float = 1.0,          # Minimum stock price
                 max_missing_days: int = 10,      # Max consecutive missing days
                 volume_threshold: int = 10000):   # Minimum daily volume
        
        self.min_data_points = min_data_points
        self.max_daily_return = max_daily_return
        self.min_price = min_price
        self.max_missing_days = max_missing_days
        self.volume_threshold = volume_threshold
        
        self.validation_issues = []
        self.data_stats = {}
        
    def validate_complete_dataset(self, 
                                price_data: Dict[str, np.ndarray],
                                raw_data: Optional[Dict] = None,
                                config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main validation function - validates your entire dataset.
        
        Args:
            price_data: Your price_data dict from data_loader.get_closing_prices()
            raw_data: Your raw_data from data_loader.download_data() (optional)
            config: Your config dict for additional validation rules
            
        Returns:
            validation_report: Comprehensive validation report
        """
        print("🔍 Running comprehensive data validation...")
        
        # Clear previous validation results
        self.validation_issues = []
        self.data_stats = {}
        
        # 1. Basic data structure validation
        self._validate_data_structure(price_data)
        
        # 2. Individual ticker validation
        for ticker, prices in price_data.items():
            if prices is not None and len(prices) > 0:
                self._validate_single_ticker(ticker, prices, raw_data)
        
        # 3. Cross-ticker validation
        self._validate_cross_ticker_consistency(price_data)
        
        # 4. Configuration-specific validation
        if config:
            self._validate_config_requirements(price_data, config)
        
        # 5. Generate comprehensive report
        report = self._generate_validation_report()
        
        # 6. Print summary
        self._print_validation_summary(report)
        
        return report
    
    def _validate_data_structure(self, price_data: Dict[str, np.ndarray]):
        """Validate basic data structure and format"""
        if not price_data:
            self.validation_issues.append(ValidationIssue(
                ticker="ALL",
                issue_type="structure",
                severity=ValidationSeverity.CRITICAL,
                message="No price data provided",
                recommendation="Check data loading process and internet connection"
            ))
            return
        
        # Check for empty tickers
        empty_tickers = [ticker for ticker, prices in price_data.items() 
                        if prices is None or len(prices) == 0]
        
        if empty_tickers:
            self.validation_issues.append(ValidationIssue(
                ticker=", ".join(empty_tickers),
                issue_type="structure",
                severity=ValidationSeverity.ERROR,
                message=f"Empty data for {len(empty_tickers)} ticker(s)",
                recommendation="Remove these tickers or check data source"
            ))
    
    def _validate_single_ticker(self, 
                               ticker: str, 
                               prices: np.ndarray,
                               raw_data: Optional[Dict] = None):
        """Comprehensive validation for a single ticker"""
        
        # Ensure prices is a 1D array
        if prices.ndim > 1:
            prices = prices.flatten()
        
        # Store basic stats
        self.data_stats[ticker] = self._calculate_ticker_stats(prices)
        
        # 1. Data sufficiency check
        if len(prices) < self.min_data_points:
            self.validation_issues.append(ValidationIssue(
                ticker=ticker,
                issue_type="sufficiency",
                severity=ValidationSeverity.ERROR,
                message=f"Insufficient data: {len(prices)} points (need {self.min_data_points})",
                data_affected=len(prices),
                recommendation="Extend date range or remove ticker"
            ))
        
        # 2. Price validity checks
        self._validate_price_data(ticker, prices)
        
        # 3. Return analysis
        self._validate_returns(ticker, prices)
        
        # 4. Missing data detection
        self._detect_missing_data(ticker, prices)
        
        # 5. Volume analysis (if available)
        if raw_data and ticker in raw_data:
            self._validate_volume_data(ticker, raw_data[ticker])
        
        # 6. Structural breaks detection
        self._detect_structural_breaks(ticker, prices)
    
    def _validate_price_data(self, ticker: str, prices: np.ndarray):
        """Validate price data quality"""
        
        # Check for NaN values
        nan_count = np.sum(np.isnan(prices))
        if nan_count > 0:
            self.validation_issues.append(ValidationIssue(
                ticker=ticker,
                issue_type="quality",
                severity=ValidationSeverity.ERROR,
                message=f"Contains {nan_count} NaN values",
                data_affected=nan_count,
                recommendation="Interpolate or remove NaN values"
            ))
        
        # Check for zero or negative prices
        invalid_prices = np.sum(prices <= 0)
        if invalid_prices > 0:
            self.validation_issues.append(ValidationIssue(
                ticker=ticker,
                issue_type="quality",
                severity=ValidationSeverity.ERROR,
                message=f"Contains {invalid_prices} zero/negative prices",
                data_affected=invalid_prices,
                recommendation="Check data source or remove invalid prices"
            ))
        
        # Check for extremely low prices (potential penny stocks)
        low_prices = np.sum(prices < self.min_price)
        if low_prices > len(prices) * 0.1:  # More than 10% below threshold
            self.validation_issues.append(ValidationIssue(
                ticker=ticker,
                issue_type="quality",
                severity=ValidationSeverity.WARNING,
                message=f"Many low prices (<${self.min_price}): {low_prices} occurrences",
                data_affected=low_prices,
                recommendation="Consider if this is a penny stock unsuitable for modeling"
            ))
        
        # Check for constant prices (no movement)
        if len(set(prices)) == 1:
            self.validation_issues.append(ValidationIssue(
                ticker=ticker,
                issue_type="quality",
                severity=ValidationSeverity.ERROR,
                message="All prices identical - no price movement",
                recommendation="Check data source or remove ticker"
            ))
    
    def _validate_returns(self, ticker: str, prices: np.ndarray):
        """Validate return characteristics"""
        
        if len(prices) < 2:
            return
        
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        valid_returns = returns[np.isfinite(returns)]
        
        if len(valid_returns) == 0:
            self.validation_issues.append(ValidationIssue(
                ticker=ticker,
                issue_type="returns",
                severity=ValidationSeverity.ERROR,
                message="No valid returns calculated",
                recommendation="Check price data quality"
            ))
            return
        
        # Check for extreme returns
        extreme_returns = np.abs(valid_returns) > self.max_daily_return
        extreme_count = np.sum(extreme_returns)
        
        if extreme_count > 0:
            severity = ValidationSeverity.WARNING if extreme_count < 5 else ValidationSeverity.ERROR
            self.validation_issues.append(ValidationIssue(
                ticker=ticker,
                issue_type="returns",
                severity=severity,
                message=f"{extreme_count} extreme daily returns (>{self.max_daily_return:.0%})",
                data_affected=extreme_count,
                recommendation="Investigate for stock splits, earnings, or data errors"
            ))
        
        # Check return distribution
        return_std = np.std(valid_returns)
        if return_std > 0.1:  # Daily volatility > 10%
            self.validation_issues.append(ValidationIssue(
                ticker=ticker,
                issue_type="returns",
                severity=ValidationSeverity.WARNING,
                message=f"Very high volatility: {return_std*100:.1f}% daily",
                recommendation="Consider if this asset is suitable for modeling"
            ))
        elif return_std < 0.001:  # Daily volatility < 0.1%
            self.validation_issues.append(ValidationIssue(
                ticker=ticker,
                issue_type="returns",
                severity=ValidationSeverity.WARNING,
                message=f"Extremely low volatility: {return_std*100:.3f}% daily",
                recommendation="Check if this is an actively traded asset"
            ))
    
    def _detect_missing_data(self, ticker: str, prices: np.ndarray):
        """Detect missing data patterns"""
        
        # This is a simplified check - in real implementation, you'd compare against trading calendar
        # For now, we check for repeated identical prices (potential missing data)
        
        consecutive_identical = 0
        max_consecutive = 0
        
        for i in range(1, len(prices)):
            if prices[i] == prices[i-1]:
                consecutive_identical += 1
                max_consecutive = max(max_consecutive, consecutive_identical)
            else:
                consecutive_identical = 0
        
        if max_consecutive > self.max_missing_days:
            self.validation_issues.append(ValidationIssue(
                ticker=ticker,
                issue_type="missing_data",
                severity=ValidationSeverity.WARNING,
                message=f"Potential missing data: {max_consecutive} consecutive identical prices",
                data_affected=max_consecutive,
                recommendation="Check for holidays, trading halts, or data gaps"
            ))
    
    def _validate_volume_data(self, ticker: str, ticker_data):
        """Validate volume data if available"""
        
        try:
            if hasattr(ticker_data, 'Volume'):
                volume = ticker_data.Volume.values
                
                # Check for zero volume days
                zero_volume_days = np.sum(volume == 0)
                if zero_volume_days > len(volume) * 0.05:  # More than 5%
                    self.validation_issues.append(ValidationIssue(
                        ticker=ticker,
                        issue_type="volume",
                        severity=ValidationSeverity.WARNING,
                        message=f"{zero_volume_days} zero-volume days ({zero_volume_days/len(volume):.1%})",
                        data_affected=zero_volume_days,
                        recommendation="Check for trading halts or illiquid periods"
                    ))
                
                # Check average volume
                avg_volume = np.mean(volume[volume > 0])
                if avg_volume < self.volume_threshold:
                    self.validation_issues.append(ValidationIssue(
                        ticker=ticker,
                        issue_type="volume",
                        severity=ValidationSeverity.WARNING,
                        message=f"Low average volume: {avg_volume:,.0f}",
                        recommendation="Consider liquidity constraints for trading"
                    ))
        
        except Exception as e:
            # Volume data not available or accessible
            pass
    
    def _detect_structural_breaks(self, ticker: str, prices: np.ndarray):
        """Detect potential structural breaks in price series"""
        
        if len(prices) < 60:  # Need sufficient data
            return
        
        # Simple structural break detection using rolling volatility
        window = 30
        returns = np.diff(prices) / prices[:-1]
        returns = returns[np.isfinite(returns)]
        
        if len(returns) < window * 2:
            return
        
        rolling_vol = []
        for i in range(window, len(returns) - window):
            vol = np.std(returns[i-window:i+window])
            rolling_vol.append(vol)
        
        rolling_vol = np.array(rolling_vol)
        
        # Detect sudden volatility changes
        vol_changes = np.abs(np.diff(rolling_vol)) / rolling_vol[:-1]
        extreme_changes = np.sum(vol_changes > 2.0)  # 200% volatility change
        
        if extreme_changes > 0:
            self.validation_issues.append(ValidationIssue(
                ticker=ticker,
                issue_type="structural_break",
                severity=ValidationSeverity.INFO,
                message=f"Detected {extreme_changes} potential structural breaks",
                recommendation="Consider regime-aware modeling approaches"
            ))
    
    def _validate_cross_ticker_consistency(self, price_data: Dict[str, np.ndarray]):
        """Validate consistency across tickers"""
        
        valid_tickers = {k: v for k, v in price_data.items() 
                        if v is not None and len(v) > 0}
        
        if len(valid_tickers) < 2:
            return
        
        # Check data length consistency
        lengths = {ticker: len(prices) for ticker, prices in valid_tickers.items()}
        min_length = min(lengths.values())
        max_length = max(lengths.values())
        
        if max_length - min_length > 10:  # More than 10 days difference
            inconsistent_tickers = [ticker for ticker, length in lengths.items() 
                                  if abs(length - min_length) > 5]
            
            self.validation_issues.append(ValidationIssue(
                ticker=", ".join(inconsistent_tickers),
                issue_type="consistency",
                severity=ValidationSeverity.WARNING,
                message=f"Inconsistent data lengths: {min_length} to {max_length} days",
                recommendation="Align data to common time period"
            ))
        
        # Check correlation patterns (basic sanity check)
        if len(valid_tickers) >= 2:
            self._check_correlation_sanity(valid_tickers)
    
    def _check_correlation_sanity(self, valid_tickers: Dict[str, np.ndarray]):
        """Basic correlation sanity checks"""
        
        tickers = list(valid_tickers.keys())
        
        # Check for perfectly correlated assets (potential duplicates)
        for i, ticker1 in enumerate(tickers):
            for ticker2 in tickers[i+1:]:
                prices1 = valid_tickers[ticker1]
                prices2 = valid_tickers[ticker2]
                
                # Align lengths
                min_len = min(len(prices1), len(prices2))
                p1_aligned = prices1[-min_len:]
                p2_aligned = prices2[-min_len:]
                
                if min_len > 30:  # Need sufficient data
                    correlation = np.corrcoef(p1_aligned, p2_aligned)[0, 1]
                    
                    if correlation > 0.99:
                        self.validation_issues.append(ValidationIssue(
                            ticker=f"{ticker1}, {ticker2}",
                            issue_type="correlation",
                            severity=ValidationSeverity.WARNING,
                            message=f"Nearly perfect correlation: {correlation:.3f}",
                            recommendation="Check for duplicate tickers or related instruments"
                        ))
    
    def _validate_config_requirements(self, price_data: Dict[str, np.ndarray], config: Dict):
        """Validate data meets configuration requirements"""
        
        # Check target ticker exists and has sufficient data
        target_ticker = config.get('data', {}).get('target_ticker')
        if target_ticker:
            if target_ticker not in price_data:
                self.validation_issues.append(ValidationIssue(
                    ticker=target_ticker,
                    issue_type="config",
                    severity=ValidationSeverity.CRITICAL,
                    message="Target ticker not found in data",
                    recommendation="Check ticker symbol or data loading"
                ))
            elif len(price_data[target_ticker]) < config.get('model', {}).get('sequence_length', 30):
                seq_len = config.get('model', {}).get('sequence_length', 30)
                self.validation_issues.append(ValidationIssue(
                    ticker=target_ticker,
                    issue_type="config",
                    severity=ValidationSeverity.ERROR,
                    message=f"Target ticker has insufficient data for sequence length ({seq_len})",
                    recommendation="Reduce sequence length or extend data period"
                ))
    
    def _calculate_ticker_stats(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive statistics for a ticker"""
        
        if len(prices) < 2:
            return {}
        
        returns = np.diff(prices) / prices[:-1]
        valid_returns = returns[np.isfinite(returns)]
        
        if len(valid_returns) == 0:
            return {}
        
        stats = {
            'total_observations': len(prices),
            'valid_returns': len(valid_returns),
            'price_min': float(np.min(prices)),
            'price_max': float(np.max(prices)),
            'price_mean': float(np.mean(prices)),
            'price_std': float(np.std(prices)),
            'total_return': float((prices[-1] - prices[0]) / prices[0]),
            'return_mean': float(np.mean(valid_returns)),
            'return_std': float(np.std(valid_returns)),
            'return_min': float(np.min(valid_returns)),
            'return_max': float(np.max(valid_returns)),
            'volatility_annualized': float(np.std(valid_returns) * np.sqrt(252)),
            'sharpe_estimate': float(np.mean(valid_returns) / np.std(valid_returns) * np.sqrt(252)) if np.std(valid_returns) > 0 else 0
        }
        
        return stats
    
    def _generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        
        # Categorize issues by severity
        issues_by_severity = {
            ValidationSeverity.CRITICAL: [],
            ValidationSeverity.ERROR: [],
            ValidationSeverity.WARNING: [],
            ValidationSeverity.INFO: []
        }
        
        for issue in self.validation_issues:
            issues_by_severity[issue.severity].append(issue)
        
        # Calculate summary statistics
        total_issues = len(self.validation_issues)
        critical_count = len(issues_by_severity[ValidationSeverity.CRITICAL])
        error_count = len(issues_by_severity[ValidationSeverity.ERROR])
        warning_count = len(issues_by_severity[ValidationSeverity.WARNING])
        
        # Determine overall data quality
        if critical_count > 0:
            quality_score = "CRITICAL"
        elif error_count > 3:
            quality_score = "POOR"
        elif error_count > 0 or warning_count > 5:
            quality_score = "FAIR"
        elif warning_count > 0:
            quality_score = "GOOD"
        else:
            quality_score = "EXCELLENT"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_quality': quality_score,
            'total_issues': total_issues,
            'issues_by_severity': {
                'critical': critical_count,
                'error': error_count, 
                'warning': warning_count,
                'info': len(issues_by_severity[ValidationSeverity.INFO])
            },
            'detailed_issues': [
                {
                    'ticker': issue.ticker,
                    'type': issue.issue_type,
                    'severity': issue.severity.value,
                    'message': issue.message,
                    'data_affected': issue.data_affected,
                    'recommendation': issue.recommendation
                }
                for issue in self.validation_issues
            ],
            'ticker_statistics': self.data_stats,
            'validation_passed': critical_count == 0 and error_count == 0
        }
        
        return report
    
    def _print_validation_summary(self, report: Dict[str, Any]):
        """Print validation summary to console"""
        
        print("\n" + "="*70)
        print("DATA VALIDATION REPORT")
        print("="*70)
        
        print(f"Overall Data Quality: {report['overall_quality']}")
        print(f"Total Issues Found: {report['total_issues']}")
        
        if report['total_issues'] > 0:
            print(f"  • Critical: {report['issues_by_severity']['critical']}")
            print(f"  • Errors: {report['issues_by_severity']['error']}")
            print(f"  • Warnings: {report['issues_by_severity']['warning']}")
            print(f"  • Info: {report['issues_by_severity']['info']}")
        
        print(f"Validation Passed: {'✓ YES' if report['validation_passed'] else '✗ NO'}")
        
        # Print detailed issues if any
        if report['total_issues'] > 0:
            print("\nDETAILED ISSUES:")
            print("-" * 50)
            
            for issue in report['detailed_issues']:
                severity_symbol = {
                    'CRITICAL': '🚫',
                    'ERROR': '❌', 
                    'WARNING': '⚠️',
                    'INFO': 'ℹ️'
                }
                
                symbol = severity_symbol.get(issue['severity'], '•')
                print(f"{symbol} {issue['ticker']}: {issue['message']}")
                if issue['recommendation']:
                    print(f"   → {issue['recommendation']}")
        
        # Print data statistics summary
        if self.data_stats:
            print(f"\nDATA STATISTICS SUMMARY:")
            print("-" * 50)
            for ticker, stats in self.data_stats.items():
                if stats:
                    print(f"{ticker}: {stats['total_observations']} obs, "
                          f"{stats['total_return']:.1%} return, "
                          f"{stats['volatility_annualized']:.1%} vol, "
                          f"Sharpe≈{stats['sharpe_estimate']:.2f}")
        
        print("="*70)
    
    def save_validation_report(self, report: Dict[str, Any], filepath: str):
        """Save validation report to file"""
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✓ Validation report saved to: {filepath}")
    
    def get_clean_data(self, 
                      price_data: Dict[str, np.ndarray],
                      remove_problematic: bool = True) -> Dict[str, np.ndarray]:
        """
        Return cleaned data based on validation results.
        
        Args:
            price_data: Original price data
            remove_problematic: Whether to remove tickers with critical/error issues
            
        Returns:
            clean_data: Cleaned price data dictionary
        """
        
        if not remove_problematic:
            return price_data
        
        # Get tickers with critical or error issues
        problematic_tickers = set()
        
        for issue in self.validation_issues:
            if issue.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]:
                # Handle multiple tickers in one issue
                tickers_in_issue = [t.strip() for t in issue.ticker.split(',')]
                problematic_tickers.update(tickers_in_issue)
        
        # Remove problematic tickers
        clean_data = {
            ticker: prices for ticker, prices in price_data.items()
            if ticker not in problematic_tickers and prices is not None and len(prices) > 0
        }
        
        removed_count = len(price_data) - len(clean_data)
        if removed_count > 0:
            print(f"🧹 Removed {removed_count} problematic ticker(s): {list(problematic_tickers)}")
        
        return clean_data


class ValidationIntegrator:
    """
    Integration class to add validation to your existing pipeline.
    This modifies your existing classes to include validation.
    """
    
    @staticmethod
    def integrate_with_data_loader(data_loader_class):
        """Add validation methods to your existing StockDataLoader"""
        
        def validate_downloaded_data(self, auto_clean: bool = True):
            """Add this method to your StockDataLoader class"""
            
            # Get the data
            price_data = self.get_closing_prices()
            raw_data = getattr(self, 'data', None)
            
            # Run validation
            validator = DataQualityChecker()
            validation_report = validator.validate_complete_dataset(
                price_data=price_data,
                raw_data=raw_data
            )
            
            # Save validation report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"results/validation/data_validation_{timestamp}.json"
            validator.save_validation_report(validation_report, report_path)
            
            # Return cleaned data if auto_clean is True
            if auto_clean:
                clean_data = validator.get_clean_data(price_data)
                return clean_data, validation_report
            else:
                return price_data, validation_report
        
        # Add the method to the class
        data_loader_class.validate_downloaded_data = validate_downloaded_data
        return data_loader_class
    
    @staticmethod
    def integrate_with_run_analysis(analyzer_class):
        """Add validation to your ComprehensiveAnalyzer class"""
        
        def validate_and_analyze_data(self):
            """Enhanced version of analyze_raw_data with validation"""
            
            print("=" * 70)
            print("PHASE 1: DATA VALIDATION & ANALYSIS")
            print("=" * 70)
            
            # Load data (your existing code)
            self.data_loader = StockDataLoader(
                tickers=self.config['data']['tickers'],
                start_date=self.config['data']['start_date'],
                end_date=self.config['data']['end_date']
            )
            
            self.raw_data = self.data_loader.download_data()
            price_data = self.data_loader.get_closing_prices()
            
            # NEW: Add comprehensive data validation
            print("🔍 Running data validation...")
            validator = DataQualityChecker()
            validation_report = validator.validate_complete_dataset(
                price_data=price_data,
                raw_data=self.raw_data,
                config=self.config
            )
            
            # Store validation results
            self.analysis_results['data_validation'] = validation_report
            
            # Save validation report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"results/validation/validation_{self.experiment_name}_{timestamp}.json"
            validator.save_validation_report(validation_report, report_path)
            
            # Decide whether to continue based on validation
            if not validation_report['validation_passed']:
                print("⚠️  Data validation failed. You can:")
                print("   1. Continue with problematic data (risky)")
                print("   2. Use auto-cleaned data (recommended)")
                print("   3. Fix data issues manually and re-run")
                
                # For automation, use cleaned data
                clean_price_data = validator.get_clean_data(price_data, remove_problematic=True)
                
                if len(clean_price_data) < len(self.config['data']['tickers']) / 2:
                    print("❌ Too many tickers removed. Please check data sources.")
                    raise ValueError("Data quality too poor for analysis")
                
                price_data = clean_price_data
                print(f"✓ Continuing with {len(price_data)} validated tickers")
            else:
                print("✅ All data validation checks passed!")
            
            # Continue with your existing analysis using validated data
            # ... rest of your analyze_raw_data method ...
            
            return price_data  # Return the validated data
        
        # Replace the method
        analyzer_class.validate_and_analyze_data = validate_and_analyze_data
        return analyzer_class


# Utility functions for easy integration
def quick_validate_data(tickers: List[str], 
                       start_date: str = "2020-01-01", 
                       end_date: str = "2024-12-31") -> Dict[str, Any]:
    """
    Quick validation function for testing.
    
    Usage on your Jetson Nano:
    from src.data.data_validator import quick_validate_data
    report = quick_validate_data(['AAPL', 'MSFT', 'GOOGL'])
    """
    
    # Import your existing data loader
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from data.data_loader import StockDataLoader
    
    # Load data
    loader = StockDataLoader(tickers, start_date, end_date)
    raw_data = loader.download_data()
    price_data = loader.get_closing_prices()
    
    # Validate
    validator = DataQualityChecker()
    report = validator.validate_complete_dataset(price_data, raw_data)
    
    return report


def validate_existing_data(price_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Validate data you already have loaded.
    
    Usage:
    from src.data.data_validator import validate_existing_data
    report = validate_existing_data(your_price_data)
    """
    
    validator = DataQualityChecker()
    report = validator.validate_complete_dataset(price_data)
    return report


# Example usage and testing functions
if __name__ == "__main__":
    """
    Test the validation system
    Run this on your Jetson Nano: python src/data/data_validator.py
    """
    
    print("🧪 Testing Data Validation System...")
    print("=" * 50)
    
    # Test with your typical tickers
    test_tickers = ['^GSPC', 'AGG', 'QQQ', 'AAPL']
    
    try:
        # Quick validation test
        print("Running quick validation test...")
        report = quick_validate_data(test_tickers)
        
        print(f"\n✅ Validation test completed!")
        print(f"   Overall quality: {report['overall_quality']}")
        print(f"   Total issues: {report['total_issues']}")
        print(f"   Validation passed: {report['validation_passed']}")
        
        # Save test report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_report_path = f"results/validation/test_validation_{timestamp}.json"
        
        validator = DataQualityChecker()
        validator.save_validation_report(report, test_report_path)
        
        print(f"   Test report saved to: {test_report_path}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Make sure you have internet connection and required packages installed")


# Integration examples for your specific pipeline
class ExampleIntegrations:
    """
    Examples showing exactly how to integrate validation into your existing code
    """
    
    @staticmethod
    def integrate_into_run_analysis():
        """
        Example: How to modify your run_analysis.py to include validation
        """
        
        example_code = '''
# Add this to the top of your run_analysis.py imports:
from data.data_validator import DataQualityChecker, ValidationSeverity

# Replace your analyze_raw_data method with this enhanced version:
def analyze_raw_data(self):
    """Enhanced version with data validation"""
    print("=" * 70)
    print("PHASE 1: DATA VALIDATION & ANALYSIS") 
    print("=" * 70)

    # 1. Load data (your existing code)
    print("Loading market data...")
    self.data_loader = StockDataLoader(
        tickers=self.config['data']['tickers'],
        start_date=self.config['data']['start_date'],
        end_date=self.config['data']['end_date']
    )

    self.raw_data = self.data_loader.download_data()
    price_data = self.data_loader.get_closing_prices()

    # 2. NEW: Comprehensive data validation
    print("🔍 Running comprehensive data validation...")
    validator = DataQualityChecker(
        min_data_points=100,        # Minimum data points required
        max_daily_return=0.25,      # Flag returns > 25%
        min_price=1.0,              # Minimum stock price
        max_missing_days=10,        # Max consecutive missing days
        volume_threshold=10000      # Minimum daily volume
    )
    
    validation_report = validator.validate_complete_dataset(
        price_data=price_data,
        raw_data=self.raw_data,
        config=self.config
    )
    
    # 3. Store validation results
    self.analysis_results['data_validation'] = validation_report
    
    # 4. Save detailed validation report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    validation_report_path = f"results/validation/validation_{self.experiment_name}_{timestamp}.json"
    validator.save_validation_report(validation_report, validation_report_path)
    
    # 5. Handle validation results
    if not validation_report['validation_passed']:
        print("⚠️  Data validation issues detected!")
        
        # Check severity of issues
        critical_issues = validation_report['issues_by_severity']['critical']
        error_issues = validation_report['issues_by_severity']['error']
        
        if critical_issues > 0:
            print(f"❌ {critical_issues} critical issues found - analysis may fail")
            print("   Consider fixing these issues before continuing:")
            for issue in validation_report['detailed_issues']:
                if issue['severity'] == 'CRITICAL':
                    print(f"   • {issue['ticker']}: {issue['message']}")
        
        if error_issues > 0:
            print(f"⚠️  {error_issues} error issues found - using cleaned data")
            # Get cleaned data (removes problematic tickers)
            clean_price_data = validator.get_clean_data(price_data, remove_problematic=True)
            
            if len(clean_price_data) < 2:
                raise ValueError("Too few valid tickers remaining after cleaning")
            
            price_data = clean_price_data
            print(f"✓ Continuing with {len(price_data)} validated tickers")
    else:
        print("✅ All data validation checks passed!")

    # 6. Continue with your existing analysis using validated data
    # ... rest of your existing analyze_raw_data code ...
    
    return price_data  # Return validated data for use in other phases
        '''
        
        return example_code
    
    @staticmethod
    def integrate_into_data_loader():
        """
        Example: How to add validation to your StockDataLoader class
        """
        
        example_code = '''
# Add this method to your StockDataLoader class in src/data/data_loader.py:

def download_and_validate_data(self, auto_clean=True, save_report=True):
    """
    Enhanced version of download_data that includes validation
    
    Args:
        auto_clean: Automatically remove problematic tickers
        save_report: Save validation report to file
        
    Returns:
        tuple: (price_data, validation_report)
    """
    
    # Download data using existing method
    raw_data = self.download_data()
    price_data = self.get_closing_prices()
    
    # Import validator
    from data.data_validator import DataQualityChecker
    
    # Run validation
    validator = DataQualityChecker()
    validation_report = validator.validate_complete_dataset(
        price_data=price_data,
        raw_data=raw_data
    )
    
    # Save report if requested
    if save_report:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"results/validation/data_validation_{timestamp}.json"
        validator.save_validation_report(validation_report, report_path)
        print(f"✓ Validation report saved: {report_path}")
    
    # Clean data if requested
    if auto_clean and not validation_report['validation_passed']:
        cleaned_data = validator.get_clean_data(price_data)
        print(f"🧹 Cleaned data: {len(price_data)} → {len(cleaned_data)} tickers")
        return cleaned_data, validation_report
    
    return price_data, validation_report

# Usage in your scripts:
# data_loader = StockDataLoader(tickers=['AAPL', 'MSFT'], start_date='2020-01-01', end_date='2024-12-31')
# price_data, validation_report = data_loader.download_and_validate_data()
        '''
        
        return example_code
