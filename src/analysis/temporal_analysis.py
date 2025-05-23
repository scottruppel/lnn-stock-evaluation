import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import signal, stats
from sklearn.cluster import KMeans
import warnings

class TemporalAnalyzer:
    """
    Advanced temporal analysis for time series data.
    Analyzes seasonality, trends, regime changes, and temporal patterns.
    """
    
    def __init__(self):
        self.analysis_results = {}
        
    def decompose_time_series(self, data: np.ndarray, period: int = 252) -> Dict:
        """
        Decompose time series into trend, seasonal, and residual components.
        
        Args:
            data: Time series data
            period: Period for seasonal decomposition (252 for daily stock data = 1 year)
        
        Returns:
            Dictionary with decomposition components
        """
        print(f"Decomposing time series with period {period}")
        
        data_series = pd.Series(data.flatten())
        
        # Simple trend using moving average
        trend = data_series.rolling(window=period, center=True).mean()
        
        # Detrend the data
        detrended = data_series - trend
        
        # Calculate seasonal component
        seasonal = self._calculate_seasonal_component(detrended, period)
        
        # Residual component
        residual = data_series - trend - seasonal
        
        # Calculate statistics
        trend_strength = 1 - (residual.var() / detrended.var()) if detrended.var() > 0 else 0
        seasonal_strength = 1 - (residual.var() / (detrended - seasonal).var()) if (detrended - seasonal).var() > 0 else 0
        
        results = {
            'original': data_series.values,
            'trend': trend.values,
            'seasonal': seasonal.values,
            'residual': residual.values,
            'trend_strength': trend_strength,
            'seasonal_strength': seasonal_strength,
            'period': period
        }
        
        self.analysis_results['decomposition'] = results
        return results
    
    def _calculate_seasonal_component(self, detrended_data: pd.Series, period: int) -> pd.Series:
        """Calculate seasonal component from detrended data."""
        seasonal = pd.Series(index=detrended_data.index, dtype=float)
        
        # Calculate average for each position in the period
        for i in range(period):
            positions = range(i, len(detrended_data), period)
            if positions:
                seasonal_value = detrended_data.iloc[positions].mean()
                # Fill all positions with this seasonal value
                for pos in positions:
                    if pos < len(seasonal):
                        seasonal.iloc[pos] = seasonal_value
        
        # Fill any remaining NaN values
        seasonal = seasonal.fillna(0)
        return seasonal
    
    def detect_seasonality(self, data: np.ndarray, max_period: int = 365) -> Dict:
        """
        Detect seasonal patterns in the data using autocorrelation and FFT.
        
        Args:
            data: Time series data
            max_period: Maximum period to check for seasonality
        
        Returns:
            Dictionary with seasonality analysis results
        """
        print(f"Detecting seasonality up to period {max_period}")
        
        data_clean = pd.Series(data.flatten()).dropna()
        
        # Autocorrelation analysis
        autocorr_results = self._autocorrelation_seasonality(data_clean, max_period)
        
        # Frequency domain analysis
        fft_results = self._fft_seasonality(data_clean, max_period)
        
        # Combine results
        results = {
            'autocorrelation': autocorr_results,
            'fft': fft_results,
            'is_seasonal': autocorr_results['is_seasonal'] or fft_results['is_seasonal'],
            'dominant_period': autocorr_results.get('period', fft_results.get('period'))
        }
        
        self.analysis_results['seasonality'] = results
        return results
    
    def _autocorrelation_seasonality(self, data: pd.Series, max_period: int) -> Dict:
        """Detect seasonality using autocorrelation."""
        autocorrelations = []
        periods = range(2, min(max_period, len(data) // 2))
        
        for period in periods:
            if len(data) > period:
                autocorr = data.autocorr(lag=period)
                if not np.isnan(autocorr):
                    autocorrelations.append(autocorr)
                else:
                    autocorrelations.append(0)
            else:
                autocorrelations.append(0)
        
        if autocorrelations:
            max_autocorr = max(autocorrelations)
            max_period = periods[np.argmax(autocorrelations)]
            
            # Consider seasonal if autocorrelation > 0.3
            is_seasonal = max_autocorr > 0.3
            
            return {
                'is_seasonal': is_seasonal,
                'max_autocorr': max_autocorr,
                'period': max_period if is_seasonal else None,
                'autocorrelations': autocorrelations,
                'periods': list(periods)
            }
        else:
            return {'is_seasonal': False, 'max_autocorr': 0, 'period': None}
    
    def _fft_seasonality(self, data: pd.Series, max_period: int) -> Dict:
        """Detect seasonality using FFT."""
        # Remove trend
        detrended = signal.detrend(data.values)
        
        # Apply FFT
        fft_vals = np.fft.fft(detrended)
        freqs = np.fft.fftfreq(len(detrended))
        
        # Get magnitude spectrum
        magnitude = np.abs(fft_vals)
        
        # Find dominant frequencies (excluding DC component)
        dominant_freq_idx = np.argmax(magnitude[1:len(magnitude)//2]) + 1
        dominant_freq = freqs[dominant_freq_idx]
        
        # Convert frequency to period
        if dominant_freq != 0:
            dominant_period = int(1 / abs(dominant_freq))
            # Check if period is reasonable
            is_seasonal = 2 <= dominant_period <= max_period
        else:
            dominant_period = None
            is_seasonal = False
        
        return {
            'is_seasonal': is_seasonal,
            'period': dominant_period if is_seasonal else None,
            'dominant_frequency': dominant_freq,
            'magnitude_spectrum': magnitude[:len(magnitude)//2],
            'frequencies': freqs[:len(freqs)//2]
        }
    
    def detect_regime_changes(self, data: np.ndarray, n_regimes: int = 3, 
                            window_size: int = 50) -> Dict:
        """
        Detect regime changes in the time series using statistical methods.
        
        Args:
            data: Time series data
            n_regimes: Number of regimes to detect
            window_size: Window size for rolling statistics
        
        Returns:
            Dictionary with regime change analysis
        """
        print(f"Detecting {n_regimes} regimes with window size {window_size}")
        
        data_series = pd.Series(data.flatten())
        
        # Calculate rolling statistics
        rolling_mean = data_series.rolling(window=window_size).mean()
        rolling_std = data_series.rolling(window=window_size).std()
        rolling_skew = data_series.rolling(window=window_size).skew()
        
        # Create feature matrix for clustering
        features = pd.DataFrame({
            'mean': rolling_mean,
            'std': rolling_std,
            'skew': rolling_skew
        }).dropna()
        
        # K-means clustering to identify regimes
        kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        regime_labels = kmeans.fit_predict(features)
        
        # Extend regime labels to full series length
        full_regime_labels = np.full(len(data_series), -1)
        start_idx = len(data_series) - len(regime_labels)
        full_regime_labels[start_idx:] = regime_labels
        
        # Calculate regime statistics
        regime_stats = self._calculate_regime_statistics(data_series, full_regime_labels, n_regimes)
        
        # Detect change points
        change_points = self._detect_change_points(regime_labels)
        
        results = {
            'regime_labels': full_regime_labels,
            'n_regimes': n_regimes,
            'regime_statistics': regime_stats,
            'change_points': change_points,
            'kmeans_centers': kmeans.cluster_centers_,
            'window_size': window_size
        }
        
        self.analysis_results['regime_changes'] = results
        return results
    
    def _calculate_regime_statistics(self, data: pd.Series, labels: np.ndarray, n_regimes: int) -> Dict:
        """Calculate statistics for each regime."""
        regime_stats = {}
        
        for regime in range(n_regimes):
            regime_data = data[labels == regime]
            if len(regime_data) > 0:
                regime_stats[regime] = {
                    'mean': regime_data.mean(),
                    'std': regime_data.std(),
                    'count': len(regime_data),
                    'min': regime_data.min(),
                    'max': regime_data.max(),
                    'duration_periods': len(regime_data)
                }
        
        return regime_stats
    
    def _detect_change_points(self, regime_labels: np.ndarray) -> List[int]:
        """Detect points where regime changes occur."""
        change_points = []
        
        for i in range(1, len(regime_labels)):
            if regime_labels[i] != regime_labels[i-1]:
                change_points.append(i)
        
        return change_points
    
    def analyze_autocorrelation(self, data: np.ndarray, max_lags: int = 50) -> Dict:
        """
        Analyze autocorrelation structure of the time series.
        
        Args:
            data: Time series data
            max_lags: Maximum number of lags to analyze
        
        Returns:
            Dictionary with autocorrelation analysis
        """
        print(f"Analyzing autocorrelation up to {max_lags} lags")
        
        data_series = pd.Series(data.flatten()).dropna()
        
        # Calculate autocorrelations
        autocorrelations = []
        for lag in range(1, max_lags + 1):
            if len(data_series) > lag:
                autocorr = data_series.autocorr(lag=lag)
                autocorrelations.append(autocorr if not np.isnan(autocorr) else 0)
            else:
                autocorrelations.append(0)
        
        # Calculate partial autocorrelations (simplified)
        partial_autocorr = self._calculate_partial_autocorr(data_series, max_lags)
        
        # Ljung-Box test for white noise
        lb_stat, lb_pvalue = self._ljung_box_test(data_series, lags=min(10, len(data_series)//4))
        
        # Find significant lags
        significant_lags = [i+1 for i, ac in enumerate(autocorrelations) if abs(ac) > 0.1]
        
        results = {
            'autocorrelations': autocorrelations,
            'partial_autocorrelations': partial_autocorr,
            'lags': list(range(1, max_lags + 1)),
            'significant_lags': significant_lags,
            'ljung_box_stat': lb_stat,
            'ljung_box_pvalue': lb_pvalue,
            'is_white_noise': lb_pvalue > 0.05 if lb_pvalue is not None else None
        }
        
        self.analysis_results['autocorrelation'] = results
        return results
    
    def _calculate_partial_autocorr(self, data: pd.Series, max_lags: int) -> List[float]:
        """Calculate partial autocorrelations (simplified approach)."""
        partial_autocorr = []
        
        for lag in range(1, max_lags + 1):
            if len(data) > lag * 2:
                # Simple approach: correlation between x_t and x_{t-lag} after removing linear dependence
                # on x_{t-1}, ..., x_{t-lag+1}
                try:
                    y = data.iloc[lag:].values
                    X = np.column_stack([data.iloc[i:-lag+i].values for i in range(lag)])
                    
                    # Linear regression to remove intermediate dependencies
                    if X.shape[0] > X.shape[1]:
                        coef = np.linalg.lstsq(X, y, rcond=None)[0]
                        residuals = y - X @ coef
                        
                        # Correlation between residuals and lagged variable
                        lagged_var = data.iloc[:-lag].values
                        if len(residuals) == len(lagged_var):
                            partial_corr = np.corrcoef(residuals, lagged_var)[0, 1]
                            partial_autocorr.append(partial_corr if not np.isnan(partial_corr) else 0)
                        else:
                            partial_autocorr.append(0)
                    else:
                        partial_autocorr.append(0)
                except:
                    partial_autocorr.append(0)
            else:
                partial_autocorr.append(0)
        
        return partial_autocorr
    
    def _ljung_box_test(self, data: pd.Series, lags: int) -> Tuple[Optional[float], Optional[float]]:
        """Simplified Ljung-Box test for white noise."""
        try:
            n = len(data)
            autocorrs = [data.autocorr(lag=i) for i in range(1, lags + 1)]
            
            # Remove NaN values
            autocorrs = [ac for ac in autocorrs if not np.isnan(ac)]
            
            if not autocorrs:
                return None, None
            
            # Calculate Ljung-Box statistic
            lb_stat = n * (n + 2) * sum((ac**2) / (n - i - 1) for i, ac in enumerate(autocorrs))
            
            # Chi-square test
            from scipy.stats import chi2
            lb_pvalue = 1 - chi2.cdf(lb_stat, df=len(autocorrs))
            
            return lb_stat, lb_pvalue
        except:
            return None, None
    
    def analyze_volatility_clustering(self, returns: np.ndarray, window: int = 20) -> Dict:
        """
        Analyze volatility clustering in return series.
        
        Args:
            returns: Return series data
            window: Window for volatility calculation
        
        Returns:
            Dictionary with volatility clustering analysis
        """
        print(f"Analyzing volatility clustering with window {window}")
        
        returns_series = pd.Series(returns.flatten()).dropna()
        
        # Calculate rolling volatility
        rolling_vol = returns_series.rolling(window=window).std()
        
        # Volatility autocorrelation
        vol_autocorr = []
        for lag in range(1, min(21, len(rolling_vol)//4)):
            if len(rolling_vol) > lag:
                autocorr = rolling_vol.autocorr(lag=lag)
                vol_autocorr.append(autocorr if not np.isnan(autocorr) else 0)
        
        # ARCH test (simplified)
        arch_stat = self._arch_test(returns_series)
        
        results = {
            'rolling_volatility': rolling_vol.values,
            'volatility_autocorr': vol_autocorr,
            'volatility_lags': list(range(1, len(vol_autocorr) + 1)),
            'arch_test_stat': arch_stat,
            'has_volatility_clustering': any(abs(ac) > 0.1 for ac in vol_autocorr),
            'window': window
        }
        
        self.analysis_results['volatility_clustering'] = results
        return results
    
    def _arch_test(self, returns: pd.Series, lags: int = 5) -> Optional[float]:
        """Simplified ARCH test for heteroscedasticity."""
        try:
            squared_returns = returns**2
            
            # Calculate autocorrelation of squared returns
            arch_autocorr = []
            for lag in range(1, lags + 1):
                if len(squared_returns) > lag:
                    autocorr = squared_returns.autocorr(lag=lag)
                    arch_autocorr.append(autocorr if not np.isnan(autocorr) else 0)
            
            # Simple test statistic
            arch_stat = sum(ac**2 for ac in arch_autocorr) * len(returns)
            return arch_stat
        except:
            return None
    
    def get_comprehensive_analysis(self, data: np.ndarray, returns: Optional[np.ndarray] = None) -> Dict:
        """
        Perform comprehensive temporal analysis.
        
        Args:
            data: Price or level data
            returns: Return data (optional, will be calculated if not provided)
        
        Returns:
            Dictionary with all temporal analysis results
        """
        print("Performing comprehensive temporal analysis...")
        
        results = {}
        
        # Time series decomposition
        try:
            decomp_results = self.decompose_time_series(data)
            results['decomposition'] = decomp_results
        except Exception as e:
            print(f"Decomposition failed: {e}")
            results['decomposition'] = None
        
        # Seasonality detection
        try:
            seasonality_results = self.detect_seasonality(data)
            results['seasonality'] = seasonality_results
        except Exception as e:
            print(f"Seasonality detection failed: {e}")
            results['seasonality'] = None
        
        # Regime change detection
        try:
            regime_results = self.detect_regime_changes(data)
            results['regime_changes'] = regime_results
        except Exception as e:
            print(f"Regime detection failed: {e}")
            results['regime_changes'] = None
        
        # Autocorrelation analysis
        try:
            autocorr_results = self.analyze_autocorrelation(data)
            results['autocorrelation'] = autocorr_results
        except Exception as e:
            print(f"Autocorrelation analysis failed: {e}")
            results['autocorrelation'] = None
        
        # Volatility clustering (if returns provided or can be calculated)
        try:
            if returns is None and len(data) > 1:
                data_series = pd.Series(data.flatten())
                returns = data_series.pct_change().dropna().values
            
            if returns is not None:
                vol_results = self.analyze_volatility_clustering(returns)
                results['volatility_clustering'] = vol_results
        except Exception as e:
            print(f"Volatility analysis failed: {e}")
            results['volatility_clustering'] = None
        
        print("Comprehensive temporal analysis completed")
        return results
