import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings

def gpu_accelerated_moving_average(self, data, window):
    if torch.cuda.is_available():
        data_gpu = torch.tensor(data.values, device='cuda', dtype=torch.float32)
        kernel = torch.ones(window, device='cuda') / window
        padded = torch.nn.functional.pad(data_gpu, (window-1,0))
        result = torch.nn.functional.convld(
            padded.unsqueeze(0),unsqueeze(0)
            kernel.unsqueeze(0),unsqueeze(0)
        ).squeeze()
        return result.cpu().numpy()
    else:
        return data.rolling(window).mean()

class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for financial time series data.
    Creates sophisticated features for machine learning models.
    """
    
    def __init__(self):
        self.feature_names = []
        self.scalers = {}
        
    def create_comprehensive_features(self, 
                                    ohlcv_data: Dict[str, np.ndarray],
                                    include_advanced: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        Create comprehensive feature set from OHLCV data.
        
        Args:
            ohlcv_data: Dictionary with 'open', 'high', 'low', 'close', 'volume' arrays
            include_advanced: Whether to include computationally expensive features
        
        Returns:
            Tuple of (features_array, feature_names)
        """
        features = []
        feature_names = []
        
        # Extract basic data
        if 'close' not in ohlcv_data:
            raise ValueError("Close prices are required")
        
        close = pd.Series(ohlcv_data['close'].flatten())
        high = pd.Series(ohlcv_data.get('high', ohlcv_data['close']).flatten())
        low = pd.Series(ohlcv_data.get('low', ohlcv_data['close']).flatten())
        open_price = pd.Series(ohlcv_data.get('open', ohlcv_data['close']).flatten())
        volume = pd.Series(ohlcv_data.get('volume', np.ones_like(close)).flatten())
        
        # 1. Price-based features
        price_features, price_names = self._create_price_features(close, high, low, open_price)
        features.extend(price_features)
        feature_names.extend(price_names)
        
        # 2. Technical indicators
        tech_features, tech_names = self._create_technical_indicators(close, high, low, volume)
        features.extend(tech_features)
        feature_names.extend(tech_names)
        
        # 3. Statistical features
        stat_features, stat_names = self._create_statistical_features(close)
        features.extend(stat_features)
        feature_names.extend(stat_names)
        
        # 4. Volatility features
        vol_features, vol_names = self._create_volatility_features(close, high, low)
        features.extend(vol_features)
        feature_names.extend(vol_names)
        
        # 5. Volume features
        if 'volume' in ohlcv_data:
            volume_features, volume_names = self._create_volume_features(close, volume)
            features.extend(volume_features)
            feature_names.extend(volume_names)
        
        # 6. Advanced features (optional)
        if include_advanced:
            adv_features, adv_names = self._create_advanced_features(close, high, low)
            features.extend(adv_features)
            feature_names.extend(adv_names)
        
        # Combine all features
        combined_features = np.column_stack(features)
        
        # Handle NaN values
        feature_df = pd.DataFrame(combined_features, columns=feature_names)
        feature_df = feature_df.fillna(method='ffill').fillna(method='bfill')
        
        self.feature_names = feature_names
        return feature_df.values, feature_names
    
    def _create_price_features(self, close: pd.Series, high: pd.Series, 
                             low: pd.Series, open_price: pd.Series) -> Tuple[List[np.ndarray], List[str]]:
        """Create basic price-based features."""
        features = []
        names = []
        
        # Returns
        returns = close.pct_change()
        features.append(returns.values.reshape(-1, 1))
        names.append('returns')
        
        # Log returns
        log_returns = np.log(close / close.shift(1))
        features.append(log_returns.values.reshape(-1, 1))
        names.append('log_returns')
        
        # Price ratios
        hl_ratio = (high - low) / close
        features.append(hl_ratio.values.reshape(-1, 1))
        names.append('hl_ratio')
        
        oc_ratio = (close - open_price) / open_price
        features.append(oc_ratio.values.reshape(-1, 1))
        names.append('oc_ratio')
        
        # Gap features
        gap = (open_price - close.shift(1)) / close.shift(1)
        features.append(gap.values.reshape(-1, 1))
        names.append('gap')
        
        # Typical price
        typical_price = (high + low + close) / 3
        features.append(typical_price.values.reshape(-1, 1))
        names.append('typical_price')
        
        return features, names
    
    def _create_technical_indicators(self, close: pd.Series, high: pd.Series, 
                                   low: pd.Series, volume: pd.Series) -> Tuple[List[np.ndarray], List[str]]:
        """Create technical indicator features."""
        features = []
        names = []
        
        # Moving averages and ratios
        for window in [5, 10, 20, 50]:
            sma = close.rolling(window=window).mean()
            ema = close.ewm(span=window).mean()
            
            features.extend([
                sma.values.reshape(-1, 1),
                ema.values.reshape(-1, 1),
                (close / sma).values.reshape(-1, 1),
                (close / ema).values.reshape(-1, 1)
            ])
            
            names.extend([
                f'sma_{window}',
                f'ema_{window}',
                f'close_sma_{window}_ratio',
                f'close_ema_{window}_ratio'
            ])
        
        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9).mean()
        macd_histogram = macd - macd_signal
        
        features.extend([
            macd.values.reshape(-1, 1),
            macd_signal.values.reshape(-1, 1),
            macd_histogram.values.reshape(-1, 1)
        ])
        names.extend(['macd', 'macd_signal', 'macd_histogram'])
        
        # RSI
        rsi = self._calculate_rsi(close)
        features.append(rsi.values.reshape(-1, 1))
        names.append('rsi')
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        sma20 = close.rolling(window=bb_period).mean()
        std20 = close.rolling(window=bb_period).std()
        
        bb_upper = sma20 + (std20 * bb_std)
        bb_lower = sma20 - (std20 * bb_std)
        bb_position = (close - bb_lower) / (bb_upper - bb_lower)
        bb_width = (bb_upper - bb_lower) / sma20
        
        features.extend([
            bb_upper.values.reshape(-1, 1),
            bb_lower.values.reshape(-1, 1),
            bb_position.values.reshape(-1, 1),
            bb_width.values.reshape(-1, 1)
        ])
        names.extend(['bb_upper', 'bb_lower', 'bb_position', 'bb_width'])
        
        # Stochastic Oscillator
        stoch_k = self._calculate_stochastic_k(close, high, low)
        stoch_d = stoch_k.rolling(window=3).mean()
        
        features.extend([
            stoch_k.values.reshape(-1, 1),
            stoch_d.values.reshape(-1, 1)
        ])
        names.extend(['stoch_k', 'stoch_d'])
        
        # Williams %R
        williams_r = self._calculate_williams_r(close, high, low)
        features.append(williams_r.values.reshape(-1, 1))
        names.append('williams_r')
        
        return features, names
    
    def _create_statistical_features(self, close: pd.Series) -> Tuple[List[np.ndarray], List[str]]:
        """Create statistical features."""
        features = []
        names = []
        
        returns = close.pct_change()
        
        # Rolling statistics
        for window in [10, 20, 50]:
            # Mean and std
            rolling_mean = returns.rolling(window=window).mean()
            rolling_std = returns.rolling(window=window).std()
            
            features.extend([
                rolling_mean.values.reshape(-1, 1),
                rolling_std.values.reshape(-1, 1)
            ])
            names.extend([f'returns_mean_{window}', f'returns_std_{window}'])
            
            # Skewness and kurtosis
            rolling_skew = returns.rolling(window=window).skew()
            rolling_kurt = returns.rolling(window=window).kurt()
            
            features.extend([
                rolling_skew.values.reshape(-1, 1),
                rolling_kurt.values.reshape(-1, 1)
            ])
            names.extend([f'returns_skew_{window}', f'returns_kurt_{window}'])
        
        return features, names
    
    def _create_volatility_features(self, close: pd.Series, high: pd.Series, 
                                  low: pd.Series) -> Tuple[List[np.ndarray], List[str]]:
        """Create volatility-based features."""
        features = []
        names = []
        
        returns = close.pct_change()
        
        # Historical volatility (different windows)
        for window in [10, 20, 50]:
            hist_vol = returns.rolling(window=window).std() * np.sqrt(252)
            features.append(hist_vol.values.reshape(-1, 1))
            names.append(f'hist_vol_{window}')
        
        # True Range and Average True Range
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        
        atr_14 = true_range.rolling(window=14).mean()
        atr_20 = true_range.rolling(window=20).mean()
        
        features.extend([
            true_range.values.reshape(-1, 1),
            atr_14.values.reshape(-1, 1),
            atr_20.values.reshape(-1, 1)
        ])
        names.extend(['true_range', 'atr_14', 'atr_20'])
        
        # Volatility ratios
        vol_ratio_short_long = (returns.rolling(10).std() / returns.rolling(50).std())
        features.append(vol_ratio_short_long.values.reshape(-1, 1))
        names.append('vol_ratio_10_50')
        
        return features, names
    
    def _create_volume_features(self, close: pd.Series, volume: pd.Series) -> Tuple[List[np.ndarray], List[str]]:
        """Create volume-based features."""
        features = []
        names = []
        
        # Volume moving averages
        for window in [10, 20, 50]:
            vol_ma = volume.rolling(window=window).mean()
            vol_ratio = volume / vol_ma
            
            features.extend([
                vol_ma.values.reshape(-1, 1),
                vol_ratio.values.reshape(-1, 1)
            ])
            names.extend([f'volume_ma_{window}', f'volume_ratio_{window}'])
        
        # On-Balance Volume (OBV)
        obv = np.where(close > close.shift(1), volume, 
                      np.where(close < close.shift(1), -volume, 0)).cumsum()
        features.append(obv.reshape(-1, 1))
        names.append('obv')
        
        # Volume Rate of Change
        vol_roc = volume.pct_change(periods=10)
        features.append(vol_roc.values.reshape(-1, 1))
        names.append('volume_roc_10')
        
        return features, names
    
    def _create_advanced_features(self, close: pd.Series, high: pd.Series, 
                                low: pd.Series) -> Tuple[List[np.ndarray], List[str]]:
        """Create advanced mathematical features."""
        features = []
        names = []
        
        returns = close.pct_change()
        
        # Autocorrelation features
        for lag in [1, 5, 10]:
            autocorr = returns.rolling(window=50).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
            )
            features.append(autocorr.values.reshape(-1, 1))
            names.append(f'autocorr_lag_{lag}')
        
        # Rolling correlation with lagged returns
        for lag in [1, 3, 5]:
            lagged_returns = returns.shift(lag)
            rolling_corr = returns.rolling(window=20).corr(lagged_returns)
            features.append(rolling_corr.values.reshape(-1, 1))
            names.append(f'return_lag_corr_{lag}')
        
        # Momentum features
        momentum_periods = [5, 10, 20]
        for period in momentum_periods:
            momentum = close / close.shift(period) - 1
            features.append(momentum.values.reshape(-1, 1))
            names.append(f'momentum_{period}')
        
        return features, names
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_stochastic_k(self, close: pd.Series, high: pd.Series, 
                              low: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Stochastic %K."""
        lowest_low = low.rolling(window=window).min()
        highest_high = high.rolling(window=window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        return k_percent
    
    def _calculate_williams_r(self, close: pd.Series, high: pd.Series, 
                            low: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    
    def get_feature_importance_by_category(self) -> Dict[str, List[str]]:
        """Group features by category for analysis."""
        categories = {
            'price': [name for name in self.feature_names if any(x in name for x in ['returns', 'ratio', 'gap', 'typical'])],
            'technical': [name for name in self.feature_names if any(x in name for x in ['sma', 'ema', 'macd', 'rsi', 'bb', 'stoch', 'williams'])],
            'statistical': [name for name in self.feature_names if any(x in name for x in ['mean', 'std', 'skew', 'kurt'])],
            'volatility': [name for name in self.feature_names if any(x in name for x in ['vol', 'atr', 'true_range'])],
            'volume': [name for name in self.feature_names if 'volume' in name or 'obv' in name],
            'advanced': [name for name in self.feature_names if any(x in name for x in ['autocorr', 'corr', 'momentum'])]
        }
        return categories

class SimpleFeatureEngineer:
    """
    Simplified feature engineer for basic use cases.
    Creates essential technical indicators without complexity.
    """
    
    def __init__(self):
        self.feature_names = []
    
    def create_basic_features(self, prices: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Create basic feature set from price data.
        
        Args:
            prices: Array of closing prices
        
        Returns:
            Tuple of (features_array, feature_names)
        """
        close = pd.Series(prices.flatten())
        features = []
        names = []
        
        # Returns
        returns = close.pct_change()
        features.append(returns.values.reshape(-1, 1))
        names.append('returns')
        
        # Simple moving averages
        for window in [5, 10, 20]:
            sma = close.rolling(window=window).mean()
            sma_ratio = close / sma
            
            features.extend([
                sma.values.reshape(-1, 1),
                sma_ratio.values.reshape(-1, 1)
            ])
            names.extend([f'sma_{window}', f'sma_{window}_ratio'])
        
        # RSI
        rsi = self._calculate_rsi(close)
        features.append(rsi.values.reshape(-1, 1))
        names.append('rsi')
        
        # Volatility
        volatility = returns.rolling(window=20).std()
        features.append(volatility.values.reshape(-1, 1))
        names.append('volatility_20')
        
        # Combine features
        combined_features = np.column_stack(features)
        feature_df = pd.DataFrame(combined_features, columns=names)
        feature_df = feature_df.fillna(method='ffill').fillna(method='bfill')
        
        self.feature_names = names
        return feature_df.values, names
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
