import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from typing import Dict, Tuple, List, Optional, Union
import warnings

class StockDataPreprocessor:
    """
    Comprehensive preprocessing class for stock data.
    Handles scaling, feature engineering, and data preparation for LNN training.
    """
    
    def __init__(self, 
                 scaling_method: str = 'minmax',
                 feature_range: Tuple[float, float] = (-1, 1),
                 handle_missing: str = 'forward_fill'):
        """
        Initialize the preprocessor.
        
        Args:
            scaling_method: 'minmax', 'standard', or 'robust'
            feature_range: Range for MinMaxScaler (only used if scaling_method='minmax')
            handle_missing: 'forward_fill', 'backward_fill', 'interpolate', or 'drop'
        """
        self.scaling_method = scaling_method
        self.feature_range = feature_range
        self.handle_missing = handle_missing
        self.scalers = {}
        self.is_fitted = False
        
        # Initialize scaler based on method
        if scaling_method == 'minmax':
            self.scaler_class = lambda: MinMaxScaler(feature_range=feature_range)
        elif scaling_method == 'standard':
            self.scaler_class = lambda: StandardScaler()
        elif scaling_method == 'robust':
            self.scaler_class = lambda: RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        if self.handle_missing == 'forward_fill':
            return data.fillna(method='ffill')
        elif self.handle_missing == 'backward_fill':
            return data.fillna(method='bfill')
        elif self.handle_missing == 'interpolate':
            return data.interpolate()
        elif self.handle_missing == 'drop':
            return data.dropna()
        else:
            raise ValueError(f"Unknown missing value handling method: {self.handle_missing}")
    
    def fit(self, price_data: Dict[str, np.ndarray]) -> 'StockDataPreprocessor':
        """
        Fit scalers to the training data.
        
        Args:
            price_data: Dictionary with ticker symbols as keys and price arrays as values
        
        Returns:
            self: Returns the fitted preprocessor
        """
        print("Fitting scalers to training data...")
        
        for ticker, prices in price_data.items():
            # Handle missing values
            prices_clean = self._clean_array(prices)
            
            # Initialize and fit scaler
            scaler = self.scaler_class()
            scaler.fit(prices_clean.reshape(-1, 1))
            self.scalers[ticker] = scaler
            
            print(f"Fitted scaler for {ticker}: shape {prices_clean.shape}")
        
        self.is_fitted = True
        return self
    
    def transform(self, price_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Transform price data using fitted scalers.
        
        Args:
            price_data: Dictionary with ticker symbols as keys and price arrays as values
        
        Returns:
            scaled_data: Dictionary with scaled price arrays
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming data")
        
        scaled_data = {}
        
        for ticker, prices in price_data.items():
            if ticker not in self.scalers:
                raise ValueError(f"No scaler found for ticker {ticker}")
            
            # Handle missing values
            prices_clean = self._clean_array(prices)
            
            # Transform using fitted scaler
            scaled_prices = self.scalers[ticker].transform(prices_clean.reshape(-1, 1))
            scaled_data[ticker] = scaled_prices
            
        return scaled_data
    
    def fit_transform(self, price_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Fit scalers and transform data in one step."""
        return self.fit(price_data).transform(price_data)
    
    def inverse_transform(self, scaled_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Reverse the scaling transformation.
        
        Args:
            scaled_data: Dictionary with scaled price arrays
        
        Returns:
            original_data: Dictionary with original scale price arrays
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before inverse transforming")
        
        original_data = {}
        
        for ticker, scaled_prices in scaled_data.items():
            if ticker not in self.scalers:
                raise ValueError(f"No scaler found for ticker {ticker}")
            
            # Inverse transform
            original_prices = self.scalers[ticker].inverse_transform(scaled_prices.reshape(-1, 1))
            original_data[ticker] = original_prices
            
        return original_data
    
    def inverse_transform_single(self, ticker: str, scaled_values: np.ndarray) -> np.ndarray:
        """
        Inverse transform values for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            scaled_values: Scaled values to transform back
        
        Returns:
            original_values: Values in original scale
        """
        if ticker not in self.scalers:
            raise ValueError(f"No scaler found for ticker {ticker}")
        
        # Ensure proper shape for inverse transform
        if scaled_values.ndim == 1:
            scaled_values = scaled_values.reshape(-1, 1)
        
        return self.scalers[ticker].inverse_transform(scaled_values)
    
    def _clean_array(self, arr: np.ndarray) -> np.ndarray:
        """Clean numpy array by handling missing values."""
        # Convert to pandas Series for easier missing value handling
        series = pd.Series(arr.flatten())
        
        # Handle missing values
        if self.handle_missing == 'forward_fill':
            series = series.fillna(method='ffill')
        elif self.handle_missing == 'backward_fill':
            series = series.fillna(method='bfill')
        elif self.handle_missing == 'interpolate':
            series = series.interpolate()
        elif self.handle_missing == 'drop':
            series = series.dropna()
        
        # Convert back to numpy array
        return series.values
    
    def get_scaling_info(self) -> Dict[str, Dict[str, float]]:
        """Get scaling information for each ticker."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        
        scaling_info = {}
        
        for ticker, scaler in self.scalers.items():
            info = {'scaling_method': self.scaling_method}
            
            if hasattr(scaler, 'data_min_'):
                info['data_min'] = float(scaler.data_min_[0])
                info['data_max'] = float(scaler.data_max_[0])
                info['data_range'] = float(scaler.data_range_[0])
            
            if hasattr(scaler, 'mean_'):
                info['mean'] = float(scaler.mean_[0])
                info['std'] = float(scaler.scale_[0])
            
            if hasattr(scaler, 'center_'):
                info['center'] = float(scaler.center_[0])
                info['scale'] = float(scaler.scale_[0])
            
            scaling_info[ticker] = info
        
        return scaling_info

class FeatureEngineer:
    """
    Feature engineering class for creating additional features from stock price data.
    """
    
    def __init__(self):
        self.feature_names = []
    
    def create_technical_indicators(self, 
                                  prices: np.ndarray, 
                                  volume: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create technical indicators from price data.
        
        Args:
            prices: Array of closing prices
            volume: Optional array of volume data
        
        Returns:
            features: Array of engineered features
        """
        features = []
        feature_names = []
        
        # Convert to pandas Series for easier calculation
        price_series = pd.Series(prices.flatten())
        
        # Simple Moving Averages
        for window in [5, 10, 20, 50]:
            sma = price_series.rolling(window=window).mean()
            features.append(sma.values.reshape(-1, 1))
            feature_names.append(f'SMA_{window}')
            
            # Price relative to SMA
            price_ratio = (price_series / sma).values.reshape(-1, 1)
            features.append(price_ratio)
            feature_names.append(f'Price_SMA_{window}_Ratio')
        
        # Exponential Moving Averages
        for span in [12, 26]:
            ema = price_series.ewm(span=span).mean()
            features.append(ema.values.reshape(-1, 1))
            feature_names.append(f'EMA_{span}')
        
        # MACD
        ema12 = price_series.ewm(span=12).mean()
        ema26 = price_series.ewm(span=26).mean()
        macd = (ema12 - ema26).values.reshape(-1, 1)
        features.append(macd)
        feature_names.append('MACD')
        
        # RSI
        rsi = self._calculate_rsi(price_series).values.reshape(-1, 1)
        features.append(rsi)
        feature_names.append('RSI')
        
        # Bollinger Bands
        sma20 = price_series.rolling(window=20).mean()
        std20 = price_series.rolling(window=20).std()
        bb_upper = (sma20 + 2 * std20).values.reshape(-1, 1)
        bb_lower = (sma20 - 2 * std20).values.reshape(-1, 1)
        bb_position = ((price_series - bb_lower.flatten()) / 
                      (bb_upper.flatten() - bb_lower.flatten())).values.reshape(-1, 1)
        
        features.extend([bb_upper, bb_lower, bb_position])
        feature_names.extend(['BB_Upper', 'BB_Lower', 'BB_Position'])
        
        # Volatility features
        returns = price_series.pct_change()
        volatility = returns.rolling(window=20).std().values.reshape(-1, 1)
        features.append(volatility)
        feature_names.append('Volatility_20')
        
        # Volume features (if provided)
        if volume is not None:
            volume_series = pd.Series(volume.flatten())
            volume_sma = volume_series.rolling(window=20).mean().values.reshape(-1, 1)
            volume_ratio = (volume_series / volume_sma.flatten()).values.reshape(-1, 1)
            features.extend([volume_sma, volume_ratio])
            feature_names.extend(['Volume_SMA_20', 'Volume_Ratio'])
        
        self.feature_names = feature_names
        
        # Combine all features
        combined_features = np.concatenate(features, axis=1)
        
        # Handle NaN values (forward fill)
        combined_df = pd.DataFrame(combined_features, columns=feature_names)
        combined_df = combined_df.fillna(method='ffill').fillna(method='bfill')
        
        return combined_df.values
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_feature_names(self) -> List[str]:
        """Get the names of engineered features."""
        return self.feature_names.copy()

def prepare_model_data(price_data: Dict[str, np.ndarray], 
                      target_ticker: str,
                      sequence_length: int = 30,
                      train_ratio: float = 0.8,
                      add_features: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for model training.
    
    Args:
        price_data: Dictionary of scaled price data
        target_ticker: Ticker symbol to predict
        sequence_length: Length of input sequences
        train_ratio: Ratio of data to use for training
        add_features: Whether to add engineered features
    
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    # Combine input features (all tickers except target)
    input_features = []
    for ticker, prices in price_data.items():
        if ticker != target_ticker:
            input_features.append(prices)
    
    # If we want to add engineered features
    if add_features:
        engineer = FeatureEngineer()
        for ticker, prices in price_data.items():
            if ticker != target_ticker:
                tech_features = engineer.create_technical_indicators(prices)
                input_features.append(tech_features)
    
    # Combine all input features
    input_data = np.concatenate(input_features, axis=1)
    target_data = price_data[target_ticker]
    
    # Ensure same length
    min_length = min(len(input_data), len(target_data))
    input_data = input_data[:min_length]
    target_data = target_data[:min_length]
    
    # Split into train/test
    train_size = int(train_ratio * len(input_data))
    
    train_input = input_data[:train_size]
    train_target = target_data[:train_size]
    test_input = input_data[train_size:]
    test_target = target_data[train_size:]
    
    # Create sequences
    from src.models.lnn_model import create_sequences
    
    X_train, y_train = create_sequences(train_input, train_target, sequence_length)
    X_test, y_test = create_sequences(test_input, test_target, sequence_length)
    
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
    
    return X_train, y_train, X_test, y_test
