import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy.signal import find_peaks, argrelextrema
from sklearn.cluster import DBSCAN
import warnings

class PatternRecognizer:
    """
    Advanced pattern recognition for stock price data.
    Identifies support/resistance levels, trends, and common patterns.
    """
    
    def __init__(self, min_pattern_length: int = 5):
        """
        Initialize pattern recognizer.
        
        Args:
            min_pattern_length: Minimum length for pattern detection
        """
        self.min_pattern_length = min_pattern_length
        self.patterns_found = {}
        
    def find_support_resistance(self, 
                              prices: np.ndarray, 
                              window: int = 20,
                              min_touches: int = 2) -> Dict[str, List[float]]:
        """
        Identify support and resistance levels.
        
        Args:
            prices: Array of price data
            window: Window size for local extrema detection
            min_touches: Minimum number of touches to confirm level
        
        Returns:
            Dictionary with 'support' and 'resistance' levels
        """
        prices_flat = prices.flatten()
        
        # Find local maxima (potential resistance)
        resistance_indices = argrelextrema(prices_flat, np.greater, order=window)[0]
        resistance_prices = prices_flat[resistance_indices]
        
        # Find local minima (potential support)
        support_indices = argrelextrema(prices_flat, np.less, order=window)[0]
        support_prices = prices_flat[support_indices]
        
        # Cluster similar levels together
        support_levels = self._cluster_levels(support_prices, min_touches)
        resistance_levels = self._cluster_levels(resistance_prices, min_touches)
        
        return {
            'support': support_levels,
            'resistance': resistance_levels,
            'support_indices': support_indices.tolist(),
            'resistance_indices': resistance_indices.tolist()
        }
    
    def _cluster_levels(self, levels: np.ndarray, min_touches: int) -> List[float]:
        """
        Cluster price levels that are close together.
        
        Args:
            levels: Array of price levels
            min_touches: Minimum touches to consider a valid level
        
        Returns:
            List of confirmed support/resistance levels
        """
        if len(levels) < min_touches:
            return []
        
        # Use DBSCAN to cluster nearby levels
        levels_reshaped = levels.reshape(-1, 1)
        eps = np.std(levels) * 0.02  # 2% of standard deviation as epsilon
        
        clustering = DBSCAN(eps=eps, min_samples=min_touches).fit(levels_reshaped)
        
        confirmed_levels = []
        for cluster_id in set(clustering.labels_):
            if cluster_id != -1:  # Ignore noise points
                cluster_levels = levels[clustering.labels_ == cluster_id]
                confirmed_levels.append(np.mean(cluster_levels))
        
        return sorted(confirmed_levels)
    
    def detect_trend(self, 
                    prices: np.ndarray, 
                    window: int = 20) -> Dict[str, any]:
        """
        Detect overall trend in price data.
        
        Args:
            prices: Array of price data
            window: Window for trend analysis
        
        Returns:
            Dictionary with trend information
        """
        prices_flat = prices.flatten()
        
        # Calculate moving average slope
        ma = pd.Series(prices_flat).rolling(window=window).mean().dropna()
        
        # Calculate trend strength using linear regression
        x = np.arange(len(ma))
        slope, intercept = np.polyfit(x, ma.values, 1)
        
        # Determine trend direction
        if slope > 0.001:
            trend_direction = "upward"
        elif slope < -0.001:
            trend_direction = "downward"
        else:
            trend_direction = "sideways"
        
        # Calculate trend strength (R-squared)
        y_pred = slope * x + intercept
        ss_res = np.sum((ma.values - y_pred) ** 2)
        ss_tot = np.sum((ma.values - np.mean(ma.values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'direction': trend_direction,
            'slope': slope,
            'strength': r_squared,
            'moving_average': ma.values,
            'trend_line': y_pred
        }
    
    def find_triangles(self, 
                      prices: np.ndarray, 
                      high_prices: Optional[np.ndarray] = None,
                      low_prices: Optional[np.ndarray] = None) -> Dict[str, List[Dict]]:
        """
        Detect triangle patterns (ascending, descending, symmetrical).
        
        Args:
            prices: Array of closing prices
            high_prices: Array of high prices (optional)
            low_prices: Array of low prices (optional)
        
        Returns:
            Dictionary with detected triangle patterns
        """
        if high_prices is None:
            high_prices = prices
        if low_prices is None:
            low_prices = prices
            
        triangles = {
            'ascending': [],
            'descending': [],
            'symmetrical': []
        }
        
        # Find swing highs and lows
        swing_highs = self._find_swing_points(high_prices.flatten(), 'high')
        swing_lows = self._find_swing_points(low_prices.flatten(), 'low')
        
        # Analyze triangle patterns
        for i in range(len(swing_highs) - 3):
            for j in range(len(swing_lows) - 3):
                pattern = self._analyze_triangle_pattern(
                    swing_highs[i:i+4], swing_lows[j:j+4]
                )
                if pattern['type'] != 'none':
                    triangles[pattern['type']].append(pattern)
        
        return triangles
    
    def _find_swing_points(self, prices: np.ndarray, point_type: str) -> List[Dict]:
        """Find swing high or low points."""
        if point_type == 'high':
            indices = argrelextrema(prices, np.greater, order=5)[0]
        else:
            indices = argrelextrema(prices, np.less, order=5)[0]
        
        swing_points = []
        for idx in indices:
            swing_points.append({
                'index': idx,
                'price': prices[idx],
                'type': point_type
            })
        
        return swing_points
    
    def _analyze_triangle_pattern(self, highs: List[Dict], lows: List[Dict]) -> Dict:
        """Analyze if swing points form a triangle pattern."""
        if len(highs) < 2 or len(lows)
