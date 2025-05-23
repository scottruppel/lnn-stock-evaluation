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
        if len(highs) < 2 or len(lows) < 2:
            return {'type': 'none'}
        
        # Calculate slopes for high and low trend lines
        high_indices = [h['index'] for h in highs]
        high_prices = [h['price'] for h in highs]
        low_indices = [l['index'] for l in lows]
        low_prices = [l['price'] for l in lows]
        
        # Linear regression for trend lines
        high_slope = np.polyfit(high_indices, high_prices, 1)[0] if len(high_indices) > 1 else 0
        low_slope = np.polyfit(low_indices, low_prices, 1)[0] if len(low_indices) > 1 else 0
        
        # Determine triangle type
        if abs(high_slope) < 0.001 and low_slope > 0.001:
            pattern_type = 'ascending'
        elif high_slope < -0.001 and abs(low_slope) < 0.001:
            pattern_type = 'descending'
        elif high_slope < -0.001 and low_slope > 0.001:
            pattern_type = 'symmetrical'
        else:
            pattern_type = 'none'
        
        return {
            'type': pattern_type,
            'high_slope': high_slope,
            'low_slope': low_slope,
            'start_index': min(high_indices + low_indices),
            'end_index': max(high_indices + low_indices),
            'highs': highs,
            'lows': lows
        }
    
    def detect_double_patterns(self, prices: np.ndarray) -> Dict[str, List[Dict]]:
        """
        Detect double top and double bottom patterns.
        
        Args:
            prices: Array of price data
        
        Returns:
            Dictionary with detected double patterns
        """
        prices_flat = prices.flatten()
        patterns = {'double_top': [], 'double_bottom': []}
        
        # Find peaks and troughs
        peaks, _ = find_peaks(prices_flat, distance=20, prominence=np.std(prices_flat)*0.5)
        troughs, _ = find_peaks(-prices_flat, distance=20, prominence=np.std(prices_flat)*0.5)
        
        # Check for double tops
        for i in range(len(peaks) - 1):
            for j in range(i + 1, len(peaks)):
                if self._is_double_pattern(prices_flat, peaks[i], peaks[j], 'top'):
                    patterns['double_top'].append({
                        'first_peak': {'index': peaks[i], 'price': prices_flat[peaks[i]]},
                        'second_peak': {'index': peaks[j], 'price': prices_flat[peaks[j]]},
                        'pattern_strength': self._calculate_pattern_strength(
                            prices_flat, peaks[i], peaks[j], 'top'
                        )
                    })
        
        # Check for double bottoms
        for i in range(len(troughs) - 1):
            for j in range(i + 1, len(troughs)):
                if self._is_double_pattern(prices_flat, troughs[i], troughs[j], 'bottom'):
                    patterns['double_bottom'].append({
                        'first_trough': {'index': troughs[i], 'price': prices_flat[troughs[i]]},
                        'second_trough': {'index': troughs[j], 'price': prices_flat[troughs[j]]},
                        'pattern_strength': self._calculate_pattern_strength(
                            prices_flat, troughs[i], troughs[j], 'bottom'
                        )
                    })
        
        return patterns
    
    def _is_double_pattern(self, prices: np.ndarray, idx1: int, idx2: int, pattern_type: str) -> bool:
        """Check if two points form a valid double pattern."""
        # Minimum distance between points
        if abs(idx2 - idx1) < 20:
            return False
        
        # Similar price levels (within 2% of each other)
        price_diff = abs(prices[idx1] - prices[idx2]) / max(prices[idx1], prices[idx2])
        if price_diff > 0.02:
            return False
        
        # Check if there's a significant valley/peak between them
        between_prices = prices[min(idx1, idx2):max(idx1, idx2)]
        
        if pattern_type == 'top':
            # For double top, there should be a valley between peaks
            min_between = np.min(between_prices)
            if (prices[idx1] - min_between) / prices[idx1] < 0.05:
                return False
        else:
            # For double bottom, there should be a peak between troughs
            max_between = np.max(between_prices)
            if (max_between - prices[idx1]) / prices[idx1] < 0.05:
                return False
        
        return True
    
    def _calculate_pattern_strength(self, prices: np.ndarray, idx1: int, idx2: int, pattern_type: str) -> float:
        """Calculate the strength/reliability of a pattern."""
        # Price similarity (closer = stronger)
        price_similarity = 1 - (abs(prices[idx1] - prices[idx2]) / max(prices[idx1], prices[idx2]))
        
        # Volume consideration (if available) - placeholder for now
        volume_confirmation = 0.5
        
        # Time span consideration (not too close, not too far)
        time_span = abs(idx2 - idx1)
        optimal_span = 40  # ~2 months of daily data
        time_factor = 1 - abs(time_span - optimal_span) / optimal_span
        time_factor = max(0, min(1, time_factor))
        
        # Combined strength score
        strength = (price_similarity * 0.5 + volume_confirmation * 0.3 + time_factor * 0.2)
        return strength
    
    def detect_head_and_shoulders(self, prices: np.ndarray) -> List[Dict]:
        """
        Detect head and shoulders patterns.
        
        Args:
            prices: Array of price data
        
        Returns:
            List of detected head and shoulders patterns
        """
        prices_flat = prices.flatten()
        patterns = []
        
        # Find significant peaks
        peaks, properties = find_peaks(prices_flat, distance=15, prominence=np.std(prices_flat)*0.3)
        
        if len(peaks) < 3:
            return patterns
        
        # Check each set of 3 consecutive peaks
        for i in range(len(peaks) - 2):
            left_shoulder = peaks[i]
            head = peaks[i + 1]
            right_shoulder = peaks[i + 2]
            
            if self._is_head_and_shoulders(prices_flat, left_shoulder, head, right_shoulder):
                # Calculate neckline
                left_trough = self._find_trough_between(prices_flat, left_shoulder, head)
                right_trough = self._find_trough_between(prices_flat, head, right_shoulder)
                
                patterns.append({
                    'left_shoulder': {'index': left_shoulder, 'price': prices_flat[left_shoulder]},
                    'head': {'index': head, 'price': prices_flat[head]},
                    'right_shoulder': {'index': right_shoulder, 'price': prices_flat[right_shoulder]},
                    'left_trough': left_trough,
                    'right_trough': right_trough,
                    'neckline_slope': self._calculate_neckline_slope(left_trough, right_trough),
                    'pattern_strength': self._calculate_hs_strength(
                        prices_flat, left_shoulder, head, right_shoulder
                    )
                })
        
        return patterns
    
    def _is_head_and_shoulders(self, prices: np.ndarray, left: int, head: int, right: int) -> bool:
        """Check if three peaks form a head and shoulders pattern."""
        # Head should be higher than both shoulders
        if prices[head] <= prices[left] or prices[head] <= prices[right]:
            return False
        
        # Shoulders should be roughly similar height (within 5%)
        shoulder_diff = abs(prices[left] - prices[right]) / max(prices[left], prices[right])
        if shoulder_diff > 0.05:
            return False
        
        # Head should be significantly higher than shoulders (at least 3%)
        left_diff = (prices[head] - prices[left]) / prices[left]
        right_diff = (prices[head] - prices[right]) / prices[right]
        
        if left_diff < 0.03 or right_diff < 0.03:
            return False
        
        return True
    
    def _find_trough_between(self, prices: np.ndarray, start_idx: int, end_idx: int) -> Dict:
        """Find the lowest point between two indices."""
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        
        segment = prices[start_idx:end_idx+1]
        min_idx = np.argmin(segment) + start_idx
        
        return {'index': min_idx, 'price': prices[min_idx]}
    
    def _calculate_neckline_slope(self, left_trough: Dict, right_trough: Dict) -> float:
        """Calculate the slope of the neckline."""
        if left_trough['index'] == right_trough['index']:
            return 0.0
        
        slope = (right_trough['price'] - left_trough['price']) / (right_trough['index'] - left_trough['index'])
        return slope
    
    def _calculate_hs_strength(self, prices: np.ndarray, left: int, head: int, right: int) -> float:
        """Calculate head and shoulders pattern strength."""
        # Symmetry of shoulders
        shoulder_symmetry = 1 - abs(prices[left] - prices[right]) / max(prices[left], prices[right])
        
        # Head prominence
        head_prominence = min(
            (prices[head] - prices[left]) / prices[left],
            (prices[head] - prices[right]) / prices[right]
        )
        head_prominence = min(1.0, head_prominence / 0.1)  # Normalize to 0-1
        
        # Time symmetry
        left_time = head - left
        right_time = right - head
        time_symmetry = 1 - abs(left_time - right_time) / max(left_time, right_time)
        
        # Combined strength
        strength = (shoulder_symmetry * 0.4 + head_prominence * 0.4 + time_symmetry * 0.2)
        return strength
    
    def get_pattern_summary(self, prices: np.ndarray) -> Dict:
        """
        Get a comprehensive summary of all detected patterns.
        
        Args:
            prices: Array of price data
        
        Returns:
            Dictionary with all pattern analysis results
        """
        # Run all pattern detection methods
        support_resistance = self.find_support_resistance(prices)
        trend_info = self.detect_trend(prices)
        triangles = self.find_triangles(prices)
        double_patterns = self.detect_double_patterns(prices)
        hs_patterns = self.detect_head_and_shoulders(prices)
        
        summary = {
            'support_resistance': support_resistance,
            'trend': trend_info,
            'triangles': triangles,
            'double_patterns': double_patterns,
            'head_and_shoulders': hs_patterns,
            'pattern_count': {
                'support_levels': len(support_resistance['support']),
                'resistance_levels': len(support_resistance['resistance']),
                'triangles': sum(len(v) for v in triangles.values()),
                'double_patterns': sum(len(v) for v in double_patterns.values()),
                'head_and_shoulders': len(hs_patterns)
            }
        }
        
        return summary
