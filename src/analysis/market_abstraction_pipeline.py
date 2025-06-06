#!/usr/bin/env python3
"""
Market Abstraction System integrated with your existing LNN pipeline.
This adds high-level market intelligence to your feature engineering.

Usage on Jetson Nano:
1. Save this as: src/analysis/market_abstraction_pipeline.py
2. Modify your run_analysis.py to include abstraction features
3. Run: python scripts/run_analysis.py --use-abstraction
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import json
from datetime import datetime

# Add your existing paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Your existing imports
from data.data_loader import StockDataLoader
from analysis.feature_engineering import AdvancedFeatureEngineer

class StockSectorClassifier:
    """
    Automatically determines what sector a stock belongs to.
    Uses multiple data sources to classify any ticker.
    """
    
    def __init__(self):
        # Major sector definitions (expandable)
        self.sector_definitions = {
            'technology': {
                'keywords': ['software', 'semiconductor', 'computer', 'internet', 'cloud', 'AI', 'tech'],
                'known_tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'CRM', 'ORCL']
            },
            'finance': {
                'keywords': ['bank', 'insurance', 'financial', 'credit', 'loan', 'mortgage'],
                'known_tickers': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'BLK', 'AXP']
            },
            'healthcare': {
                'keywords': ['pharma', 'biotech', 'medical', 'health', 'drug', 'hospital'],
                'known_tickers': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'LLY', 'MRK', 'BMY', 'AMGN']
            },
            'energy': {
                'keywords': ['oil', 'gas', 'energy', 'petroleum', 'renewable', 'solar', 'wind'],
                'known_tickers': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'KMI', 'OKE', 'WMB']
            },
            'consumer_discretionary': {
                'keywords': ['retail', 'restaurant', 'automotive', 'luxury', 'entertainment', 'travel'],
                'known_tickers': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'F', 'GM']
            },
            'consumer_staples': {
                'keywords': ['food', 'beverage', 'household', 'grocery', 'consumer goods'],
                'known_tickers': ['WMT', 'PG', 'KO', 'PEP', 'COST', 'WBA', 'CL', 'KMB', 'GIS', 'K']
            },
            'industrials': {
                'keywords': ['manufacturing', 'aerospace', 'defense', 'construction', 'transportation'],
                'known_tickers': ['BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'LMT', 'DE', 'UNP']
            },
            'utilities': {
                'keywords': ['electric', 'utility', 'power', 'water', 'gas utility'],
                'known_tickers': ['NEE', 'DUK', 'SO', 'D', 'EXC', 'XEL', 'SRE', 'AEP', 'ES', 'AWK']
            },
            'materials': {
                'keywords': ['mining', 'chemicals', 'metals', 'steel', 'aluminum', 'copper'],
                'known_tickers': ['LIN', 'APD', 'SHW', 'FCX', 'NUE', 'PPG', 'ECL', 'DD', 'NEM', 'DOW']
            },
            'real_estate': {
                'keywords': ['real estate', 'REIT', 'property', 'land', 'commercial real estate'],
                'known_tickers': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SBAC', 'DLR', 'REG', 'AVB', 'EQR']
            },
            'communication': {
                'keywords': ['telecom', 'media', 'broadcasting', 'wireless', 'cable'],
                'known_tickers': ['T', 'VZ', 'CMCSA', 'DIS', 'NFLX', 'GOOGL', 'META', 'CHTR', 'TMUS', 'DISH']
            }
        }
    
    def classify_ticker(self, ticker: str) -> str:
        """
        Automatically classify a ticker into a sector.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            
        Returns:
            sector: Sector classification (e.g., 'technology')
        """
        ticker = ticker.upper().replace('^', '')  # Clean ticker
        
        # Method 1: Check known tickers first (fastest)
        for sector, definition in self.sector_definitions.items():
            if ticker in definition['known_tickers']:
                return sector
        
        # Method 2: Try to get company info from yfinance
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Check sector from yfinance (if available)
            if 'sector' in info and info['sector']:
                yf_sector = info['sector'].lower()
                
                # Map yfinance sectors to our definitions
                sector_mapping = {
                    'technology': 'technology',
                    'financial services': 'finance',
                    'healthcare': 'healthcare',
                    'energy': 'energy',
                    'consumer cyclical': 'consumer_discretionary',
                    'consumer defensive': 'consumer_staples',
                    'industrials': 'industrials',
                    'utilities': 'utilities',
                    'basic materials': 'materials',
                    'real estate': 'real_estate',
                    'communication services': 'communication'
                }
                
                for yf_name, our_name in sector_mapping.items():
                    if yf_name in yf_sector:
                        return our_name
            
            # Check business summary for keywords
            if 'longBusinessSummary' in info and info['longBusinessSummary']:
                summary = info['longBusinessSummary'].lower()
                
                for sector, definition in self.sector_definitions.items():
                    for keyword in definition['keywords']:
                        if keyword in summary:
                            return sector
        
        except Exception as e:
            print(f"   Warning: Could not classify {ticker} automatically: {e}")
        
        # Method 3: Default classification based on ticker patterns
        if ticker.startswith('QQQ') or 'TECH' in ticker:
            return 'technology'
        elif ticker.startswith('^') or 'SPY' in ticker or 'SPX' in ticker:
            return 'market_index'  # Special category for indices
        elif 'AGG' in ticker or 'BND' in ticker:
            return 'fixed_income'  # Special category for bonds
        
        # Default: classify as 'unknown' sector
        return 'unknown'
    
    def get_sector_peers(self, target_ticker: str, all_tickers: List[str]) -> Dict[str, List[str]]:
        """
        Group all your tickers by sector, with target ticker's sector emphasized.
        
        Returns:
            sector_groups: Dict mapping sectors to ticker lists
        """
        sector_groups = {}
        target_sector = self.classify_ticker(target_ticker)
        
        print(f"   Target ticker {target_ticker} classified as: {target_sector}")
        
        for ticker in all_tickers:
            sector = self.classify_ticker(ticker)
            
            if sector not in sector_groups:
                sector_groups[sector] = []
            
            sector_groups[sector].append(ticker)
        
        print(f"   Sector groupings:")
        for sector, tickers in sector_groups.items():
            print(f"     {sector}: {tickers}")
        
        return sector_groups, target_sector


class MarketAbstractionEngine:
    """
    Creates high-level market abstractions from your existing data.
    Integrates seamlessly with your current pipeline.
    """
    
    def __init__(self, all_tickers: List[str], target_ticker: str):
        self.all_tickers = all_tickers
        self.target_ticker = target_ticker
        
        # Initialize sector classifier
        self.sector_classifier = StockSectorClassifier()
        self.sector_groups, self.target_sector = self.sector_classifier.get_sector_peers(
            target_ticker, all_tickers
        )
        
        # Initialize feature scalers
        self.scalers = {
            'sector_rotation': StandardScaler(),
            'macro_regime': StandardScaler(),
            'microstructure': StandardScaler()
        }
        
        self.is_fitted = False
        
    def create_sector_rotation_features(self, price_data: Dict[str, np.ndarray], window: int = 20) -> np.ndarray:
        """
        Create features that capture sector rotation patterns.
        """
        print(f"   Creating sector rotation features (window={window})...")
        
        # Define sector representatives (you can modify these based on your tickers)
        sectors = {
            'market': ['^GSPC', 'SPY'],
            'tech': ['QQQ', 'XLK', 'AAPL', 'MSFT', 'GOOGL'],
            'bonds': ['AGG', 'TLT', 'BND'],
            'finance': ['XLF', 'JPM', 'BAC'],
            'energy': ['XLE', 'XOM', 'CVX'],
            'healthcare': ['XLV', 'JNJ', 'UNH'],
            'consumer': ['XLY', 'AMZN', 'TSLA']
        }
    
        # Calculate sector performance
        sector_returns = {}
        min_length = float('inf')
    
        for sector, tickers in sectors.items():
            # Find available tickers in this sector
            available_tickers = [t for t in tickers if t in price_data]
        
            if available_tickers:
                # Get prices for available tickers
                sector_prices = []
                for ticker in available_tickers:
                    prices = price_data[ticker]
                    if prices.ndim > 1:
                        prices = prices.flatten()
                    sector_prices.append(prices)
            
                # Find minimum length among sector tickers
                sector_min_length = min(len(p) for p in sector_prices)
                min_length = min(min_length, sector_min_length)
            
                # Align and average sector prices
                aligned_sector_prices = np.column_stack([p[-sector_min_length:] for p in sector_prices])
                avg_sector_price = np.mean(aligned_sector_prices, axis=1)
            
                # Calculate returns
                if len(avg_sector_price) > 1:
                    returns = np.diff(avg_sector_price) / avg_sector_price[:-1]
                    sector_returns[sector] = returns
    
        if not sector_returns or min_length == float('inf'):
            print("   WARNING: No valid sector data found")
            # Return dummy features
            n_samples = len(list(price_data.values())[0]) - window + 1
            return np.zeros((n_samples, 10))  # 10 dummy features
    
        # Align all sector returns to same length
        min_return_length = min(len(r) for r in sector_returns.values())
        aligned_returns = {s: r[-min_return_length:] for s, r in sector_returns.items()}
    
        # Calculate rolling features
        n_samples = min_return_length - window + 1
        rotation_features = []
    
        for i in range(n_samples):
            window_features = []
        
            # Get window returns for each sector
            window_sector_returns = {}
            for sector, returns in aligned_returns.items():
                if i + window <= len(returns):
                    window_sector_returns[sector] = returns[i:i+window]
        
            if len(window_sector_returns) < 2:
                # Not enough sectors, add dummy features
                rotation_features.append([0] * 10)
                continue
        
            # Calculate relative strength for each sector
            sector_strengths = {}
            for sector, returns in window_sector_returns.items():
                if len(returns) > 0:
                    # Average return over window
                    avg_return = np.mean(returns)
                    # Volatility
                    vol = np.std(returns) if len(returns) > 1 else 0
                    # Sharpe-like ratio
                    sharpe = avg_return / vol if vol > 0 else 0
                    sector_strengths[sector] = {
                        'return': avg_return,
                        'volatility': vol,
                        'sharpe': sharpe
                    }
        
            # Create features
            if sector_strengths:
                # 1. Best performing sector return
                best_return = max(s['return'] for s in sector_strengths.values())
                # 2. Worst performing sector return
                worst_return = min(s['return'] for s in sector_strengths.values())
                # 3. Spread between best and worst
                performance_spread = best_return - worst_return
                # 4. Average sector volatility
                avg_vol = np.mean([s['volatility'] for s in sector_strengths.values()])
                # 5. Tech vs Market relative strength
                tech_strength = sector_strengths.get('tech', {}).get('return', 0)
                market_strength = sector_strengths.get('market', {}).get('return', 0)
                tech_vs_market = tech_strength - market_strength
                # 6. Bonds vs Market relative strength (risk on/off indicator)
                bonds_strength = sector_strengths.get('bonds', {}).get('return', 0)
                bonds_vs_market = bonds_strength - market_strength
                # 7. Number of positive sectors
                positive_sectors = sum(1 for s in sector_strengths.values() if s['return'] > 0)
                # 8. Sector correlation (simplified)
                returns_matrix = [list(r) for r in window_sector_returns.values() if len(r) == window]
                if len(returns_matrix) > 1:
                    try:
                        corr_matrix = np.corrcoef(returns_matrix)
                        avg_correlation = np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
                    except:
                        avg_correlation = 0
                else:
                    avg_correlation = 0
                # 9. Momentum indicator (recent vs earlier performance)
                if window > 10:
                    recent_avg = np.mean([np.mean(r[-5:]) for r in window_sector_returns.values() if len(r) >= 5])
                    earlier_avg = np.mean([np.mean(r[:5]) for r in window_sector_returns.values() if len(r) >= 5])
                    momentum = recent_avg - earlier_avg
                else:
                    momentum = 0
                # 10. Volatility regime indicator
                high_vol_threshold = np.percentile([s['volatility'] for s in sector_strengths.values()], 75)
                high_vol_sectors = sum(1 for s in sector_strengths.values() if s['volatility'] > high_vol_threshold)
            
                window_features = [
                    best_return,
                    worst_return,
                    performance_spread,
                    avg_vol,
                    tech_vs_market,
                    bonds_vs_market,
                    positive_sectors / len(sector_strengths),  # Normalize
                    avg_correlation,
                    momentum,
                    high_vol_sectors / len(sector_strengths)  # Normalize
                ]
            else:
                window_features = [0] * 10
        
            rotation_features.append(window_features)
    
        rotation_array = np.array(rotation_features)
        print(f"   Created sector rotation features: {rotation_array.shape}")
    
        return rotation_array
    
    def create_market_regime_features(self, price_data: Dict[str, np.ndarray], window: int = 20) -> np.ndarray:
        """
        Identify market regimes (trending, ranging, volatile).
        """
        print(f"   Creating market regime features...")
    
        # Debug: Check input data
        if not price_data:
            print("   WARNING: No price data provided")
            return np.array([])
    
        # Get all price series and ensure they're 1D arrays
        price_arrays = []
        for ticker, prices in price_data.items():
            # Ensure prices is a 1D array
            if prices.ndim > 1:
                prices = prices.flatten()
            if len(prices) > 0:
                price_arrays.append(prices)
    
        if not price_arrays:
            print("   WARNING: No valid price data found")
            return np.array([])
    
        # Find minimum length
        min_length = min(len(p) for p in price_arrays)
    
        # Align all prices to same length
        aligned_prices = np.column_stack([p[-min_length:] for p in price_arrays])
    
        # Calculate regime features
        n_samples = len(aligned_prices) - window + 1
        regime_features = []
    
        for i in range(n_samples):
            window_prices = aligned_prices[i:i+window]
        
            # Skip if window is too small
            if len(window_prices) < 2:
                continue
        
            # Calculate returns safely
            try:
                # Calculate returns for each asset in the window
                window_returns = []
                for col in range(window_prices.shape[1]):
                    asset_prices = window_prices[:, col]
                    if len(asset_prices) > 1 and asset_prices[0] != 0:
                        asset_returns = np.diff(asset_prices) / asset_prices[:-1]
                        window_returns.append(asset_returns)
            
                if not window_returns:
                    # If no valid returns, create dummy features
                    regime_features.append([0, 0, 0, 0])
                    continue
                
                # Convert to array for calculations
                window_returns = np.array(window_returns).T  # Shape: (window-1, n_assets)
            
                # Calculate regime indicators
                # 1. Average correlation between assets
                if window_returns.shape[1] > 1:
                    corr_matrix = np.corrcoef(window_returns.T)
                    avg_correlation = np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
                else:
                    avg_correlation = 0
            
                # 2. Market volatility (average volatility across assets)
                volatilities = np.std(window_returns, axis=0)
                avg_volatility = np.mean(volatilities)
            
                # 3. Trend strength (average absolute return)
                avg_abs_return = np.mean(np.abs(window_returns))
            
                # 4. Volatility regime (high/low volatility indicator)
                volatility_percentile = np.percentile(volatilities, 75)
                high_vol_indicator = 1 if avg_volatility > volatility_percentile else 0
            
                regime_features.append([
                    avg_correlation,
                    avg_volatility,
                    avg_abs_return,
                    high_vol_indicator
                ])
            
            except Exception as e:
                print(f"   Warning: Error calculating regime features for window {i}: {e}")
                # Add dummy features for this window
                regime_features.append([0, 0, 0, 0])
    
        regime_array = np.array(regime_features)
        print(f"   Created regime features: {regime_array.shape}")
    
        return regime_array
    
    def create_cross_asset_features(self, price_data: dict, window: int = 20) -> np.ndarray:
        """
        Create cross-asset correlation and relative strength features.
    
        Args:
            price_data: Dictionary of {ticker: price_array}
            window: Rolling window size for calculations
        
        Returns:
            cross_asset_features: Array of shape [n_samples, n_cross_features]
        """
        if len(price_data) < 2:
            print("Warning: Need at least 2 assets for cross-asset features")
            return np.array([[0.0]])  # Return minimal array to avoid breaking pipeline
    
        tickers = list(price_data.keys())
        n_assets = len(tickers)
    
        # Get the target ticker (usually first in list or specified)
        target_ticker = tickers[0]  # You might want to make this configurable
    
        # Ensure all price arrays are 1D and same length
        processed_prices = {}
        min_length = float('inf')
    
        for ticker, prices in price_data.items():
            # Convert to numpy array and flatten to ensure 1D
            price_array = np.array(prices).flatten()
        
            # Skip if empty
            if len(price_array) == 0:
                print(f"Warning: Empty price data for {ticker}, skipping cross-asset features")
                return np.array([[0.0]])
        
            processed_prices[ticker] = price_array
            min_length = min(min_length, len(price_array))
    
        # Truncate all arrays to same length
        for ticker in processed_prices:
            processed_prices[ticker] = processed_prices[ticker][:min_length]
    
        print(f"Creating cross-asset features with {n_assets} assets, {min_length} samples")
    
        # Calculate the number of valid windows
        if min_length <= window:
            print(f"Warning: Data length ({min_length}) <= window size ({window})")
            return np.array([[0.0]])
    
        n_samples = min_length - window
        n_features = n_assets * (n_assets - 1) + n_assets  # Correlations + relative strengths
    
        cross_features = np.zeros((n_samples, n_features))
    
        target_prices = processed_prices[target_ticker]
    
        for t in range(window, min_length):
            feature_idx = 0
        
            try:
                # Calculate returns for target asset in this window
                target_window = target_prices[t-window:t]
                if len(target_window) < 2:
                    continue
                
                target_returns = np.diff(target_window) / target_window[:-1]
            
                # Skip if we have any invalid returns
                if len(target_returns) == 0 or np.any(np.isnan(target_returns)) or np.any(np.isinf(target_returns)):
                    continue
            
                # Cross-correlations with other assets
                for other_ticker in tickers:
                    if other_ticker == target_ticker:
                        continue
                
                    other_prices = processed_prices[other_ticker]
                    other_window = other_prices[t-window:t]
                
                    if len(other_window) < 2:
                        cross_features[t-window, feature_idx] = 0.0
                        feature_idx += 1
                        continue
                
                    other_returns = np.diff(other_window) / other_window[:-1]
                
                    # Calculate correlation if we have valid data
                    if (len(other_returns) == len(target_returns) and 
                        len(target_returns) > 1 and
                        not np.any(np.isnan(other_returns)) and 
                        not np.any(np.isinf(other_returns))):
                    
                        correlation = np.corrcoef(target_returns, other_returns)[0, 1]
                    
                        # Handle NaN correlations (constant returns)
                        if np.isnan(correlation):
                            correlation = 0.0
                    
                        cross_features[t-window, feature_idx] = correlation
                    else:
                        cross_features[t-window, feature_idx] = 0.0
                
                    feature_idx += 1
            
                # Relative strength indicators
                for ticker in tickers:
                    prices = processed_prices[ticker]
                    window_prices = prices[t-window:t]
                
                    if len(window_prices) >= 2:
                        # Simple relative strength: (current - start) / start
                        rel_strength = (window_prices[-1] - window_prices[0]) / window_prices[0]
                    
                        if np.isnan(rel_strength) or np.isinf(rel_strength):
                            rel_strength = 0.0
                    
                        cross_features[t-window, feature_idx] = rel_strength
                    else:
                        cross_features[t-window, feature_idx] = 0.0
                
                    feature_idx += 1
                
            except Exception as e:
                print(f"Warning: Error processing sample {t}: {e}")
                # Fill with zeros for this sample
                cross_features[t-window, :] = 0.0
                continue
    
        print(f"✓ Created cross-asset features: {cross_features.shape}")
        return cross_features
    
    def create_abstracted_features(self, price_data: Dict[str, np.ndarray],
                                 sequence_length: int = 30) -> Tuple[np.ndarray, List[str]]:
        """
        Create comprehensive abstracted features for your LNN.
        
        This is the main function that creates all abstraction features.
        """
        print("🧠 Creating abstracted market intelligence features...")
        
        # Create different types of abstracted features
        sector_features = self.create_sector_rotation_features(price_data, window=20)
        regime_features = self.create_market_regime_features(price_data, window=20)
        cross_asset_features = self.create_cross_asset_features(price_data, window=20)
        
        # Combine all abstracted features
        min_length = min(len(sector_features), len(regime_features), len(cross_asset_features))
        
        combined_features = np.concatenate([
            sector_features[:min_length],
            regime_features[:min_length],
            cross_asset_features[:min_length]
        ], axis=1)
        
        # Scale features if not fitted yet
        if not self.is_fitted:
            # Fit scalers on the entire dataset
            n_sector = sector_features.shape[1]
            n_regime = regime_features.shape[1]
            n_cross = cross_asset_features.shape[1]
            
            self.scalers['sector_rotation'].fit(sector_features)
            self.scalers['macro_regime'].fit(regime_features)
            self.scalers['microstructure'].fit(cross_asset_features)
            
            self.is_fitted = True
        
        # Apply scaling
        n_sector = sector_features.shape[1]
        n_regime = regime_features.shape[1]
        
        scaled_sector = self.scalers['sector_rotation'].transform(sector_features[:min_length])
        scaled_regime = self.scalers['macro_regime'].transform(regime_features[:min_length])
        scaled_cross = self.scalers['microstructure'].transform(cross_asset_features[:min_length])
        
        scaled_features = np.concatenate([scaled_sector, scaled_regime, scaled_cross], axis=1)
        
        # Generate feature names
        feature_names = self._generate_feature_names()
        
        print(f"✅ Created {scaled_features.shape[1]} abstracted features over {scaled_features.shape[0]} time steps")
        
        return scaled_features, feature_names
    
    def _generate_feature_names(self) -> List[str]:
        """Generate descriptive names for abstracted features."""
        names = []
        
        # Sector rotation features
        for sector in self.sector_groups.keys():
            names.extend([f'sector_{sector}_return', f'sector_{sector}_volatility'])
        
        # Market regime features
        names.extend([
            'market_volatility_regime',
            'market_trend_strength',
            'market_momentum_change',
            'market_mean_reversion'
        ])
        
        # Cross-asset features
        names.extend([
            'cross_asset_avg_correlation',
            'cross_asset_max_correlation',
            'cross_asset_min_correlation',
            'cross_asset_correlation_dispersion'
        ])
        
        return names


class EnhancedFeatureEngineer(AdvancedFeatureEngineer):
    """
    Enhanced version of your existing feature engineer that includes abstractions.
    This extends your current AdvancedFeatureEngineer class.
    """
    
    def __init__(self, use_abstractions: bool = True):
        super().__init__()
        self.use_abstractions = use_abstractions
        self.abstraction_engine = None
    
    def create_features_with_abstractions(self, 
                                        price_data: Dict[str, np.ndarray],
                                        target_ticker: str,
                                        ohlcv_data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """
        Create features combining your existing approach with abstractions.
        
        Args:
            price_data: Your price data dict (from data_loader.get_closing_prices())
            target_ticker: Your target ticker
            ohlcv_data: Your OHLCV approximation (same as run_analysis.py)
            
        Returns:
            features: Combined feature matrix
            feature_names: List of all feature names
        """
        print("🔧 Creating enhanced features with market abstractions...")
        
        all_features = []
        all_feature_names = []
        
        # 1. Create your existing technical features
        print("   Creating traditional technical features...")
        technical_features, technical_names = self.create_comprehensive_features(
            ohlcv_data, include_advanced=True
        )
        
        all_features.append(technical_features)
        all_feature_names.extend(technical_names)
        
        print(f"   Traditional features: {len(technical_names)}")
        
        # 2. Create abstracted features (if enabled)
        if self.use_abstractions:
            print("   Creating abstracted market intelligence features...")
            
            # Initialize abstraction engine
            all_tickers = list(price_data.keys())
            self.abstraction_engine = MarketAbstractionEngine(all_tickers, target_ticker)
            
            # Create abstracted features
            abstracted_features, abstracted_names = self.abstraction_engine.create_abstracted_features(
                price_data, sequence_length=30
            )
            
            # Align lengths (abstracted features start later due to windowing)
            min_length = min(len(technical_features), len(abstracted_features))
            
            technical_features = technical_features[-min_length:]  # Take last min_length samples
            abstracted_features = abstracted_features[:min_length]  # Take first min_length samples
            
            all_features = [technical_features, abstracted_features]
            all_feature_names.extend(abstracted_names)
            
            print(f"   Abstracted features: {len(abstracted_names)}")
        
        # 3. Combine all features
        if len(all_features) > 1:
            combined_features = np.concatenate(all_features, axis=1)
        else:
            combined_features = all_features[0]
        
        print(f"✅ Total enhanced features: {len(all_feature_names)}")
        print(f"   Feature matrix shape: {combined_features.shape}")
        
        return combined_features, all_feature_names


def integrate_abstractions_with_your_pipeline(config_path="config/config.yaml"):
    """
    Integration function showing how to add abstractions to run_analysis.py
    
    This is what you'd add to your existing run_analysis.py script.
    """
    
    print("🚀 INTEGRATING MARKET ABSTRACTIONS")
    print("=" * 50)
    
    # Load your existing configuration
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 1. Load data using your existing pipeline
    print("📊 Loading data with your existing pipeline...")
    data_loader = StockDataLoader(
        tickers=config['data']['tickers'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date']
    )
    
    raw_data = data_loader.download_data()
    price_data = data_loader.get_closing_prices()
    target_ticker = config['data']['target_ticker']
    
    print(f"   Loaded {len(config['data']['tickers'])} assets")
    print(f"   Target ticker: {target_ticker}")
    
    # 2. Create OHLCV approximation (same as your current approach)
    target_prices = price_data[target_ticker]
    ohlcv_data = {
        'close': target_prices,
        'high': target_prices * 1.02,
        'low': target_prices * 0.98,
        'open': target_prices,
        'volume': np.ones_like(target_prices) * 1000000
    }
    
    # 3. Create enhanced features (traditional + abstracted)
    print("🧠 Creating enhanced features...")
    enhanced_engineer = EnhancedFeatureEngineer(use_abstractions=True)
    
    enhanced_features, enhanced_feature_names = enhanced_engineer.create_features_with_abstractions(
        price_data=price_data,
        target_ticker=target_ticker,
        ohlcv_data=ohlcv_data
    )
    
    # 4. Show the difference
    print("\n📈 FEATURE COMPARISON:")
    print("-" * 30)
    
    # Traditional features only
    traditional_features, traditional_names = enhanced_engineer.create_comprehensive_features(
        ohlcv_data, include_advanced=True
    )
    
    print(f"Traditional approach: {len(traditional_names)} features")
    print(f"Enhanced approach: {len(enhanced_feature_names)} features")
    print(f"Abstraction added: {len(enhanced_feature_names) - len(traditional_names)} features")
    
    # 5. Show abstracted feature categories
    print(f"\n🧠 ABSTRACTED FEATURES ADDED:")
    print("-" * 30)
    abstracted_features = [name for name in enhanced_feature_names if name not in traditional_names]
    
    feature_categories = {
        'Sector Rotation': [f for f in abstracted_features if 'sector_' in f],
        'Market Regime': [f for f in abstracted_features if 'market_' in f],
        'Cross-Asset': [f for f in abstracted_features if 'cross_asset_' in f]
    }
    
    for category, features in feature_categories.items():
        print(f"{category}: {len(features)} features")
        for feature in features[:3]:  # Show first 3
            print(f"  - {feature}")
        if len(features) > 3:
            print(f"  - ... and {len(features)-3} more")
    
    print(f"\n✅ Integration example complete!")
    print(f"💡 To use in your pipeline:")
    print(f"   1. Replace AdvancedFeatureEngineer with EnhancedFeatureEngineer")
    print(f"   2. Use create_features_with_abstractions() instead of create_comprehensive_features()")
    print(f"   3. Your LNN will now have market intelligence features!")
    
    return enhanced_features, enhanced_feature_names


def modify_run_analysis_for_abstractions():
    """
    Shows exactly what to change in your run_analysis.py to add abstractions.
    """
    
    print("📝 MODIFICATIONS FOR run_analysis.py:")
    print("=" * 50)
    
    print("""
# BEFORE (your current approach in run_analysis.py):
from analysis.feature_engineering import AdvancedFeatureEngineer

feature_engineer = AdvancedFeatureEngineer()
features, feature_names = feature_engineer.create_comprehensive_features(
    ohlcv_data, include_advanced=True
)

# AFTER (with abstractions):
from analysis.market_abstraction_pipeline import EnhancedFeatureEngineer

enhanced_engineer = EnhancedFeatureEngineer(use_abstractions=True)
features, feature_names = enhanced_engineer.create_features_with_abstractions(
    price_data=price_data,        # Your existing price_data dict
    target_ticker=target_ticker,  # Your target ticker
    ohlcv_data=ohlcv_data        # Your existing OHLCV approximation
)
""")
    
    print("\n🎯 That's it! The rest of your pipeline stays exactly the same.")
    print("   Your LNN will now train on both technical and abstracted features.")


if __name__ == "__main__":
    # Run the integration example
    try:
        enhanced_features, feature_names = integrate_abstractions_with_your_pipeline()
        print("\n🎉 Market abstraction integration successful!")
        
    except Exception as e:
        print(f"\n❌ Error during integration: {e}")
        print("Make sure you have:")
        print("1. A valid config.yaml file")
        print("2. Internet connection for yfinance data")
        print("3. All required dependencies installed")
