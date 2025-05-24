# src/analysis/advanced_forecasting.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AdvancedForecaster:
    """
    Advanced forecasting with confidence intervals and uncertainty quantification.
    Provides 30, 60, 90-day forecasts with confidence bands.
    """
    
    def __init__(self, model, preprocessor, device='cuda'):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.forecast_horizons = [30, 60, 90]
        
    def generate_multi_horizon_forecasts(self, 
                                       last_sequence: np.ndarray,
                                       target_ticker: str,
                                       n_simulations: int = 1000) -> Dict[str, Dict]:
        """
        Generate forecasts for 30, 60, and 90 day horizons with confidence bands.
        
        Args:
            last_sequence: Last sequence of input data for forecasting
            target_ticker: Stock ticker being forecasted
            n_simulations: Number of Monte Carlo simulations for uncertainty
        
        Returns:
            Dictionary with forecasts for each horizon
        """
        print("🔮 Generating multi-horizon forecasts...")
        
        results = {}
        
        for horizon in self.forecast_horizons:
            print(f"Forecasting {horizon} days ahead...")
            
            # Generate point forecasts and confidence intervals
            forecasts = self._forecast_with_uncertainty(
                last_sequence, horizon, n_simulations
            )
            
            # Convert back to original scale
            forecasts_unscaled = self.preprocessor.inverse_transform_single(
                target_ticker, forecasts['predictions']
            )
            
            # Calculate confidence bands
            confidence_bands = self._calculate_confidence_bands(
                forecasts_unscaled, forecasts['uncertainties']
            )
            
            results[f'{horizon}_day'] = {
                'point_forecast': forecasts_unscaled,
                'confidence_bands': confidence_bands,
                'uncertainty_metrics': self._calculate_uncertainty_metrics(forecasts),
                'forecast_dates': self._generate_forecast_dates(horizon)
            }
            
        return results
    
    def _forecast_with_uncertainty(self, 
                                 last_sequence: np.ndarray, 
                                 horizon: int, 
                                 n_simulations: int) -> Dict:
        """Generate forecasts with uncertainty quantification using Monte Carlo."""
        
        self.model.eval()
        predictions = []
        uncertainties = []
        
        # Enable dropout during inference for uncertainty estimation
        def enable_dropout(m):
            if type(m) == nn.Dropout:
                m.train()
        
        with torch.no_grad():
            current_sequence = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            for day in range(horizon):
                day_predictions = []
                
                # Monte Carlo sampling
                for _ in range(n_simulations):
                    # Enable dropout for uncertainty
                    self.model.apply(enable_dropout)
                    
                    # Get prediction
                    pred = self.model(current_sequence)
                    day_predictions.append(pred.cpu().numpy())
                
                # Calculate statistics
                day_predictions = np.array(day_predictions).squeeze()
                mean_pred = np.mean(day_predictions)
                std_pred = np.std(day_predictions)
                
                predictions.append(mean_pred)
                uncertainties.append(std_pred)
                
                # Update sequence for next prediction (rolling forecast)
                if day < horizon - 1:
                    # Add prediction to sequence and remove oldest value
                    new_input = torch.tensor([[mean_pred]], dtype=torch.float32).to(self.device)
                    current_sequence = torch.cat([
                        current_sequence[:, 1:, :],  # Remove first time step
                        new_input.unsqueeze(1)      # Add new prediction
                    ], dim=1)
        
        return {
            'predictions': np.array(predictions).reshape(-1, 1),
            'uncertainties': np.array(uncertainties)
        }
    
    def _calculate_confidence_bands(self, 
                                  forecasts: np.ndarray, 
                                  uncertainties: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate confidence intervals for forecasts."""
        
        confidence_levels = [0.68, 0.95, 0.99]  # 1σ, 2σ, 3σ
        bands = {}
        
        for conf_level in confidence_levels:
            z_score = stats.norm.ppf((1 + conf_level) / 2)
            
            lower_bound = forecasts.flatten() - z_score * uncertainties
            upper_bound = forecasts.flatten() + z_score * uncertainties
            
            bands[f'{int(conf_level*100)}%'] = {
                'lower': lower_bound,
                'upper': upper_bound
            }
        
        return bands
    
    def _calculate_uncertainty_metrics(self, forecasts: Dict) -> Dict:
        """Calculate various uncertainty metrics."""
        uncertainties = forecasts['uncertainties']
        
        return {
            'mean_uncertainty': float(np.mean(uncertainties)),
            'max_uncertainty': float(np.max(uncertainties)),
            'uncertainty_trend': float(np.polyfit(range(len(uncertainties)), uncertainties, 1)[0]),
            'forecast_reliability': float(1 / (1 + np.mean(uncertainties)))  # Higher = more reliable
        }
    
    def _generate_forecast_dates(self, horizon: int) -> List[str]:
        """Generate future dates for forecasts."""
        from datetime import datetime, timedelta
        
        start_date = datetime.now()
        dates = []
        
        for i in range(1, horizon + 1):
            future_date = start_date + timedelta(days=i)
            # Skip weekends for trading days
            while future_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                future_date += timedelta(days=1)
            dates.append(future_date.strftime('%Y-%m-%d'))
        
        return dates
    
    def create_forecast_visualization(self, forecasts: Dict, actual_prices: np.ndarray) -> Dict:
        """Create visualization data for forecasts."""
        
        viz_data = {
            'historical': actual_prices[-60:].flatten().tolist(),  # Last 60 days
            'forecasts': {}
        }
        
        for horizon_name, horizon_data in forecasts.items():
            viz_data['forecasts'][horizon_name] = {
                'point_forecast': horizon_data['point_forecast'].flatten().tolist(),
                'confidence_68': horizon_data['confidence_bands']['68%'],
                'confidence_95': horizon_data['confidence_bands']['95%'],
                'dates': horizon_data['forecast_dates'],
                'uncertainty_metrics': horizon_data['uncertainty_metrics']
            }
        
        return viz_data

class OptionsAnalyzer:
    """
    Analyze option pricing vs model forecasts to identify mispricings.
    """
    
    def __init__(self):
        self.risk_free_rate = 0.05  # Approximate current risk-free rate
        
    def fetch_options_data(self, ticker: str) -> Dict:
        """
        Fetch current options data for 1-2 week expirations.
        """
        try:
            import yfinance as yf
            
            stock = yf.Ticker(ticker)
            options_dates = stock.options
            
            if not options_dates:
                return None
            
            # Get options for next 1-2 weeks
            short_term_options = []
            for date in options_dates[:4]:  # First few expiration dates
                try:
                    calls = stock.option_chain(date).calls
                    puts = stock.option_chain(date).puts
                    
                    short_term_options.append({
                        'expiration': date,
                        'calls': calls,
                        'puts': puts,
                        'days_to_expiry': self._calculate_days_to_expiry(date)
                    })
                except:
                    continue
            
            return {
                'current_price': stock.history(period='1d')['Close'].iloc[-1],
                'options': short_term_options
            }
            
        except Exception as e:
            print(f"Error fetching options data: {e}")
            return None
    
    def calculate_implied_volatility_surface(self, options_data: Dict) -> Dict:
        """Calculate implied volatility from options prices."""
        
        iv_surface = {}
        
        for option_set in options_data['options']:
            expiry = option_set['expiration']
            days_to_expiry = option_set['days_to_expiry']
            
            if days_to_expiry <= 14:  # Focus on 1-2 week options
                calls_iv = self._extract_implied_volatility(
                    option_set['calls'], 
                    options_data['current_price'],
                    days_to_expiry,
                    'call'
                )
                
                puts_iv = self._extract_implied_volatility(
                    option_set['puts'],
                    options_data['current_price'], 
                    days_to_expiry,
                    'put'
                )
                
                iv_surface[expiry] = {
                    'days_to_expiry': days_to_expiry,
                    'calls_iv': calls_iv,
                    'puts_iv': puts_iv,
                    'atm_iv': self._calculate_atm_iv(calls_iv, puts_iv, options_data['current_price'])
                }
        
        return iv_surface
    
    def identify_forecast_option_mismatches(self, 
                                          forecasts: Dict, 
                                          options_data: Dict,
                                          current_price: float) -> Dict:
        """
        Compare model forecasts with options-implied expectations.
        """
        
        if not options_data:
            return {'error': 'No options data available'}
        
        iv_surface = self.calculate_implied_volatility_surface(options_data)
        mismatches = {}
        
        # Compare 7-day and 14-day forecasts with corresponding options
        for expiry, iv_data in iv_surface.items():
            days_to_expiry = iv_data['days_to_expiry']
            
            if days_to_expiry <= 14:
                # Get corresponding forecast
                if days_to_expiry <= 7:
                    forecast_horizon = 7
                else:
                    forecast_horizon = 14
                
                # Calculate model-implied volatility
                model_volatility = self._calculate_model_implied_volatility(
                    forecasts, forecast_horizon, current_price
                )
                
                # Compare with options IV
                atm_iv = iv_data['atm_iv']
                
                if model_volatility and atm_iv:
                    volatility_diff = model_volatility - atm_iv
                    
                    mismatches[expiry] = {
                        'days_to_expiry': days_to_expiry,
                        'model_implied_vol': model_volatility,
                        'options_implied_vol': atm_iv,
                        'volatility_difference': volatility_diff,
                        'percentage_difference': (volatility_diff / atm_iv) * 100,
                        'trading_signal': self._generate_trading_signal(volatility_diff)
                    }
        
        return mismatches
    
    def _extract_implied_volatility(self, options_df, current_price, days_to_expiry, option_type):
        """Extract implied volatility from options dataframe."""
        try:
            # Filter for near-the-money options
            atm_range = current_price * 0.05  # 5% around current price
            near_atm = options_df[
                (options_df['strike'] >= current_price - atm_range) &
                (options_df['strike'] <= current_price + atm_range)
            ]
            
            if not near_atm.empty and 'impliedVolatility' in near_atm.columns:
                return near_atm['impliedVolatility'].mean()
            
        except Exception as e:
            print(f"Error extracting IV: {e}")
        
        return None
    
    def _calculate_atm_iv(self, calls_iv, puts_iv, current_price):
        """Calculate at-the-money implied volatility."""
        if calls_iv and puts_iv:
            return (calls_iv + puts_iv) / 2
        elif calls_iv:
            return calls_iv
        elif puts_iv:
            return puts_iv
        return None
    
    def _calculate_model_implied_volatility(self, forecasts, horizon, current_price):
        """Calculate implied volatility from model forecasts."""
        try:
            if horizon <= 30:
                forecast_data = forecasts['30_day']
            elif horizon <= 60:
                forecast_data = forecasts['60_day']
            else:
                forecast_data = forecasts['90_day']
            
            # Take forecasts up to the horizon
            relevant_forecasts = forecast_data['point_forecast'][:horizon]
            
            if len(relevant_forecasts) > 0:
                # Calculate realized volatility from forecasts
                returns = np.diff(relevant_forecasts.flatten()) / relevant_forecasts.flatten()[:-1]
                return np.std(returns) * np.sqrt(252)  # Annualized volatility
            
        except Exception as e:
            print(f"Error calculating model IV: {e}")
        
        return None
    
    def _calculate_days_to_expiry(self, expiry_date):
        """Calculate days to option expiry."""
        from datetime import datetime
        try:
            expiry = datetime.strptime(expiry_date, '%Y-%m-%d')
            today = datetime.now()
            return (expiry - today).days
        except:
            return None
    
    def _generate_trading_signal(self, volatility_diff):
        """Generate trading signal based on volatility mismatch."""
        if abs(volatility_diff) < 0.05:  # Less than 5% difference
            return "NEUTRAL"
        elif volatility_diff > 0.1:  # Model predicts higher vol than options
            return "BUY_VOLATILITY"  # Buy straddles/strangles
        elif volatility_diff < -0.1:  # Model predicts lower vol than options
            return "SELL_VOLATILITY"  # Sell straddles/strangles
        else:
            return "WEAK_SIGNAL"
