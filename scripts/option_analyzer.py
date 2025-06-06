#!/usr/bin/env python3
"""
Option Strategy Analyzer for LNN Models
Generates specific option recommendations based on model forecasts and real option chains.

Usage:
    python scripts/option_analyzer.py --model models/qualified_models/LOW_champion_model.pth --ticker LOW
    python scripts/option_analyzer.py --batch --model-dir models/qualified_models/
    python scripts/option_analyzer.py --model LOW_model.pth --confidence-threshold 0.85
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.file_naming import file_namer
from data.data_loader import StockDataLoader
from models.lnn_model import LiquidNetwork
from analysis.market_abstraction_pipeline import EnhancedFeatureEngineer
from sklearn.preprocessing import MinMaxScaler

class OptionStrategyAnalyzer:
    """
    Advanced option strategy analyzer that combines ML forecasts with real option chains
    to generate specific trading recommendations.
    """
    
    def __init__(self, confidence_threshold: float = 0.8):
        """
        Initialize the option analyzer.
        
        Args:
            confidence_threshold: Minimum confidence level for recommendations (default 80%)
        """
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Forecast periods in trading days
        self.forecast_periods = {
            '2w': 10,   # 2 weeks = 10 trading days
            '4w': 20,   # 4 weeks = 20 trading days  
            '3m': 63    # 3 months = ~63 trading days
        }
        
        print(f"🎯 Option Strategy Analyzer initialized")
        print(f"   Device: {self.device}")
        print(f"   Confidence threshold: {confidence_threshold:.1%}")
        
    def load_model_and_generate_forecasts(self, model_path: str, ticker: str) -> Dict:
        """Load model and generate price forecasts with confidence bounds."""
        
        print(f"\n📊 Generating forecasts for {ticker}")
        print(f"   Model: {os.path.basename(model_path)}")
        
        try:
            # Load model
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            config = checkpoint.get('config', {})
            
            # Get model architecture
            saved_weights = checkpoint['model_state_dict']['liquid_cell.input_weights']
            input_size = saved_weights.shape[0]
            hidden_size = saved_weights.shape[1]
            
            model = LiquidNetwork(input_size=input_size, hidden_size=hidden_size, output_size=1).to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Prepare recent data for forecasting
            forecast_data = self.prepare_forecast_data(ticker, config)
            if forecast_data is None:
                return None
            
            # Generate forecasts
            forecasts = self.generate_multi_period_forecasts(model, forecast_data, ticker)
            
            return forecasts
            
        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
            return None
    
    def prepare_forecast_data(self, ticker: str, config: Dict, lookback_days: int = 200) -> Dict:
        """Prepare recent data for forecasting."""
        
        try:
            # Get recent data for forecasting
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            
            # Use same tickers as training
            data_config = config.get('data', {})
            training_tickers = data_config.get('tickers', ['^GSPC', 'QQQ', ticker])
            if ticker not in training_tickers:
                training_tickers.append(ticker)
            
            print(f"   Loading recent data: {start_date} to {end_date}")
            
            # Load data
            data_loader = StockDataLoader(training_tickers, start_date, end_date)
            raw_data = data_loader.download_data()
            price_data = data_loader.get_closing_prices()
            
            target_prices = price_data.get(ticker)
            if target_prices is None or len(target_prices) < 100:
                print(f"   ⚠️ Insufficient recent data for {ticker}")
                return None
            
            # Create enhanced features (same as training)
            enhanced_engineer = EnhancedFeatureEngineer(use_abstractions=True)
            
            ohlcv_data = {
                'close': target_prices,
                'high': target_prices * 1.02,
                'low': target_prices * 0.98,
                'open': target_prices,
                'volume': np.ones_like(target_prices) * 1000000
            }
            
            features, feature_names = enhanced_engineer.create_features_with_abstractions(
                price_data=price_data,
                target_ticker=ticker,
                ohlcv_data=ohlcv_data
            )
            
            # Scale features
            scaler = MinMaxScaler(feature_range=(-1, 1))
            features_scaled = scaler.fit_transform(features)
            
            # Get current price
            current_price = float(target_prices.flatten()[-1])
            
            return {
                'features': features_scaled,
                'current_price': current_price,
                'scaler': scaler,
                'raw_data': raw_data,
                'sequence_length': config.get('model', {}).get('sequence_length', 30),
                'ticker': ticker
            }
            
        except Exception as e:
            print(f"   ❌ Error preparing forecast data: {e}")
            return None
    
    def generate_multi_period_forecasts(self, model: torch.nn.Module, 
                                      forecast_data: Dict, ticker: str) -> Dict:
        """Generate forecasts for multiple time periods with confidence bounds."""
        
        forecasts = {}
        current_price = forecast_data['current_price']
        features = forecast_data['features']
        sequence_length = forecast_data['sequence_length']
        
        print(f"   Current price: ${current_price:.2f}")
        
        for period_name, trading_days in self.forecast_periods.items():
            print(f"   Generating {period_name} forecast ({trading_days} trading days)...")
            
            try:
                # Monte Carlo simulation for confidence bounds
                n_simulations = 100
                period_predictions = []
                
                for sim in range(n_simulations):
                    # Start with recent features
                    current_sequence = features[-sequence_length:].copy()
                    simulated_price = current_price
                    
                    # Simulate forward for the period
                    for day in range(trading_days):
                        # Prepare input
                        input_tensor = torch.tensor(current_sequence.reshape(1, sequence_length, -1), 
                                                   dtype=torch.float32).to(self.device)
                        
                        # Add small noise for Monte Carlo variation
                        if sim > 0:
                            noise = torch.randn_like(input_tensor) * 0.005
                            input_tensor = input_tensor + noise
                        
                        # Generate prediction
                        with torch.no_grad():
                            predicted_return = model(input_tensor).cpu().numpy()[0, 0]
                        
                        # Update simulated price
                        simulated_price *= (1 + predicted_return)
                        
                        # Update sequence (simplified - use last known features with small variation)
                        if day < trading_days - 1:  # Don't update on last day
                            # Shift sequence and add new "features" (simplified)
                            current_sequence = np.roll(current_sequence, -1, axis=0)
                            # Add small random variation to simulate market evolution
                            current_sequence[-1] = current_sequence[-2] + np.random.normal(0, 0.01, size=current_sequence.shape[1])
                    
                    period_predictions.append(simulated_price)
                
                # Calculate statistics
                predictions_array = np.array(period_predictions)
                
                mean_price = np.mean(predictions_array)
                std_price = np.std(predictions_array)
                
                # Confidence bounds
                confidence_80_lower = np.percentile(predictions_array, 10)  # 80% confidence = 10th to 90th percentile
                confidence_80_upper = np.percentile(predictions_array, 90)
                confidence_95_lower = np.percentile(predictions_array, 2.5)
                confidence_95_upper = np.percentile(predictions_array, 97.5)
                
                # Expected return
                expected_return = (mean_price - current_price) / current_price
                
                forecasts[period_name] = {
                    'trading_days': trading_days,
                    'current_price': current_price,
                    'expected_price': mean_price,
                    'expected_return': expected_return,
                    'price_std': std_price,
                    'confidence_80_lower': confidence_80_lower,
                    'confidence_80_upper': confidence_80_upper,
                    'confidence_95_lower': confidence_95_lower,
                    'confidence_95_upper': confidence_95_upper,
                    'all_simulations': predictions_array.tolist()
                }
                
                print(f"     Expected price: ${mean_price:.2f} ({expected_return:+.1%})")
                print(f"     80% confidence: ${confidence_80_lower:.2f} - ${confidence_80_upper:.2f}")
                
            except Exception as e:
                print(f"   ❌ Error generating {period_name} forecast: {e}")
                continue
        
        return forecasts
    
    def fetch_option_chain(self, ticker: str, period: str) -> Dict:
        """Fetch real option chain data for the ticker and period."""
        
        print(f"   📋 Fetching option chain for {ticker} ({period})...")
        
        try:
            stock = yf.Ticker(ticker)
            
            # Get option expiration dates
            expiration_dates = stock.options
            
            if not expiration_dates:
                print(f"   ⚠️ No option expiration dates found for {ticker}")
                return None
            
            # Find appropriate expiration for the period
            target_days = self.forecast_periods[period]
            target_date = datetime.now() + timedelta(days=target_days)
            
            # Find closest expiration date
            best_expiration = None
            min_date_diff = float('inf')
            
            for exp_date_str in expiration_dates:
                exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d')
                date_diff = abs((exp_date - target_date).days)
                
                if date_diff < min_date_diff:
                    min_date_diff = date_diff
                    best_expiration = exp_date_str
            
            if best_expiration is None:
                print(f"   ⚠️ No suitable expiration found for {period}")
                return None
            
            print(f"   Using expiration: {best_expiration} ({min_date_diff} days from target)")
            
            # Get option chain for this expiration
            option_chain = stock.option_chain(best_expiration)
            
            calls_df = option_chain.calls
            puts_df = option_chain.puts
            
            # Clean and process the data
            calls_df = calls_df.dropna(subset=['strike', 'lastPrice', 'impliedVolatility'])
            puts_df = puts_df.dropna(subset=['strike', 'lastPrice', 'impliedVolatility'])
            
            print(f"   ✅ Found {len(calls_df)} calls and {len(puts_df)} puts")
            
            return {
                'expiration_date': best_expiration,
                'days_to_expiry': min_date_diff,
                'calls': calls_df,
                'puts': puts_df
            }
            
        except Exception as e:
            print(f"   ❌ Error fetching option chain: {e}")
            return None
    
    def analyze_option_opportunities(self, ticker: str, forecasts: Dict) -> Dict:
        """Analyze option opportunities for all periods."""
        
        print(f"\n🎯 ANALYZING OPTION OPPORTUNITIES FOR {ticker}")
        print("="*60)
        
        recommendations = {
            'ticker': ticker,
            'analysis_date': datetime.now().isoformat(),
            'current_price': forecasts[list(forecasts.keys())[0]]['current_price'],
            'periods': {}
        }
        
        for period, forecast in forecasts.items():
            print(f"\n📊 {period.upper()} ANALYSIS:")
            print("-" * 30)
            
            # Fetch option chain
            option_chain = self.fetch_option_chain(ticker, period)
            if option_chain is None:
                continue
            
            # Analyze calls and puts
            call_opportunities = self.find_call_opportunities(forecast, option_chain)
            put_opportunities = self.find_put_opportunities(forecast, option_chain)
            
            recommendations['periods'][period] = {
                'forecast': forecast,
                'option_chain': {
                    'expiration_date': option_chain['expiration_date'],
                    'days_to_expiry': option_chain['days_to_expiry']
                },
                'call_opportunities': call_opportunities,
                'put_opportunities': put_opportunities
            }
        
        return recommendations
    
    def find_call_opportunities(self, forecast: Dict, option_chain: Dict) -> List[Dict]:
        """Find profitable call option opportunities."""
        
        current_price = forecast['current_price']
        confidence_80_lower = forecast['confidence_80_lower']
        expected_price = forecast['expected_price']
        
        calls_df = option_chain['calls']
        opportunities = []
        
        print(f"   🔍 Scanning {len(calls_df)} call options...")
        
        for _, call in calls_df.iterrows():
            strike = call['strike']
            option_price = call['lastPrice']
            
            # Skip if no valid data
            if pd.isna(strike) or pd.isna(option_price) or option_price <= 0:
                continue
            
            # Calculate breakeven price (strike + premium)
            breakeven_price = strike + option_price
            
            # Check if our 80% confidence lower bound exceeds breakeven
            if confidence_80_lower > breakeven_price:
                # Calculate potential profit
                expected_value = max(0, expected_price - strike) - option_price
                confidence_80_value = max(0, confidence_80_lower - strike) - option_price
                
                # Calculate probability of profit (simplified)
                prob_profitable = np.mean(np.array(forecast['all_simulations']) > breakeven_price)
                
                opportunity = {
                    'type': 'CALL',
                    'strike': strike,
                    'option_price': option_price,
                    'breakeven_price': breakeven_price,
                    'expected_value': expected_value,
                    'confidence_80_value': confidence_80_value,
                    'probability_profitable': prob_profitable,
                    'implied_volatility': call.get('impliedVolatility', 0),
                    'volume': call.get('volume', 0),
                    'open_interest': call.get('openInterest', 0),
                    'safety_margin': confidence_80_lower - breakeven_price,
                    'max_loss': option_price,
                    'potential_return': expected_value / option_price if option_price > 0 else 0
                }
                
                opportunities.append(opportunity)
        
        # Sort by safety margin (descending)
        opportunities.sort(key=lambda x: x['safety_margin'], reverse=True)
        
        print(f"   ✅ Found {len(opportunities)} call opportunities")
        
        return opportunities[:5]  # Return top 5
    
    def find_put_opportunities(self, forecast: Dict, option_chain: Dict) -> List[Dict]:
        """Find profitable put option opportunities (for bearish forecasts)."""
        
        current_price = forecast['current_price']
        confidence_80_upper = forecast['confidence_80_upper']
        expected_price = forecast['expected_price']
        
        puts_df = option_chain['puts']
        opportunities = []
        
        # Only look for put opportunities if we expect downward movement
        if expected_price < current_price * 0.95:  # Only if expecting >5% decline
            
            print(f"   🔍 Scanning {len(puts_df)} put options...")
            
            for _, put in puts_df.iterrows():
                strike = put['strike']
                option_price = put['lastPrice']
                
                # Skip if no valid data
                if pd.isna(strike) or pd.isna(option_price) or option_price <= 0:
                    continue
                
                # Calculate breakeven price (strike - premium)
                breakeven_price = strike - option_price
                
                # Check if our 80% confidence upper bound is below breakeven
                if confidence_80_upper < breakeven_price:
                    # Calculate potential profit
                    expected_value = max(0, strike - expected_price) - option_price
                    confidence_80_value = max(0, strike - confidence_80_upper) - option_price
                    
                    # Calculate probability of profit
                    prob_profitable = np.mean(np.array(forecast['all_simulations']) < breakeven_price)
                    
                    opportunity = {
                        'type': 'PUT',
                        'strike': strike,
                        'option_price': option_price,
                        'breakeven_price': breakeven_price,
                        'expected_value': expected_value,
                        'confidence_80_value': confidence_80_value,
                        'probability_profitable': prob_profitable,
                        'implied_volatility': put.get('impliedVolatility', 0),
                        'volume': put.get('volume', 0),
                        'open_interest': put.get('openInterest', 0),
                        'safety_margin': breakeven_price - confidence_80_upper,
                        'max_loss': option_price,
                        'potential_return': expected_value / option_price if option_price > 0 else 0
                    }
                    
                    opportunities.append(opportunity)
            
            # Sort by safety margin (descending)
            opportunities.sort(key=lambda x: x['safety_margin'], reverse=True)
            
            print(f"   ✅ Found {len(opportunities)} put opportunities")
        
        else:
            print(f"   📈 Bullish forecast - skipping put analysis")
        
        return opportunities[:5]  # Return top 5
    
    def generate_recommendation_report(self, recommendations: Dict) -> str:
        """Generate a comprehensive recommendation report."""
        
        ticker = recommendations['ticker']
        current_price = recommendations['current_price']
        
        report = []
        report.append("="*80)
        report.append("OPTION STRATEGY RECOMMENDATIONS")
        report.append("="*80)
        report.append(f"Ticker: {ticker}")
        report.append(f"Current Price: ${current_price:.2f}")
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Confidence Threshold: {self.confidence_threshold:.0%}")
        report.append("")
        
        for period, analysis in recommendations['periods'].items():
            forecast = analysis['forecast']
            call_ops = analysis['call_opportunities']
            put_ops = analysis['put_opportunities']
            
            report.append(f"{period.upper()} RECOMMENDATIONS:")
            report.append("-" * 40)
            report.append(f"Expected Price: ${forecast['expected_price']:.2f} ({forecast['expected_return']:+.1%})")
            report.append(f"80% Confidence Range: ${forecast['confidence_80_lower']:.2f} - ${forecast['confidence_80_upper']:.2f}")
            report.append(f"Expiration: {analysis['option_chain']['expiration_date']}")
            report.append("")
            
            if call_ops:
                report.append("🚀 TOP CALL OPPORTUNITIES:")
                for i, opp in enumerate(call_ops[:3], 1):
                    report.append(f"  {i}. ${opp['strike']:.2f} Call @ ${opp['option_price']:.2f}")
                    report.append(f"     Breakeven: ${opp['breakeven_price']:.2f}")
                    report.append(f"     Safety Margin: ${opp['safety_margin']:.2f}")
                    report.append(f"     Expected Return: {opp['potential_return']:+.1%}")
                    report.append(f"     Probability Profitable: {opp['probability_profitable']:.1%}")
                    report.append("")
            
            if put_ops:
                report.append("📉 TOP PUT OPPORTUNITIES:")
                for i, opp in enumerate(put_ops[:3], 1):
                    report.append(f"  {i}. ${opp['strike']:.2f} Put @ ${opp['option_price']:.2f}")
                    report.append(f"     Breakeven: ${opp['breakeven_price']:.2f}")
                    report.append(f"     Safety Margin: ${opp['safety_margin']:.2f}")
                    report.append(f"     Expected Return: {opp['potential_return']:+.1%}")
                    report.append(f"     Probability Profitable: {opp['probability_profitable']:.1%}")
                    report.append("")
            
            if not call_ops and not put_ops:
                report.append("⚠️ No qualifying opportunities found for this period")
                report.append("")
        
        report.append("="*80)
        
        return "\n".join(report)
    
    def save_results(self, recommendations: Dict, output_path: str = None):
        ticker = recommendations['ticker']
    
        # Create standardized option analysis paths
        option_paths = file_namer.create_options_paths(ticker)
    
        # Save main recommendations
        if output_path is None:
            output_path = option_paths['recommendations']['options_json']
    
        with open(output_path, 'w') as f:
            json.dump(recommendations, f, indent=2, default=str)
    
        # Save human-readable report
        report = self.generate_recommendation_report(recommendations)
        with open(option_paths['recommendations']['recommendations_txt'], 'w') as f:
            f.write(report)
    
        # Save individual forecasts
        for period, analysis in recommendations['periods'].items():
            forecast_path = option_paths['forecasts'][period]
            forecast_data = {
                'ticker': ticker,
                'period': period,
                'forecast': analysis['forecast'],
                'timestamp': datetime.now().isoformat()
            }
        
            with open(forecast_path, 'w') as f:
                json.dump(forecast_data, f, indent=2, default=str)
    
        print(f"Option analysis saved to {output_path}")
        print(f"\n💾Forecasts saved to: {list(option_paths['forecasts'].values())}")

def main():
    """Main function for option strategy analysis."""
    parser = argparse.ArgumentParser(description='Option Strategy Analyzer for LNN Models')
    
    # Model specification
    parser.add_argument('--model', type=str,
                      help='Path to specific model file')
    parser.add_argument('--ticker', type=str,
                      help='Stock ticker to analyze')
    parser.add_argument('--batch', action='store_true',
                      help='Analyze all models in directory')
    parser.add_argument('--model-dir', type=str, default='models/qualified_models',
                      help='Directory containing qualified models')
    
    # Analysis parameters
    parser.add_argument('--confidence-threshold', type=float, default=0.8,
                      help='Confidence threshold for recommendations (default: 0.8)')
    parser.add_argument('--output', type=str,
                      help='Output file path')
    
    args = parser.parse_args()
    
    print("🎯 LNN OPTION STRATEGY ANALYZER")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    analyzer = OptionStrategyAnalyzer(confidence_threshold=args.confidence_threshold)
    
    try:
        if args.batch:
            # Batch analysis
            if not os.path.exists(args.model_dir):
                print(f"❌ Model directory not found: {args.model_dir}")
                return
            
            model_files = [f for f in os.listdir(args.model_dir) if f.endswith('.pth')]
            print(f"\n🔍 Found {len(model_files)} models to analyze")
            
            for model_file in model_files:
                model_path = os.path.join(args.model_dir, model_file)
                
                # Extract ticker from filename
                ticker = model_file.split('_')[0].upper()
                
                print(f"\n" + "="*60)
                print(f"📊 ANALYZING: {model_file} ({ticker})")
                print("="*60)
                
                # Generate forecasts
                forecasts = analyzer.load_model_and_generate_forecasts(model_path, ticker)
                if forecasts is None:
                    continue
                
                # Analyze options
                recommendations = analyzer.analyze_option_opportunities(ticker, forecasts)
                
                # Generate and print report
                report = analyzer.generate_recommendation_report(recommendations)
                print(report)
                
                # Save results
                analyzer.save_results(recommendations)
        
        elif args.model and args.ticker:
            # Single model analysis
            ticker = args.ticker.upper()
            
            print(f"\n🔍 Analyzing model: {args.model}")
            print(f"   Ticker: {ticker}")
            
            # Generate forecasts
            forecasts = analyzer.load_model_and_generate_forecasts(args.model, ticker)
            if forecasts is None:
                print("❌ Failed to generate forecasts")
                return
            
            # Analyze options
            recommendations = analyzer.analyze_option_opportunities(ticker, forecasts)
            
            # Generate and print report
            report = analyzer.generate_recommendation_report(recommendations)
            print(report)
            
            # Save results
            analyzer.save_results(recommendations, args.output)
        
        else:
            print("❌ Must specify either --model and --ticker, or --batch")
            return
        
        print(f"\n✅ OPTION ANALYSIS COMPLETED!")
        
    except Exception as e:
        print(f"\n❌ ERROR during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
