# Jetson Nano Optimized Configuration for Financial LNN
# Run this configuration on your Jetson Nano to avoid memory issues

import numpy as np
import pandas as pd
from enhanced_financial_lnn import *

def run_jetson_optimized_test():
    """Optimized test configuration for Jetson Nano's limited memory"""
    
    print("Running Jetson Nano optimized Financial LNN test...")
    print("=" * 50)
    
    # Reduced dataset size for memory efficiency
    np.random.seed(42)
    n_days = 150  # Reduced from 300
    
    # Generate financial data (same logic but smaller)
    returns = []
    volatility = 0.02
    momentum = 0.0
    
    for i in range(n_days):
        # Simple regime switching every 30 days (reduced complexity)
        if i % 30 == 0:
            regime = np.random.choice([0, 1])  # Only 2 regimes instead of 3
        
        if regime == 0:
            vol_target, momentum_persistence = 0.015, 0.9
        else:
            vol_target, momentum_persistence = 0.030, 0.8
        
        volatility = 0.8 * volatility + 0.2 * vol_target
        momentum = momentum_persistence * momentum + 0.1 * np.random.randn() * 0.01
        daily_return = momentum + volatility * np.random.randn()
        returns.append(daily_return)
    
    prices = 100 * np.cumprod(1 + np.array(returns))
    volumes = 1000000 * (1 + 0.3 * np.random.randn(n_days))
    
    financial_data = pd.DataFrame({
        'Close': prices,
        'Volume': volumes
    })
    
    print(f"Created optimized dataset: {len(financial_data)} observations")
    
    # Create smaller models for Jetson Nano
    comparison_framework = ModelComparisonFramework()
    
    # Smaller LNN configurations
    lnn_small = EnhancedFinancialLNN(
        input_size=4,  # Reduced feature size
        hidden_sizes=[6, 4],  # Smaller hidden layers
        output_size=1,
        adaptive=True,
        name="Jetson_LNN_Adaptive"
    )
    comparison_framework.add_model("Jetson_LNN_Adaptive", lnn_small)
    
    # Static version for comparison
    lnn_static = EnhancedFinancialLNN(
        input_size=4,
        hidden_sizes=[6, 4],
        output_size=1,
        adaptive=False,
        name="Jetson_LNN_Static"
    )
    comparison_framework.add_model("Jetson_LNN_Static", lnn_static)
    
    # Simple baseline models
    def momentum_baseline(price_feat, volume_feat, vol_feat):
        return np.mean(price_feat[-2:]) if len(price_feat) >= 2 else 0.0
    
    comparison_framework.add_model("Simple_Baseline", momentum_baseline)
    
    # Run smaller test
    print("Running optimized side-by-side test...")
    test_results = comparison_framework.run_side_by_side_test(
        financial_data,
        test_window=30,  # Reduced from 80
        prediction_horizon=1
    )
    
    # Display results
    print("\nJetson Nano Test Results:")
    print("-" * 40)
    
    for model_name, results in test_results['detailed_results'].items():
        if 'error' in results:
            print(f"{model_name}: ERROR - {results['error']}")
            continue
        
        print(f"\n{model_name}:")
        print(f"  MSE: {results['mse']:.6f}")
        print(f"  Directional Accuracy: {results['directional_accuracy']:.2%}")
    
    return test_results

def monitor_jetson_performance():
    """Monitor Jetson Nano performance during LNN execution"""
    
    try:
        # Check memory usage
        import psutil
        memory = psutil.virtual_memory()
        print(f"Memory Usage: {memory.percent}% ({memory.used/1e9:.1f}GB used)")
        
        # Check CPU temperature (Jetson specific)
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = int(f.read()) / 1000.0
                print(f"CPU Temperature: {temp:.1f}°C")
        except:
            print("Temperature monitoring not available")
            
    except ImportError:
        print("psutil not available - install with: pip install psutil")

# Memory-efficient feature extractor for Jetson
class JetsonFeatureExtractor:
    """Memory-efficient feature extraction for Jetson Nano"""
    
    @staticmethod
    def extract_basic_features(price_data: pd.DataFrame, window: int = 20) -> Dict[str, np.ndarray]:
        """Extract basic features with minimal memory footprint"""
        
        returns = price_data['Close'].pct_change().fillna(0)
        
        # Basic price features only
        returns_1 = returns.values[-window:]
        returns_5 = price_data['Close'].pct_change(5).fillna(0).values[-window:]
        
        price_features = np.column_stack([returns_1, returns_5])
        
        # Basic volume features
        volume_ratio = (price_data['Volume'] / price_data['Volume'].rolling(window).mean()).fillna(1.0)
        volume_features = volume_ratio.values[-window:].reshape(-1, 1)
        
        # Basic volatility
        volatility = returns.rolling(5).std().fillna(0).values[-window:]
        volatility_features = volatility.reshape(-1, 1)
        
        return {
            'price_features': price_features,
            'volume_features': volume_features,
            'volatility_features': volatility_features
        }

if __name__ == "__main__":
    print("Jetson Nano Optimized Financial LNN")
    print("=" * 40)
    
    # Monitor initial state
    monitor_jetson_performance()
    
    # Run optimized test
    results = run_jetson_optimized_test()
    
    # Monitor final state
    print("\nFinal Performance Check:")
    monitor_jetson_performance()
    
    print("\nOptimization Tips for Jetson Nano:")
    print("1. Use smaller batch sizes")
    print("2. Reduce neuron counts if memory issues persist")
    print("3. Implement gradient checkpointing for larger models")
    print("4. Consider using float16 precision for memory savings")
