#!/usr/bin/env python3
"""
Test script to verify all imports work correctly.
Run this before your first analysis to catch any setup issues.

Usage:
    python tests/test_imports.py
"""

import sys
import os
import traceback

# Add project root to path (not just src)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_core_imports():
    """Test core module imports."""
    print("Testing core imports...")
    
    try:
        # Test data modules - CORRECTED PATHS
        from data.data_loader import StockDataLoader
        from data.preprocessor import StockDataPreprocessor
        print("✓ Data modules imported successfully")
        
        # Test model modules - CORRECTED PATHS
        from models.lnn_model import LiquidNetwork, ModelConfig
        print("✓ Model modules imported successfully")
        
        # Test analysis modules - CORRECTED PATHS
        from analysis.pattern_recognition import PatternRecognizer
        from analysis.feature_engineering import AdvancedFeatureEngineer
        from analysis.dimensionality_reduction import DimensionalityReducer
        from analysis.temporal_analysis import TemporalAnalyzer
        print("✓ Analysis modules imported successfully")
        
        # Test utility modules - CORRECTED PATHS
        from utils.metrics import StockPredictionMetrics
        from utils.experiment_tracker import ExperimentTracker
        print("✓ Utility modules imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_dependencies():
    """Test required dependencies."""
    print("\nTesting dependencies...")
    
    required_packages = [
        'torch',
        'numpy', 
        'pandas',
        'sklearn',
        'yfinance',
        'matplotlib',
        'scipy',
        'yaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'yaml':
                import yaml
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def test_torch_cuda():
    """Test PyTorch and CUDA availability."""
    print("\nTesting PyTorch and CUDA...")
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA version: {torch.version.cuda}")
        else:
            print("⚠️  CUDA not available - will use CPU")
        
        # Test basic tensor operations
        x = torch.randn(10, 10)
        y = torch.mm(x, x.t())
        print("✓ Basic tensor operations working")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch error: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        import yaml
        
        # Check if config file exists
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print("✓ Configuration file loaded successfully")
            print(f"✓ Target ticker: {config.get('data', {}).get('target_ticker', 'Not specified')}")
        else:
            print("⚠️  No config file found - will use defaults")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_data_access():
    """Test data access (quick download test)."""
    print("\nTesting data access...")
    
    try:
        import yfinance as yf
        
        # Quick test download
        print("Testing yfinance connection...")
        ticker = yf.Ticker("AAPL")
        info = ticker.info
        
        if info and 'symbol' in info:
            print("✓ Yahoo Finance connection working")
            return True
        else:
            print("⚠️  Yahoo Finance connection may have issues")
            return False
            
    except Exception as e:
        print(f"❌ Data access error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("STOCK LNN SYSTEM - IMPORT AND DEPENDENCY TEST")
    print("=" * 60)
    
    tests = [
        ("Core Imports", test_core_imports),
        ("Dependencies", test_dependencies), 
        ("PyTorch/CUDA", test_torch_cuda),
        ("Configuration", test_configuration),
        ("Data Access", test_data_access)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("\n🎉 ALL TESTS PASSED! Your system is ready for analysis.")
        return True
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Please fix issues before running analysis.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
