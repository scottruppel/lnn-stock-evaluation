#!/usr/bin/env python3
"""
DEBUG VERSION - Multi-Stock Analysis with Enhanced Feedback
This version provides immediate feedback to help diagnose issues.
"""

import os
import sys
import time
print("=== DEBUG: Script starting ===")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Check imports immediately
print("\n=== DEBUG: Testing imports ===")
try:
    import yaml
    print("✓ yaml imported successfully")
except ImportError as e:
    print(f"❌ yaml import failed: {e}")

try:
    import numpy as np
    print("✓ numpy imported successfully")
except ImportError as e:
    print(f"❌ numpy import failed: {e}")
    np = None

try:
    import pandas as pd
    print("✓ pandas imported successfully")
except ImportError as e:
    print(f"❌ pandas import failed: {e}")

try:
    import torch
    print("✓ torch imported successfully")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU device: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"❌ torch import failed: {e}")

print("\n=== DEBUG: Checking project structure ===")
required_dirs = ['src', 'config', 'scripts', 'results']
for directory in required_dirs:
    if os.path.exists(directory):
        print(f"✓ {directory}/ exists")
    else:
        print(f"❌ {directory}/ missing")

# Check if we can import project modules
print("\n=== DEBUG: Testing project imports ===")
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from data.data_loader import StockDataLoader
    print("✓ StockDataLoader imported successfully")
except ImportError as e:
    print(f"❌ StockDataLoader import failed: {e}")

try:
    from run_analysis import ComprehensiveAnalyzer
    print("✓ ComprehensiveAnalyzer imported successfully")
except ImportError as e:
    print(f"❌ ComprehensiveAnalyzer import failed: {e}")

print("\n=== DEBUG: Testing config loading ===")
config_path = "config/config.yaml"
if os.path.exists(config_path):
    print(f"✓ Config file exists: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("✓ Config file loaded successfully")
        print(f"  Config keys: {list(config.keys())}")
        
        if 'multi_stock_analysis' in config:
            print("✓ Multi-stock analysis config found")
            msa_config = config['multi_stock_analysis']
            print(f"  Active universe: {msa_config.get('active_universe', 'not set')}")
            print(f"  Active grid: {msa_config.get('active_grid', 'not set')}")
        else:
            print("❌ Multi-stock analysis config missing")
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
else:
    print(f"❌ Config file missing: {config_path}")

print("\n=== DEBUG: Basic functionality test ===")

class DebugMultiStockAnalyzer:
    """Minimal version for testing"""
    
    def __init__(self, config_path="config/config.yaml"):
        print(f"DEBUG: Initializing analyzer with config: {config_path}")
        self.config_path = config_path
        
        # Test config loading
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            print("✓ Config loaded in analyzer")
        except Exception as e:
            print(f"❌ Config loading failed in analyzer: {e}")
            self.config = self.get_minimal_config()
    
    def get_minimal_config(self):
        return {
            'multi_stock_analysis': {
                'active_universe': 'debug',
                'active_grid': 'debug',
                'stock_universes': {
                    'debug': [
                        {'ticker': 'AAPL', 'description': 'Apple Inc.'}
                    ]
                },
                'hyperparameter_grids': {
                    'debug': [
                        {'hidden_size': 32, 'sequence_length': 20, 'description': 'Debug-Small'}
                    ]
                }
            }
        }
    
    def test_basic_functionality(self):
        print("DEBUG: Testing basic functionality...")
        
        # Test stock universe loading
        try:
            stocks = self.get_stocks_from_config()
            print(f"✓ Loaded {len(stocks)} stocks: {[s[0] for s in stocks]}")
        except Exception as e:
            print(f"❌ Stock loading failed: {e}")
        
        # Test parameter grid loading
        try:
            params = self.get_params_from_config()
            print(f"✓ Loaded {len(params)} parameter combinations")
            for i, param in enumerate(params):
                print(f"  {i+1}. {param}")
        except Exception as e:
            print(f"❌ Parameter loading failed: {e}")
    
    def get_stocks_from_config(self):
        """Simplified stock loading"""
        msa_config = self.config.get('multi_stock_analysis', {})
        universe_name = msa_config.get('active_universe', 'debug')
        universes = msa_config.get('stock_universes', {})
        
        if universe_name in universes:
            stock_list = universes[universe_name]
            return [(stock['ticker'], stock['description']) for stock in stock_list]
        else:
            return [('AAPL', 'Apple Inc.')]
    
    def get_params_from_config(self):
        """Simplified parameter loading"""
        msa_config = self.config.get('multi_stock_analysis', {})
        grid_name = msa_config.get('active_grid', 'debug')
        grids = msa_config.get('hyperparameter_grids', {})
        
        if grid_name in grids:
            return grids[grid_name]
        else:
            return [{'hidden_size': 32, 'sequence_length': 20, 'description': 'Debug-Small'}]

# Run the debug test
print("\n=== DEBUG: Running analyzer test ===")
try:
    analyzer = DebugMultiStockAnalyzer()
    analyzer.test_basic_functionality()
    print("✓ Basic analyzer functionality works")
except Exception as e:
    print(f"❌ Analyzer test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== DEBUG: Network connectivity test ===")
try:
    import urllib.request
    response = urllib.request.urlopen('https://finance.yahoo.com', timeout=10)
    print("✓ Internet connectivity OK")
except Exception as e:
    print(f"❌ Network test failed: {e}")

print("\n=== DEBUG: Complete ===")
print("If you see this message, the basic script structure is working.")
print("Any ❌ errors above need to be fixed before running the full analysis.")

# Test argument parsing
if __name__ == "__main__":
    import argparse
    print("\n=== DEBUG: Testing argument parsing ===")
    
    parser = argparse.ArgumentParser(description='Debug version')
    parser.add_argument('--dry-run', action='store_true', help='Test mode')
    parser.add_argument('--universe', type=str, default=None, help='Universe name')
    
    try:
        args = parser.parse_args()
        print(f"✓ Arguments parsed: dry_run={args.dry_run}, universe={args.universe}")
        
        if args.dry_run:
            print("✓ Dry run mode - this would show what analysis would run")
        else:
            print("⚠️  This is debug mode - no actual analysis will run")
            
    except Exception as e:
        print(f"❌ Argument parsing failed: {e}")
