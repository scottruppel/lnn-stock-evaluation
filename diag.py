#!/usr/bin/env python3
"""
Import Diagnostics Script
Run this to identify and fix import issues in your LNN project.
"""

import os
import sys

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

print("=== LNN PROJECT IMPORT DIAGNOSTICS ===")
print(f"Project root: {PROJECT_ROOT}")
print(f"Python path includes: {[p for p in sys.path if 'lnn' in p.lower()]}")

def test_import(module_path, class_names):
    """Test importing specific classes from a module."""
    try:
        module = __import__(module_path, fromlist=class_names)
        print(f"✓ Module {module_path} imported successfully")
        
        for class_name in class_names:
            if hasattr(module, class_name):
                print(f"  ✓ {class_name} available")
            else:
                print(f"  ❌ {class_name} NOT FOUND")
                available = [name for name in dir(module) if not name.startswith('_')]
                print(f"     Available: {available}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import {module_path}: {e}")
        return False

print("\n=== TESTING CORE IMPORTS ===")

# Test data loader
test_import('data.data_loader', ['StockDataLoader'])

# Test LNN model with both possible class names
print(f"\n=== TESTING LNN MODEL ===")
lnn_success = test_import('models.lnn_model', ['LiquidNetwork', 'LiquidNeuralNetwork', 'ModelConfig'])

if not lnn_success:
    print("Attempting to fix LNN model imports...")
    lnn_model_path = os.path.join(PROJECT_ROOT, 'src', 'models', 'lnn_model.py')
    
    if os.path.exists(lnn_model_path):
        print(f"Found LNN model at: {lnn_model_path}")
        
        # Read current content
        with open(lnn_model_path, 'r') as f:
            content = f.read()
        
        # Check if alias already exists
        if 'LiquidNeuralNetwork = LiquidNetwork' not in content:
            print("Adding compatibility alias...")
            
            # Add alias at the end
            alias_code = '''

# Import aliases for backward compatibility
LiquidNeuralNetwork = LiquidNetwork  # Alias for compatibility

# Export all important classes
__all__ = [
    'LiquidTimeCell',
    'LiquidNetwork', 
    'LiquidNeuralNetwork',  # Alias
    'ModelConfig',
    'create_sequences',
    'count_parameters',
    'get_model_summary'
]
'''
            
            with open(lnn_model_path, 'a') as f:
                f.write(alias_code)
            
            print("✓ Added LiquidNeuralNetwork alias to lnn_model.py")
            
            # Test import again
            importlib.reload(sys.modules.get('models.lnn_model', None))
            test_import('models.lnn_model', ['LiquidNetwork', 'LiquidNeuralNetwork'])
        else:
            print("Alias already exists")

# Test analysis modules
print(f"\n=== TESTING ANALYSIS MODULES ===")
test_import('analysis.feature_engineering', ['AdvancedFeatureEngineer'])
test_import('analysis.pattern_recognition', ['PatternRecognizer'])
test_import('analysis.temporal_analysis', ['TemporalAnalyzer'])

# Test run_analysis
print(f"\n=== TESTING RUN_ANALYSIS ===")
try:
    run_analysis_path = os.path.join(PROJECT_ROOT, 'scripts', 'run_analysis.py')
    if os.path.exists(run_analysis_path):
        print(f"✓ Found run_analysis.py at: {run_analysis_path}")
        
        # Add scripts to path temporarily
        scripts_path = os.path.join(PROJECT_ROOT, 'scripts')
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        
        try:
            from run_analysis import ComprehensiveAnalyzer
            print("✓ ComprehensiveAnalyzer imported successfully")
        except ImportError as e:
            print(f"❌ ComprehensiveAnalyzer import failed: {e}")
            
            # Check what's trying to import the wrong class name
            with open(run_analysis_path, 'r') as f:
                content = f.read()
                if 'LiquidNeuralNetwork' in content:
                    print("  Found 'LiquidNeuralNetwork' reference in run_analysis.py")
                    print("  This should be 'LiquidNetwork' instead")
                if 'from models.lnn_model import' in content:
                    print("  Found model import in run_analysis.py")
    else:
        print(f"❌ run_analysis.py not found at: {run_analysis_path}")
        
except Exception as e:
    print(f"❌ Error testing run_analysis: {e}")

print(f"\n=== TESTING PYTORCH AND CUDA ===")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__} imported")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  ⚠️  CUDA not available - check NVIDIA drivers and PyTorch installation")
        print("  For Jetson, try: pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
except ImportError:
    print("❌ PyTorch not available")

print(f"\n=== DIAGNOSTICS COMPLETE ===")
print("Run this script from your project root to identify import issues.")
print("If you see ❌ errors above, those need to be fixed before running the analysis.")

if __name__ == "__main__":
    import importlib
    # Force reload of modules to pick up any changes
    modules_to_reload = ['models.lnn_model']
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
