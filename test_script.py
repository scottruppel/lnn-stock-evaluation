#!/bin/bash
# Test Setup Script for Multi-Stock Analysis
# Run this from your LNN project root directory

echo "=== LNN MULTI-STOCK ANALYSIS SETUP TEST ==="
echo "Current directory: $(pwd)"
echo "Date: $(date)"
echo ""

# Check if we're in the right directory
echo "=== CHECKING PROJECT STRUCTURE ==="
if [ ! -d "src" ] || [ ! -d "config" ] || [ ! -d "scripts" ]; then
    echo "❌ ERROR: Not in project root directory!"
    echo "   Please run this from your LNN project root (the directory containing src/, config/, scripts/)"
    echo "   Current contents:"
    ls -la
    exit 1
fi

echo "✓ Project structure looks correct"
echo "✓ Found directories:"
for dir in src config scripts results; do
    if [ -d "$dir" ]; then
        echo "  ✓ $dir/"
    else
        echo "  ⚠️  $dir/ (will be created)"
        mkdir -p "$dir"
    fi
done

echo ""
echo "=== CHECKING CONFIG FILE ==="
if [ -f "config/config2.yaml" ]; then
    echo "✓ Found config/config2.yaml"
else
    echo "❌ config/config2.yaml not found"
    echo "Creating minimal config file..."
    cat > config/config2.yaml << 'EOF'
# Minimal config for multi-stock analysis testing
data:
  tickers: ['^GSPC', 'AGG', 'QQQ', 'AAPL']
  start_date: '2020-01-01'
  end_date: '2024-12-31'
  target_ticker: 'AAPL'

model:
  sequence_length: 30
  hidden_size: 64
  learning_rate: 0.001
  batch_size: 32
  num_epochs: 20  # Reduced for testing
  patience: 5

analysis:
  use_advanced_features: true
  pattern_analysis: true
  temporal_analysis: true
  dimensionality_reduction: true

multi_stock_analysis:
  active_universe: 'quick_test'
  active_grid: 'debug'
  output_dir: 'results/multi_stock_analysis'
  
  stock_universes:
    quick_test:
      - {ticker: "AAPL", description: "Apple Inc."}
      - {ticker: "MSFT", description: "Microsoft Corp."}
    
    debug:
      - {ticker: "AAPL", description: "Apple Inc."}
  
  hyperparameter_grids:
    debug:
      - {hidden_size: 32, sequence_length: 20, description: "Debug-Small"}
    
    quick_3:
      - {hidden_size: 32, sequence_length: 20, description: "Quick-Small"}
      - {hidden_size: 64, sequence_length: 30, description: "Quick-Medium"}
      - {hidden_size: 128, sequence_length: 30, description: "Quick-Large"}
EOF
    echo "✓ Created config/config2.yaml"
fi

echo ""
echo "=== CHECKING PYTHON ENVIRONMENT ==="
python3 -c "
import sys
print(f'Python version: {sys.version}')

# Check critical packages
packages = ['numpy', 'pandas', 'torch', 'yaml']
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✓ {pkg} available')
    except ImportError:
        print(f'❌ {pkg} missing - install with: pip install {pkg}')

# Check CUDA
try:
    import torch
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('⚠️  No CUDA - will run on CPU (slower)')
except:
    print('❌ PyTorch not available')
"

echo ""
echo "=== TESTING SCRIPT IMPORT ==="
python3 -c "
import sys
import os
sys.path.insert(0, 'src')

try:
    from data.data_loader import StockDataLoader
    print('✓ StockDataLoader can be imported')
except Exception as e:
    print(f'❌ StockDataLoader import failed: {e}')

try:
    sys.path.insert(0, 'scripts')
    from run_analysis import ComprehensiveAnalyzer
    print('✓ ComprehensiveAnalyzer can be imported')
except Exception as e:
    print(f'❌ ComprehensiveAnalyzer import failed: {e}')
"

echo ""
echo "=== TESTING MULTI-STOCK SCRIPT ==="
echo "Running dry-run test..."
python3 scripts/multi_stock_analysis.py --config config/config2.yaml --universe debug --grid debug --dry-run

echo ""
echo "=== SETUP TEST COMPLETE ==="
echo "If you see no errors above, you can run:"
echo "  python3 scripts/multi_stock_analysis.py --config config/config2.yaml --universe debug --grid debug"
echo ""
echo "For a quick real test with 2 stocks:"
echo "  python3 scripts/multi_stock_analysis.py --config config/config2.yaml --universe quick_test --grid quick_3"
