# Create a test file
cat > test_qqq_data.py << 'EOF'
import sys
import os
sys.path.append('src')

print("Testing QQQ data...")

try:
    from data.data_loader import StockDataLoader
    print("✅ DataLoader imported")
    
    import numpy as np
    print("✅ NumPy imported")
    
    print("Downloading data...")
    loader = StockDataLoader(['QQQ', '^GSPC', 'AGG', 'SPY'], '2020-01-01', '2024-12-31')
    data = loader.download_data()
    print("✅ Data downloaded")
    
    print('QQQ data shape:', data['Close']['QQQ'].shape)
    print('QQQ data range:', data['Close']['QQQ'].min(), 'to', data['Close']['QQQ'].max())
    print('QQQ has NaN values:', data['Close']['QQQ'].isna().sum())
    print('QQQ sample values:', data['Close']['QQQ'].head().values.flatten())
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
EOF

# Run the test file
python test_qqq_data.py
