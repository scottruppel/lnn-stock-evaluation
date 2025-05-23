# simple_test.py
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from data.data_loader import StockDataLoader
    print("✅ Data loader imported successfully")
    
    from models.lnn_model import LiquidNetwork
    print("✅ LNN model imported successfully")
    
    from utils.metrics import StockPredictionMetrics
    print("✅ Metrics imported successfully")
    
    print("\n🎉 All core imports working!")
    
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
