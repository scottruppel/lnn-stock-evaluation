# simple_test.py
import sys
import os

# Add the project root to Python path (not just src)
sys.path.append(os.path.dirname(__file__))

def test_file_exists(filepath):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✅ {filepath} exists")
        return True
    else:
        print(f"❌ {filepath} MISSING")
        return False

def test_import(module_path, class_name):
    """Test importing a specific class"""
    try:
        module = __import__(module_path, fromlist=[class_name])
        getattr(module, class_name)
        print(f"✅ {module_path}.{class_name} imports successfully")
        return True
    except Exception as e:
        print(f"❌ {module_path}.{class_name} failed: {e}")
        return False

print("=== CHECKING FILES EXIST ===")
files_to_check = [
    "data/data_loader.py",
    "data/preprocessor.py", 
    "models/lnn_model.py",
    "utils/metrics.py",
    "utils/experiment_tracker.py"
]

all_files_exist = True
for file_path in files_to_check:
    if not test_file_exists(file_path):
        all_files_exist = False

print("\n=== TESTING IMPORTS ===")
if all_files_exist:
    # Use correct import paths with 'src.' prefix
    test_import("data.data_loader", "StockDataLoader")
    test_import("models.lnn_model", "LiquidNetwork")
    test_import("utils.metrics", "StockPredictionMetrics")
    test_import("utils.experiment_tracker", "ExperimentTracker")
else:
    print("Cannot test imports - some files are missing!")
