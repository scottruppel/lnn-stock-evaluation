# gpu_debug.py - Add this to your repo for GPU troubleshooting

import torch
import subprocess
import os

def check_gpu_status():
    """Comprehensive GPU debugging for Jetson"""
    print("🔍 GPU DEBUG REPORT")
    print("=" * 50)
    
    # 1. Basic CUDA check
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        
        # Memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        cached = torch.cuda.memory_reserved(0)
        
        print(f"Total GPU memory: {total_memory / 1e9:.2f} GB")
        print(f"Allocated: {allocated / 1e6:.2f} MB")
        print(f"Cached: {cached / 1e6:.2f} MB")
    
    # 2. NVIDIA-SMI check
    print("\n🖥️ NVIDIA-SMI OUTPUT:")
    print("-" * 30)
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print(result.stdout)
    except:
        print("❌ nvidia-smi not available")
    
    # 3. Test GPU computation
    print("\n🧪 GPU COMPUTATION TEST:")
    print("-" * 30)
    try:
        # Create test tensors
        x = torch.randn(1000, 1000)
        if torch.cuda.is_available():
            x_gpu = x.cuda()
            y_gpu = torch.mm(x_gpu, x_gpu.t())
            print("✅ GPU computation successful!")
            print(f"Result shape: {y_gpu.shape}")
            
            # Speed test
            import time
            start = time.time()
            for _ in range(10):
                _ = torch.mm(x_gpu, x_gpu.t())
            gpu_time = time.time() - start
            
            start = time.time()
            for _ in range(10):
                _ = torch.mm(x, x.t())
            cpu_time = time.time() - start
            
            print(f"GPU time: {gpu_time:.3f}s")
            print(f"CPU time: {cpu_time:.3f}s")
            print(f"Speedup: {cpu_time/gpu_time:.2f}x")
            
        else:
            print("❌ GPU not available for computation")
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
    
    # 4. Check if model is actually using GPU
    print("\n🧠 MODEL GPU USAGE TEST:")
    print("-" * 30)
    try:
        import sys
        sys.path.append('src')
        from models.lnn_model import LiquidNetwork
        
        model = LiquidNetwork(3, 50, 1)
        print(f"Model device (before): {next(model.parameters()).device}")
        
        if torch.cuda.is_available():
            model = model.cuda()
            print(f"Model device (after .cuda()): {next(model.parameters()).device}")
            
            # Test forward pass
            test_input = torch.randn(32, 30, 3).cuda()
            output = model(test_input)
            print(f"✅ Model forward pass on GPU successful!")
            print(f"Output device: {output.device}")
        else:
            print("❌ GPU not available for model")
            
    except Exception as e:
        print(f"❌ Model GPU test failed: {e}")

if __name__ == "__main__":
    check_gpu_status()
