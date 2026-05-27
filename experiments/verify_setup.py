import torch
import pynvml
import os
# script to verify PyTorch installation, CUDA drivers, NVML (for future frequency scaling), VRAM, and DRAM are all communicating correctly.
def check_system_ram():
    print("\n--- 1. SYSTEM RAM (DRAM) CHECK ---")
    try:
        with open('/proc/meminfo', 'r') as mem:
            mem_info = mem.read()
        for line in mem_info.split('\n'):
            if "MemTotal" in line or "MemAvailable" in line:
                print(f"✅ {line.strip()}")
    except Exception as e:
        print(f"❌ Could not read system RAM: {e}")

def check_cuda_and_vram():
    print("\n--- 2. PYTORCH & CUDA VRAM CHECK ---")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ PyTorch sees CUDA! Found {gpu_count} GPU(s).")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            # Convert bytes to GB
            vram_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"✅ GPU {i}: {gpu_name}")
            print(f"✅ VRAM Total: {vram_total:.2f} GB")
            
            if vram_total < 7.5:
                print("⚠️ WARNING: You have less than 7.5GB of VRAM. Vicuna-13B (NF4) requires ~7GB. This will be a very tight fit!")
            else:
                print("✅ VRAM capacity is sufficient for 4-bit Vicuna-13B.")
    else:
        print("❌ ERROR: PyTorch cannot see your GPU. Check CUDA installation in WSL.")

def check_nvml_access():
    print("\n--- 3. NVML (GPU CONTROL) CHECK ---")
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        print(f"✅ NVML Initialized successfully. Sees {device_count} device(s).")
        
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
        print(f"✅ NVML can read Power Limit: {power_limit} W")
        
        # Test if NVML can read clocks (Crucial for SpecDVFS)
        clock_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
        print(f"✅ NVML can read current GPU Clock: {clock_mhz} MHz")
        
    except pynvml.NVMLError as e:
        print(f"❌ NVML Error: {e}")
        print("   (Note: Controlling clocks later will require running Python with sudo)")

def check_quantization():
    print("\n--- 4. QUANTIZATION (BITSANDBYTES) CHECK ---")
    try:
        import bitsandbytes as bnb
        print("✅ bitsandbytes imported successfully! NF4 4-bit loading will work.")
    except ImportError as e:
        print(f"❌ Error loading bitsandbytes: {e}")

if __name__ == "__main__":
    print("="*50)
    print("SpecDVFS - WSL & Hardware Verification Script")
    print("="*50)
    
    check_system_ram()
    check_cuda_and_vram()
    check_nvml_access()
    check_quantization()
    
    print("\n" + "="*50)
    print("If all checks have a ✅, you are ready to run replicate_dutta.py!")