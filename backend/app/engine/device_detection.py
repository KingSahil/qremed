"""
device_detection.py
===================
Detects hardware accelerators (NVIDIA GPU via nvidia-smi / CUDA runtime,
CPU cores) to dynamically train models on GPU when present, or fall back to CPU.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import psutil


def detect_gpu() -> dict:
    """Detects if an NVIDIA GPU with CUDA support is available on this system."""
    gpu_info = {
        "has_gpu": False,
        "device_type": "cpu",
        "device_name": "CPU",
        "gpu_name": None,
        "vram_total_mb": None,
        "driver_version": None,
        "cuda_version": None,
        "cuda_device": None,
        "status_text": "CPU Mode (No GPU detected)",
    }

    # 1. Try nvidia-smi query
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            # Query GPU name, memory total, driver version
            cmd = [
                nvidia_smi,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            if res.returncode == 0 and res.stdout.strip():
                lines = res.stdout.strip().split("\n")
                if lines:
                    parts = [p.strip() for p in lines[0].split(",")]
                    gpu_name = parts[0] if len(parts) > 0 else "NVIDIA GPU"
                    vram_mb = int(float(parts[1])) if len(parts) > 1 and parts[1].replace(".", "").isdigit() else None
                    driver_ver = parts[2] if len(parts) > 2 else None

                    # Verify if XGBoost or CUDA can use this device
                    gpu_info.update({
                        "has_gpu": True,
                        "device_type": "gpu",
                        "device_name": gpu_name,
                        "gpu_name": gpu_name,
                        "vram_total_mb": vram_mb,
                        "driver_version": driver_ver,
                        "cuda_device": "cuda:0",
                        "status_text": f"NVIDIA GPU Active: {gpu_name} ({vram_mb} MB VRAM)",
                    })
                    return gpu_info
        except Exception:
            pass

    # 2. Check if PyTorch or CuPy sees CUDA
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
            gpu_info.update({
                "has_gpu": True,
                "device_type": "gpu",
                "device_name": name,
                "gpu_name": name,
                "vram_total_mb": vram,
                "cuda_device": "cuda:0",
                "status_text": f"NVIDIA GPU Active: {name} ({vram} MB VRAM)",
            })
            return gpu_info
    except Exception:
        pass

    return gpu_info


def get_system_hardware_summary() -> dict:
    """Returns complete summary of host compute resources (GPU + CPU)."""
    gpu = detect_gpu()
    cpu_logical = psutil.cpu_count(logical=True) or 1
    cpu_physical = psutil.cpu_count(logical=False) or 1
    mem = psutil.virtual_memory()

    return {
        "gpu": gpu,
        "cpu": {
            "cores_logical": cpu_logical,
            "cores_physical": cpu_physical,
            "ram_total_gb": round(mem.total / (1024 ** 3), 2),
            "ram_available_gb": round(mem.available / (1024 ** 3), 2),
        },
        "preferred_device": "GPU (CUDA)" if gpu["has_gpu"] else "CPU",
        "acceleration_enabled": gpu["has_gpu"],
    }
