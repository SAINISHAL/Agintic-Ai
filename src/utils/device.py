from __future__ import annotations

import torch


def select_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Selected device: CUDA ({name}, {memory_gb:.1f} GB)")
        return device

    print("Selected device: CPU")
    return torch.device("cpu")


def device_summary() -> dict:
    cuda = torch.cuda.is_available()
    info = {"cuda_available": cuda, "device": "cuda" if cuda else "cpu"}
    if cuda:
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
        )
    return info
