
from __future__ import annotations

import logging
from typing import Tuple, Dict

import torch 

log = logging.getLogger("gaitguard.device")


def select_device(prefer_gpu: bool = True) -> Tuple[str, bool]:
    if prefer_gpu and torch.cuda.is_available():
        device = "cuda"
        use_half = True
        name = torch.cuda.get_device_name(0)
        log.info("Using CUDA device: %s", name)
        
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info("GPU Memory Capacity: %.2f GB", total_memory)
    else:
        device = "cpu"
        use_half = False
        if prefer_gpu:
            log.warning("CUDA requested but not available. Falling back to CPU.")
        else:
            log.info("Using CPU (GPU explicitly disabled).")

    return device, use_half


def get_gpu_memory_stats() -> Dict[str, float | str]:
    if not torch.cuda.is_available():
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "total_gb": 0.0,
            "percent_used": 0.0,
            "status": "CPU_MODE"
        }

    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    percent_used = (allocated / total * 100) if total > 0 else 0

    if percent_used > 85:
        status = "CRITICAL"
    elif percent_used > 75:
        status = "WARNING"
    else:
        status = "OK"

    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "total_gb": total,
        "percent_used": percent_used,
        "status": status
    }
