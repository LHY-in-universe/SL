"""
Resource management for models.
"""

import logging
import psutil
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Resource types."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    GPU_MEMORY = "gpu_memory"


@dataclass
class ResourceUsage:
    """Resource usage information."""
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    gpu_memory_mb: Optional[Dict[int, float]] = None
    gpu_utilization: Optional[Dict[int, float]] = None


class ResourceManager:
    """
    Manages system resources for model deployment.

    Tracks CPU, memory, and GPU usage to ensure models
    don't exceed available resources.
    """

    def __init__(self, max_memory_percent: float = 80.0):
        """
        Initialize resource manager.

        Args:
            max_memory_percent: Maximum memory usage percentage (0-100)
        """
        self.max_memory_percent = max_memory_percent
        self.cuda_available = torch.cuda.is_available()

        if self.cuda_available:
            self.num_gpus = torch.cuda.device_count()
            logger.info(f"CUDA available with {self.num_gpus} GPU(s)")
        else:
            self.num_gpus = 0
            logger.info("CUDA not available, using CPU only")

    def get_current_usage(self) -> ResourceUsage:
        """
        Get current resource usage.

        Returns:
            ResourceUsage object with current metrics
        """
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_mb = (memory.total - memory.available) / (1024 ** 2)
        memory_percent = memory.percent

        usage = ResourceUsage(
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            memory_percent=memory_percent
        )

        # GPU metrics if available
        if self.cuda_available:
            gpu_memory = {}
            gpu_util = {}

            for i in range(self.num_gpus):
                try:
                    # Memory
                    allocated = torch.cuda.memory_allocated(i) / (1024 ** 2)
                    gpu_memory[i] = allocated

                    # Note: GPU utilization requires nvidia-ml-py3
                    # We'll skip it here for simplicity
                    gpu_util[i] = 0.0

                except Exception as e:
                    logger.warning(f"Failed to get GPU {i} metrics: {e}")

            usage.gpu_memory_mb = gpu_memory
            usage.gpu_utilization = gpu_util

        return usage

    def check_available_resources(
        self,
        required_memory_mb: Optional[float] = None,
        required_gpu: bool = False
    ) -> bool:
        """
        Check if required resources are available.

        Args:
            required_memory_mb: Required memory in MB
            required_gpu: Whether GPU is required

        Returns:
            True if resources are available
        """
        usage = self.get_current_usage()

        # Check memory
        if required_memory_mb:
            total_memory_mb = psutil.virtual_memory().total / (1024 ** 2)
            max_allowed_mb = total_memory_mb * (self.max_memory_percent / 100.0)

            if usage.memory_mb + required_memory_mb > max_allowed_mb:
                logger.warning(
                    f"Insufficient memory: current={usage.memory_mb:.0f}MB, "
                    f"required={required_memory_mb:.0f}MB, "
                    f"max_allowed={max_allowed_mb:.0f}MB"
                )
                return False

        # Check GPU
        if required_gpu and not self.cuda_available:
            logger.warning("GPU required but CUDA not available")
            return False

        return True

    def find_best_device(
        self,
        prefer_gpu: bool = True,
        min_free_memory_mb: float = 1000.0
    ) -> str:
        """
        Find best available device for model loading.

        Args:
            prefer_gpu: Prefer GPU over CPU if available
            min_free_memory_mb: Minimum free GPU memory required (MB)

        Returns:
            Device string ("cpu", "cuda:0", etc.)
        """
        if not prefer_gpu or not self.cuda_available:
            return "cpu"

        # Find GPU with most free memory
        best_gpu = None
        max_free_memory = 0.0

        for i in range(self.num_gpus):
            try:
                # Get free memory
                total_memory = torch.cuda.get_device_properties(i).total_memory
                allocated_memory = torch.cuda.memory_allocated(i)
                free_memory_mb = (total_memory - allocated_memory) / (1024 ** 2)

                if free_memory_mb >= min_free_memory_mb and free_memory_mb > max_free_memory:
                    max_free_memory = free_memory_mb
                    best_gpu = i

            except Exception as e:
                logger.warning(f"Failed to check GPU {i}: {e}")

        if best_gpu is not None:
            logger.info(
                f"Selected cuda:{best_gpu} with {max_free_memory:.0f}MB free"
            )
            return f"cuda:{best_gpu}"

        logger.warning("No suitable GPU found, falling back to CPU")
        return "cpu"

    def get_gpu_info(self) -> List[Dict[str, any]]:
        """
        Get information about available GPUs.

        Returns:
            List of dictionaries with GPU information
        """
        if not self.cuda_available:
            return []

        gpu_info = []

        for i in range(self.num_gpus):
            try:
                props = torch.cuda.get_device_properties(i)
                total_memory_mb = props.total_memory / (1024 ** 2)
                allocated_mb = torch.cuda.memory_allocated(i) / (1024 ** 2)
                reserved_mb = torch.cuda.memory_reserved(i) / (1024 ** 2)
                free_mb = total_memory_mb - allocated_mb

                info = {
                    "device_id": i,
                    "name": props.name,
                    "total_memory_mb": round(total_memory_mb, 2),
                    "allocated_mb": round(allocated_mb, 2),
                    "reserved_mb": round(reserved_mb, 2),
                    "free_mb": round(free_mb, 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                }

                gpu_info.append(info)

            except Exception as e:
                logger.warning(f"Failed to get info for GPU {i}: {e}")

        return gpu_info

    def log_resource_usage(self) -> None:
        """Log current resource usage."""
        usage = self.get_current_usage()

        logger.info(
            f"Resource Usage - "
            f"CPU: {usage.cpu_percent:.1f}%, "
            f"Memory: {usage.memory_mb:.0f}MB ({usage.memory_percent:.1f}%)"
        )

        if usage.gpu_memory_mb:
            for gpu_id, mem_mb in usage.gpu_memory_mb.items():
                logger.info(f"GPU {gpu_id}: {mem_mb:.0f}MB allocated")
