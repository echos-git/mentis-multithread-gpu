"""
GPU-optimized configuration system for CAD processing pipeline.
Designed for NVIDIA RTX A6000 and other CUDA-capable GPUs.
"""

import os
import sys
import platform
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import subprocess
import json


@dataclass
class GPUInfo:
    """Information about a detected GPU."""
    index: int
    name: str
    memory_total: int  # MB
    memory_free: int   # MB
    compute_capability: Tuple[int, int]
    cuda_cores: Optional[int] = None
    is_rtx_a6000: bool = False


@dataclass
class SystemSpecs:
    """System hardware specifications."""
    cpu_count: int
    total_memory_gb: float
    gpus: List[GPUInfo]
    platform: str
    python_version: str


class GPUConfigManager:
    """Manages GPU detection and configuration for optimal CAD processing."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.system_specs = self._detect_system()
        self.gpu_config = self._configure_gpu_settings()
        self.pool_config = self._configure_pools()
        self._setup_environment()
    
    def _detect_system(self) -> SystemSpecs:
        """Detect system hardware specifications."""
        import psutil
        import multiprocessing as mp
        
        # Basic system info
        cpu_count = mp.cpu_count()
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        platform_info = platform.platform()
        python_version = platform.python_version()
        
        # GPU detection
        gpus = self._detect_gpus()
        
        return SystemSpecs(
            cpu_count=cpu_count,
            total_memory_gb=total_memory_gb,
            gpus=gpus,
            platform=platform_info,
            python_version=python_version
        )
    
    def _detect_gpus(self) -> List[GPUInfo]:
        """Detect NVIDIA GPUs using nvidia-ml-py3 and fallback methods."""
        gpus = []
        
        # Try nvidia-ml-py3 first (most reliable)
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                
                # Memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_total = mem_info.total // (1024 * 1024)  # Convert to MB
                memory_free = mem_info.free // (1024 * 1024)
                
                # Compute capability
                major = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[0]
                minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[1]
                
                # Special detection for RTX A6000
                is_rtx_a6000 = "A6000" in name.upper()
                cuda_cores = self._estimate_cuda_cores(name, major, minor)
                
                gpu_info = GPUInfo(
                    index=i,
                    name=name,
                    memory_total=memory_total,
                    memory_free=memory_free,
                    compute_capability=(major, minor),
                    cuda_cores=cuda_cores,
                    is_rtx_a6000=is_rtx_a6000
                )
                gpus.append(gpu_info)
                
        except ImportError:
            self.logger.warning("pynvml not available, trying nvidia-smi fallback")
            gpus = self._detect_gpus_nvidia_smi()
        except Exception as e:
            self.logger.warning(f"GPU detection failed: {e}")
            gpus = self._detect_gpus_nvidia_smi()
        
        if gpus:
            self.logger.info(f"Detected {len(gpus)} GPU(s)")
            for gpu in gpus:
                self.logger.info(f"  {gpu.name}: {gpu.memory_total}MB VRAM")
        else:
            self.logger.warning("No NVIDIA GPUs detected")
        
        return gpus
    
    def _detect_gpus_nvidia_smi(self) -> List[GPUInfo]:
        """Fallback GPU detection using nvidia-smi."""
        gpus = []
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=index,name,memory.total,memory.free',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True)
            
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 4:
                        index = int(parts[0])
                        name = parts[1]
                        memory_total = int(parts[2])
                        memory_free = int(parts[3])
                        
                        # Estimate compute capability (basic heuristic)
                        compute_capability = self._estimate_compute_capability(name)
                        is_rtx_a6000 = "A6000" in name.upper()
                        cuda_cores = self._estimate_cuda_cores(name, *compute_capability)
                        
                        gpu_info = GPUInfo(
                            index=index,
                            name=name,
                            memory_total=memory_total,
                            memory_free=memory_free,
                            compute_capability=compute_capability,
                            cuda_cores=cuda_cores,
                            is_rtx_a6000=is_rtx_a6000
                        )
                        gpus.append(gpu_info)
                        
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.warning("nvidia-smi not available")
            
        return gpus
    
    def _estimate_compute_capability(self, gpu_name: str) -> Tuple[int, int]:
        """Estimate compute capability based on GPU name."""
        name_upper = gpu_name.upper()
        
        # RTX 30/40 series and A6000 are typically compute capability 8.6
        if any(x in name_upper for x in ['RTX 30', 'RTX 40', 'A6000', 'RTX 3', 'RTX 4']):
            return (8, 6)
        # RTX 20 series
        elif any(x in name_upper for x in ['RTX 20', 'RTX 2']):
            return (7, 5)
        # GTX 16 series
        elif any(x in name_upper for x in ['GTX 16', 'GTX 1']):
            return (7, 5)
        # Default to a reasonable modern capability
        else:
            return (7, 0)
    
    def _estimate_cuda_cores(self, gpu_name: str, major: int, minor: int) -> Optional[int]:
        """Estimate CUDA cores based on GPU name and compute capability."""
        name_upper = gpu_name.upper()
        
        # RTX A6000 has 10752 CUDA cores
        if "A6000" in name_upper:
            return 10752
        # Add more specific GPU mappings as needed
        elif "RTX 4090" in name_upper:
            return 16384
        elif "RTX 4080" in name_upper:
            return 9728
        elif "RTX 3090" in name_upper:
            return 10496
        elif "RTX 3080" in name_upper:
            return 8704
        
        return None
    
    def _configure_gpu_settings(self) -> Dict:
        """Configure GPU-specific settings based on detected hardware."""
        config = {
            'cuda_available': len(self.system_specs.gpus) > 0,
            'primary_gpu': None,
            'vram_pool_size_mb': 0,
            'max_concurrent_renders': 1,
            'batch_size_renders': 1,
            'batch_size_pointclouds': 1,
            'use_unified_memory': False,
            'memory_fraction': 0.8,  # Use 80% of available VRAM
        }
        
        if self.system_specs.gpus:
            # Use the first (primary) GPU
            primary_gpu = self.system_specs.gpus[0]
            config['primary_gpu'] = primary_gpu
            
            # Calculate VRAM pool size (80% of available memory)
            available_vram = min(primary_gpu.memory_free, primary_gpu.memory_total)
            config['vram_pool_size_mb'] = int(available_vram * config['memory_fraction'])
            
            # RTX A6000 specific optimizations
            if primary_gpu.is_rtx_a6000:
                config.update({
                    'max_concurrent_renders': 8,    # Can handle many concurrent renders
                    'batch_size_renders': 6,        # Batch 6 views at once
                    'batch_size_pointclouds': 32,   # Large batches for point cloud generation
                    'use_unified_memory': True,     # Enable for large meshes
                    'memory_fraction': 0.85,        # Can use more VRAM on A6000
                })
            # Other high-end GPUs
            elif primary_gpu.memory_total >= 20000:  # 20GB+
                config.update({
                    'max_concurrent_renders': 6,
                    'batch_size_renders': 4,
                    'batch_size_pointclouds': 24,
                    'use_unified_memory': True,
                })
            # Mid-range GPUs
            elif primary_gpu.memory_total >= 10000:  # 10GB+
                config.update({
                    'max_concurrent_renders': 4,
                    'batch_size_renders': 3,
                    'batch_size_pointclouds': 16,
                })
            # Lower-end GPUs
            else:
                config.update({
                    'max_concurrent_renders': 2,
                    'batch_size_renders': 2,
                    'batch_size_pointclouds': 8,
                })
            
            # Recalculate VRAM pool with updated memory fraction
            available_vram = min(primary_gpu.memory_free, primary_gpu.memory_total)
            config['vram_pool_size_mb'] = int(available_vram * config['memory_fraction'])
        
        return config
    
    def _configure_pools(self) -> Dict:
        """Configure process pools based on GPU and CPU resources."""
        cpu_cores = self.system_specs.cpu_count
        has_gpu = len(self.system_specs.gpus) > 0
        
        if has_gpu:
            primary_gpu = self.system_specs.gpus[0]
            
            # GPU-optimized pool configuration
            if primary_gpu.is_rtx_a6000:
                # RTX A6000: Favor GPU-heavy workloads
                config = {
                    'stl_generation_workers': min(cpu_cores, 8),     # STL gen is CPU-bound
                    'render_workers': 2,                             # Few workers, high GPU utilization
                    'pointcloud_workers': 4,                         # Moderate for GPU+CPU hybrid
                    'gpu_render_streams': 4,                         # Multiple CUDA streams
                    'task_batch_size': 16,                          # Large batches
                }
            else:
                # Other GPUs: More conservative
                config = {
                    'stl_generation_workers': min(cpu_cores, 6),
                    'render_workers': 1,
                    'pointcloud_workers': 3,
                    'gpu_render_streams': 2,
                    'task_batch_size': 8,
                }
        else:
            # CPU-only fallback
            config = {
                'stl_generation_workers': min(cpu_cores, 6),
                'render_workers': min(cpu_cores // 2, 4),
                'pointcloud_workers': min(cpu_cores, 8),
                'gpu_render_streams': 0,
                'task_batch_size': 4,
            }
        
        # Memory and task limits
        config.update({
            'memory_threshold_mb': 4096,         # 4GB per worker
            'task_limit_per_worker': 100,        # Recycle workers after 100 tasks
            'enable_worker_recycling': True,
        })
        
        return config
    
    def _setup_environment(self):
        """Setup environment variables for optimal GPU operation."""
        # CUDA settings
        if self.gpu_config['cuda_available']:
            # Enable CUDA caching
            os.environ['CUDA_CACHE_DISABLE'] = '0'
            os.environ['CUDA_CACHE_MAXSIZE'] = '2147483648'  # 2GB cache
            
            # Memory management
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            
            # VTK GPU settings
            os.environ['VTK_DEFAULT_OPENGL_WINDOW'] = 'vtkOpenGLRenderWindow'
            os.environ['VTK_USE_CUDA'] = '1'
            
            # Disable MSAA for performance (can be enabled later if needed)
            os.environ['VTK_FORCE_MSAA'] = '0'
        
        # PyVista settings
        os.environ['PYVISTA_OFF_SCREEN'] = 'true'
        os.environ['PYVISTA_USE_PANEL'] = 'false'
        
        # Set multiprocessing method
        import multiprocessing as mp
        if mp.get_start_method(allow_none=True) is None:
            # Use 'spawn' for CUDA compatibility
            mp.set_start_method('spawn', force=True)
    
    def get_optimal_settings(self) -> Dict:
        """Get the complete configuration dictionary."""
        return {
            'system': {
                'cpu_count': self.system_specs.cpu_count,
                'total_memory_gb': self.system_specs.total_memory_gb,
                'platform': self.system_specs.platform,
                'python_version': self.system_specs.python_version,
            },
            'gpu': self.gpu_config,
            'pools': self.pool_config,
            'gpus': [
                {
                    'index': gpu.index,
                    'name': gpu.name,
                    'memory_total_mb': gpu.memory_total,
                    'memory_free_mb': gpu.memory_free,
                    'compute_capability': gpu.compute_capability,
                    'cuda_cores': gpu.cuda_cores,
                    'is_rtx_a6000': gpu.is_rtx_a6000,
                }
                for gpu in self.system_specs.gpus
            ]
        }
    
    def log_configuration(self):
        """Log the detected configuration."""
        settings = self.get_optimal_settings()
        
        self.logger.info("=== GPU Configuration ===")
        self.logger.info(f"Platform: {settings['system']['platform']}")
        self.logger.info(f"CPU Cores: {settings['system']['cpu_count']}")
        self.logger.info(f"System RAM: {settings['system']['total_memory_gb']:.1f}GB")
        
        if settings['gpu']['cuda_available']:
            self.logger.info(f"CUDA Available: Yes")
            primary_gpu = settings['gpu']['primary_gpu']
            if primary_gpu:
                self.logger.info(f"Primary GPU: {primary_gpu.name}")
                self.logger.info(f"VRAM: {primary_gpu.memory_total}MB total, {primary_gpu.memory_free}MB free")
                self.logger.info(f"Compute Capability: {primary_gpu.compute_capability}")
                if primary_gpu.cuda_cores:
                    self.logger.info(f"CUDA Cores: {primary_gpu.cuda_cores}")
        else:
            self.logger.info("CUDA Available: No")
        
        self.logger.info(f"Pool Configuration:")
        for key, value in settings['pools'].items():
            self.logger.info(f"  {key}: {value}")


# Global configuration instance
CONFIG = GPUConfigManager() 