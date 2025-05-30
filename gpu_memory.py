"""
GPU memory management system for CAD processing pipeline.
Handles CUDA memory pools, VTK resource management, and GPU-CPU transfers.
"""

import os
import gc
import logging
import weakref
from typing import Optional, Dict, Any, List, Union
from threading import Lock
from contextlib import contextmanager
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import pyvista as pv


class CUDAMemoryPool:
    """Manages CUDA memory allocation and recycling."""
    
    def __init__(self, initial_size_mb: int = 1024, max_size_mb: int = 8192):
        self.logger = logging.getLogger(f"{__name__}.CUDAPool")
        self.initial_size_mb = initial_size_mb
        self.max_size_mb = max_size_mb
        self._lock = Lock()
        self._initialized = False
        
        self._setup_memory_pool()
    
    def _setup_memory_pool(self):
        """Initialize CUDA memory pool if available."""
        if CUPY_AVAILABLE:
            try:
                # Configure CuPy memory pool
                cp.cuda.MemoryPool().set_limit(size=self.max_size_mb * 1024 * 1024)
                cp.cuda.MemoryPool().set_growth_policy('exponential')
                self._initialized = True
                self.logger.info(f"CuPy memory pool initialized: {self.max_size_mb}MB limit")
            except Exception as e:
                self.logger.warning(f"Failed to initialize CuPy memory pool: {e}")
        elif TORCH_AVAILABLE:
            try:
                # Configure PyTorch CUDA memory
                torch.cuda.empty_cache()
                # Set memory fraction if needed
                # torch.cuda.set_per_process_memory_fraction(0.8)
                self._initialized = True
                self.logger.info("PyTorch CUDA memory management initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize PyTorch CUDA: {e}")
    
    @contextmanager
    def cuda_context(self, device_id: int = 0):
        """Context manager for CUDA device operations."""
        if not self._initialized:
            yield
            return
        
        current_device = None
        try:
            if CUPY_AVAILABLE:
                current_device = cp.cuda.Device().id
                cp.cuda.Device(device_id).use()
            elif TORCH_AVAILABLE:
                current_device = torch.cuda.current_device()
                torch.cuda.set_device(device_id)
            
            yield
            
        finally:
            # Restore previous device if needed
            if current_device is not None:
                if CUPY_AVAILABLE:
                    cp.cuda.Device(current_device).use()
                elif TORCH_AVAILABLE:
                    torch.cuda.set_device(current_device)
    
    def allocate_gpu_array(self, shape: tuple, dtype=np.float32, device_id: int = 0):
        """Allocate GPU array with the available framework."""
        with self.cuda_context(device_id):
            if CUPY_AVAILABLE:
                return cp.zeros(shape, dtype=dtype)
            elif TORCH_AVAILABLE:
                return torch.zeros(shape, dtype=torch.float32, device=f'cuda:{device_id}')
            else:
                # Fallback to CPU
                return np.zeros(shape, dtype=dtype)
    
    def free_memory(self, device_id: int = 0):
        """Free unused GPU memory."""
        with self._lock:
            try:
                if CUPY_AVAILABLE:
                    with self.cuda_context(device_id):
                        cp.get_default_memory_pool().free_all_blocks()
                        cp.get_default_pinned_memory_pool().free_all_blocks()
                elif TORCH_AVAILABLE:
                    torch.cuda.empty_cache()
                
                self.logger.debug(f"Freed GPU memory on device {device_id}")
            except Exception as e:
                self.logger.warning(f"Error freeing GPU memory: {e}")
    
    def get_memory_info(self, device_id: int = 0) -> Dict[str, int]:
        """Get current GPU memory usage."""
        try:
            if CUPY_AVAILABLE:
                with self.cuda_context(device_id):
                    mem_pool = cp.get_default_memory_pool()
                    return {
                        'used_bytes': mem_pool.used_bytes(),
                        'total_bytes': mem_pool.total_bytes(),
                        'free_bytes': mem_pool.free_bytes(),
                    }
            elif TORCH_AVAILABLE:
                return {
                    'allocated_bytes': torch.cuda.memory_allocated(device_id),
                    'cached_bytes': torch.cuda.memory_reserved(device_id),
                    'max_allocated_bytes': torch.cuda.max_memory_allocated(device_id),
                }
        except Exception as e:
            self.logger.warning(f"Error getting memory info: {e}")
        
        return {'used_bytes': 0, 'total_bytes': 0, 'free_bytes': 0}


class PyVistaGPUManager:
    """Manages PyVista plotters and VTK resources for GPU operations."""
    
    def __init__(self, max_plotters: int = 4, plotter_reuse_limit: int = 50):
        self.logger = logging.getLogger(f"{__name__}.PyVistaGPU")
        self.max_plotters = max_plotters
        self.plotter_reuse_limit = plotter_reuse_limit
        
        self._plotter_pool: List[pv.Plotter] = []
        self._plotter_usage: Dict[int, int] = {}
        self._lock = Lock()
        
        # Ensure off-screen rendering is enabled
        pv.OFF_SCREEN = True
        
        # Configure VTK for GPU operation
        self._setup_vtk_gpu()
    
    def _setup_vtk_gpu(self):
        """Configure VTK for optimal GPU operation."""
        try:
            # Set VTK to use OpenGL with GPU acceleration
            os.environ['VTK_DEFAULT_OPENGL_WINDOW'] = 'vtkOpenGLRenderWindow'
            
            # Enable VTK GPU features if available
            import vtk
            
            # Try to enable GPU-based operations
            if hasattr(vtk, 'vtkOpenGLGPUVolumeRayCastMapper'):
                self.logger.debug("VTK GPU volume rendering available")
            
            self.logger.info("VTK configured for GPU operation")
            
        except Exception as e:
            self.logger.warning(f"Error configuring VTK for GPU: {e}")
    
    @contextmanager
    def get_plotter(self, **plotter_kwargs):
        """Context manager for getting and releasing plotters."""
        plotter = None
        plotter_id = None
        
        try:
            with self._lock:
                # Try to reuse an existing plotter
                if self._plotter_pool:
                    plotter = self._plotter_pool.pop()
                    plotter_id = id(plotter)
                    
                    # Check if plotter needs replacement
                    usage_count = self._plotter_usage.get(plotter_id, 0)
                    if usage_count >= self.plotter_reuse_limit:
                        self._cleanup_plotter(plotter)
                        plotter = None
                        plotter_id = None
                
                # Create new plotter if needed
                if plotter is None:
                    plotter = pv.Plotter(off_screen=True, **plotter_kwargs)
                    plotter_id = id(plotter)
                    self._plotter_usage[plotter_id] = 0
                    
                    self.logger.debug(f"Created new PyVista plotter {plotter_id}")
            
            yield plotter
            
        finally:
            if plotter is not None and plotter_id is not None:
                try:
                    # Clean the plotter for reuse
                    plotter.clear()
                    plotter.remove_all_lights()
                    plotter.reset_camera()
                    
                    with self._lock:
                        self._plotter_usage[plotter_id] += 1
                        
                        # Return to pool if under limit
                        if (len(self._plotter_pool) < self.max_plotters and 
                            self._plotter_usage[plotter_id] < self.plotter_reuse_limit):
                            self._plotter_pool.append(plotter)
                        else:
                            self._cleanup_plotter(plotter)
                            
                except Exception as e:
                    self.logger.warning(f"Error returning plotter to pool: {e}")
                    self._cleanup_plotter(plotter)
    
    def _cleanup_plotter(self, plotter):
        """Clean up a plotter and remove from tracking."""
        try:
            plotter_id = id(plotter)
            plotter.close()
            
            # Remove from usage tracking
            self._plotter_usage.pop(plotter_id, None)
            
            self.logger.debug(f"Cleaned up plotter {plotter_id}")
            
        except Exception as e:
            self.logger.warning(f"Error cleaning up plotter: {e}")
    
    def cleanup_all(self):
        """Clean up all plotters and VTK resources."""
        with self._lock:
            # Clean up pooled plotters
            for plotter in self._plotter_pool:
                self._cleanup_plotter(plotter)
            
            self._plotter_pool.clear()
            self._plotter_usage.clear()
            
            # Global VTK cleanup
            try:
                pv.close_all()
                
                # VTK garbage collection
                import vtk
                if hasattr(vtk, 'vtkGarbageCollector'):
                    vtk.vtkGarbageCollector.CollectGarbage()
                    
            except Exception as e:
                self.logger.warning(f"Error in global VTK cleanup: {e}")


class GPUResourceManager:
    """Main GPU resource management class."""
    
    def __init__(self, config_manager=None):
        self.logger = logging.getLogger(__name__)
        
        # Import config here to avoid circular imports
        if config_manager is None:
            from gpu_config import CONFIG
            self.config = CONFIG
        else:
            self.config = config_manager
        
        # Initialize memory and plotter managers
        gpu_config = self.config.gpu_config
        
        # CUDA memory pool
        vram_size_mb = gpu_config.get('vram_pool_size_mb', 1024)
        self.cuda_pool = CUDAMemoryPool(
            initial_size_mb=min(1024, vram_size_mb // 4),
            max_size_mb=vram_size_mb
        )
        
        # PyVista plotter manager
        max_concurrent = gpu_config.get('max_concurrent_renders', 2)
        self.pyvista_manager = PyVistaGPUManager(
            max_plotters=max_concurrent,
            plotter_reuse_limit=50
        )
        
        # Process cleanup registry
        self._cleanup_hooks = []
        
        # Register cleanup on process exit
        import atexit
        atexit.register(self.cleanup_all)
    
    @contextmanager
    def gpu_context(self, device_id: int = 0):
        """Context manager for GPU operations."""
        try:
            with self.cuda_pool.cuda_context(device_id):
                yield self
        finally:
            # Cleanup after operation
            self.cuda_pool.free_memory(device_id)
            gc.collect()
    
    @contextmanager
    def get_plotter(self, **kwargs):
        """Get a managed PyVista plotter."""
        with self.pyvista_manager.get_plotter(**kwargs) as plotter:
            yield plotter
    
    def allocate_gpu_array(self, shape: tuple, dtype=np.float32, device_id: int = 0):
        """Allocate GPU array through the memory pool."""
        return self.cuda_pool.allocate_gpu_array(shape, dtype, device_id)
    
    def transfer_to_gpu(self, cpu_array: np.ndarray, device_id: int = 0):
        """Transfer CPU array to GPU."""
        if CUPY_AVAILABLE:
            with self.cuda_pool.cuda_context(device_id):
                return cp.asarray(cpu_array)
        elif TORCH_AVAILABLE:
            return torch.from_numpy(cpu_array).cuda(device_id)
        else:
            return cpu_array  # Fallback to CPU
    
    def transfer_to_cpu(self, gpu_array) -> np.ndarray:
        """Transfer GPU array to CPU."""
        if CUPY_AVAILABLE and isinstance(gpu_array, cp.ndarray):
            return cp.asnumpy(gpu_array)
        elif TORCH_AVAILABLE and isinstance(gpu_array, torch.Tensor):
            return gpu_array.cpu().numpy()
        else:
            return gpu_array  # Already CPU array
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        stats = {
            'cuda_available': CUPY_AVAILABLE or TORCH_AVAILABLE,
            'cuda_memory': self.cuda_pool.get_memory_info(),
            'active_plotters': len(self.pyvista_manager._plotter_pool),
            'plotter_usage': dict(self.pyvista_manager._plotter_usage),
        }
        
        # Add PyVista memory info if available
        try:
            import psutil
            process = psutil.Process()
            stats['process_memory_mb'] = process.memory_info().rss / 1024 / 1024
        except Exception:
            pass
        
        return stats
    
    def cleanup_all(self):
        """Clean up all GPU resources."""
        self.logger.info("Cleaning up GPU resources")
        
        # Run custom cleanup hooks
        for hook in self._cleanup_hooks:
            try:
                hook()
            except Exception as e:
                self.logger.warning(f"Error in cleanup hook: {e}")
        
        # Clean up managers
        self.pyvista_manager.cleanup_all()
        self.cuda_pool.free_memory()
        
        # Final garbage collection
        gc.collect()
    
    def register_cleanup_hook(self, cleanup_func):
        """Register a function to be called during cleanup."""
        self._cleanup_hooks.append(cleanup_func)


# Global GPU resource manager
_gpu_manager: Optional[GPUResourceManager] = None
_manager_lock = Lock()

def get_gpu_manager() -> GPUResourceManager:
    """Get the global GPU resource manager (singleton)."""
    global _gpu_manager
    
    if _gpu_manager is None:
        with _manager_lock:
            if _gpu_manager is None:
                _gpu_manager = GPUResourceManager()
    
    return _gpu_manager

def cleanup_gpu_resources():
    """Clean up all GPU resources."""
    global _gpu_manager
    
    if _gpu_manager is not None:
        _gpu_manager.cleanup_all()
        _gpu_manager = None 