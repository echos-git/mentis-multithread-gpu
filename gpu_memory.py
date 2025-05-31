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
                self._configure_cupy_memory_pool()
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
    
    def _configure_cupy_memory_pool(self) -> bool:
        """Configure CuPy memory pool for optimal GPU utilization."""
        try:
            import cupy
            
            # Get memory pool
            mempool = cupy.get_default_memory_pool()
            
            # Set memory limit to 85% of available VRAM
            total_memory = cupy.cuda.Device().mem_info[1]  # Total memory
            limit = int(total_memory * 0.85)
            mempool.set_limit(size=limit)
            
            # Try to set growth policy (newer CuPy versions)
            try:
                if hasattr(mempool, 'set_growth_policy'):
                    mempool.set_growth_policy(cupy.cuda.memory.GROWTH_POLICY_LOG)
                else:
                    self.logger.debug("CuPy memory pool growth policy not available (older version)")
            except AttributeError:
                self.logger.debug("CuPy growth policy not supported in this version")
            
            self.logger.info(f"CuPy memory pool configured: {limit / 1024**3:.1f}GB limit")
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize CuPy memory pool: {e}")
            return False
    
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
        # Force plotter_reuse_limit to 0 (or 1) to ensure plotters are closed after each use.
        # This might help with GLX context issues in threaded environments.
        self.plotter_reuse_limit = 0 # Was plotter_reuse_limit
        self.logger.info(f"PyVistaGPUManager initialized with max_plotters={self.max_plotters}, plotter_reuse_limit={self.plotter_reuse_limit}")
        
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
            self._vtk_cleanup()
    
    def _vtk_cleanup(self):
        """Safe VTK cleanup that works across versions."""
        try:
            import vtk
            
            # Try modern VTK cleanup first
            if hasattr(vtk, 'vtkGarbageCollector'):
                gc_class = getattr(vtk, 'vtkGarbageCollector')
                if hasattr(gc_class, 'DeferredCollectionPush'):
                    gc_class.DeferredCollectionPush()
                if hasattr(gc_class, 'DeferredCollectionPop'):
                    gc_class.DeferredCollectionPop()
                
                # Try different garbage collection methods
                if hasattr(gc_class, 'Collect'):
                    gc_class.Collect()
                elif hasattr(gc_class, 'CollectGarbage'):
                    gc_class.CollectGarbage()  # Older versions
                else:
                    self.logger.debug("VTK garbage collection method not found")
            
            # Try render window cleanup
            if hasattr(vtk, 'vtkRenderWindow'):
                rw_class = getattr(vtk, 'vtkRenderWindow')
                if hasattr(rw_class, 'GlobalWarningDisplayOff'):
                    rw_class.GlobalWarningDisplayOff()
            
            self.logger.debug("VTK cleanup completed successfully")
            
        except Exception as e:
            self.logger.warning(f"Error in VTK cleanup: {e}")


class GPUResourceManager:
    """Centralized GPU resource management for the entire application."""
    
    _instance = None
    _initialized = False
    _cleanup_in_progress = False  # Prevent cleanup loops
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = logging.getLogger(__name__)
            self.cuda_pool = CUDAMemoryPool()
            self.pyvista_manager = PyVistaGPUManager()
            self._cleanup_hooks = []
            self._atexit_registered = False
            
            # Register cleanup only once
            if not self._atexit_registered:
                import atexit
                atexit.register(self._safe_cleanup)
                self._atexit_registered = True
            
            self.__class__._initialized = True
            self.logger.info("GPU Resource Manager initialized")
    
    def _safe_cleanup(self):
        """Thread-safe cleanup that prevents loops."""
        if self.__class__._cleanup_in_progress:
            return  # Already cleaning up, avoid loops
        
        self.__class__._cleanup_in_progress = True
        try:
            self.cleanup_all()
        except Exception as e:
            print(f"Warning: Error in final cleanup: {e}")
        finally:
            self.__class__._cleanup_in_progress = False
    
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
        try:
            self.logger.info("Cleaning up GPU resources")
            
            # Clean up PyVista plotters
            self.pyvista_manager.cleanup_all()
            
            # Clean up CUDA memory
            self.cuda_pool.free_memory()
            
            # Safe VTK cleanup
            self.pyvista_manager._vtk_cleanup()
            
        except Exception as e:
            self.logger.warning(f"Error in global VTK cleanup: {e}")
        finally:
            # Always attempt garbage collection
            import gc
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