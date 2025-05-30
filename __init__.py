"""
GPU-accelerated CAD processing pipeline.

This package provides GPU-optimized implementations for:
- Point cloud generation from STL meshes
- Memory management for CUDA operations
- Hardware detection and configuration
- Multi-view rendering (coming in Phase 2)

Example usage:
    from multithread_gpu import stl_to_pointcloud_gpu, CONFIG
    
    # Generate point cloud
    points = stl_to_pointcloud_gpu("mesh.stl", n_points=8192)
    
    # Check GPU configuration
    CONFIG.log_configuration()
"""

# Core configuration and memory management
from .gpu_config import CONFIG, GPUConfigManager, GPUInfo, SystemSpecs
from .gpu_memory import (
    get_gpu_manager, 
    cleanup_gpu_resources,
    GPUResourceManager,
    CUDAMemoryPool,
    PyVistaGPUManager
)

# Point cloud generation
from .gpu_pointclouds import (
    stl_to_pointcloud_gpu,
    batch_stl_to_pointclouds_gpu,
    GPUPointCloudGenerator
)

# GPU rendering (Phase 2)
from .gpu_render import (
    render_stl_gpu,
    batch_render_stls_gpu,
    GPUBatchRenderer,
    GPUCanonicalViews,
    GPULightingSystem
)

# GPU workers and pipeline
from .gpu_workers import (
    TaskResult,
    GPUWorkerManager,
    stl_generation_worker_gpu,
    pointcloud_generation_worker_gpu,
    rendering_worker_gpu,
    batch_pointcloud_worker_gpu,
    batch_rendering_worker_gpu,
    full_pipeline_worker_gpu,
    process_cadquery_file_gpu,
    process_stl_file_gpu
)

__version__ = "1.0.0"
__author__ = "CS190A GPU Optimization Team"

# Convenience aliases
generate_pointcloud = stl_to_pointcloud_gpu
batch_generate_pointclouds = batch_stl_to_pointclouds_gpu
render_stl = render_stl_gpu
batch_render_stls = batch_render_stls_gpu
process_cadquery_file = process_cadquery_file_gpu
process_stl_file = process_stl_file_gpu

def get_gpu_info():
    """Get current GPU configuration information."""
    return CONFIG.get_optimal_settings()

def test_gpu_setup():
    """Quick GPU setup test."""
    try:
        # Test basic functionality
        gpu_manager = get_gpu_manager()
        stats = gpu_manager.get_memory_stats()
        
        print(f"✓ GPU Manager: {'Available' if stats['cuda_available'] else 'CPU-only'}")
        print(f"✓ Configuration: {len(CONFIG.system_specs.gpus)} GPU(s) detected")
        
        if CONFIG.system_specs.gpus:
            primary = CONFIG.system_specs.gpus[0]
            print(f"✓ Primary GPU: {primary.name} ({primary.memory_total}MB VRAM)")
        
        return True
        
    except Exception as e:
        print(f"✗ GPU setup test failed: {e}")
        return False

def get_performance_summary():
    """Get GPU performance capabilities summary."""
    settings = CONFIG.get_optimal_settings()
    
    summary = {
        'gpu_available': settings['gpu']['cuda_available'],
        'max_concurrent_renders': settings['gpu'].get('max_concurrent_renders', 1),
        'batch_size_renders': settings['gpu'].get('batch_size_renders', 1),
        'batch_size_pointclouds': settings['gpu'].get('batch_size_pointclouds', 1),
        'vram_pool_mb': settings['gpu'].get('vram_pool_size_mb', 0),
    }
    
    if settings['gpus']:
        primary_gpu = settings['gpus'][0]
        summary.update({
            'gpu_name': primary_gpu['name'],
            'gpu_memory_total_mb': primary_gpu['memory_total_mb'],
            'is_rtx_a6000': primary_gpu['is_rtx_a6000'],
            'cuda_cores': primary_gpu.get('cuda_cores'),
        })
    
    return summary

# Auto-test on import (can be disabled by setting environment variable)
import os
if os.environ.get("SKIP_GPU_TEST", "").lower() not in ("1", "true", "yes"):
    try:
        # Quick validation that imports work
        _ = CONFIG.system_specs.cpu_count
    except Exception:
        print("Warning: GPU configuration failed to initialize. Run test_gpu_setup.py for details.")

__all__ = [
    # Configuration
    "CONFIG", "GPUConfigManager", "GPUInfo", "SystemSpecs",
    # Memory management  
    "get_gpu_manager", "cleanup_gpu_resources", "GPUResourceManager",
    "CUDAMemoryPool", "PyVistaGPUManager",
    # Point cloud generation
    "stl_to_pointcloud_gpu", "batch_stl_to_pointclouds_gpu", "GPUPointCloudGenerator",
    # GPU rendering
    "render_stl_gpu", "batch_render_stls_gpu", "GPUBatchRenderer",
    "GPUCanonicalViews", "GPULightingSystem",
    # GPU workers and pipeline
    "TaskResult", "GPUWorkerManager",
    "stl_generation_worker_gpu", "pointcloud_generation_worker_gpu", "rendering_worker_gpu",
    "batch_pointcloud_worker_gpu", "batch_rendering_worker_gpu", "full_pipeline_worker_gpu",
    "process_cadquery_file_gpu", "process_stl_file_gpu",
    # Convenience functions
    "generate_pointcloud", "batch_generate_pointclouds", 
    "render_stl", "batch_render_stls",
    "process_cadquery_file", "process_stl_file",
    "get_gpu_info", "test_gpu_setup", "get_performance_summary",
] 