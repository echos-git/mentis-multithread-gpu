#!/usr/bin/env python3
"""
GPU setup test script for CAD processing pipeline.
Tests CUDA availability, GPU memory, and basic functionality.
"""

import sys
import logging
import traceback
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_basic_imports():
    """Test basic Python imports."""
    print("=" * 60)
    print("TESTING BASIC IMPORTS")
    print("=" * 60)
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
        
        import psutil
        print(f"✓ psutil {psutil.__version__}")
        
        import trimesh
        print(f"✓ trimesh {trimesh.__version__}")
        
        import pyvista as pv
        print(f"✓ PyVista {pv.__version__}")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_gpu_frameworks():
    """Test GPU framework availability."""
    print("\n" + "=" * 60)
    print("TESTING GPU FRAMEWORKS")
    print("=" * 60)
    
    cupy_available = False
    torch_available = False
    
    # Test CuPy
    try:
        import cupy as cp
        print(f"✓ CuPy {cp.__version__}")
        
        # Test basic CUDA operation
        test_array = cp.array([1, 2, 3, 4, 5])
        result = cp.sum(test_array)
        print(f"✓ CuPy CUDA test: sum([1,2,3,4,5]) = {result}")
        cupy_available = True
        
    except ImportError:
        print("✗ CuPy not available")
    except Exception as e:
        print(f"✗ CuPy error: {e}")
    
    # Test PyTorch
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ PyTorch CUDA available, devices: {torch.cuda.device_count()}")
            
            # Test basic CUDA operation
            test_tensor = torch.tensor([1, 2, 3, 4, 5]).cuda()
            result = torch.sum(test_tensor)
            print(f"✓ PyTorch CUDA test: sum([1,2,3,4,5]) = {result}")
            torch_available = True
        else:
            print("✗ PyTorch CUDA not available")
            
    except ImportError:
        print("✗ PyTorch not available")
    except Exception as e:
        print(f"✗ PyTorch error: {e}")
    
    return cupy_available or torch_available

def test_gpu_detection():
    """Test GPU detection and configuration."""
    print("\n" + "=" * 60)
    print("TESTING GPU DETECTION")
    print("=" * 60)
    
    try:
        from gpu_config import CONFIG
        
        # Log configuration
        CONFIG.log_configuration()
        
        # Get settings
        settings = CONFIG.get_optimal_settings()
        
        print(f"\nConfiguration Summary:")
        print(f"CUDA Available: {settings['gpu']['cuda_available']}")
        print(f"GPUs detected: {len(settings['gpus'])}")
        
        if settings['gpus']:
            primary_gpu = settings['gpus'][0]
            print(f"Primary GPU: {primary_gpu['name']}")
            print(f"VRAM: {primary_gpu['memory_total_mb']}MB")
            print(f"Compute Capability: {primary_gpu['compute_capability']}")
            
            if primary_gpu['is_rtx_a6000']:
                print("✓ RTX A6000 detected - optimal configuration applied")
        
        return True
        
    except Exception as e:
        print(f"✗ GPU configuration error: {e}")
        traceback.print_exc()
        return False

def test_gpu_memory_management():
    """Test GPU memory management."""
    print("\n" + "=" * 60)
    print("TESTING GPU MEMORY MANAGEMENT")
    print("=" * 60)
    
    try:
        from gpu_memory import get_gpu_manager
        
        gpu_manager = get_gpu_manager()
        
        # Test memory stats
        stats = gpu_manager.get_memory_stats()
        print(f"GPU Manager initialized: {stats['cuda_available']}")
        
        if stats['cuda_available']:
            print(f"CUDA Memory stats: {stats['cuda_memory']}")
            
            # Test GPU context
            with gpu_manager.gpu_context() as gm:
                print("✓ GPU context manager working")
                
                # Test GPU array allocation
                test_array = gm.allocate_gpu_array((1000, 3))
                print(f"✓ GPU array allocation: shape {test_array.shape}")
        
        # Test PyVista plotter management
        with gpu_manager.get_plotter() as plotter:
            print("✓ PyVista plotter management working")
        
        return True
        
    except Exception as e:
        print(f"✗ GPU memory management error: {e}")
        traceback.print_exc()
        return False

def test_point_cloud_generation():
    """Test GPU point cloud generation (requires sample STL)."""
    print("\n" + "=" * 60)
    print("TESTING POINT CLOUD GENERATION")
    print("=" * 60)
    
    try:
        from gpu_pointclouds import GPUPointCloudGenerator
        import numpy as np
        
        # Create a simple test mesh
        test_stl_content = create_test_stl()
        test_stl_path = Path("test_cube.stl")
        
        # Write test STL
        with open(test_stl_path, 'w') as f:
            f.write(test_stl_content)
        
        try:
            # Test GPU point cloud generation
            generator = GPUPointCloudGenerator()
            
            print(f"GPU acceleration available: {generator.use_gpu}")
            
            # Generate point cloud
            points = generator.stl_to_pointcloud_gpu(test_stl_path, n_points=1000)
            
            print(f"✓ Point cloud generated: {points.shape}")
            print(f"  Points range: X=[{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
            print(f"  Points range: Y=[{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
            print(f"  Points range: Z=[{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
            
            return True
            
        finally:
            # Clean up test file
            if test_stl_path.exists():
                test_stl_path.unlink()
                
    except Exception as e:
        print(f"✗ Point cloud generation error: {e}")
        traceback.print_exc()
        return False

def create_test_stl():
    """Create a simple STL file content for testing."""
    return """solid cube
  facet normal 0.0 0.0 1.0
    outer loop
      vertex 0.0 0.0 1.0
      vertex 1.0 0.0 1.0
      vertex 1.0 1.0 1.0
    endloop
  endfacet
  facet normal 0.0 0.0 1.0
    outer loop
      vertex 0.0 0.0 1.0
      vertex 1.0 1.0 1.0
      vertex 0.0 1.0 1.0
    endloop
  endfacet
  facet normal 0.0 0.0 -1.0
    outer loop
      vertex 0.0 0.0 0.0
      vertex 1.0 1.0 0.0
      vertex 1.0 0.0 0.0
    endloop
  endfacet
  facet normal 0.0 0.0 -1.0
    outer loop
      vertex 0.0 0.0 0.0
      vertex 0.0 1.0 0.0
      vertex 1.0 1.0 0.0
    endloop
  endfacet
endsolid cube"""

def test_system_resources():
    """Test system resource detection."""
    print("\n" + "=" * 60)
    print("SYSTEM RESOURCE INFORMATION")
    print("=" * 60)
    
    try:
        import psutil
        import platform
        
        print(f"Platform: {platform.platform()}")
        print(f"Python: {platform.python_version()}")
        print(f"CPU cores: {psutil.cpu_count()}")
        print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f}GB")
        
        # GPU information via nvidia-smi if available
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                                   capture_output=True, text=True, check=True)
            print(f"GPU Info (nvidia-smi):")
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    print(f"  {line.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("nvidia-smi not available")
        
        return True
        
    except Exception as e:
        print(f"Error getting system info: {e}")
        return False

def main():
    """Run all tests."""
    print("GPU CAD Processing Pipeline Test Suite")
    print("This script tests GPU setup and basic functionality")
    print()
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("System Resources", test_system_resources),
        ("GPU Frameworks", test_gpu_frameworks),
        ("GPU Detection", test_gpu_detection),
        ("GPU Memory Management", test_gpu_memory_management),
        ("Point Cloud Generation", test_point_cloud_generation),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:<30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your GPU setup is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 