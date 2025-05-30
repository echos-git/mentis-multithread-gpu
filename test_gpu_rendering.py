#!/usr/bin/env python3
"""
GPU rendering test script for Phase 2 validation.
Tests multi-view rendering, batch processing, and GPU acceleration.
"""

import sys
import logging
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_gpu_rendering_imports():
    """Test GPU rendering module imports."""
    print("=" * 60)
    print("TESTING GPU RENDERING IMPORTS")
    print("=" * 60)
    
    try:
        from gpu_render import (
            GPUBatchRenderer, GPUCanonicalViews, GPULightingSystem,
            render_stl_gpu, batch_render_stls_gpu
        )
        print("✓ GPU rendering imports successful")
        
        from gpu_workers import (
            TaskResult, GPUWorkerManager, rendering_worker_gpu,
            batch_rendering_worker_gpu, full_pipeline_worker_gpu
        )
        print("✓ GPU workers imports successful")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_canonical_views():
    """Test canonical viewpoint calculations."""
    print("\n" + "=" * 60)
    print("TESTING CANONICAL VIEWPOINTS")
    print("=" * 60)
    
    try:
        from gpu_render import GPUCanonicalViews
        
        # Test mesh parameters
        mesh_center = (0.0, 0.0, 0.0)
        mesh_bounds = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        mesh_length = 2.0
        
        # Calculate viewpoints
        viewpoints = GPUCanonicalViews.calculate_viewpoints(
            mesh_center, mesh_bounds, mesh_length
        )
        
        print(f"✓ Generated {len(viewpoints)} canonical viewpoints")
        
        # Verify all expected views are present
        expected_views = {
            'front', 'back', 'left', 'right', 'top', 'bottom',
            'front_left', 'front_right', 'back_left', 'back_right',
            'top_front', 'top_back', 'top_left', 'top_right'
        }
        
        actual_views = set(viewpoints.keys())
        if actual_views == expected_views:
            print("✓ All 14 canonical views generated correctly")
        else:
            missing = expected_views - actual_views
            extra = actual_views - expected_views
            if missing:
                print(f"✗ Missing views: {missing}")
            if extra:
                print(f"✗ Extra views: {extra}")
            return False
        
        # Test a specific viewpoint
        front_view = viewpoints['front']
        camera_pos, focal_point, view_up = front_view
        print(f"✓ Front view: camera={camera_pos}, focal={focal_point}, up={view_up}")
        
        return True
        
    except Exception as e:
        print(f"✗ Canonical views test failed: {e}")
        return False

def test_lighting_system():
    """Test GPU lighting calculations."""
    print("\n" + "=" * 60)
    print("TESTING LIGHTING SYSTEM")
    print("=" * 60)
    
    try:
        from gpu_render import GPULightingSystem
        
        # Test camera parameters
        camera_params = (
            (0.0, -5.0, 0.0),  # camera position
            (0.0, 0.0, 0.0),   # focal point
            (0.0, 0.0, 1.0)    # view up
        )
        
        # Calculate view-specific lights
        lights = GPULightingSystem.calculate_lights_for_view(
            camera_params, GPULightingSystem.LIGHT_INTENSITY
        )
        
        print(f"✓ Generated {len(lights)} lights for view")
        
        if len(lights) != GPULightingSystem.NUM_LIGHTS_PER_VIEW:
            print(f"✗ Expected {GPULightingSystem.NUM_LIGHTS_PER_VIEW} lights, got {len(lights)}")
            return False
        
        # Test global lighting
        all_camera_params = [camera_params] * 3  # Simulate 3 views
        global_lights = GPULightingSystem.calculate_global_lights(all_camera_params)
        
        expected_global_lights = len(all_camera_params) * GPULightingSystem.NUM_LIGHTS_PER_VIEW
        print(f"✓ Generated {len(global_lights)} global lights (expected {expected_global_lights})")
        
        if len(global_lights) != expected_global_lights:
            print(f"✗ Global lighting count mismatch")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Lighting system test failed: {e}")
        return False

def create_test_stl(output_path: Path):
    """Create a test STL file for rendering tests."""
    stl_content = """solid cube
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
  facet normal 1.0 0.0 0.0
    outer loop
      vertex 1.0 0.0 0.0
      vertex 1.0 1.0 1.0
      vertex 1.0 0.0 1.0
    endloop
  endfacet
  facet normal 1.0 0.0 0.0
    outer loop
      vertex 1.0 0.0 0.0
      vertex 1.0 1.0 0.0
      vertex 1.0 1.0 1.0
    endloop
  endfacet
endsolid cube"""
    
    with open(output_path, 'w') as f:
        f.write(stl_content)

def test_single_stl_rendering():
    """Test single STL file GPU rendering."""
    print("\n" + "=" * 60)
    print("TESTING SINGLE STL RENDERING")
    print("=" * 60)
    
    try:
        from gpu_render import render_stl_gpu
        
        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            stl_path = temp_path / "test_cube.stl"
            renders_dir = temp_path / "renders"
            
            # Create test STL
            create_test_stl(stl_path)
            print(f"✓ Created test STL: {stl_path}")
            
            # Test GPU rendering
            start_time = time.time()
            rendered_paths = render_stl_gpu(
                stl_path, renders_dir, use_global_lighting=False, force_overwrite=True
            )
            render_time = time.time() - start_time
            
            print(f"✓ Rendering completed in {render_time:.2f}s")
            print(f"✓ Generated {len(rendered_paths)} view images")
            
            # Check that images were created
            successful_renders = [
                view for view, path in rendered_paths.items()
                if Path(path).exists() and "FAILED_RENDERING" not in str(path)
            ]
            
            print(f"✓ Successful renders: {len(successful_renders)}/{len(rendered_paths)}")
            
            if len(successful_renders) < len(rendered_paths) // 2:
                print(f"✗ Too many failed renders")
                return False
            
            # Test specific views
            if 'front' in successful_renders:
                front_path = Path(rendered_paths['front'])
                if front_path.exists():
                    file_size = front_path.stat().st_size
                    print(f"✓ Front view image: {file_size} bytes")
                else:
                    print("✗ Front view image not found")
                    return False
            
            return True
            
    except Exception as e:
        print(f"✗ Single STL rendering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_rendering():
    """Test batch STL rendering."""
    print("\n" + "=" * 60)
    print("TESTING BATCH STL RENDERING")
    print("=" * 60)
    
    try:
        from gpu_render import batch_render_stls_gpu
        
        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            renders_dir = temp_path / "batch_renders"
            
            # Create multiple test STL files
            stl_paths = []
            for i in range(3):
                stl_path = temp_path / f"test_cube_{i}.stl"
                create_test_stl(stl_path)
                stl_paths.append(str(stl_path))
            
            print(f"✓ Created {len(stl_paths)} test STL files")
            
            # Test batch rendering
            start_time = time.time()
            batch_results = batch_render_stls_gpu(
                stl_paths, renders_dir, use_global_lighting=False, 
                force_overwrite=True, max_concurrent=2
            )
            batch_time = time.time() - start_time
            
            print(f"✓ Batch rendering completed in {batch_time:.2f}s")
            print(f"✓ Processed {len(batch_results)} STL files")
            
            # Check results
            total_views = 0
            successful_stls = 0
            
            for stl_name, rendered_paths in batch_results.items():
                if rendered_paths:
                    successful_renders = [
                        view for view, path in rendered_paths.items()
                        if Path(path).exists() and "FAILED_RENDERING" not in str(path)
                    ]
                    total_views += len(successful_renders)
                    if len(successful_renders) > 0:
                        successful_stls += 1
                    
                    print(f"  {stl_name}: {len(successful_renders)} views rendered")
            
            print(f"✓ Total views rendered: {total_views}")
            print(f"✓ Successful STL files: {successful_stls}/{len(stl_paths)}")
            
            if successful_stls < len(stl_paths) // 2:
                print("✗ Too many failed STL renders")
                return False
            
            return True
            
    except Exception as e:
        print(f"✗ Batch rendering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rendering_worker():
    """Test GPU rendering worker function."""
    print("\n" + "=" * 60)
    print("TESTING RENDERING WORKER")
    print("=" * 60)
    
    try:
        from gpu_workers import rendering_worker_gpu
        
        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            stl_path = temp_path / "test_cube.stl"
            renders_dir = temp_path / "worker_renders"
            
            # Create test STL
            create_test_stl(stl_path)
            print(f"✓ Created test STL: {stl_path}")
            
            # Test worker function
            start_time = time.time()
            result = rendering_worker_gpu(
                str(stl_path), str(renders_dir), 
                use_global_lighting=False, force_overwrite=True
            )
            worker_time = time.time() - start_time
            
            print(f"✓ Worker completed in {worker_time:.2f}s")
            print(f"✓ Worker success: {result.success}")
            print(f"✓ Worker PID: {result.worker_pid}")
            print(f"✓ Execution time: {result.execution_time_s:.2f}s")
            
            if not result.success:
                print(f"✗ Worker failed: {result.error}")
                return False
            
            # Check worker data
            if result.data:
                rendered_paths = result.data.get('rendered_paths', {})
                failed_views = result.data.get('failed_views', [])
                
                print(f"✓ Rendered paths: {len(rendered_paths)}")
                print(f"✓ Failed views: {len(failed_views)}")
                
                if len(failed_views) > len(rendered_paths) // 2:
                    print("✗ Too many failed views in worker")
                    return False
            
            return True
            
    except Exception as e:
        print(f"✗ Rendering worker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_benchmark():
    """Run a performance benchmark for GPU rendering."""
    print("\n" + "=" * 60)
    print("TESTING PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    try:
        from gpu_render import GPUBatchRenderer
        from gpu_config import CONFIG
        
        # Show GPU configuration
        settings = CONFIG.get_optimal_settings()
        gpu_info = settings.get('gpu', {})
        
        print(f"GPU Available: {gpu_info.get('cuda_available', False)}")
        print(f"Max Concurrent Renders: {gpu_info.get('max_concurrent_renders', 1)}")
        print(f"Batch Size Renders: {gpu_info.get('batch_size_renders', 1)}")
        
        if settings.get('gpus'):
            primary_gpu = settings['gpus'][0]
            print(f"Primary GPU: {primary_gpu['name']}")
            print(f"VRAM: {primary_gpu['memory_total_mb']}MB")
        
        # Create renderer and check stats
        renderer = GPUBatchRenderer()
        
        try:
            # Create test files for benchmark
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                stl_paths = []
                
                # Create multiple test STL files
                for i in range(2):  # Small benchmark
                    stl_path = temp_path / f"bench_cube_{i}.stl"
                    create_test_stl(stl_path)
                    stl_paths.append(str(stl_path))
                
                renders_dir = temp_path / "benchmark_renders"
                
                # Run benchmark
                start_time = time.time()
                batch_results = renderer.batch_render_stls_gpu(
                    stl_paths, renders_dir, force_overwrite=True
                )
                benchmark_time = time.time() - start_time
                
                # Get renderer stats
                render_stats = renderer.get_rendering_stats()
                
                print(f"\nBenchmark Results:")
                print(f"Total time: {benchmark_time:.2f}s")
                print(f"STL files: {len(stl_paths)}")
                print(f"Total views rendered: {render_stats.get('total_rendered', 0)}")
                print(f"Average views/second: {render_stats.get('avg_views_per_second', 0):.2f}")
                print(f"Average time/view: {render_stats.get('avg_time_per_view', 0):.3f}s")
                print(f"Errors: {render_stats.get('errors', 0)}")
                
        finally:
            renderer.cleanup()
        
        return True
        
    except Exception as e:
        print(f"✗ Performance benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all GPU rendering tests."""
    print("GPU Rendering Test Suite (Phase 2)")
    print("Testing GPU-accelerated multi-view rendering capabilities")
    print()
    
    tests = [
        ("GPU Rendering Imports", test_gpu_rendering_imports),
        ("Canonical Views", test_canonical_views),
        ("Lighting System", test_lighting_system),
        ("Single STL Rendering", test_single_stl_rendering),
        ("Batch Rendering", test_batch_rendering),
        ("Rendering Worker", test_rendering_worker),
        ("Performance Benchmark", test_performance_benchmark),
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
        print("\n🎉 All GPU rendering tests passed! Phase 2 implementation is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 