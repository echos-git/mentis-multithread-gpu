"""
GPU-optimized worker functions for CAD processing pipeline.
Integrates STL generation, point cloud generation, and rendering.
"""

import os
import gc
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import numpy as np

from gpu_config import CONFIG
from gpu_memory import get_gpu_manager, cleanup_gpu_resources
from gpu_pointclouds import GPUPointCloudGenerator
from gpu_render import GPUBatchRenderer


@dataclass
class TaskResult:
    """Result of a worker task execution."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    worker_pid: Optional[int] = None
    memory_usage_mb: Optional[float] = None
    execution_time_s: Optional[float] = None


class GPUWorkerManager:
    """Manages GPU workers for different processing tasks."""
    
    def __init__(self, device_id: int = 0):
        self.logger = logging.getLogger(__name__)
        self.device_id = device_id
        self.gpu_manager = get_gpu_manager()
        
        # Initialize specialized processors
        self.pointcloud_generator = GPUPointCloudGenerator(device_id=device_id)
        self.batch_renderer = GPUBatchRenderer(device_id=device_id)
        
        # Performance tracking
        self.stats = {
            'stl_generated': 0,
            'pointclouds_generated': 0,
            'views_rendered': 0,
            'total_time': 0.0,
            'errors': 0
        }
    
    def cleanup(self):
        """Clean up all GPU resources."""
        try:
            self.batch_renderer.cleanup()
            cleanup_gpu_resources()
            gc.collect()
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")


def init_gpu_worker_process():
    """Initialize a worker process for GPU operations."""
    try:
        # Set up logging for worker
        logging.basicConfig(
            level=logging.WARNING,  # Reduce log noise from workers
            format=f"%(asctime)s - GPU-WORKER-{os.getpid()} - %(levelname)s - %(message)s"
        )
        
        # Initialize GPU manager
        _ = get_gpu_manager()
        
        # Worker-specific setup
        logger = logging.getLogger(__name__)
        logger.info(f"GPU worker process {os.getpid()} initialized")
        
    except Exception as e:
        logging.error(f"Failed to initialize GPU worker process: {e}")


def cleanup_gpu_worker_process():
    """Clean up worker process before exit."""
    try:
        cleanup_gpu_resources()
        gc.collect()
    except Exception as e:
        logging.warning(f"Error cleaning up GPU worker process: {e}")


def stl_generation_worker_gpu(cadquery_file_path: str, stl_output_path: str) -> TaskResult:
    """
    GPU-optimized STL generation worker.
    Note: STL generation is inherently CPU-bound, but we optimize memory management.
    """
    start_time = time.time()
    
    try:
        init_gpu_worker_process()
        
        # Import here to avoid import issues in multiprocessing
        from cadquerytostl import cq_to_stl  # Import from local module
        
        # Ensure output directory exists
        Path(stl_output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert CadQuery to STL
        cq_to_stl(cadquery_file_path, stl_output_path)
        
        # Verify output was created
        if not Path(stl_output_path).exists():
            raise FileNotFoundError(f"STL file was not created: {stl_output_path}")
        
        execution_time = time.time() - start_time
        
        return TaskResult(
            success=True,
            data={
                'stl_path': str(Path(stl_output_path).resolve()),
                'input_path': cadquery_file_path
            },
            worker_pid=os.getpid(),
            execution_time_s=execution_time
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        return TaskResult(
            success=False,
            error=f"STL generation failed: {str(e)}",
            data={'input_path': cadquery_file_path, 'traceback': traceback.format_exc()},
            worker_pid=os.getpid(),
            execution_time_s=execution_time
        )
    finally:
        cleanup_gpu_worker_process()


def pointcloud_generation_worker_gpu(stl_file_path: str, 
                                    pointcloud_output_path: str,
                                    n_points: int = 8192,
                                    device_id: int = 0) -> TaskResult:
    """GPU-accelerated point cloud generation worker."""
    start_time = time.time()
    
    try:
        init_gpu_worker_process()
        
        # Ensure output directory exists
        Path(pointcloud_output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Generate point cloud using GPU acceleration
        generator = GPUPointCloudGenerator(device_id=device_id)
        points = generator.stl_to_pointcloud_gpu(stl_file_path, n_points=n_points)
        
        # Save point cloud
        np.save(pointcloud_output_path, points)
        
        # Verify output was created
        if not Path(pointcloud_output_path).exists():
            raise FileNotFoundError(f"Point cloud file was not created: {pointcloud_output_path}")
        
        execution_time = time.time() - start_time
        
        # Get memory stats
        gpu_manager = get_gpu_manager()
        memory_stats = gpu_manager.get_memory_stats()
        memory_usage = memory_stats.get('process_memory_mb', 0)
        
        return TaskResult(
            success=True,
            data={
                'pointcloud_path': str(Path(pointcloud_output_path).resolve()),
                'stl_path': stl_file_path,
                'n_points': len(points),
                'gpu_acceleration': generator.use_gpu
            },
            worker_pid=os.getpid(),
            memory_usage_mb=memory_usage,
            execution_time_s=execution_time
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        return TaskResult(
            success=False,
            error=f"Point cloud generation failed: {str(e)}",
            data={'stl_path': stl_file_path, 'traceback': traceback.format_exc()},
            worker_pid=os.getpid(),
            execution_time_s=execution_time
        )
    finally:
        cleanup_gpu_worker_process()


def rendering_worker_gpu(stl_file_path: str,
                        renders_base_dir: str,
                        use_global_lighting: bool = False,
                        force_overwrite: bool = False,
                        device_id: int = 0) -> TaskResult:
    """GPU-accelerated multi-view rendering worker."""
    start_time = time.time()
    
    try:
        init_gpu_worker_process()
        
        # Render using GPU acceleration
        renderer = GPUBatchRenderer(device_id=device_id)
        rendered_paths = renderer.render_stl_multiview_gpu(
            stl_file_path, renders_base_dir, use_global_lighting, force_overwrite
        )
        
        if not rendered_paths:
            raise RuntimeError("No rendered images were generated")
        
        # Check for failed renderings
        failed_views = [
            view for view, path in rendered_paths.items() 
            if "FAILED_RENDERING" in str(path)
        ]
        
        success = len(failed_views) == 0
        execution_time = time.time() - start_time
        
        # Get performance stats
        render_stats = renderer.get_rendering_stats()
        
        # Get memory stats
        gpu_manager = get_gpu_manager()
        memory_stats = gpu_manager.get_memory_stats()
        memory_usage = memory_stats.get('process_memory_mb', 0)
        
        return TaskResult(
            success=success,
            data={
                'rendered_paths': rendered_paths,
                'stl_path': stl_file_path,
                'failed_views': failed_views,
                'render_stats': render_stats,
                'gpu_acceleration': True
            },
            worker_pid=os.getpid(),
            memory_usage_mb=memory_usage,
            execution_time_s=execution_time
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        return TaskResult(
            success=False,
            error=f"Rendering failed: {str(e)}",
            data={'stl_path': stl_file_path, 'traceback': traceback.format_exc()},
            worker_pid=os.getpid(),
            execution_time_s=execution_time
        )
    finally:
        cleanup_gpu_worker_process()


def batch_pointcloud_worker_gpu(stl_file_paths: List[str],
                               pointcloud_output_dir: str,
                               n_points: int = 8192,
                               device_id: int = 0) -> List[TaskResult]:
    """Batch GPU-accelerated point cloud generation worker."""
    start_time = time.time()
    
    try:
        init_gpu_worker_process()
        
        # Ensure output directory exists
        Path(pointcloud_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate point clouds in batch
        generator = GPUPointCloudGenerator(device_id=device_id)
        batch_size = CONFIG.gpu_config.get('batch_size_pointclouds', 8)
        
        results = []
        
        for stl_path in stl_file_paths:
            try:
                stl_name = Path(stl_path).stem
                output_path = Path(pointcloud_output_dir) / f"{stl_name}.npy"
                
                # Generate point cloud
                points = generator.stl_to_pointcloud_gpu(stl_path, n_points=n_points)
                
                # Save point cloud
                np.save(str(output_path), points)
                
                results.append(TaskResult(
                    success=True,
                    data={
                        'pointcloud_path': str(output_path.resolve()),
                        'stl_path': stl_path,
                        'n_points': len(points)
                    }
                ))
                
            except Exception as e:
                results.append(TaskResult(
                    success=False,
                    error=f"Failed to process {stl_path}: {str(e)}",
                    data={'stl_path': stl_path}
                ))
        
        execution_time = time.time() - start_time
        
        # Update execution time for all results
        for result in results:
            result.execution_time_s = execution_time / len(results)
            result.worker_pid = os.getpid()
        
        return results
        
    except Exception as e:
        execution_time = time.time() - start_time
        return [TaskResult(
            success=False,
            error=f"Batch point cloud generation failed: {str(e)}",
            data={'stl_paths': stl_file_paths, 'traceback': traceback.format_exc()},
            worker_pid=os.getpid(),
            execution_time_s=execution_time
        )]
    finally:
        cleanup_gpu_worker_process()


def batch_rendering_worker_gpu(stl_file_paths: List[str],
                              renders_base_dir: str,
                              use_global_lighting: bool = False,
                              force_overwrite: bool = False,
                              device_id: int = 0) -> List[TaskResult]:
    """Batch GPU-accelerated rendering worker."""
    start_time = time.time()
    
    try:
        init_gpu_worker_process()
        
        # Render batch using GPU acceleration
        renderer = GPUBatchRenderer(device_id=device_id)
        batch_results = renderer.batch_render_stls_gpu(
            stl_file_paths, renders_base_dir, use_global_lighting, force_overwrite
        )
        
        results = []
        for stl_path in stl_file_paths:
            stl_name = Path(stl_path).stem
            rendered_paths = batch_results.get(stl_name, {})
            
            # Check for failed renderings
            failed_views = [
                view for view, path in rendered_paths.items() 
                if "FAILED_RENDERING" in str(path)
            ]
            
            success = len(rendered_paths) > 0 and len(failed_views) == 0
            
            results.append(TaskResult(
                success=success,
                data={
                    'rendered_paths': rendered_paths,
                    'stl_path': stl_path,
                    'failed_views': failed_views
                }
            ))
        
        execution_time = time.time() - start_time
        
        # Update execution time for all results
        for result in results:
            result.execution_time_s = execution_time / len(results)
            result.worker_pid = os.getpid()
        
        return results
        
    except Exception as e:
        execution_time = time.time() - start_time
        return [TaskResult(
            success=False,
            error=f"Batch rendering failed: {str(e)}",
            data={'stl_paths': stl_file_paths, 'traceback': traceback.format_exc()},
            worker_pid=os.getpid(),
            execution_time_s=execution_time
        )]
    finally:
        cleanup_gpu_worker_process()


def full_pipeline_worker_gpu(cadquery_file_path: str,
                            output_base_dir: str,
                            n_points: int = 8192,
                            use_global_lighting: bool = False,
                            force_overwrite: bool = False,
                            device_id: int = 0) -> TaskResult:
    """
    Complete GPU-accelerated pipeline worker: CadQuery -> STL -> Point Cloud -> Renders.
    """
    start_time = time.time()
    pipeline_results = {}
    
    try:
        init_gpu_worker_process()
        
        # Setup paths
        cq_name = Path(cadquery_file_path).stem
        output_dir = Path(output_base_dir)
        stl_dir = output_dir / "stls"
        pointcloud_dir = output_dir / "pointclouds"
        renders_dir = output_dir / "renders"
        
        # Create directories
        for dir_path in [stl_dir, pointcloud_dir, renders_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Generate STL
        stl_path = stl_dir / f"{cq_name}.stl"
        stl_result = stl_generation_worker_gpu(cadquery_file_path, str(stl_path))
        pipeline_results['stl'] = stl_result
        
        if not stl_result.success:
            raise RuntimeError(f"STL generation failed: {stl_result.error}")
        
        # Step 2: Generate Point Cloud
        pointcloud_path = pointcloud_dir / f"{cq_name}.npy"
        pc_result = pointcloud_generation_worker_gpu(
            str(stl_path), str(pointcloud_path), n_points, device_id
        )
        pipeline_results['pointcloud'] = pc_result
        
        # Step 3: Render Views (continue even if point cloud fails)
        render_result = rendering_worker_gpu(
            str(stl_path), str(renders_dir), use_global_lighting, force_overwrite, device_id
        )
        pipeline_results['rendering'] = render_result
        
        # Determine overall success
        success = stl_result.success and render_result.success
        # Point cloud failure is not fatal for the pipeline
        
        execution_time = time.time() - start_time
        
        return TaskResult(
            success=success,
            data={
                'cadquery_path': cadquery_file_path,
                'pipeline_results': pipeline_results,
                'stl_path': str(stl_path) if stl_result.success else None,
                'pointcloud_path': str(pointcloud_path) if pc_result.success else None,
                'rendered_views': render_result.data.get('rendered_paths', {}) if render_result.success else {}
            },
            worker_pid=os.getpid(),
            execution_time_s=execution_time
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        return TaskResult(
            success=False,
            error=f"Full pipeline failed: {str(e)}",
            data={
                'cadquery_path': cadquery_file_path,
                'pipeline_results': pipeline_results,
                'traceback': traceback.format_exc()
            },
            worker_pid=os.getpid(),
            execution_time_s=execution_time
        )
    finally:
        cleanup_gpu_worker_process()


# Convenience wrapper functions
def process_cadquery_file_gpu(cadquery_file_path: str,
                             output_base_dir: str,
                             n_points: int = 8192,
                             use_global_lighting: bool = False,
                             force_overwrite: bool = False) -> TaskResult:
    """Convenience function to process a single CadQuery file through the full GPU pipeline."""
    return full_pipeline_worker_gpu(
        cadquery_file_path, output_base_dir, n_points, 
        use_global_lighting, force_overwrite
    )


def process_stl_file_gpu(stl_file_path: str,
                        output_base_dir: str,
                        n_points: int = 8192,
                        use_global_lighting: bool = False,
                        force_overwrite: bool = False) -> Dict[str, TaskResult]:
    """Convenience function to process a single STL file (point cloud + renders)."""
    stl_name = Path(stl_file_path).stem
    output_dir = Path(output_base_dir)
    pointcloud_dir = output_dir / "pointclouds"
    renders_dir = output_dir / "renders"
    
    # Create directories
    pointcloud_dir.mkdir(parents=True, exist_ok=True)
    renders_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Generate point cloud
    pointcloud_path = pointcloud_dir / f"{stl_name}.npy"
    results['pointcloud'] = pointcloud_generation_worker_gpu(
        stl_file_path, str(pointcloud_path), n_points
    )
    
    # Render views
    results['rendering'] = rendering_worker_gpu(
        stl_file_path, str(renders_dir), use_global_lighting, force_overwrite
    )
    
    return results 