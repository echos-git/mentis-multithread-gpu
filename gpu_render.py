"""
GPU-accelerated multi-view rendering system for STL meshes.
Optimized for NVIDIA RTX A6000 with batch processing and parallel rendering.
"""

import os
import logging
import math
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time

import pyvista as pv
from pyvista.plotting.lights import LightType

from gpu_config import CONFIG
from gpu_memory import get_gpu_manager

# Constants for lighting system
NUM_LIGHTS_PER_VIEW = 3
THETA_CONE_DEGREES = 55.0
THETA_CONE_RADIANS = np.radians(THETA_CONE_DEGREES)
LIGHT_INTENSITY = 0.5
LIGHTING_MULTIPLIER = 2


class GPUCanonicalViews:
    """Manages canonical viewpoint calculations for 3D meshes."""
    
    # 14 canonical viewpoints as defined in the original system
    CANONICAL_VIEWS = {
        'front': ([0, -1, 0], [0, 0, 0], [0, 0, 1]),
        'back': ([0, 1, 0], [0, 0, 0], [0, 0, 1]),
        'left': ([-1, 0, 0], [0, 0, 0], [0, 0, 1]),
        'right': ([1, 0, 0], [0, 0, 0], [0, 0, 1]),
        'top': ([0, 0, 1], [0, 0, 0], [0, 1, 0]),
        'bottom': ([0, 0, -1], [0, 0, 0], [0, 1, 0]),
        'front_left': ([-1, -1, 0], [0, 0, 0], [0, 0, 1]),
        'front_right': ([1, -1, 0], [0, 0, 0], [0, 0, 1]),
        'back_left': ([-1, 1, 0], [0, 0, 0], [0, 0, 1]),
        'back_right': ([1, 1, 0], [0, 0, 0], [0, 0, 1]),
        'top_front': ([0, -1, 1], [0, 0, 0], [0, 0, 1]),
        'top_back': ([0, 1, 1], [0, 0, 0], [0, 0, 1]),
        'top_left': ([-1, 0, 1], [0, 0, 0], [0, 0, 1]),
        'top_right': ([1, 0, 1], [0, 0, 0], [0, 0, 1]),
    }
    
    @classmethod
    def calculate_viewpoints(cls, mesh_center: Tuple[float, float, float], 
                           mesh_bounds: Tuple[float, ...], 
                           mesh_length: float) -> Dict[str, Tuple]:
        """Calculate camera parameters for all canonical viewpoints."""
        # Calculate optimal view distance
        diag_length = math.sqrt(
            (mesh_bounds[1] - mesh_bounds[0])**2 +
            (mesh_bounds[3] - mesh_bounds[2])**2 +
            (mesh_bounds[5] - mesh_bounds[4])**2
        )
        view_distance = diag_length * 1.8
        
        if view_distance == 0:
            view_distance = mesh_length * 2.5
        if view_distance == 0:
            view_distance = 5.0
        
        focal_point = tuple(c for c in mesh_center)
        
        viewpoints = {}
        for view_name, (direction, _, view_up) in cls.CANONICAL_VIEWS.items():
            # Calculate camera position
            camera_pos = tuple(
                focal_point[i] + direction[i] * view_distance 
                for i in range(3)
            )
            
            viewpoints[view_name] = (camera_pos, focal_point, view_up)
        
        return viewpoints


class GPULightingSystem:
    """GPU-optimized lighting calculations for rendering."""
    
    # Lighting constants from original system
    THETA_CONE_DEGREES = THETA_CONE_DEGREES
    THETA_CONE_RADIANS = THETA_CONE_RADIANS
    LIGHT_INTENSITY = LIGHT_INTENSITY
    NUM_LIGHTS_PER_VIEW = NUM_LIGHTS_PER_VIEW
    LIGHTING_MULTIPLIER = LIGHTING_MULTIPLIER
    
    @classmethod
    def get_azimuth_angles(cls):
        """Get azimuth angles for lighting."""
        return [i * (2 * np.pi / cls.NUM_LIGHTS_PER_VIEW) for i in range(cls.NUM_LIGHTS_PER_VIEW)]
    
    @classmethod
    def calculate_lights_for_view(cls, camera_view_params: Tuple, base_intensity: float) -> List[pv.Light]:
        """Calculate a triad of lights for a given camera view."""
        lights = []
        cam_pos_np = np.array(camera_view_params[0])
        focal_pt_np = np.array(camera_view_params[1])
        view_up_for_camera_np = np.array(camera_view_params[2])
        
        # Calculate coordinate system
        z_vec = focal_pt_np - cam_pos_np
        if np.linalg.norm(z_vec) < 1e-9:
            z_vec = np.array([0, 0, 1])
        else:
            z_vec = z_vec / np.linalg.norm(z_vec)
        
        if np.linalg.norm(view_up_for_camera_np) < 1e-9:
            view_up_for_camera_np = np.array([0, 0, 1])
        else:
            view_up_for_camera_np = view_up_for_camera_np / np.linalg.norm(view_up_for_camera_np)
        
        x_vec = np.cross(view_up_for_camera_np, z_vec)
        if np.linalg.norm(x_vec) < 1e-6:
            if abs(z_vec[0]) > 0.999 and abs(z_vec[1]) < 1e-6 and abs(z_vec[2]) < 1e-6:
                x_vec = np.cross(np.array([0.0, 1.0, 0.0]), z_vec)
            else:
                x_vec = np.cross(np.array([1.0, 0.0, 0.0]), z_vec)
            if np.linalg.norm(x_vec) < 1e-6:
                x_vec = np.cross(np.array([0.0, 0.0, 1.0]), z_vec)
                if np.linalg.norm(x_vec) < 1e-6:
                    x_vec = np.cross(np.array([0.0, 1.0, 0.0]), z_vec)
        x_vec = x_vec / np.linalg.norm(x_vec)
        
        y_vec = np.cross(z_vec, x_vec)
        y_vec = y_vec / np.linalg.norm(y_vec)
        
        # Create lights
        azimuth_angles = cls.get_azimuth_angles()
        for phi_azimuth in azimuth_angles:
            lx_local = np.sin(cls.THETA_CONE_RADIANS) * np.cos(phi_azimuth)
            ly_local = np.sin(cls.THETA_CONE_RADIANS) * np.sin(phi_azimuth)
            lz_local = np.cos(cls.THETA_CONE_RADIANS)
            
            world_light_direction = (x_vec * lx_local +
                                   y_vec * ly_local +
                                   z_vec * lz_local)
            world_light_direction = world_light_direction / np.linalg.norm(world_light_direction)
            
            light = pv.Light()
            light.positional = False
            light.intensity = base_intensity
            light.color = (1.0, 1.0, 1.0)
            light.light_type = LightType.SCENE_LIGHT
            light.position = tuple(-c for c in world_light_direction)
            lights.append(light)
        
        return lights
    
    @classmethod
    def calculate_global_lights(cls, all_camera_params: List[Tuple]) -> List[pv.Light]:
        """Calculate global lighting from all camera viewpoints."""
        all_lights = []
        num_views = len(all_camera_params)
        
        if num_views > 0:
            global_intensity_per_light = (
                (cls.LIGHT_INTENSITY * cls.NUM_LIGHTS_PER_VIEW) / 
                (num_views * cls.NUM_LIGHTS_PER_VIEW) * cls.LIGHTING_MULTIPLIER
            )
            
            for cam_params in all_camera_params:
                lights = cls.calculate_lights_for_view(cam_params, global_intensity_per_light)
                all_lights.extend(lights)
        
        return all_lights


class GPUBatchRenderer:
    """GPU-optimized batch rendering system for STL meshes."""
    
    def __init__(self, device_id: int = 0, image_size: Tuple[int, int] = (1024, 1024)):
        self.logger = logging.getLogger(__name__)
        self.device_id = device_id
        self.image_size = image_size
        self.gpu_manager = get_gpu_manager()
        
        # Get GPU configuration
        self.gpu_config = CONFIG.gpu_config
        self.max_concurrent_renders = self.gpu_config.get('max_concurrent_renders', 2)
        self.batch_size_renders = self.gpu_config.get('batch_size_renders', 2)
        
        # Thread pool for parallel rendering
        self.render_pool = ThreadPoolExecutor(max_workers=self.max_concurrent_renders)
        self._pool_lock = Lock()
        
        # Rendering statistics
        self.stats = {
            'total_rendered': 0,
            'total_time': 0.0,
            'batch_count': 0,
            'errors': 0
        }
    
    def render_stl_multiview_gpu(self, 
                                stl_path: Union[str, Path],
                                output_base_dir: Union[str, Path],
                                use_global_lighting: bool = False,
                                force_overwrite: bool = False,
                                views_to_render: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Render multiple views of an STL file using GPU acceleration.
        
        Args:
            stl_path: Path to STL file
            output_base_dir: Base directory for rendered images
            use_global_lighting: Use global lighting setup
            force_overwrite: Overwrite existing images
            views_to_render: Specific views to render (default: all 14)
            
        Returns:
            Dictionary mapping view names to output image paths
        """
        start_time = time.time()
        
        try:
            # Load mesh
            mesh = pv.read(str(stl_path))
            if mesh is None:
                raise ValueError(f"Failed to load mesh from {stl_path}")
            
            # Setup output directory
            stl_name = Path(stl_path).stem
            output_dir = Path(output_base_dir) / stl_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Calculate viewpoints
            mesh_center = mesh.center
            mesh_bounds = mesh.bounds
            mesh_length = max(mesh.bounds[1] - mesh.bounds[0],
                            mesh.bounds[3] - mesh.bounds[2],
                            mesh.bounds[5] - mesh.bounds[4])
            
            viewpoints = GPUCanonicalViews.calculate_viewpoints(mesh_center, mesh_bounds, mesh_length)
            
            # Filter views if specified
            if views_to_render:
                viewpoints = {k: v for k, v in viewpoints.items() if k in views_to_render}
            
            # Prepare mesh for rendering (compute normals for better visualization)
            mesh.compute_normals(point_normals=True, cell_normals=False, inplace=True)
            
            # Render views in batches
            results = self._render_views_batched(
                mesh, stl_path, viewpoints, output_dir, 
                use_global_lighting, force_overwrite
            )
            
            # Update statistics
            elapsed = time.time() - start_time
            with self._pool_lock:
                self.stats['total_rendered'] += len(results)
                self.stats['total_time'] += elapsed
                self.stats['batch_count'] += 1
            
            self.logger.info(f"Rendered {len(results)} views for {stl_name} in {elapsed:.2f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to render {stl_path}: {e}")
            with self._pool_lock:
                self.stats['errors'] += 1
            return {}
        finally:
            # Cleanup mesh
            if 'mesh' in locals():
                del mesh
            gc.collect()
    
    def batch_render_stls_gpu(self,
                             stl_paths: List[Union[str, Path]],
                             output_base_dir: Union[str, Path],
                             use_global_lighting: bool = False,
                             force_overwrite: bool = False,
                             max_concurrent_stls: Optional[int] = None) -> Dict[str, Dict[str, str]]:
        """
        Render multiple STL files with GPU batch processing.
        
        Args:
            stl_paths: List of STL file paths
            output_base_dir: Base directory for all rendered images
            use_global_lighting: Use global lighting setup
            force_overwrite: Overwrite existing images
            max_concurrent_stls: Maximum STL files to process simultaneously
            
        Returns:
            Dictionary mapping STL names to their rendered view paths
        """
        if max_concurrent_stls is None:
            max_concurrent_stls = min(len(stl_paths), self.max_concurrent_renders)
        
        results = {}
        start_time = time.time()
        
        # Process STL files in concurrent batches
        with ThreadPoolExecutor(max_workers=max_concurrent_stls) as executor:
            # Submit all rendering tasks
            future_to_stl = {
                executor.submit(
                    self.render_stl_multiview_gpu,
                    stl_path, output_base_dir, use_global_lighting, force_overwrite
                ): stl_path for stl_path in stl_paths
            }
            
            # Collect results
            for future in as_completed(future_to_stl):
                stl_path = future_to_stl[future]
                stl_name = Path(stl_path).stem
                
                try:
                    stl_results = future.result()
                    results[stl_name] = stl_results
                except Exception as e:
                    self.logger.error(f"Failed to render {stl_path}: {e}")
                    results[stl_name] = {}
        
        elapsed = time.time() - start_time
        total_views = sum(len(views) for views in results.values())
        
        self.logger.info(f"Batch rendered {total_views} views from {len(stl_paths)} STL files in {elapsed:.2f}s")
        return results
    
    def _render_views_batched(self,
                             mesh: pv.DataSet,
                             stl_path: Union[str, Path],
                             viewpoints: Dict[str, Tuple],
                             output_dir: Path,
                             use_global_lighting: bool,
                             force_overwrite: bool) -> Dict[str, str]:
        """Render views using GPU batch processing."""
        results = {}
        view_items = list(viewpoints.items())
        
        # Calculate lighting setup
        if use_global_lighting:
            all_lights = GPULightingSystem.calculate_global_lights(list(viewpoints.values()))
        else:
            all_lights = None
        
        # Process views in batches
        for i in range(0, len(view_items), self.batch_size_renders):
            batch_views = view_items[i:i + self.batch_size_renders]
            
            # Render batch of views in parallel
            with ThreadPoolExecutor(max_workers=len(batch_views)) as executor:
                future_to_view = {
                    executor.submit(
                        self._render_single_view_gpu,
                        mesh, view_name, camera_params, output_dir,
                        use_global_lighting, all_lights, force_overwrite
                    ): view_name for view_name, camera_params in batch_views
                }
                
                # Collect batch results
                for future in as_completed(future_to_view):
                    view_name = future_to_view[future]
                    try:
                        output_path = future.result()
                        if output_path:
                            results[view_name] = str(output_path)
                        else:
                            results[view_name] = "FAILED_RENDERING"
                    except Exception as e:
                        self.logger.error(f"Failed to render view {view_name}: {e}")
                        results[view_name] = "FAILED_RENDERING"
        
        return results
    
    def _render_single_view_gpu(self,
                               mesh: pv.DataSet,
                               view_name: str,
                               camera_params: Tuple,
                               output_dir: Path,
                               use_global_lighting: bool,
                               global_lights: Optional[List[pv.Light]],
                               force_overwrite: bool) -> Optional[str]:
        """Render a single view using GPU acceleration."""
        output_path = output_dir / f"{view_name}.png"
        
        # Skip if exists and not overwriting
        if output_path.exists() and not force_overwrite:
            return str(output_path)
        
        try:
            # Use GPU-managed plotter
            with self.gpu_manager.get_plotter() as plotter:
                # Clear any existing state
                plotter.remove_all_lights()
                
                # Add mesh with normal-based coloring for better visualization
                point_normals = mesh.point_normals
                rgb_colors = (point_normals + 1.0) / 2.0
                rgb_colors = np.clip(rgb_colors, 0.0, 1.0)
                
                plotter.add_mesh(mesh, scalars=rgb_colors, rgb=True, smooth_shading=True)
                
                # Set camera position
                plotter.camera_position = camera_params
                plotter.reset_camera_clipping_range()
                
                # Setup lighting
                if use_global_lighting and global_lights:
                    for light in global_lights:
                        plotter.add_light(light)
                else:
                    # View-specific lighting
                    view_lights = GPULightingSystem.calculate_lights_for_view(
                        camera_params, GPULightingSystem.LIGHT_INTENSITY
                    )
                    for light in view_lights:
                        plotter.add_light(light)
                
                # Configure for quality rendering
                plotter.show_grid()
                
                # Render to file
                plotter.screenshot(str(output_path))
                
                return str(output_path)
                
        except Exception as e:
            self.logger.error(f"Error rendering view {view_name}: {e}")
            return None
    
    def get_rendering_stats(self) -> Dict:
        """Get rendering performance statistics."""
        with self._pool_lock:
            stats = dict(self.stats)
            
        if stats['total_time'] > 0:
            stats['avg_views_per_second'] = stats['total_rendered'] / stats['total_time']
            stats['avg_time_per_view'] = stats['total_time'] / max(stats['total_rendered'], 1)
        else:
            stats['avg_views_per_second'] = 0
            stats['avg_time_per_view'] = 0
            
        return stats
    
    def cleanup(self):
        """Clean up rendering resources."""
        with self._pool_lock:
            self.render_pool.shutdown(wait=True)
        
        # GPU cleanup handled by context managers


# Convenience functions for easy usage
def render_stl_gpu(stl_path: Union[str, Path],
                  output_base_dir: Union[str, Path],
                  use_global_lighting: bool = False,
                  force_overwrite: bool = False,
                  device_id: int = 0) -> Dict[str, str]:
    """
    Convenience function for GPU-accelerated STL rendering.
    
    Args:
        stl_path: Path to STL file
        output_base_dir: Base directory for rendered images
        use_global_lighting: Use global lighting setup
        force_overwrite: Overwrite existing images
        device_id: GPU device ID to use
        
    Returns:
        Dictionary mapping view names to output image paths
    """
    renderer = GPUBatchRenderer(device_id=device_id)
    try:
        return renderer.render_stl_multiview_gpu(
            stl_path, output_base_dir, use_global_lighting, force_overwrite
        )
    finally:
        renderer.cleanup()


def batch_render_stls_gpu(stl_paths: List[Union[str, Path]],
                         output_base_dir: Union[str, Path],
                         use_global_lighting: bool = False,
                         force_overwrite: bool = False,
                         device_id: int = 0,
                         max_concurrent: Optional[int] = None) -> Dict[str, Dict[str, str]]:
    """
    Convenience function for GPU-accelerated batch STL rendering.
    
    Args:
        stl_paths: List of STL file paths
        output_base_dir: Base directory for all rendered images
        use_global_lighting: Use global lighting setup
        force_overwrite: Overwrite existing images
        device_id: GPU device ID to use
        max_concurrent: Maximum concurrent STL files to process
        
    Returns:
        Dictionary mapping STL names to their rendered view paths
    """
    renderer = GPUBatchRenderer(device_id=device_id)
    try:
        return renderer.batch_render_stls_gpu(
            stl_paths, output_base_dir, use_global_lighting, 
            force_overwrite, max_concurrent
        )
    finally:
        renderer.cleanup() 