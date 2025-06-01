"""
GPU-accelerated point cloud generation from STL meshes.
Uses CUDA for parallel mesh sampling and processing.
"""

import logging
import numpy as np
from typing import Union, Tuple, Optional
from pathlib import Path
import gc

try:
    import cupy as cp
    import cupyx.scipy.spatial.distance as cup_distance
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
    TorchTensor = torch.Tensor
except ImportError:
    TORCH_AVAILABLE = False
    TorchTensor = None

import trimesh
import pyvista as pv

from gpu_memory import get_gpu_manager


class GPUPointCloudGenerator:
    """GPU-accelerated point cloud generation from meshes."""
    
    def __init__(self, device_id: int = 0):
        self.logger = logging.getLogger(__name__)
        self.device_id = device_id
        self.gpu_manager = get_gpu_manager()
        
        # Check GPU availability
        self.use_gpu = CUPY_AVAILABLE or TORCH_AVAILABLE
        if not self.use_gpu:
            self.logger.warning("No GPU acceleration available, falling back to CPU")
    
    def stl_to_pointcloud_gpu(self, 
                             stl_file_path: Union[str, Path], 
                             n_points: int = 8192,
                             batch_size: Optional[int] = None) -> np.ndarray:
        """
        Convert STL file to point cloud using GPU acceleration.
        
        Args:
            stl_file_path: Path to STL file
            n_points: Number of points to sample
            batch_size: Batch size for GPU processing (auto-detected if None)
            
        Returns:
            Point cloud as numpy array of shape (n_points, 3)
        """
        if batch_size is None:
            # Auto-detect batch size based on available memory
            batch_size = self._estimate_batch_size(n_points)
        
        try:
            # Load mesh using trimesh (still CPU-based loading)
            mesh = trimesh.load(stl_file_path)
            if mesh is None:
                raise ValueError(f"Failed to load mesh from {stl_file_path}")
            
            # Preprocess mesh (centering and scaling)
            mesh = self._preprocess_mesh(mesh)
            
            # Generate point cloud on GPU
            if self.use_gpu:
                points = self._sample_mesh_gpu(mesh, n_points, batch_size)
            else:
                points = self._sample_mesh_cpu(mesh, n_points)
            
            return points
            
        except Exception as e:
            self.logger.error(f"Failed to generate point cloud from {stl_file_path}: {e}")
            # Fallback to CPU method
            return self._fallback_cpu_sampling(stl_file_path, n_points)
        finally:
            # Cleanup
            if 'mesh' in locals():
                del mesh
            gc.collect()
    
    def batch_stl_to_pointclouds_gpu(self, 
                                    stl_paths: list, 
                                    n_points: int = 8192,
                                    max_batch_size: int = 8) -> list:
        """
        Process multiple STL files in GPU batches.
        
        Args:
            stl_paths: List of STL file paths
            n_points: Number of points per point cloud
            max_batch_size: Maximum number of meshes to process simultaneously
            
        Returns:
            List of point clouds as numpy arrays
        """
        results = []
        
        # Process in batches
        for i in range(0, len(stl_paths), max_batch_size):
            batch_paths = stl_paths[i:i + max_batch_size]
            
            try:
                batch_results = self._process_mesh_batch_gpu(batch_paths, n_points)
                results.extend(batch_results)
            except Exception as e:
                self.logger.warning(f"GPU batch processing failed: {e}, falling back to individual processing")
                # Fallback to individual processing
                for path in batch_paths:
                    try:
                        pc = self.stl_to_pointcloud_gpu(path, n_points)
                        results.append(pc)
                    except Exception as e2:
                        self.logger.error(f"Failed to process {path}: {e2}")
                        results.append(None)
        
        return results
    
    def _preprocess_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Preprocess mesh (center and scale to unit cube)."""
        # Center at origin
        mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)
        
        # Scale to fit in unit cube
        max_extent = max(mesh.extents)
        if max_extent > 0:
            mesh.apply_scale(2.0 / max_extent)
        
        return mesh
    
    def _sample_mesh_gpu(self, 
                        mesh: trimesh.Trimesh, 
                        n_points: int, 
                        batch_size: int) -> np.ndarray:
        """Sample points from mesh using GPU acceleration."""
        if CUPY_AVAILABLE:
            return self._sample_mesh_cupy(mesh, n_points, batch_size)
        elif TORCH_AVAILABLE:
            return self._sample_mesh_pytorch(mesh, n_points, batch_size)
        else:
            return self._sample_mesh_cpu(mesh, n_points)
    
    def _sample_mesh_cupy(self, 
                         mesh: trimesh.Trimesh, 
                         n_points: int, 
                         batch_size: int) -> np.ndarray:
        """Sample points using CuPy acceleration."""
        with self.gpu_manager.cuda_pool.cuda_context(self.device_id):
            # Transfer mesh data to GPU
            vertices_gpu = cp.asarray(mesh.vertices.astype(np.float32))
            faces_gpu = cp.asarray(mesh.faces.astype(np.int32))
            
            # Calculate face areas on GPU
            face_areas = self._calculate_face_areas_cupy(vertices_gpu, faces_gpu)
            face_areas_normalized = face_areas / cp.sum(face_areas)
            
            # Sample points in batches
            all_points = []
            points_per_batch = min(batch_size, n_points)
            
            for i in range(0, n_points, points_per_batch):
                current_batch_size = min(points_per_batch, n_points - i)
                
                # Sample faces based on area
                face_indices = cp.random.choice(
                    len(faces_gpu), 
                    size=current_batch_size, 
                    p=face_areas_normalized
                )
                
                # Sample points within selected faces
                batch_points = self._sample_points_in_faces_cupy(
                    vertices_gpu, faces_gpu, face_indices
                )
                
                all_points.append(batch_points)
            
            # Combine all batches
            result_gpu = cp.vstack(all_points)
            
            # Transfer back to CPU
            return cp.asnumpy(result_gpu)
    
    def _sample_mesh_pytorch(self, 
                            mesh: trimesh.Trimesh, 
                            n_points: int, 
                            batch_size: int) -> np.ndarray:
        """Sample points using PyTorch acceleration."""
        device = f'cuda:{self.device_id}'
        
        # Transfer mesh data to GPU
        vertices_gpu = torch.from_numpy(mesh.vertices.astype(np.float32)).to(device)
        faces_gpu = torch.from_numpy(mesh.faces.astype(np.int64)).to(device)
        
        # Calculate face areas on GPU
        face_areas = self._calculate_face_areas_pytorch(vertices_gpu, faces_gpu)
        face_probs = face_areas / torch.sum(face_areas)
        
        # Sample points in batches
        all_points = []
        points_per_batch = min(batch_size, n_points)
        
        for i in range(0, n_points, points_per_batch):
            current_batch_size = min(points_per_batch, n_points - i)
            
            # Sample faces based on area
            face_indices = torch.multinomial(
                face_probs, 
                num_samples=current_batch_size, 
                replacement=True
            )
            
            # Sample points within selected faces
            batch_points = self._sample_points_in_faces_pytorch(
                vertices_gpu, faces_gpu, face_indices
            )
            
            all_points.append(batch_points)
        
        # Combine all batches
        result_gpu = torch.cat(all_points, dim=0)
        
        # Transfer back to CPU
        return result_gpu.cpu().numpy()
    
    def _calculate_face_areas_cupy(self, vertices: cp.ndarray, faces: cp.ndarray) -> cp.ndarray:
        """Calculate face areas using CuPy."""
        # Get face vertices
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        
        # Calculate cross product and area
        cross = cp.cross(v1 - v0, v2 - v0)
        areas = 0.5 * cp.linalg.norm(cross, axis=1)
        
        return areas
    
    def _calculate_face_areas_pytorch(self, vertices, faces):
        """Calculate face areas using PyTorch."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
            
        # Get face vertices
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        
        # Calculate cross product and area
        cross = torch.cross(v1 - v0, v2 - v0, dim=1)
        areas = 0.5 * torch.norm(cross, dim=1)
        
        return areas
    
    def _sample_points_in_faces_cupy(self, 
                                    vertices: cp.ndarray, 
                                    faces: cp.ndarray, 
                                    face_indices: cp.ndarray) -> cp.ndarray:
        """Sample random points within triangular faces using CuPy."""
        # Get vertices of selected faces
        selected_faces = faces[face_indices]
        v0 = vertices[selected_faces[:, 0]]
        v1 = vertices[selected_faces[:, 1]]
        v2 = vertices[selected_faces[:, 2]]
        
        # Generate random barycentric coordinates
        r1 = cp.random.random(len(face_indices))
        r2 = cp.random.random(len(face_indices))
        
        # Ensure points are inside triangle
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        
        # Convert to Cartesian coordinates
        r3 = 1 - r1 - r2
        points = (r1[:, None] * v0 + 
                 r2[:, None] * v1 + 
                 r3[:, None] * v2)
        
        return points
    
    def _sample_points_in_faces_pytorch(self, 
                                       vertices, 
                                       faces, 
                                       face_indices):
        """Sample random points within triangular faces using PyTorch."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
            
        # Get vertices of selected faces
        selected_faces = faces[face_indices]
        v0 = vertices[selected_faces[:, 0]]
        v1 = vertices[selected_faces[:, 1]]
        v2 = vertices[selected_faces[:, 2]]
        
        # Generate random barycentric coordinates
        r1 = torch.rand(len(face_indices), device=vertices.device)
        r2 = torch.rand(len(face_indices), device=vertices.device)
        
        # Ensure points are inside triangle
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        
        # Convert to Cartesian coordinates
        r3 = 1 - r1 - r2
        points = (r1.unsqueeze(1) * v0 + 
                 r2.unsqueeze(1) * v1 + 
                 r3.unsqueeze(1) * v2)
        
        return points
    
    def _sample_mesh_cpu(self, mesh: trimesh.Trimesh, n_points: int) -> np.ndarray:
        """Fallback CPU-based sampling using trimesh."""
        points, _ = trimesh.sample.sample_surface(mesh, n_points)
        return points
    
    def _process_mesh_batch_gpu(self, stl_paths: list, n_points: int) -> list:
        """Process multiple meshes simultaneously on GPU."""
        # This is a simplified batch processing - in practice, you'd want
        # to optimize memory usage and handle varying mesh sizes
        results = []
        
        for path in stl_paths:
            try:
                pc = self.stl_to_pointcloud_gpu(path, n_points)
                results.append(pc)
            except Exception as e:
                self.logger.error(f"Failed to process {path} in batch: {e}")
                results.append(None)
        
        return results
    
    def _estimate_batch_size(self, n_points: int) -> int:
        """Estimate optimal batch size based on available GPU memory."""
        try:
            memory_stats = self.gpu_manager.get_memory_stats()
            
            if 'cuda_memory' in memory_stats:
                free_bytes = memory_stats['cuda_memory'].get('free_bytes', 0)
                
                # Rough estimate: each point needs ~12 bytes (3 float32)
                # Add safety factor
                estimated_points = (free_bytes * 0.5) // (12 * 4)  # Safety factor of 0.5, 4x for intermediate calculations
                
                # Clamp to reasonable range
                batch_size = max(1024, min(int(estimated_points), 32768))
                
                self.logger.debug(f"Estimated batch size: {batch_size} points")
                return batch_size
        except Exception as e:
            self.logger.warning(f"Could not estimate batch size: {e}")
        
        # Default batch size
        return 8192
    
    def _fallback_cpu_sampling(self, stl_file_path: Union[str, Path], n_points: int) -> np.ndarray:
        """Fallback to CPU sampling if GPU fails."""
        self.logger.warning(f"Falling back to CPU sampling for {stl_file_path}")
        try:
            mesh = trimesh.load(stl_file_path)
            mesh = self._preprocess_mesh(mesh)
            return self._sample_mesh_cpu(mesh, n_points)
        except Exception as e:
            self.logger.error(f"Fallback CPU sampling failed for {stl_file_path}: {e}")
            # Return empty array as last resort
            return np.zeros((n_points, 3), dtype=np.float32)


def stl_to_pointcloud_gpu(stl_file_path: Union[str, Path], 
                         n_points: int = 8192,
                         device_id: int = 0) -> np.ndarray:
    """
    Standalone function to convert STL to point cloud using GPU.

    Args:
        stl_file_path: Path to STL file
        n_points: Number of points to sample.
        device_id: GPU device ID.
        
    Returns:
        Point cloud as numpy array
    """
    try:
        generator = GPUPointCloudGenerator(device_id=device_id)
        return generator.stl_to_pointcloud_gpu(stl_file_path, n_points)
    except Exception as e:
        logging.error(f"Error in standalone stl_to_pointcloud_gpu for {stl_file_path}: {e}")
        return np.zeros((n_points, 3), dtype=np.float32)


def batch_stl_to_pointclouds_gpu(stl_paths: list, 
                                n_points: int = 8192,
                                device_id: int = 0,
                                max_batch_size: int = 8) -> list:
    """
    Convenience function for batch GPU-accelerated point cloud generation.
    
    Args:
        stl_paths: List of STL file paths
        n_points: Number of points per point cloud
        device_id: GPU device ID to use
        max_batch_size: Maximum batch size
        
    Returns:
        List of point clouds as numpy arrays
    """
    generator = GPUPointCloudGenerator(device_id=device_id)
    return generator.batch_stl_to_pointclouds_gpu(stl_paths, n_points, max_batch_size) 