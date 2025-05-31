import time
import logging
import traceback
import numpy as np
import pyvista as pv
from pathlib import Path
from typing import Dict, Any, Tuple
import json # For json.dumps in the main block

# Attempt to import GPU utilities, but allow to fail gracefully if not immediately used
# or if this processor is intended to be runnable in environments without full GPU setup.
try:
    from gpu_memory import get_gpu_manager
    GPU_MEMORY_AVAILABLE = True
except ImportError:
    GPU_MEMORY_AVAILABLE = False
    # Define a placeholder if not available, so the rest of the code doesn't break
    # This would need more robust handling if GPU features are critical and expected.
    class PlaceholderGPUManager:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
        def get_plotter(self, **kwargs):
            # Return a basic plotter if PyVista is available
            if pv:
                return pv.Plotter(off_screen=True, **kwargs)
            raise RuntimeError("PyVista not available and GPUManager mock cannot create plotter.")

    def get_gpu_manager():
        logging.warning("gpu_memory.py not found, using placeholder GPU manager. GPU functionalities will be limited or unavailable.")
        return PlaceholderGPUManager()

def process_cad_file_sequentially(
    cad_script_path: str,
    base_output_dir: str,
    num_points: int = 10000,
    image_size: Tuple[int, int] = (1024, 1024),
    force_overwrite: bool = False,
    use_global_lighting: bool = False
) -> Dict[str, Any]:
    """
    Processes a single CadQuery .py file to generate an STL, a point cloud,
    and 14 rendered views, sequentially.

    Args:
        cad_script_path: Absolute or relative path to the CadQuery Python script.
        base_output_dir: The base directory where all outputs for this file 
                         (and others) will be stored. A subdirectory named after 
                         the CAD file stem will be created here.
        num_points: Number of points to generate for the point cloud.
        image_size: Tuple (width, height) for rendered images.
        force_overwrite: If True, overwrite existing output files.
        use_global_lighting: If True, use a global lighting setup for rendering,
                             otherwise use view-specific lights.

    Returns:
        A dictionary containing processing results:
        - On success:
            {
                'success': True,
                'file': str (input file path),
                'stl_path': str (generated STL file path),
                'pointcloud_path': str (generated .npy file path),
                'render_results': Dict[str, str] (view names → PNG paths),
                'processing_time': float (seconds elapsed),
                'num_renders': int (number of successfully rendered views),
                'output_dir': str (base output directory for this specific file)
            }
        - On failure:
            {
                'success': False,
                'file': str (input file path),
                'error': str (error message),
                'traceback': str (full Python traceback)
            }
    """
    logger = logging.getLogger(__name__)
    function_start_time = time.time()

    try:
        # --- 2.2 Path Management ---
        script_path_obj = Path(cad_script_path)
        base_output_dir_obj = Path(base_output_dir)

        if not script_path_obj.is_file():
            raise FileNotFoundError(f"Input CadQuery script not found: {cad_script_path}")

        file_stem = script_path_obj.stem
        file_specific_output_dir = base_output_dir_obj / file_stem
        
        stl_dir = file_specific_output_dir / "stls"
        pointcloud_dir = file_specific_output_dir / "pointclouds"
        render_dir = file_specific_output_dir / "renders" # Target for final renders

        # Create output directories
        stl_dir.mkdir(parents=True, exist_ok=True)
        pointcloud_dir.mkdir(parents=True, exist_ok=True)
        render_dir.mkdir(parents=True, exist_ok=True)

        stl_output_path = stl_dir / f"{file_stem}.stl"
        pointcloud_output_path = pointcloud_dir / f"{file_stem}.npy"

        logger.info(f"Processing {script_path_obj.name} -> {file_specific_output_dir}")

        # --- 2.3 PyVista Setup (Headless Rendering) ---
        pv.start_xvfb()
        pv.OFF_SCREEN = True
        # pv.set_plot_theme("document") # Optional: consider if it affects output

        # --- 2.4 GPU Resource Manager ---
        # This will initialize the manager if it's the first call,
        # or return the existing instance.
        # The actual GPU resources are typically used within context managers
        # by the specialized classes (PointCloudGenerator, BatchRenderer).
        if GPU_MEMORY_AVAILABLE:
            gpu_manager = get_gpu_manager()
            logger.info("GPU Manager obtained.")
        else:
            logger.warning("Using placeholder GPU manager due to missing gpu_memory module.")
            gpu_manager = get_gpu_manager()

        logger.info("Phase 1 (Setup) complete. STL generation next.")
        
        actual_stl_path = None # Will store the path if STL is successfully generated/found

        # --- 3. Integrate STL Generation ---
        try:
            # Attempt to import cq_to_stl from cadquerytostl
            from cadquerytostl import cq_to_stl
            STL_GENERATION_POSSIBLE = True
        except ImportError:
            STL_GENERATION_POSSIBLE = False
            logger.error("cadquerytostl.py not found or cq_to_stl function cannot be imported. STL generation will be skipped.")
            # This is a critical failure for the pipeline if STL is needed for subsequent steps.
            # Consider how to handle this: either raise an error or allow pipeline to continue if STL is optional.
            # For now, let's assume it's critical and leads to failure of this function if STL cannot be made.
            raise RuntimeError("STL generation prerequisite (cadquerytostl.py) not available.")

        if not stl_output_path.exists() or force_overwrite:
            logger.info(f"Generating STL file: {stl_output_path}")
            if not STL_GENERATION_POSSIBLE:
                 # This check is somewhat redundant given the raise above, but good for clarity
                raise RuntimeError("Cannot generate STL: cadquerytostl module not available.")
            try:
                cq_to_stl(str(script_path_obj), str(stl_output_path))
                if not stl_output_path.exists():
                    raise FileNotFoundError(f"STL file was not created by cq_to_stl: {stl_output_path}")
                logger.info(f"STL file generated successfully: {stl_output_path}")
                actual_stl_path = stl_output_path
            except Exception as stl_e:
                logger.error(f"STL generation failed for {script_path_obj.name}: {stl_e}")
                # Propagate the error to the main try-except block
                raise RuntimeError(f"STL generation failed: {stl_e}") from stl_e
        else:
            logger.info(f"STL file already exists and force_overwrite is False: {stl_output_path}")
            actual_stl_path = stl_output_path
        
        actual_pointcloud_path = None # Will store path if successful

        # --- 4. Integrate Point Cloud Generation ---
        if actual_stl_path: # Only proceed if STL was successfully obtained
            try:
                from gpu_pointclouds import GPUPointCloudGenerator
                POINTCLOUD_GENERATION_POSSIBLE = True
            except ImportError:
                POINTCLOUD_GENERATION_POSSIBLE = False
                logger.error("gpu_pointclouds.py not found or GPUPointCloudGenerator cannot be imported. Point cloud generation will be skipped.")
                # Depending on requirements, this could be a critical failure.
                # For now, let's assume it is, to ensure the full pipeline integrity.
                raise RuntimeError("Point cloud generation prerequisite (gpu_pointclouds.py) not available.")

            if not pointcloud_output_path.exists() or force_overwrite:
                logger.info(f"Generating point cloud: {pointcloud_output_path} from {actual_stl_path}")
                if not POINTCLOUD_GENERATION_POSSIBLE:
                    raise RuntimeError("Cannot generate point cloud: gpu_pointclouds module not available.")
                try:
                    # Assuming device_id 0 for single-threaded focused processing for now.
                    # gpu_manager is available if GPU_MEMORY_AVAILABLE is True.
                    # GPUPointCloudGenerator might use get_gpu_manager() internally if needed for context.
                    pc_generator = GPUPointCloudGenerator(device_id=0) 
                    points_array = pc_generator.stl_to_pointcloud_gpu(str(actual_stl_path), num_points)
                    
                    np.save(str(pointcloud_output_path), points_array)
                    if not pointcloud_output_path.exists():
                        raise FileNotFoundError(f"Point cloud file (.npy) was not created: {pointcloud_output_path}")
                    logger.info(f"Point cloud generated successfully: {pointcloud_output_path}")
                    actual_pointcloud_path = pointcloud_output_path
                except Exception as pc_e:
                    logger.error(f"Point cloud generation failed for {actual_stl_path.name}: {pc_e}")
                    raise RuntimeError(f"Point cloud generation failed: {pc_e}") from pc_e
            else:
                logger.info(f"Point cloud file already exists and force_overwrite is False: {pointcloud_output_path}")
                actual_pointcloud_path = pointcloud_output_path
        else:
            logger.warning("Skipping point cloud generation as STL file was not available.")
            # This implies a failure in STL step, which should have already raised an error and exited.
            # However, if STL was optional, this path would be taken. For now, an error in STL gen is fatal.

        actual_render_results = {} # Stores {view_name: path_to_png}
        actual_num_renders = 0

        # --- 5. Integrate Multi-View Rendering ---
        if actual_stl_path: # Only proceed if STL was successfully obtained
            try:
                from gpu_render import GPUBatchRenderer
                RENDERING_POSSIBLE = True
            except ImportError:
                RENDERING_POSSIBLE = False
                logger.error("gpu_render.py not found or GPUBatchRenderer cannot be imported. Rendering will be skipped.")
                raise RuntimeError("Rendering prerequisite (gpu_render.py) not available.")

            expected_views = [
                'front', 'back', 'right', 'left', 'top', 'bottom', 
                'above_ne', 'above_nw', 'above_se', 'above_sw',
                'below_ne', 'below_nw', 'below_se', 'below_sw'
            ]

            # render_dir is the TARGET directory for final images (e.g., .../my_model/renders/)
            # renderer_source_output_dir is where GPUBatchRenderer will place files initially
            # (e.g., .../my_model/renders/my_model/)
            renderer_source_output_dir = render_dir / file_stem 

            # Check if all renders already exist in the FINAL TARGET directory
            all_renders_exist_in_target = True
            if not force_overwrite:
                for view_name in expected_views:
                    expected_target_path = render_dir / f"{view_name}.png"
                    if not expected_target_path.exists():
                        all_renders_exist_in_target = False
                        break
            
            if all_renders_exist_in_target and not force_overwrite:
                logger.info(f"All {len(expected_views)} render files already exist in target {render_dir} and force_overwrite is False.")
                for view_name in expected_views:
                    actual_render_results[view_name] = str((render_dir / f"{view_name}.png").resolve())
                actual_num_renders = len(expected_views)
            else:
                logger.info(f"Proceeding with rendering for {actual_stl_path.name}. Target: {render_dir}")
                if not RENDERING_POSSIBLE:
                    raise RuntimeError("Cannot render: gpu_render module not available.")
                
                # Ensure the source directory for the renderer is clean if force_overwrite is true
                if force_overwrite and renderer_source_output_dir.exists():
                    for item in renderer_source_output_dir.iterdir():
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir(): # Should not happen with current renderer, but for safety
                            import shutil
                            shutil.rmtree(item)
                renderer_source_output_dir.mkdir(parents=True, exist_ok=True) # Ensure it exists for renderer

                try:
                    renderer = GPUBatchRenderer(device_id=0, image_size=image_size)
                    
                    # GPUBatchRenderer output_base_dir is render_dir.
                    # It will create a subdir file_stem within it: render_dir / file_stem / view.png
                    render_results_from_call = renderer.render_stl_multiview_gpu(
                        stl_path=str(actual_stl_path),
                        output_base_dir=str(render_dir), 
                        use_global_lighting=use_global_lighting,
                        force_overwrite=force_overwrite 
                    )

                    if not render_results_from_call:
                        # This case implies renderer itself decided not to produce results (e.g. all files existed in its view with its force_overwrite=False)
                        # OR a more fundamental issue occurred within the renderer before it even attempted files.
                        # If all_renders_exist_in_target was false, this means we expected rendering but got nothing.
                        if not all_renders_exist_in_target:
                            raise RuntimeError("Rendering call returned no results, but files were expected.")
                        else: # This means files pre-existed in target, and renderer also found them pre-existing in its source (less likely scenario) or simply did nothing.
                             logger.info("Renderer returned no results, assuming pre-existing files handled.")
                             # Repopulate from target as a fallback if this path is hit unexpectedly.
                             for view_name in expected_views:
                                actual_render_results[view_name] = str((render_dir / f"{view_name}.png").resolve())
                             actual_num_renders = len(expected_views)

                    temp_moved_results = {}
                    successful_renders_count = 0
                    if render_results_from_call: # If renderer provided a list of what it did
                        for view_name, G_PathStr_source in render_results_from_call.items():
                            source_file_path = Path(G_PathStr_source)
                            if "FAILED_RENDERING" not in G_PathStr_source and source_file_path.exists():
                                target_file_name = source_file_path.name 
                                target_file_path = render_dir / target_file_name

                                target_file_path.parent.mkdir(parents=True, exist_ok=True)
                                if target_file_path.exists() and force_overwrite and target_file_path != source_file_path:
                                    target_file_path.unlink()
                                
                                if source_file_path != target_file_path:
                                    source_file_path.rename(target_file_path)
                                
                                temp_moved_results[view_name] = str(target_file_path.resolve())
                                successful_renders_count += 1
                            else:
                                logger.warning(f"View '{view_name}' failed by renderer or source file not found: {G_PathStr_source}")
                        
                        actual_render_results = temp_moved_results
                        actual_num_renders = successful_renders_count

                    if actual_num_renders < len(expected_views) and (not all_renders_exist_in_target or force_overwrite):
                        logger.warning(f"Expected {len(expected_views)} renders, but only {actual_num_renders} were successful post-processing.")
                        if actual_num_renders == 0:
                            raise RuntimeError("Rendering and post-processing produced no valid image files.")
                    
                    logger.info(f"Rendering and file placement completed. {actual_num_renders} views in {render_dir}")

                    # Clean up the renderer's specific source output subdirectory if it's empty and different
                    if renderer_source_output_dir.exists() and renderer_source_output_dir != render_dir:
                        try:
                            if not any(renderer_source_output_dir.iterdir()): # Check if empty
                                renderer_source_output_dir.rmdir()
                                logger.info(f"Cleaned up empty renderer source output subdirectory: {renderer_source_output_dir}")
                        except OSError as e:
                            logger.warning(f"Could not remove renderer source output subdirectory {renderer_source_output_dir}: {e}")
                
                except Exception as render_e:
                    logger.error(f"Rendering process failed for {actual_stl_path.name}: {render_e}")
                    raise RuntimeError(f"Rendering process failed: {render_e}") from render_e
        else:
            logger.warning("Skipping rendering as STL file was not available.")

        # --- Result Compilation ---
        processing_time = time.time() - function_start_time
        
        # Determine overall success based on critical components
        # For this example, STL and at least one Point Cloud and one Render are critical if attempted
        final_success_status = False
        if actual_stl_path and actual_pointcloud_path and actual_num_renders > 0:
            final_success_status = True
        elif actual_stl_path and actual_pointcloud_path and not RENDERING_POSSIBLE: # If rendering wasn't possible but others were
            final_success_status = True # Or False, depending on strictness
        elif actual_stl_path and not POINTCLOUD_GENERATION_POSSIBLE and not RENDERING_POSSIBLE:
             final_success_status = True # If only STL was possible and made
        # More nuanced success conditions can be added here.
        # For a simple start, let's say success means all attempted steps that *were possible* completed.
        
        # A more robust check: if a step was supposed to run (its module was available) and it didn't produce output, it's a failure.
        pipeline_steps_completed = True
        if not actual_stl_path: pipeline_steps_completed = False # STL is always critical
        
        # Check Pointcloud success if its generation was possible
        try: from gpu_pointclouds import GPUPointCloudGenerator 
        except ImportError: POINTCLOUD_GENERATION_POSSIBLE = False
        else: POINTCLOUD_GENERATION_POSSIBLE = True
        if POINTCLOUD_GENERATION_POSSIBLE and not actual_pointcloud_path: pipeline_steps_completed = False

        # Check Rendering success if its generation was possible
        try: from gpu_render import GPUBatchRenderer
        except ImportError: RENDERING_POSSIBLE = False
        else: RENDERING_POSSIBLE = True
        if RENDERING_POSSIBLE and actual_num_renders < len(expected_views):
             pipeline_steps_completed = False # Or actual_num_renders == 0 for a more lenient check

        final_success_status = pipeline_steps_completed

        return {
            'success': final_success_status,
            'file': str(script_path_obj.resolve()),
            'stl_path': str(actual_stl_path.resolve()) if actual_stl_path else None,
            'pointcloud_path': str(actual_pointcloud_path.resolve()) if actual_pointcloud_path else None,
            'render_results': actual_render_results, 
            'processing_time': processing_time,
            'num_renders': actual_num_renders,
            'output_dir': str(file_specific_output_dir.resolve())
        }

    except Exception as e:
        logger.error(f"Processing failed for {cad_script_path}: {e}")
        return {
            'success': False,
            'file': str(cad_script_path), # Use original path string
            'error': str(e),
            'traceback': traceback.format_exc()
        }

if __name__ == "__main__":
    # Basic logging setup for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Create dummy files and directories for testing
    # In a real scenario, these paths would point to actual data.
    dummy_cad_dir = Path("_test_cad_files")
    dummy_cad_dir.mkdir(exist_ok=True)
    dummy_cad_file = dummy_cad_dir / "test_model.py"
    
    # Create a minimal CadQuery script for testing
    # This script should be runnable by cadquerytostl.py
    dummy_cad_content = """
import cadquery as cq
# Create a simple box
result = cq.Workplane("XY").box(1, 2, 3) 
# If your cadquerytostl.py expects specific variable names like 'part' or 'model', adjust here.
# For example, if it looks for 'part':
# part = cq.Workplane("XY").box(1, 2, 3)
"""
    with open(dummy_cad_file, "w") as f:
        f.write(dummy_cad_content)

    test_output_dir = Path("_test_output_dir")
    test_output_dir.mkdir(exist_ok=True)

    logger.info(f"Running test processing for: {dummy_cad_file}")
    
    result = process_cad_file_sequentially(
        cad_script_path=str(dummy_cad_file),
        base_output_dir=str(test_output_dir),
        num_points=500, # Smaller for faster test
        force_overwrite=True
    )

    # Construct log message separately to avoid potential linter parsing issues with complex f-strings
    log_message_content = json.dumps(result, indent=2)
    log_message = f"Processing Result:\n{log_message_content}"
    logger.info(log_message)

    if result['success']:
        logger.info(f"Test successful. Outputs should be in: {result.get('output_dir')}")
    else:
        logger.error(f"Test failed. Error: {result.get('error')}")

    # Optional: cleanup dummy files (manual for now)
    # import shutil
    # shutil.rmtree(dummy_cad_dir)
    # shutil.rmtree(test_output_dir)

    # If gpu_memory was available and initialized, consider cleanup
    if GPU_MEMORY_AVAILABLE:
        try:
            from gpu_memory import cleanup_gpu_resources
            logger.info("Attempting global GPU resource cleanup.")
            cleanup_gpu_resources()
            logger.info("Global GPU resource cleanup finished.")
        except ImportError:
            logger.warning("Could not import cleanup_gpu_resources.")
        except Exception as e:
            logger.error(f"Error during GPU resource cleanup: {e}")

