import time
import logging
import traceback
import numpy as np
import pyvista as pv
from pathlib import Path
from typing import Dict, Any, Tuple, Literal
import json # For json.dumps in the main block
import shutil
# Global lock for PyVista rendering operations to prevent concurrent access issues
# GLX context errors often arise from multiple threads using graphics resources simultaneously.
# _pyvista_render_lock = threading.Lock() # REMOVED for multiprocessing

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
        def get_renderer(self): # Matched to actual manager
            if pv and _GPUBatchRenderer_class:
                 # This placeholder might not be fully functional if _GPUBatchRenderer_class is the mock
                return _GPUBatchRenderer_class(placeholder=True) 
            raise RuntimeError("PyVista or GPUBatchRenderer not available for placeholder.")
        def get_generator(self): # Matched to actual manager
            if pv and _GPUPointCloudGenerator_class:
                return _GPUPointCloudGenerator_class(placeholder=True)
            raise RuntimeError("PyVista or GPUPointCloudGenerator not available for placeholder.")

    def get_gpu_manager(manager_type: str = "renderer"):
        logging.warning(f"gpu_memory.py not found, using placeholder GPU manager for type '{manager_type}'.")
        return PlaceholderGPUManager()

# Global flag to track if pv.start_xvfb() has been called in the current process
_xvfb_started_in_process = False

DEFAULT_VIEWS: Tuple[str, ...] = (
    "iso", "front", "back", "right", "left", "top", "bottom",
    "above_ne", "above_nw", "above_se", "above_sw",
    "back_right", "back_left" # "top", "bottom" are often less informative for CAD
)

# These are defined at module level but will be passed as args for robustness with multiprocessing
# try:
#     from cadquerytostl import cq_to_stl
#     CADQUERYTOSTL_AVAILABLE = True 
# except ImportError:
#     CADQUERYTOSTL_AVAILABLE = False
#     def cq_to_stl(*args, **kwargs): # Mock function
#         logging.error("cadquerytostl.cq_to_stl is not available.")
#         return None
#
# try:
#     from gpu_pointclouds import GPUPointCloudGenerator
#     GPU_POINTCLOUD_GENERATOR_AVAILABLE = True
# except ImportError:
#     GPU_POINTCLOUD_GENERATOR_AVAILABLE = False
#     # ... mock class ...
#
# try:
#     from gpu_render import GPUBatchRenderer
#     GPU_BATCH_RENDERER_AVAILABLE = True
# except ImportError:
#     GPU_BATCH_RENDERER_AVAILABLE = False
#     # ... mock class ...

# --- Keep the mock definitions for when the imports fail at module level ---
# These ensure the script can still be imported and potentially run with placeholders
# if the dependencies are missing, though full functionality will be impaired.

_CADQUERYTOSTL_AVAILABLE = False
_cq_to_stl_func = None
try:
    from cadquerytostl import cq_to_stl as _cq_to_stl_func_imported
    _cq_to_stl_func = _cq_to_stl_func_imported
    _CADQUERYTOSTL_AVAILABLE = True
except ImportError:
    def _cq_to_stl_mock(*args, **kwargs):
        logging.error("cadquerytostl.cq_to_stl is not available (mock used).")
        return None
    _cq_to_stl_func = _cq_to_stl_mock
    _CADQUERYTOSTL_AVAILABLE = False

_GPU_POINTCLOUD_GENERATOR_AVAILABLE = False
_GPUPointCloudGenerator_class = None
try:
    from gpu_pointclouds import GPUPointCloudGenerator as _GPUPointCloudGenerator_class_imported
    _GPUPointCloudGenerator_class = _GPUPointCloudGenerator_class_imported
    _GPU_POINTCLOUD_GENERATOR_AVAILABLE = True
except ImportError:
    class _GPUPointCloudGeneratorMock:
        def __init__(self, *args, **kwargs):
            logging.error("gpu_pointclouds.GPUPointCloudGenerator is not available (mock used).")
        def stl_to_pointcloud_gpu(self, *args, **kwargs):
            logging.error("GPUPointCloudGenerator.stl_to_pointcloud_gpu is not available (mock used).")
            return np.array([])
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
    _GPUPointCloudGenerator_class = _GPUPointCloudGeneratorMock
    _GPU_POINTCLOUD_GENERATOR_AVAILABLE = False

_GPU_BATCH_RENDERER_AVAILABLE = False
_GPUBatchRenderer_class = None
try:
    from gpu_render import GPUBatchRenderer as _GPUBatchRenderer_class_imported
    _GPUBatchRenderer_class = _GPUBatchRenderer_class_imported
    _GPU_BATCH_RENDERER_AVAILABLE = True
except ImportError:
    class _GPUBatchRendererMock:
        def __init__(self, *args, **kwargs):
            logging.error("gpu_render.GPUBatchRenderer is not available (mock used).")
        def render_stl_multiview_gpu(self, *args, **kwargs):
            logging.error("GPUBatchRenderer.render_stl_multiview_gpu is not available (mock used).")
            return {}
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
    _GPUBatchRenderer_class = _GPUBatchRendererMock
    _GPU_BATCH_RENDERER_AVAILABLE = False

def process_cad_file_sequentially(
    cad_script_path: str,
    base_output_dir: str,
    views: Tuple[str, ...] = DEFAULT_VIEWS,
    num_points: int = 4096,
    force_overwrite: bool = False,
    render_width: int = 1024,
    render_height: int = 1024,
    processing_stage: Literal["all", "geometry", "render"] = "all",
    cadquerytostl_available_flag: bool = True,
    gpu_pointcloud_generator_available_flag: bool = True,
    gpu_batch_renderer_available_flag: bool = True
) -> Dict[str, Any]:
    """
    Processes a single CadQuery .py file based on the specified processing_stage.
    - "geometry": Converts to STL and generates a point cloud.
    - "render": Renders multiple views from an existing STL.
    - "all": Performs both geometry and render stages.

    Args:
        cad_script_path: Absolute path to the CadQuery .py script.
        base_output_dir: The root directory where outputs will be stored.
        views: A tuple of view names for rendering.
        num_points: Number of points for the point cloud.
        force_overwrite: If True, overwrite existing files for the current stage(s).
        render_width: Width of the rendered image.
        render_height: Height of the rendered image.
        processing_stage: "all", "geometry", or "render".
        cadquerytostl_available_flag: Boolean indicating if cq_to_stl is available.
        gpu_pointcloud_generator_available_flag: Boolean indicating if GPUPointCloudGenerator is available.
        gpu_batch_renderer_available_flag: Boolean indicating if GPUBatchRenderer is available.

    Returns:
        A dictionary containing processing results and metadata:
        {
            "cad_file": str (path to the input CAD script),
            "output_subdir": str (path to the dedicated output subdirectory for this file),
            "stl_file": str (path to the STL file, expected or generated),
            "pointcloud_file": str (path to the point cloud .npy file, expected or generated),
            "render_files": Dict[str, str] (view_name -> path to render .png file),
            "status": str (Overall status, e.g., "Completed_All_Stages", "Error_STL", "Warning_No_Render_Generator", "Completed_Skipped_All_Exist_Or_Unavailable"),
            "stages_processed": List[str] (chronological list of operations performed or skipped, e.g., ["stl_generated", "pointcloud_skipped_exists"]),
            "error": Optional[str] (error message if an error status is set),
            "traceback": Optional[str] (Python traceback if an error status is set),
            "processing_time_seconds": float (total time taken for processing this file)
        }
    """
    logger = logging.getLogger(f"{__name__}.{Path(cad_script_path).stem}")
    function_start_time = time.time()

    try:
        # Set PyVista to off-screen mode (critical for headless operation with Xvfb or OSMesa)
        pv.OFF_SCREEN = True

        # --- 2.2 Path Management ---
        script_path_obj = Path(cad_script_path)
        base_output_dir_obj = Path(base_output_dir)

        if not script_path_obj.is_file():
            raise FileNotFoundError(f"Input CadQuery script not found: {cad_script_path}")

        filename_no_ext = script_path_obj.stem
        # filename_no_ext is already defined from cad_script_path
        
        # Ensure output directories exist
        # Modified to include the parent directory name of the input file (e.g., 'batch_00')
        # in the output path to avoid collisions when processing multiple input batches.
        input_batch_dir_name = script_path_obj.parent.name 
        output_sub_dir = base_output_dir_obj / input_batch_dir_name / filename_no_ext
        
        stl_dir = output_sub_dir / "stls"
        pointcloud_dir = output_sub_dir / "pointclouds"
        render_dir = output_sub_dir / "renders" # Target for final renders

        # Create output directories
        stl_dir.mkdir(parents=True, exist_ok=True)
        pointcloud_dir.mkdir(parents=True, exist_ok=True)
        render_dir.mkdir(parents=True, exist_ok=True)

        stl_output_path = stl_dir / f"{filename_no_ext}.stl"
        pointcloud_output_path = pointcloud_dir / f"{filename_no_ext}.npy"

        # Initialize results dictionary HERE
        results = {
            "cad_file": cad_script_path,
            "output_subdir": str(output_sub_dir),
            "stl_file": str(stl_output_path), # Use defined stl_output_path
            "pointcloud_file": str(pointcloud_output_path), # Use defined pointcloud_output_path
            "render_files": {},
            "status": "Scheduled", 
            "stages_processed": [],
            "error": None,
            "traceback": None,
            "processing_time_seconds": 0.0,
        }

        logger.info(f"Processing {script_path_obj.name} -> {output_sub_dir} (Stage: {processing_stage})")

        # --- STAGE: GEOMETRY (STL and Point Cloud) ---
        if processing_stage in ["all", "geometry"]:
            results["status"] = "Processing_Geometry"
            # --- 2.4 STL Generation ---
            stl_file_path = Path(results["stl_file"])

            if stl_file_path.exists() and not force_overwrite:
                logger.info(f"STL file already exists and force_overwrite is False: {stl_file_path}")
                results["stages_processed"].append("stl_skipped_exists")
            elif not cadquerytostl_available_flag: # Use passed flag
                logger.error("cadquerytostl.cq_to_stl is not available (flag). Skipping STL generation.")
                results["status"] = "Error_STL_No_Generator"
                results["error"] = "cq_to_stl not available"
                results["processing_time_seconds"] = time.time() - function_start_time
                return results
            else:
                try:
                    logger.info("Starting STL generation...")
                    stl_dir.mkdir(parents=True, exist_ok=True)
                    generated_stl_path_str = _cq_to_stl_func( # Use the (potentially mock) function from module level
                        script_path=cad_script_path,
                        output_path=str(stl_file_path),
                        object_name=None, 
                        tolerance=0.1, 
                        angular_tolerance=0.1,
                        logger_name=logger.name
                    )
                    if generated_stl_path_str and Path(generated_stl_path_str).exists():
                        results["stl_file"] = generated_stl_path_str
                        logger.info(f"STL file generated successfully: {generated_stl_path_str}")
                        results["stages_processed"].append("stl_generated")
                    else:
                        raise RuntimeError(f"cq_to_stl did not return a valid path or file was not created: {generated_stl_path_str}")
                except Exception as e:
                    logger.error(f"Error during STL generation for {filename_no_ext}: {e}")
                    logger.debug(traceback.format_exc())
                    results["status"] = "Error_STL"
                    results["error"] = str(e)
                    results["traceback"] = traceback.format_exc()
                    results["processing_time_seconds"] = time.time() - function_start_time
                    return results

            # --- 2.5 Point Cloud Generation (within "all" or "geometry" stage) ---
            pointcloud_file_path = Path(results["pointcloud_file"])
            if not Path(results["stl_file"]).exists():
                logger.error(f"Cannot generate point cloud, STL file is missing: {results['stl_file']}")
                results["status"] = "Error_PointCloud_No_STL"
                if not results["error"]: results["error"] = "STL file missing for point cloud generation."
                results["stages_processed"].append("pointcloud_skipped_no_stl")
                if processing_stage == "geometry": # If only doing geometry, this is a terminal error for the stage
                    results["processing_time_seconds"] = time.time() - function_start_time
                    return results 

            elif pointcloud_file_path.exists() and not force_overwrite:
                logger.info(f"Point cloud file already exists and force_overwrite is False: {pointcloud_file_path}")
                results["stages_processed"].append("pointcloud_skipped_exists")
            elif not gpu_pointcloud_generator_available_flag: # Use passed flag
                logger.warning("GPUPointCloudGenerator not available (flag). Skipping point cloud generation.")
                results["stages_processed"].append("pointcloud_skipped_no_generator")
                if "Error" not in results["status"]: results["status"] = "Warning_No_PointCloud_Generator"
            elif Path(results["stl_file"]).exists(): # Added explicit check again, though theoretically covered by first if
                try:
                    logger.info("Starting point cloud generation...")
                    pointcloud_dir.mkdir(parents=True, exist_ok=True)
                    pc_generator_manager = get_gpu_manager("pointcloud")
                    with pc_generator_manager.get_generator() as pc_generator_instance:
                        if pc_generator_instance is None: 
                            # This path might be taken if get_gpu_manager returns a placeholder that yields None
                            # or if the actual manager fails. The flag should ideally prevent this if generator truly unavailable.
                            if gpu_pointcloud_generator_available_flag: # If flag said it was available, this is an unexpected runtime failure
                                raise RuntimeError("Failed to acquire PointCloudGenerator from manager, though availability flag was true.")
                            else: # Flag already indicated not available, this is consistent, log and skip.
                                logger.warning("PointCloudGenerator acquisition failed, consistent with availability flag being false.")
                                results["stages_processed"].append("pointcloud_skipped_no_generator_runtime")
                                if "Error" not in results["status"]: results["status"] = "Warning_No_PointCloud_Generator"
                                # Skip this specific try-block for point cloud generation
                                pass 
                        else:
                            logger.info(f"Using PointCloudGenerator: {pc_generator_instance}")
                            point_cloud_np = pc_generator_instance.stl_to_pointcloud_gpu(
                                stl_file_path=results["stl_file"],
                                num_points=num_points,
                                device_id=0
                            )
                    np.save(pointcloud_file_path, point_cloud_np)
                    logger.info(f"Point cloud generated and saved: {pointcloud_file_path}")
                    results["stages_processed"].append("pointcloud_generated")
                except Exception as e:
                    logger.error(f"Error during point cloud generation for {filename_no_ext}: {e}")
                    logger.debug(traceback.format_exc())
                    results["status"] = "Error_PointCloud"
                    results["error"] = str(e)
                    results["traceback"] = traceback.format_exc()
                    if processing_stage == "geometry": # If only doing geometry, this is a terminal error for the stage
                        results["processing_time_seconds"] = time.time() - function_start_time
                        return results
        
        # --- STAGE: RENDER ---
        if processing_stage in ["all", "render"]:
            # Update status, but preserve earlier errors from geometry stage if processing_stage is 'all'
            if results["status"] not in ["Error_STL", "Error_PointCloud", "Error_STL_No_Generator", "Error_PointCloud_No_STL"]:
                results["status"] = "Processing_Render"
            
            stl_file_for_render = Path(results["stl_file"])
            if not stl_file_for_render.exists():
                logger.error(f"Rendering stage: STL file not found at {stl_file_for_render}. Cannot render.")
                results["status"] = "Error_Render_No_STL"
                if not results["error"]: results["error"] = f"STL file not found for rendering: {stl_file_for_render}"
                results["processing_time_seconds"] = time.time() - function_start_time
                return results # Critical error for render stage if STL is missing

            # Overwrite check for rendering
            all_renders_exist = False
            if not force_overwrite:
                all_renders_exist = True # Assume true, then check
                render_dir.mkdir(parents=True, exist_ok=True) # Ensure dir exists for check (using render_dir)
                for view_name in views:
                    if not (render_dir / f"{view_name}.png").exists(): # using render_dir
                        all_renders_exist = False
                        break
                if all_renders_exist:
                    logger.info(f"All render files exist in {render_dir} and force_overwrite is False. Skipping rendering.") # using render_dir
                    for view_name in views:
                        results["render_files"][view_name] = str(render_dir / f"{view_name}.png") # using render_dir
                    results["stages_processed"].append("render_skipped_exists")
            
            if not all_renders_exist: # Proceed if not all exist or if force_overwrite
                if not gpu_batch_renderer_available_flag: # Use passed flag
                    logger.warning("GPUBatchRenderer not available (flag). Skipping rendering.")
                    results["stages_processed"].append("render_skipped_no_generator")
                    if "Error" not in results["status"] and "Warning" not in results["status"]:
                        results["status"] = "Warning_No_Render_Generator"
                else:
                    try:
                        logger.info(f"Starting multi-view rendering for {stl_file_for_render.name}...")
                        render_dir.mkdir(parents=True, exist_ok=True) # using render_dir
                        
                        temp_render_output_base = output_sub_dir 
                        
                        renderer_manager = get_gpu_manager("renderer")
                        with renderer_manager.get_renderer() as renderer_instance:
                            if renderer_instance is None:
                                if gpu_batch_renderer_available_flag:
                                    raise RuntimeError("Failed to acquire GPUBatchRenderer from manager, though availability flag was true.")
                                else:
                                    logger.warning("GPUBatchRenderer acquisition failed, consistent with availability flag being false.")
                                    results["stages_processed"].append("render_skipped_no_generator_runtime")
                                    if "Error" not in results["status"] and "Warning" not in results["status"]:
                                        results["status"] = "Warning_No_Render_Generator"
                                    # Skip this specific try-block for rendering
                                    pass
                            else:
                                logger.info(f"Using GPUBatchRenderer: {renderer_instance}")
                                renderer_instance.render_stl_multiview_gpu(
                                    stl_file_paths=[str(stl_file_for_render)],
                                    output_dir=str(temp_render_output_base),
                                    views=views,
                                    image_width=render_width,
                                    image_height=render_height,
                                    device_id=0
                                )

                        source_render_subdir = temp_render_output_base / "renders" / filename_no_ext
                        if source_render_subdir.exists() and source_render_subdir.is_dir():
                            logger.info(f"Moving renders from {source_render_subdir} to {render_dir}") # using render_dir
                            for view_file in source_render_subdir.glob("*.png"):
                                target_file_path = render_dir / view_file.name # using render_dir
                                shutil.move(str(view_file), str(target_file_path))
                                results["render_files"][view_file.stem] = str(target_file_path)
                                logger.debug(f"Moved {view_file} to {target_file_path}")
                            
                            try:
                                source_render_subdir.rmdir()
                                if not any((temp_render_output_base / "renders").iterdir()):
                                    (temp_render_output_base / "renders").rmdir()
                            except OSError as e:
                                logger.warning(f"Could not clean up temporary render directory {source_render_subdir} or its parent: {e}")
                        else:
                            logger.warning(f"Expected source render directory {source_render_subdir} not found after rendering.")
                            if "Error" not in results["status"] and "Warning" not in results["status"]:
                                results["status"] = "Warning_Render_Output_Missing"
                        
                        logger.info(f"Rendering completed for {filename_no_ext}.")
                        results["stages_processed"].append("render_generated")

                    except Exception as e:
                        logger.error(f"Error during rendering for {filename_no_ext}: {e}")
                        logger.debug(traceback.format_exc())
                        results["status"] = "Error_Render"
                        results["error"] = str(e)
                        results["traceback"] = traceback.format_exc()

        # --- Final Status Determination ---
        if "Error" in results["status"]:
            pass # Error status is already set and is specific
        elif results["status"] == "Scheduled": # No stages were actually run (e.g. bad processing_stage input - unlikely with Literal)
            results["status"] = "No_Action_Taken"
        elif processing_stage == "all":
            if all(item in results["stages_processed"] for item in ["stl_generated", "pointcloud_generated", "render_generated"]) or \
               all(item in results["stages_processed"] for item in ["stl_skipped_exists", "pointcloud_skipped_exists", "render_skipped_exists"]):
                results["status"] = "Completed_All_Stages"
            elif "Warning" in results["status"]:
                pass # Preserve warning status (e.g. Warning_No_Render_Generator)
            else: # Partial success or unexpected state for 'all'
                results["status"] = f"Completed_All_Stages_Partial: {results['stages_processed']}"
        elif processing_stage == "geometry":
            if all(item in results["stages_processed"] for item in ["stl_generated", "pointcloud_generated"]) or \
               all(item in results["stages_processed"] for item in ["stl_skipped_exists", "pointcloud_skipped_exists"]):
                results["status"] = "Completed_Geometry_Stage"
            elif "Warning" in results["status"]:
                pass
            else:
                results["status"] = f"Completed_Geometry_Stage_Partial: {results['stages_processed']}"
        elif processing_stage == "render":
            if "render_generated" in results["stages_processed"] or "render_skipped_exists" in results["stages_processed"]:
                results["status"] = "Completed_Render_Stage"
            elif "Warning" in results["status"]:
                pass
            else:
                results["status"] = f"Completed_Render_Stage_Partial: {results['stages_processed']}"

    except Exception as e: # Main try-except block's catch for unhandled exceptions
        logger.error(f"Unhandled exception processing {filename_no_ext} (Stage: {processing_stage}): {e}")
        logger.debug(traceback.format_exc())
        results["status"] = "Error_Unhandled"
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()
        # Ensure essential keys exist even if error happened before results was fully populated
        results.setdefault("cad_file", cad_script_path)
        results.setdefault("output_subdir", str(output_sub_dir) if 'output_sub_dir' in locals() else base_output_dir)
        results.setdefault("stages_processed", [])
        results.setdefault("render_files", {})

    results["processing_time_seconds"] = time.time() - function_start_time
    
    # Adjust final log status message for clarity if no actual processing occurred due to skips
    final_log_display_status = results["status"]
    if not results["error"] and not any(s.endswith("_generated") for s in results["stages_processed"]):
        if all(s.endswith(("_skipped_exists", "_skipped_no_generator", "_skipped_no_stl")) for s in results["stages_processed"]):
            final_log_display_status = "Completed_Skipped_All_Exist_Or_Unavailable"
        elif not results["stages_processed"] and results["status"] == "Scheduled": # Should be caught by No_Action_Taken
             final_log_display_status = "No_Action_Taken"

    logger.info(
        f"Finished processing {filename_no_ext} in {results['processing_time_seconds']:.2f}s. "
        f"Stage: {processing_stage}, Final Status: {final_log_display_status}, Stages Detail: {results['stages_processed']}"
    )
    return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    test_logger = logging.getLogger(__name__)

    dummy_cad_script_content = """
import cadquery as cq
result = cq.Workplane("XY").box(1, 2, 3)
# show_object(result) # Not needed for stl export, but useful for direct CadQuery debugging
"""
    test_dir = Path("_test_single_file_processor_workspace") # More specific name
    test_output_dir = test_dir / "outputs"
    dummy_script_path = test_dir / "dummy_box.py"

    # Define input_batch_dir_name for cleanup path construction, mirroring logic in process_cad_file_sequentially
    input_batch_dir_name = dummy_script_path.parent.name

    # Ensure clean state for testing
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    test_output_dir.mkdir(parents=True, exist_ok=True)

    with open(dummy_script_path, "w") as f:
        f.write(dummy_cad_script_content)
    test_logger.info(f"Created dummy script: {dummy_script_path}")

    def check_results(results_dict, expected_status_part, stage_name, expect_error=False):
        log_dump = json.dumps(results_dict, indent=2)
        test_logger.info(f"Results ({stage_name}):\n{log_dump}")
        if expect_error:
            assert "Error" in results_dict["status"], f"{stage_name} expected an error status, got {results_dict['status']}"
            assert results_dict["error"] is not None, f"{stage_name} expected an error message"
        else:
            assert "Error" not in results_dict["status"], f"{stage_name} had unexpected error: {results_dict['status']} - {results_dict['error']}"
            assert expected_status_part in results_dict["status"], f"{stage_name} failed: status {results_dict['status']} does not contain {expected_status_part}"
            if "Warning" not in results_dict["status"] and results_dict["status"] != "No_Action_Taken" and "Skipped" not in results_dict["status"]:
                 assert results_dict["stages_processed"], f"{stage_name} ({results_dict['status']}) should have processed or skipped stages, but 'stages_processed' is empty."

    # Test Case 1: Geometry Only (force_overwrite=True)
    test_logger.info(f"--- Test Case 1: Geometry Only (force_overwrite=True) ---")
    results_geom = process_cad_file_sequentially(
        str(dummy_script_path), str(test_output_dir), force_overwrite=True, processing_stage="geometry"
    )
    check_results(results_geom, "Completed_Geometry_Stage", "Test 1: Geometry Only")
    assert Path(results_geom["stl_file"]).exists(), "STL file missing after geometry stage (Test 1)"
    assert Path(results_geom["pointcloud_file"]).exists(), "Point cloud missing after geometry stage (Test 1)"
    assert not results_geom["render_files"], "Render files should not exist after geometry stage (Test 1)"
    assert "stl_generated" in results_geom["stages_processed"]
    assert "pointcloud_generated" in results_geom["stages_processed"]

    # Test Case 2: Render Only (force_overwrite=True, uses geometry from Test 1)
    test_logger.info(f"--- Test Case 2: Render Only (force_overwrite=True, uses geometry from Test 1) ---")
    results_render = process_cad_file_sequentially(
        str(dummy_script_path), str(test_output_dir), force_overwrite=True, processing_stage="render"
    )
    check_results(results_render, "Completed_Render_Stage", "Test 2: Render Only")
    assert Path(results_geom["stl_file"]).exists(), "STL file from geometry stage should still exist (Test 2)"
    assert results_render["render_files"], "Render files should exist after render stage (Test 2)"
    assert "render_generated" in results_render["stages_processed"]
    for f_path in results_render["render_files"].values():
        assert Path(f_path).exists(), f"Render file {f_path} should exist (Test 2)"

    # Test Case 3: Geometry Only (force_overwrite=False, files exist)
    test_logger.info(f"--- Test Case 3: Geometry Only (force_overwrite=False, files exist) ---")
    results_geom_skip = process_cad_file_sequentially(
        str(dummy_script_path), str(test_output_dir), force_overwrite=False, processing_stage="geometry"
    )
    check_results(results_geom_skip, "Completed_Geometry_Stage", "Test 3: Geometry Only (Skip)")
    assert "stl_skipped_exists" in results_geom_skip["stages_processed"]
    assert "pointcloud_skipped_exists" in results_geom_skip["stages_processed"]

    # Test Case 4: Render Only (force_overwrite=False, files exist)
    test_logger.info(f"--- Test Case 4: Render Only (force_overwrite=False, files exist) ---")
    results_render_skip = process_cad_file_sequentially(
        str(dummy_script_path), str(test_output_dir), force_overwrite=False, processing_stage="render"
    )
    check_results(results_render_skip, "Completed_Render_Stage", "Test 4: Render Only (Skip)")
    assert "render_skipped_exists" in results_render_skip["stages_processed"]

    # Test Case 5: All Stages (force_overwrite=True, clean start)
    test_logger.info(f"--- Test Case 5: All Stages (force_overwrite=True, clean start) ---")
    shutil.rmtree(test_output_dir / input_batch_dir_name / dummy_script_path.stem) # Clean specific output subdir
    results_all = process_cad_file_sequentially(
        str(dummy_script_path), str(test_output_dir), force_overwrite=True, processing_stage="all"
    )
    check_results(results_all, "Completed_All_Stages", "Test 5: All Stages")
    assert Path(results_all["stl_file"]).exists(), "STL file missing (Test 5)"
    assert Path(results_all["pointcloud_file"]).exists(), "Point cloud missing (Test 5)"
    assert results_all["render_files"], "Render files missing (Test 5)"
    assert "stl_generated" in results_all["stages_processed"]
    assert "pointcloud_generated" in results_all["stages_processed"]
    assert "render_generated" in results_all["stages_processed"]

    # Test Case 6: Render Only (STL missing)
    test_logger.info(f"--- Test Case 6: Render Only (STL missing) ---")
    shutil.rmtree(test_output_dir / input_batch_dir_name / dummy_script_path.stem) # Clean specific output subdir to remove STL
    results_render_no_stl = process_cad_file_sequentially(
        str(dummy_script_path), str(test_output_dir), processing_stage="render"
    )
    check_results(results_render_no_stl, "Error_Render_No_STL", "Test 6: Render Only (No STL)", expect_error=True)
    assert not results_render_no_stl["render_files"], "Render files should not exist (Test 6)"

    # Test Case 7: Geometry only (CadQuery to STL fails)
    test_logger.info(f"--- Test Case 7: Geometry only (CadQuery to STL fails) ---")
    faulty_cad_script_content = "import cadquery as cq\nresult = cq.Workplane('XY').circle(1).extrude('not_a_number') # This will fail"
    faulty_script_path = test_dir / "faulty_box.py"
    with open(faulty_script_path, "w") as f: f.write(faulty_cad_script_content)
    
    # Clean up specific output for faulty script if it exists from a previous failed run
    faulty_output_subdir = test_output_dir / faulty_script_path.parent.name / faulty_script_path.stem
    if faulty_output_subdir.exists():
        shutil.rmtree(faulty_output_subdir)

    results_geom_fail_stl = process_cad_file_sequentially(
        str(faulty_script_path), str(test_output_dir), processing_stage="geometry"
    )
    check_results(results_geom_fail_stl, "Error_STL", "Test 7: Geometry (STL Fail)", expect_error=True)

    # Test Case 8: All stages (force_overwrite=False, all files exist from Test 5 run)
    test_logger.info(f"--- Test Case 8: All Stages (force_overwrite=False, files exist) ---")
    # Re-run test 5 outputs
    shutil.rmtree(test_output_dir / input_batch_dir_name / dummy_script_path.stem) # Clean specific output subdir
    process_cad_file_sequentially( str(dummy_script_path), str(test_output_dir), force_overwrite=True, processing_stage="all") # Create all files
    # Now run with force_overwrite=False
    results_all_skip = process_cad_file_sequentially(
        str(dummy_script_path), str(test_output_dir), force_overwrite=False, processing_stage="all"
    )
    check_results(results_all_skip, "Completed_Skipped_All_Exist_Or_Unavailable", "Test 8: All Stages (Skip)")
    assert "stl_skipped_exists" in results_all_skip["stages_processed"]
    assert "pointcloud_skipped_exists" in results_all_skip["stages_processed"]
    assert "render_skipped_exists" in results_all_skip["stages_processed"]

    test_logger.info("--- All single_file_processor stage tests completed. ---")
    test_logger.info(f"Note: Test files and directories are in: {test_dir}")
    test_logger.info("Consider manually deleting it if not needed further.")

    # Cleanup GPU resources (if applicable and imported)
    if GPU_MEMORY_AVAILABLE:
        try:
            # from gpu_memory import cleanup_gpu_resources # Already imported if available
            if 'cleanup_gpu_resources' in globals() or 'cleanup_gpu_resources' in locals():
                test_logger.info("Attempting global GPU resource cleanup.")
                cleanup_gpu_resources()
                test_logger.info("Global GPU resource cleanup finished.")
            else:
                test_logger.warning("cleanup_gpu_resources not found in globals/locals despite GPU_MEMORY_AVAILABLE=True.")
        except NameError: # Should be caught by the check above, but defensive
             test_logger.warning("cleanup_gpu_resources not defined despite GPU_MEMORY_AVAILABLE=True.")
        except Exception as e:
            test_logger.error(f"Error during GPU resource cleanup: {e}")

