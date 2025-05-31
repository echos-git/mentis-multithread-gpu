import os
import sys
import json
import time
import logging
import argparse
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime

# Assuming single_file_processor.py is in the same directory or accessible in PYTHONPATH
try:
    from single_file_processor import process_cad_file_sequentially
    SINGLE_FILE_PROCESSOR_AVAILABLE = True
except ImportError as e:
    print(f"Critical Error: Could not import 'process_cad_file_sequentially' from 'single_file_processor.py'. Error: {e}")
    print("Please ensure 'single_file_processor.py' is in the same directory or in PYTHONPATH.")
    SINGLE_FILE_PROCESSOR_AVAILABLE = False
    # Define a dummy function to allow script structure to be parsed if needed, but it will fail at runtime.
    def process_cad_file_sequentially(*args, **kwargs):
        raise ImportError("process_cad_file_sequentially could not be imported.")

try:
    from gpu_memory import cleanup_gpu_resources
    GPU_MEMORY_CLEANUP_AVAILABLE = True
except ImportError:
    GPU_MEMORY_CLEANUP_AVAILABLE = False
    def cleanup_gpu_resources():
        logging.warning("cleanup_gpu_resources not available from gpu_memory.py.")


def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Setup comprehensive logging for the batch process."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"batch_processing_{timestamp}.log"

    # Define a more detailed formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(process)d - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
    )
    
    # File handler for all logs (DEBUG level)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG) 
    
    # Console handler for important messages (INFO or user-defined level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter) # Using the same detailed formatter
    try:
        console_handler.setLevel(getattr(logging, log_level.upper()))
    except AttributeError:
        console_handler.setLevel(logging.INFO)
        logging.warning(f"Invalid log level '{log_level}'. Defaulting to INFO for console.")

    # Configure the root logger
    # Remove any existing handlers to avoid duplicate logs if script is re-run in same session (e.g. in IPython)
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler])
    
    # Return a specific logger for this script
    return logging.getLogger("BatchProcessor")


class SequentialProgressTracker:
    """Tracks processing progress for sequential batch jobs."""
    
    def __init__(self, output_dir: Path, resume: bool = False):
        self.output_dir = output_dir
        self.progress_file = self.output_dir / "sequential_processing_progress.json"
        self.stats_file = self.output_dir / "sequential_processing_stats.json"
        
        self.processed_files: Set[str] = set()
        self.failed_files: Set[str] = set()
        self.processing_stats: Dict[str, Any] = {
            "total_files_processed_successfully": 0,
            "total_files_failed": 0,
            "total_processing_time_seconds": 0.0,
            "start_time": None,
            "end_time": None,
            "individual_file_times": {}, # file_path: time_taken
            "error_summary": {} # error_type: count
        }
        self.current_file_index = 0
        self.total_files_to_process = 0

        if resume and self.progress_file.exists():
            self._load_progress()
        else:
            self.processing_stats["start_time"] = datetime.now().isoformat()

    def _load_progress(self):
        logger = logging.getLogger("BatchProcessor.ProgressTracker")
        try:
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
            self.processed_files = set(data.get("processed_files", []))
            self.failed_files = set(data.get("failed_files", []))
            # Load stats carefully, allowing defaults if keys are missing
            loaded_stats = data.get("processing_stats", {})
            self.processing_stats["total_files_processed_successfully"] = loaded_stats.get("total_files_processed_successfully", 0)
            self.processing_stats["total_files_failed"] = loaded_stats.get("total_files_failed", 0)
            self.processing_stats["total_processing_time_seconds"] = loaded_stats.get("total_processing_time_seconds", 0.0)
            self.processing_stats["start_time"] = loaded_stats.get("start_time", datetime.now().isoformat()) # Keep original start time
            self.processing_stats["individual_file_times"] = loaded_stats.get("individual_file_times", {})
            self.processing_stats["error_summary"] = loaded_stats.get("error_summary", {})
            
            logger.info(f"Resumed progress: {len(self.processed_files)} processed, {len(self.failed_files)} failed.")
        except Exception as e:
            logger.warning(f"Could not load progress from {self.progress_file}: {e}. Starting fresh.")
            self.processing_stats["start_time"] = datetime.now().isoformat()


    def _save_progress(self):
        logger = logging.getLogger("BatchProcessor.ProgressTracker")
        data_to_save = {
            "processed_files": list(self.processed_files),
            "failed_files": list(self.failed_files),
            "processing_stats": self.processing_stats,
            "last_update": datetime.now().isoformat()
        }
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(data_to_save, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress to {self.progress_file}: {e}")

    def start_batch(self, files_to_process: List[str]):
        self.total_files_to_process = len(files_to_process)
        self.current_file_index = 0
        if self.processing_stats["start_time"] is None: # Should be set in init
            self.processing_stats["start_time"] = datetime.now().isoformat()

    def record_success(self, file_path: str, processing_time: float):
        self.processed_files.add(file_path)
        self.processing_stats["total_files_processed_successfully"] += 1
        self.processing_stats["total_processing_time_seconds"] += processing_time
        self.processing_stats["individual_file_times"][file_path] = processing_time
        self._save_progress()

    def record_failure(self, file_path: str, error_message: str, processing_time: float = 0.0):
        self.failed_files.add(file_path)
        self.processing_stats["total_files_failed"] += 1
        self.processing_stats["total_processing_time_seconds"] += processing_time # Time spent before failure
        self.processing_stats["individual_file_times"][file_path] = processing_time
        
        # Simplified error summary (first line of error message as type)
        error_type = error_message.split('\n')[0][:100] # Cap length
        self.processing_stats["error_summary"][error_type] = self.processing_stats["error_summary"].get(error_type, 0) + 1
        self._save_progress()

    def get_summary(self) -> str:
        successful = self.processing_stats["total_files_processed_successfully"]
        failed = self.processing_stats["total_files_failed"]
        total_attempted = successful + failed
        total_time_s = self.processing_stats["total_processing_time_seconds"]
        
        avg_time_per_file = total_time_s / total_attempted if total_attempted > 0 else 0
        
        summary_lines = [
            "Batch Processing Summary:",
            f"  Successfully processed: {successful}",
            f"  Failed to process:      {failed}",
            f"  Total attempted:        {total_attempted}",
            f"  Total processing time:  {timedelta(seconds=total_time_s)}",
            f"  Average time per file:  {avg_time_per_file:.2f} seconds"
        ]
        if self.processing_stats["error_summary"]:
            summary_lines.append("  Error Summary:")
            for error, count in self.processing_stats["error_summary"].items():
                summary_lines.append(f"    - '{error}': {count} times")
        return "\n".join(summary_lines)

    def complete_batch(self):
        self.processing_stats["end_time"] = datetime.now().isoformat()
        self._save_progress() # Final save


def collect_cad_files(data_dir: str, file_extension=".py") -> List[str]:
    """Recursively collects all files with the given extension from the data directory."""
    logger = logging.getLogger("BatchProcessor.FileCollector")
    data_path = Path(data_dir)
    if not data_path.is_dir():
        logger.error(f"Data directory not found or is not a directory: {data_dir}")
        return []
    
    collected_files = []
    for item in data_path.rglob(f"*{file_extension}"): # rglob for recursive
        if item.is_file():
            collected_files.append(str(item.resolve()))
            
    logger.info(f"Collected {len(collected_files)} files from {data_dir}")
    return sorted(collected_files) # Sort for consistent processing order


def main():
    parser = argparse.ArgumentParser(description="Batch process CadQuery files sequentially using single_file_processor.")
    parser.add_argument("--data-dir", required=True, help="Path to the root directory containing CadQuery (.py) files.")
    parser.add_argument("--output-dir", required=True, help="Root directory for all processed outputs.")
    parser.add_argument("--num-points", type=int, default=10000, help="Points per point cloud for each file.")
    parser.add_argument("--image-width", type=int, default=1024, help="Width of rendered images.")
    parser.add_argument("--image-height", type=int, default=1024, help="Height of rendered images.")
    parser.add_argument("--force-overwrite", action="store_true", help="Overwrite existing output files for each CadQuery file.")
    parser.add_argument("--use-global-lighting", action="store_true", help="Use global lighting for rendering instead of view-specific.")
    parser.add_argument("--resume", action="store_true", help="Resume processing from the last saved progress.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Console logging level.")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of files to process (for testing).")

    args = parser.parse_args()

    if not SINGLE_FILE_PROCESSOR_AVAILABLE:
        sys.exit("Exiting due to missing single_file_processor.py.")

    output_root_dir = Path(args.output_dir)
    log_dir = output_root_dir / "_batch_logs"
    logger = setup_logging(log_dir, args.log_level)

    logger.info("Starting Sequential Batch CAD Processor.")
    logger.info(f"  Data directory: {args.data_dir}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Force overwrite: {args.force_overwrite}")
    logger.info(f"  Resume: {args.resume}")
    if args.max_files is not None:
        logger.info(f"  Processing at most {args.max_files} files.")

    tracker = SequentialProgressTracker(output_root_dir, args.resume)
    
    all_cad_files = collect_cad_files(args.data_dir)

    if not all_cad_files:
        logger.info("No CadQuery files found to process. Exiting.")
        sys.exit(0)

    files_to_process = []
    if args.resume:
        for f_path in all_cad_files:
            # Process if not already processed or failed (failed files are re-attempted on resume)
            if f_path not in tracker.processed_files:
                 files_to_process.append(f_path)
        logger.info(f"Resuming. Found {len(all_cad_files)} total files, {len(files_to_process)} remaining to process or re-attempt failed.")
    else:
        files_to_process = all_cad_files
        logger.info(f"Starting new run. {len(files_to_process)} files to process.")

    if args.max_files is not None:
        files_to_process = files_to_process[:args.max_files]
        logger.info(f"Limiting processing to the first {len(files_to_process)} files due to --max-files argument.")
        
    tracker.start_batch(files_to_process) # Inform tracker about the total for this run

    image_size = (args.image_width, args.image_height)

    overall_start_time = time.time()

    try:
        for i, cad_file_path_str in enumerate(files_to_process):
            tracker.current_file_index = i
            file_path_obj = Path(cad_file_path_str)
            
            logger.info(f"Processing file {i+1}/{len(files_to_process)}: {file_path_obj.name}")
            file_process_start_time = time.time()

            try:
                result = process_cad_file_sequentially(
                    cad_script_path=cad_file_path_str,
                    base_output_dir=args.output_dir, # single_file_processor creates its own subdir based on file_stem
                    num_points=args.num_points,
                    image_size=image_size,
                    force_overwrite=args.force_overwrite,
                    use_global_lighting=args.use_global_lighting
                )
                
                file_processing_time = result.get('processing_time', time.time() - file_process_start_time)

                if result.get('success', False):
                    logger.info(f"Successfully processed {file_path_obj.name} in {file_processing_time:.2f}s. Output: {result.get('output_dir')}")
                    tracker.record_success(cad_file_path_str, file_processing_time)
                else:
                    error_msg = result.get('error', 'Unknown error from single_file_processor')
                    logger.error(f"Failed to process {file_path_obj.name}. Error: {error_msg}")
                    if result.get('traceback'):
                        logger.debug(f"Traceback for {file_path_obj.name}:\n{result.get('traceback')}")
                    tracker.record_failure(cad_file_path_str, error_msg, file_processing_time)

            except Exception as e_inner: # Catch errors from the call to process_cad_file_sequentially itself
                file_processing_time = time.time() - file_process_start_time
                logger.critical(f"Critical error during processing of {file_path_obj.name}: {e_inner}", exc_info=True)
                tracker.record_failure(cad_file_path_str, f"Critical batch error: {str(e_inner)}", file_processing_time)
            
            # Optional: Log memory usage or other stats periodically
            if (i + 1) % 100 == 0: # Every 100 files
                logger.info(f"--- Processed {i+1} files. Current progress: ---")
                logger.info(tracker.get_summary())


    except KeyboardInterrupt:
        logger.warning("Batch processing interrupted by user (KeyboardInterrupt).")
    except Exception as e_outer:
        logger.critical(f"An unexpected error occurred during batch processing: {e_outer}", exc_info=True)
    finally:
        logger.info("Batch processing finished or was interrupted.")
        tracker.complete_batch()
        logger.info(tracker.get_summary())
        
        if GPU_MEMORY_CLEANUP_AVAILABLE:
            logger.info("Attempting final GPU resource cleanup...")
            try:
                cleanup_gpu_resources()
                logger.info("GPU resource cleanup successful.")
            except Exception as e_cleanup:
                logger.error(f"Error during final GPU resource cleanup: {e_cleanup}")

        overall_end_time = time.time()
        logger.info(f"Total batch script execution time: {timedelta(seconds=(overall_end_time - overall_start_time))}")
        logger.info(f"Log file saved to: {log_dir}")

if __name__ == "__main__":
    if not SINGLE_FILE_PROCESSOR_AVAILABLE:
        # Message already printed, just exit
        sys.exit(1)
    main() 