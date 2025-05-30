#!/usr/bin/env python3
"""
Production CAD Dataset Processing Script
Processes the complete CAD-recode v1.5 dataset with GPU acceleration
"""

import os
import sys
import json
import time
import logging
import argparse
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Setup logging
def setup_logging(output_dir: str, log_level: str = "INFO"):
    """Setup comprehensive logging"""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler for all logs
    file_handler = logging.FileHandler(log_dir / f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler for important messages
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Setup root logger
    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler])
    
    return logging.getLogger(__name__)

def setup_environment():
    """Setup PyVista for headless rendering"""
    import pyvista as pv
    pv.start_xvfb()
    pv.set_plot_theme("document")
    os.environ['PYVISTA_OFF_SCREEN'] = 'true'
    os.environ['PYVISTA_USE_PANEL'] = 'false'

class ProgressTracker:
    """Track processing progress with statistics"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.progress_file = self.output_dir / "processing_progress.json"
        self.stats_file = self.output_dir / "processing_stats.json"
        
        # Load existing progress
        self.progress = self.load_progress()
        self.stats = self.load_stats()
        
    def load_progress(self) -> Dict[str, Any]:
        """Load processing progress from file"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {
            "processed_files": set(),
            "failed_files": set(),
            "current_batch": 0,
            "total_files_processed": 0,
            "total_files_failed": 0,
            "start_time": None,
            "last_update": None
        }
    
    def load_stats(self) -> Dict[str, Any]:
        """Load processing statistics"""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {
            "processing_times": [],
            "batch_times": [],
            "memory_usage": [],
            "error_counts": {},
            "file_sizes": {
                "stl": [],
                "pointcloud": [],
                "renders": []
            }
        }
    
    def save_progress(self):
        """Save current progress"""
        # Convert sets to lists for JSON serialization
        progress_copy = self.progress.copy()
        progress_copy["processed_files"] = list(self.progress["processed_files"])
        progress_copy["failed_files"] = list(self.progress["failed_files"])
        progress_copy["last_update"] = datetime.now().isoformat()
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress_copy, f, indent=2)
    
    def save_stats(self):
        """Save current statistics"""
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def start_processing(self):
        """Mark start of processing"""
        if not self.progress["start_time"]:
            self.progress["start_time"] = datetime.now().isoformat()
        self.save_progress()
    
    def add_processed_file(self, file_path: str, processing_time: float, 
                          file_sizes: Dict[str, int]):
        """Add successfully processed file"""
        self.progress["processed_files"].add(file_path)
        self.progress["total_files_processed"] += 1
        
        self.stats["processing_times"].append(processing_time)
        for file_type, size in file_sizes.items():
            if file_type in self.stats["file_sizes"]:
                self.stats["file_sizes"][file_type].append(size)
        
        self.save_progress()
        self.save_stats()
    
    def add_failed_file(self, file_path: str, error: str):
        """Add failed file"""
        self.progress["failed_files"].add(file_path)
        self.progress["total_files_failed"] += 1
        
        error_type = type(error).__name__ if isinstance(error, Exception) else "Unknown"
        self.stats["error_counts"][error_type] = self.stats["error_counts"].get(error_type, 0) + 1
        
        self.save_progress()
        self.save_stats()
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get current progress summary"""
        total_processed = self.progress["total_files_processed"]
        total_failed = self.progress["total_files_failed"]
        
        if self.stats["processing_times"]:
            avg_time = np.mean(self.stats["processing_times"])
            std_time = np.std(self.stats["processing_times"])
        else:
            avg_time = std_time = 0
        
        return {
            "total_processed": total_processed,
            "total_failed": total_failed,
            "success_rate": total_processed / (total_processed + total_failed) * 100 if (total_processed + total_failed) > 0 else 0,
            "avg_processing_time": avg_time,
            "std_processing_time": std_time,
            "estimated_time_remaining": None  # Will be calculated based on remaining files
        }

def collect_dataset_files(data_dir: str, start_batch: int = 0, 
                         end_batch: Optional[int] = None) -> List[str]:
    """Collect all CadQuery files from dataset"""
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    
    all_files = []
    batch_dirs = sorted(train_dir.glob("batch_*"))
    
    # Filter by batch range
    if end_batch is not None:
        batch_dirs = [b for b in batch_dirs if start_batch <= int(b.name.split('_')[1]) <= end_batch]
    else:
        batch_dirs = [b for b in batch_dirs if int(b.name.split('_')[1]) >= start_batch]
    
    for batch_dir in batch_dirs:
        py_files = list(batch_dir.glob("*.py"))
        all_files.extend([str(f) for f in py_files])
    
    return all_files

def process_single_file(file_path: str, output_base_dir: str, 
                       num_points: int = 10000) -> Dict[str, Any]:
    """Process a single CadQuery file"""
    from gpu_workers import full_pipeline_worker_gpu
    
    start_time = time.time()
    
    try:
        # Process the file - this will create all outputs in structured directories
        result = full_pipeline_worker_gpu(
            cadquery_file_path=file_path,
            output_base_dir=output_base_dir,
            n_points=num_points,
            use_global_lighting=False,
            force_overwrite=True,
            device_id=0
        )
        
        processing_time = time.time() - start_time
        
        if result.success:
            # Get actual output paths from result data
            stl_path = result.data.get('stl_path')
            pointcloud_path = result.data.get('pointcloud_path')
            rendered_views = result.data.get('rendered_views', {})
            
            # Calculate file sizes
            file_sizes = {}
            
            if stl_path and Path(stl_path).exists():
                file_sizes["stl"] = Path(stl_path).stat().st_size
            else:
                file_sizes["stl"] = 0
                
            if pointcloud_path and Path(pointcloud_path).exists():
                file_sizes["pointcloud"] = Path(pointcloud_path).stat().st_size
            else:
                file_sizes["pointcloud"] = 0
            
            # Calculate total render file sizes
            render_size = 0
            render_count = 0
            render_dir = None
            
            if rendered_views:
                for view_name, view_path in rendered_views.items():
                    if Path(view_path).exists():
                        render_size += Path(view_path).stat().st_size
                        render_count += 1
                        if render_dir is None:
                            render_dir = str(Path(view_path).parent)
            
            file_sizes["renders"] = render_size
            
            return {
                "success": True,
                "file_path": file_path,
                "processing_time": processing_time,
                "file_sizes": file_sizes,
                "outputs": {
                    "stl": stl_path,
                    "pointcloud": pointcloud_path,
                    "renders": render_dir,
                    "render_count": render_count,
                    "rendered_views": rendered_views
                },
                "pipeline_results": result.data.get('pipeline_results', {})
            }
        else:
            return {
                "success": False,
                "file_path": file_path,
                "processing_time": processing_time,
                "error": result.error,
                "pipeline_results": result.data.get('pipeline_results', {}) if result.data else {}
            }
            
    except Exception as e:
        processing_time = time.time() - start_time
        return {
            "success": False,
            "file_path": file_path,
            "processing_time": processing_time,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def process_batch(files: List[str], output_dir: str, tracker: ProgressTracker,
                 batch_size: int = 100, max_workers: int = 4, 
                 num_points: int = 10000) -> Dict[str, Any]:
    """Process a batch of files with parallel processing"""
    logger = logging.getLogger(__name__)
    
    batch_start_time = time.time()
    batch_results = {
        "successful": 0,
        "failed": 0,
        "total_time": 0,
        "avg_time_per_file": 0
    }
    
    logger.info(f"Starting batch processing: {len(files)} files, {max_workers} workers")
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all files
        future_to_file = {
            executor.submit(process_single_file, file_path, output_dir, num_points): file_path
            for file_path in files
        }
        
        # Process completed futures
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            
            try:
                result = future.result()
                
                if result["success"]:
                    batch_results["successful"] += 1
                    tracker.add_processed_file(
                        file_path, 
                        result["processing_time"],
                        result["file_sizes"]
                    )
                    logger.info(f"✓ Processed {Path(file_path).name} in {result['processing_time']:.2f}s")
                else:
                    batch_results["failed"] += 1
                    tracker.add_failed_file(file_path, result["error"])
                    logger.warning(f"✗ Failed {Path(file_path).name}: {result['error']}")
                    
            except Exception as e:
                batch_results["failed"] += 1
                tracker.add_failed_file(file_path, str(e))
                logger.error(f"✗ Exception processing {Path(file_path).name}: {str(e)}")
    
    batch_time = time.time() - batch_start_time
    batch_results["total_time"] = batch_time
    batch_results["avg_time_per_file"] = batch_time / len(files) if files else 0
    
    logger.info(f"Batch completed: {batch_results['successful']}/{len(files)} successful, {batch_time:.2f}s total")
    
    return batch_results

def monitor_gpu_memory():
    """Monitor GPU memory usage"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        return {
            "total_gb": info.total / 1024**3,
            "used_gb": info.used / 1024**3,
            "free_gb": info.free / 1024**3,
            "utilization_percent": (info.used / info.total) * 100
        }
    except Exception:
        return None

def main():
    """Main processing function"""
    parser = argparse.ArgumentParser(description="Process CAD-recode dataset with GPU acceleration")
    parser.add_argument("--data-dir", required=True, help="Path to CAD-recode dataset")
    parser.add_argument("--output-dir", required=True, help="Output directory for processed files")
    parser.add_argument("--start-batch", type=int, default=0, help="Starting batch number")
    parser.add_argument("--end-batch", type=int, help="Ending batch number (inclusive)")
    parser.add_argument("--batch-size", type=int, default=100, help="Files per processing batch")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum parallel workers")
    parser.add_argument("--num-points", type=int, default=10000, help="Points per point cloud")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(args.output_dir, args.log_level)
    setup_environment()
    
    # Initialize progress tracker
    tracker = ProgressTracker(args.output_dir)
    tracker.start_processing()
    
    logger.info("🚀 Starting CAD Dataset Processing")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Batch range: {args.start_batch} - {args.end_batch or 'end'}")
    logger.info(f"Processing config: {args.batch_size} files/batch, {args.max_workers} workers")
    
    # Monitor GPU
    gpu_info = monitor_gpu_memory()
    if gpu_info:
        logger.info(f"GPU Memory: {gpu_info['used_gb']:.1f}/{gpu_info['total_gb']:.1f} GB ({gpu_info['utilization_percent']:.1f}%)")
    
    try:
        # Collect all files to process
        logger.info("📁 Collecting dataset files...")
        all_files = collect_dataset_files(args.data_dir, args.start_batch, args.end_batch)
        
        # Filter out already processed files if resuming
        if args.resume:
            remaining_files = [f for f in all_files if f not in tracker.progress["processed_files"]]
            logger.info(f"📊 Resume mode: {len(all_files)} total, {len(remaining_files)} remaining")
            all_files = remaining_files
        
        logger.info(f"📊 Found {len(all_files)} files to process")
        
        if not all_files:
            logger.info("✅ No files to process")
            return
        
        # Process in batches
        processed_batches = 0
        total_batches = (len(all_files) + args.batch_size - 1) // args.batch_size
        
        for i in range(0, len(all_files), args.batch_size):
            batch_files = all_files[i:i + args.batch_size]
            processed_batches += 1
            
            logger.info(f"🔄 Processing batch {processed_batches}/{total_batches} ({len(batch_files)} files)")
            
            # Process batch
            batch_results = process_batch(
                batch_files, 
                args.output_dir, 
                tracker,
                args.batch_size,
                args.max_workers,
                args.num_points
            )
            
            # Log batch results
            logger.info(f"✓ Batch {processed_batches} complete: {batch_results['successful']}/{len(batch_files)} successful")
            
            # Update progress summary
            progress = tracker.get_progress_summary()
            logger.info(f"📊 Total progress: {progress['total_processed']} processed, {progress['total_failed']} failed, {progress['success_rate']:.1f}% success rate")
            
            # Monitor GPU memory periodically
            if processed_batches % 5 == 0:  # Every 5 batches
                gpu_info = monitor_gpu_memory()
                if gpu_info:
                    logger.info(f"🖥️  GPU Memory: {gpu_info['used_gb']:.1f}/{gpu_info['total_gb']:.1f} GB ({gpu_info['utilization_percent']:.1f}%)")
        
        # Final summary
        final_progress = tracker.get_progress_summary()
        logger.info("🎉 Processing complete!")
        logger.info(f"📊 Final Results:")
        logger.info(f"   • Total processed: {final_progress['total_processed']}")
        logger.info(f"   • Total failed: {final_progress['total_failed']}")
        logger.info(f"   • Success rate: {final_progress['success_rate']:.1f}%")
        logger.info(f"   • Average time per file: {final_progress['avg_processing_time']:.2f}s")
        
        if tracker.progress["start_time"]:
            start_time = datetime.fromisoformat(tracker.progress["start_time"])
            total_time = datetime.now() - start_time
            logger.info(f"   • Total processing time: {total_time}")
        
    except KeyboardInterrupt:
        logger.info("⚠️  Processing interrupted by user")
        logger.info("💾 Progress has been saved. Use --resume to continue")
    except Exception as e:
        logger.error(f"❌ Processing failed: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 