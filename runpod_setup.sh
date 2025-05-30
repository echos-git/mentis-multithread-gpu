#!/bin/bash
# RunPod Setup Script for GPU-Accelerated CAD Processing
# Usage: bash runpod_setup.sh

set -e

echo "🚀 Starting RunPod CAD Processing Setup..."

# Update system
echo "📦 Updating system packages..."
apt-get update && apt-get install -y git wget curl build-essential

# Verify GPU
echo "🔍 Verifying GPU setup..."
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

# Install dependencies
echo "📚 Installing CAD processing dependencies..."
pip install --upgrade pip
pip install pynvml nvidia-ml-py3 cupy-cuda12x
pip install PyVista vtk pyvista[all] 
pip install trimesh[easy] open3d
pip install cadquery==2.3.0 opencascade-python
pip install huggingface_hub datasets

# Install our GPU-accelerated modules
pip install -r requirements_gpu.txt

# Test GPU setup
echo "🧪 Testing GPU acceleration..."
python test_gpu_setup.py

# Create processing directories
echo "📁 Setting up workspace..."
mkdir -p /workspace/{data_input,data_output,checkpoints,logs}

echo "✅ Setup complete! Ready for CAD processing."
echo ""
echo "Next steps:"
echo "1. Download your CAD data to /workspace/data_input/"
echo "2. Run: python gpu_workers.py --input /workspace/data_input --output /workspace/data_output"
echo "3. Massage data: python ../datamassage.py --input_dir /workspace/data_output"
echo "4. Upload to HF: huggingface-cli upload your-dataset-name /workspace/final_dataset.parquet" 