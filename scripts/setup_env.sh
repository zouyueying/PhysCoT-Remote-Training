#!/bin/bash
# PhysCoT Remote Training — Environment Setup
# Tested with: Python 3.10+, CUDA 12.x, 8x H100
set -e

echo "=== PhysCoT Environment Setup ==="

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.57.1
pip install peft==0.17.1
pip install llamafactory==0.9.4
pip install deepspeed
pip install accelerate
pip install qwen-vl-utils
pip install scikit-learn
pip install flash-attn --no-build-isolation 2>/dev/null || echo "[WARN] flash-attn failed, will use sdpa"

python3 -c "
import torch, transformers, deepspeed
print(f'PyTorch:      {torch.__version__}')
print(f'CUDA:         {torch.cuda.is_available()} ({torch.cuda.device_count()} GPUs)')
print(f'Transformers: {transformers.__version__}')
print(f'DeepSpeed:    {deepspeed.__version__}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem/1e9:.0f}GB)')
"
echo "=== Setup Complete ==="
