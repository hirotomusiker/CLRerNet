# TensorRT Deployment Guide (Jetson AGX Orin)

> Author: [@seyedehsan-taheri](https://github.com/seyedehsan-taheri)  
> Tested on Jetson AGX Orin (JetPack 6, TensorRT 10.3, CUDA 12.3)  
> Branch: `feature/tensorrt-export-jetson`  

---

## 🚀 Overview

This document explains how to **export, convert, and deploy the CLRerNet model on NVIDIA Jetson devices** using TensorRT.
It covers every stage from PyTorch → ONNX → TensorRT `.engine` conversion and inference testing.

### ✳️ Why This Is Needed
While the official repository supports PyTorch inference via `demo/image_demo.py`,  
there is no documented workflow for **high-performance deployment** using TensorRT.  
This guide introduces a stable export and runtime pipeline tested on embedded GPUs (sm_87 architecture).

---

## 🧩 1. Environment Setup

### Tested Environment

| Component | Version / Notes |
|------------|----------------|
| JetPack | 6.0 (L4T R36.4.7, CUDA 12.3) |
| TensorRT | 10.3.0 |
| Python | 3.10 |
| PyTorch | 2.3.0+cu121 |
| mmcv | 2.1.0 |
| mmdet | 3.3.0 |
| mmengine | 0.10.5 |
| numpy | `<2.0` |
| OpenCV | `opencv-python-headless<4.8` |

### Install Dependencies
```bash
sudo apt install python3-pip libopencv-dev libopenblas-dev
pip install --upgrade pip setuptools wheel

pip install numpy<2.0 opencv-python-headless<4.8
pip install mmengine==0.10.5
pip install mmcv==2.1.0
pip install mmdet==3.3.0
```

⚠️ Important:
MMCV must be installed with ops (MMCV_WITH_OPS=1), otherwise you’ll get:
ModuleNotFoundError: No module named 'mmcv._ext'

If building from source:
MMCV_WITH_OPS=1 pip install -v -e ./mmcv


🧠 2. Export PyTorch → ONNX
Step 1: Use the Export Script

A new export script is provided at:

tools/export_onnx.py


Run:

python tools/export_onnx.py


Expected output:

✅ Exported: clrernet_raw.onnx


This script:

Builds CLRerNet using MMDet’s registry (MODELS.build(cfg.model))

Bypasses model.forward() and calls:

extract_feat(x)

bbox_head.forward(features)

Exports the 4 core outputs:

cls_logits, anchor_params, lengths, xs


Removes NMS and post-processing for ONNX compatibility.

⚙️ 3. Convert ONNX → TensorRT Engine
Step 1: Create or use deploy config

File:

configs/deploy/trt_config_standalone.py


Defines:
backend_config = dict(
    type='tensorrt',
    common_config=dict(fp16_mode=True),
    model_inputs=[dict(input_shapes=dict(
        input=dict(min_shape=[1,3,320,800],
                   opt_shape=[1,3,320,800],
                   max_shape=[1,3,320,800])
    ))],
    
    tensorrt_params=dict(cuda_arch='sm_87', workspace_size=1<<30)
)

onnx_config = dict(
    input_shape=None,
    output_names=['cls_logits','anchor_params','lengths','xs'],
    opset_version=16
)

Step 2: Convert with MMDeploy TensorRT backend

Run:

python -m mmdeploy.backend.tensorrt.onnx2tensorrt \
    configs/deploy/trt_config_standalone.py \
    clrernet_raw.onnx \
    clrernet_trt.engine


Expected:

✅ TensorRT engine successfully built: clrernet_trt.engine

🧪 4. Validate TensorRT Engine (Inference)

A minimal test script is included:

tools/test_trt_engine.py


Usage:

python tools/test_trt_engine.py \
  --engine clrernet_trt.engine \
  --image demo/demo.jpg


Example output:

TENSORS: ['input', 'cls_logits', 'anchor_params', 'lengths', 'xs']
shapes: [(1,192,2), (1,192,3), (1,192,1), (1,192,72)]
stats: cls_max=3.55 len_max=0.98 xs_mean=0.44
✅ Done.


This confirms the exported TensorRT engine runs successfully on-device.

🧮 5. Optional: Visualize ONNX/TensorRT Output

A helper script is available:

tools/onnx_infer_image.py


Command:

python tools/onnx_infer_image.py \
  --onnx clrernet_raw.onnx \
  --image demo/demo.jpg \
  --out vis_onnx.jpg \
  --conf 0.5 \
  --len_thr 0.2


This performs:

BGR crop [270:590, 0:1640]

Resize to (800×320)

Lane reconstruction using xs, lengths, and cls_logits

Visualization saved as vis_onnx.jpg.

⚡ 6. Known Issues & Fixes
Issue	Cause	Fix
mmcv._ext not found	Missing compiled ops	Reinstall MMCV_WITH_OPS=1
KeyError: ROIGather	Custom module not in registry	Ensure libs/models/__init__.py imports all components
RuntimeError: data dependence	ONNX tracer failing on Python loops	Simplified forward using Wrapper class
num_io_tensors vs num_bindings	TensorRT v10 API change	Updated inference script
numpy 2.x conflicts	mmcv incompatible with NumPy ≥2.0	Force numpy<2.0
Checkpoint mismatch	EMA weights missing segmentation loss	Set loss_seg.weight=0.0 in config
🧾 7. Deployment Summary
Stage	Input	Output	Tool
PyTorch → ONNX	.pth	clrernet_raw.onnx	tools/export_onnx.py
ONNX → TRT Engine	.onnx	clrernet_trt.engine	onnx2tensorrt
Runtime Inference	.engine	lane tensors	tools/test_trt_engine.py
🔧 8. Example Full Pipeline (Jetson AGX Orin)
# Activate venv
source ~/venv/bin/activate

# Export model to ONNX
python tools/export_onnx.py

# Convert to TensorRT engine
python -m mmdeploy.backend.tensorrt.onnx2tensorrt \
  configs/deploy/trt_config_standalone.py \
  clrernet_raw.onnx \
  clrernet_trt.engine

# Run inference
python tools/test_trt_engine.py \
  --engine clrernet_trt.engine \
  --image demo/demo.jpg

✅ 9. Results
Model	Backend	Precision	Device	FPS	Notes
CLRerNet-DLA34	PyTorch	FP32	RTX 4080	~20	Original
CLRerNet-DLA34	TensorRT	FP16	Jetson Orin	~105 FPS	Optimized
📎 10. References

CLRerNet Official Repo

MMDetection v3.3.0

MMDeploy v1.3.1

NVIDIA TensorRT Documentation

CULane Dataset

🏁 Maintainer Notes

This deployment path intentionally excludes post-processing (NMS, lane fitting) from the ONNX graph to ensure export stability.
It outputs raw lane tensor predictions (cls_logits, anchor_params, lengths, xs)
that can be decoded externally for visualization or downstream logic.

For questions or collaboration:

📧 Contact: [ehsan.taheri1400@gmail.com]

PR branch: feature/tensorrt-export-jetson
