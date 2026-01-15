import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import numpy as np
import time
import argparse
import os
import sys

clrernet_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CLRerNet'))
if clrernet_path not in sys.path:
    sys.path.insert(0, clrernet_path)

try:
    from libs.utils.visualizer import visualize_lanes
except ImportError:
    print(f"Error: Could not import visualize_lanes...")
    sys.exit(1)

# === Model Parameters ===
IMG_W = 800
IMG_H = 320
NUM_POINTS = 72
NUM_PRIORS = 192
NUM_OFFSETS = 73
N_STRIPS = NUM_OFFSETS - 1
CONF_THRESHOLD = 0.43
# Original Dims (Not needed here anymore, read from preprocess)
# ORI_IMG_W = 1640
# ORI_IMG_H = 590
# CUT_HEIGHT = 270

prior_ys = np.linspace(1, 0, num=NUM_OFFSETS, dtype=np.float64)

# --- TRT Helper Functions (Unchanged) ---
class HostDeviceMem(object):
    # ... (Keep the class definition exactly as before) ...
    def __init__(self, host_mem, device_mem, name, dtype, shape):
        self.host = host_mem
        self.device = device_mem
        self.name = name
        self.dtype = dtype
        self.shape = shape
    def __str__(self):
        return f"Tensor '{self.name}': Shape={self.shape}, Dtype={self.dtype}\nHost:\n{self.host}\nDevice:\n{self.device}"
    def __repr__(self):
        return self.__str__()

def allocate_buffers_by_name(engine, context):
    inputs = {}
    outputs = {}
    bindings = [0] * engine.num_io_tensors
    stream = cuda.Stream()
    print("   Allocating Buffers (by name):")
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        is_input = engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
        shape = tuple(context.get_tensor_shape(tensor_name))
        if -1 in shape:
             print(f"      Warning: Tensor '{tensor_name}' has dynamic dimensions {shape}. Using known shape.")
             if is_input: shape = (1, 3, 320, 800)
             else:
                  if tensor_name == 'cls_logits': shape = (1, NUM_PRIORS, 2)
                  elif tensor_name == 'anchor_params': shape = (1, NUM_PRIORS, 3)
                  elif tensor_name == 'lengths': shape = (1, NUM_PRIORS, 1)
                  elif tensor_name == 'xs': shape = (1, NUM_PRIORS, NUM_POINTS) # Corrected to 72 based on logs
                  else: raise ValueError(f"Unknown output binding: {tensor_name}")
             print(f"         Resolved shape to: {shape}")
        size = trt.volume(shape)
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings[i] = int(device_mem)
        mem_info = HostDeviceMem(host_mem, device_mem, tensor_name, dtype, shape)
        if is_input: inputs[tensor_name] = mem_info
        else: outputs[tensor_name] = mem_info
        print(f"      Allocated for '{tensor_name}': Shape={shape}, Dtype={dtype}, Size={host_mem.nbytes} bytes")
    return inputs, outputs, bindings, stream

def do_inference_v2_by_name(context, bindings, inputs_dict, outputs_dict, stream):
    for name, inp in inputs_dict.items():
        cuda.memcpy_htod_async(inp.device, inp.host, stream)
    executed = False
    # Simplified execution call based on last successful log
    if hasattr(context, "execute_v2"):
        try:
             context.execute_v2(bindings=bindings) # Use synchronous execute_v2
             print(f"      Used TensorRT execution method: execute_v2")
             executed = True
        except Exception as e:
             print(f"      Attempt to run execute_v2 failed: {e}")
    if not executed and hasattr(context, "execute_async"): # Fallback
         try:
              context.execute_async(bindings=bindings, stream_handle=stream.handle)
              print(f"      Used TensorRT execution method: execute_async")
              executed = True
         except Exception as e:
              print(f"      Attempt to run execute_async failed: {e}")
    # Add other fallbacks if necessary (execute_async_v2, execute)

    if not executed: raise RuntimeError("No suitable TensorRT execution method found.")

    results = {}
    for name, out in outputs_dict.items():
        cuda.memcpy_dtoh_async(out.host, out.device, stream)
        results[name] = out.host
    stream.synchronize()
    for name, out in outputs_dict.items():
         results[name] = results[name].reshape(out.shape).astype(out.dtype)
    return results
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Test CLRerNet TensorRT engine with scaling.')
    parser.add_argument('img', help='Image file (e.g., ../CLRerNet/demo/demo.jpg)')
    parser.add_argument('engine_file', help='TensorRT engine file (e.g., clrernet_trt.engine)')
    parser.add_argument('--out-file', default='result_trt_scaled.png', help='Path to output file')
    parser.add_argument('--conf-threshold', type=float, default=CONF_THRESHOLD, help='Confidence threshold')
    return parser.parse_args()

def preprocess(img_path):
    """Preprocesses image: read -> resize -> normalize -> transpose -> batch."""
    if not os.path.exists(img_path): raise FileNotFoundError(f"Image not found: {img_path}")
    img = cv2.imread(img_path)
    if img is None: raise ValueError(f"Could not read image: {img_path}")
    src_img_orig = img.copy() # Keep original for final visualization
    orig_h, orig_w = src_img_orig.shape[:2]
    img_resized = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
    img_float = img_resized.astype(np.float32) / 255.0
    img_chw = np.transpose(img_float, (2, 0, 1))
    input_tensor = np.expand_dims(img_chw, axis=0)
    input_tensor = np.ascontiguousarray(input_tensor)
    return input_tensor, src_img_orig, (orig_h, orig_w) # Return original image and its dims

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

 
def postprocess(trt_outputs_dict, conf_threshold):
    """Postprocesses raw TRT outputs to relative lane points [0, 1]."""
    required_keys = ['cls_logits', 'anchor_params', 'lengths', 'xs']
    if not all(key in trt_outputs_dict for key in required_keys): return []

    cls_logits_trt = trt_outputs_dict['cls_logits']
    anchor_params_trt = trt_outputs_dict['anchor_params']
    lengths_trt = trt_outputs_dict['lengths']
    xs_trt = trt_outputs_dict['xs'] # Shape (1, 192, 72)

    batch_size = cls_logits_trt.shape[0]
    if batch_size == 0: return []

    all_scores = softmax(cls_logits_trt, axis=2)[0, :, 1]
    keep_inds = all_scores >= conf_threshold

    scores = all_scores[keep_inds]
    xs = xs_trt[0, keep_inds] # Shape (N_kept, 72)
    lengths = lengths_trt[0, keep_inds]
    anchor_params = anchor_params_trt[0, keep_inds]

    if xs.shape[0] == 0: return []

    final_lanes_relative = [] # List to store lanes with relative coords
    lengths_strips = np.round(lengths * N_STRIPS)

    for lane_xs, lane_param, length, score in zip(xs, anchor_params, lengths_strips, scores):
        start_strip_index = min(max(0, int(round((1.0 - lane_param[0]) * N_STRIPS))), N_STRIPS)
        num_strips_lane = int(round(length[0]))
        end_strip_index = min(start_strip_index + num_strips_lane - 1, N_STRIPS)
        start_strip_index = max(0, start_strip_index)
        end_strip_index = max(start_strip_index, end_strip_index)

        # Extend bottom logic (Simplified/disabled)
        # ...

        # Slice Y coordinates (relative)
        current_lane_ys_rel = prior_ys[start_strip_index : end_strip_index + 1]

        # Slice X coordinates (relative) based on Y indices, handling 72 points
        if lane_xs.shape[0] == NUM_POINTS: # Check against 72
            start_xs_idx = min(start_strip_index, NUM_POINTS - 1)
            end_xs_idx = min(end_strip_index, NUM_POINTS - 1)
            if start_xs_idx > end_xs_idx: continue

            current_lane_xs_rel = lane_xs[start_xs_idx : end_xs_idx + 1]
            # Adjust Y slice to match X slice length
            num_xs_points = current_lane_xs_rel.shape[0]
            if num_xs_points == 0: continue
            actual_end_strip_index = min(start_strip_index + num_xs_points - 1, N_STRIPS)
            current_lane_ys_rel = prior_ys[start_strip_index : actual_end_strip_index + 1]

            if current_lane_xs_rel.shape[0] != current_lane_ys_rel.shape[0]:
                print(f"Warning: Mismatch lengths: xs:{current_lane_xs_rel.shape[0]}, ys:{current_lane_ys_rel.shape[0]}. Skip.")
                continue
        else:
            print(f"Error: Unexpected x-points: {lane_xs.shape[0]}. Expected {NUM_POINTS}. Skip.")
            continue

        # Flip to bottom-up order
        current_lane_xs_rel = np.flipud(current_lane_xs_rel).astype(np.float64)
        current_lane_ys_rel = np.flipud(current_lane_ys_rel)

        if current_lane_xs_rel.size <= 1: continue

      
        # Stack X and Y (already relative)
        points_relative = np.stack(
            (current_lane_xs_rel[:, np.newaxis], current_lane_ys_rel[:, np.newaxis]),
            axis=1
        ).squeeze(axis=2)
        # =================================

        final_lanes_relative.append(points_relative) # Append the (N, 2) array of relative points

    return final_lanes_relative # Return list of arrays with relative [0,1] coordinates
# ---------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()

    # --- 1. Preprocess & Get Original Image/Dims ---
    print("Loading image and preprocessing...")
    try:
        input_tensor, src_img_orig, (orig_h, orig_w) = preprocess(args.img)
        print(f"   Original image size: {orig_w}x{orig_h}")
        print(f"   Input tensor size: {input_tensor.shape}")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        sys.exit(1)

    # --- 2. Load Engine ---
    print(f"Loading TensorRT engine: {args.engine_file}...")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    try:
        with open(args.engine_file, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None: raise RuntimeError("Deserialize failed")
        context = engine.create_execution_context()
        if context is None: raise RuntimeError("Create context failed")
        print(f"   Engine loaded. Num IO Tensors: {engine.num_io_tensors}")
    except Exception as e:
        print(f"Error loading TRT engine: {e}")
        sys.exit(1)

    # --- 3. Allocate Buffers ---
    print("Allocating buffers...")
    try:
        inputs_dict, outputs_dict, bindings_ordered, stream = allocate_buffers_by_name(engine, context)
    except Exception as e:
        print(f"Error allocating buffers: {e}")
        sys.exit(1)

    # --- 4. Copy Input ---
    input_name = 'input'
    if input_name not in inputs_dict: # ... (error check as before) ...
        print(f"Error: Input tensor '{input_name}' not found.")
        sys.exit(1)
    input_mem_info = inputs_dict[input_name]
    if input_tensor.dtype != input_mem_info.dtype: # ... (dtype check as before) ...
         input_tensor = input_tensor.astype(input_mem_info.dtype)
    if input_tensor.shape != input_mem_info.shape: # ... (shape check as before) ...
         print(f"Error: Input tensor shape {input_tensor.shape} != engine shape {input_mem_info.shape}.")
         sys.exit(1)
    np.copyto(input_mem_info.host, input_tensor.ravel())

    # --- 5. Inference ---
    print("Running TensorRT inference...")
    start_time = time.time()
    try:
        trt_outputs_dict = do_inference_v2_by_name(context, bindings=bindings_ordered, inputs_dict=inputs_dict, outputs_dict=outputs_dict, stream=stream)
    except Exception as e:
        print(f"Error during TRT inference: {e}")
        sys.exit(1)
    end_time = time.time()
    print(f"   Inference took: {end_time - start_time:.4f} seconds")

    # --- 6. Postprocess (returns relative points) ---
    print("Postprocessing outputs...")
    try:
        predicted_lanes_relative = postprocess(trt_outputs_dict, args.conf_threshold)
        print(f"   Detected {len(predicted_lanes_relative)} lanes (before NMS).")
    except Exception as e:
        print(f"Error during postprocessing: {e}")
        predicted_lanes_relative = []

    # --- 7. Scale to Original Image and Visualize ---
    print("Scaling coordinates and visualizing results...")
    scaled_lanes = []
    if predicted_lanes_relative:
        scale_x = orig_w
        scale_y = orig_h
        for lane_rel in predicted_lanes_relative:
            # Scale x by width, y by height
            lane_scaled = lane_rel.copy()
            lane_scaled[:, 0] *= scale_x
            lane_scaled[:, 1] *= scale_y
            scaled_lanes.append(lane_scaled.astype(np.int32)) # Convert to int for drawing

    try:
        # Visualize on the ORIGINAL image using SCALED points
        dst = visualize_lanes(src_img_orig.copy(), scaled_lanes, save_path=args.out_file)
        print(f"   Output saved to {args.out_file}")
    except Exception as e:
        print(f"Error during visualization: {e}")

    print("Done.")